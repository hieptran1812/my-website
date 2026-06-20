---
title: "Loan Pricing: Cost of Funds, the Risk Premium, and the Capital Charge"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank actually prices a loan, built up one component at a time — cost of funds, expected loss, the cost of the equity held against it, operating cost, and margin — and why RAROC, not the headline rate, decides whether the loan is worth making."
tags: ["banking", "loan-pricing", "cost-of-funds", "risk-premium", "expected-loss", "capital-charge", "raroc", "risk-based-pricing", "credit-risk", "net-interest-margin"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank does not pick a loan rate out of the air; it stacks up its own costs and adds a margin, and the headline rate is the *output*, not the input.
>
> - A loan rate is built from five pieces: **cost of funds** (what the bank pays for the money it lends), the **expected loss** or risk premium (PD × LGD — the average loss baked into the price), the **capital charge** (the cost of the shareholder equity the regulator forces the bank to hold against the loan), **operating cost**, and a **target margin**.
> - **Risk-based pricing** means the risk premium is not a flat add-on — it scales with the borrower's probability of default, so a near-prime BBB borrower pays around 5.8% while a deeply sub-investment-grade CCC borrower pays over 17% for the same product.
> - The piece almost everyone outside banking forgets is the **capital charge**. A loan that looks profitable on cash margin alone can quietly earn *below* the bank's cost of equity once you charge it for the capital it locks up — which is value-destroying even though it makes money.
> - The number that actually decides whether a loan gets made is **RAROC** — risk-adjusted return on capital — measured against a **hurdle rate** equal to the bank's cost of equity (roughly 10–12%). Price below the hurdle and you are subsidising the borrower with your shareholders' money. The one number to remember: **expected loss = PD × LGD × EAD.**

A bank manager in a small Midwestern town once told a regulator, with complete sincerity, that he priced his commercial loans by "what feels right for the relationship." Two years later his bank was on the FDIC's problem list. The loans were performing — borrowers were paying on time — and yet the bank was bleeding capital. The rates felt right. They were also, every single one of them, wrong.

This is the strange thing about loan pricing: a bank can make a loan that gets fully repaid, on schedule, with no default at all, and still lose money on it. Not lose money in some abstract accounting sense — lose money in the sense that its shareholders would have done better if the bank had simply handed the cash back to them and let them buy a government bond. The rate "felt right." It cleared the cost of the money. It even left a visible cash spread. And it was a slow leak in the hull of the bank.

To understand why, you have to stop thinking of a loan rate as a price the banker chooses and start thinking of it as a *sum the banker is forced to assemble*. The figure below is the mental model for this entire post: a loan rate is a stack. Each layer is a real cost the bank has to cover, and only the very top sliver — the margin — is profit. Get any layer wrong and the whole thing is mispriced. The most commonly missed layer, the one that sank our Midwestern banker, is not even a cash cost. It is the cost of the *equity* the bank must park behind every loan it makes.

![Stacked layers showing cost of funds plus expected loss plus capital charge plus operating cost plus margin equals the quoted loan rate](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-1.png)

This is the operations-level companion to the spread story the whole series keeps returning to. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and cheap, lends long and dear, and lives on the gap. Loan pricing is where that gap is *manufactured*, one loan at a time. If the [net interest margin](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) is the bank's aggregate spread, loan pricing is the microscope view of how each individual deal is built to deliver it — or fails to.

## Foundations: every term you need, from zero

Before we build a single rate, let's define the parts, because each one carries a specific meaning that the everyday word doesn't capture. We'll explain each one in plain English first, then name the industry term so you can look it up later.

**The corner-shop model.** Start with a corner shop that, instead of selling groceries, lends out cash. It gets its cash from three places: customers who leave money in a "savings drawer" at the shop (and expect a little interest), money it borrows from a bigger shop down the street, and the owner's own savings tipped into the till. It lends that cash to neighbours and charges them interest. The shop's whole profit is *the interest it charges minus the interest it pays minus the loans that never come back minus the cost of running the shop*. Every one of those minuses is a layer in our stack. The owner's own savings — the equity — is special: it's the cushion that absorbs the loans that go bad, and the owner expects a fat return on it for taking that risk. Keep this shop in mind; everything below is just this shop with precise vocabulary.

### Cost of funds — what the money costs the bank

A bank doesn't lend its own money. It lends *other people's* money — depositors' cash, money borrowed from other banks, money raised by issuing bonds. Every one of those sources has a price. **Cost of funds** is the blended, weighted-average interest rate the bank pays across all the money it has raised. If a bank funds itself 71% from deposits paying 2%, 10% from wholesale markets paying 5%, and so on, its cost of funds is the weighted average of those rates.

Cost of funds is the *floor* under every loan. A bank cannot lend at a rate below what its money costs it and survive — that's lending at a loss before you even count anything else. A cheap, sticky deposit base (lots of checking accounts paying near 0%) is the single biggest structural advantage a bank can have, because it lowers this floor for every loan the bank makes. That is why the deposit franchise is, quite literally, the whole game.

It helps to see the *mix*. A typical large bank funds itself roughly 71% from deposits, 10% from wholesale and repo markets, 7% from long-term debt, 4% from other liabilities, and about 8% from its own equity. Those sources cost wildly different amounts: a non-interest checking deposit might cost the bank close to 0% (plus a little servicing overhead), a money-market savings account might cost 3–4%, wholesale funding tracks the interbank rate, and long-term bonds carry a credit spread on top. The blended cost of funds is the weighted average of all of them — which is why the *composition* of a bank's funding matters as much as the level of rates. Two banks facing the same Fed funds rate can have cost-of-funds floors a full percentage point apart, purely because one has a richer mix of cheap checking deposits and the other leans on expensive wholesale money. We unpack where the cheap money actually comes from in the [retail-deposit franchise](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) story; for pricing, the only thing that matters is that this blended floor is where every rate starts.

There's a crucial wrinkle: cost of funds is not static within a rate cycle, and it doesn't move in lockstep with the central bank. When the Fed raises rates, loan yields tend to reprice quickly (many loans float, and new loans are priced at the new level), but deposit rates lag — savers don't immediately demand higher interest, and banks don't immediately offer it. The fraction of a Fed rate move that passes through to deposit costs is called the **deposit beta**. Early in a hiking cycle, deposit beta is low (around 0.10 — only a tenth of the hike passes to depositors), so the cost-of-funds floor barely rises while loan yields jump, *widening* the spread. Late in the cycle, deposit beta climbs toward 0.50 or higher as competition for deposits intensifies, the floor catches up, and the spread compresses again. This is why bank margins tend to expand at the start of a tightening cycle and fade as it matures — the floor under the loan-pricing stack is itself moving, on a lag.

### Funds transfer pricing (FTP) — the internal price of money

Here's a subtlety that trips up beginners. A bank is not one pool of money. The branch that gathers a deposit and the desk that makes a loan are different teams with different bosses. So how does the lending desk know what "the money" costs? The answer is **funds transfer pricing**, or **FTP**: an internal accounting system in which the treasury department acts as the bank's internal bank. The lending desk "buys" funds from treasury at an internal rate (the FTP rate), and the deposit-gathering team "sells" its deposits to treasury at an FTP rate. The FTP rate is typically anchored to a market benchmark of the same maturity as the loan.

Why does this matter for pricing? Because **the cost of funds a loan officer uses is the FTP rate for that loan's maturity, not the bank's average deposit rate.** A 5-year loan is funded as if it were borrowed for 5 years in the market, even if the actual cash came from an overnight checking account. This is how the bank charges the loan for the *interest-rate risk* of locking money up for 5 years, and keeps the lending desk from looking artificially profitable just because deposits happen to be cheap today. FTP is the unsung machinery that makes honest loan pricing possible.

#### Worked example: why FTP, not the deposit rate, is the cost of funds

A loan officer wants to make a \$1,000,000 5-year fixed loan. The bank's checking accounts cost almost nothing — say 0.50% blended. Tempting to price the loan off that 0.50% floor: it would let the bank quote a very low rate and still look profitable. But that ignores the maturity mismatch. The deposit can leave tomorrow; the loan is locked for 5 years. If rates rise, the bank will have to pay up to keep funding that 5-year loan, while the loan's yield stays fixed — exactly the trap that destroyed the savings-and-loans.

FTP fixes this. Treasury charges the lending desk the 5-year funding rate — say **3.50%** — regardless of where the cash physically came from. So the loan's cost of funds for pricing purposes is 3.50%, not 0.50%. The 3.00% gap between the 0.50% deposit cost and the 3.50% FTP rate doesn't vanish — it's booked as profit to the *deposit-gathering* side of the bank (which gathered a 5-year-equivalent deposit for 0.50% and "sold" it to treasury for 3.50%), not to the lending side. The lending desk earns only the spread it adds *above* the maturity-matched funding cost.

**The intuition:** FTP splits the bank's spread into two honest halves — the value of cheap deposits (a deposit-franchise profit) and the value of credit and capital (a lending profit) — so neither desk can claim a margin that really belongs to the other, and no loan looks profitable just because today's deposits happen to be cheap.

### Expected loss — the risk premium (PD × LGD × EAD)

Some loans will not be repaid. The bank doesn't know *which* ones in advance, but across a large book it can estimate, statistically, *how much* it will lose on average. That average expected loss is built into the price as a **risk premium** — an extra slice of interest charged so that the borrowers who do repay collectively cover the losses from the borrowers who don't.

The expected loss on a loan is the product of three numbers:

$$ \text{Expected Loss} = PD \times LGD \times EAD $$

- **PD — probability of default:** the chance, over a year, that the borrower stops paying. A *basis point* is one hundredth of a percent (0.01%), and PDs are often quoted in them. A pristine corporate borrower might have a one-year PD of 2–6 bps (0.02–0.06%); a shaky one, several percent.
- **LGD — loss given default:** *if* the borrower defaults, what fraction of the exposure the bank actually loses after recovering collateral and chasing the borrower. A well-secured mortgage might have an LGD of 25% (you recover 75 cents on the dollar by selling the house); an unsecured credit-card balance, 55% or more.
- **EAD — exposure at default:** how much the borrower owes at the moment they default. For a term loan that's roughly the outstanding balance; for a credit line it's the drawn amount plus an estimate of how much more they'll draw on the way down.

The risk premium baked into the rate is the expected loss *as a percentage of the exposure* — that is, PD × LGD. (EAD scales the dollar loss but cancels out when you express the premium as a rate.) We go far deeper on this engine in the [PD-LGD-EAD credit-risk post](/blog/trading/banking/credit-analysis-the-five-cs-and-how-a-loan-gets-approved); here we just need it as the second layer of the stack.

### The capital charge — the cost of the equity behind the loan

This is the layer outsiders never see and bankers can never forget. Regulators require a bank to fund a portion of every loan with *its own equity* rather than with borrowed money — this is the [capital requirement](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) at the heart of Basel. The point of that equity is to absorb *unexpected* losses (the losses beyond the average, in a bad year). But shareholders don't provide that equity for free. They demand a return on it — the bank's **cost of equity**, typically 10–12%.

So every loan ties up a chunk of expensive shareholder capital, and the *cost of that tied-up capital* must be charged to the loan. That charge is the **capital charge**. It is not a cash outflow; it is an *opportunity cost* — the return shareholders could have earned elsewhere on the equity now imprisoned behind this loan. A loan that earns a decent cash spread but doesn't cover its capital charge is destroying shareholder value even as it books a profit. This is the leak that sinks "feels-right" pricing.

### Operating cost — running the loan

Loans cost money to make and to service: the loan officer's time, the underwriting, the systems, the collections team, the branch overhead. **Operating cost** (often just "opex" or the *cost-to-serve*) is the slice of those expenses allocated to the loan, expressed as a percentage of the balance. A jumbo corporate loan has tiny opex per dollar (one deal, lots of dollars); a portfolio of small personal loans has high opex per dollar (many small accounts, each needing servicing).

### Target margin, the hurdle rate, and RAROC

Whatever is left after the bank covers all four costs above is its **margin** — its profit. But "profit" alone doesn't tell the bank whether the loan was *worth making*. For that you need to compare the profit to the *capital it consumed*, and compare that ratio to a threshold.

- **RAROC — risk-adjusted return on capital:** the loan's profit *after* subtracting expected loss, divided by the equity capital held against it. It answers: "for every dollar of shareholder equity this loan locks up, how much profit does it throw off, after we've already paid for the average losses?"
- **Hurdle rate:** the minimum RAROC the bank will accept, set equal to its **cost of equity**. If shareholders demand 12%, then any loan returning less than 12% RAROC is, by definition, earning them less than they require — value-destroying.

The decision rule is brutally simple: **make the loan only if RAROC ≥ hurdle rate.** Price the loan so that it clears the hurdle. That is the entire job. Everything else is figuring out the inputs.

Now we have all the parts. Let's build a rate.

## Building a loan rate, one layer at a time

The cleanest way to understand loan pricing is to assemble a rate in *basis points*, the way a pricing desk actually does it. The chart below builds a representative commercial loan rate from its five components, and the worked example walks the same arithmetic.

![Stacked bar chart showing the loan rate built from cost of funds, expected loss, capital charge, operating cost, and target margin in basis points](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-2.png)

The visual claim is the thesis of the post: the quoted rate is the *top* of a stack, and only the green sliver at the top is profit. Notice how small the risk premium and operating cost are relative to the cost of funds — for a healthy borrower in a normal rate environment, the cost of funds dominates everything. That has a consequence we'll come back to: when central-bank rates move, the floor under every loan moves with them, and the spread layers barely change.

#### Worked example: building a rate from the five components

Let's price a 5-year term loan to a solid, investment-grade mid-sized company. We'll use round numbers you can carry in your head.

Start with the **cost of funds**. The 5-year FTP rate — what treasury charges the lending desk for 5-year money — is **3.50%**. That's our floor.

Add the **expected loss**. The borrower is rated BBB internally, with a one-year PD of **0.24%**. The loan is senior unsecured, so LGD is **45%**. The risk premium is:

$$ PD \times LGD = 0.24\% \times 0.45 = 0.108\% \approx 0.11\% $$

So far we're at 3.50% + 0.11% = 3.61%. The borrower's average expected loss costs the bank about 11 bps a year — small, because investment-grade default is rare.

Add the **capital charge**. Suppose the regulator requires the bank to hold **8.5%** equity against this loan's risk-weighted amount, and the bank's cost of equity (its hurdle) is **10%**. We'll do the precise version later; for now use the rule-of-thumb that the capital charge ≈ (capital ratio) × (cost of equity) ≈ 8.5% × 10% = **0.85%** of the loan. Running total: 3.61% + 0.85% = 4.46%.

Add **operating cost** — the cost to underwrite and service this loan, allocated at **0.50%**. Total: 4.46% + 0.50% = 4.96%.

Finally, the **target margin**. The bank wants a profit slice of **0.80%** on top — the relationship is competitive, so it can't pile on more. Final quoted rate:

$$ 3.50\% + 0.11\% + 0.85\% + 0.50\% + 0.80\% = 5.76\% $$

(In the cover figure we used a slightly riskier borrower — a 0.45% risk premium and an 0.80% margin — to land on a round 7.10% illustration; the *method* is identical, only the borrower's risk differs.)

**The intuition:** the rate the borrower sees, 5.76%, is not a markup on the cost of funds — it's the sum of five distinct costs, four of which exist whether or not the borrower ever misses a payment. Change the borrower's risk and only one layer (the risk premium) moves; change the rate environment and only the bottom layer (cost of funds) moves.

## The risk premium up close: why two borrowers pay different rates

The layer that makes loan pricing *interesting* is the risk premium, because it is the only one that scales with *who the borrower is*. This is the heart of **risk-based pricing**: instead of charging everyone the same rate and hoping the good borrowers subsidise the bad, the bank charges each borrower a premium calibrated to their own probability of default.

The chart below shows the quoted rate climbing across internal rating grades. The fixed costs (cost of funds, capital charge, opex, margin) set a floor of about 5.65%; on top of that floor, the expected-loss layer grows from almost nothing for AAA to enormous for CCC.

![Line chart of the quoted loan rate and the expected-loss risk premium rising across rating grades from AAA to CCC](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-3.png)

The shape is the whole story. For the top four grades — AAA through BBB — the line is almost flat, because default is so rare that the risk premium is a rounding error: the rate creeps from 5.65% to 5.76%. Then it bends sharply. A BB borrower (PD 1.20%) pays 6.19%; a B borrower (PD 5.50%) pays 8.12%; a CCC borrower (PD 26%) pays 17.35%. The risk premium is *non-linear* in default risk because PD itself explodes as credit quality falls. This is why high-yield lending is a fundamentally different business from prime lending — the prices aren't a little higher, they're in another universe.

#### Worked example: pricing a prime borrower vs a subprime borrower

Two borrowers walk into the same bank for the same \$200,000, 5-year, senior-unsecured loan. The fixed layers are identical: cost of funds 3.50%, capital charge 0.85%, opex 0.50%, margin 0.80% — a fixed floor of **5.65%**. The only thing that differs is the risk premium.

**Borrower A — prime, rated A.** One-year PD **0.06%**, LGD **45%**. Risk premium:

$$ 0.06\% \times 0.45 = 0.027\% \approx 0.03\% $$

Quoted rate: 5.65% + 0.03% = **5.68%**. On \$200,000, the expected loss the bank prices in is 0.027% × \$200,000 = **\$54 a year**. Almost nothing.

**Borrower B — subprime, rated B.** One-year PD **5.50%**, LGD **45%**. Risk premium:

$$ 5.50\% \times 0.45 = 2.475\% \approx 2.48\% $$

Quoted rate: 5.65% + 2.48% = **8.13%**. On \$200,000, the expected loss priced in is 2.475% × \$200,000 = **\$4,950 a year** — almost a hundred times Borrower A's.

The gap in their rates, 8.13% − 5.68% = **2.45%**, is *entirely* the difference in their risk premiums. Borrower B isn't being punished or judged; B is being charged the actuarially fair cost of B's own default probability, the same way a riskier driver pays a higher car-insurance premium.

**The intuition:** risk-based pricing turns the rate into a thermometer for the borrower's credit quality. If you ever see a lender quoting wildly different rates to two people for the same product, the difference is almost always the risk premium — PD × LGD made visible.

### LGD: why the same borrower can get two different rates

Notice that the risk premium has *two* moving parts, and we've only varied one. PD is about the *borrower*; LGD is about the *structure of the loan* — chiefly, the collateral. The same borrower, with the same PD, can be charged a different risk premium depending on how the loan is secured, because better security means the bank loses less when default happens.

![Horizontal bar chart of loss given default by collateral type, from senior secured to subordinated](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-4.png)

The chart shows LGD by how the loan is secured. A **senior secured** loan — first claim on good collateral — has an LGD around 25%: in default the bank recovers 75 cents on the dollar. A **senior unsecured** loan recovers less, LGD around 45%. A **subordinated** loan, which only gets paid after the senior lenders are made whole, has an LGD around 65%. Same borrower, same PD — but the risk premium scales directly with LGD.

#### Worked example: how collateral changes the rate

Take Borrower B again — rated B, PD **5.50%** — but now offer two versions of the loan.

**Version 1 — senior secured (LGD 25%).** Risk premium = 5.50% × 0.25 = **1.375%**. Quoted rate = 5.65% + 1.38% = **7.03%**.

**Version 2 — senior unsecured (LGD 45%).** Risk premium = 5.50% × 0.45 = **2.475%**. Quoted rate = 5.65% + 2.48% = **8.13%**.

The collateral is worth **1.10%** a year to the borrower — 8.13% − 7.03% — purely by cutting the bank's loss-given-default from 45% to 25%. On \$200,000 that's \$2,200 a year of interest saved by pledging an asset.

**The intuition:** this is why lenders push so hard for collateral and why secured loans are cheaper than unsecured ones for the *same* borrower. Collateral doesn't change the chance you default; it changes how much the bank loses if you do, and that flows straight into the price. The [credit-spread machinery in fixed income](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default) is the bond-market version of exactly this PD-and-LGD calculation.

### Point-in-time vs through-the-cycle: which PD goes into the price?

There's a deceptively hard question lurking inside the risk premium: *which* probability of default should the bank use? The answer depends on whether you measure PD at this exact moment or as an average across good times and bad — and the choice quietly determines whether a bank's pricing amplifies the credit cycle or dampens it.

A **point-in-time (PIT) PD** estimates the borrower's default probability given today's conditions. In a roaring economy, defaults are rare, so PIT PDs are low and the risk premium shrinks; in a recession, PIT PDs spike and the premium balloons. A **through-the-cycle (TTC) PD** averages over a full economic cycle, so it's more stable — it doesn't collapse in the boom or explode in the bust. Both are legitimate; they answer different questions.

The trap is using PIT PDs to price long-dated loans during a boom. If a bank originates a 5-year loan in the best year of an expansion, the borrower's PIT PD might be a fraction of its through-the-cycle average. Price the risk premium off that rosy PIT number and the loan is *underpriced for most of its life* — the boom-time PD will not hold for five years, and when the cycle turns, the realized losses blow through the thin premium the bank charged. This is the mechanical heart of why credit losses always seem to arrive "unexpectedly": banks priced for the conditions at origination, not for the cycle the loan would actually live through.

#### Worked example: the boom-time underpricing trap

A bank prices a 5-year loan to a BB borrower during a boom. The borrower's *point-in-time* PD looks like only **0.50%** (defaults are scarce this year), against a *through-the-cycle* average of **1.20%**. Using LGD of 45%:

- Priced on the PIT PD: risk premium = 0.50% × 0.45 = **0.23%**.
- Priced on the TTC PD: risk premium = 1.20% × 0.45 = **0.54%**.

The bank that prices off the boom-time PIT number charges 0.31% too little — and earns it on every dollar of the loan for five years. On a \$10,000,000 loan that's \$31,000 a year, \$155,000 over the life, of risk premium the bank never collected. When the cycle turns and realized defaults climb toward the through-the-cycle rate, the loan's thin premium can't absorb the losses, and what looked like a profitable booking becomes a loss. The disciplined bank prices long loans off something closer to the through-the-cycle PD even when current conditions look benign.

**The intuition:** the risk premium is supposed to cover *average* losses over the loan's whole life, not the unusually-good conditions at the moment it's signed. Pricing off point-in-time PDs in a boom is how a bank builds a portfolio that looks brilliant until the cycle does what cycles do. This is why bank earnings are so [pro-cyclical](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the underpricing is invisible until the losses show up all at once.

## The capital charge: the layer that decides everything

Now we come to the layer that separates bankers who survive from bankers who end up on the problem list. The capital charge is invisible on a cash-flow statement, it never shows up as a payment, and it is the single most common reason a "profitable" loan is actually destroying value.

Let's rebuild the intuition slowly. Regulators don't let a bank fund a loan entirely with borrowed money (deposits, bonds). They require a fraction of it — call it the **capital ratio** — to be funded with the bank's *own equity*, so that if the loan goes bad in an unexpectedly severe way, the loss eats the shareholders' money, not the depositors'. We covered the leverage version of this in [why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion); here we need its *pricing* consequence.

The mechanism works like this. To make a \$1,000,000 loan, the bank must hold, say, \$85,000 of equity against it (an 8.5% capital ratio) and can fund the other \$915,000 with cheap deposits or debt. That \$85,000 of equity is *expensive*: shareholders demand a return on it. If the bank's cost of equity is 10%, then locking up \$85,000 of equity costs the bank 10% × \$85,000 = **\$8,500 a year** in required shareholder return. Spread that over the \$1,000,000 loan, and it's **0.85%** of the loan balance — that's the capital charge.

So the capital charge formula is simply:

$$ \text{Capital charge (\% of loan)} = \text{capital ratio} \times \text{cost of equity} $$

A loan that ignores this layer prices itself 0.85% too cheap. And here is the trap: that 0.85% is *not* a cash cost. The bank still collects its cash spread; the income statement still shows a profit. The damage is silent — the loan simply doesn't earn enough to compensate shareholders for the equity it consumed. The chart below makes the trap concrete: the same loan, priced two ways.

![Before-and-after comparison of a loan priced ignoring versus including the capital charge, showing the RAROC difference](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-5.png)

On the left, the bank prices the loan covering cost of funds, expected loss, opex, and margin — and quotes 5.25%. It looks competitive and it books a cash profit. But the RAROC is only 6%, far below a 12% hurdle: the loan earns shareholders half what they demand. On the right, the bank adds the 0.85% capital charge, quotes 6.10%, and the RAROC jumps to 14% — now above the hurdle. The extra 0.85% isn't greed; it's the difference between creating and destroying shareholder value.

#### Worked example: how the capital charge changes the price

A bank prices a \$1,000,000 corporate loan. Cost of funds 3.50%, expected loss 0.40%, opex 0.45%, target margin 0.50%. Add it up *without* the capital charge:

$$ 3.50\% + 0.40\% + 0.45\% + 0.50\% = 4.85\% $$

That feels like a fine, competitive rate. Now let's check what it actually earns the shareholders. The bank holds 8.5% × \$1,000,000 = **\$85,000** of equity against the loan. The loan's net income, after paying the cost of funds and the expected loss and the opex, is its margin: 0.50% × \$1,000,000 = **\$5,000**. So RAROC is:

$$ \frac{\$5,000}{\$85,000} = 5.9\% $$

Against a 12% hurdle, this loan returns *less than half* of what shareholders require. It books a \$5,000 cash profit and quietly destroys value.

Now price it *with* the capital charge. The capital charge is 8.5% × 12% (using the cost of equity as the required return) = **1.02%**. Add it in:

$$ 3.50\% + 0.40\% + 0.45\% + 0.50\% + 1.02\% = 5.87\% $$

At 5.87%, the loan's net income above expected loss is now 0.50% + 1.02% = 1.52% × \$1,000,000 = **\$15,200**, and:

$$ \text{RAROC} = \frac{\$15,200}{\$85,000} = 17.9\% $$

— comfortably above the 12% hurdle. The single extra layer, 1.02%, is the difference between a loan that subsidises the borrower and a loan that pays the shareholders what they're owed.

**The intuition:** the capital charge is the price of the regulatory equity a loan locks up, and a loan that doesn't cover it loses money in the only sense that ultimately matters — it earns shareholders less than they could get elsewhere for the same risk. "Profitable" and "worth doing" are not the same sentence.

### Why riskier loans carry a bigger capital charge

There's a second-order effect that makes the capital charge even more important for risky loans. Regulators don't require the same capital ratio against every loan — they require *more* equity against riskier loans, via **risk weights**. A loan to a strong corporate might carry a 50% risk weight (so the bank holds capital against half the loan amount); a loan to a weak one might carry 150% (capital against one and a half times the amount). We unpack the machinery in [how risk-weighted assets work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work); the pricing consequence is that **riskier borrowers pay twice** — once through a bigger expected-loss premium, and again through a bigger capital charge, because the bank must hold more equity behind their loan.

This compounding is why the rate curve in the risk-based-pricing chart bends so steeply at the bottom grades. The CCC borrower isn't paying 17% just because of expected loss; they're also tying up far more of the bank's scarce equity, and that equity has to earn its hurdle too.

## RAROC vs the hurdle: the decision the whole post is building toward

Everything so far has been about constructing a *rate*. But a bank's loan committee doesn't approve rates — it approves *deals*, and the test it applies is RAROC against the hurdle. RAROC reframes every loan in a single comparable currency: return on the bank's scarcest resource, equity.

The reason this matters is that loans compete for capital. A bank has a finite amount of equity. Every loan it makes consumes some. So the right question is never "does this loan make money?" — almost any loan above the cost of funds makes *some* money. The right question is "does this loan make *enough* money for the equity it consumes, compared to the next-best use of that equity?" RAROC answers exactly that, and the hurdle rate is the bar.

The chart below shows RAROC across rating grades under two pricing regimes: proper risk-based pricing, and a single flat rate for everyone.

![Grouped bar chart of RAROC by rating grade comparing risk-based pricing against one flat rate, with the hurdle rate marked](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-6.png)

The blue bars — risk-based pricing — sit flat at the 12% hurdle for every grade. That's not a coincidence; it's the *design*. When you price each loan to cover its own expected loss and capital charge plus the hurdle return on capital, every loan earns exactly the hurdle, regardless of risk. The bank is indifferent between lending to AAA and CCC, because each is priced to the same return on capital.

Now look at the amber bars — a single flat 6.50% rate for everyone. The pattern is a disaster. The safe AAA, AA, and A borrowers throw off RAROC far above the hurdle (45%+), so the bank is *overcharging* them — and will lose them to a competitor who prices properly. The risky B borrower scrapes a RAROC of 0.2%, and the CCC borrower posts a RAROC of **−72%** — a catastrophic value destroyer. Flat pricing systematically drives away your best customers and piles up your worst. This is *adverse selection*, and it's exactly the trap our "feels-right" Midwestern banker fell into.

#### Worked example: a RAROC-vs-hurdle decision

A relationship manager wants to make a \$500,000 loan to a BB-rated company at **6.00%**. The bank's hurdle is 12%. Should the loan committee approve it?

Work out the loan's economics per the components. Cost of funds 3.50%; expected loss = PD 1.20% × LGD 45% = 0.54%; opex 0.50%. The capital held: BB carries a 115% risk weight, the capital ratio is 8.5%, so capital held = 8.5% × 115% × \$500,000 = **\$48,875**.

Net income after expected loss = (rate − cost of funds − expected loss − opex) × loan = (6.00% − 3.50% − 0.54% − 0.50%) × \$500,000 = 1.46% × \$500,000 = **\$7,300**.

$$ \text{RAROC} = \frac{\$7,300}{\$48,875} = 14.9\% $$

That's above the 12% hurdle, so the committee approves — the loan clears. But now suppose the relationship manager, under competitive pressure, wants to cut the rate to **5.40%** to win the deal. Re-run it: net income = (5.40% − 3.50% − 0.54% − 0.50%) × \$500,000 = 0.86% × \$500,000 = **\$4,300**, so RAROC = \$4,300 / \$48,875 = **8.8%** — *below* the hurdle. At 5.40% the loan destroys value. The committee should either hold the line at a rate that clears 12% (about 5.74% is the break-even), find non-credit revenue from the relationship to make up the gap, or walk away.

**The intuition:** RAROC turns "what rate should we charge?" into "does this clear our cost of equity?" The minimum acceptable rate is the rate that makes RAROC equal the hurdle — and any rate below it is a transfer from your shareholders to the borrower.

### What drives each component — and why pricing is mostly out of the banker's hands

Step back and look at the five layers together. The striking thing is how *little* of the rate the banker actually controls. The matrix below lays out each component, its typical size, and who really sets it.

![Matrix of the five loan-pricing components with their drivers, typical sizes, and who controls each](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-7.png)

Read down the "who sets it" column. The cost of funds is set by the central bank and by savers' appetite for the bank's deposits. The expected loss is set by the *borrower's* riskiness. The capital charge is set by Basel and by the shareholders' required return. Operating cost is the one genuinely internal lever — the bank's own efficiency. Only the target margin is set by "the market," and even that is squeezed by competition. The banker is less a price-*setter* than a price-*assembler*, snapping together pieces that other forces dictate. The freedom is in the margin and in choosing *which* borrowers to serve — not in conjuring a rate.

This is why the realized spread the whole industry earns moves with the cycle rather than with any single bank's cleverness. The next chart shows the aggregate margin loan pricing has to deliver, across a full rate cycle.

![Line chart of US commercial bank net interest margin from 2010 to 2024 showing the ZIRP trough and the post-2022 recovery](/imgs/blogs/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge-8.png)

This is the FDIC's aggregate net interest margin for US banks. It fell to a ZIRP-era trough of **2.56% in 2021** — when the cost of funds was floored near zero but so were loan yields, compressing the spread — then jumped back to **3.23% by 2024** as the Fed hiked and loans repriced faster than deposits. Every loan a bank prices is a tiny contribution to this aggregate line, and the line shows that no amount of pricing discipline can fully escape the rate environment: when the cost-of-funds floor moves, every loan's economics move with it.

#### Worked example: the margin a book of loans must clear

Suppose a bank has \$10 billion of loans funded at a blended cost of funds of 2.50%, with average expected loss of 0.40%, opex of 1.00% of loans, and a cost of equity of 12% on equity equal to 8% of loans. What average loan yield does the book need just to hit its hurdle?

The required margin over cost of funds has to cover: expected loss (0.40%) + opex (1.00%) + capital charge (8% × 12% = 0.96%) = **2.36%**. So the book must yield at least 2.50% + 2.36% = **4.86%** to earn exactly the hurdle. That implies a net interest margin (loan yield minus cost of funds) of about 2.36% before opex, which lines up with why the industry sweats when NIM compresses toward 2.5% — at that level there is almost no room left for opex and the capital charge, let alone profit. On \$10 billion, every 0.10% of margin is **\$10 million** of pre-tax income.

**The intuition:** loan pricing isn't done one loan at a time in isolation — each loan has to pull its weight in delivering a book-level margin that clears the bank's costs *and* its cost of equity. When the aggregate NIM falls toward the cost stack, the whole industry's pricing discipline gets tested at once.

## Common misconceptions

**"The loan rate is the bank's profit margin."** No — the rate is the *top of a cost stack*, and the profit is only the thin slice left after cost of funds, expected loss, capital charge, and opex. In our worked example a 5.76% rate contained roughly 0.80% of actual margin; the other 4.96% was covering costs. When a borrower complains that "the bank is charging 6% and only pays me 2%, so it's pocketing 4%," they're forgetting that most of that 4% pays for default losses, regulatory capital, and the cost of running the loan.

**"A loan that gets repaid was a good loan."** Not necessarily. A loan can be repaid in full and still have destroyed value, because it never covered its capital charge — it earned shareholders less than their required return on the equity it tied up. Our \$1,000,000 example booked a \$5,000 cash profit and a 5.9% RAROC against a 12% hurdle: fully performing, and a value destroyer. Repayment is necessary, not sufficient.

**"Risky borrowers are charged more to punish them."** They're charged more because they *cost* more — both in expected loss (a higher PD means more of them default) and in capital charge (a higher risk weight means more equity locked up). It's actuarial, not moral. The B borrower's 8.13% versus the A borrower's 5.68% is the fair price of B's 5.50% default probability, the same way a teenager pays more for car insurance than a 50-year-old.

**"Cheaper deposits just mean more profit."** Cheaper deposits lower the cost-of-funds *floor*, but in a competitive market that advantage gets partly passed through to borrowers as lower loan rates — the bank with the cheapest funding can win deals others can't. The deposit advantage shows up as *market share and resilience* as much as headline margin. And in a falling-rate environment, deposit costs can't fall below zero, so the floor stops dropping while loan yields keep falling, compressing the spread — exactly what happened into the 2021 NIM trough.

**"Loan pricing is mostly about beating competitors on rate."** Competition sets the *ceiling* on the margin, but the *floor* is set by the bank's own cost stack, and a bank that prices below its RAROC hurdle to win share is not competing — it's subsidising. The disciplined response to a competitor's low rate isn't to match it blindly; it's to check whether the deal still clears the hurdle, and to walk if it doesn't. The banks that grew fastest into the 2008 and 2023 cycles were often the ones who'd quietly abandoned hurdle discipline to win volume.

**"The risk premium is there to cover *this* loan's losses."** A single loan either defaults or it doesn't — a 0.24% PD doesn't mean a borrower loses 0.24% of the loan; it means roughly one in four hundred such borrowers defaults entirely while the rest pay in full. The risk premium is a *portfolio* device: it's charged to every borrower in a risk pool so that the many who repay collectively fund the few who don't. This is why risk-based pricing only works at scale — a lender with three subprime loans is gambling, not pricing; a lender with thirty thousand is running an insurance book where the law of large numbers makes the expected loss a reliable cost. Misjudge the *pool's* average default rate, as the subprime lenders did, and the premium charged across the whole book is too thin no matter how carefully any individual loan was assessed.

**"Once the loan is priced, the price is locked in."** For a fixed-rate loan the *contract* rate is locked, but the loan's *economics* keep moving. If the borrower's credit deteriorates after origination, the realized risk premium the bank is collecting becomes too thin for the now-higher PD — the loan is underpriced in hindsight, and there's nothing the bank can do but watch its margin of safety erode. This is why banks monitor loans continuously and why a deteriorating loan gets moved to a worse internal rating (and a bigger provision) even though its contract rate never changes. The price was right at origination and wrong six months later, through no change in the contract — only in the borrower.

## How it shows up in real banks

**The savings-and-loan disaster of mispricing maturity (1980s).** The S&L crisis is, at its root, a loan-pricing failure dressed up as an interest-rate disaster. Thrifts made 30-year fixed mortgages at, say, 6%, funded by short-term deposits. When the Fed pushed short rates to ~18% to break inflation, the thrifts' cost of funds rocketed above their loan yields — they were paying 12% for money lent out at 6%. Modern FTP would have priced those 30-year mortgages off a 30-year funding cost, exposing the maturity bet at origination. Over 1,000 thrifts failed and the cleanup cost taxpayers roughly \$124 billion. The lesson burned into every treasury department since: price the loan against funds of its own maturity, or you're making an unhedged rate bet you didn't mean to make.

**RAROC's birth at Bankers Trust (1970s–80s).** RAROC isn't an academic nicety — it was invented at Bankers Trust precisely to answer "which of our loans are actually worth the capital they consume?" Before RAROC, banks ranked loans by cash spread, which systematically favoured big, low-margin, capital-light deals and undervalued smaller, higher-return ones. Once Bankers Trust started measuring return *per unit of capital at risk*, whole product lines that looked profitable on spread turned out to be value destroyers, and the bank reallocated capital toward where it actually earned the hurdle. Every modern bank's loan-approval system is a descendant of that idea.

**The post-2008 capital-charge repricing.** When Basel III roughly tripled the effective equity a bank had to hold against many loans, it raised the capital-charge layer of the stack for the entire industry overnight. Banks responded exactly as the pricing model predicts: they repriced or exited the loans whose capital charge had jumped most — long-dated corporate facilities, certain trade-finance lines, low-rated exposures — because at the old rates those loans no longer cleared the hurdle once their capital charge rose. This is why "regulation made some lending uneconomic" wasn't a complaint about red tape; it was the capital-charge layer of the pricing stack mechanically rising and pushing certain rates above what the market would bear.

**The 2021 net-interest-margin squeeze.** When the FDIC's aggregate NIM hit 2.56% in 2021, it wasn't because banks forgot how to price — it was because the cost-of-funds floor had collapsed toward zero *and* loan yields had collapsed with it, leaving almost no room above the cost stack. Banks couldn't lower deposit rates below zero, so the floor stopped falling while asset yields kept dropping. Pricing discipline held, but the *environment* compressed the achievable margin for everyone simultaneously — a reminder that the bottom layer of the stack is set by the central bank, not the bank.

**Auto and credit-card lending as pure risk-based pricing.** Consumer lending is where risk-based pricing is most visible to ordinary people. Two applicants for the same car loan can be quoted 4% and 14% — a 10-point gap that is almost entirely the difference in their risk premiums (PD × LGD) plus, at the bottom, a fatter capital charge for the riskier book. The subprime auto lenders who blew up in various cycles typically did so not because their *rates* were wrong but because their *PD estimates* were — they priced for a 6% default rate and got 15%, so the risk-premium layer was too thin to cover the losses that actually arrived. The pricing framework was sound; the input was optimistic.

**The "win the relationship, lose on the loan" trap.** Commercial banking is full of deals priced below the hurdle on the loan itself, justified by future fee income — cash management, FX, advisory — that the relationship is supposed to throw off. Sometimes that's genuine: the loan is a loss leader for a profitable relationship. But it curdles when the promised ancillary revenue never shows up and the bank is left holding a book of sub-hurdle loans. The 2023 regional-bank stress exposed several lenders who'd grown by pricing aggressively for relationships that turned out to be thin, leaving them with assets that didn't earn their capital just as their funding costs spiked.

**Mortgage pricing and the index-plus-margin rule.** A retail mortgage is the cleanest real-world expression of the pricing stack, because regulators essentially force lenders to show their work. A typical adjustable mortgage is quoted as "index plus margin" — the index (a published market rate, the lender's cost-of-funds proxy) plus a fixed margin that bundles the expected loss, capital charge, opex, and profit. The margin a borrower is offered moves directly with their credit score and loan-to-value ratio: a borrower with a high score and a 60% loan-to-value (low PD, low LGD because of the equity cushion in the home) gets a thin margin; a borrower with a weak score and a 95% loan-to-value (higher PD, much higher LGD because there's little equity to recover) gets a fat one. The 1–2 percentage-point gap between the best and worst quoted mortgage rates in any given week is the risk-premium and capital-charge layers, made transparent by the index-plus-margin format. When you shop a mortgage and watch your rate change as your down payment changes, you are watching LGD get repriced in real time.

**The dash for trade-finance and leverage-ratio arbitrage.** After Basel III introduced a flat leverage-ratio backstop alongside the risk-weighted capital rules, some banks discovered that low-risk-weight, high-volume lending — like short-term trade finance — could become unattractive not because of its *risk-weighted* capital charge but because of the *leverage-ratio* one, which charges capital against the raw loan amount regardless of risk. Suddenly a safe, thin-margin trade-finance loan that cleared the risk-weighted hurdle easily failed the leverage-ratio hurdle, because the leverage rule demanded more capital than the loan's thin margin could pay for. Several global banks pulled back from trade finance for exactly this reason — a vivid demonstration that the capital charge isn't a single number but the *binding* one among several regulatory constraints, and that a loan's price must clear whichever constraint bites hardest.

## The takeaway: how to use this

Once you see a loan rate as a stack rather than a number, you can never un-see it, and it changes how you read everything from your own mortgage quote to a bank's earnings call.

**When you're the borrower,** you now know which layer to push on. You can't argue the cost of funds (the Fed set it) or the capital charge (Basel set it). But the risk premium is *yours* — improve your credit profile and your PD falls; pledge collateral and your LGD falls; both shrink the only layer that's specifically about you. The difference between a 5.68% and an 8.13% rate on the same loan was entirely risk premium, and risk premium is the layer a borrower can actually move. It also reframes how you should negotiate: asking a lender to "just lower the rate" is asking them to cut into the margin or, worse, to price below their hurdle — which a disciplined lender won't do. Asking instead "what would lower my risk premium?" gives them something they *can* say yes to, because a bigger down payment or additional collateral genuinely lowers the bank's expected loss and capital charge, and a lower true cost can flow back to you as a lower rate. You're not haggling over the bank's profit; you're changing the inputs to its cost stack.

**When you're reading a bank,** the loan-pricing stack tells you where to look for trouble. A bank growing its loan book fast while its NIM is flat or falling is almost certainly winning volume by pricing below the hurdle — buying market share with its shareholders' capital. A bank whose RAROC discipline is real will *shrink* in a too-competitive market rather than chase deals that don't clear. The most dangerous sentence in banking is "we're growing the book in a tough rate environment," because it usually means the cost stack is being ignored to hit a volume target — the precise error that put our Midwestern banker on the problem list and that recurs in every credit cycle.

**The deepest point** ties straight back to the series' spine. A bank is a leveraged, confidence-funded maturity-transformation machine that lives on the spread. Loan pricing is the act of *manufacturing that spread one deal at a time* — and the capital charge is the layer that connects the individual loan back to the bank's thin equity cushion. Price every loan to cover its capital charge and clear the hurdle, and the thin cushion earns its keep and grows. Price below the hurdle — even on loans that get fully repaid — and you are slowly spending the very cushion that's supposed to keep you alive. The "feels-right" rate is the one that quietly does the latter. The disciplined rate, built layer by layer and tested against the hurdle, is the one that keeps the machine running.

## Further reading & cross-links

- [Credit analysis: the five Cs and how a loan gets approved](/blog/trading/banking/credit-analysis-the-five-cs-and-how-a-loan-gets-approved) — where PD, LGD, and the credit decision behind the risk premium actually get made.
- [Net interest margin and the spread business explained](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the aggregate spread that loan pricing must deliver, across the rate cycle.
- [Risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) — the machinery that sets the capital charge layer and makes riskier loans cost more equity.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — why the equity behind every loan is so expensive, and so scarce.
- [Credit spreads: pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default) — the bond-market version of the same PD-and-LGD risk premium, priced as a spread over the risk-free rate.
