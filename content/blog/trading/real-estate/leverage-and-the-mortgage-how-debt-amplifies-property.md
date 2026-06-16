---
title: "Leverage and the Mortgage: How Debt Amplifies Property — Both Ways"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into how a mortgage multiplies your property returns up and down, what LTV, amortization and DSCR really mean, and why Vietnam's teaser-then-reset loans are a payment-shock trap."
tags: ["real-estate", "property", "mortgage", "leverage", "ltv", "amortization", "dscr", "vietnam", "negative-equity", "risk-management"]
category: "trading"
subcategory: "Real Estate"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A mortgage lets a small down payment control a whole property, so it multiplies the percentage move in price onto your equity — symmetrically, up *and* down.
>
> - Leverage doesn't change the asset; it changes how far a price fall is from wiping out your equity. That single sentence is the most important risk fact in property.
> - Three dials govern the risk: **LTV** (how much you borrow), **amortization** (how the loan is repaid — early payments are almost all interest), and **DSCR** (whether the rent covers the payment).
> - At **80% LTV a 20% price fall wipes your equity to zero**; at 50% LTV it takes a 50% fall. The borrowed share *is* the cushion you don't have.
> - Vietnam's mortgages add a second trap: a ~8% teaser rate for 1–2 years that **resets to 12–14%**, jumping the monthly payment ~40% overnight. The one number to remember: on Minh's ₫5.6bn loan, the ₫43.2M teaser payment becomes ₫62.2M on reset.

In 2007, a school teacher outside Las Vegas bought a \$300,000 house with \$15,000 down — a 5% down payment, a 95% loan. For eighteen months she felt like a genius: the house "rose" to \$360,000 and her \$15,000 stake had, on paper, quadrupled. Then prices in her zip code fell 35%. The house was worth \$195,000; she owed \$282,000. Her \$15,000 was gone, and she was \$87,000 *in the hole* — she would have to bring a cheque to closing just to sell. The house never moved. The asset was the same three-bedroom it had always been. What moved was the *debt* stacked on top of it.

Fifteen years and ten thousand kilometres away, a Ho Chi Minh City marketing manager we'll call **Minh** signs for a ₫7.0 billion (≈ \$270,000) apartment with ₫1.4 billion (≈ \$54,000) down. The bank lends him the other ₫5.6 billion (≈ \$216,000) at a cheerful 8% "preferential" rate. Two years later the rate resets to 13% and his monthly payment jumps from ₫43.2 million to ₫62.2 million — a 44% increase on a payment that already swallowed most of his salary. Minh didn't buy a different apartment. He bought the *same* apartment with a loan whose price he didn't fully read.

These two people, on two continents, are living the same equation. The diagram above is the mental model for the whole post: **a mortgage is a lever, and a lever multiplies force in both directions.** Push the price up 10% and your equity can leap 50%; push it down 10% and your equity can fall 50% just as fast. This article builds that lever from the ground up — what a mortgage actually is, why early payments are almost all interest, how to tell whether the rent covers the loan, and exactly how far a price can fall before your equity hits zero. We will keep Minh in HCMC and a US investor named **Dana** as our running examples, in ₫ and \$ side by side, so both a Vietnamese and an international reader stays oriented the whole way.

![Two panels showing a property bought with a 20 percent down payment: a 10 percent price rise turns into a 50 percent equity gain, and a 10 percent price fall turns into a 50 percent equity loss](/imgs/blogs/leverage-and-the-mortgage-how-debt-amplifies-property-1.png)

## Foundations: how a mortgage actually works

Before we can talk about amplification, we have to define the parts. Skim this if you already know them; do not skip it if you don't, because every later section leans on these exact words.

A **mortgage** is a loan secured by a piece of property. "Secured" means the property itself is the collateral: if you stop paying, the lender has a legal claim to take the property and sell it to recover the money. The word comes from old French — *mort* (dead) + *gage* (pledge) — literally a "dead pledge," because the pledge dies either when you pay it off or when the lender takes the house. That dark little etymology is worth remembering: a mortgage is a deadly serious promise backed by the roof over your head.

The money you borrow is the **principal** — the ₫5.6 billion Minh owes the bank, the \$282,000 the teacher owed. The price the bank charges you for the use of that money, expressed as a yearly percentage, is the **interest rate**. The **interest** is the actual money that percentage produces: 11% on ₫5.6 billion is roughly ₫616 million of interest in the first year alone.

The money you put in yourself — the part you don't borrow — is the **down payment**. Minh's is ₫1.4 billion; the teacher's was \$15,000. The down payment matters far more than its size suggests, because of the next two definitions.

**Loan-to-value (LTV)** is the loan divided by the property's value, as a percentage. Minh borrows ₫5.6 billion against a ₫7.0 billion flat, so his LTV is 5.6 / 7.0 = **80%**. The teacher's was 282 / 297 ≈ 95%. LTV is the single most important number in this entire post; hold onto it. A high LTV (90%+) means you borrowed almost the whole price and put in almost nothing. A low LTV (50%) means you put in half yourself.

**Equity** is what you actually own: the property's value minus the loan. On day one, Minh's equity is ₫7.0bn − ₫5.6bn = ₫1.4 billion — exactly his down payment. Equity is the part of the asset that is *yours*; the rest belongs, in a sense, to the bank until you repay it. The whole game of leverage is about how that equity number moves.

**Amortization** is the schedule by which you repay the loan in equal periodic payments that cover both interest and a slice of principal, so the balance reaches zero exactly at the end of the **term** — the loan's life, typically 20–30 years. A crucial, counter-intuitive fact lives inside amortization: in the early years almost the entire payment is interest, and only a sliver pays down principal. We will prove this with Minh's numbers and a chart in a moment. The plain-English name for it is **interest front-loading**.

A **fixed-rate** loan keeps the same interest rate for the whole term — the payment never changes. A **floating-rate** (or "variable" / "adjustable") loan has a rate that moves with the market, so the payment changes when rates change. Most US 30-year mortgages are fixed; most Vietnamese mortgages are floating after a short fixed window. That window is the next term.

A **teaser rate** (Vietnamese banks call it a "preferential" rate, *lãi suất ưu đãi*) is a low introductory rate offered for the first 12–24 months to win your business — often around 8% in Vietnam, sometimes as low as 5–7%. When that window ends, the loan **resets** to a floating rate, typically the bank's reference rate plus a margin, landing around 12–14%. The reset is not a penalty or a default — it is the deal you signed. It is also where most payment shock comes from, and we will give it its own section.

**Debt service coverage ratio (DSCR)** is the property's income divided by its loan payment. "Debt service" just means the loan payment. If a rental brings in ₫60 million a year and the mortgage costs ₫45 million a year, the DSCR is 60 / 45 ≈ 1.33 — the rent covers the loan with a third to spare. A DSCR below 1.0 means the rent does *not* cover the loan and the owner must top up the difference out of pocket every month. Banks and serious investors live and die by this number.

The **leverage ratio** is just the inverse of how much you put down: with 20% down you control five times your money (₫7.0bn of asset on ₫1.4bn of equity = 5:1), so your leverage ratio is 5×. With 50% down it's 2×. Higher leverage multiplies both gains and losses by exactly that factor — that's the whole mechanism, and the next section is devoted to it.

**Negative equity** (being **"underwater"**) is when you owe more than the property is worth — equity has gone *below zero*. The Las Vegas teacher, owing \$282,000 on a \$195,000 house, had −\$87,000 of equity. You can still live in the house, but you cannot sell without bringing cash to the table, and you cannot refinance.

**Margin of safety** is how far the price can fall before your equity hits zero — before you go underwater. As we'll see, it's almost exactly equal to your down payment percentage. At 20% down, a 20% fall wipes you out; at 50% down, you can survive a 50% fall.

Finally, **recourse vs non-recourse**. A **recourse** loan lets the lender come after your *other* assets and income if selling the house doesn't cover the debt — they can sue you for the shortfall, garnish wages, seize savings. A **non-recourse** loan limits the lender to the house itself: hand back the keys and you walk away, even if the house is worth less than the loan. Most Vietnamese mortgages are recourse; many US residential mortgages, in practice and in some states by law, behave closer to non-recourse. This distinction decides how badly negative equity actually hurts you, and we'll return to it.

That's the vocabulary. Now the engine.

## How leverage multiplies returns: the symmetric amplification

Here is the heart of the matter, and it is simpler than the jargon makes it sound. **You control the whole asset, but you only put in a slice of it. So any change in the whole asset's value lands entirely on your slice.**

Start with the simplest possible case: no loan at all. Dana, our US investor, buys a \$100,000 condo for cash. If it rises 10% to \$110,000, she made \$10,000 on \$100,000 — a **10% return**. If it falls 10% to \$90,000, she lost \$10,000 — a **−10% return**. With no leverage, the return on your money equals the return on the asset, one for one. Nothing is amplified.

Now add a loan. Dana buys the same \$100,000 condo, but puts \$20,000 down and borrows \$80,000 (80% LTV, 5× leverage). The condo rises 10% to \$110,000. The loan is still \$80,000 — *the debt does not share in the gain*. So her equity went from \$20,000 to \$110,000 − \$80,000 = \$30,000. She made \$10,000 on \$20,000 of her own money: a **50% return**. The asset moved 10%; her equity moved 50%. That factor of five is exactly her leverage ratio.

The price you pay for that magic is perfect symmetry on the downside. The condo falls 10% to \$90,000. The loan is still \$80,000. Her equity went from \$20,000 to \$90,000 − \$80,000 = \$10,000. She lost \$10,000 on \$20,000: a **−50% return**. The lever doesn't know which way you want it to push. It multiplies up and down by the same factor.

This is the picture in Figure 1, and it's worth stating the rule as a formula because it makes the symmetry undeniable:

$$\text{equity return} = \text{price return} \times \frac{\text{property value}}{\text{equity}} = \text{price return} \times \text{leverage ratio}$$

where the leverage ratio is the property value divided by your equity (your down payment, on day one). Every symbol here is something we defined above: price return is the percent the property moved, equity is your own money in the deal, and the leverage ratio is how many times your money the asset is worth. The fraction is always ≥ 1, so leverage can only magnify — never shrink — the percentage move. That's the entire mechanism.

#### Worked example: Minh's ±10% on ₫1.4 billion of equity

Minh buys a ₫7.0 billion (≈ \$270,000) flat with ₫1.4 billion (≈ \$54,000) down and a ₫5.6 billion (≈ \$216,000) loan. His leverage ratio is 7.0 / 1.4 = **5×**.

Prices in his district rise 10%. The flat is now worth ₫7.7 billion — a gain of ₫700 million (≈ \$27,000). His loan is unchanged at ₫5.6 billion, so his equity is now ₫7.7bn − ₫5.6bn = ₫2.1 billion. He turned ₫1.4 billion into ₫2.1 billion: a **+50% return** on his money from a 10% move in the flat. 10% × 5 = 50%. The lever worked.

Now run it the other way. Prices fall 10%. The flat is worth ₫6.3 billion — a loss of ₫700 million. The loan is still ₫5.6 billion, so his equity is ₫6.3bn − ₫5.6bn = ₫700 million (≈ \$27,000). He turned ₫1.4 billion into ₫700 million: a **−50% return**. Same flat, same lever, opposite sign.

Notice the asset only ever moved ₫700 million in either direction. What changed dramatically was the *percentage* impact on Minh, because his slice was small. The ₫700 million gain or loss is a small fraction of a ₫7 billion flat but half of Minh's ₫1.4 billion equity.

*Leverage doesn't make the property move more; it makes the property's move land harder on the smaller pile of money you actually put in.*

This is why real estate makes — and destroys — so much wealth. Property prices are not especially volatile compared to stocks; a 10% annual move is normal, not extreme. But almost nobody buys property with cash. The typical buyer is levered 4–5×, which turns an ordinary 10% market move into a life-changing 40–50% swing in their net worth. The leverage, not the asset, is where the drama lives. (If you want the cousin of this idea in stocks and REITs, where the same equation governs how rate moves hit levered property companies, the cross-asset view is in [real estate, REITs, income, leverage and rates](/blog/trading/cross-asset/real-estate-reits-income-leverage-and-rates).)

The symmetry of the formula is mathematically perfect, but human psychology is not symmetric, and this asymmetry is what makes leverage so dangerous in a boom. When prices are rising, the lever's *upside* is loud and visible: Minh's neighbour turned ₫1.4 billion into ₫2.1 billion in eighteen months and won't stop talking about it. The *downside* is silent — it's a counterfactual that hasn't happened yet, so it carries no emotional weight. Worse, a rising market continuously *rewards* the most leveraged players: the buyer who put 5% down made a far bigger percentage return than the cautious one who put 50% down, so the boom systematically transfers status, confidence, and capital to whoever took the most risk. This is exactly backwards from safety, and it compounds. As prices rise, banks see falling default rates and *loosen* lending (higher LTVs, thinner documentation), buyers see only winners and bid more aggressively, and the average leverage in the whole market quietly climbs to its maximum at precisely the moment prices are most stretched. The crowd is most leveraged exactly when the margin of safety is thinnest — which is why busts are so violent. The same lever that felt like genius on the way up is sitting under everyone's feet on the way down, and it multiplies the fall by the same factor it multiplied the climb. No participant chose to be reckless; the *boom itself* selected for leverage and called it skill.

## Amortization: where your payment really goes

We've been treating the loan balance as a fixed ₫5.6 billion. Over time it does shrink — but astonishingly slowly at first, and understanding *why* is essential to understanding leverage, because for the first decade you are barely building equity through repayment at all.

A mortgage payment is computed so that the same amount, paid every month, exactly zeroes out the loan at the end of the term. The formula is the standard annuity payment:

$$\text{payment} = P \cdot \frac{r(1+r)^N}{(1+r)^N - 1}$$

where $P$ is the principal (₫5.6 billion), $r$ is the *monthly* interest rate (the annual rate divided by 12), and $N$ is the number of monthly payments (25 years × 12 = 300). For Minh's loan at 11% annual, $r$ = 0.11/12 ≈ 0.917% per month, and the formula spits out a monthly payment of **₫54.9 million** (≈ \$2,120). That payment never changes if the rate is fixed — but where the money *goes* inside the payment changes enormously over the years.

Where does that intimidating formula come from? You don't need the derivation to use it, but the intuition is worth ten seconds because it demystifies the whole thing. The bank is solving one puzzle: "what equal monthly amount, paid $N$ times, has a *present value* today equal to the ₫5.6 billion I'm lending?" Money in the future is worth less than money today — a dong owed to the bank in month 300 is worth far less to it now than a dong owed next month, because the bank could have lent that dong out again in the meantime. Discounting all 300 equal payments back to today and setting the total equal to the principal produces exactly the fraction above. The numerator $r(1+r)^N$ and the denominator $(1+r)^N - 1$ are just the compressed algebra of "sum a stream of 300 equal future payments, each shrunk by compound interest, back to one number." The practical upshot is the only thing you must keep: a *higher rate* or a *shorter term* both push the payment up, and the rate matters far more than intuition suggests because it compounds across hundreds of months. This is the same present-value machinery that prices a bond — the mortgage is a bond you've issued to the bank, and you are the one paying the coupon. (If you want that present-value engine built from scratch, it is the [seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds): higher rate, lower present value, exactly as here.)

Each month, the interest you owe is computed on the *remaining* balance. Early on, the balance is huge, so almost the whole payment is eaten by interest, leaving only a sliver to reduce principal. As the balance slowly falls, the interest portion falls and the principal portion grows. By the final years almost the entire payment goes to principal. This is the interest front-loading we named earlier, and Figure 2 shows it for Minh's full 25-year loan.

![Stacked bars showing for each year of a twenty-five year loan how much of the payment is interest versus principal, with interest dominating the early years and principal dominating the late years](/imgs/blogs/leverage-and-the-mortgage-how-debt-amplifies-property-2.png)

#### Worked example: Minh's first payment vs his payment in year 15

Minh's monthly payment is ₫54.9 million, fixed (assume 11% for the life of the loan for now). Let's open up the very first payment.

Interest on the first payment = balance × monthly rate = ₫5,600 million × 0.917% = **₫51.3 million**. That leaves only ₫54.9M − ₫51.3M = **₫3.6 million** to actually pay down the loan. On his very first payment, **93% of Minh's money goes to the bank as pure interest** and 7% builds his equity. He pays ₫54.9 million and his loan balance drops from ₫5,600.0 million to ₫5,596.4 million.

Now jump to the start of year 15. By then he's paid the loan down to about ₫4,192 million. Interest on that payment = ₫4,192M × 0.917% = ₫38.4 million, leaving ₫54.9M − ₫38.4M = **₫16.5 million** of principal. The split has shifted from 93/7 toward 70/30 — but it took fourteen years of payments to get there.

Add it up across the whole loan and the scale is sobering: on a ₫5.6 billion loan at 11% over 25 years, Minh pays roughly ₫16.5 billion in total — about **₫10.9 billion of it pure interest**, nearly twice the amount he borrowed. The house cost ₫7 billion; the *financing* cost almost another ₫11 billion.

*For the first decade of a mortgage, you are renting money from the bank far more than you are buying the house from yourself.*

The reason interest is front-loaded isn't a trick the bank plays — it falls straight out of the math, and seeing why removes the suspicion that something unfair is happening. The payment is fixed, but the interest portion is recomputed *every month on whatever you still owe*. In month one you owe the whole ₫5.6 billion, so the interest charge is enormous and only a crumb is left for principal. That crumb shrinks the balance by a hair, so next month's interest is a hair smaller, leaving a slightly bigger crumb for principal. Each tiny extra dong of principal makes the *next* month's principal portion grow a tiny bit more — principal repayment compounds in your favour, but it starts from almost nothing, so it takes years to gather speed. The curve isn't a straight line down; it's a slow start that accelerates. That's why the midpoint of the *balance* — the month you've repaid half the loan — comes nowhere near the midpoint of the *term*. On Minh's 25-year loan, he doesn't cross the halfway-repaid mark until roughly year 19.

#### Worked example: what one extra point of interest costs Minh

The payment formula hides how violently the *rate* drives the cost, so let's make it visible. Hold everything fixed — ₫5.6 billion, 25 years — and only change the rate.

At 11%, Minh's payment is ₫54.9 million and his lifetime interest is about ₫10.9 billion. Bump the rate one single point to 12%: the payment rises to about **₫59.0 million** and lifetime interest to roughly **₫12.1 billion**. One percentage point of rate — the kind of move a central bank can deliver in a single meeting — adds about **₫1.2 billion of interest** over the life of the loan, more than 20% of the entire amount Minh originally borrowed. Drop instead to 8% and the payment falls to ₫43.2 million with lifetime interest near ₫7.4 billion. The spread between an 8% loan and a 13% loan on the *same flat* is over ₫6 billion in interest — almost the price of a second apartment.

*The rate, not the price of the house, is often the most expensive number in the whole transaction — and it's the one buyers scrutinise least.*

This front-loading has a direct consequence for leverage and risk, which the next figure makes vivid. Because principal barely moves early on, your equity in the early years comes almost entirely from price appreciation, not from paying down the loan. If prices are flat for the first five years, you've built almost no equity through your payments — you've mostly paid interest. The loan balance, your denominator of danger, stays stubbornly high.

![A declining curve showing the outstanding loan balance over twenty-five years, with the balance still near eighty-six percent of the original amount after ten years](/imgs/blogs/leverage-and-the-mortgage-how-debt-amplifies-property-3.png)

#### Worked example: how little Minh owes less after ten years

Intuition says that after 10 years of a 25-year loan — 40% of the way through — you'd have paid off a good chunk, maybe a third. Let's check.

Minh has paid ₫54.9 million every month for 120 months — about ₫6.59 billion in total payments, more than the ₫5.6 billion he borrowed. Yet his remaining balance is **₫4.83 billion** (≈ \$186,000). He has repaid only ₫771 million of principal — **just 14% of the loan** — after sending the bank ₫6.59 billion in cash. The other ₫5.82 billion was interest.

Put differently: after a full decade, Minh still owes **86%** of what he originally borrowed. His LTV has barely improved, so his margin of safety — how far prices can fall before he's underwater — has barely improved either. Almost all of his progress toward owning the flat outright comes in the *back* half of the loan.

*Amortization repays the loan, but it does so on a curve so back-loaded that for the first ten years you should assume the loan barely shrinks when you're sizing your risk.*

## DSCR: does the rent cover the loan?

So far we've assumed Minh lives in his flat. But a huge share of leveraged property is bought to rent out — and for a rental, there's a make-or-break question that has nothing to do with whether prices rise: **does the rent cover the loan payment?** The number that answers it is the debt service coverage ratio (DSCR), and it separates an investment that pays for itself from one that quietly bleeds you dry.

DSCR is income divided by debt service. Above 1.0, the rent covers the payment and the property is self-funding. Below 1.0, you are feeding the property every month out of your own pocket — and a property you have to feed becomes a forced sale the moment your other income wobbles. Banks set minimum DSCRs (often 1.2–1.25×) precisely so the loan survives a few empty months. Figure 6 shows the two worlds side by side.

![Two panels comparing a rental with debt service coverage below one where the owner must subsidise the property against a rental with coverage above one where rent covers the loan with room to spare](/imgs/blogs/leverage-and-the-mortgage-how-debt-amplifies-property-6.png)

#### Worked example: Minh tries to rent the flat out

Suppose Minh decides to rent the flat instead of living in it. The going rent for a flat like his is about ₫15 million a month (≈ \$580). His mortgage payment, we computed, is ₫54.9 million a month.

DSCR = rent ÷ payment = ₫15M ÷ ₫54.9M = **0.27**.

The rent covers barely a quarter of the loan payment. To hold this flat, Minh must top up roughly ₫40 million *every month* out of his salary — almost ₫480 million a year — just to keep the bank paid, before a single dong of property tax, maintenance, or a vacant month. This is the brutal arithmetic of buying residential property to rent in Vietnam right now: prices are so high relative to rents (HCMC trades around 32 years of income to buy, one of the steepest ratios on Earth) that the rental yield is tiny — roughly ₫180 million of annual rent on a ₫7 billion flat is about 2.6% gross — while the borrowing rate is 11–14%. When your yield is 2.6% and your loan costs 11%, leverage works *against* you on the income line; you pay more to borrow than the asset earns. Minh is betting entirely on price appreciation to bail out a deeply negative carry.

Now contrast Dana in a higher-yield US market. She buys a \$300,000 rental, puts \$60,000 down, borrows \$240,000 at 7%, and rents it for \$2,400 a month (\$28,800 a year). Her annual loan payment is about \$19,200. DSCR = 28,800 ÷ 19,200 = **1.5** — the rent covers the loan with half again to spare. Her property funds itself and throws off cash even in a flat market. Same leverage mechanics; completely different survivability, because the rent-to-price ratio is completely different. (How that rent-to-price relationship is valued — the cap rate and NOI — is the subject of [cap rates, NOI and the income approach](/blog/trading/real-estate/cap-rate-noi-and-the-income-approach); a low DSCR is just a low cap rate meeting a high loan rate.)

There's a subtlety worth pulling out, because the headline DSCR is more optimistic than reality. The number we computed used *gross* rent against the *loan payment alone*. A real owner pays more than the mortgage: property management (often 8–10% of rent), maintenance and repairs (budget ~1% of property value a year), insurance, any property tax, and — the silent killer — **vacancy**. A unit that sits empty two months a year is earning only ten months of rent but still owing twelve months of mortgage. Banks underwrite against this by computing DSCR on *net* operating income (rent minus operating costs, but before the loan) and by demanding a cushion above 1.0 — typically 1.2 to 1.25× — precisely so a few empty months or a surprise repair don't tip the property below break-even. A DSCR of exactly 1.0 isn't "safe"; it's the knife's edge, where the first vacant month or broken air-conditioner forces the owner to reach into their own pocket.

#### Worked example: how a vacancy and costs eat Dana's cushion

Dana's headline DSCR was 1.5, which sounds comfortable. Let's make it honest. Take her \$28,800 of annual rent and subtract real costs: 8% management (\$2,304), \$3,000 maintenance, \$1,800 insurance, \$3,000 property tax — about \$10,100 of operating cost, leaving net operating income of roughly **\$18,700**. Now her *true* DSCR against the \$19,200 loan payment is 18,700 ÷ 19,200 = **0.97** — she's actually slightly underwater on cash flow before a single vacant month. Add one month vacant (lose \$2,400 of rent) and net income drops to ~\$16,300, a DSCR of **0.85**: she now feeds the property roughly \$240 a month. The same deal that looked like it threw off cash at the gross-rent level is, once you count what landlords actually pay, a modest monthly subsidy. This is why experienced investors never quote gross DSCR — and why Minh's situation, already 0.27 on *gross* rent, is far worse than even that grim number once costs are layered in.

*The gross-rent coverage on the brochure is the best case; the number that decides whether you sleep at night is net of vacancy and every cost the rent has to cover before it reaches the bank.*

There is a deeper point hiding in Minh's 0.27. When the rental yield (≈2.6%) is far below the borrowing rate (11–14%), the gap is a **negative carry** — every month you hold the asset, the financing cost exceeds the income, so the position bleeds cash by construction. A levered asset with negative carry is not an income investment at all; it is a *pure bet on price*, dressed up as property. The borrower is effectively paying the bank a large monthly fee for the right to be exposed to price appreciation, and if appreciation doesn't arrive on schedule, the carry slowly grinds the equity down even with prices flat. This is the structural reason Vietnamese residential-for-rent maths so rarely works at current prices: you are not buying a yield, you are renting upside and paying dearly for it.

*A property whose rent can't cover its loan isn't an investment that pays you — it's a monthly bill you're hoping to sell to someone else at a profit before it drains you.*

## LTV and the margin of safety: how far price can fall before equity = 0

We now arrive at the most important risk idea in all of property, and it falls straight out of the definitions. Your **margin of safety** — how far the price can fall before your equity is wiped to zero and you go underwater — is almost exactly your down payment percentage. The borrowed share *is* the cushion you don't have.

The logic is one line. You go underwater when the property's value drops to equal the loan. Your loan is your LTV times the original value. So the value can fall by (1 − LTV) before it hits the loan. At 80% LTV, value can fall 20% before equity = 0. At 50% LTV, it can fall 50%. The cushion is exactly (1 − LTV), which is your down payment percentage. Figure 5 shows the two cases.

![A branching diagram showing that at eighty percent loan-to-value a twenty percent price fall wipes equity to zero while at fifty percent loan-to-value it takes a fifty percent fall](/imgs/blogs/leverage-and-the-mortgage-how-debt-amplifies-property-5.png)

This is also why the *capital stack* matters — the order in which a price fall is absorbed. The bank's loan is **senior**: it gets repaid first from any sale. Your equity is **junior**: you get whatever is left over *after* the bank is made whole. So the first losses always land on your equity, not the bank's loan. Figure 4 draws this stack — debt on top, paid first; equity underneath, first to lose.

![A two-layer capital stack with senior mortgage debt that is repaid first sitting above the owner's equity that absorbs the first loss in value](/imgs/blogs/leverage-and-the-mortgage-how-debt-amplifies-property-4.png)

#### Worked example: Minh at 80% LTV vs a more cautious 50% LTV

Minh bought at 80% LTV: ₫1.4 billion down, ₫5.6 billion loan, on a ₫7.0 billion flat.

How far can the price fall before he's underwater? The flat hits the loan value when it falls to ₫5.6 billion — a drop of ₫1.4 billion, which is **20%**. A 20.1% fall and Minh owes more than the flat is worth. His margin of safety is a thin 20%, exactly his down payment percentage. Given that HCMC apartment prices have swung far more than 20% inside a single cycle, that is a genuinely fragile position.

Now imagine Minh had instead put 50% down: ₫3.5 billion of his own money (≈ \$135,000), borrowing only ₫3.5 billion. His LTV is 50%. The flat now has to fall all the way to ₫3.5 billion — a **50%** crash — before his equity is gone. He can ride out a downturn that would have annihilated the 80%-LTV version of himself. The cost is that he tied up ₫2.1 billion more cash and his upside is smaller (his leverage ratio dropped from 5× to 2×, so a 10% price rise now lifts his equity 20%, not 50%).

That is the entire trade leverage offers, in one comparison: **more leverage buys more upside per dong and a thinner margin of safety; less leverage buys a thicker cushion and a smaller multiplier.** There is no free lunch and no setting that is "correct" for everyone — only a choice about how close to the edge you're willing to stand.

*Leverage doesn't change the asset; it sets how far a price fall is from wiping you out — and that distance is just your down payment percentage.*

#### Worked example: Minh's flat falls 25% — past the cushion

The 20% margin of safety is the point where equity hits exactly zero. But prices don't politely stop there. Suppose HCMC corrects 25% — well within historical range for a single cycle — and Minh's ₫7.0 billion flat is now worth **₫5.25 billion**. His loan, after a year or two of glacial amortization, is still essentially ₫5.6 billion. His equity is ₫5.25bn − ₫5.6bn = **−₫350 million**. He doesn't just have *zero* equity; he is ₫350 million (≈ \$13,500) **underwater** — he owes the bank ₫350 million more than the entire flat would fetch in a sale.

Stack the human consequences. His original ₫1.4 billion down payment is gone — 100% loss on his own money from a 25% fall in the asset, the leverage working in full reverse. He cannot sell without finding ₫350 million in cash to hand the bank at closing just to clear the loan. He cannot refinance, because no lender writes a loan on a flat worth less than the balance. And because Vietnamese mortgages are recourse (more on this shortly), walking away doesn't end it — the ₫350 million shortfall is a debt that follows him. A 25% price move — five points past his cushion — converts a ₫1.4 billion stake into a ₫1.75 billion hole.

*Once price falls past your down-payment cushion, leverage stops merely erasing your money and starts manufacturing debt you didn't have before — you can lose more than everything you put in.*

## The Vietnam teaser-then-reset trap: payment shock

Everything so far assumed Minh's rate stays put. In Vietnam it usually doesn't, and this is where leverage acquires a second, distinctly local fang. Vietnamese mortgages are overwhelmingly structured as a short **teaser** ("preferential," *ưu đãi*) period — a low fixed rate for the first 12 to 24 months — that then **resets** to a floating rate for the remaining 20-plus years. The teaser is the rate the salesperson quotes and the rate that makes the monthly payment look affordable at signing. It is not the rate you'll mostly pay.

In 2024–2026, teaser rates ran around 8% (some promotions as low as 5.3–7.2%), and the post-teaser floating rate landed around 12–14% — one state-owned-bank reference ceiling was cited near 13.9% in 2026. The reset roughly doubles the *rate*, which more than offsets the small amount of principal repaid during the teaser, and the monthly payment jumps sharply. Figure 7 shows exactly what Minh's payment does.

![A step chart of the monthly payment staying flat through the preferential period then jumping sharply when the loan resets to the higher floating rate](/imgs/blogs/leverage-and-the-mortgage-how-debt-amplifies-property-7.png)

#### Worked example: Minh's payment shock at reset

Minh's ₫5.6 billion loan starts at the 8% teaser for 24 months. At 8% over 25 years, his payment is **₫43.2 million** a month (≈ \$1,670). That's the number on the brochure, the number he budgeted around, the number that made him feel the flat was just barely affordable.

Two years in, he's repaid only about ₫153 million of principal — the loan is still ₫5,447 million — and the rate resets to 13% floating. Re-amortizing ₫5,447 million at 13% over the remaining 23 years gives a new payment of **₫62.2 million** a month (≈ \$2,400). His payment just jumped ₫19.0 million — a **44% increase** — overnight, on a loan he's barely begun to pay down.

For most Vietnamese households this is a genuine emergency. If ₫43.2 million was already 45% of household income, ₫62.2 million is 65% — past the point most budgets can absorb. The options are all bad: stretch and cut everything else, refinance into another teaser (if a bank will have you, and the new teaser also resets), or sell. And here is the cruel timing: payment shocks cluster when rates have risen across the whole market, which is exactly when property prices are soft and buyers are scarce — so the forced sale lands into a falling market. The teaser-reset trap is leverage's amplification working through the *payment* instead of the price, and it is the single most underappreciated risk in Vietnamese home buying.

*The payment you signed for is a marketing rate with an expiry date; the payment you'll actually live with is the reset rate, and you should stress-test your budget against that number before you sign, not after.*

## Recourse, refinancing, and rollover risk

Two more mechanisms decide how badly all of this can hurt — and they're the ones buyers think about least.

**Recourse** decides what happens after a forced sale that doesn't cover the loan. Recall the distinction: a non-recourse loan limits the lender to the property, so handing back the keys ends the story even if you're underwater. A recourse loan lets the lender pursue your other assets and income for the shortfall. In much of the US, residential mortgages are non-recourse in practice or by state law, which is why the 2008 wave produced millions of "strategic defaults" — owners who were deeply underwater simply mailed the keys to the bank ("jingle mail") and walked, because the law let them. **Vietnamese mortgages are recourse.** If Minh is forced to sell his ₫7 billion flat for ₫5 billion while still owing ₫5.4 billion, the ₫400 million shortfall doesn't vanish — the bank can pursue him for it, and his ₫1.4 billion down payment is gone on top. Negative equity in a recourse system isn't a clean exit; it's a debt that follows you.

This distinction isn't just legal trivia — it reshapes how borrowers *behave* in a crash, and the difference is large enough to move whole markets. Under non-recourse, a deeply underwater owner faces a clean financial calculation: if the house is worth \$195,000 and you owe \$282,000, continuing to pay is throwing good money after bad, because you're servicing \$282,000 of debt on a \$195,000 asset and the law lets you cap your loss at the house. Default becomes a rational option the moment negative equity is deep enough, and millions exercised it in 2008 — which *accelerated* the price collapse, because each strategic default dumped another distressed house onto the market, pushing prices down further and tipping the next marginal owner underwater. The non-recourse default is, perversely, a feedback loop that speeds the bust. Under recourse, that escape valve is welded shut: the underwater borrower keeps paying long past the point of financial sense, because walking away doesn't cap the loss — the bank chases the shortfall, garnishes wages, and freezes savings. Recourse borrowers default *less* and later, which dampens the fire-sale cascade but loads the pain onto households instead, who grind on under debt for years. Vietnam's recourse regime means a property bust there looks less like 2008's wave of walk-aways and more like Japan's slow, silent generation of over-indebted owners servicing loans on assets worth a fraction of the balance.

**Refinancing** is taking out a new loan to pay off the old one, usually to get a lower rate or a longer term. It's the escape hatch buyers count on — "if the reset hurts, I'll just refinance." Sometimes you can. But refinancing depends on three things that all tend to fail at once: the property must still appraise high enough (an LTV the new lender will accept), your income must still qualify, and *rates must be available that beat your current one*. When Minh's reset hits, it's because market rates rose — so the refinance market offers him 13%+ too, not relief. And if prices fell, his LTV may have ballooned past what any lender will refinance. The escape hatch is bolted shut exactly when you need it.

#### Worked example: the reset that no refinance can rescue

Picture Minh two years in, his 8% teaser expiring, and trace what each "escape" actually offers. His balance is about ₫5,447 million. The reset path re-amortizes that at the new 13% floating rate over the remaining 23 years: payment jumps from the ₫43.2 million teaser to **₫62.2 million** — the ₫19.0 million, 44% shock from the previous section. Now try to refinance out of it. Option one: a new bank's fresh teaser. But the new teaser is also ~8% for only 24 months and then resets to *its* floating rate — Minh would be paying ₫43.2 million again briefly, then facing the same ₫62.2 million wall two years later, having paid fees twice to kick the can. Option two: a genuine fixed refinance at the prevailing market rate. But the reset happened *because* market rates rose, so the best fixed loan on offer is itself near 13% — re-amortizing ₫5,447 million at 13% gives essentially the same ₫62.2 million payment. There is no rate in the market that beats his reset, because his reset *is* the market. Option three: sell. But rates rose market-wide, which cooled prices and thinned buyers, so the flat that was "worth" ₫7 billion now draws bids near ₫5.6 billion in a slow market — barely enough to clear the ₫5,447 million loan after a 2% transfer tax, leaving Minh with roughly nothing and his ₫1.4 billion down payment fully consumed. Every door leads to the same ₫62.2 million payment or a wipeout sale.

*Refinancing only rescues you when rates have fallen and your equity has held — and a reset that hurts is, by definition, proof that neither of those is true.*

That dependence on being able to roll into a new loan is **rollover risk** (or refinancing risk), and it's most acute for short-term and interest-only structures. Many commercial property loans, and Vietnam's developer financing, don't amortize to zero over a long term — they run for a few years and then the whole balance comes due as a **balloon payment** that the borrower must refinance or repay. If credit is tight or values have fallen when the balloon matures, the borrower can't roll it and defaults on an otherwise-performing loan. This is precisely what froze Vietnamese developers in 2022–23: a wave of corporate bonds came due, the bond market had seized after the Tân Hoàng Minh and Vạn Thịnh Phát arrests, and projects that were fundamentally sound on paper couldn't roll their financing. Rollover risk is leverage's quietest killer — the loan doesn't have to go bad for the borrower to go bust; the *refinancing* just has to be unavailable on the day it's due. (Why credit availability swings like this, drying up exactly when everyone needs it, is the liquidity cycle — covered in [global liquidity, the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide).)

## Common misconceptions

**"Leverage is free money."** Leverage is borrowed money, and it amplifies losses as ruthlessly as gains — by the exact same factor. The "free money" feeling comes from only ever looking at the upside case. A 5× lever that turns a 10% gain into 50% turns a 10% loss into −50% and a 20% loss into a total wipeout. There is no version of leverage that magnifies the good and not the bad; they're the same multiplier with opposite signs. What leverage actually buys you is *exposure you couldn't otherwise afford*, at the price of *fragility you might not survive*.

**"I can always refinance if it gets tight."** Refinancing requires that you still qualify, the property still appraises, and a better rate actually exists — and all three tend to disappear together, in a downturn, which is exactly when you need to refinance. Counting on a refinance is counting on the credit market being open and friendly on a future date you don't control. Stress-test your budget against the reset rate as if refinancing won't be available, because often it won't.

**"My payment is fixed forever."** In Vietnam, almost no mortgage is fixed for its full term — the comfortable rate is a 12–24 month teaser, after which the payment floats and typically jumps 30–50%. Even in the US, where 30-year fixed loans are genuinely fixed, that's a relatively unusual feature of one country's mortgage market, not a law of nature. Read your loan: find the words "preferential period," "reset," "reference rate plus margin," and compute what the payment becomes when the teaser ends.

**"Property prices never fall, so leverage is safe here."** This belief is the fuel for every property bust in history. US national prices fell about 27% from 2006 to 2012; Japanese property fell for two decades after 1990; HCMC and Hanoi had a deep freeze in 2011–13. Prices absolutely fall, and when they do, leverage is precisely what converts a survivable price dip into personal insolvency. The more confident a market is that prices can't fall, the more leverage it takes on — and the bigger the wipeout when they do.

**"A bigger loan means a bigger return."** A bigger loan means a bigger *multiplier* on whatever the price does — bigger gains if it rises, bigger losses if it falls, and a thinner cushion before zero. It only means a bigger return in the scenarios where the price rises. Across the full range of outcomes, more leverage means more *dispersion* — a wider spread of possible results, fatter on both tails — not a higher guaranteed return.

**"Paying the mortgage builds equity quickly."** For the first decade, amortization is so back-loaded that your payments barely dent the principal — Minh repays only 14% of his loan in 10 years despite paying in more cash than he borrowed. Early equity comes overwhelmingly from price appreciation, not from your payments. Don't assume that "I've been paying for years" means "I owe much less."

**"If the bank approved my loan, I can afford it."** A bank's approval protects the *bank*, not you. It checks that you can probably make the *teaser* payment today and that the property covers its own loan if it has to seize and sell — and even that second check is the whole point of the **LTV cap**. When a regulator or bank limits loans to, say, 70% of value, it isn't protecting your budget; it's ensuring the bank has a 30% cushion to recover its money in a forced sale. LTV caps are in fact a **macro-prudential tool** — central banks tighten them to cool a credit-fuelled boom and loosen them to stimulate, because the maximum LTV in a market directly sets how much leverage the whole system can take on. Vietnam's SBV and Singapore's MAS both move LTV ceilings deliberately to lean against property cycles. So an approval tells you the bank thinks *it* is safe at your LTV; whether *you* can absorb the reset, a vacancy, or a price fall is a question only your own stress-test answers — and it's a different question entirely.

**"Recourse vs non-recourse is lawyer stuff that won't affect me."** It decides whether negative equity is an inconvenience or a catastrophe. In a non-recourse system you can hand back the keys and cap your loss at the house; in a recourse system — which is most of the world, including Vietnam — the shortfall after a distressed sale becomes a personal debt the bank can chase through your wages and savings for years. Two identical underwater borrowers, one in California and one in HCMC, face completely different futures: one walks away, the other keeps paying on a flat worth less than the loan. Know which one you are *before* you sign, because it changes how much leverage you can responsibly carry.

## How it shows up in real markets

**The US 2008 negative-equity wave.** Between 2004 and 2007, US lenders wrote enormous volumes of high-LTV mortgages — 95%, 100%, even "piggyback" structures that left buyers with essentially no equity — on the shared belief that national prices couldn't fall. The S&P/Case-Shiller national index peaked at 184.6 in July 2006 and fell to 134 by February 2012, roughly **−27% peak to trough**. For a buyer with 5% down, a 5% fall already erased their equity; a 27% fall left a vast cohort tens of thousands of dollars underwater. At the depth of the crisis, roughly a *quarter* of all mortgaged US homes had negative equity. Because many of those loans were effectively non-recourse, millions chose strategic default — handing back keys they were no longer legally chained to. The lesson is the spine of this post: the houses didn't change; the leverage stacked on them turned an ordinary cyclical price decline into the worst financial crisis since the Depression.

Make it concrete with one household, using illustrative-but-representative numbers. Picture a buyer in Phoenix, Arizona — one of the hardest-hit markets — who bought a \$250,000 house in 2006 with a 100% loan (zero down, two stacked mortgages totalling \$250,000). On day one their equity was \$0 and their margin of safety was nothing: *any* fall put them underwater. Phoenix prices roughly halved from the peak, so by 2011 the house was worth about \$130,000 while they still owed close to \$245,000 — roughly **\$115,000 underwater** on a house that had been their entire down-payment-free bet on never-falling prices. Selling was impossible without bringing six figures to closing. But Arizona is a state where purchase mortgages are largely non-recourse, so the financially rational move was stark: stop paying, let the bank foreclose, and walk away owing nothing further. Multiply that single decision by millions of households making the same call, and you have the mechanism by which leverage didn't just amplify one family's loss — it manufactured the supply of distressed homes that drove the next price leg down. The 100% loan that felt like free homeownership in 2006 was, in structure, a coin flip with no cushion.

**Vietnam 2022–23, when the rates reset.** Vietnam's property and credit markets seized in late 2022. The Tân Hoàng Minh bond cancellation (April 2022) and the Vạn Thịnh Phát / SCB arrests (October 2022) froze the corporate-bond market that developers relied on to roll their financing — a textbook rollover-risk event. At the household level, the SBV's rate environment pushed floating mortgage rates toward 13–15%, and buyers who had signed during the cheap-money window watched their teasers reset into a market where prices had stalled and selling was slow. Transactions in many segments collapsed. Through 2023 the SBV cut policy rates four times and Decree 08 gave developers room to restructure bonds — policy stepping in precisely because leverage had over-tightened the whole chain from developer to buyer. It was the same machine as 2008, running on teaser resets and balloon maturities instead of subprime ARMs.

Put a household on it, with illustrative numbers in the spirit of Minh. A young HCMC couple signs in 2021 for a ₫4.0 billion (≈ \$155,000) apartment with ₫1.0 billion down, borrowing ₫3.0 billion at a 9% teaser — a payment near ₫25 million a month that took perhaps half their combined income. Two things then happen at once in 2022–23. First, the teaser expires and the rate floats to ~14%, pushing the payment to roughly ₫36 million — a ~45% jump that blows straight past what their budget can absorb. Second, the secondary market for their building cools so hard that realistic bids drift toward ₫3.4–3.6 billion, against a loan still near ₫2.9 billion. They are not underwater — they still have thin positive equity — but they are *cash-flow* insolvent: they cannot make the new payment and cannot sell quickly enough to escape it. With recourse looming over any shortfall and refinancing unavailable at a better rate, the realistic outcome is a distressed sale that hands back most of the original ₫1.0 billion down payment to transaction costs and the rate spread. No fraud, no exotic loan — just a teaser that reset into a market that had turned. That is the Vietnamese version of payment shock, and it ran through thousands of ordinary balance sheets in 2022–23.

**Japan after 1990 — leverage that never came back.** At the peak of Japan's bubble, the grounds of the Imperial Palace were said to be worth more than all the real estate in California, and buyers and corporations were levered to the hilt against ever-rising land. When the bubble burst, urban land prices fell for *more than a decade*, in some categories down 70–80% from the peak. Owners and banks who'd borrowed against bubble-era valuations were left with loans far exceeding collateral that kept shrinking — "zombie" borrowers servicing debt on assets worth a fraction of the loan. Japan shows the slow-motion version of the same truth: high leverage into a price peak doesn't just cause a sharp crash, it can sterilize an economy for a generation as everyone spends years paying down debt instead of investing.

**The 2021 US ultra-low-rate window — leverage's friendlier face.** It's not all carnage. In January 2021 the US 30-year fixed mortgage hit a record low of **2.65%**, and crucially it was *fixed for 30 years*. Buyers who locked that rate and saw prices rise — the national index climbed from ~260 in 2021 toward ~330 by 2026 — enjoyed leverage's upside with almost none of the reset risk that defines Vietnam, because their payment genuinely couldn't move. When rates later rose past 7%, those owners were sitting on a sub-3% loan they'd never refinance away — the "golden handcuffs" that froze the US existing-home market. This is the mirror image of the Vietnamese trap: the same leverage, but with a fixed rate, became a source of stability rather than shock. The structure of the loan, not just its size, decides whether leverage helps or hurts.

## When this matters / Further reading

This matters the moment you sign a mortgage — which for most people is the largest financial commitment of their lives. Before you sign, you can compute every number in this post for your own deal in ten minutes: your LTV (loan ÷ price), your margin of safety (1 − LTV, the price fall that wipes you out), your DSCR if you're renting it out (rent ÷ payment), and — if you're in Vietnam — your reset payment (re-amortize the balance at the floating rate, not the teaser). If the reset payment, the price fall you could survive, or the rent coverage looks frightening, that fear is information: it's telling you the leverage is too high for the cushion you have. The fix is always the same lever — more down payment lowers your LTV, thickens your margin of safety, and shrinks the multiplier on a price fall.

None of this is advice to buy, sell, or borrow any particular amount — it's the mechanism, so that whatever you decide, you decide it knowing exactly how the lever cuts both ways. The honest summary is the one we started with: leverage doesn't change the asset; it changes how far a price fall is from wiping you out.

To go deeper from here:

- [What is real estate as an asset class](/blog/trading/real-estate/what-is-real-estate-as-an-asset-class) — the bigger picture of why property is the largest, most levered asset on Earth.
- [Rent vs buy: the real math](/blog/trading/real-estate/rent-vs-buy-the-real-math) — how the financing cost in this post feeds the single most common housing decision.
- [Cap rates, NOI and the income approach](/blog/trading/real-estate/cap-rate-noi-and-the-income-approach) — the rental-yield side of DSCR, and why a low cap rate plus a high loan rate is a negative-carry trap.
- [Real estate, REITs, income, leverage and rates](/blog/trading/cross-asset/real-estate-reits-income-leverage-and-rates) — the same leverage equation seen through listed property companies.
- [Global liquidity, the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide) — why credit floods in and drains out on a cycle, driving the rate resets and rollover risk that decide when leverage bites.
