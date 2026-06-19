---
title: "Real-estate and housing law: zoning, rent control, and the GSEs"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Real-estate returns are set as much by law as by location. Zoning constrains supply, rent control caps income, and the government-sponsored enterprises plus mortgage rules set the cost of the loan that prices every home. Learn to read the legal layer and you understand the asset's return."
tags: ["regulation", "real-estate", "housing", "zoning", "rent-control", "fannie-mae", "freddie-mac", "mortgage", "gses", "1031-exchange", "property-tax", "valuation"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A home's price and a property's return are set as much by law as by location: zoning decides how much supply can be built, rent control caps how fast the income can grow, and the government-sponsored enterprises plus mortgage rules set the cost of the loan that prices every house.
>
> - A property is worth its **net operating income divided by the cap rate** — the yield buyers demand. The cap rate moves with interest rates, so the same building can be worth wildly different amounts in a 3% world and a 7% world with no change to the rent roll.
> - **Zoning is the supply lever.** Where the law makes building hard (discretionary review, density caps), supply cannot rise to meet demand, so prices climb and rental yields compress. The legal regime, not the land, explains why San Francisco and Houston price so differently.
> - **The 30-year fixed mortgage and the conforming market are legal artifacts.** Fannie Mae and Freddie Mac — the GSEs — buy and securitize "conforming" loans, which is *why* the 30-year fixed exists and why the bond market, not your local bank, sets the mortgage rate. That is the channel through which Fed policy reaches your house.
> - The single number to remember: a buyer with a fixed **\$1,686/month** budget could borrow **\$400,000 at a 3% mortgage rate but only about \$253,000 at 7%** — a 37% collapse in borrowing power, and the main reason a rate cycle reprices housing.

In the spring of 2019, two very different legal events were quietly resetting the value of American real estate, and almost nobody outside the industry connected them. The first was in Albany: in June, New York State passed the Housing Stability and Tenant Protection Act, the most sweeping expansion of rent regulation in a generation. Overnight, owners of roughly a million rent-stabilized apartments in New York City lost most of the levers they had used to raise rents over time. The second was in the bond market: the ten-year Treasury yield was sliding from about 2.7% toward 1.5% as the Federal Reserve pivoted from hiking to cutting. One event capped how fast a building's income could grow. The other lowered the rate at which that income was discounted into a price. Both moved real-estate values — in opposite directions, through completely different legal machinery.

This is the thing about real estate that most people miss. They treat it as a story about location, neighborhoods, and "they're not making any more land." All of that is real. But underneath every property's price sits a stack of laws that set the actual numbers in the valuation: how much can be built (zoning), how fast the income can grow (rent regulation), how cheap the financing is (the mortgage and GSE rules), and how the gains and the cash flows are taxed. Change a line in any of those bodies of law and you have changed the asset's return — often by more than any renovation or neighborhood trend ever could.

This post builds the whole picture from zero. We start with the one valuation idea everything hangs on — income, the cap rate, and value — and define every term a beginner needs. Then we go layer by legal layer: zoning as the supply lever, rent control as a cap on income growth, the GSEs and the 30-year fixed as the mortgage-rate channel, the conforming-versus-jumbo spread, the 1031 exchange and the tax timing it creates, and finally how a single mortgage-rule or GSE-reform change can reprice the entire asset class. Throughout we stay strictly neutral: we describe *how* these laws move prices — the mechanism, the magnitude, the timing — never whether any of them is good or bad policy.

![Diagram showing three legal layers feeding into property value: zoning sets supply, rent control sets income, mortgage law sets the rate](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-1.png)

The figure above is the mental model for the entire post. A property's value is fed by three legal layers. Zoning law sets how much supply exists. Rent control sets how much income the building is allowed to earn over time. GSE and mortgage law set the rate at which that income is financed and discounted. Read down the three columns and you have read the asset's return. Every section below is one column of this picture.

## Foundations: how real estate is valued, and the laws behind every input

Before we can trade anything, we need the plumbing. This section assumes no finance or law background and builds every term from scratch. The math is not hard — it is mostly division — but the division is exactly where the legal force lives.

### A property is worth its income divided by a yield

Start with the simplest possible idea. A piece of income-producing real estate — an apartment building, an office, a rental house — is a machine that produces cash. The question "what is it worth?" reduces to "how much cash does it produce, and what yield do buyers demand to own that cash stream?"

The cash the property produces, after running costs but before financing and taxes, is called the **net operating income**, or **NOI**. Take the rent the building collects, subtract the operating costs (property management, maintenance, insurance, property tax, vacancy), and what is left is NOI. It is the building's profit as an operating business, ignoring the mortgage.

The yield buyers demand is called the **capitalization rate**, or **cap rate**. It is simply NOI divided by price, expressed as a percent — the unlevered annual yield on the purchase price. A building that produces \$84,000 of NOI and sells for \$1.4 million has a cap rate of \$84,000 ÷ \$1,400,000 = 6%. Run that backward and you have the master valuation equation of all of real estate:

> **Value = NOI ÷ cap rate.**

That one line is the entire game. Value goes *up* when income goes up, and value goes *up* when the cap rate goes *down* (buyers accept a lower yield, usually because interest rates are low). Value goes *down* when income is capped or the cap rate rises. Every legal lever in this post works by pushing on one of those two numbers.

![Pipeline showing gross rent minus operating costs equals NOI, divided by cap rate, equals property value](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-2.png)

The figure traces the calculation left to right: gross rent comes in, operating costs come out, what is left is NOI, you divide by the cap rate buyers demand, and you have the value. Keep this in mind; the rest of the post is about which law sets which box.

#### Worked example: cap-rate valuation and the rate-driven repricing

Take an apartment building that collects \$120,000 a year in gross rent. Operating costs — management, maintenance, insurance, taxes, an allowance for vacancy — run \$36,000. NOI is \$120,000 − \$36,000 = **\$84,000**.

Now suppose the market cap rate for buildings like this is 6%, because interest rates are moderate. The value is:

- Value = \$84,000 ÷ 0.06 = **\$1,400,000**.

Then interest rates rise sharply, and buyers — who can now earn more on a Treasury bond and who finance purchases with pricier debt — demand a higher yield to own real estate. The cap rate widens to 7.5%. The rent roll has not changed by a dollar:

- New value = \$84,000 ÷ 0.075 = **\$1,120,000**.

The building lost **\$280,000, or 20% of its value**, with identical income, purely because the cap rate moved 1.5 points. The core idea: a property is a perpetual income stream, and when the yield the market demands rises, the price of that stream falls — exactly like a bond.

### The mortgage-rate sensitivity: why housing trades like a long-duration bond

For owner-occupied homes — the largest asset class on earth — there is no NOI and no cap rate. But the same logic runs through a different door: the mortgage. Most buyers do not pay cash; they buy the largest house whose *monthly payment* they can afford. So home prices are set less by the sticker price and more by the **monthly payment at the prevailing mortgage rate**.

That makes housing behave like a long-duration bond: its price is acutely sensitive to interest rates. When the mortgage rate falls, a given monthly budget supports a bigger loan, buyers bid up prices, and home values rise. When the mortgage rate rises, the same budget supports a far smaller loan, and prices come under pressure. We will quantify this precisely in a worked example below, because the magnitude surprises people.

### The four bodies of law that set the inputs

Here is the map of where this post is going. Four distinct legal regimes set the four inputs to a property's return:

- **Zoning and land-use law** set the *supply* — how much can be built, and therefore how scarce (and expensive) the existing stock is.
- **Rent control and rent stabilization** set a ceiling on the *income* — specifically on how fast the rent, and thus the NOI, can grow.
- **The GSEs, the conforming-loan rules, and the 30-year fixed mortgage** set the *cost and availability of financing*, which sets both the mortgage rate for homes and the cap rate for income property.
- **The tax code** — the mortgage-interest deduction, the 1031 exchange, and the property tax — sets the *after-tax return* and the *timing* of transactions.

Let us define each, then go deep on how each one moves prices.

### Zoning, by-right versus discretionary

**Zoning** is the body of local law that dictates what may be built on a piece of land: residential or commercial, how tall, how dense, how many parking spaces, how far back from the street. It originated in the early twentieth century in the United States and is now the single most important determinant of housing supply.

The crucial distinction is **by-right versus discretionary** approval. Under a **by-right** regime, if your project complies with the written zoning code, you get your permit more or less automatically — no public vote, no negotiation. Under a **discretionary** regime, even a compliant project must run a gauntlet of hearings, environmental review, and political approvals, any of which can delay it for years or kill it. By-right systems let supply expand to meet demand; discretionary systems let neighbors and politicians choke supply. That single legal difference explains an enormous share of the price gap between cities.

### Rent control and rent stabilization

**Rent control** (and its softer cousin, **rent stabilization**) is a law that caps how much a landlord may charge or raise the rent on a regulated unit. Hard rent control fixes the rent; stabilization, the more common modern form, limits annual increases to a percentage set by a public board and gives tenants strong renewal rights. Either way, the legal effect is the same in our valuation equation: it caps the growth rate of the income, which — as we will see — caps the value.

### The GSEs, conforming loans, and the 30-year fixed

This is the piece most investors least understand, so we will define it carefully. **Fannie Mae** (the Federal National Mortgage Association) and **Freddie Mac** (the Federal Home Loan Mortgage Corporation) are the two **government-sponsored enterprises**, or **GSEs**. They do not lend to home buyers directly. Instead, they *buy* mortgages from the banks that originate them, bundle thousands of those loans into a **mortgage-backed security** (MBS), guarantee the security against default, and sell it to bond investors worldwide. This is **securitization** — turning a pool of individual loans into a tradable bond.

A mortgage the GSEs are allowed to buy is called a **conforming loan**: it must be under a size cap (the **conforming-loan limit**) and meet underwriting standards. A loan above the cap is a **jumbo loan**, which the GSEs will not buy, so it stays on the originating bank's balance sheet. We will see the price difference this creates.

Two facts follow that are easy to miss. First, the **30-year fixed-rate mortgage** — a loan whose rate never changes for three decades, with no prepayment penalty — barely exists outside the United States. It is not a natural market product; it is a *legal and policy artifact* created by the New Deal-era mortgage system and sustained by the GSE securitization machine, because only a deep, liquid, government-backed MBS market can absorb the interest-rate and prepayment risk that a 30-year fixed loan throws off. Second, because conforming mortgages are securitized and sold as bonds, **the mortgage rate is set by the bond market, not by your local bank**. That is the channel through which Federal Reserve policy reaches your house.

### The tax layer: the MID, 1031, and property tax

Three tax features round out the legal stack. The **mortgage-interest deduction** (MID) lets homeowners (who itemize) deduct mortgage interest from taxable income, lowering the after-tax cost of borrowing. The **1031 like-kind exchange** lets an investor sell one investment property and roll the proceeds into another *without paying capital-gains tax now* — a deferral so valuable it shapes the entire transaction calendar of commercial real estate. And the **property tax** — an annual levy on assessed value — is a direct, recurring drag on NOI and on the cost of holding a home. Each of these is a lever the law can move.

With the vocabulary in place, we can now go deep on each lever, starting with the one that sets the supply.

## Zoning: the supply lever that sets the price floor

Of the four legal layers, zoning is the one that most directly explains why two physically similar cities can have home prices that differ by a factor of five. The mechanism is pure supply and demand, with the law standing on the supply hose.

Demand for housing in a desirable, job-rich metro grows roughly with population and incomes. If the legal regime lets builders respond — if new apartments and houses can go up by-right when prices rise — then supply expands, the new units absorb the demand, and prices grow only modestly. Rents track incomes, and rental yields (cap rates) stay healthy because you are not paying a scarcity premium. If instead the legal regime makes building slow, uncertain, and politically fraught, supply cannot respond. Demand piles into a fixed stock of housing, prices ratchet up year after year, and the yield on that ever-more-expensive stock compresses.

![Before-and-after comparison of by-right permitting versus discretionary review and their effect on supply, price, and yield](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-3.png)

The figure contrasts the two regimes. On the left, by-right permitting lets supply rise to meet demand, so price stays roughly flat and the rental yield stays near 6%. On the right, discretionary review — hearings, vetoes, multi-year approvals — keeps supply stuck while demand builds, so price runs up and the yield compresses toward 3.5%. The land did not change. The law did.

This is the deep point about constrained markets that investors must internalize: **a low cap rate is not always a sign of a safe, premium asset — it is often a sign of a legally constrained supply.** In a place like coastal California or much of greater London, the 3% cap rate on an apartment building reflects decades of restrictive zoning that has made the existing stock scarce. The scarcity is durable precisely because it is legal, not physical — you cannot build your way out of it, because the law will not let you. That makes the income premium sticky, but it also means the asset is priced for very low future income growth and is extraordinarily sensitive to interest rates (a low cap rate is a long duration). By contrast, a permissive market like Houston or much of the US Sun Belt runs higher cap rates because supply can always respond, so no scarcity premium accrues, but income can grow with the metro.

#### Worked example: what a cap rate implies about supply and growth

Take two identical apartment buildings, each producing \$84,000 of NOI today. One sits in a tightly zoned coastal city and trades at a **3.5% cap rate**; the other sits in a permissive Sun Belt metro and trades at a **6.5% cap rate**.

- Coastal value = \$84,000 ÷ 0.035 = **\$2,400,000**.
- Sun Belt value = \$84,000 ÷ 0.065 = **\$1,292,308**.

The coastal building is worth **86% more for the exact same income today**. The market is paying that premium for two reasons baked into the zoning: durable scarcity (no new supply will compete the rent down) and expected rent growth (constrained supply means rents rise with incomes). The core idea: the cap-rate gap between two cities is, to a large degree, a *zoning-stringency gap* priced into the asset — you are buying the local land-use law as much as the building.

The practitioner reads zoning as the supply signal. A market loosening its zoning — upzoning, by-right multifamily, eliminating parking minimums — is one where future supply will rise and scarcity premiums may erode, a headwind for existing-owner values even as it helps affordability. A market tightening (new growth controls, downzoning, moratoria) is one where the scarcity premium will harden. The legal change is the leading indicator; the price change follows.

There is a second-order effect that makes zoning even more powerful than the simple supply story suggests, and it is worth spelling out because it is where the durable scarcity premium really comes from. When the legal regime makes new supply slow and uncertain, it does not just raise prices today — it makes future supply *unresponsive* to future demand. Economists describe this as the "supply elasticity" of a housing market: in an elastic market (permissive zoning), a surge in demand is met by a surge in building, so prices barely move; in an inelastic market (restrictive zoning), the same demand surge crashes entirely into prices because supply cannot answer. The consequence for an investor is that constrained markets are not just expensive — they are *more volatile* in price terms, because every shift in demand or rates lands fully on the price rather than being absorbed by quantity. A market that cannot build is a market where prices do all the adjusting. That is why coastal supply-constrained metros show both the highest price levels and some of the largest peak-to-trough price swings in downturns: the law removed the safety valve.

This also resolves a puzzle that confuses many newcomers: how can a city have a severe "housing shortage" and sky-high prices at the same time as plenty of physically available land? The answer is that the shortage is *legal*, not physical. There may be land, but the zoning forbids building the dense housing that demand wants on it — single-family-only zoning, height caps, minimum lot sizes, and discretionary review combine to make the legally buildable supply far smaller than the physically buildable supply. The gap between what could be built and what is allowed to be built is the shortage, and it is written entirely in the land-use code. An investor who understands this stops asking "is there land?" and starts asking "what does the code allow, and is that likely to change?" — because the second question is the one that prices the asset.

## Rent control: a legal cap on income growth, and therefore on value

If zoning works on the *supply* side of our equation, rent control works directly on the *income* side — and through the valuation equation, straight onto the price.

Recall that Value = NOI ÷ cap rate, and that a building's value compounds when its NOI grows. A market-rate building can raise rents to whatever the market will bear; over a decade of healthy demand, its NOI might grow 3–4% a year, and because the value is the income capitalized, the value compounds right along with it. Now impose rent stabilization: a public board limits annual rent increases to, say, 1–2%, and tenants have near-permanent renewal rights so you cannot reset to market when a unit turns over. The NOI growth rate is now capped by law at a level far below market. The building is worth dramatically less — not because today's rent fell, but because the *future* income stream the buyer is purchasing has been legally truncated.

![Before-and-after comparison of an uncapped building with growing NOI versus a rent-stabilized building with capped NOI and lower value](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-5.png)

The figure lays the two side by side. The uncapped, market-rate building resets rents to market, grows NOI around 4% a year, and compounds in value at a low cap rate. The rent-stabilized building has increases capped by a board, sees NOI growth limited to around 1%, and reprices down 30–45% at a higher cap rate. Same bricks, two values, set by the rent law.

This is not a hypothetical. New York's 2019 law is the cleanest real-world case. Before 2019, owners of rent-stabilized apartments had legal pathways to raise rents over time — increases tied to capital improvements, vacancy bonuses when a tenant left, and eventual deregulation once rents crossed a threshold. The 2019 act closed nearly all of them: it eliminated vacancy decontrol, sharply limited what owners could recover for renovations, and made stabilization effectively permanent. The market repriced the affected buildings almost immediately. Publicly traded landlords with heavy rent-stabilized exposure saw their share prices and the appraised value of their portfolios fall, and transaction prices for stabilized buildings in the following years dropped by roughly a third or more, with some estimates of value declines for the most-affected assets in the 40-45% range. The income did not collapse; the *legally permitted growth* of the income collapsed, and the value followed.

#### Worked example: the value hit from capping NOI growth

Take a stabilized building that earns \$100,000 of NOI today. A buyer values it as a growing income stream. The right way to capitalize a *growing* stream is the Gordon growth form: Value = NOI × (1 + g) ÷ (cap rate − g), where g is the income growth rate. To keep the arithmetic clean, we will use a simpler comparison that gives the same result: the cap rate buyers demand widens when growth is capped, because a slower-growing stream is worth a lower multiple.

Before the law, the building can grow NOI ~4% a year, and buyers pay a **4.5% cap rate**:

- Uncapped value = \$100,000 ÷ 0.045 = **\$2,222,222**.

After the law, NOI growth is capped near ~1%, the building is now a no-growth asset, and buyers demand a **6.5% cap rate** to own a stream that will not keep up with costs or inflation:

- Capped value = \$100,000 ÷ 0.065 = **\$1,538,462**.

The value falls **\$683,760, or about 31%**, on identical current income. The core idea: rent control does not need to cut today's rent to destroy value — it only has to cap the *growth*, and because a building is priced as a growing perpetuity, capping the growth widens the cap rate and slices the price.

The practitioner reads pending rent-control legislation the way an equity analyst reads a pending tax hike on a single sector: as a direct repricing event for any landlord with exposure. The signal is the bill, the committee vote, the ballot measure; the trade is the repricing of the exposed names *before* the law takes full effect, because the value hit is in the expected future income, which the market discounts immediately.

## The GSEs and the mortgage-rate channel: how Fed policy reaches your house

We now come to the layer that prices the largest asset class of all — owner-occupied homes — and that quietly sets the cap rate for income property too. This is the financing layer, and at its center sit Fannie Mae and Freddie Mac.

Here is the chain, step by step. A bank originates a 30-year fixed mortgage to a home buyer. The bank does not want to hold a 30-year loan — it ties up capital and exposes the bank to thirty years of interest-rate and prepayment risk. So the bank sells the loan to a GSE. The GSE pools thousands of such conforming loans into a mortgage-backed security, stamps it with a guarantee against borrower default, and sells the MBS to bond investors — pension funds, insurers, foreign central banks, the Federal Reserve itself. Those bond investors price the MBS off the broader bond market: roughly the ten-year Treasury yield plus a spread for prepayment risk and the cost of the guarantee. The yield the MBS market demands *is* the mortgage rate offered back to the next home buyer.

![Diagram of the mortgage-rate transmission chain from Fed policy and the GSE guarantee through MBS yields to the 30-year rate and home price](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-4.png)

The figure traces it end to end. Fed policy (and its bond-buying or selling, called quantitative easing or tightening) moves the ten-year Treasury yield. The Treasury yield plus a spread sets the agency MBS yield. The GSE guarantee is what makes those loans poolable and the MBS market deep enough to clear at a tight spread. The MBS yield sets the 30-year fixed mortgage rate, which sets affordability — the monthly payment — which sets the home price. Pull on the policy rate at the left and the home price at the right moves.

This is *why* you cannot understand housing without understanding the bond market — a connection we develop in our companion piece on [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) and the policy-transmission mechanics covered across the [macro-trading](/blog/trading/macro-trading) series. The GSEs are the legal plumbing that converts a Fed decision into a mortgage rate. When the Fed bought trillions of dollars of agency MBS during its quantitative-easing programs, it was directly compressing the MBS spread and pushing mortgage rates to record lows — a policy choice that flowed straight into home prices. The mechanism is the same one behind [quantitative easing](/blog/trading/finance/quantitative-easing-explained-printing-money), pointed specifically at the mortgage market.

Two features of this legal architecture deserve a closer look, because they explain why American housing finance behaves so differently from the rest of the world's. The first is the **30-year fixed rate itself**. In most countries, the standard mortgage is variable-rate or fixed only for a short initial period, because no private lender wants to bet its own balance sheet on what interest rates will do over thirty years. The United States is the great exception, and the reason is entirely legal and historical: the federal mortgage system built in the 1930s (and the GSE securitization machine that grew out of it) created a market deep enough to absorb that risk. The lender originates the 30-year fixed loan but does not keep it — it sells the interest-rate and prepayment risk into the MBS market, where thousands of institutional investors price and hold it. Take that securitization machine away and the 30-year fixed, as a mass-market product, would likely disappear. It exists because the law and the GSEs make it exist.

The second feature is **prepayment risk**, which is why an MBS yields more than a Treasury of the same maturity. A US homeowner can refinance or pay off the mortgage at any time with no penalty — another legal feature, not a market given. That means the bond investor who owns the MBS faces an asymmetric bet: when rates fall, borrowers refinance and hand the investor their money back early (right when reinvesting it is least attractive); when rates rise, borrowers sit on their cheap loans and the investor is stuck holding a below-market bond. To compensate for that "negative convexity," MBS investors demand a spread over Treasuries. That spread is itself a price input: when it widens (in a crisis, or when the Fed stops buying MBS), mortgage rates rise relative to Treasury yields, and housing gets more expensive to finance even if the Fed has not moved at all. A practitioner watching the housing market watches the agency-MBS spread, not just the policy rate.

Now to the magnitude, which is the part people underestimate.

![Step chart of the Fed funds upper bound rising from 0.5 percent in 2022 to 5.5 percent in 2023](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-6.png)

The chart shows the Federal Reserve's policy-rate cycle from 2022 to 2024: the upper bound of the federal funds target rose from 0.5% in early 2022 to 5.5% by mid-2023. Through the MBS channel above, the 30-year fixed mortgage rate moved roughly in parallel, from about 3% to about 7%. That move did more to reprice American housing than any change in incomes, demographics, or construction costs over the same period.

#### Worked example: the payment, and the price it implies at constant affordability

Take a buyer who can afford a fixed monthly principal-and-interest payment. We will compute the loan that payment supports at two mortgage rates using the standard amortization formula for a 30-year (360-month) loan.

A **\$400,000** loan at a **3%** mortgage rate carries a monthly payment of about **\$1,686**. Now hold that exact \$1,686 budget fixed and ask: at a **7%** rate, how big a loan does the same payment support?

- Run the amortization formula in reverse with the payment fixed at \$1,686 and the rate at 7%, and the supported loan is about **\$253,500**.

So the same monthly budget that bought a \$400,000 mortgage at 3% buys only a **\$253,500** mortgage at 7% — a **37% collapse in borrowing power**. If buyers across the market are payment-constrained (most are), this is enormous downward pressure on the price a given household can bid.

![Bar chart showing a fixed 1,686 dollar monthly budget supports a 400,000 dollar loan at 3 percent but only 254,000 dollars at 7 percent](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-7.png)

The chart makes the cliff vivid: the same \$1,686/month budget supports \$400,000 at a 3% rate, \$314,000 at 5%, and just \$253,500 at 7% — green to amber to red as the borrowing power erodes. Put plainly: because buyers bid with monthly payments, the mortgage rate is a lever on the price the whole market can pay, and a doubling of the rate from 3% to 7% knocks more than a third off purchasing power.

Why, then, did US home *prices* not fall 37% when rates roughly doubled in 2022-23? Because of two legal artifacts that throttle supply at the worst possible moment. First, the 30-year fixed with no prepayment penalty created the **lock-in effect**: tens of millions of homeowners holding 3% mortgages had no reason to sell into a 7% market, because moving meant trading their cheap loan for an expensive one. Existing-home supply dried up. Second, new construction is gated by the same zoning that limits supply in good times. With both resale and new supply choked, prices stayed far stickier than the affordability math alone would predict — a vivid demonstration that the legal structure of the mortgage (the fixed rate, the no-penalty prepayment) feeds back into price dynamics, not just into the monthly payment.

## The conforming-versus-jumbo spread: the price of a legal boundary

The conforming-loan limit is a single number set each year, and it draws a hard legal line through the mortgage market. A loan at or below the limit is conforming — the GSEs will buy it, it gets securitized into the deep agency MBS market, and it prices off that liquid market. A dollar above the limit and the loan is jumbo — the GSEs will not touch it, so it has to be held on a bank's balance sheet or sold into the smaller, less liquid private market. The same borrower, the same house, the same down payment can get a different rate depending on which side of that legal line the loan falls.

Historically jumbo loans carried a *higher* rate than conforming loans — the "jumbo spread" — because they lacked the GSE guarantee and the deep MBS bid. In some periods, particularly after the 2008 crisis when banks competed hard for wealthy borrowers, the spread inverted and jumbos were briefly *cheaper*. But the point for our purposes is structural: a government-drawn line in the loan-size distribution creates a pricing discontinuity that has nothing to do with the borrower's creditworthiness and everything to do with whether the GSE securitization machine is available.

#### Worked example: the cost of the conforming-vs-jumbo spread

Take a borrower with a **\$700,000** mortgage in a year when the conforming limit is, say, \$766,550 in most areas — so this loan is conforming and qualifies for the GSE-backed rate of **6.5%**. Now take the same borrower in a high-cost transaction needing an **\$850,000** loan, which exceeds the limit and is therefore jumbo, priced at a **6.9%** rate — a **0.40-point** spread for crossing the legal boundary.

Compare the first-year interest cost on each (interest ≈ balance × rate in the early years, when little principal has amortized):

- Conforming \$700,000 at 6.5%: first-year interest ≈ \$700,000 × 0.065 = **\$45,500**.
- Jumbo \$850,000 at 6.9%: first-year interest ≈ \$850,000 × 0.069 = **\$58,650**.

To isolate the *spread* cost specifically, hold the balance constant: an \$850,000 loan at 6.9% versus a hypothetical 6.5% costs an extra \$850,000 × 0.004 = **\$3,400 a year**, or roughly **\$102,000 over the 30-year life** in extra interest, purely because the loan sits on the jumbo side of a legally drawn line. The takeaway: the conforming limit is a piece of administrative law that prices into every mortgage near the boundary — cross it and you pay for the loss of the GSE guarantee, regardless of how good a borrower you are.

This is also why the conforming limit itself is a policy lever worth watching. When regulators raise the limit (as they do most years to track home-price inflation), they pull more loans into the cheaper conforming bucket, marginally lowering financing costs for buyers near the old boundary and supporting prices at the high end of the conforming market. The legal number moves; the financing cost moves; the price follows.

## 1031 exchanges: a tax rule that times the entire market

We turn now from the rate layer to the tax layer, where one provision does more to shape the *timing* of commercial real-estate transactions than any market force: the 1031 like-kind exchange.

Normally, when an investor sells a property for more than they paid, the gain is a taxable capital gain. Section 1031 of the US tax code creates an exception for investment real estate: if you sell one property and reinvest the proceeds into another "like-kind" property within strict deadlines (45 days to identify the replacement, 180 days to close), you defer the capital-gains tax entirely. You can chain these exchanges for decades, rolling gains from property to property without ever paying tax — and under current law, if you hold until death, your heirs receive a stepped-up basis that can erase the deferred gain altogether ("swap till you drop").

The market effect is profound. The 1031 rule means a huge share of commercial real-estate sellers are not really sellers at all — they are *exchangers* who must redeploy their capital into another property fast, or face a large tax bill. This creates a constant, tax-driven bid for replacement properties and links transactions together: a sale on one side of the country forces a purchase on the other within 180 days. It also creates "lock-in," where an investor sitting on a large embedded gain is reluctant to sell at all, because selling without a 1031 triggers the tax. Any proposal to limit or repeal 1031 (which surfaces periodically in tax-reform debates) would be a genuine repricing event for the asset class, because it would remove a structural source of demand and unlock a wave of long-deferred selling. This is the same after-tax-return logic we develop in [tax law as a market force](/blog/trading/law-and-geopolitics/tax-law-as-a-market-force): the code does not just tax the return, it steers the timing of capital.

#### Worked example: the deferral value of a 1031 exchange

Take an investor who bought a property for \$1,000,000 and is selling it for \$1,600,000 — a **\$600,000** capital gain. Assume a combined federal-plus-state capital-gains rate of **30%** (long-term federal 20% plus the 3.8% net investment income tax plus a state rate, rounded).

- Tax due on a straight sale = \$600,000 × 0.30 = **\$180,000**.

Without a 1031, the investor nets \$1,600,000 − \$180,000 = \$1,420,000 to reinvest. *With* a 1031, the full \$1,600,000 rolls into the next property and the \$180,000 stays invested and compounding. If that deferred \$180,000 compounds at 7% a year for 10 years before any tax is eventually paid, it grows to \$180,000 × (1.07)^10 ≈ **\$354,000** — so the deferral is worth roughly **\$174,000** of extra compounded wealth on this single transaction (and potentially the entire \$180,000 if held until a stepped-up-basis event).

The takeaway: a 1031 is an interest-free, government-extended loan equal to the tax you would otherwise owe, and over a multi-decade holding chain that deferral compounds into one of the largest sources of return in all of real-estate investing. Change the rule and you change both the return and the willingness to transact.

## The mortgage-interest deduction and the property tax: subsidy and drag

Two more tax features deserve a section of their own, because they sit on opposite sides of the ledger and both feed straight into demand and value. One subsidizes ownership; the other taxes it every year.

The **mortgage-interest deduction** lets a homeowner who itemizes deductions subtract the interest paid on a home loan from taxable income. The legal effect is to lower the *after-tax* cost of borrowing: if you are in a 32% bracket and pay \$20,000 of mortgage interest, the deduction saves you \$20,000 × 0.32 = \$6,400 in tax, so your effective interest cost is \$13,600, not \$20,000. By making mortgage debt cheaper after tax than it looks before tax, the MID raises how much house a given household will rationally buy, which feeds into demand and price. Crucially, the MID is a *policy lever* — when US tax reform in 2017 roughly doubled the standard deduction and capped the deductibility of state-and-local and mortgage interest, far fewer households itemized, the marginal value of the MID fell for most owners, and economists estimated a measurable (if modest) drag on home prices at the high end of the market where the deduction had mattered most. The legal change moved the after-tax cost of owning, and the after-tax cost is what buyers actually optimize. This is precisely the wedge logic developed in [tax law as a market force](/blog/trading/law-and-geopolitics/tax-law-as-a-market-force): the code sets the after-tax return, and capital flows to the highest after-tax return.

The **property tax** runs the other way: it is an annual levy, set by local governments as a percentage (the "mill rate") of a property's assessed value, and it is a direct, recurring cost. For an income property, the property tax is subtracted right inside the NOI calculation — it is one of the operating costs — so a higher property tax lowers NOI and, through the valuation equation, lowers value. For a homeowner, the property tax is part of the monthly carrying cost alongside the mortgage payment, so it eats into the budget that determines how much house the household can afford. Reassessments, rate changes, and caps (like California's Proposition 13, which limits how fast assessed values can rise) are all legal events that move the recurring cost of holding real estate, and therefore its price.

#### Worked example: a property-tax change as a NOI drag

Take an income property with gross rent of \$120,000 and operating costs (before property tax) of \$24,000, in a jurisdiction with a property tax of \$12,000 a year. NOI is \$120,000 − \$24,000 − \$12,000 = **\$84,000**, and at a 6% cap rate the value is \$84,000 ÷ 0.06 = **\$1,400,000**.

Now the local government raises the property tax by half, to **\$18,000** a year — a \$6,000 increase. Nothing else changes:

- New NOI = \$120,000 − \$24,000 − \$18,000 = **\$78,000**.
- New value = \$78,000 ÷ 0.06 = **\$1,300,000**.

A \$6,000 annual tax increase cut the property's value by **\$100,000** — a 7% hit — because the tax is capitalized: at a 6% cap rate, every recurring \$1 of cost destroys about \$16.67 of value (1 ÷ 0.06). The takeaway: a property tax is not just an annual bill; it is a permanent claim on the income stream, and the market capitalizes a change in it straight into the price, exactly as it would a change in rent.

## How a mortgage-rule or GSE-reform change reprices the whole asset class

We have now walked all four legal layers. The deepest point of the post is that because these laws set the *inputs* to value rather than the value directly, a change to any one of them reprices not a single building but the entire asset class at once. The figure below collects the four layers and the law behind each.

![Matrix mapping each valuation input to the law behind it and its pricing effect: supply, income, rate, and taxes](/imgs/blogs/real-estate-and-housing-law-zoning-rent-control-and-the-gses-8.png)

The matrix is the post in one frame: supply is set by zoning and reprices through scarcity; income is set by rent control and reprices through capped NOI growth; the rate is set by the GSEs, the conforming limit, and the 30-year fixed, and reprices through the mortgage and cap rate; taxes are set by the MID, 1031, and property tax, and reprice through the after-tax return and the transaction timing. Read across any row and you have a complete cause-and-effect chain from statute to price.

Take the GSEs specifically, because they are the largest latent legal risk in the asset class. Fannie Mae and Freddie Mac have been in federal **conservatorship** since the 2008 crisis — effectively run by their regulator under an implicit government backstop, but not formally guaranteed by the full faith and credit of the Treasury. This is the source of the third common misconception we address below: people assume the GSEs are private companies, or alternatively that they are fully government-guaranteed, and neither is quite true. They occupy a deliberately ambiguous legal middle, and that ambiguity is itself a price input — the MBS market trades agency paper at a tight spread to Treasuries *because* it believes the government would not let the GSEs fail, even though the law does not require a bailout.

Now run the thought experiment that periodically appears in policy debates: a formal **GSE reform** that ended the implicit backstop, raised guarantee fees, or shrank the conforming footprint. Any of those would widen the MBS spread, raise the 30-year mortgage rate by some increment for the same Fed policy, and — through the affordability channel we quantified — push home prices down. The reverse is equally true: a reform that made the backstop explicit could tighten the spread and lower mortgage rates. The point is that **the price of every American home is partly a bet on the legal status of two companies most homeowners have never thought about.** That is the through-line of this entire series: read the rule, and you have read the asset.

## Common misconceptions

**"Rent control helps affordability."** Whether rent control helps the specific tenants who hold a regulated lease is one question; whether it improves affordability for a city overall is a different one, and the evidence on the second is the relevant fact for an investor. By capping the income a building can earn, rent control reduces the value of rental property, which reduces the incentive to build and maintain rental housing. The classic studied case is San Francisco's expansion of rent control in 1994: research found it reduced the supply of rental housing on affected properties by around 15% as owners converted units to condos or other uses, and that the supply reduction pushed up rents on the *unregulated* stock — partly offsetting, at the city level, the benefit to regulated tenants. The number to hold: a policy that protects existing tenants can simultaneously shrink the future rental stock, so "controls help affordability" is true for some renters and counterproductive in aggregate, depending on design. The valuation effect, by contrast, is unambiguous: cap NOI growth and you cut the building's value, as our 31% worked example showed.

**"Home prices only track interest rates."** Rates are the single most powerful short-term driver — our affordability example showed a 37% swing in borrowing power as rates moved from 3% to 7%. But prices did *not* fall 37% in 2022-23, because supply is set by a different body of law entirely. The 30-year-fixed lock-in effect froze resale supply (sellers would not give up 3% loans), and zoning kept new construction throttled. Rates set demand-side affordability; zoning and the mortgage's legal structure set supply; the price is where those two legal forces meet. Anyone modeling housing off rates alone will be badly wrong whenever the supply law dominates — which is often.

**"The GSEs are private companies."** Fannie Mae and Freddie Mac were once shareholder-owned, publicly traded companies, and their common shares still trade. But since 2008 they have been in federal conservatorship, run by the Federal Housing Finance Agency, with the Treasury holding senior preferred stock and an effective backstop. They are neither fully private (the government controls them and stands implicitly behind their MBS) nor fully public (there is no statutory Treasury guarantee, and the law could in principle let them fail). That hybrid status is exactly why the conforming mortgage market is so cheap and deep — and why GSE reform is a real, if slow-moving, repricing risk. The number to hold: the GSEs guarantee or hold a large majority of US residential mortgage debt, so their legal status is not a niche issue — it is the foundation of the mortgage rate on most American homes.

## How it shows up in real markets

**A rate-driven repricing (2020-2023).** The cleanest recent case is the round trip in mortgage rates. In 2020-21, the Fed cut its policy rate to near zero and bought agency MBS aggressively, compressing the spread and pushing 30-year mortgage rates to record lows near 3%. Through the affordability channel, borrowing power surged and US home prices rose roughly 40% in two years. Then in 2022-23 the Fed reversed: the policy-rate step-up shown in the chart above drove mortgage rates to about 7%. Affordability collapsed — the same payment now bought a third less house — yet prices fell only modestly, because the 30-year-fixed lock-in and zoning constraints starved the market of supply. The episode is a textbook demonstration of the whole chain: policy → MBS → mortgage rate → affordability, with the legal structure of the mortgage cushioning the price on the way down.

**A rent-control law hitting a REIT's value (New York, 2019).** When New York passed the Housing Stability and Tenant Protection Act in June 2019, publicly traded landlords with heavy rent-stabilized exposure repriced almost immediately. The largest owners of New York stabilized apartments saw the appraised value of those portfolios and their stock prices decline, and the private transaction market for stabilized buildings dropped by roughly a third or more over the following years, with the most-affected assets estimated down 40-45%. No building lost its tenants or its current rent; what was destroyed was the *legally permitted growth* of the income — and because real estate is priced as a growing income stream, the value fell with it.

**A zoning change unlocking value.** The supply lever cuts both ways. When a city upzones a parcel — say, rezoning a single-family lot or a low-rise commercial strip to allow dense multifamily — it can multiply the land's value overnight, because the legally buildable income on that land just jumped. Developers and land investors who anticipate or influence such changes capture the difference between the as-zoned value and the rezoned value. The mirror image is a downzoning or a new moratorium, which strands the development potential and cuts land value. In both directions the price moves on the legal change, often well before a single brick is laid — the market discounts the new zoning the moment it becomes likely, exactly as it discounts any rule change. This is the same regulatory-risk-as-a-price-input logic we develop in [regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor).

#### Worked example: the value an upzoning unlocks

Take a parcel currently zoned for a single house worth **\$800,000**. The city upzones it to allow a small apartment building that, once built, would produce \$180,000 of NOI and, at a 6% cap rate, would be worth \$180,000 ÷ 0.06 = **\$3,000,000**. Suppose construction of that building would cost **\$1,800,000**.

- As-zoned land value: **\$800,000** (worth one house).
- Upzoned residual land value = finished value − construction cost = \$3,000,000 − \$1,800,000 = **\$1,200,000**.

The legal change lifted the land's value from \$800,000 to \$1,200,000 — a **\$400,000, or 50%, gain** — without anyone touching the dirt, simply because the law now permits a more valuable use. The takeaway: land is worth the present value of its *legally permitted* best use, so a zoning change is a direct, immediate repricing of the land, captured by whoever owns it when the code changes.

## How to trade it: the housing-law playbook

Everything above lands here. Real-estate returns are set by four bodies of law, so reading the legal layer is reading the return. Here is the practitioner's checklist.

**Read the supply law first.** For any market, classify the zoning regime: by-right or discretionary, loosening or tightening. A constrained, discretionary market carries durable scarcity premiums (low cap rates) and the most interest-rate sensitivity; a permissive market carries higher cap rates and more income-growth potential but no scarcity moat. Watch for legal changes — upzoning, by-right multifamily, moratoria, growth controls — as the leading indicator of future supply and therefore of the scarcity premium. The legal change precedes the price change.

**Read the income law.** Map any rent-control or stabilization exposure in a property or a REIT. Pending legislation, ballot measures, and committee votes are the catalysts; the trade is the repricing of exposed landlords before the law bites, because the value hit lives in the *future* income and the market discounts it immediately. Remember the mechanism: rent control caps NOI growth, capped growth widens the cap rate, and a wider cap rate slices the value — a 31% hit in our worked example with no change to today's rent.

**Quantify the mortgage-rate sensitivity.** Because buyers bid with monthly payments, the 30-year mortgage rate is the master lever on demand-side affordability. Track the chain — Fed policy and QE/QT → ten-year Treasury → agency MBS spread → mortgage rate → borrowing power — and remember the magnitude: a move from 3% to 7% cut borrowing power 37%. But never model price off rates alone: the 30-year-fixed lock-in and zoning set the supply that determines how much of that affordability hit actually reaches prices. Rates move demand; the legal supply structure decides the price outcome.

**Watch the GSE and conforming-limit policy.** The conforming-loan limit, GSE guarantee fees, and any move on conservatorship or GSE reform are direct inputs to the mortgage rate and therefore to home prices. A rising conforming limit pulls more loans into the cheaper bucket and supports high-end prices; a reform that widens the MBS spread raises rates for the same Fed policy and pressures prices. The agency-MBS spread is the real-time signal to watch.

**Use the tax timing.** The 1031 exchange creates a structural, deadline-driven bid for replacement properties and links transactions across the country; understanding the 45/180-day clock helps you read commercial deal flow and anticipate the wave of selling that any 1031 repeal would unleash. The mortgage-interest deduction and property-tax rules set the after-tax cost of owning, and changes to either (a cap on the MID, a property-tax reassessment) feed straight into demand.

**Know what invalidates the view.** A bullish constrained-market thesis is invalidated by a credible supply unlock — a state preemption of local zoning, a by-right upzoning, a building boom — that erodes the scarcity premium. A bearish rate-driven thesis is invalidated when supply is so legally throttled (deep lock-in, tight zoning) that prices simply refuse to fall despite the affordability hit, as in 2022-23. A landlord thesis is invalidated by a rent-control expansion that caps the income growth you were underwriting. In every case the *law* is the variable to monitor, because the law sets the input and the input sets the price.

The discipline is the same one that runs through this whole series: a statute or a rule changes the rules of the game; markets discount the change into prices before it fully bites; the practitioner reads the legal layer early, sizes the repricing through the valuation equation, and knows which legal change would prove the view wrong. In real estate more than almost anywhere, the asset's return is written in the law before it is written in the land.

## Further reading & cross-links

- [Regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor) — how pending rules (including zoning and rent control) price into an asset as a discount before they take effect.
- [The legal mandate of a central bank](/blog/trading/law-and-geopolitics/the-legal-mandate-of-a-central-bank) — the statutory powers behind the Fed policy and QE that move mortgage rates through the GSE channel.
- [Tax law as a market force](/blog/trading/law-and-geopolitics/tax-law-as-a-market-force) — the after-tax-return logic behind the MID, the 1031 exchange, and how the code steers capital and transaction timing.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the policy mechanism that sits at the head of the mortgage-rate transmission chain.
- [Quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money) — how central-bank MBS buying directly compresses the spread that sets the mortgage rate.
- The [real-estate](/blog/trading/real-estate) and [macro-trading](/blog/trading/macro-trading) series — the cycle, valuation, and liquidity mechanics this legal lens sits on top of.
