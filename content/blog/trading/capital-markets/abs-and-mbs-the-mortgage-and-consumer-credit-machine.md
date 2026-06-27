---
title: "ABS and MBS: The Mortgage and Consumer-Credit Machine"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How millions of household loans get pooled, sliced, and turned into the second-most-liquid bond market on earth — and why prepayment, not default, is the risk that defines it."
tags: ["capital-markets", "securitization", "mbs", "abs", "mortgage-backed-securities", "prepayment-risk", "cmo", "fixed-income", "structured-finance"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Securitization turns millions of ordinary household loans (mortgages, car loans, credit-card balances) into tradable bonds, and the two giant families are MBS (mortgage-backed) and ABS (everything else).
>
> - **Agency MBS** — guaranteed by Fannie Mae, Freddie Mac, or Ginnie Mae — is the second-most-liquid market on earth after US Treasuries, with roughly \$1.5 trillion issued in a single year.
> - The defining risk of MBS is not default; it is **prepayment**. Homeowners refinance when rates fall (handing you your cash back to reinvest at a worse yield) and stay put when rates rise (stretching your money out at a below-market coupon). You lose both ways — that's **negative convexity**, and it's why MBS pay more than Treasuries.
> - **CMOs** carve a mortgage pool into time-based tranches so different investors can buy the prepayment profile they want; **ABS** does the same trick for auto, card, and student loans.
> - The one number to remember: a refinancing wave can cut your bond's expected life from **7 years to 2** almost overnight.

## A Tuesday in 2003

Consider a portfolio manager at an insurance company in early 2003. She owns a big slug of mortgage-backed bonds yielding a comfortable 5.5%, bought when mortgage rates were higher. Her models say these bonds have an average life of about seven years — plenty of time to clip those coupons. Then the Federal Reserve cuts rates, 30-year mortgage rates fall toward 5%, and across America millions of homeowners do the rational thing: they refinance. They take out a new, cheaper loan and pay off the old one in full.

Every one of those payoffs flows straight back to her. Within months, a bond she expected to hold for years is handing her principal back in a flood. She now has to reinvest that cash — but rates have fallen, so the best she can do is buy new mortgage bonds yielding 5%. Her 5.5% income stream is evaporating exactly when she'd most like to keep it. She did nothing wrong. She simply owned the one bond in finance that gets *shorter* precisely when you want it to stay long.

That asymmetry — the borrower holds an option, and you are short it — is the heart of this entire market. It is also a perfect illustration of the series' spine: securitization is a *primary-market technology* for manufacturing tradable securities out of loans, but it only works because a deep *secondary market* will buy and trade those securities every single day. Nobody would lend \$300 million against a pool of mortgages they couldn't sell tomorrow morning. Let's build the machine from the ground up.

![Homeowner payments flow through a servicer and SPV to investors](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-1.png)

## Foundations: what securitization actually does

Start with a single loan. A bank lends you \$300,000 to buy a house. That loan is an *asset* to the bank: a promise that you'll pay roughly \$1,800 a month for 30 years. It's a perfectly good asset, but it's illiquid — the bank can't easily sell one random mortgage, and it ties up capital the bank could be lending again.

**Securitization** is the trick that fixes this. You take thousands of these loans, drop them into a legal box, and sell pieces of the box as bonds. If you've never read the primer, the full first-principles version lives in [securitization from first principles: turning loans into bonds](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) — here we'll move fast through the basics and spend our time on the two specific product families.

A few terms you need before anything else makes sense:

- **The pool** — the collection of loans (the mortgages, the car loans) whose payments back the bonds. Also called the *collateral*.
- **The SPV (special-purpose vehicle)** — the legal "box," a bankruptcy-remote trust that legally *owns* the pool. It exists for one deal and nothing else. Bankruptcy-remote means if the bank that made the loans goes bust, the SPV's loans are walled off and still belong to the bondholders.
- **The servicer** — the company that does the day-to-day grunt work: collects each borrower's monthly payment, chases late payers, handles foreclosures, and passes the cash up to the trust. It keeps a small fee (often 0.25%–0.50% a year of the pool) for doing this.
- **The certificate or note** — the bond the investor actually buys: a claim on the pool's cash flows.

So the assembly line is: **loans → pool → SPV → bonds → investors**, with the servicer standing between borrowers and the trust, sweeping cash upward every month. The genius is what it does to liquidity. One mortgage is unsellable; a \$1 billion pool sliced into standardized, rated bonds trades in a market with thousands of buyers. The same loan that was frozen on a bank's balance sheet becomes a security a pension fund in Tokyo can buy before lunch. That conversion — *illiquid household credit into deep, tradable secondary markets* — is the whole point of the machine.

### The two families: MBS and ABS

Everything securitized falls into two big families, split by what's in the pool:

- **MBS — mortgage-backed securities.** The pool is home loans. This is by far the biggest securitized market.
- **ABS — asset-backed securities.** The pool is anything *else*: auto loans, credit-card receivables, student loans, equipment leases. ABS is the non-mortgage cousin.

The reason we split them is that mortgages behave differently from car loans, and that difference drives the entire design of each market. We'll take MBS first because it's bigger, older, and stranger.

### Why this was invented in the first place

It's worth pausing on *why* anyone built this contraption, because the motivation explains every design choice that follows. Before securitization, a bank that made a mortgage held it to maturity. That created three problems. First, the bank's capital was frozen for 30 years — every loan it made was a loan it couldn't make again until the borrower paid off. Second, the bank carried all the risk of that single loan on its own books: if the local economy soured, its whole mortgage book soured together. Third, mortgage credit was *local* — a bank in a slow-growing town had spare deposits but few good borrowers, while a bank in a booming one had eager borrowers but not enough deposits, and there was no easy pipe to move money between them.

Securitization solved all three at once. The originating bank sells the loan into a pool, gets its cash back immediately, and lends again — capital recycles instead of freezing. The risk gets spread across thousands of investors instead of concentrated on one balance sheet. And savings from anywhere on earth can now fund a mortgage anywhere in America, because the connecting pipe is a liquid bond. That last point is the series' spine in one sentence: **securitization is how the capital market routes savings to their best use across geography and time.** The whole machine exists to turn a frozen, local, lumpy asset into a liquid, national, standardized one.

## MBS: the biggest securitized market

How big? Look at where mortgage debt sits inside the broader US bond market.

![US bond issuance by type in 2023 shown on a log scale](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-2.png)

In a normal year the US issues well over a trillion dollars of mortgage-backed securities — second only to Treasuries among debt sectors, and dwarfing corporate, municipal, and non-mortgage ABS issuance. The total stock of US mortgage debt runs into the *tens* of trillions. When people say "the bond market," a huge slice of what they mean is mortgages wearing a bond's clothing.

### Agency vs private-label

MBS splits into two halves, and the distinction matters more than almost anything else in the market.

**Agency MBS** is backed by mortgages that conform to the standards of, and are guaranteed by, one of three government-linked entities:

- **Fannie Mae** and **Freddie Mac** — government-sponsored enterprises (GSEs) that buy conforming mortgages and guarantee the bonds against borrower default.
- **Ginnie Mae (GNMA)** — a literal arm of the US government that guarantees bonds backed by FHA/VA loans, carrying the full faith and credit of the United States.

The guarantee means the investor takes *no credit risk* on the borrowers. If a homeowner defaults, the agency makes the bondholder whole. What you're left holding is essentially a US-government-credit bond — which is why **agency MBS is the second-most-liquid market in the world after Treasuries**. Hundreds of billions of dollars trade daily, often through a clever forward market called **TBA (to-be-announced)**, where you buy "\$10 million of 5.5% Fannie 30-year for July delivery" without knowing the exact pools you'll receive. That fungibility is what makes the market so deep.

**Private-label MBS (PLS)** has no agency guarantee. It's backed by loans that don't conform — too big (jumbo loans), or made to weaker borrowers (the infamous subprime). Here the investor *does* bear credit risk, so these deals are sliced into credit tranches (senior/mezzanine/equity) to redistribute default losses. Private-label was the epicenter of 2008; we'll come back to that. For now, hold onto the split: **agency = no credit risk, all about prepayment; private-label = credit risk too.**

It's worth being precise about what the agency guarantee does and doesn't cover. The guarantee makes the bondholder whole on *credit* — if a borrower defaults and the foreclosure recovers less than the loan balance, the agency pays the difference. But a default is, from the bondholder's cash-flow point of view, just an *involuntary prepayment*: the loan is removed from the pool and the principal is returned (by the agency) at par. So even the agency guarantee doesn't shield you from timing — a wave of defaults shortens your bond exactly like a wave of refinancing would. This is the subtle thing beginners miss: the guarantee converts credit risk *into* prepayment risk. You never escape timing; you only choose how much credit risk to layer on top of it.

### How the TBA market makes agency MBS trade like cash

A big reason agency MBS is so liquid is a market structure that has no equivalent in corporate bonds: the **TBA (to-be-announced)** forward. When a trader buys "\$25 million of Fannie Mae 5.5% 30-year for August settlement," they are *not* buying specific identified pools. They're buying a promise to receive *some* pools meeting agreed criteria (issuer, coupon, maturity, settlement month), with the exact pool numbers "announced" only 48 hours before settlement. Because any conforming pool of that coupon is an acceptable delivery, the bonds are **fungible** — a buyer doesn't care which Ohio or Texas mortgages they get, only the coupon and term. That fungibility is what lets agency MBS trade in enormous size with tiny bid-ask spreads, the way Treasuries do. It also gives mortgage lenders a forward market to *hedge* loans they haven't even closed yet: a lender who quotes you a rate today can immediately sell a TBA to lock in the price at which it'll later sell your loan. The liquidity of the secondary TBA market is, quite literally, what lets your local lender quote you a rate on the spot — the spine again.

### The pass-through structure

The simplest MBS is a **pass-through**. The name is literal: the homeowners' monthly payments are collected and *passed through*, pro-rata, to the bondholders. There's no slicing, no reordering — if you own 1% of the pool, you get 1% of every dollar that comes in that month, whether it's scheduled interest, scheduled principal, or an early payoff.

Each monthly check to an investor blends three things:

1. **Interest** on the outstanding balance.
2. **Scheduled principal** — the small chunk of the loan that amortizes each month.
3. **Prepayments** — extra principal from anyone who refinanced, sold their house, or just paid early.

That third bucket is the wild card, and it's why an MBS feels nothing like a Treasury. A Treasury pays you fixed coupons and then your whole principal back on one known date. An MBS dribbles principal back every single month at a rate *you can't predict*, because it depends on what millions of strangers decide to do with their home loans.

#### Worked example: a pass-through on a \$300M pool

Take a \$300,000,000 pool of 30-year mortgages with a 6% weighted-average coupon, and suppose the servicer keeps a 0.50% fee, so investors receive a 5.5% **pass-through rate**.

- **Interest to investors in month 1:** the pool earns 6% annually on \$300M, which is \$300{,}000{,}000 \times 0.06 / 12 = \$1{,}500{,}000 gross. The servicer skims 0.50%/12 on the balance, about \$125{,}000, leaving roughly \$1{,}375{,}000 of interest passed through.
- **Scheduled principal in month 1:** on a 30-year amortizing pool the first month's scheduled principal is small — for these numbers, about \$300{,}000.
- **Prepayments:** suppose 0.5% of the pool pays off early this month — that's another \$300{,}000{,}000 \times 0.005 = \$1{,}500{,}000 of principal arriving unscheduled.

So the investor's month-1 check is roughly \$1{,}375{,}000 + \$300{,}000 + \$1{,}500{,}000 = \$3{,}175{,}000, of which \$1.8M is principal being returned. The intuition: more than half of your "income" this month was actually your own principal coming back early, which you now have to reinvest — and that's before rates have even moved.

### How the market measures prepayment: CPR and PSA

To trade these bonds you need a language for "how fast is the pool prepaying." The market uses two:

- **CPR (conditional prepayment rate)** — the *annualized* fraction of the pool that prepays. A 6% CPR means about 6% of the remaining balance pays off early each year. The monthly version, **SMM (single monthly mortality)**, is just the monthly equivalent.
- **PSA** — a benchmark *ramp* published by the (old) Public Securities Association. **100% PSA** means prepayments ramp from 0.2% CPR in month 1, rising 0.2% each month, up to 6% CPR in month 30, then a flat 6% CPR thereafter. Quotes are stated as a multiple: "200% PSA" is twice that speed (peaking at 12% CPR); "50% PSA" is half.

Why a ramp and not a flat number? Because new loans rarely prepay — people don't refinance a mortgage they took out last month — so prepayment realistically *builds* over the first couple of years as the pool seasons. PSA encodes that seasoning curve. When a trader says a bond is priced "at 150 PSA," they're telling you the entire assumed cash-flow schedule in three characters.

#### Worked example: the same pool at 100 vs 300 PSA

Take a seasoned \$300,000,000 pool currently running at **6% CPR (100% PSA)**, and suppose a rate drop pushes it to **18% CPR (300% PSA)**.

- At 6% CPR, roughly \$300{,}000{,}000 \times 0.06 = \$18{,}000{,}000 of principal prepays over the next year (plus scheduled amortization).
- At 18% CPR, roughly \$300{,}000{,}000 \times 0.18 = \$54{,}000{,}000 prepays — *three times* as much cash hits your account in the same year.
- If you paid a **premium** for this bond (say 102 cents on the dollar because its 6% coupon beat the market), every prepaid dollar comes back at *par* (100), so you eat a 2-cent loss on \$54M instead of \$18M — a far bigger hit to your premium.

The intuition: PSA isn't academic — tripling the prepayment speed triples the rate at which a premium bond bleeds its premium back to par, which is exactly why falling rates hurt MBS holders.

## Prepayment risk: the feature that defines MBS

Here is the single most important idea in the whole post. When a homeowner takes a mortgage, they get a free **option to prepay** — to pay off the loan early without penalty. They didn't pay for it; US mortgages just come with it. And every option that one party holds is an option the other party is *short*. As an MBS investor, **you are short the homeowner's refinancing option.** That one fact reorganizes everything.

![Prepayment hurts whether rates fall or rise](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-3.png)

Watch how it plays out in both directions:

- **When rates fall**, refinancing becomes attractive. A homeowner with a 6% loan refinances into a 4.5% loan, paying off your bond at par. You get your cash back early — and you reinvest it at the new, *lower* market yield. Your high-coupon bond vanishes exactly when high coupons are scarce. Your **average life shortens** (this is *contraction risk*).
- **When rates rise**, nobody refinances — why trade a 6% loan for a 7.5% one? Prepayments dry up, your bond keeps paying you that 6% for years longer than expected. Sounds good? It isn't: you're now locked into a *below-market* yield when you'd rather have your cash to buy the new 7.5% bonds. Your **average life extends** (this is *extension risk*).

So you lose both ways. When rates fall you'd love your bond to stay long, and it shortens. When rates rise you'd love it to be short so you can reinvest, and it lengthens. The bond always moves the *wrong* way for you. That property has a name.

### Negative convexity

Most normal bonds have **positive convexity**: when rates fall, their price rises *faster and faster*; when rates rise, their price falls *slower and slower*. It's a friendly curvature — the bond rewards you on the upside and cushions you on the downside.

MBS has the opposite: **negative convexity**. As rates fall, the price *should* rise, but the looming wave of prepayments caps the gain — your bond can't trade much above par because everyone knows it'll just get called away at par by refinancing. The price ceiling is sometimes called the **prepayment wall**. As rates rise, extension makes the bond longer and more rate-sensitive right when you don't want it, so losses accelerate. Capped upside, accelerated downside: the worst of both.

This has a brutal practical consequence for anyone *hedging* MBS. A bond's **duration** (its sensitivity to rates) is supposed to be a stable number you can hedge against. But MBS duration *changes as rates move* — it shortens when rates fall (prepayments loom) and lengthens when rates rise (extension). So a manager who hedged their MBS book yesterday wakes up today, after a rate move, holding a position with the *wrong* hedge — the duration drifted out from under them. They have to constantly re-hedge, *selling* into falling-rate rallies and *buying* into rising-rate selloffs to keep the hedge matched. That forced trading is one reason big rate moves can become self-reinforcing in the Treasury and swap markets: a wall of MBS hedgers all dynamically adjusting in the same direction at the same time. The negative convexity of a household's refinancing decision propagates, through hedging, all the way into the price of government bonds. The full mechanics of duration belong to [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance); the takeaway here is that the homeowner's free option doesn't just sit quietly in your portfolio — it forces you to trade.

#### Worked example: a refinancing wave halves your average life

Suppose you bought \$10,000,000 face of a 5.5% pass-through expecting a **7-year average life**, planning to earn 5.5% for those seven years. Rates drop 1.25%, and a refinancing wave hits. Prepayments surge, and the bond's expected life collapses to **2 years** — most of your principal comes back inside 24 months.

- You expected: \$10{,}000{,}000 earning 5.5% for ~7 years.
- You get: most of your \$10M back inside 2 years, which you must reinvest at the new market rate of, say, **4.25%**.
- The lost income on the early-returned cash is roughly \$10{,}000{,}000 \times (5.5\% - 4.25\%) = \$125{,}000 per year of yield you no longer earn, compounding over the years you *thought* you had the 5.5% locked in.

The intuition: your seven-year 5.5% income stream got amputated to two years exactly when 5.5% became unavailable — the option you were short got exercised against you at the worst moment.

### Why MBS yields more than Treasuries

If you're short an option, you should be *paid* for it. You are. Agency MBS carries no credit risk (the agency guarantees it), yet it trades at a yield meaningfully above the comparable Treasury — historically anywhere from ~0.4% to well over 1% extra. That spread, the **option-adjusted spread (OAS)** once you strip out the prepayment option's value, is your compensation for being short the homeowner's refi option and for living with negative convexity. The deeper math of how that spread relates to the yield curve lives in [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance); the point here is simpler: **MBS pays more than Treasuries not because it's riskier on default, but because it's riskier on timing.**

## CMOs: tranching by time, not just credit

A plain pass-through forces every investor to eat the same prepayment profile. But different investors want different things — a bank wants short, stable cash; a pension fund wants long, predictable cash. The market's answer is the **CMO (collateralized mortgage obligation)**: take the pool's cash flows and carve them into tranches *by timing*. This is the key insight that separates MBS from ordinary credit securitization. In a corporate-loan deal you tranche by **credit** (who eats the first default loss). In a CMO you also tranche by **prepayment timing** (who gets principal first).

### Sequential-pay

The simplest CMO is **sequential-pay**. You create tranches A, B, C, (and often a Z). The rule: *all* principal — scheduled and prepaid — goes to tranche A first until A is completely paid off. Only then does principal start flowing to B, then to C. Every tranche earns interest the whole time, but principal marches strictly in order.

![Sequential-pay CMO directs principal to tranche A first then B then C](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-4.png)

The effect is to *manufacture* maturities out of one messy pool. Tranche A becomes a short bond (it retires first), C becomes a long bond, and B sits in between. A bank that wants a 2-year asset buys A; a pension fund that wants a 15-year asset buys C. The prepayment risk hasn't disappeared — it's been *redistributed*. Tranche A absorbs the first wave of prepayments (so it's very exposed to contraction), while C is shielded until A and B are gone.

#### Worked example: principal cascades to tranche A first

Take a \$500,000,000 pool split sequentially: tranche A = \$200M, B = \$200M, C = \$100M.

- In month 1, the pool throws off \$3,000,000 of total principal (scheduled + prepaid). **All \$3M goes to A.** A's balance drops to \$197M. B and C get only their interest.
- Suppose over the first two years the pool returns \$200M of principal. **A is fully retired** — every dollar went to A. Only in month 25-ish does B start receiving principal.
- If a refinancing wave hits in year 1, A absorbs it and retires even faster (great if you wanted short); C barely notices.

The intuition: by stacking principal payments in strict order, a single unpredictable pool is reshaped into a short bond, a medium bond, and a long bond — each sold to the investor who wants that exact timing.

### PAC and support tranches

Sequential-pay redistributes timing risk but doesn't *remove* it — tranche A's life still swings with prepayment speeds. The next refinement is the **PAC (planned amortization class)**. A PAC tranche gets a *scheduled* principal payment — a fixed amortization schedule it sticks to across a wide band of prepayment speeds. The magic is a companion **support tranche** (also called a **companion** or **TAC**) that absorbs the variability: when prepayments come in fast, the extra principal is dumped on the support tranche to keep the PAC on schedule; when prepayments are slow, the support tranche waits so the PAC still gets its planned amount.

So the PAC buyer gets near-bond-like predictability, and the support buyer gets paid a fat yield for eating *all* the prepayment uncertainty for both of them. It's risk transfer, pure and simple: the PAC's stability is *manufactured* out of the support tranche's instability. When prepayments blow past the PAC's protective band (the "collar"), even the PAC breaks its schedule — which is exactly what burned a lot of "safe" CMO buyers in fast-prepay years. Tranching is powerful, but it relocates risk; it never destroys it. (The recursion of tranching tranches — taking the mezzanine slices and re-securitizing *them* — is the CDO story, told in [CDOs, CLOs, and the tranching of tranches](/blog/trading/capital-markets/cdos-clos-and-the-tranching-of-tranches).)

### IO and PO strips: the most leveraged prepayment bets

The most extreme CMO carve-up splits a pool's cash into its two raw ingredients. Remember every monthly payment is part **interest** and part **principal**. An **IO (interest-only) strip** receives *only* the interest; a **PO (principal-only) strip** receives *only* the principal. These are the purest possible bets on prepayment, and they move in opposite directions.

- A **PO** is bought at a deep discount (you pay, say, 65 cents for a dollar of principal you'll eventually receive). You *want* fast prepayments — the sooner that dollar comes back, the higher your return. So a PO *loves* falling rates and refinancing waves.
- An **IO** is the mirror image. It only earns interest on the *outstanding* balance. The moment a loan prepays, that loan stops paying interest — your income source vanishes. So an IO *hates* prepayments and *loves* rising rates that keep loans alive. An IO is one of the few fixed-income instruments whose price *rises* when rates rise.

#### Worked example: an IO strip torched by a refi wave

Suppose you buy the IO strip off a \$100,000,000 pool of 6% mortgages, paying \$4,000,000 for the right to collect interest only.

- At a slow 6% CPR, the balance stays high, so you collect close to \$100{,}000{,}000 \times 0.06 = \$6{,}000{,}000 of interest in year one — a great return on your \$4M.
- A rate drop pushes prepayments to 40% CPR. Within a year, \$40{,}000{,}000 of the pool has paid off, so by year-end you're earning 6% on only ~\$60M — and the loans keep vanishing. Your future interest stream collapses, and the IO can lose half its value or more.

The intuition: an IO is a bet that loans *stay alive*; a refinancing wave is the one event that kills them all at once, which is why IOs are the most violent prepayment instrument in the market.

## ABS: the non-mortgage cousins

Now swap mortgages out of the pool and put in other household debt. That's **ABS**, and the big four collateral types are auto loans, credit-card receivables, student loans, and equipment leases. The mechanics rhyme with MBS — pool, SPV, servicer, tranches, waterfall — but the *behavior* of the collateral is different, and that changes the design.

The single most useful distinction is **amortizing vs revolving** pools.

- **Amortizing pools (auto, equipment, student).** Each loan pays down a fixed schedule to zero, just like a mortgage. The pool shrinks over time. Auto-loan ABS is the cleanest example: a 5-year car loan amortizes predictably, and — crucially — *prepayment matters far less than in mortgages*. Why? Because the loan is small and the rate is high relative to the hassle of refinancing a \$25,000 car loan, and because cars depreciate, so there's no "refi when rates drop" reflex. Auto ABS therefore has a tight, predictable life. That predictability is why it's a favorite of conservative buyers.
- **Revolving pools (credit cards).** A credit-card "loan" has no fixed schedule — you borrow and repay constantly. So a card-ABS pool is a *revolving* structure: during a multi-year **revolving period**, principal collections are used to *buy more receivables* rather than pay down bonds, keeping the bond balance flat. Then an **amortization period** kicks in and principal finally flows to investors. The bond behaves like a bullet (stable, then pays off in a window) — totally unlike the slow monthly drip of a mortgage.

What makes each pool tick is *its own* behavior. Student loans have deferment and forbearance (borrowers in school don't pay), so the timing is stretched and policy-sensitive — and a large slice of US student-loan ABS is backed by government-guaranteed loans, which removes most credit risk and leaves policy and timing risk. Equipment leases tie to business cycles and to the residual value of the equipment at lease-end. The art of ABS is matching the bond structure to how that specific borrower base actually pays.

### Credit enhancement: how ABS manufactures safety

Because ABS investors usually *do* bear credit risk (there's no government guarantee on a car loan), every deal is wrapped in layers of **credit enhancement** — the engineering that turns a pool of mediocre consumer loans into a AAA senior bond:

- **Subordination** — the junior tranches that absorb losses first (the waterfall we'll see next). This is the main lever.
- **Overcollateralization (OC)** — putting *more* loans in the pool than bonds issued. A \$1 billion deal might be backed by \$1.06 billion of loans; that extra \$60M is a buffer that absorbs losses before any bond is touched.
- **Excess spread** — the gap between what the loans earn (say 9% on auto loans) and what the bonds pay (say 5%) plus fees. That ~4% annual cushion is the *first* line of defense: losses get netted against it every month before they ever eat into a tranche.
- **A reserve account** — cash set aside up front to cover shortfalls.

These stack on top of each other, which is why a senior auto-ABS tranche can be genuinely safe even though the underlying borrowers are ordinary people with ordinary cars. The senior holder is protected by excess spread *plus* overcollateralization *plus* the entire subordinated stack beneath them. The danger — and 2008's lesson — is when the enhancement is calibrated to good-times loss rates and the bad times turn out to be much worse.

### The servicer and the waterfall

Every securitization, MBS or ABS, runs on the same monthly ritual: the **servicer** collects all the cash, and a **waterfall** decides who gets paid in what order. The waterfall is a strict priority list — the defining feature of structured finance.

![Monthly cash cascades down the waterfall from fees to senior to equity](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-7.png)

The order is almost always:

1. **Fees first** — servicer fee, trustee fee, any swap counterparty.
2. **Senior interest, then senior principal** — the AAA tranche gets its due.
3. **Mezzanine interest, then principal** — the BBB tranche.
4. **Equity / residual last** — the first-loss piece keeps whatever's left.

The point of the order is **subordination**: the junior tranches stand below the senior ones, so they absorb losses first and get paid last. That's what lets the senior tranche earn a AAA rating even when the underlying loans are merely OK. The senior holder is protected by a cushion of subordinated paper beneath them.

![How a 100-unit deal absorbs losses across senior, mezzanine, and equity](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-6.png)

#### Worked example: an auto-ABS senior tranche's protection

Take a \$1,000,000,000 auto-loan ABS structured as: **Senior (A) = \$800M (80%)**, **Mezzanine (B) = \$150M (15%)**, **Equity = \$50M (5%)**.

- The equity tranche is **first-loss**: it absorbs the first \$50M of pool losses before any other tranche is touched. \$50M / \$1{,}000M = **5%** of the pool can default with zero loss to anyone above equity.
- The mezzanine absorbs the *next* \$150M. So the pool can lose \$50M + \$150M = \$200M — a **20%** cumulative default loss — before the senior tranche loses a single dollar.
- Auto-loan pools historically run cumulative losses of only a few percent even in bad years. So a 20% loss cushion makes the senior tranche extraordinarily safe — hence its AAA rating and its low yield.

The intuition: the senior holder isn't trusting the car buyers; they're trusting the \$200M of subordinated paper standing between them and the first loss. Subordination, not borrower quality, is what manufactures the AAA.

## Where this sits in the bond universe

Step back and see the scale. Securitized debt — MBS plus ABS — is a structural pillar of the global bond market, not a niche.

![Where securitized debt sits within the global bond market](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-8.png)

The global bond market is roughly \$140 trillion, and mortgage-related debt alone is one of its single largest components. This is the spine of the series made concrete: the securitization machine has converted *millions of individual household loans* — each one illiquid, each one a private contract between a bank and a borrower — into one of the deepest, most-traded secondary markets that exists. A pension fund can hold a claim on 5,000 anonymous Ohio car loans and sell it in seconds. That's not a financial curiosity; it's the reason credit is cheap and abundant. Liquid secondary markets for securitized paper are what let banks keep lending: they originate, securitize, sell, and lend again.

## Common misconceptions

**"MBS is risky because homeowners default."** For *agency* MBS, default risk is essentially zero — Fannie, Freddie, and Ginnie eat the credit losses. The risk you're actually paid for is **prepayment timing**, not default. (For *private-label* MBS, credit risk is real — which is precisely why it, not agency MBS, blew up in 2008.)

**"A pass-through is just like a Treasury bond."** No. A Treasury returns all principal on one known date; a pass-through dribbles principal back every month at an unpredictable speed. The Treasury has positive convexity; the MBS has *negative* convexity. They behave like opposite instruments when rates move.

**"Tranching makes risk disappear."** Tranching *relocates* risk; it never destroys it. A CMO's PAC tranche is stable only because a support tranche is eating all its volatility. Slicing a pool into senior/mezz/equity doesn't reduce total losses — it just decides who absorbs them first. Forgetting this was central to 2008.

**"AAA means the loans are high quality."** A senior tranche can be AAA even on a pool of mediocre loans, because the rating comes from **subordination** — the cushion of junior tranches below it — not from the borrowers. When the subordination is too thin for the real loss rate (as with 2006-era subprime), the "AAA" is fiction.

**"Auto and card ABS behave like mortgages."** They don't. Auto ABS amortizes fast and predictably with little prepayment sensitivity; card ABS is a *revolving* bullet structure. The collateral's payment behavior drives the bond's behavior — that's the whole craft of ABS.

## How it shows up in real markets

**The 2003 refi wave.** When mortgage rates fell sharply in 2002-2003, prepayment speeds exploded. Investors who'd modeled 7-year average lives watched bonds return principal in two — the exact scenario in our worked example. It taught a generation of fixed-income managers what "negative convexity" feels like in a P&L.

**2008: when the private-label machine broke.** The crisis was not an agency-MBS or auto-ABS event — it was a *private-label subprime MBS* event. Subprime origination ran up from ~\$190B in 2001 to ~\$625B in 2005 before collapsing.

![US non-agency securitization issuance with the 2008-09 collapse marked](/imgs/blogs/abs-and-mbs-the-mortgage-and-consumer-credit-machine-5.png)

When house prices fell, default losses blew straight through the thin subordination on "AAA" subprime tranches, and the non-agency issuance market — \$700B+ in 2007 — fell off a cliff to ~\$150B by 2009. The agency market, with its government guarantee, kept functioning throughout; the private-label market simply *closed*. The full autopsy is in [2008: when the securitization machine broke](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study). The lesson that survived: securitization works only when the secondary market *trusts* the paper. The moment investors stopped believing the ratings, the machine seized — no buyers, no trades, no new issuance, no new lending.

**Who actually buys this stuff.** The buyer base tells you why the market is shaped the way it is. Banks love short, AAA, floating-rate ABS and short MBS tranches because they pair well with their deposit funding and qualify as high-quality liquid assets. Insurance companies and pension funds — with 20- and 30-year liabilities — buy the long CMO tranches and the PAC tranches that promise predictable cash. Money-market funds buy the very shortest, highest-rated slices. The Federal Reserve itself became the largest single holder of agency MBS during its quantitative-easing programs, buying trillions to push mortgage rates down. Each buyer wants a different timing and credit profile, and tranching exists precisely to *manufacture* a slice for each of them. That's the deep reason the secondary market is so deep: the product was engineered, slice by slice, to match the natural demand of every major class of saver.

#### Worked example: matching a tranche to a pension's liability

Suppose a pension fund owes retirees roughly \$50,000,000 a year, ten to fifteen years out, and wants a bond whose principal arrives in that window — not before (reinvestment headache) and not after (it can't pay retirees).

- A plain pass-through is useless: principal dribbles back every month from year one, unpredictably.
- A **sequential-pay tranche C**, which receives no principal until A and B retire (~years 1–9), starts paying down right in the 10–15 year window — a much better match.
- Even better, a **PAC tranche** with a scheduled amortization band locks the timing across a wide range of prepayment speeds, so the fund can plan around it.

The intuition: the pension isn't buying "a mortgage bond" — it's buying a *manufactured maturity* that didn't exist in the raw pool, which is the entire reason CMOs were invented.

**The slow recovery.** Notice in the chart that non-agency issuance climbed back to ~\$650B by 2021. The machine wasn't destroyed; it was rebuilt with thicker subordination, more honest ratings, and better disclosure. That recovery is itself the spine's proof: when trust returned to the secondary market, primary issuance came back with it.

## The takeaway: a machine built on timing

If you remember one thing, make it this: **the securitized markets run on *timing risk*, not just credit risk.** Treasuries pay you on a schedule; corporate bonds add default risk; but MBS and ABS add a third dimension — *when* your principal comes back is itself uncertain and moves against you. Agency MBS strips out credit risk entirely and leaves you holding pure prepayment risk, which is why it's the second-most-liquid market on earth and yet still pays a spread over Treasuries.

The deeper point for understanding the capital-markets machine is this. Securitization is the ultimate expression of the series' thesis. It is a *primary-market* technology — it manufactures brand-new tradable securities out of raw household credit that could never be sold on its own. But it only functions because a deep, trusting *secondary market* will buy and trade those securities every day. CMOs and ABS tranching exist purely to *engineer* the timing and credit profiles that secondary-market buyers actually want — short bonds for banks, long bonds for pensions, AAA bullets for the risk-averse. The machine doesn't just convert loans into bonds; it *shapes* them to fit demand. When that shaping is honest, it makes credit cheap for every household in the country. When it's dishonest — thin subordination dressed as AAA — it breaks, and 2008 is what breaking looks like.

## Further reading & cross-links

- [Securitization from first principles: turning loans into bonds](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) — the primer this post builds on.
- [CDOs, CLOs, and the tranching of tranches](/blog/trading/capital-markets/cdos-clos-and-the-tranching-of-tranches) — what happens when you re-securitize the tranches.
- [2008: when the securitization machine broke](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study) — the case study of private-label MBS failure.
- [How a bond is issued: auctions, syndication, and the deal](/blog/trading/capital-markets/how-a-bond-is-issued-auctions-syndication-and-the-deal) — how these securities get sold into the primary market.
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — the rate backdrop that drives prepayment and spread.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the commercial-bank view of the same machine.
