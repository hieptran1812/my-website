---
title: "CDOs, CLOs, and the Tranching of Tranches: When Securitization Built Structures Out of Structured Products"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How a CDO manufactures fresh AAA out of mostly-BBB collateral, why one assumption about default correlation blew it up in 2008, and why its cousin the CLO survived."
tags: ["capital-markets", "cdo", "clo", "securitization", "structured-finance", "correlation", "synthetic-cdo", "credit", "2008-crisis", "tranches", "shadow-banking"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A CDO is securitization pointed at itself: it pools the leftover, lower-rated middle tranches of *other* securitization deals and re-tranches them, manufacturing brand-new "AAA" bonds out of collateral that was mostly rated BBB.
>
> - The whole trick rests on one assumption — that the loans default *independently* of each other. When you assume low correlation, the senior tranche almost never gets hit, so it can be rated AAA. When housing fell everywhere at once in 2008, defaults became correlated, and the "safe" tranche took losses it was never supposed to take.
> - CDO-squared deals re-securitized the tranches of *other CDOs* — leverage stacked on leverage, opacity stacked on opacity, until nobody could trace a bond back to the houses underneath it.
> - The CLO — the leveraged-*loan* cousin — mostly survived 2008 and is a large, healthy market today, because its collateral is diversified senior-secured corporate loans with real subordination and an active manager, not a thin slice of correlated subprime risk.
> - The one number to remember: a subprime mezzanine CDO might give its senior tranche only a few percent of loss cushion, so a pool-wide default rate above roughly **5–8%** could wipe a bond the rating agencies called as safe as a Treasury.

## A bond that was three deals deep

In 2006 you could buy a bond rated AAA — the same rating the rating agencies gave to US Treasuries and the bluest of blue-chip corporations — that paid you a yield noticeably higher than a Treasury. That should have been impossible. A higher yield is the market's way of pricing in more risk, and AAA is supposed to mean almost no risk. Yet here was a security offering both. Money managers, pension funds, and foreign central banks bought hundreds of billions of dollars of them precisely because they looked like a free lunch: Treasury-grade safety with a fatter coupon.

If you had pulled the thread on one of those bonds, here is what you would have found. The bond was a tranche of a **collateralized debt obligation** (a CDO). The CDO did not own any loans directly. What it owned was a pile of *other* bonds — specifically, the middle, BBB-rated slices of dozens of mortgage securitizations. Pull the thread on one of *those* slices and you would reach a pool of several thousand home mortgages, many of them subprime: loans to borrowers with weak credit, often with no income documentation, made in 2005 and 2006 at the very top of a housing bubble. The "AAA" you bought sat three securitization layers above an adjustable-rate loan on a house in Stockton, California that the borrower could not afford the moment the teaser rate reset.

That is the subject of this post: **re-securitization** — the practice of building new structured products out of the pieces of old ones. It is the most extreme expression of a single idea that runs through this entire series. Securitization is a *primary-market technology* for creating tradable securities out of illiquid loans. It works only when the secondary market trusts the result enough to buy and trade it. The CDO took that technology and pointed it at its own output, manufacturing trust out of structure rather than out of the actual quality of the loans underneath. When the trust evaporated, the whole machine seized — and the chain between the borrower and the investor turned out to be so long that nobody could see the collateral. Let us build the whole thing up from the loan, one layer at a time.

![Re-securitization stack from MBS mezzanine tranches into a CDO and new AAA, mezzanine, and equity tranches](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-1.png)

## Foundations: what a tranche actually is

Before we can stack securitizations on top of each other, we need to be completely clear on what one securitization does, because the CDO is just that operation applied twice.

Start with the everyday version. Suppose you and two friends jointly lend \$300,000 — \$100,000 each — to a small property developer. The developer might pay you all back in full, or might default and only return part of the money. You are all equal partners, so if \$60,000 is lost, each of you eats \$20,000. That is *pari passu*: everyone shares losses in proportion.

Now change one rule. Instead of sharing losses equally, you agree on an *order* in which losses hit. Friend C agrees to absorb the **first** \$30,000 of any loss. Friend B absorbs the **next** \$60,000. You take losses only after the first \$90,000 is gone. In exchange for taking the dangerous bottom position, Friend C gets the highest interest rate; you, sitting safely at the top, accept the lowest rate because you are very unlikely to lose anything. You have just **tranched** the loan. A tranche (French for "slice") is a claim on the *same* pool of cash flows, but with a defined position in the loss-absorption queue.

This is the entire engine of structured finance. Take a pool of loans, collect all the interest and principal they pay, and pour that cash into a **waterfall**: the top tranche gets paid first and loses last; the bottom tranche gets paid last and loses first. The bottom slice — usually called the **equity** or **first-loss** tranche — is the shock absorber. The slices above it are **mezzanine** (the middle), and the slice on top is **senior**. Because the senior tranche only suffers if losses are large enough to burn through everything below it, it can carry a much higher credit rating than the average loan in the pool. A pool of BBB-rated mortgages can produce a senior tranche the agencies will stamp AAA, simply because the equity and mezzanine beneath it stand in front of the losses.

It helps to separate two cash-flow rules that a waterfall actually governs, because people blur them. There is the **interest waterfall** — the monthly question of who gets paid coupons first while the deal is healthy — and there is the **principal/loss waterfall** — the question of who eats losses when borrowers default. The senior tranche wins on both: it is paid its coupon before anyone below it sees a dollar, and it loses its principal only after everyone below it has been wiped out. The equity tranche is the mirror image: it is paid last (it gets the *residual* — whatever is left after every senior claim is satisfied) and it loses first. This asymmetry is exactly why the senior tranche can be rated so much higher than the pool average: it has both a payment-priority cushion and a loss-absorption cushion standing in front of it. The price of that safety is a low coupon; the reward for the equity's danger is a fat, leveraged residual in good years and a zero in bad ones.

There is one more concept the rest of this post leans on: the **attachment** and **detachment** points of a tranche. A tranche that absorbs losses from, say, 10% to 30% of pool losses is said to *attach* at 10% and *detach* at 30%. Below 10% it is untouched; above 30% it is fully wiped and losses pass to the tranche above. The AAA senior tranche is simply the one with the highest attachment point — it does not start losing until a very large fraction of the pool is gone. The whole rating exercise reduces to one question: *how likely is it that losses climb past this tranche's attachment point?* And that probability, for a pool of many loans, depends overwhelmingly on whether the loans default independently or together — the correlation question we return to below.

> [!note]
> **Why this is legitimate, in moderation.** Tranching is not a trick by itself. If the loans are genuinely diversified and the subordination is real, the senior tranche really *is* safer than the pool average — for the same reason that the most senior bond of a healthy company is safer than its stock. The danger appears only when the collateral is *not* diversified and the cushion is *thinner than it looks*. Hold that thought; it is the whole story.

A **CDO** — collateralized debt obligation — is this same waterfall, but the "loans" in the pool are themselves *bonds*: corporate bonds, asset-backed securities, or, most fatefully, the mezzanine tranches of other mortgage deals. A CDO is a securitization of securitizations. Everything that makes a single securitization work or fail applies to a CDO, only squared.

If you want the single-layer version in full, this series covers it in [securitization from first principles](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) and the mortgage-specific machinery in [ABS and MBS, the mortgage and consumer-credit machine](/blog/trading/capital-markets/abs-and-mbs-the-mortgage-and-consumer-credit-machine). This post assumes that foundation and builds the second story on top of it.

## The CDO idea: manufacturing AAA from BBB

Here is the problem the CDO was invented to solve, from the point of view of a Wall Street desk in 2005.

When you securitize a pool of subprime mortgages, the waterfall produces a lot of senior AAA bonds (those sell instantly to safety-seeking buyers) and a thin layer of equity (hedge funds and the deal sponsor buy that for the high yield). But it also produces a chunk of **mezzanine** — the BBB-rated middle. BBB is the lowest rung of investment grade; it is the slice nobody especially wants. It yields more than AAA but carries real default risk, and the natural buyer base is small. Desks were producing far more BBB mezzanine than they could sell. That inventory was clogging the machine.

The CDO unclogged it. Take the unsold BBB mezzanine tranches from thirty or forty different mortgage deals, pool them, and run the *same* waterfall over the pool. Out comes — astonishingly — a new senior tranche, perhaps 70–80% of the deal, that the rating agencies will rate AAA. You have taken a pile of BBB bonds that the market priced as risky and manufactured a majority of brand-new AAA bonds out of it. The leftover middle and bottom of the CDO are smaller and easier to place. The inventory clears. The fees roll in.

#### Worked example: re-tranching a pile of BBB into 70% "AAA"

Suppose a CDO buys \$1,000,000,000 (one billion dollars) of BBB-rated mezzanine tranches collected from many mortgage deals. The bank structures the CDO's own capital stack like this:

- **Senior tranche: \$700,000,000 (70%)** — rated AAA. Pays the lowest coupon, say 5.5%.
- **Mezzanine tranche: \$200,000,000 (20%)** — rated A down to BBB. Pays a higher coupon, say 7%.
- **Equity tranche: \$100,000,000 (10%)** — unrated, first-loss. Takes whatever cash is left after the others are paid; the residual could be 15%+ in good times or zero in bad.

The loss waterfall runs bottom-up: the equity absorbs the first \$100M of losses, the mezzanine the next \$200M, and only after \$300M (30% of the pool) is gone does the AAA senior take its first dollar of loss. The rating agency's model says a diversified BBB pool would essentially never lose 30%, so 70% of the structure earns the AAA stamp.

Step back and notice what just happened. The collateral was 100% BBB. The output is 70% AAA. **No new safety was created** — the underlying loans are exactly as risky as before. All that changed is how the losses were *sliced*. The AAA-ness was manufactured entirely by the ordering of the waterfall plus the rating model's assumption that the BBB bonds would not all go bad together. That assumption is the load-bearing wall of the entire edifice, and it is the subject of the next section.

It is worth being precise about *why a desk wanted to do this*, because the incentives explain the volume. The bank earned an upfront structuring fee on every CDO it arranged — typically 1–2% of the deal size, so \$10–20M on a \$1B CDO. It earned more by warehousing the BBB collateral and selling it into the CDO at a markup. And critically, it cleared inventory: the unsellable BBB mezzanine that was tying up the bank's balance sheet got repackaged into AAA bonds that *did* sell. Every party in the chain was paid at the moment the deal closed — the originator who made the loan, the bank that securitized it, the bank that re-securitized it, the rating agency that stamped it, the manager who ran it. The losses, by design, arrived years later and landed on the end investor. When the people who *create* a security are paid upfront and bear none of its later losses, you should expect them to create a great deal of it. They did.

The volume here was not small. US securitization issuance — the broad pipe that fed both honest ABS and the CDO machine — ran near \$700–750 billion a year in 2006–2007 before collapsing by roughly 75% in the crisis. At the peak, CDOs were buying a large majority of all the BBB-rated subprime mezzanine being produced; without the CDO bid, that mezzanine could not have been sold, which means the subprime origination boom itself could not have been funded at the scale it reached. Re-securitization was not a sideshow to the bubble. It was the mechanism that let the bubble inflate, because it manufactured a buyer for the riskiest slice of every deal.

![US securitization issuance by year showing the run-up to 2007 and the collapse in 2008](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-3.png)

## The correlation assumption at the heart of it

Why can a pool of BBB bonds produce a AAA senior tranche? The honest answer is: *only if the BBB bonds default independently of one another.* Everything hangs on the word **correlation**.

Think about flipping a hundred coins. If each coin is independent, the chance that, say, 90 of them come up heads at once is vanishingly small — the outcomes average out. That is **idiosyncratic** risk: each loan can go bad for its own private reason (a borrower loses a job, gets divorced, has a medical bill), and those reasons are unrelated across borrowers. In an idiosyncratic world, a diversified pool's loss rate is stable and predictable, the equity and mezzanine cushion easily covers it, and the senior tranche is genuinely safe.

Now glue the coins together so they tend to land the same way. If one comes up heads, the others probably do too. That is **systematic** (correlated) risk: a single common factor — say, national house prices — drives *all* the loans at once. In a correlated world, the pool no longer averages out. Either almost everyone pays (small loss) or a huge fraction defaults together (catastrophic loss). The middle outcomes that the cushion was sized for rarely happen. You get a bimodal distribution: mostly fine, occasionally apocalyptic.

The rating models — most famously a tool called the **Gaussian copula**, which reduced the whole question of "how likely are these loans to default together?" to a single correlation number — were calibrated on a period when US house prices had never fallen nationally. So they plugged in a *low* correlation. Low correlation means losses average out, which means a thin cushion is plenty, which means most of the structure can be AAA. The entire AAA rating was a bet that mortgage defaults across the country were closer to independent coins than to glued-together coins.

![Before and after view of the correlation assumption, low correlation versus systematic shock](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-2.png)

That bet was catastrophically wrong. House prices were the common factor. When the national housing market turned in 2007–2008, borrowers everywhere went underwater at the same time, defaulted at the same time, and the "diversified" pool of mortgages from California, Florida, Nevada, and Arizona behaved like one giant correlated bet on a single thing: American home prices going up forever.

#### Worked example: how a small rise in correlation wipes the senior tranche

Take our \$1B CDO with its \$300M of subordination (equity + mezz) protecting a \$700M AAA tranche. The agency model assumed that in a stress scenario the BBB collateral might lose around 10% — \$100M — which the \$100M equity tranche absorbs entirely. The AAA tranche stays untouched. So far so good.

But the CDO's collateral was itself *the BBB mezzanine of subprime deals*. Those mezzanine tranches are highly sensitive: they are the slice that gets wiped out once the underlying mortgage pool's losses climb past the equity layer of *that* deal. In a correlated downturn, it was not one mezzanine bond going bad — it was *all of them* going bad together, because they were all exposed to the same falling house prices.

Run the correlated scenario. Suppose pool-wide losses on the CDO's collateral come in not at 10% but at 45% — \$450,000,000 — because the mezzanine tranches it held defaulted *together*:

- Equity (\$100M): wiped out. Loss absorbed: \$100M.
- Mezzanine (\$200M): wiped out. Loss absorbed: \$200M.
- Cumulative absorbed: \$300M. Remaining loss: \$450M − \$300M = **\$150M, which lands on the "AAA" senior tranche.**

The AAA tranche, sold as essentially risk-free, takes a \$150M loss — about 21% of its \$700M face value. The holders who bought it as a Treasury substitute discover they own something closer to a junk bond. Notice that the difference between "totally safe" and "21% loss" was not a change in the loans — it was a change in the *assumed correlation*. A small move in that one input, from "low" to "high", is the difference between the model's world and the real one. **The senior tranche of a correlation-dependent structure is only as safe as the correlation assumption, and that assumption is invisible on the rating label.**

There is a subtle, deadly feature of the correlation parameter worth dwelling on: **the senior tranche is the one most sensitive to it.** This is counterintuitive — surely the risky equity tranche is the one that cares about everything? But no. The equity tranche gets wiped out in almost any bad scenario regardless of correlation; its fate is nearly settled either way. The *senior* tranche only ever takes a loss in the tail — the scenario where so many loans default that losses climb all the way to its attachment point. And that tail scenario is *precisely* the correlated one: it can only happen if defaults cluster. So as you dial correlation up from low to high, the equity tranche's value barely moves, while the senior tranche's value falls off a cliff. The AAA buyer, who thought they had bought the safest, least-assumption-dependent slice, had actually bought the slice whose entire value rested on a single, unobservable, historically-miscalibrated number. The safety they paid for was the safety most exposed to the one input nobody could see.

The formal mathematics of dependence and copulas belongs to [quantitative finance](/blog/trading/quantitative-finance/order-book-simulator-quant-research) and we link out rather than re-derive it. The point for the *machine* is structural: re-securitization concentrated correlation risk while the rating system priced it as if it had been diversified away. The single-deal MBS at the bottom had *some* genuine diversification across borrowers. But when you pooled the mezzanine of forty such deals into a CDO, you did not add diversification — you added concentration, because every one of those forty mezzanine slices was a bet on the same national housing factor. The CDO looked more diversified (forty bonds instead of one) and was in fact less, because it had stripped away the idiosyncratic part of each deal and kept only the part driven by the common factor. Re-securitization is, in this exact sense, a *concentration* technology dressed up as a diversification technology.

## CDO-squared: leverage on leverage, opacity on opacity

If you can pool the BBB tranches of mortgage deals into a CDO, you can pool the BBB tranches of *CDOs* into a new CDO. That is a **CDO-squared** (CDO²). And yes, a handful of **CDO-cubed** deals existed too. Each layer repeats the same manufacturing step — pool the unwanted middle, re-tranche, mint fresh AAA — one level further from the actual houses.

![CDO-squared structure pooling the mezzanine of other CDOs into a third layer of AAA](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-4.png)

Two things compound viciously at each layer.

First, **leverage**. Each tranching step concentrates risk into a thin slice. The equity of a CDO is a leveraged bet on the CDO's collateral; the equity of a CDO-squared is a leveraged bet on *that*. Small moves in the underlying mortgage losses get amplified at every level, the way a small move in a stock price becomes a huge move in a deep out-of-the-money option. The result is that the bottom tranches of a CDO² could swing from "paying handsomely" to "worth zero" on a modest change in national default rates.

Second, **opacity**. To value a single CDO² tranche correctly, you would have to know the loans inside every mortgage pool, inside every MBS deal, inside every CDO, inside the CDO². In practice nobody did this. The deals referenced each other, sometimes circularly (CDOs bought each other's tranches), and the documentation ran to thousands of pages. The rating agencies modeled the layers with the same low-correlation assumption at every level, so the errors did not cancel — they multiplied.

#### Worked example: a CDO-squared's leverage

Take a CDO² whose collateral is the mezzanine tranches of underlying CDOs. Each of *those* CDOs already had a ~10% equity cushion below its mezzanine. Now the CDO² puts only, say, an 8% equity cushion below *its* senior tranche.

Trace a 5% loss on the original mortgage pools through the stack:

- At the first securitization, a 5% mortgage loss might burn through the deal's ~5% equity entirely — so the BBB mezzanine of that deal starts taking losses. Call it a 50% loss on the mezzanine.
- The first CDO holds those mezzanine bonds. A 50% loss on its collateral blows through its 10% equity and most of its 20% mezzanine — so the CDO's *own* mezzanine is largely wiped, call it 80% gone.
- The CDO² holds *that* mezzanine. An 80% loss on its collateral obliterates its 8% equity and its mezzanine, and slams hard into its "AAA" senior tranche.

A **5% loss at the bottom became near-total loss at the top.** That is the leverage of re-securitization: each layer multiplies the sensitivity, so a downturn that would have been a manageable haircut on the raw mortgages became a wipeout three layers up. The investor in the CDO²'s AAA tranche thought they owned the safest thing in the deal. They owned the most leveraged.

This is the precise mechanism by which the chain between borrower and investor stretched past the breaking point — the theme this series picks up in the [2008 case study, when the securitization machine broke](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study).

## The CLO: the cousin that survived

Here is the part that surprises people. A structure that looks almost identical on paper — pool a bunch of debt, tranche it, sell AAA off the top — *mostly survived 2008 and is a large, healthy market today*. That structure is the **CLO**: the collateralized **loan** obligation.

A CLO pools **leveraged loans** — floating-rate, senior-secured loans made to below-investment-grade companies (the debt that funds private-equity buyouts and corporate borrowing). It tranches them into AAA down to equity, exactly like a CDO. So why did subprime CDOs detonate while CLOs came through with their AAA and AA tranches taking essentially zero principal losses, even through 2008–2009?

Four structural differences, all of which trace back to the same root cause — *real diversification and real subordination versus manufactured safety*:

1. **The collateral is genuinely diversified.** A CLO holds loans to 150–300 different companies across many industries. One retailer going bankrupt is idiosyncratic — it does not mean the software company and the hospital chain and the packaging firm also default. A subprime CDO's "diversification" was an illusion: dozens of mezzanine tranches that were all bets on the *same* national housing market. Different names, one risk factor.
2. **The loans are senior and secured.** Leveraged loans sit at the top of the borrower's capital structure and are backed by the company's assets. When a borrower defaults, the loan *recovers* a large fraction — historically around 60–70 cents on the dollar — versus far lower recoveries on a defaulted subprime mortgage tranche. Losses, when they come, are partial, not total.
3. **The subordination is thicker and real.** A CLO's AAA tranche typically sits on top of 35–40% subordination. To lose a dollar of AAA principal, more than a third of the *entire* loan pool has to be wiped out *after recoveries* — a default-and-loss rate far beyond anything corporate credit has ever produced, even in 2008.
4. **It is actively managed.** A CLO has a portfolio manager who can trade out of deteriorating loans, reinvest principal, and steer the pool during its life. A static subprime CDO just held its bonds and watched them rot.

![Tranche stack of a 100-unit deal showing senior, mezzanine, and equity loss-absorption order](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-7.png)

#### Worked example: a CLO surviving a default rate that would destroy a subprime CDO

Take a \$500,000,000 CLO with a AAA tranche of \$325,000,000 (65% of the deal) — so the AAA sits on **35% subordination** (\$175M of mezzanine and equity beneath it).

Now hit it with a brutal recession: **15% of the loans in the pool default** over the cycle. That sounds devastating. But these are senior-secured loans, so each defaulted loan recovers about **65 cents on the dollar**. The actual *loss* is therefore:

- Defaulted notional: 15% × \$500M = \$75,000,000.
- Recovery at 65%: the pool gets back 0.65 × \$75M = \$48,750,000.
- Net loss: \$75M − \$48.75M = **\$26,250,000**, or about 5.25% of the \$500M pool.

That \$26.25M loss is absorbed entirely by the equity and lower mezzanine tranches. The \$175M of subordination is barely scratched — the AAA, AA, and even most mezzanine tranches take **zero principal loss**. The CLO shrugs off a 15% default rate.

Now run the same 15% default rate through the subprime CDO from earlier. Its collateral was BBB mezzanine tranches with near-total loss-given-default (a wiped mezzanine recovers close to nothing) and only \$300M / \$1,000M = 30% subordination — but, fatally, that 30% was correlated and the losses were not partial. A 15% pool loss with low recovery, *correlated across the whole pool*, blows through the equity and eats deep into the mezzanine; push to the 30–45% correlated losses we computed and the AAA itself is hit. **Same waterfall shape, opposite outcome — because the CLO's losses were partial and diversified, and the CDO's were total and correlated.** The structure is not the risk. The collateral and the honesty of the subordination are the risk.

There is also a governance layer that protects CLO investors and had no real analogue in the subprime CDO world: **coverage tests.** A CLO's indenture sets out overcollateralization (OC) and interest-coverage (IC) tests that the deal must pass every payment period. If the pool deteriorates — too many loans downgraded to CCC, or the value of collateral falling relative to the notes — the tests fail, and cash that would have flowed to the equity and lower tranches is instead *diverted to pay down the senior notes early.* The structure deleverages itself automatically in a downturn, protecting the top of the stack at the expense of the bottom. The equity holders hate it (their payout is cut exactly when things go wrong) but it is precisely why the senior tranches survive. The subprime CDO had nothing comparable: it just held its rotting bonds and kept paying everyone until it couldn't pay anyone.

This is why, after 2008, the CDO label became toxic while the CLO market kept growing — it stood near \$1 trillion in the US by the mid-2020s, and its AAA tranches have a track record of near-zero losses across multiple cycles. The lesson the market learned was not "tranching is bad." It was "tranching diversified, senior, well-cushioned, actively-managed collateral is fine; tranching correlated, junior, thinly-cushioned, static collateral and calling it AAA is a time bomb." Same waterfall, same vocabulary, opposite collateral — and the collateral is what determines whether the AAA stamp is a fact or a fiction.

## Synthetic CDOs: betting on a pool you do not own

There is one more layer to add, and it is the one that made *The Big Short* possible. A **synthetic CDO** does not own any bonds at all. It references them.

Recall a **credit default swap** (CDS): an insurance-like contract where the protection buyer pays a periodic premium and, if a specified bond defaults, the protection seller pays out the loss. (The mechanics live in [options and derivatives](/blog/trading/quantitative-finance/market-making-simulator-quant-research); we use the result, not the pricing.) A synthetic CDO is built entirely out of CDS written on a **reference pool** of mortgage bonds that the CDO never buys.

![Synthetic CDO using credit default swaps to reference a pool of mortgage bonds without owning them](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-6.png)

The investors in the synthetic CDO's tranches are *selling protection* — they collect the CDS premiums as their "coupon" and, in exchange, agree to pay out if the referenced mortgage bonds default. They are the **long** side: long the housing market, betting it stays healthy. On the other side sits the **short** investor — the protection buyer — paying the premiums in exchange for a giant payout if the housing market collapses. That short position is exactly the trade made famous in 2007 by the handful of investors who saw the bubble.

Two consequences make synthetic CDOs especially dangerous:

- **They have no natural size limit.** A cash CDO can only be as big as the actual mortgages that exist. But you can write CDS referencing the *same* mortgage bond over and over. Synthetic CDOs let the financial system place *multiples* of the real subprime exposure — several dollars of bets riding on every one dollar of actual mortgages. When the reference bonds defaulted, the losses were larger than the underlying market, because the synthetic layer had multiplied the exposure.
- **They turned a housing downturn into a counterparty crisis.** The protection sellers (long investors, and famously AIG, which sold protection on enormous notional amounts) owed payouts they could not make. The risk did not stay contained in mortgages; it propagated through the web of CDS counterparties — the same contagion mechanism that, in a different market, brought down [LTCM in 1998](/blog/trading/finance/ltcm-1998-when-genius-failed).

#### Worked example: the leverage of a synthetic bet

Suppose \$1,000,000,000 of actual subprime mortgage bonds exist for a given vintage. The cash market can only hold \$1B of exposure. But suppose synthetic CDOs and standalone CDS write protection referencing those same bonds five times over: \$5,000,000,000 of notional exposure.

If the reference bonds ultimately lose 40% of their value:

- The real, cash loss on the underlying bonds: 40% × \$1B = \$400,000,000.
- The synthetic loss the protection sellers owe: 40% × \$5B = **\$2,000,000,000.**

The financial system lost five times the size of the actual collateral, because the synthetic layer let bets stack on a fixed pile of loans. **Synthetic structures decouple the size of the risk from the size of the real economy underneath it** — which is why a relatively contained housing correction became a multi-trillion-dollar solvency crisis.

## Who rated this, and why it mattered

One actor sits at the center of the whole story and deserves its own section: the **rating agency.** A CDO's senior tranche could only be sold as AAA because an agency — Moody's, S&P, or Fitch — said it was AAA. Institutional buyers like pension funds and many foreign banks are *required* by their own rules to hold mostly highly-rated paper; the AAA stamp was the key that unlocked that enormous, price-insensitive pool of buyers. Without the rating, the CDO machine has no customers.

The problem was threefold. First, the agencies were **paid by the issuers** — the very banks structuring the deals — which created a pull toward generous ratings, because a bank that did not like one agency's model could shop the deal to another. Second, the agencies rated the deals with the **same low-correlation models** the desks used, so the rating did not provide an independent check on the central assumption; it embedded the same flaw. Third, and most importantly for our theme, the agencies rated CDOs and CDO-squareds by feeding in the *ratings* of the underlying tranches rather than re-examining the actual loans. A CDO model treated a pile of BBB-rated mezzanine bonds as "BBB collateral" and applied a generic BBB default assumption — it never reached down to ask whether those particular BBB bonds were all bets on the same housing market. The rating of the re-securitization was built on the rating of the securitization, which was built on a model of the loans. Errors at the bottom did not get caught; they propagated upward and compounded.

When the agencies finally downgraded — in 2007 and 2008, often by ten or more notches at once, taking bonds from AAA to junk in a single action — the effect was violent. Forced sellers (funds that could not hold downgraded paper) dumped tranches into a market with no buyers, prices gapped down, and the marks fed back into the next round of downgrades. The rating, which had been the source of the trust that made the market, became the trigger that destroyed it. This is the cleanest illustration of the series' spine in reverse: when the secondary market stopped believing the AAA label, there was no liquidity, no price, and the primary machine that depended on selling those bonds shut off entirely.

## Common misconceptions

**"AAA means safe, full stop."** No. AAA is a statement about the *probability of loss under a model's assumptions*. A CDO's AAA tranche was AAA only under a low-correlation assumption that turned out to be false. The rating never described what happens when the assumption breaks — and on a structured product, the assumption *is* the risk. A AAA corporate bond and a AAA CDO tranche from 2006 were not remotely the same animal.

**"Tranching is financial alchemy — it creates safety out of nothing."** It does not create safety; it *redistributes* it. Total risk in the pool is conserved. Tranching moves risk from the senior holders down onto the equity holders, who are paid for taking it. That is legitimate when the buyers of each tranche understand what they hold. The 2008 failure was not tranching per se — it was tranching *correlated, low-recovery* collateral and mislabeling the result.

**"CLOs are just CDOs with a new name — the next one will blow up too."** The collateral is fundamentally different: diversified, senior-secured corporate loans with ~60–70% recoveries and 35–40% subordination, versus correlated subprime mezzanine with near-zero recoveries and thin cushions. CLOs have already survived multiple credit cycles, including 2008 and 2020, with negligible losses in their senior tranches. They are not risk-free, but they are not the same instrument.

**"Synthetic CDOs were just hedging."** Some CDS use is genuine hedging. But synthetic CDOs let the system manufacture *new* exposure to subprime far beyond the stock of real mortgages. They were net risk-*creating*, not risk-transferring, because the protection sellers had no offsetting position — they were taking on fresh leveraged exposure to housing.

**"Everyone knew it was junk and did it anyway."** Mostly not. Many buyers — banks, pension funds, foreign institutions — genuinely believed the AAA ratings. The deeper failure was that the chain from borrower to investor was so long, and the deals so opaque, that *almost nobody could actually see the collateral* even if they wanted to. That opacity, more than malice, is the structural lesson.

## How it shows up in real markets

**The 2006–2007 peak and the 2008 collapse.** US securitization issuance ran near \$700–750 billion a year at the peak and then fell roughly 75% in 2008–2009 as the machine seized. The CDO segment — and especially the subprime mezzanine CDOs and CDO-squareds — effectively went to zero. Many of the AAA tranches that buyers had treated as cash-equivalents were marked down to a fraction of par; some mezzanine and equity tranches went to zero outright.

![US subprime mortgage origination by year peaking in 2005 and collapsing after 2007](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-5.png)

The subprime origination chart tells the upstream story: origination ballooned from under \$200B in 2001 to over \$600B in 2005, then collapsed to \$23B by 2008 as the loans stopped being made and the defaults came due. Every one of those loans at the 2004–2006 peak was raw material for the CDO machine. The re-securitization layer sat on top of this curve, amplifying its every move.

**AIG and the synthetic exposure.** AIG's financial-products unit had sold protection on tens of billions of dollars of CDO and mortgage exposure through CDS. When the reference assets deteriorated and AIG faced collateral calls it could not meet, it required a US government rescue measured in the tens of billions — a direct consequence of the synthetic layer letting one firm accumulate enormous one-sided exposure to housing without owning a single mortgage.

**The CLO market after the crisis.** While the CDO disappeared, the CLO market grew steadily after 2010 and exceeded \$1 trillion outstanding in the US by the mid-2020s. Through the 2020 COVID shock, even as corporate defaults spiked, the AAA and AA tranches of CLOs took essentially no principal losses — the subordination and recoveries did their job. The CLO is the living proof that the *structure* was never the villain.

**Where the issuance sits today.** In the broad debt-issuance picture, structured products like ABS (and the CLOs within that family) are a meaningful but not dominant slice next to the giant Treasury and mortgage-agency pipes. The chart below puts the asset-backed segment in proportion against the rest of US debt issuance.

![US debt issuance by type in 2023 with ABS shown against Treasury, corporate, and other segments](/imgs/blogs/cdos-clos-and-the-tranching-of-tranches-8.png)

The picture to carry away: re-securitization is a small slice of the total issuance machine, but in 2008 it was the slice that broke, because it stretched the distance between the saver's money and the borrower's house further than anyone could see across.

## The takeaway: how to read any structured product

Re-securitization is the clearest illustration of this series' core idea, taken to its breaking point. A capital market turns savings into investment by creating securities (the primary market) that people trust enough to trade (the secondary market). Securitization is the technology for building those securities out of illiquid loans. The CDO pointed that technology at its own output and manufactured *trust* — the AAA stamp — out of structure rather than out of the genuine quality of the collateral. It worked right up until the secondary market stopped believing the structure, at which point there was no liquidity, no price, and no way to see what was actually inside.

So when you meet any structured product — a CLO, an ABS, a CDO, whatever the next acronym is — ask four questions, in order:

1. **What is the actual collateral, and is it genuinely diversified?** Many different names that all depend on one factor (national house prices, one commodity, one borrower type) is fake diversification.
2. **How thick is the subordination below the tranche I am buying, and is it real?** A 35% cushion of well-recovering senior loans is real. A 5% cushion of correlated junior tranches is theater.
3. **What does the rating actually assume?** If the AAA depends on a correlation number you cannot see, the rating is a bet on that number, not a guarantee.
4. **How far am I from the actual cash flow?** Every layer of re-securitization is another wall between you and the loan. By the third layer, nobody can see the house.

The CLO answers those four questions well and survives. The 2006 subprime CDO-squared answered all four badly and detonated. The instruments looked nearly identical on a tombstone. The difference was never in the structure — it was in the collateral underneath and the honesty of the cushion. That is the whole lesson of re-securitization: **the structure can only ever be as trustworthy as the loans at the bottom, and the more layers you build, the harder it becomes to see them.**

## Further reading & cross-links

- [Securitization from first principles: turning loans into bonds](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) — the single-layer foundation this post builds on.
- [ABS and MBS: the mortgage and consumer-credit machine](/blog/trading/capital-markets/abs-and-mbs-the-mortgage-and-consumer-credit-machine) — where the collateral for CDOs came from.
- [2008: when the securitization machine broke (case study)](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study) — the full crisis narrative.
- [Covered bonds, ABCP, and the shadow-funding chain](/blog/trading/capital-markets/covered-bonds-abcp-and-the-shadow-funding-chain) — how these structures were funded short-term, and how that funding ran.
- [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — an earlier lesson in correlation and leverage breaking together.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the banking-side view of the same machine.
