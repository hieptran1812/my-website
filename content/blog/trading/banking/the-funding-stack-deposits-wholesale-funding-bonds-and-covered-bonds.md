---
title: "The Funding Stack: Deposits, Wholesale Funding, Bonds and Covered Bonds"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank funds itself across the whole stack — from cheap, flighty deposits to costly, loss-absorbing equity — and why the cost-versus-stability trade-off, the creditor waterfall, and the bail-in-able layers decide whether a bank lives or dies."
tags: ["banking", "funding", "deposits", "wholesale-funding", "repo", "covered-bonds", "senior-unsecured", "subordinated-debt", "mrel-tlac", "bail-in", "cost-of-funds", "treasury"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank funds itself in layers, and every layer is a deliberate trade between cost and stability: the cheapest money is also the most likely to run, and the money that stays the longest is the most expensive.
>
> - The stack runs, cheapest to dearest: deposits (about 71% of funding) → wholesale and repo (about 10%) → senior unsecured bonds → subordinated debt → equity (about 8%). Cost climbs as you go up; loyalty climbs with it.
> - Funding splits two ways that decide who gets hurt in a failure: **secured vs unsecured** (does a lender hold collateral?) and the **creditor hierarchy** (who absorbs loss first?). Equity is wiped before any creditor; insured depositors are protected last of all.
> - A **covered bond** is the one trick that buys stability cheaply: the lender gets both a claim on the bank *and* a ring-fenced pool of mortgages, so it accepts a lower rate — often 30 to 60 basis points under senior unsecured.
> - **MREL and TLAC** are rules that force a bank to keep a thick band of bail-in-able layers (equity, AT1, Tier 2, senior) so that in a collapse the loss is absorbed by investors who signed up for it, not by depositors or taxpayers.
> - The one number to remember: a deposit-rich bank funds its book near **2%**, while a wholesale-reliant one can pay well over **3%** for the same assets — and that gap is the difference between a franchise and a casualty.

In March 2023, Silicon Valley Bank had a balance sheet that looked, on paper, perfectly solvent. Its problem was never that its assets were worth less than its liabilities by some catastrophic margin. Its problem was the *shape* of its funding. The bank had grown on a flood of large, uninsured corporate deposits — money that felt like a stable franchise but was, in truth, the most flighty kind of funding a bank can hold. When confidence cracked, those depositors did not write angry letters. They opened an app and moved \$42 billion in a single day, with another \$100 billion queued behind it. No bank on earth funds itself in a way that survives losing more than a fifth of its base in twenty-four hours.

That is the lesson this post is built around, and it is bigger than one bank. A bank is, at its heart, a machine that borrows short and lends long — it takes in money that can leave quickly and turns it into loans that cannot be called back quickly. The entire art of running a bank is managing the *liability side* of that trade: deciding what mix of funding to hold, how much it costs, and — crucially — how fast each piece can disappear when the weather turns. This is the **funding stack**, and learning to read it is learning to see a bank's fragility before the market does.

The diagram above is the mental model for everything that follows: a stack of funding layers, cheap and flighty at the base, costly and loyal at the core, ending in equity — the dearest money of all, and the only money that never has to be paid back.

![Funding stack from cheap flighty deposits to costly loss-absorbing equity](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-1.png)

## Foundations: how a bank actually funds itself

Before we go deep, let us build every term from zero. If you have never thought about where a bank gets its money, this section is your foundation; if you have, skim it.

### What "funding" even means for a bank

When you run a lemonade stand, you fund it with your own pocket money — that is your equity. A bank is different in degree to the point of being different in kind. To make money, a bank lends. To lend, it needs cash to hand out. And almost none of that cash is its own. The vast majority is *borrowed* — from depositors, from other banks, from bond investors. A **funding source**, then, is simply any pool of money a bank borrows in order to fund its assets (its loans and securities). The collection of all these sources, weighted by how much of each it uses, is the bank's **funding mix** or funding stack.

Here is the first surprising fact, and it is the whole reason banks are fragile: a typical large bank funds itself with only about **8% equity**. The other ~92% is borrowed. We call that **leverage** — using borrowed money to control assets larger than your own capital. Eight percent equity means the bank controls assets worth about `1 / 0.08 ≈ 12.5` times its own money. A 12.5-times leveraged machine is wonderful when things go well and lethal when they do not, because an asset fall of just 8% wipes out *all* the equity. Everything in this post is about the 92% — the borrowed part — and how its composition decides whether the bank survives a bad day.

### Secured versus unsecured funding

The first great divide in funding is whether the lender holds **collateral**. *Collateral* is an asset the borrower pledges that the lender can seize if the borrower does not pay back.

- **Unsecured funding** is a loan backed by nothing but the bank's promise to repay. A senior bond, an interbank loan, most deposits — these are unsecured. The lender's only protection is the bank's overall health and its place in the creditor queue.
- **Secured funding** is a loan backed by a specific pledged asset. If the bank fails to repay, the lender simply keeps (or sells) the collateral. A **repo** and a **covered bond** are the two big secured-funding instruments, and we will dissect both.

Why does this matter? Because a secured lender is far safer, it will accept a far lower interest rate. Secured funding is therefore cheaper — but it works only as long as the bank has good collateral to pledge, and pledging collateral to one lender means it is no longer available to everyone else. That hidden cost — called **encumbrance** — is the catch we will return to.

### Repo, in one paragraph

A **repurchase agreement** (repo) is the workhorse of short-term secured funding. It sounds exotic but it is just a one-day secured loan dressed up as a sale. The bank sells a bond it owns for cash today, and agrees to buy that same bond back tomorrow at a slightly higher price. The price difference is the interest. The lender holds the bond overnight as collateral, so if the bank vanishes, the lender keeps the bond. To protect against the bond falling in value, the lender lends slightly less than the bond is worth — a discount called a **haircut**. Repo is cheap, secured, and enormous, and it can evaporate overnight. We have a whole diagram for it later.

### Covered bonds

A **covered bond** is a long-term bond a bank issues that gives the investor *two* claims at once — what the market calls **dual recourse**. First, like any bond, the investor has a claim on the bank itself. Second — and this is the magic — the investor also has a claim on a ring-fenced **cover pool** of high-quality assets (usually mortgages) that stays on the bank's balance sheet but is legally walled off for the bondholders. If the bank fails, the cover pool is reserved for covered-bond holders before anyone else touches it. Two claims for the price of one means covered bonds are exceptionally safe, so investors accept a notably lower interest rate. They are a staple of European bank funding and increasingly elsewhere.

### Senior versus subordinated

When a bank borrows by issuing a bond, the bond sits somewhere in the **creditor hierarchy** — the legally fixed pecking order that decides who gets paid first if the bank is wound up.

- **Senior** debt is paid before junior debt. Senior unsecured bonds, deposits, and most counterparties sit at or near the top of the unsecured queue.
- **Subordinated** (or "junior") debt is paid only after senior creditors are fully satisfied. Because it absorbs loss sooner, it is riskier, so it pays a higher coupon. The two regulatory flavors are **Tier 2** subordinated debt and **Additional Tier 1** (AT1, also called CoCos — contingent convertibles), the most junior debt of all, which can be written off or converted to equity while the bank is still alive.

### The creditor waterfall and MREL/TLAC

Put the hierarchy together and you get the **creditor waterfall**: in a failure, loss is absorbed strictly top-down. Equity is wiped first. Then AT1. Then Tier 2. Then senior unsecured. Only after every one of those layers is exhausted do uninsured depositors take a loss — and insured depositors (up to \$250,000 per depositor per bank in the US) are made whole regardless. Picture a literal waterfall of losses cascading down through the layers; the depositors are the pool at the bottom that the water should never reach.

To guarantee it never does, regulators invented two acronyms. **MREL** (Minimum Requirement for own funds and Eligible Liabilities, the European rule) and **TLAC** (Total Loss-Absorbing Capacity, the global standard for the biggest banks) both force a bank to keep a *minimum thickness* of bail-in-able layers — equity plus AT1, Tier 2, and qualifying senior debt — large enough to absorb a realistic failure. The word **bail-in** is the opposite of bail-out: instead of taxpayers rescuing the bank from outside, the bank's own investors absorb the loss from inside by being wiped out or converted. MREL and TLAC make sure there is always enough of that bail-in-able money standing between losses and the public.

That is the whole vocabulary. Now let us go deep.

## The funding mix: what a real bank's liability side looks like

Theory is tidy; balance sheets are not. So let us look at the real proportions. The chart below shows the funding mix of a typical large universal bank as a single 100% stacked bar — every dollar of the liability-and-equity side, sorted by source.

![Stacked bar of a large bank funding mix deposits wholesale debt and equity](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-2.png)

The dominant block is unmistakable: **deposits are about 71% of funding**. This is the franchise. Everything good about banking economics flows from a thick, cheap, sticky deposit base, which is exactly why a sister post calls cheap deposits [the whole game](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise). Above that sits **wholesale and repo at about 10%** — fast, market-priced money from other banks and the repo market. Then **long-term debt at about 7%** — the senior and subordinated bonds the bank issues into capital markets. A small **4% of other liabilities** (things like trade payables and derivative margin), and finally **equity at about 8%** — the owners' capital that anchors the whole structure.

Sit with those numbers for a moment, because the proportions *are* the strategy. A bank that pushes its deposit share toward 80% is choosing cheap and sticky over expensive and reliable. A bank that lets wholesale creep toward 25% has chosen the opposite, often because it is growing faster than its deposits can. Northern Rock, the British lender that failed in 2007, had funded roughly three-quarters of its mortgage book from wholesale markets rather than deposits — and when those markets froze, it died within weeks. The mix is not an accounting curiosity. It is a confession of how the bank intends to survive.

### Not all deposits are the same deposit

It is tempting to treat that 71% deposit block as one homogeneous, comfortable mass. That is exactly the mistake SVB's regulators made. Deposits are really three quite different animals wearing the same coat, and a treasurer prices and stress-tests each separately:

- **Operational / transaction deposits** — the checking accounts and corporate cash-management balances that sit at the bank because the customer needs them to run daily life or a business, not to earn interest. They pay almost nothing and they barely move, even when rates rise, because the customer's payroll and bills flow through them. This is the gold of the funding stack: cheap *and* sticky. Banks fight viciously for it through current accounts and corporate treasury services.
- **Retail savings and term deposits** — money parked to earn a return. It is stickier than wholesale because retail savers are slow and insured, but it is rate-sensitive: raise rates too little and it slowly migrates to a competitor or a money-market fund. The fraction of a rate hike a bank must pass through to keep this money is called the **deposit beta**, and through the 2022 to 2024 hiking cycle that beta climbed from about 0.10 to around 0.55 — meaning by the end, banks were passing roughly 55 cents of every dollar of rate increase straight through to savers.
- **Large uninsured / wholesale-like deposits** — big corporate and institutional balances far above the insurance limit. These look like deposits on the balance sheet but behave like wholesale funding: professionally managed, yield-hungry, and gone at the first whiff of trouble. They are the most dangerous line on the liability side precisely because they are filed under the reassuring word "deposits."

The treasurer's nightmare is a balance sheet that *reports* as 71% deposit-funded but is, on closer reading, stuffed with the third kind. That is the gap between how a funding stack looks and how it behaves — and closing that gap is most of the job.

#### Worked example: how deposit beta erodes the cheap-funding advantage

Watch what a rising deposit beta does to the spread. A bank holds \$100 billion of savings deposits. The central bank raises rates by 4 percentage points over a cycle.

- Early in the cycle, deposit beta is **0.10**: the bank passes through only `4% × 0.10 = 0.40%`, so its deposit cost rises from, say, 0.2% to 0.6%. Cheap funding stays cheap. On \$100 billion, extra interest paid: `\$100bn × 0.40% = \$400 million` a year.
- Late in the cycle, beta has climbed to **0.55**: cumulatively the bank must pass through `4% × 0.55 = 2.2%`, lifting deposit cost from 0.2% toward 2.4%. Extra interest paid versus the start: `\$100bn × 2.2% = \$2.2 billion` a year.

The same deposits cost the bank an extra `\$2.2bn − \$0.4bn = \$1.8 billion` a year simply because the beta rose. The intuition: the cheap-deposit advantage is not a fixed gift; it is a *contest* the bank wins only as long as depositors stay loyal at a low rate. Push beta up — or worse, let depositors leave for a money-market fund yielding the full policy rate — and the bottom layer of the funding stack quietly turns expensive. The whole franchise is the ability to keep deposit beta low.

#### Worked example: the blended cost of funds across the stack

Let us make the central trade-off concrete. Suppose a bank funds \$100 of assets and pays these rates on each layer (illustrative, in a roughly 4.5% policy-rate world):

- Deposits, 71% of funding, blended cost **1.6%** (a mix of near-free checking and pricier savings)
- Wholesale and repo, 10%, cost **4.6%**
- Long-term debt, 7%, cost **5.8%** (a blend of senior and subordinated)
- Other liabilities, 4%, cost **0%** (non-interest-bearing)
- Equity, 8%, *cost of capital* **12%** (not an interest expense, but the return owners demand)

The blended **cost of funds** — the weighted average interest the bank pays on its *borrowed* money — counts only the interest-bearing debt, not equity (equity earns profit, it is not borrowed). Weight the four debt layers:

`(0.71 × 1.6%) + (0.10 × 4.6%) + (0.07 × 5.8%) + (0.04 × 0%)`
`= 1.136% + 0.46% + 0.406% + 0% = 2.00%`

So this bank's all-in cost of funds is about **2.0%**. If it lends that \$100 out at, say, 5.5%, its gross spread is `5.5% − 2.0% = 3.5%`. That spread, multiplied across hundreds of billions of dollars, is the bank's lifeblood. The single biggest lever on it is the deposit share: shift just 15 points of funding from 1.6% deposits to 4.6% wholesale and the cost of funds jumps by `0.15 × (4.6% − 1.6%) = 0.45%` — nearly half a point of margin gone. **Cheap funding is not a perk; it is the business model.**

## Climbing the stack: cost rises with every step

The funding mix tells you *how much* of each source a bank uses. The next question is *how much each one costs* — and the answer is a near-perfect ladder. The chart below ranks the all-in cost of each funding layer from cheapest to dearest.

![Bar chart of funding cost from checking deposits to equity cost of capital](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-4.png)

Read it left to right and the logic of the whole post appears. **Checking deposits** cost almost nothing — about 0.3% — because customers leave money in transaction accounts for convenience, not yield. **Savings and term deposits** cost more, around 2.5%, because savers chase rates. A **covered bond** comes in near the policy rate itself, about 4.5%, because its collateral makes it nearly riskless to the lender. **Senior unsecured** debt costs a touch more, say 5.4%, paying a spread over the policy rate for the privilege of being unsecured. Then the steps get steeper: **Tier 2** subordinated debt around 7%, **AT1 / CoCos** around 9%, and finally **equity** — whose "cost" is the ~12% return shareholders demand — sitting at the top.

Notice the dashed line marking the roughly 4.5% policy rate. Everything *below* it (deposits, covered bonds) is funding the bank gets for less than the central bank's own rate — a genuine subsidy that comes from customer trust and collateral. Everything *above* it is funding the bank pays a risk premium for. The art of treasury is to fund as much of the book as possible from below that line.

### Why the ladder slopes the way it does

The ordering is not arbitrary; it falls straight out of two principles we have already met. **Risk to the lender** sets the price: the safer the lender's position, the lower the rate it accepts. A secured covered-bond holder is safest, so it is cheapest. An equity holder is wiped first in any failure, so it demands the most. **Stability to the bank** runs in the opposite direction and is the consolation prize: the expensive layers are also the loyal ones. A covered bond or a ten-year senior note cannot run; it sits on the balance sheet for years by contract. Deposits and repo are cheap precisely because they can leave at will — and that optionality, handed to the lender, is exactly what makes them dangerous.

#### Worked example: a covered bond's lower rate versus senior unsecured

Imagine a bank wants to raise \$1 billion of five-year funding. It has two options.

- Issue a **senior unsecured** bond. Investors have one claim — on the bank. To compensate for that, they demand the policy rate plus a spread of, say, 90 basis points. (A *basis point* is one hundredth of a percent — 0.01%.) At a 4.5% policy rate, the coupon is `4.5% + 0.90% = 5.40%`. Annual interest: `\$1,000,000,000 × 5.40% = \$54,000,000`.
- Issue a **covered bond** backed by a ring-fenced pool of prime mortgages. Investors now have two claims — on the bank *and* on the cover pool — so they accept a spread of only about 30 basis points: `4.5% + 0.30% = 4.80%`. Annual interest: `\$1,000,000,000 × 4.80% = \$48,000,000`.

The covered bond saves the bank `\$54,000,000 − \$48,000,000 = \$6,000,000` a year, or \$30 million over the five-year life of the bond, for the *identical* amount and tenor. That \$6 million a year is the literal price the market puts on dual recourse. The catch, which never shows up on the coupon, is that the bank has now **encumbered** a pool of its best mortgages — pledged them away — so if it later needs to raise emergency cash by selling assets, those mortgages are spoken for. The intuition: a covered bond buys cheap, stable funding by mortgaging your safest assets to the new lender, which quietly weakens everyone behind them in the queue.

## Reading any funding source on four axes

By now you can see that every funding source is a bundle of attributes that trade off against each other. The cleanest way to hold them all in your head is a single table: each source rated on cost, stability, whether it is secured, and whether it can be bailed in. The matrix below lays it out.

![Matrix of funding sources rated on cost stability secured and bail-in status](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-3.png)

Walk the rows and the personalities emerge:

- **Retail deposits** — cheapest of all, but flighty (they can run in hours); unsecured, though insured up to the limit; and *protected* in the waterfall, taking loss only after everything above them is gone. The dream funding, with one nightmare attached: the run.
- **Wholesale and repo** — cheap when markets are calm and priced near the policy rate, but the flightiest source of all because it rolls over daily or weekly. Repo is secured (the lender holds collateral); plain interbank lending is not. Secured repo steps around the bail-in question entirely, which is its own kind of problem we will see.
- **Covered bond** — low cost, high stability (a 5-to-10-year contractual term), fully secured, and *not* bail-in-able thanks to dual recourse. The best of both worlds, paid for with encumbrance.
- **Senior unsecured bond** — moderate cost, stable (3-to-10-year term), unsecured, and bail-in-able after the subordinated layers are exhausted. The bulk of a bank's bond funding and the workhorse of MREL/TLAC.
- **Subordinated (Tier 2 / AT1)** — the most expensive debt, very stable (long-dated or perpetual), deeply junior, and the *first* debt to absorb loss. This is the bail-in cushion, paid for with a fat coupon.

The pattern in the table is the thesis of the whole post: **cheap money is flighty and protected; expensive money is loyal and loss-absorbing.** A covered bond is the single exception that escapes the trade-off — and it does so only by handing over collateral, which is why no bank can fund itself entirely with covered bonds. Encumber everything and there is nothing left to reassure the unsecured lenders, who then demand a higher rate or simply leave.

## Repo: the cheapest, fastest, most fragile funding of all

Wholesale funding deserves its own dissection, because it is the layer that kills banks. And the heart of wholesale funding is the **repo market**. A bank with a portfolio of high-quality bonds can turn those bonds into cash overnight, again and again, for a rate barely above the policy rate. It is astonishingly cheap. It is also the single most treacherous funding a bank can rely on. The diagram below walks one repo transaction end to end.

![Pipeline of a repo selling a bond for cash and buying it back next day](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-6.png)

Follow the steps. The bank owns a bond worth \$100. It sells that bond to a money-market fund for \$98 in cash — the \$2 gap is the **haircut**, the lender's cushion against the bond falling in value. The bank now has \$98 of cash to fund its operations for the day. The next morning, it buys the bond back for \$98 plus a sliver of interest — say \$98.01 — and gets its bond returned. The loan is repaid, the collateral is back, and the whole thing repeats tomorrow.

#### Worked example: refinancing risk on wholesale funding

Now see why this is dangerous. Suppose a bank funds \$20 billion of its assets with overnight repo, rolled every single day. On a normal day, rolling \$20 billion is a non-event — the same lenders show up, the same bonds are pledged, the cash recycles. The bank's *refinancing risk* — the risk that it cannot replace maturing funding when it comes due — feels like zero.

Then confidence wobbles. Lenders get nervous about either the bank or its collateral. They do two things at once. First, they raise the haircut from 2% to 8%. Overnight, the same \$20 billion of bonds now raises only `\$20bn × (1 − 0.08) = \$18.4bn` instead of `\$20bn × (1 − 0.02) = \$19.6bn` — a `\$1.2bn` hole to fill from somewhere else, today. Second, some lenders simply stop showing up. If even a quarter of them walk, the bank must replace `\$20bn × 0.25 = \$5bn` of funding by this afternoon. Stack the two effects and the bank has to find well over \$6 billion of cash in hours, against assets it cannot sell that fast without crushing their price.

That is **refinancing risk**, and it is the precise mechanism by which Lehman Brothers died in 2008 and the repo market froze again in 2019. The intuition is brutal and worth memorizing: overnight funding is cheap because the lender keeps the option to leave every single morning — and on the one morning they all leave together, no amount of solvency saves you. Lehman, funding hundreds of billions in short-term markets at over 30 times leverage, even resorted to a trick called Repo 105 to temporarily shuffle \$50 billion off its balance sheet at quarter-end to look less leveraged than it was. The deeper plumbing of all this lives in a dedicated post on [the repo market and how banks fund overnight](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight) and in the system-level view of [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).

### The interbank market: banks lending to each other

Alongside repo sits the older cousin of wholesale funding: the **interbank market**, where banks lend cash to each other directly, usually unsecured and usually very short-term — overnight to a few months. A bank that ends the day with surplus cash lends it to a bank that ends the day short, at a rate near the central bank's policy rate. For decades the benchmark for this was LIBOR (the London Interbank Offered Rate); today it is rates like SOFR in the US and ESTR in Europe. The interbank market is the system's shock absorber: on any normal day, cash sloshes from cash-rich banks to cash-poor ones and the whole system clears.

But because interbank lending is *unsecured* — no collateral, just one bank's faith in another — it is the first thing to vanish when trust cracks. In a crisis, banks stop lending to each other precisely when they most need to, because no one can tell which counterparty is about to fail. The interbank market did exactly this in 2008: lending between banks seized almost completely, spreads blew out, and the system froze until central banks stepped in as lender of last resort. The lesson folds neatly into our theme: unsecured wholesale funding is cheap and convenient in good times and *structurally unavailable* in the bad times when you need it most. That is why a prudent bank treats interbank lines as a convenience, never as a load-bearing pillar.

### Encumbrance: the hidden cost of secured funding

We have praised secured funding — repo, covered bonds — for being cheap. Now the bill. Every dollar of secured funding requires pledging collateral, and pledged collateral is **encumbered**: legally promised to one lender and therefore unavailable to anyone else. The more of a bank's good assets are encumbered, the *less* is left to reassure its unsecured creditors and depositors, who get whatever remains after the secured lenders have helped themselves. High encumbrance quietly subordinates everyone unsecured — and they notice.

#### Worked example: how encumbrance subordinates unsecured creditors

A bank has \$100 billion of assets and \$90 billion of liabilities. Start with zero secured funding. If the bank fails, all \$100 billion of assets are available to all \$90 billion of creditors — they would recover the full \$90 billion, or close to it. Now the bank raises \$40 billion of cheap covered-bond and repo funding by pledging \$45 billion of its best assets (the extra \$5 billion is the overcollateralization the secured lenders demand). Those \$45 billion of assets are now encumbered.

If the bank fails, the secured lenders take their \$45 billion of pledged assets and walk away whole. That leaves `\$100bn − \$45bn = \$55 billion` of assets for the remaining `\$90bn − \$40bn = \$50 billion` of unsecured creditors and depositors — still covered here, but watch what happens with a loss. Suppose assets are actually worth only \$80 billion in the failure. Secured lenders still take their \$45 billion in full. The unsecured creditors are left fighting over `\$80bn − \$45bn = \$35 billion` against \$50 billion of claims — a recovery of only `\$35bn / \$50bn = 70 cents` on the dollar. Had there been no encumbrance, all \$80 billion would have been shared across all \$90 billion of claims, a recovery of about 89 cents. The intuition: secured funding looks free on the coupon, but it shifts risk onto the unsecured layers, who eventually demand a higher rate or impose a cap on how much a bank may encumber. There is, once again, no free lunch in the funding stack — only a relocation of the bill.

## The funding pyramid: cheap-and-flighty versus expensive-and-stable

Step back and the whole stack resolves into one trade-off you cannot escape: **cost versus stability**. Plot every funding source on those two axes and they fall on a downward-sloping line. The cheapest sources (checking deposits, overnight repo) are the least stable. The most stable sources (long-dated bonds, equity) are the most expensive. There is no source that is both free *and* permanent — except the one trick, the covered bond, which buys a little stability at low cost by pledging collateral.

This is why a bank's treasury does not simply minimize cost. If it did, it would fund everything with overnight repo and free checking deposits, and it would die in the first panic. Nor does it simply maximize stability — fund everything with ten-year bonds and equity and the cost of funds would be so high the bank could not lend profitably. The treasury's job is to find the *blend* that funds the book cheaply enough to make money but stably enough to survive a stress. That blending decision, run through a committee called ALCO and a discipline called funds transfer pricing, is the subject of [bank treasury and asset-liability management](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit).

### The funding ladder: spreading out when the money comes due

There is a second dimension to stability that the cost-versus-source picture hides: *when* each piece of funding matures. A bank can hold a perfectly sensible mix of bonds and still be fragile if all of those bonds happen to come due in the same quarter. If \$30 billion of senior bonds all mature in March, the bank must refinance \$30 billion in a single window — and if markets happen to be shut that month, it has a problem no amount of long-dated funding solved. The defense is a **funding ladder** (also called a maturity ladder): deliberately staggering bond maturities so that only a manageable slice rolls off in any one period.

Think of it the way a careful saver staggers term deposits — some maturing each year — so there is always cash coming free and never a single cliff. A bank does the same with its bond stack: it might target no more than, say, \$5 billion of bonds maturing in any single quarter, so that even a closed market for a few months forces it to refinance only a slice it can cover from its liquidity buffer. Banks that ignored the ladder — that let large amounts of wholesale funding bunch up at the same near-term maturity — are the ones that turned an awkward market into a fatal one. The ladder converts a single terrifying refinancing cliff into a series of small, survivable steps.

#### Worked example: a maturity cliff versus a smooth ladder

Two banks each fund \$20 billion with five-year senior bonds. Bank A issued all \$20 billion in one go three years ago, so the entire \$20 billion matures in the same quarter two years from now. Bank B issued \$4 billion a year for five years, so it always has \$4 billion maturing each year and \$4 billion of fresh issuance to do — a smooth ladder.

Now imagine the bond market shuts for six months at exactly the wrong moment. Bank A must refinance the full \$20 billion in a window when no one is buying; if it cannot, it must dump assets at fire-sale prices to raise \$20 billion of cash, potentially crystallizing losses that eat its capital. Bank B, in the same shut market, only needs to refinance the \$4 billion maturing that year — and can cover most of that from its liquidity buffer until the market reopens. Same total funding, same instrument, same cost — but Bank A faces a `\$20bn` cliff and Bank B faces a `\$4bn` step. The intuition: stability is not only about *what* you fund with but *when it comes due*; a laddered maturity profile is the cheapest insurance a treasurer can buy against the market being closed on the one day you need it open.

### How the blend shifts the cost of funds

The next chart makes the trade-off quantitative. It shows the bank's blended cost of funds as the mix slides from deposit-rich on the left toward wholesale-reliant on the right.

![Line chart of blended cost of funds rising as wholesale funding share grows](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-7.png)

The line slopes up, and steeply. A **deposit-rich bank** — only 10% of its funding from wholesale and bonds — funds itself at about **1.99%**. A **balanced bank** at 25% wholesale pays about **2.58%**. A **wholesale-reliant bank** at 45% pays about **3.36%**. That is a 1.37-point swing in the cost of funding the exact same assets.

#### Worked example: what the cost-of-funds gap is worth

Take two banks, each with a \$200 billion loan book yielding 5.5%, identical in every way except their funding mix.

- Bank A is deposit-rich: cost of funds 1.99%. Its gross spread is `5.5% − 1.99% = 3.51%`, earning `\$200bn × 3.51% = \$7.02 billion` of net interest before costs.
- Bank B is wholesale-reliant: cost of funds 3.36%. Its gross spread is `5.5% − 3.36% = 2.14%`, earning `\$200bn × 2.14% = \$4.28 billion`.

Same assets, same loan rates — and Bank A out-earns Bank B by `\$7.02bn − \$4.28bn = \$2.74 billion` a year, purely on the funding mix. And that is before the stability difference: in a stress, Bank B's wholesale funding can vanish while Bank A's insured deposits mostly stay. The intuition: the funding stack is not a back-office detail; it is the largest single determinant of both a bank's profitability *and* its survival. Cheap, stable funding is the closest thing in banking to a free lunch — and it is earned over decades, through branch networks and customer trust, not bought overnight.

## The creditor waterfall: who gets paid, who eats the loss

We have priced the stack and we have weighed its stability. The third dimension — the one that only matters when a bank fails, but matters absolutely then — is the **creditor hierarchy**. The before-and-after diagram below shows the same funding stack in two states: a solvent bank, where every layer is whole and the ordering is invisible, and a failing bank, where the waterfall runs top-down.

![Before and after of the creditor waterfall in a solvent versus failing bank](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-5.png)

On the left, the solvent bank, nobody thinks about rank. Equity earns the profit and quietly absorbs the small day-to-day losses. AT1 and Tier 2 collect their fat coupons. Senior bonds and repo roll over and get repaid on schedule. Depositors are paid on demand, never once asked where they stand in a queue. The hierarchy exists, but it is dormant.

On the right, the failing bank, the order becomes everything. Loss cascades strictly top-down. **Equity is wiped first** — shareholders get zero before any creditor loses a cent. Then **AT1 is written off**, then **Tier 2** absorbs loss. Only if the loss is deep enough does it reach **senior unsecured** bonds, which are bailed in next. And only after all of those layers are exhausted do uninsured depositors take a hit — with insured depositors made whole regardless. The whole point of stacking the layers in this order is to put the most loss-absorbing money (equity, then the riskiest debt) between the disaster and the ordinary depositor.

#### Worked example: the creditor waterfall in a failure

Make it numeric. A bank fails with a hole in its balance sheet — its assets are worth \$10 billion less than its liabilities. Here is the funding stack it failed with, and how the \$10 billion of loss cascades down:

- **Equity: \$8 billion.** Absorbs loss first. Fully wiped. Remaining loss to allocate: `\$10bn − \$8bn = \$2 billion`.
- **AT1: \$1.5 billion.** Next in line. Fully written off. Remaining: `\$2bn − \$1.5bn = \$0.5 billion`.
- **Tier 2: \$1 billion.** Absorbs the rest. It takes a `\$0.5bn` hit — a 50% loss — and `\$0.5bn` survives. Remaining: **\$0.**

The loss is fully absorbed before it ever reaches senior bonds or any depositor. Equity holders lose everything, AT1 holders lose everything, Tier 2 holders lose half — and senior creditors and depositors lose *nothing*. That is the system working as designed. Had the hole been \$12 billion instead, the extra \$2 billion would have chewed into senior unsecured bonds (bailing them in) before any depositor was touched. The intuition: the funding stack is also a loss-absorption stack, built so that the people who signed up for risk — and got paid extra for it — are the ones who actually bear it. A senior bond's lower coupon and a deposit's near-zero rate are the *price of standing further back in the queue*, paid in advance.

## Bail-in, MREL and TLAC: making the waterfall mandatory

The waterfall only protects depositors and taxpayers if there is *enough* loss-absorbing material stacked above them. In 2008, there often was not — banks held thin slivers of equity and junior debt, the loss blew straight through, and governments bailed them out with public money. The post-crisis fix was to make the cushion mandatory. The stack diagram below shows the bail-in hierarchy and the band that MREL and TLAC require a bank to maintain.

![Stack of the bail-in hierarchy from equity to deposits with MREL TLAC cushion](/imgs/blogs/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds-8.png)

The order, outside-in, is the same waterfall: **equity (CET1)** absorbs loss first, then **AT1**, then **Tier 2**, then **senior** (often a special "senior non-preferred" class created precisely to sit just above deposits in the queue), and finally the protected **deposit core**. What MREL and TLAC add is a *floor*: a bank must hold a combined amount of these bail-in-able layers — equity plus AT1, Tier 2, and qualifying senior debt — large enough that, in a realistic failure, the loss is fully absorbed before the line marked "deposits." For the biggest global banks, TLAC requires loss-absorbing capacity of roughly 18% or more of risk-weighted assets, and MREL sets a comparable European bar.

#### Worked example: sizing the MREL/TLAC cushion

Suppose a bank has \$100 billion of risk-weighted assets and a TLAC requirement of 18%. It must therefore hold `\$100bn × 18% = \$18 billion` of loss-absorbing capacity. Say its capital stack is:

- CET1 equity: \$8 billion
- AT1: \$1.5 billion
- Tier 2: \$2 billion

That is `\$8bn + \$1.5bn + \$2bn = \$11.5 billion` of capital — well short of the \$18 billion required. To close the `\$18bn − \$11.5bn = \$6.5 billion` gap, the bank issues \$6.5 billion of qualifying **senior non-preferred** bonds, which count toward TLAC because they are bail-in-able before depositors. Now the bank carries an \$18 billion wall of money that can be wiped or converted before a single depositor or taxpayer is touched. The intuition: MREL and TLAC turned the creditor waterfall from a hope into a building code. A modern bank is *required* to fund a defined slice of itself with money that has explicitly agreed to die first — and the cost of that mandated bail-in debt is one of the quiet new expenses of being a bank.

This is also the layer where the theory becomes painfully real. When Credit Suisse failed in March 2023, its **AT1 bonds were written down to zero** — about CHF 16 billion wiped out — even as shareholders received CHF 3 billion in the UBS rescue. AT1 holders were furious, but the bail-in worked exactly as the contracts said it would: the loss-absorbing layer absorbed the loss. The system-level account of that episode lives in [the SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

## Common misconceptions

**"Deposits are the safest funding because they're insured."** Safe for the *depositor*, yes — up to \$250,000 in the US. Stable for the *bank*, not necessarily. Insurance covers small retail balances, but large corporate and wealthy depositors routinely hold far more than the limit, and *uninsured* deposits are the flightiest money on the balance sheet. SVB's deposits were 94% uninsured; that is why \$42 billion could leave in a day. Insured retail deposits are sticky; uninsured wholesale-like deposits are a wholesale run waiting to happen.

**"Wholesale funding is dangerous, so a good bank avoids it entirely."** No bank avoids it, and avoiding it isn't even desirable. Wholesale and repo funding is cheap, flexible, and lets a bank fund growth faster than deposits arrive. The danger is not its existence but its *share* and its *tenor*. A bank with 10% short-term wholesale funding is fine; one with 40% rolled overnight is a Northern Rock waiting to happen. The metric that matters is how much funding must be refinanced in a stress window, not whether wholesale appears at all.

**"A covered bond is just a safer senior bond."** It is structurally different. A senior bond gives one claim — on the bank — and is bail-in-able. A covered bond gives *two* claims (dual recourse) — on the bank and on a ring-fenced cover pool — and is generally *exempt* from bail-in. That is why it prices 30 to 60 basis points cheaper. The hidden cost is encumbrance: the assets in the cover pool are pledged away and can no longer protect unsecured creditors, so loading up on covered bonds quietly subordinates everyone else.

**"Bail-in means depositors can lose their money."** For *insured* depositors, essentially never — they sit below the bail-in line and are made whole. The entire architecture of MREL and TLAC exists to stack enough equity and bail-in-able debt above depositors that the loss is exhausted before it reaches them. In a properly capitalized bank, bail-in burns equity holders, AT1 holders, Tier 2 holders, and possibly senior bondholders — investors who were paid extra to stand in front. Depositors are the people the system is designed to protect.

**"Equity is just another funding source the bank pays for."** Equity is not borrowed and is never repaid, so it is not "funding" in the same sense — there is no maturity, no coupon, no run risk. But it is the *most expensive* claim on the bank, because shareholders demand a high return (the cost of capital, ~12%) for taking first loss. A bank does not minimize equity because it is cheap (it is the dearest money of all); it holds equity because it is the only money that absorbs loss without triggering a default. Equity is the price of being allowed to run a leveraged machine at all.

## How it shows up in real banks

**Silicon Valley Bank, 2023 — the uninsured-deposit run.** SVB looked deposit-funded and therefore "stable," but its deposits were overwhelmingly large, uninsured tech-company balances — the wholesale-like end of the deposit spectrum. With 94% of deposits uninsured, the moment confidence broke, depositors raced for the exit: \$42 billion gone on March 9, \$100 billion queued for March 10. The lesson of the funding stack made flesh: *the label "deposit" does not tell you the stability; the depositor's insurance status and concentration do.* A funding base that looks like the cheap, sticky bottom of the pyramid can behave like the flighty top.

**Northern Rock, 2007 — wholesale dependence.** The British lender funded roughly three-quarters of its mortgage book not from retail deposits but from short-term wholesale and securitization markets. The model was cheap and let it grow blazingly fast. When the wholesale market froze in August 2007, the funding it relied on simply was not there to roll, and Britain saw its first depositor run since 1866. It was nationalized within months. Northern Rock is the canonical case of choosing the cheap-and-flighty layers over the cheap-and-sticky one and paying with the bank's life.

**Lehman Brothers, 2008 — the repo run.** Lehman, an investment bank with almost no retail deposits, funded itself at over 30 times leverage largely through short-term repo. When lenders raised haircuts and stopped rolling, the refinancing hole opened in days. Lehman even used Repo 105 to temporarily move \$50 billion off its balance sheet at quarter-end to mask the leverage. When the repo market lost faith, no amount of asset value could be turned into cash fast enough, and the firm filed for bankruptcy on September 15, 2008. The repo layer giveth cheap funding and taketh away overnight.

**Credit Suisse, 2023 — the AT1 wipeout.** A decade of scandals eroded the one thing a bank cannot function without — trust — and CHF 110 billion of outflows in late 2022 turned a slow bleed into a crisis. In the March 2023 rescue, regulators triggered the bail-in: roughly CHF 16 billion of AT1 bonds were written to zero while shareholders salvaged CHF 3 billion in the UBS takeover. It was the largest AT1 loss in history and a live demonstration that the bail-in-able layers do exactly what they were designed to do — absorb loss so the broader system does not.

**The European covered-bond market — funding through the storm.** While unsecured bank funding seized up repeatedly during the 2008 crisis and the 2011 eurozone debt crisis, the covered-bond market — centuries old, dominated by German Pfandbriefe and their cousins — kept functioning. Banks could still raise term money against ring-fenced mortgage pools when no one would lend to them unsecured. That resilience is the entire reason covered bonds exist and a large reason regulators treat them favorably: dual recourse makes them the funding source most likely to survive a panic, which is exactly when funding is most precious.

**The 2019 repo spike — even the giants are exposed.** In September 2019, overnight repo rates briefly spiked from around 2% to nearly 10% as cash mysteriously drained from the system and lenders pulled back. No bank failed, but the episode jolted even the largest, best-funded institutions and forced the Federal Reserve to inject hundreds of billions to calm the market. The lesson: the cheap, fast repo layer can seize up for plumbing reasons that have nothing to do with any single bank's health — which is why no prudent treasury lets it become a load-bearing share of funding.

**The savings-and-loan crisis — funding short while lending long.** The American thrifts of the 1980s offer the oldest and clearest case in this post. They funded themselves with short-term deposits and lent the money out as 30-year fixed-rate mortgages — the maturity-transformation trade at its most extreme. When the Federal Reserve drove short-term rates into the high teens to fight inflation, the thrifts had to pay those soaring rates on their *funding* while their *assets* were locked into old, low-rate mortgages. Their cost of funds blew straight past the yield on their loans, and the spread that was supposed to keep them alive turned negative. More than a thousand thrifts failed, at a cleanup cost to taxpayers of roughly \$124 billion. The funding stack was not flighty here — it was simply mispriced against the assets. It is the same disease that felled SVB four decades later: a funding side that reprices faster than the asset side, dressed up in different clothes.

**A well-funded survivor — the deposit fortress.** It is worth naming the counter-case, because survival is as instructive as collapse. The banks that sailed through 2008 and 2023 with the least drama were the ones with thick, granular, insured retail-deposit bases — millions of small, sticky accounts spread across a vast branch network, the operational deposits we met earlier. When wholesale markets froze and uninsured money fled the weak banks, these institutions actually *gained* deposits as frightened customers fled to perceived safety. Their cost of funds barely moved, their funding never had to be refinanced in a panic, and several of them were able to scoop up failed rivals at fire-sale prices. The lesson is the whole post in one sentence: the bank with the cheapest *and* stickiest base of the funding stack does not merely survive a crisis — it inherits the spoils of the banks that didn't.

## The takeaway / How to use this

If you remember one thing about how a bank funds itself, make it this: **the funding stack is a single, unavoidable trade between cost and stability, and where a bank sits on that trade tells you almost everything about how it will behave in a crisis.** Cheap money is flighty; loyal money is dear; and the only escape — the covered bond — is bought by mortgaging your safest assets to a new lender, which weakens everyone behind them.

So when you read a bank, do not stop at "it's deposit-funded." Read the stack the way the bank's own treasurer does. Ask three questions, in order. First, *how cheap is the funding* — what share comes from below the policy-rate line (sticky deposits, covered bonds) versus above it (wholesale, subordinated debt)? That sets the bank's profitability. Second, *how stable is it* — how much of the funding can leave in a 30-day stress window, and how concentrated and uninsured are the deposits? That sets the bank's survival odds. Third, *who eats the loss if it fails* — how thick is the bail-in-able band of equity, AT1, Tier 2, and senior debt that MREL and TLAC require to stand between losses and depositors? That tells you whether the next failure is absorbed quietly by investors or violently by the public.

This connects straight back to the spine of the whole series. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and lends long, earns the spread, and lives only as long as its funders keep the faith and its thin equity absorbs losses faster than they arrive. The funding stack *is* the borrowing-short side of that trade, laid out layer by priced layer. SVB did not fail because it ran out of assets; it failed because the cheapest-looking part of its funding turned out to be the flightiest, and there was no time to climb the stack to something stickier. Read the funding stack well, and you will see that fragility — the gap between what a bank's funding costs and how long it will actually stay — long before it shows up in the share price.

Notice, too, how the three questions interlock rather than trade off cleanly. The cheapest funding (operational deposits) is also among the stickiest, which is why a great deposit franchise is so prized — it is the rare corner of the stack where cost and stability point the same way. Everywhere else, you pay for stability: in coupon (bonds over deposits), in collateral (covered bonds and repo over unsecured), or in dilution (equity over debt). A bank's whole liability-side strategy is a series of bets about where on that surface it can afford to sit given how it lends. A bank that makes long, illiquid loans *must* fund longer and stickier, even though it costs more, or it is building a duration mismatch with a fuse on it. A bank that makes short, liquid loans can afford flightier funding. The funding stack and the asset book are two halves of one decision, and reading either in isolation will fool you. SVB's mistake was funding a long-duration bond portfolio with the flightiest deposits in the country — a mismatch the funding stack made visible to anyone who bothered to read both sides at once.

A closing note in the spirit of honesty: none of this is investment advice. It is a way of reading a bank's machinery. But it is the same lens supervisors and bank treasurers use, and it is the most reliable early-warning system finance has — because in banking, the asset side tells you how much money the bank can make, and the funding side tells you whether it will live long enough to make it.

## Further reading & cross-links

- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — the bottom of the stack in full: CASA ratios, deposit stickiness, and why a cheap deposit base is the whole game.
- [The repo market and how banks fund overnight](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight) — haircuts, rehypothecation, and the freezes that turn the cheapest funding into the deadliest.
- [Bank treasury and asset-liability management: the balance-sheet cockpit](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit) — the ALCO and funds-transfer-pricing machinery that actually decides the funding mix.
- [Basel I, II, III and the capital rules that govern every bank](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank) — where the capital stack (CET1, Tier 1, Tier 2, AT1) and the loss-absorption rules come from.
- [SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level account of the two failures this post leans on.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — how secured wholesale funding works beyond the regulated banking system.
