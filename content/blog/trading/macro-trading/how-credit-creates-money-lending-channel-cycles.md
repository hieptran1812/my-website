---
title: "How Credit Creates Money: The Lending Channel That Drives Every Cycle"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Most money is created by commercial banks when they make loans, not printed by the central bank — and because credit IS money creation, the credit cycle is the real engine behind every boom, bust, and bubble that traders try to time."
tags: ["macro", "monetary-policy", "credit-cycle", "money-creation", "banking", "credit-impulse", "liquidity", "asset-bubbles", "lending-standards", "macro-trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Most money in a modern economy is not printed by the central bank; it is created by commercial banks the instant they make a loan, and because credit *is* money creation, the credit cycle is the real engine behind every boom, bust, and bubble.
>
> - When a bank makes a loan it simultaneously creates a deposit — both sides of its balance sheet expand and brand-new money exists that did not a second before. Loans create deposits, not the other way around.
> - Reserves do not constrain lending. Bank capital, profitable loan demand, and regulation do. The textbook "money multiplier" gets the causality backwards.
> - The credit cycle is self-reinforcing: easy lending lifts asset prices, higher prices raise collateral values, richer collateral justifies more lending — until the loop snaps into a bust.
> - The single best leading signal is the **credit impulse** — the *change in the flow* of new credit. It turns up or down two to three quarters before growth and equities react. The one number to remember: US M2 grew **+24% in 2020**, the fastest broad-money surge in postwar history, and CPI peaked at **9.1%** eighteen months later.

In late 2020, with the world locked down and the economy supposedly broken, something strange was happening in the plumbing of money. US bank deposits were exploding. By the end of the year, the broad money supply — M2, the total of cash plus checking and savings balances the public holds — had grown by roughly a quarter in a single twelve-month stretch. Nothing like it had happened since the Second World War. House prices took off. Stocks staged one of the sharpest recoveries on record. Crypto went vertical. SPACs, meme stocks, used cars, lumber — everything that could be bought with borrowed or freshly created money went up.

A year and a half after that money surge, inflation printed 9.1%, a forty-year high. Everyone has a story about why: supply chains, energy, stimulus checks, war in Ukraine. All of those mattered at the margin. But the deepest story is the one almost nobody tells cleanly, because it requires unlearning the textbook: a flood of *new money* had been created, most of it not by the Federal Reserve's printing press but by the ordinary act of banks and the government injecting credit and deposits into the system — and a flood of new money chasing a constrained supply of goods and assets is what a credit boom looks like from the inside.

This post is about the mechanism underneath that story, and about how to *trade* it. The mechanism is deceptively simple and almost universally misunderstood: **commercial banks create money when they lend.** Credit and money are, to a first approximation, the same thing being born at the same instant. Once you see that, the credit cycle stops being an abstraction in a textbook and becomes the single most important thing a macro trader watches — because credit growth and the credit impulse turn *before* prices do. The figure below is the mental model for the whole post: watch what happens to a bank's balance sheet the moment it makes a loan.

![Bank balance sheet before and after a new loan showing a deposit appearing](/imgs/blogs/how-credit-creates-money-lending-channel-cycles-1.png)

If you internalize that one picture — a loan asset and a matching deposit popping into existence together, both sides of the ledger growing at once — the rest follows. You will understand why reserves do not constrain lending, why quantitative easing did not directly create the broad money everyone feared, why credit booms reliably end in busts, and most importantly, which handful of signals tell you where in the cycle you are *before* the crowd figures it out. Let us build the whole thing from zero, defining every term as we go, and grounding each step in arithmetic you can check on a napkin.

## Foundations: what a bank actually does

Before we can say how credit creates money, we have to be honest about what a bank really is, because the everyday picture is wrong in a way that matters.

Here is the everyday picture, the one most people carry around: a bank is a *middleman for savings*. Savers deposit money. The bank keeps a fraction in the vault and lends the rest out to borrowers. The bank earns the difference between the interest it pays savers and the interest it charges borrowers. In this story, the bank is a warehouse that moves *existing* money from people who have it to people who need it. It is intuitive, it is what the word "loan" suggests, and it is how fractional-reserve banking is taught in most introductory economics classes.

It is also, as a description of how money is created, backwards.

Let me define the terms first so nothing is hand-waved.

**A bank** is a licensed institution that takes deposits and makes loans, and — crucially — whose deposit liabilities are accepted by the public *as money*. When you have \$5,000 "in the bank," you do not own \$5,000 of cash sitting in a vault with your name on it. You own a *promise from the bank* to pay you \$5,000 of cash on demand. That promise is so reliable, so instantly spendable by card and transfer, that for all practical purposes the number in your account *is* money. This is the single most important fact about modern banking: **a bank deposit is a liability of the bank that the public treats as money.** Banks are special precisely because their IOUs circulate as money.

**A deposit** is money the public holds *at* a bank — the balance in your checking or savings account. To you it is an asset (the bank owes you). To the bank it is a *liability* (the bank owes you). The same balance is an asset on your books and a liability on the bank's books. Hold that mirror image; it is the key to everything.

**A loan** is the reverse mirror. When the bank lends you \$200,000, *you* owe *the bank*. To the bank the loan is an *asset* (you owe it, with interest). To you the loan is a *liability* (you owe the bank). Again, the same contract sits on both balance sheets with opposite signs.

**A balance sheet** is the two-column ledger every bank keeps. On the left, **assets**: everything the bank owns or is owed — loans it has made, bonds it holds, reserves at the central bank, cash. On the right, **liabilities plus equity**: everything the bank owes — deposits it owes customers, bonds it has issued, money it has borrowed — plus **equity** (also called *capital*), which is the owners' stake, the cushion that absorbs losses. By the iron law of accounting, the two columns must equal: assets = liabilities + equity. Every transaction that changes one side must change the other side, or change two entries on the same side that cancel. There are no exceptions. This rule is what makes money creation *visible* — you can literally watch it happen as new lines on the ledger.

**A T-account** is just a stripped-down balance sheet drawn as a "T," with assets on the left of the vertical bar and liabilities plus equity on the right. It is the standard tool for showing what a single transaction does to a bank. We will use it constantly.

So what does a bank actually *do*? It does not warehouse and re-lend your cash. It manufactures money. When a creditworthy borrower wants a loan and the bank agrees, the bank does not go find existing money to hand over. It writes two new entries into its own ledger: a new loan on the asset side, and a new deposit on the liability side. The borrower walks away with a fresh deposit — new money — that did not exist before the loan was signed. The Bank of England said exactly this in plain language in its 2014 Quarterly Bulletin paper "Money creation in the modern economy": *"Whenever a bank makes a loan, it simultaneously creates a matching deposit in the borrower's bank account, thereby creating new money."* This is not a fringe or heterodox claim. It is the official position of central banks. It is just rarely the version taught first.

The reason this matters so much for a trader: if money is mostly created by lending, then *the pace of lending is the pace of money creation*, and the pace of money creation drives spending, asset prices, and inflation. Watching credit is watching the money supply being born in real time. Everything in this post builds on that single inversion of the textbook.

## Loans create deposits: the T-account walkthrough

Let us make it concrete and follow real balance-sheet numbers. We will watch a single mortgage get made and follow every dollar.

Imagine a small bank with a tidy balance sheet. On the asset side it holds \$50 of reserves and other assets. On the liability side it owes \$40 of existing deposits and has \$10 of equity (capital). Assets of \$50 equal liabilities-plus-equity of \$50. The books balance. (I am using small round numbers so the arithmetic is trivial; multiply by a billion if you want realism.)

Now a customer walks in and qualifies for a \$200 loan — say, a mortgage. The bank approves it. Watch precisely what happens on the ledger, and notice what does *not* happen.

What does *not* happen: the bank does not move \$200 from its existing reserves to the borrower. It does not call other depositors and ask to borrow their balances. It does not wait to receive new deposits. None of that.

What *does* happen, in a single accounting instant: the bank creates a new **asset** — a \$200 loan, the borrower's promise to repay — and a matching new **liability** — a \$200 deposit credited to the borrower's account. Two new lines, one on each side, both \$200. The borrower's account now shows \$200 they can spend. That \$200 is brand-new money. It was not taken from anyone. It was *created* by the act of lending.

#### Worked example: a \$200,000 mortgage conjures a \$200,000 deposit

Let us scale to a real mortgage. You qualify for a \$200,000 home loan. The morning of closing, you have, say, \$5,000 in your checking account. The bank approves the mortgage. Here is the bank's T-account, before and after, in dollars:

```
BEFORE the loan
  Assets                       Liabilities + Equity
  Reserves + other   $50.000B    Deposits          $40.000B
                                 Equity            $10.000B
  --------------------------     -----------------------------
  Total              $50.000B    Total             $50.000B

AFTER making your $200,000 mortgage
  Assets                       Liabilities + Equity
  Reserves + other   $50.000B    Deposits          $40.000B
  NEW loan to you    $0.0002B    NEW deposit (you)  $0.0002B
                                 Equity            $10.000B
  --------------------------     -----------------------------
  Total              $50.0002B   Total             $50.0002B
```

Both sides of the bank's balance sheet just grew by exactly \$200,000. Your loan (the bank's asset) is \$200,000; the deposit the bank credited to the seller's account when the home purchase settles — or to your account first, then transferred — is \$200,000. The bank did not reach into a vault. It did not lend out somebody's savings. It typed \$200,000 into existence on both sides of its ledger at once, and the iron law of accounting (assets = liabilities + equity) held perfectly throughout because the two new entries are equal and opposite. The intuition: **a loan does not move money from saver to borrower; it manufactures new money out of a creditworthy promise.**

Now, two natural objections, because a sharp reader will raise them.

*"Doesn't the money leave the bank when I spend it?"* Yes — and this is where reserves enter, but not as a constraint on creation. When you spend your \$200,000 (you pay the home seller), the deposit moves to the seller's bank. To settle that, your bank transfers \$200,000 of *reserves* (its balance at the central bank) to the seller's bank. So your bank does need reserves *to settle the payment* — but it needs them *after* the loan is made, and it can borrow them in the interbank market or from the central bank, which supplies reserves on demand to keep its policy interest rate on target. The reserve need is a *plumbing* problem solved after the fact, not a *funding* gate that has to clear before lending. We will come back to this; it is the crux of why "banks lend out reserves" is a myth.

*"Doesn't this mean banks can create infinite money?"* No — and the real limits are the whole subject of a later section. But notice already that the limit is *not* reserves and *not* a vault of savings. Banks are limited by capital, by whether creditworthy borrowers actually want to borrow, and by regulation. Those are the brakes. Reserves are not.

When the loan is repaid, the whole thing runs in reverse: the borrower hands back \$200,000 of deposits, the bank cancels the loan asset and the deposit liability, and that \$200,000 of money *ceases to exist*. Money is created when loans are made and destroyed when they are repaid. This is why the money supply is not a fixed pool being shuffled around — it *breathes*, expanding when credit grows and contracting when credit shrinks. A trader who grasps this has a different relationship with the phrase "the money supply" than someone who pictures a fixed stack of bills.

This breathing is not a curiosity; it is the mechanical reason credit busts are so violent. In a boom, net *new* lending exceeds repayments, so money is created on net and the supply grows. In a bust, the arithmetic flips: borrowers repay and default faster than banks make new loans, so on net money is *destroyed*. The deposits that funded spending simply vanish from the system, line by line, as loans are paid down or written off. That is why a credit contraction is deflationary and self-amplifying — it is not merely that people *choose* to spend less; it is that the very *money* they would have spent is being deleted from existence. The Great Depression and the 2008 crisis both featured this monetary implosion: broad money shrank as credit collapsed, and the shrinking money supply turned a financial shock into an economic one. Keep this picture — money expanding and contracting with the credit cycle — because it is the bridge from a single T-account to the macro cycle, and it is what makes credit data *lead* the economy: the money is being created or destroyed before the spending it funds shows up in GDP.

For the full mechanics of money creation — how reserves are supplied, how the central bank fits in, how money is destroyed on repayment — the companion piece [How Money Is Actually Created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) walks every T-account in detail. Here we take loans-create-deposits as established and push toward the *cycle* and the *trade*.

## The money-multiplier myth versus reality

Now we have to demolish the model most people learned, because it is the source of nearly every wrong intuition about banking, QE, and inflation.

**The money multiplier** is the textbook story. It goes like this. The central bank creates some base money (reserves). Banks are required to hold a fraction — say 10% — of deposits as reserves (the *reserve requirement*). So when \$100 of reserves enters the system, a bank lends out \$90, that \$90 gets deposited somewhere, that bank keeps \$9 and lends \$81, and so on. The geometric series sums to \$100 / 0.10 = \$1,000. The conclusion: a given quantity of reserves "multiplies" into a fixed, larger quantity of broad money, and the central bank controls the money supply by controlling reserves. Reserves come *first*; lending follows; the multiplier is mechanical.

This model is taught everywhere. It is also, as a description of causality, wrong — and central banks themselves say so. Here is why, point by point.

**Causality runs the other way.** Banks do not wait for reserves and then lend a multiple of them. They make loans first — creating deposits, as we just saw — and then acquire whatever reserves they need to settle payments and meet requirements, borrowing them from each other or from the central bank. The Bank of England paper is blunt: *"Rather than banks receiving deposits when households save and then lending them out, bank lending creates deposits."* Lending leads; reserves follow. The multiplier describes a sequence that runs backwards from reality.

**The central bank supplies reserves on demand.** Because the central bank targets an *interest rate*, not a *quantity of reserves*, it must provide whatever reserves the banking system needs to keep that rate on target. If it refused, the overnight interbank rate would spike above target and the central bank would miss its policy goal. So in practice reserves are *elastic*: the system gets the reserves it needs. A quantity that adjusts to demand cannot be the binding constraint on the quantity of lending.

**The reserve requirement is often zero.** In March 2020, the Federal Reserve cut the reserve requirement ratio to **0%**. Banks in the US are now under no obligation to hold reserves against deposits at all. If the multiplier model were how the world worked, a 0% reserve requirement would imply an *infinite* money multiplier and infinite money. That obviously did not happen, because the multiplier was never the mechanism. Lending is constrained by capital and demand, not reserves.

**This is the key distinction: exogenous versus endogenous money.** The multiplier model treats money as **exogenous** — set from outside by the central bank, which dials the quantity up or down. The reality is **endogenous money**: the money supply is determined *from within* the economy, by the demand for credit and banks' willingness and ability to supply it. The central bank influences this not by rationing reserves but by setting the *price* of credit — the interest rate — which raises or lowers loan demand. Money is created by the private decisions of borrowers and lenders, with the central bank steering the price, not metering the quantity. (For why the interest rate is the master lever here, see the companion [Interest Rates: The Price of Money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).)

Why does this matter for a trader? Because the multiplier myth produces a specific, expensive forecasting error: it makes people predict inflation every time central-bank reserves rise. After 2008, the Fed's balance sheet ballooned and reserves went from tens of billions to over a trillion dollars. Multiplier-thinkers — including some very famous investors — predicted runaway inflation and hyperinflation. It did not come. For a decade, inflation ran *below* target. The reason: those reserves sat *in the banking system*. They did not become broad money, because broad money is created by *lending*, and lending was weak (households and banks were deleveraging after a credit bust). Reserves are not loanable funds the public can spend; they are a settlement asset banks hold among themselves. Confusing the two is the single most common and costly macro mistake. We will return to this when we dissect QE.

## What really limits lending: capital, demand, regulation

If reserves do not limit lending, what does? Three things, and a trader should know all three cold, because each one is a signal.

![What limits bank lending corrected to capital demand and regulation not reserves](/imgs/blogs/how-credit-creates-money-lending-channel-cycles-4.png)

**1. Capital.** This is the real, binding, quantitative constraint, so we will spend the most time here. Every loan a bank makes is a *risky asset* — the borrower might default. To absorb potential losses without going bust, regulators require banks to fund a fraction of their risky assets with **capital** (equity) rather than borrowed money. The rule is expressed as a **capital ratio**: capital divided by *risk-weighted assets* (RWA, the bank's assets weighted by how risky each is). Under the Basel international framework, the minimum total capital ratio is around 8% of RWA, and with buffers, real-world banks run higher — often 12–15%. The mechanism: a bank with a fixed amount of capital can only support a limited stack of risk-weighted loans on top of it. To lend more, it must either raise more capital (issue equity, retain earnings) or hold less of something else. Capital, not reserves, is the wall lending runs into.

#### Worked example: 8% capital supports \$125B of loans, not infinite

Suppose a bank has **\$10 billion of capital** (equity) and faces a minimum total capital ratio of **8%** on risk-weighted assets. How big a loan book can it support?

The ratio says: capital ÷ risk-weighted assets ≥ 8%. Rearranging for the maximum risk-weighted assets:

```
max RWA = capital / capital_ratio
        = $10B / 0.08
        = $125B
```

So \$10 billion of capital supports roughly **\$125 billion** of risk-weighted loans — a leverage of about 12.5 to 1. Push past that and the bank's capital ratio drops below the minimum, regulators step in, and the bank must stop lending or raise more equity. Notice what is *absent* from this calculation: reserves. Nowhere does the quantity of reserves appear. The binding constraint is the \$10 billion of capital and the 8% rule. If the bank wants to make another \$10 billion of mortgages (risk-weighted, say, at 50%, so \$5B of RWA), it needs \$5B × 8% = \$400 million of additional capital to back them. The intuition: **a bank's lending capacity is set by its capital and the rules on top of it, and to grow credit the system has to grow capital — which is exactly why credit booms eventually run out of room.**

This is also why bank *profitability* matters for the credit cycle: retained earnings are the cheapest way to build capital, so when banks are profitable they can expand lending, and when they are taking losses (loan defaults eat into capital) they must *shrink* lending to restore their ratios — which is precisely how a bust feeds on itself.

**2. Loan demand.** A bank cannot create money by lending unless someone *wants to borrow* and is *creditworthy enough to lend to*. This sounds obvious but it is the constraint that the multiplier model completely ignores, and it is the one that dominates in downturns. After a credit bust, households and firms are over-indebted and want to *pay down* debt, not take on more. Banks can be flush with capital and reserves and willing to lend, and credit still does not grow, because nobody wants the loan. Economists call the extreme version a *balance-sheet recession* (the term is Richard Koo's): the entire private sector is repaying debt at once, so credit — and therefore money — *contracts* no matter how easy the central bank makes it. This is why central banks can be "pushing on a string": cutting rates to zero does nothing if loan demand is dead. For a trader, loan *demand* is often the more important variable than loan *supply*, and it is why the credit impulse (a flow measure) beats the level of rates as a signal.

**3. Regulation.** Beyond capital ratios, a thicket of rules caps how fast and how much banks can lend: liquidity requirements (the Liquidity Coverage Ratio, which forces banks to hold enough liquid assets to survive a 30-day stress), leverage caps (a simple non-risk-weighted limit), supervisory stress tests, and in some countries outright **credit ceilings** — hard quantitative limits on loan growth set by the central bank. Vietnam, for instance, runs an explicit annual credit-growth ceiling as a primary policy tool; see [Vietnam's Monetary Policy and Credit Ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling). When a regulator tightens any of these — raises capital requirements, tightens liquidity rules, lowers a credit ceiling — credit growth slows even if demand and capital are ample. Regulatory shifts are a slow-moving but powerful brake, and a trader should track them as part of the credit picture.

So: reserves do not limit lending; capital, demand, and regulation do. When you see a headline like "banks are flush with reserves, so lending will boom," you now know it is a non-sequitur. The right questions are: Is bank *capital* growing or shrinking? Do creditworthy borrowers *want* to borrow? Are regulators tightening or loosening? Those three determine whether money — credit — expands or contracts. And that expansion-or-contraction is the credit cycle, which is where we turn next.

## The credit cycle: the engine behind every boom and bust

Here is the payoff of everything so far. If credit creation *is* money creation, and money creation drives spending and asset prices, then the *credit cycle* — the rhythm of credit expanding and contracting — is the master cycle that the business cycle, the asset-price cycle, and the inflation cycle all dance to. Most of what looks like separate phenomena (a housing boom, a stock bubble, a recession, a banking crisis) are phases of one underlying credit cycle. Understanding its mechanics is the difference between reacting to prices and anticipating them.

The credit cycle is **self-reinforcing** in both directions, and that feedback is what makes it powerful and dangerous. Walk the loop on the way up:

![Credit cycle loop from easy lending to rising prices to more collateral to bust](/imgs/blogs/how-credit-creates-money-lending-channel-cycles-2.png)

1. **Easy lending.** Banks loosen standards — lower rates, smaller down payments, looser income checks. Credit flows freely. Remember: each new loan *creates new money*, so the money supply is expanding as fast as credit.

2. **Asset prices rise.** That new money chases a roughly fixed stock of assets — houses, stocks, land. People borrow to buy them. Prices go up. This is the crucial link the textbook misses: credit creation is not neutral; the new money flows disproportionately into *assets*, inflating them.

3. **Collateral is worth more.** Most lending is collateralized — the house secures the mortgage, the stock secures the margin loan. When asset prices rise, the collateral backing existing loans is worth more, and new loans against that richer collateral *look safer*. A \$500,000 house that rose to \$700,000 can now "safely" support a bigger loan.

4. **Even more lending.** Richer collateral and rising confidence justify still more credit. Borrowers take out home-equity loans against their gains; investors lever up against appreciated portfolios. Leverage climbs across the system. More credit means still more new money, which means step 2 again, *amplified*.

5. **Overextension.** The loop runs hot. Debt outruns income. Marginal, less-creditworthy buyers get pulled in at the top (the "greater fool" stage — subprime mortgages in 2006, leveraged everything in 2021). The system is now fragile: it depends on prices continuing to rise to keep collateral valuable and borrowers solvent.

6. **The bust.** Something stops the rise — rates rise, a default cluster appears, sentiment cracks. Prices stop going up. Now the loop runs *violently in reverse*: falling prices shrink collateral, shrinking collateral forces banks to call loans and tighten standards, tighter credit means less new money, less money means lower spending and lower prices — which shrinks collateral further. Loans default, banks take losses, those losses eat capital, and (per the capital constraint above) shrinking capital forces banks to lend *even less*. The same feedback that inflated the boom now deflates it, often faster than it inflated, because fear moves quicker than greed and because forced selling and margin calls have no patience.

The economist Hyman Minsky built a whole theory around this: stability breeds instability, because a long calm period encourages ever-riskier lending (from "hedge" borrowers who can repay principal and interest, to "speculative" borrowers who can only service interest, to "Ponzi" borrowers who need rising asset prices just to roll the debt) until the structure collapses under its own leverage. The moment the bust begins is sometimes called a "Minsky moment." 2008 was the canonical one: a decade of easy mortgage credit inflated US housing, collateral and lending fed each other, the marginal subprime borrower was pulled in at the top, prices stalled in 2006, and the whole leveraged structure imploded into the Global Financial Crisis. 2020–21 was a compressed, stimulus-fueled version that ended not in a banking crisis but in the 2022 inflation and rate shock.

For a trader, the lesson is directional and timing-based: **you do not have to call the exact top.** You have to know *which phase of the credit cycle you are in*, because the phase tells you the regime — whether to be long risk and leverage (early-to-mid expansion, credit accelerating, standards easing) or to be defensive and short fragility (late cycle, credit decelerating, standards tightening, spreads widening). The phase is *observable* in credit data before it is obvious in prices. That observability is the edge, and the credit impulse is the sharpest tool for reading it.

## The credit impulse: a leading indicator you can actually use

Now we get to the single most useful concept in this entire post for a working trader. It is subtle, it is underused by the crowd, and it leads markets. It is the **credit impulse**.

Start with the distinction between a *level*, a *flow*, and a *change in the flow* — because the credit impulse is the third, and confusing them is why most people misread credit data.

- The **stock** of credit is the *level* — total debt outstanding, say \$50 trillion. It moves slowly and tells you little about the margin.
- The **flow** of credit is *new credit per period* — how much *new* borrowing happened this year, the *change* in the stock. This is closer to what drives spending: it is the new money being created.
- The **credit impulse** is the **change in the flow** — this year's flow of new credit minus last year's flow, usually scaled by GDP. It is, in calculus terms, the *second derivative* of the credit stock: the acceleration of credit, not its level or even its growth.

Why the *second* derivative? Because GDP growth is itself a flow (spending per period), and a large body of empirical work (the concept was popularized by Michael Biggs and colleagues around 2008) shows that the *change* in GDP tracks the *change in the flow* of credit — the impulse — far better than it tracks the *level* of credit or even credit growth. In plain terms: the economy responds to whether credit creation is *accelerating or decelerating*, not to how much total debt exists. A heavily indebted economy where credit is *re-accelerating* will grow; a low-debt economy where credit is *decelerating* will slow. The impulse, not the level, is what moves the margin.

And here is the trader's gold: **the credit impulse turns before growth, and growth turns before equities and risk assets.** The change in the flow of credit shows up first — because credit *funds* the spending and investment that later become GDP and corporate earnings. So a turn in the credit impulse typically leads economic growth by **two to three quarters**, and leads equity and credit markets by a similar margin. You see the engine change speed before the car visibly speeds up or slows down.

![Credit impulse leads growth then equities by two to three quarters timeline](/imgs/blogs/how-credit-creates-money-lending-channel-cycles-6.png)

China is the cleanest laboratory for this, because Chinese credit (measured by Total Social Financing, TSF) is large, policy-driven, and turns sharply when Beijing decides to stimulate or restrain. The Chinese credit impulse has repeatedly led not just Chinese growth but *global* manufacturing, commodity prices, and cyclical equities by several quarters: it surged in 2009, 2012, 2016, and 2020 ahead of global reflations, and rolled over ahead of the 2011, 2014, and 2018 slowdowns. Traders who watch the Chinese credit impulse get a multi-quarter heads-up on the global cycle that the GDP data only confirms much later.

It is worth being precise about *why* the second derivative, rather than the level or even the first derivative, is what leads. Spending out of credit is, mechanically, the *flow* of new credit: the money created by this period's lending is what gets spent this period. GDP is itself a flow of spending. So the *level* of GDP tracks the *flow* of new credit. Take the derivative of both sides: the *change* in GDP — i.e. GDP growth, the thing markets actually price — tracks the *change* in the flow of new credit, which is the credit impulse. That is the entire mathematical content of the idea, and it explains the empirical lead: by the time GDP growth turns, the impulse that drove it turned a couple of quarters earlier, when the flow of new lending first accelerated or decelerated. The trader's job is to watch the flow's acceleration directly instead of waiting for the growth it later produces. A practical refinement: because the relationship is a *change* relationship, the impulse is inherently noisy quarter-to-quarter, so smooth it (a rolling few-quarter average) and watch the *trend* and the *sign change*, not every wiggle. A clean sign flip from negative to positive in a smoothed credit impulse is one of the more reliable "the cycle is turning up" signals available, and it typically arrives before any equity index or PMI confirms it.

#### Worked example: computing a credit impulse and reading it

Let me show the arithmetic, because it is simpler than it sounds and seeing it once makes the concept stick. Suppose an economy has nominal GDP of **\$1,000** (use any unit; this scales). Track the *stock* of total credit at three year-ends:

```
Year 0 end:  credit stock = $2,000
Year 1 end:  credit stock = $2,200
Year 2 end:  credit stock = $2,260
```

Now compute the **flow** of new credit each year (the change in the stock):

```
Year 1 flow of new credit = $2,200 - $2,000 = $200
Year 2 flow of new credit = $2,260 - $2,200 = $60
```

Now the **credit impulse** — the change in the flow, scaled by GDP:

```
credit impulse (Year 2) = (this year's flow - last year's flow) / GDP
                        = ($60 - $200) / $1,000
                        = -$140 / $1,000
                        = -14% of GDP
```

Look at what just happened. The *stock* of credit **rose** in Year 2 (from \$2,200 to \$2,260 — debt is still growing). A level-watcher sees "credit is up, all good." But the *flow* of new credit **collapsed** — from \$200 to \$60 — and so the credit impulse is a brutal **−14% of GDP**. That negative impulse predicts a sharp *slowdown* in growth two to three quarters out, even though total debt never fell. The intuition: **growth is driven by the acceleration of credit, so a credit slowdown can tank the economy even while the debt pile keeps growing — which is exactly the trap that catches level-watchers at every cycle top.**

This is why the credit impulse is so powerful as a signal: it can flash a warning while every headline number (debt level, even credit growth) still looks fine. The turn is in the *acceleration*, and the acceleration leads.

## Money growth as a real-world proxy for the credit surge

In an ideal world you would track a clean bank-credit series and compute the impulse directly. In practice, a fast and surprisingly good proxy for the whole credit-and-money story is **broad money growth** — M2 — because, as we established, the bulk of broad money *is* deposits created by lending (plus, in 2020, deposits created when the government deficit-spent and the Fed bought the bonds). When credit and money creation surge, M2 surges; when credit contracts, M2 contracts. The 2020–21 episode is the textbook case, and the data is stark.

![US M2 broad money year over year growth bar chart with 2020 boom and 2023 contraction](/imgs/blogs/how-credit-creates-money-lending-channel-cycles-3.png)

The chart above computes year-over-year M2 growth from year-end levels. The story it tells:

#### Worked example: the +24% M2 surge of 2020 and its echo

Take the real year-end M2 levels (from `data.M2_YEAREND`, sourced from FRED's M2SL series):

```
2019 year-end M2 = $15.40 trillion
2020 year-end M2 = $19.13 trillion
2021 year-end M2 = $21.49 trillion
2022 year-end M2 = $21.36 trillion
2023 year-end M2 = $20.87 trillion
```

Compute the year-over-year growth rates:

```
2020 growth = (19.13 - 15.40) / 15.40 = +24.2%
2021 growth = (21.49 - 19.13) / 19.13 = +12.3%
2022 growth = (21.36 - 21.49) / 21.49 = -0.6%
2023 growth = (20.87 - 21.36) / 21.36 = -2.3%
```

The **+24.2%** print for 2020 is the fastest single-year broad-money growth in postwar US history — a flood of newly created deposits from emergency lending, deficit spending, and QE all at once. Then look at the other end: 2023 M2 *shrank* by **−2.3%**, the first sustained contraction in broad money since the 1930s, as the Fed hiked rates aggressively, credit creation slowed, and the prior surge unwound. The intuition: **money and credit are the same surge seen from two angles, and a +24% money explosion followed by a −2% contraction is the entire 2020–23 macro regime — boom then bust — written in a single series.**

That money surge had a consequence the credit cycle predicts: inflation. A flood of new money chasing a constrained supply of goods is the textbook setup, and it played out on a lag.

![US CPI inflation year over year peaking at 9.1% after the 2020 money boom](/imgs/blogs/how-credit-creates-money-lending-channel-cycles-5.png)

The CPI chart shows inflation accelerating through 2021 and peaking at **9.1%** in June 2022 (`data.CPI_PEAK`), a forty-year high — roughly eighteen months *after* the money boom began. The lead-lag is exactly what the credit-and-money framework predicts: the money surge comes first, the inflation comes later, and the trader who watched M2 explode in 2020 had a year-and-a-half head start on the inflation trade (short duration, long inflation breakevens, long commodities) that the CPI prints only confirmed in 2021–22. Money and credit lead; prices follow.

A caveat worth stating plainly: M2 is a *proxy*, not a perfect credit measure. The 2020 M2 surge was driven as much by fiscal deficits monetized via QE as by private bank lending, and the relationship between M2 and inflation is loose and variable (it broke down for the decade after 2008, when M2 grew but inflation stayed low because the *velocity* of money — how fast it circulates — collapsed). M2 is a useful, fast, free signal, but it is one input, not an oracle. We will fold it into a fuller dashboard at the end.

## Common misconceptions

The credit-money mechanism breeds a specific set of confident, wrong beliefs. Each one has cost traders real money. Here are the five worth inoculating against, each corrected with a number.

**Misconception 1: "Banks lend out their reserves (or your deposits)."** No. As the T-account walkthrough showed, a bank creates a *new* deposit when it lends; it does not lend out existing reserves or existing deposits. Reserves are a *settlement* asset used to move money *between* banks after a loan is spent, sourced on demand. The proof by number: the US reserve requirement has been **0%** since March 2020, yet banks lend trillions — impossible if lending required reserves. Lending is constrained by capital, not reserves.

**Misconception 2: "The money multiplier is a mechanical, fixed relationship."** No. The multiplier is not a control knob; it is, at best, an *ex-post ratio* (broad money divided by base money) that wanders all over the place. After 2008, US base money rose roughly fivefold while broad money grew only modestly — the "multiplier" *collapsed*, because banks held the new reserves rather than lending them. A relationship that can fall by 80% is not a mechanical multiplier; it is the residue of independent lending decisions. Causality runs from lending to reserves, not reserves to lending.

**Misconception 3: "QE directly creates broad money and must cause inflation."** This is the costliest myth, so handle it carefully. Quantitative easing — the central bank buying bonds — creates *reserves* (base money), not broad money *directly*. When the Fed buys a bond from a bank, it credits the bank's reserve account; reserves rise, but the public's deposits don't automatically. QE *can* feed broad money indirectly (it lowers yields, lifts asset prices, eases financial conditions, and may encourage lending) and when the Fed buys bonds from *non-banks* it does create deposits — but it is not a direct money-printing-into-your-pocket mechanism. The proof by number: from 2008 to 2014 the Fed did multiple rounds of QE, base money exploded, and inflation ran *below* its 2% target for most of the period. The hyperinflation predicted by QE-equals-printing never came, because broad money is created by *lending*, and lending was weak. (2020–21 was different because QE coincided with *enormous fiscal transfers* that put money directly in households' hands *and* a lending rebound — money creation through several channels at once. See [Quantitative Easing Explained](/blog/trading/finance/quantitative-easing-explained-printing-money) for the full mechanism.) The lesson: QE alone is not inflationary; QE plus credit growth plus fiscal transfers is.

**Misconception 4: "More central-bank money always means more inflation."** No — it depends on whether that money becomes *spending*. The missing variable is **velocity** (how often each dollar is spent per year) and whether the new money reaches spenders. Reserves piled in the banking system have near-zero velocity. The 2008–2019 period proves it: base money up massively, velocity down, inflation muted. The quantity-theory identity makes this precise: money times velocity equals prices times real output (MV = PY). If money (M) jumps but velocity (V) falls by roughly as much, the product MV — total spending — barely moves, so prices barely move. That is exactly what happened post-2008: M2 velocity fell from about 2.0 in 2007 to roughly 1.1 by 2020, offsetting much of the money growth. Inflation needs new money *in the hands of spenders*, which is why the 2020 fiscal transfers (checks straight to households, which get spent immediately, raising V) were inflationary in a way that a decade of reserves-piled-in-banks QE was not. The trading lesson: never trade a money surge as automatically inflationary — check whether velocity is rising (money reaching spenders) or falling (money stuck in the financial system).

**Misconception 5: "Rising total debt always means more growth and inflation ahead."** No — what matters is the *impulse* (the change in the flow), not the level. As the credit-impulse worked example showed, the debt *stock* can keep rising while the *flow* of new credit collapses, producing a negative impulse and a coming slowdown. Level-watchers see growing debt and stay bullish into the top; impulse-watchers see the flow rolling over and de-risk early. The number to remember: the credit impulse can go from +5% to −10% of GDP while total debt never falls a single dollar.

## How it shows up in real markets

Theory is cheap. Here is how the credit-money mechanism has driven real markets, with dates and numbers, so the framework earns its keep.

**2003–2007: the US housing credit boom.** Easy mortgage credit — subprime lending, low teaser rates, securitization that let banks offload risk and lend more — drove the classic credit cycle. US home prices roughly doubled from 2000 to 2006. The loop ran exactly as drawn: cheap credit lifted house prices, rising prices made mortgages look safe (rising collateral), safer-looking mortgages justified more and looser lending, leverage climbed across the financial system (banks, shadow banks, households). The marginal subprime borrower was pulled in at the top in 2005–06. When prices stalled in 2006 and defaults clustered in 2007, the loop reversed: falling collateral forced deleveraging, deleveraging crushed credit, the credit contraction became the Global Financial Crisis of 2008. A trader watching mortgage credit growth and lending standards (the SLOOS, below) saw the deterioration in 2006–07, well before the equity market peaked in October 2007. Much of the leverage hid in the *shadow* banking system — repo, money-market funds, securitization vehicles — which is its own essential topic; see [Shadow Banking and the Repo Market](/blog/trading/finance/shadow-banking-and-the-repo-market).

**2020–2021: the stimulus credit-and-money boom.** As the charts above show, M2 grew **+24%** in 2020. New money from emergency lending, deficit spending, and QE flooded the system. Asset prices did what the credit cycle predicts: the S&P 500 roughly doubled off its March 2020 low into late 2021; house prices rose ~40% from early 2020 to mid-2022; crypto, SPACs, and speculative tech went parabolic — every asset that could absorb new money inflated. Then the money surge produced inflation on a lag, peaking at **9.1%** in mid-2022, forcing the Fed to hike from 0% to over 5% (the fastest hiking cycle in 40 years), which slammed the credit impulse negative, contracted M2 (**−2.3%** in 2023), and burst the speculative excess (crypto crashed, unprofitable tech fell 70–80%, SPACs collapsed). The whole 2020–23 arc — boom, inflation, bust — is one credit cycle compressed into three years and visible the entire time in the money data.

**China's credit pulses, 2009–present.** China runs its economy substantially through directed credit, so its credit impulse is policy-driven and sharp, and it has repeatedly led the *global* cycle. The 2009 credit explosion (a ~4 trillion yuan stimulus, credit impulse spiking) pulled the world out of the GFC and sent commodities and emerging markets soaring. The 2016 credit pulse drove the 2016–17 global reflation and the commodity rally. When China *restrains* credit (2011, 2014, 2018, 2021's property crackdown), the impulse rolls over and global cyclicals, commodities, and EM weaken several quarters later. Traders watch China's Total Social Financing and its derived credit impulse as a leading indicator for global growth-sensitive assets — copper, miners, EM equities, the Australian dollar.

The common thread across all three: **the credit signal led the price.** Mortgage credit deteriorated before equities peaked in 2007. M2 exploded before inflation in 2021. China's credit impulse turned before global cyclicals moved. In every case, the trader watching credit had a multi-quarter information edge over the trader watching prices.

## How to trade it: the credit playbook

Everything above converges on a practical question: *what do you actually watch, and how do you position?* Here is the playbook — the signals, the regime read, the position, and the invalidation. The organizing dashboard:

![Trader credit dashboard tracking bank credit growth lending standards and spreads](/imgs/blogs/how-credit-creates-money-lending-channel-cycles-7.png)

**The signals to track, in priority order:**

1. **Bank credit growth.** The level and rate-of-change of total bank credit and commercial-and-industrial (C&I) loans. In the US, FRED series like Total Bank Credit (TOTBKCR) and C&I loans give you this. *Rising and accelerating* = expansion, risk-on. *Decelerating or rolling over* = late cycle, start de-risking. (Caveat: a clean bank-credit series was not available in this post's dataset — I used M2 growth as the proxy; in live trading, pull TOTBKCR and the loan-growth series directly.)

2. **The credit impulse.** The change in the flow of new credit, scaled by GDP — the second derivative. Compute it for the US, and especially for **China** (from Total Social Financing), where it is largest and most policy-driven and leads the global cycle. This is your earliest signal: it turns two to three quarters before growth and equities. A rising impulse is a green light for cyclicals and risk; a falling impulse is your earliest warning to reduce leverage and rotate defensive.

3. **SLOOS — the Senior Loan Officer Opinion Survey.** Every quarter the Fed surveys banks on whether they are *tightening or easing* lending standards and seeing stronger or weaker loan demand. This is gold because it is a direct read on the credit *supply* and *demand* that drive money creation, and it leads. *Net tightening* of standards has preceded every US recession in the survey's history; when the net percentage of banks tightening spikes, credit (and the credit impulse) will slow and a downturn risk rises 2–4 quarters out. Watch the C&I and CRE (commercial real estate) standards series specifically.

4. **Credit spreads.** The extra yield investment-grade and high-yield bonds pay over Treasuries. *Tight and stable* spreads = easy credit, complacency, late cycle. *Widening* spreads = credit stress, the bust phase loading. Spreads are a fast, market-priced, real-time read on credit conditions — they widen before equities fall in a credit-led downturn. The high-yield (junk) spread is the most sensitive.

5. **Broad money (M2) growth.** The fast, free proxy when granular credit data is thin (as in this post). A money surge (like 2020's +24%) flags a credit-and-stimulus boom and a coming inflation/asset-price impulse; a money contraction (like 2023's −2.3%) flags a credit bust and disinflation. Use it as a confirming input, not a standalone signal.

6. **Defaults and delinquencies.** Bankruptcy filings, loan charge-off rates, delinquency rates. These *lag* — they confirm the bust is underway rather than predict it — so use them to confirm a turn, not to anticipate one. Rising delinquencies on credit cards and auto loans are an early-*ish* read on household stress.

**How to position by regime.** Read the signals *together* — that is the whole skill — and they place you on the credit cycle:

- **Early-to-mid expansion** (credit accelerating, impulse rising, SLOOS easing, spreads tight and stable, M2 growing): *risk-on*. Be long equities, especially cyclicals and credit-sensitive sectors (banks, homebuilders, industrials); be long high-yield credit; lean into leverage; favor growth-sensitive currencies and commodities. The money is being created and it is flowing into assets — ride it.

- **Late cycle** (credit decelerating, impulse rolling over, SLOOS starting to tighten, spreads still tight but no longer falling, M2 growth slowing): *reduce risk early*. Trim leverage, rotate from cyclicals to quality and defensives, take profits in the most speculative positions, start buying downside protection while it is cheap. This is where the credit signal earns its keep: you de-risk *before* prices roll over, paying a small opportunity cost to avoid the bust.

- **Bust / contraction** (credit contracting, impulse deeply negative, SLOOS sharply tightening, spreads widening, M2 shrinking, defaults rising): *defensive and short fragility*. Long duration (Treasuries rally as growth and inflation fall and the central bank cuts), long the dollar (a credit bust is dollar-positive as global funding tightens), underweight or short equities and high-yield credit, avoid leverage. Wait for the impulse to turn back up — the earliest signal of the next expansion — before re-risking.

**The trade, distilled.** Go risk-on when the credit impulse is *rising*, lending standards are *easing*, and spreads are *tight*. Go defensive when the impulse is *rolling over*, standards are *tightening*, and spreads are *widening*. Position on the lead (credit), not the lag (prices and the data).

**The invalidation — what tells you you are wrong.** This is essential, because every framework fails sometimes and you must know when to abandon it. Stand down or reverse the credit read if: (1) the central bank is actively and aggressively offsetting the credit signal (massive QE or rate cuts can keep asset prices up even as private credit weakens — the "central bank put" can overwhelm a negative impulse for a while, as in 2019); (2) the relationship between money/credit and inflation has clearly broken down via *velocity* (M2 surging but velocity collapsing, as in 2009–2019, means the money is not becoming spending — do not blindly trade the money surge as inflationary); (3) a non-credit shock dominates (a pure supply shock, a war, a pandemic can move markets independently of the credit cycle for a stretch). And always size for the chance that the lead-lag is *longer* than usual — the credit impulse can turn many quarters before the market does, and being early is, in trading, indistinguishable from being wrong if you are over-leveraged into the wait. The discipline is to treat the credit signal as a probabilistic *tilt* on your positioning and time horizon, not a precise market-timing trigger: it changes the odds and the regime, sizing and direction follow, but the exact turn date still belongs to price.

The deepest reason this playbook works is the one we started with: **credit is money, money creation drives the cycle, and credit data turns before prices.** Most market participants watch prices and the lagging economic data. You will be watching the engine — the creation and destruction of money through lending — which changes speed before the car does. That informational lead, read through bank credit, the credit impulse, SLOOS, and spreads, is the edge. Track the engine, position on the lead, and let the lagging crowd confirm your view at worse prices.

## Further reading & cross-links

- [How Money Is Actually Created: Banks, Central Banks, and the Money Multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) — the full T-account mechanics of money creation and destruction, in detail.
- [What Money Really Is: Base Money, Broad Money, and What Traders Watch](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders) — the base-versus-broad-money distinction this post leans on.
- [Interest Rates: The Price of Money, the Master Variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — how the central bank steers credit by setting the price, not the quantity.
- [Shadow Banking and the Repo Market](/blog/trading/finance/shadow-banking-and-the-repo-market) — where much of the leverage in a credit boom actually hides.
- [Quantitative Easing Explained: Printing Money?](/blog/trading/finance/quantitative-easing-explained-printing-money) — why QE creates reserves, not broad money directly, and when it is and is not inflationary.
- [Vietnam's Monetary Policy and the Credit Ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling) — a real-world example of regulating credit growth directly.
- The Bank of England, "Money creation in the modern economy," *Quarterly Bulletin* 2014 Q1 — the canonical central-bank statement that loans create deposits.
