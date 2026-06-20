---
title: "Corporate and Commercial Lending: Term Loans, Revolvers, and Syndication"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank lends to companies rather than people: the difference between a term loan and a revolving credit line, how a single huge loan is split across many banks through an arranger and an agent, what covenants really do, and why the relationship matters more than the loan."
tags: ["banking", "corporate-lending", "commercial-lending", "term-loan", "revolver", "syndicated-loan", "covenants", "leveraged-loan", "relationship-banking", "credit"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Lending to a company is not lending to a person at a larger scale; it is a structured, negotiated, often shared transaction in which the loan itself is frequently the smallest part of the relationship.
>
> - A **term loan** pays out one lump sum the company repays over years; a **revolver** (revolving credit facility) is a credit line the company can draw, repay, and redraw at will, and the bank charges a **commitment fee** on the unused part for keeping the money available.
> - A loan too big for one bank to hold safely is **syndicated**: a lead **arranger** structures it, sells slices to **participant** banks, and an **agent** bank administers it. No single bank ends up holding more than it can afford to lose.
> - **Covenants** are the promises in the contract that let the bank act before a default becomes a loss. **Maintenance** covenants are tested every quarter (e.g. Debt/EBITDA must stay below 4.0x); **incurrence** covenants are only tested when the company does something new (borrows more, pays a dividend).
> - The one number to remember: corporate loans are priced as a **benchmark plus a spread** — a strong company might borrow at the benchmark plus 1.5%, a leveraged borrower at the benchmark plus 4.5% or more, because each extra turn of leverage raises default risk faster than the last.

In the spring of 2007, before anyone was using the word "crisis", a private-equity firm agreed to buy a large, boring industrial company. The cheque was enormous — call it several billion dollars — and most of it would be borrowed. No single bank wanted to lend the whole sum; the risk of one company failing and taking a chunk out of one balance sheet was too concentrated. So one bank stepped forward as the *lead arranger*. It structured the debt, signed a commitment to fund the entire amount itself if it had to, and then quietly spent the next few weeks on the phone selling pieces of that loan to twenty other banks and a dozen institutional funds. By the time the deal closed, the original arranger held only a sliver of what it had committed. The loan that looked like one bank lending billions was actually thirty lenders each holding a manageable slice, stitched together by a single contract and run by one administrative agent.

That is corporate lending, and it looks almost nothing like the mortgage or the car loan most people picture when they hear the word "loan". A person borrows a fixed sum and pays it back on a schedule. A company borrows in *facilities* — flexible, layered, renegotiable instruments — and the bank that lends rarely keeps the whole thing. The loan is a starting point, not an endpoint: it opens a relationship through which the bank hopes to sell foreign-exchange hedging, cash management, bond underwriting, and advice on the next acquisition. The interest on the loan is often the least profitable thing in the whole arrangement.

This post is about how banks actually lend to companies. We will build up the three core shapes of corporate credit — the term loan, the revolver, and the syndicated loan — define every term from zero, and then go deep on the two things that make corporate lending its own discipline: **syndication** (how banks share one loan so no one of them is endangered by it) and **covenants** (the contractual tripwires that let a lender act early). The figure above is the mental model to keep in your head the whole way through: one borrower, one arranger who structures and sells the deal, many banks each funding a slice, and one agent who runs it.

![Diagram of a syndicated loan from borrower to lead arranger to participant banks to agent to funded loan](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-1.png)

## Foundations: term loans, revolvers, syndicates, covenants, and the relationship

Corporate lending has its own vocabulary, and almost every word means something more specific than its everyday sense. Let us define the building blocks from absolute zero before we go anywhere.

### A loan facility, not a loan

When a bank lends to a company, it rarely makes a single "loan". It grants a **facility** — a standing agreement under which the company may borrow on agreed terms. A facility has a *commitment* (the maximum the bank will lend), a *tenor* (how long the facility lasts), a *price* (the interest rate and fees), and a set of *conditions* (the covenants). A company typically has several facilities at once: a term loan to fund a long-term asset, a revolver for day-to-day liquidity, perhaps a separate facility for trade finance. Think of a facility as a pre-agreed permission to borrow, not the borrowing itself.

### Term loan

A **term loan** is the closest cousin to a personal loan. The bank advances a lump sum at the start (the *drawdown*), and the company repays it over a fixed *term* — say five or seven years — either in instalments (an *amortising* term loan) or in one payment at the end (a *bullet* loan). Once repaid, the money is gone; the company cannot redraw it. Term loans fund things that have a long life and a clear cost: buying a factory, refinancing old debt, paying for an acquisition. The defining feature is **certainty of amount and schedule** — both sides know exactly how much was borrowed and roughly when it comes back.

### Revolving credit facility (the revolver)

A **revolver** is the opposite of certainty: it is a flexible credit line the company can draw down, repay, and draw again, as many times as it likes, up to the commitment, for the life of the facility. It works like a corporate credit card with a very large limit and a much lower interest rate. A company uses a revolver for *working capital* — the cash that sloshes in and out as it pays suppliers before customers pay it — and as a *liquidity backstop* it can tap in an emergency. Because the bank must keep the full commitment available even when the company has not drawn it, the company pays a **commitment fee** (also called a facility fee) on the *undrawn* portion: a small percentage, often 0.25% to 0.50% a year, for the privilege of being able to borrow on demand.

### Syndicate, arranger, participant, and agent

When a loan is too large for one bank to hold prudently, several banks lend together. The group is a **syndicate**, and the loan is a **syndicated loan**. Three roles matter:

- The **lead arranger** (sometimes the *mandated lead arranger*, or MLA) wins the mandate from the borrower, structures the deal, sets the terms, and sells slices of the loan to other banks. The arranger usually *underwrites* the deal — commits to fund the whole amount itself if it cannot sell enough — which is why arranging is a skill, and a risk, in its own right.
- A **participant** (or syndicate member) is any bank that takes a slice of the loan. Each participant lends only its committed share and is exposed only to that share.
- The **agent** (the *administrative agent* or *facility agent*) is one bank that runs the loan after closing: it collects interest from the borrower and distributes it to the lenders pro rata, monitors covenant compliance, and coordinates the syndicate if anything goes wrong. The agent is paid an annual *agency fee* for the plumbing.

The same bank can play several roles — typically the lead arranger also becomes the agent and keeps a slice as a participant — but the roles are distinct, and the contract names who does what.

### Covenant

A **covenant** is a promise the borrower makes in the loan contract. *Affirmative* covenants say what the company must do (deliver audited accounts, pay its taxes, maintain insurance). *Negative* covenants say what it must not do without the lenders' consent (take on more debt beyond a limit, sell major assets, pay large dividends). *Financial* covenants are numeric tests the company must satisfy (keep leverage below a ratio, keep interest coverage above a ratio). Breaching a covenant is an *event of default* — even if the company is paying its interest on time — and it hands the lenders the right to act. We will spend a whole section on the crucial distinction between *maintenance* and *incurrence* covenants.

### Leveraged loan

A **leveraged loan** is a loan to a company that already carries a lot of debt relative to its earnings — typically a borrower rated below investment grade (we will define ratings shortly), often a company owned by a private-equity firm. Leveraged loans pay a much higher spread because the borrower is riskier, and they are the engine of the *leveraged finance* market that funds buyouts. A useful rule of thumb: a loan is "leveraged" when the borrower's total debt is roughly four times its annual operating earnings or more.

### Relationship banking

**Relationship banking** is the model under which a bank lends to a company not mainly to earn interest, but to be the company's primary financial partner — and to earn the fee income that flows from that primacy. The loan is the entry ticket; the profit comes from the cash management, the hedging, the bond and equity underwriting, and the advisory work the relationship unlocks. This is the single most important idea in corporate lending, and it explains pricing decisions that look irrational if you only watch the interest rate.

With the vocabulary in place, here is the comparison that organises everything that follows.

![Comparison matrix of term loan revolver and syndicated loan by purpose draw pricing and holder](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-2.png)

## The term loan: certainty of amount, schedule, and risk

Start with the simplest instrument. A company wants to buy a competitor for \$300 million. It has \$60 million of its own cash and wants to borrow the rest. A bank — or a syndicate — advances a **term loan** of \$240 million, drawn in full at closing. The company repays it over seven years.

The economics of a term loan are clean because everything is fixed up front. The bank knows the principal (\$240 million), the tenor (seven years), and the repayment shape (amortising or bullet). Its job is to price the loan so that the interest it earns covers four things: the bank's own cost of borrowing the money, the *expected loss* if the borrower defaults, the cost of the capital regulators force it to hold against the loan, and a margin of profit. We will return to pricing in detail; for now, hold the idea that a term loan is a bet on one company over a defined horizon.

### Term loan A versus term loan B, and who actually holds them

In the leveraged-loan market, term loans come in two main flavours, and the difference shapes who ends up owning the debt. A **term loan A** (TLA) amortises steadily — the company repays a chunk of principal every quarter — and is typically held by banks, which like the predictable paydown. A **term loan B** (TLB) amortises only a token amount (often 1% a year) with a large bullet at the end, runs longer, and pays a wider spread; it is designed to be sold to *institutional investors* — the loan funds, insurance companies, and especially the **CLOs** (collateralised loan obligations) that package leveraged loans into bonds. The distinction matters because it tells you who bears the risk. A TLA is bank balance-sheet risk; a TLB is mostly sold out of the banking system into the capital markets, which is why a single bank can arrange a multi-billion-dollar TLB while holding very little of it.

#### Worked example: amortising versus bullet repayment

Take that \$240 million loan over seven years at the 9.0% all-in cost from above, and compare two repayment shapes. As an **amortising** term loan A, suppose the company repays 10% of the original principal — \$24 million — each year, with a balloon at the end. In year one it pays \$24 million of principal plus 9.0% interest on the full \$240 million, or \$21.6 million of interest — about \$45.6 million of total cash out. By year seven, the principal owed has shrunk to perhaps \$60 million, so the interest bill that year is only about \$5.4 million. The debt *and* the interest fall together.

As a **bullet** term loan B, the company pays only token amortisation — say 1%, or \$2.4 million a year — and refinances or repays the remaining ~\$223 million at maturity. Every year it pays close to the full \$21.6 million of interest, because the principal barely moves, but it keeps almost all the cash in the business in the meantime. The intuition: amortisation de-risks the lender (the loan shrinks year by year, so the exposure at default falls) while a bullet keeps the borrower's cash flexible but leaves the lender fully exposed until the very end — which is precisely why bullet TLBs pay a wider spread and tend to be sold to investors hungry for that extra yield rather than held by conservative banks.

### Collateral, seniority, and why they change the spread

A term loan can be **secured** — backed by a claim on specific assets (a *lien* over the company's plant, receivables, or even all its assets) — or **unsecured**. Security matters because it drives the *loss given default*. A senior secured loan that recovers, say, 75 cents on the dollar after a default loses far less than an unsecured loan that recovers 45 cents. Recall that the spread compensates for *expected loss*, which is probability of default multiplied by loss given default. Two loans to the same borrower — one senior secured, one subordinated unsecured — share the same default probability but have very different loss-given-default, so the subordinated loan must pay a wider spread. This is why corporate credit is layered into a *capital structure*: senior secured loans at the bottom (lowest risk, lowest spread, paid first in a bankruptcy), then senior unsecured, then subordinated debt, with equity at the very top taking the first loss. Where a loan sits in that stack is half of its risk.

#### Worked example: pricing a term loan over a benchmark

Corporate loans are almost never quoted as a flat rate like "7%". They are quoted as a **benchmark plus a spread**. The benchmark is a published, market-wide cost of short-term money — for U.S. dollar loans today that is **SOFR** (the Secured Overnight Financing Rate), which replaced the discredited LIBOR. The *spread* (also called the *margin*) is the extra rate the bank charges this particular borrower for its particular risk.

Suppose SOFR is **5.0%**. Our acquirer is a solid but leveraged company, so the bank quotes "SOFR plus 450 basis points". A *basis point* is one hundredth of a percent — 0.01% — so 450 basis points is 4.50%. The headline rate is therefore:

$$\text{Rate} = \text{SOFR} + \text{spread} = 5.0\% + 4.5\% = 9.5\%$$

But there is one more wrinkle. Leveraged loans are often sold to investors at a small discount to face value — an **original issue discount** (OID). If the lenders pay \$99 for every \$100 of loan, that \$1 discount is extra yield spread over the life of the loan, worth roughly **0.5%** a year on a multi-year loan. From the borrower's side, an upfront fee or OID lowers the effective ongoing cost slightly relative to the headline; from the lender's side it raises the all-in return. Netting the pieces, the all-in economics settle near:

$$5.0\%\ (\text{SOFR}) + 4.5\%\ (\text{spread}) - 0.5\%\ (\text{OID benefit to borrower}) \approx 9.0\%$$

On \$240 million, a 9.0% all-in cost is about **\$21.6 million** of interest in year one. The intuition: a corporate loan's price is built like a stack — a market benchmark everyone pays, plus a credit spread that is *entirely* about this borrower's risk, adjusted by fees. When you read "SOFR + 450", you are reading the bank's verdict on how likely it thinks the company is to default.

![Waterfall chart showing a leveraged term loan priced as SOFR plus a credit spread minus an upfront fee](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-5.png)

### Why the spread is the whole story

The benchmark is the same for everyone — it is the cost of money in the economy, set by the Federal Reserve's policy and the repo market, and a bank earns nothing by passing it through. The *spread* is where the bank's craft lives. It is the compensation for taking the chance that this company stops paying. To set it, the bank estimates the borrower's **probability of default** (PD) — the chance, usually over one year, that the company fails to meet its obligations — and the **loss given default** (LGD) — the fraction of the loan the bank would not recover after a default, net of collateral.

The relationship between credit quality and default risk is steep and non-linear, and it is worth seeing in numbers. Banks and rating agencies map borrowers onto a ladder of grades from AAA (safest) down to CCC (most likely to default). The chart below shows representative one-year default probabilities by grade. Notice that moving from BBB (the bottom of *investment grade*) to BB (the top of *high yield*, or *leveraged*) roughly quintuples the default rate, and a CCC borrower defaults more than two thousand times as often as a AAA one.

![Bar chart of one-year corporate default probability by credit rating from AAA to CCC](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-3.png)

This is why a leveraged loan pays SOFR + 450 while an investment-grade company might pay SOFR + 100: the spread is the bank pricing that default ladder. It is also why the spread, not the benchmark, is what moves through a credit cycle. When the economy weakens and defaults rise, spreads widen even if the central bank holds the benchmark flat — the market is repricing the risk, not the cost of money.

## The revolver: paying for the option to borrow

The revolver is where corporate lending diverges most sharply from consumer lending, because the bank is selling something subtle: not money, but the *option* to demand money. A company with a \$100 million revolver may draw nothing for months, then suddenly need \$60 million when a customer pays late or a supplier demands cash up front. The bank must keep the full \$100 million available the whole time, which ties up its own funding and its regulatory capital even when nothing is drawn. So the bank charges for availability, not just for use.

#### Worked example: a revolver drawdown and the commitment fee

A company has a \$100 million revolving credit facility. The terms are:

- Interest on **drawn** balances: SOFR + 2.00% (200 basis points). Say SOFR is 5.0%, so drawn money costs 7.0%.
- A **commitment fee** on the **undrawn** balance: 0.375% (37.5 basis points) a year.

In a quiet quarter, the company draws **\$40 million** and leaves \$60 million undrawn. Over that quarter (a quarter of a year):

Interest on the drawn \$40 million:
$$\$40{,}000{,}000 \times 7.0\% \times \tfrac{1}{4} = \$700{,}000$$

Commitment fee on the undrawn \$60 million:
$$\$60{,}000{,}000 \times 0.375\% \times \tfrac{1}{4} = \$56{,}250$$

Total cost for the quarter: **\$756,250**. Now suppose the company draws the full \$100 million for the next quarter and pays no commitment fee (nothing is undrawn):

$$\$100{,}000{,}000 \times 7.0\% \times \tfrac{1}{4} = \$1{,}750{,}000$$

The intuition: the commitment fee is rent on a promise. The company pays \$56,250 a quarter — about \$225,000 a year — purely to keep \$60 million of borrowing power on standby. That fee looks small, but it is almost pure profit for the bank, because the bank earns it without advancing a cent. Across a large corporate book, commitment fees on undrawn revolvers are a quiet, steady, low-risk income stream — and a reason banks compete hard to be a company's revolver provider even when the loan rarely gets used.

There is one more subtlety worth naming. A revolver almost always carries a *utilisation*-sensitive price: the spread and even the commitment fee can step up as the company draws more of the line, because heavy utilisation is itself a warning sign that the company is short of cash. A revolver sitting mostly undrawn is a healthy company keeping a backstop; a revolver pinned near its limit is a company leaning on its bank. Banks watch revolver utilisation across their corporate book as an early-warning indicator of stress — a rising tide of drawdowns across many clients at once is one of the first signs that a credit cycle is turning, because companies reach for their committed liquidity before they admit, even to themselves, that trouble has arrived.

### Why the revolver is the relationship's anchor

A revolver is the facility a company touches most often, and the one it cannot do without, so it almost always goes to the company's lead relationship bank — the bank it trusts to actually fund the draw on a bad day. That gives the bank a privileged position: it sees the company's cash flows in near real time, it is first in line for the next piece of business, and it earns the commitment fee in the meantime. Banks have been known to provide a revolver at a price that barely covers its cost, treating it as a loss leader, precisely because it locks in the relationship. The revolver is rarely where the money is made; it is where the relationship is *kept*.

This is also where the bank's own fragility shows through. A revolver is a *committed* line: the bank has promised to fund it. In a crisis, many companies draw their revolvers at once to hoard cash — exactly what happened in March 2020, when investment-grade firms drew tens of billions of dollars of revolvers in a few weeks as the pandemic froze markets. The bank that sold the option to borrow must now honour it, funding huge unexpected outflows precisely when its own funding is hardest to come by. The revolver, in other words, is a maturity-transformation trap of its own: the bank has written a liquidity insurance policy that gets claimed exactly when liquidity is scarcest.

## Syndication: how banks share one loan so none of them dies from it

Now to the heart of corporate lending. A bank's whole life, as this series keeps insisting, is a leveraged, confidence-funded balancing act: it lends long with money it borrows short, and its thin equity cushion must absorb losses faster than they arrive. A single \$2 billion loan to one company is a direct threat to that cushion — if the company fails and the recovery is poor, the loss could be a meaningful fraction of the bank's entire equity. The answer is not to refuse the loan; the answer is to *share* it.

**Syndication** is the machinery for sharing. The process runs like this. A borrower needs a large loan and gives one or more banks the *mandate* to arrange it. The lead arranger negotiates the terms with the borrower and writes the *information memorandum* describing the deal. Then it markets the loan to other banks and institutional investors, who each agree to take a slice — say \$100 million here, \$50 million there — until the whole amount is committed. The arranger often *underwrites* the deal first, guaranteeing the borrower the full amount, then sells down its position; if it cannot sell enough, it is left holding more than it wanted (this is *syndication risk*, and it bit hard in 2007 when arrangers were stuck with billions of unsellable buyout debt as markets seized). After closing, one bank acts as the **agent**, collecting and distributing cash and minding the covenants.

#### Worked example: splitting a syndicated loan across banks

A company needs **\$500 million**. No bank wants more than about \$150 million of exposure to a single name. The lead arranger structures a five-bank syndicate. Suppose the allocations come out as:

| Bank | Commitment | Share of loan |
|---|---|---|
| Lead bank | \$150 million | 30% |
| Bank A | \$150 million | 30% |
| Bank B | \$120 million | 24% |
| Bank C | \$80 million | 16% |
| **Total** | **\$500 million** | **100%** |

The borrower signs *one* loan agreement and deals with *one* agent. But behind that single contract, the \$500 million risk is split four ways. If the company defaults and the recovery is, say, 60 cents on the dollar, the \$200 million loss is shared pro rata: the lead bank and Bank A each absorb \$60 million, Bank B absorbs \$48 million, Bank C absorbs \$32 million. No single bank takes a \$200 million hit.

Now do the arithmetic that makes syndication worth the effort. Imagine a bank with \$8 billion of equity (recall this series' base case of about 8% equity, roughly 12.5x leverage). A \$200 million loss is 2.5% of that bank's entire equity from one borrower. A \$60 million loss — the syndicated slice — is 0.75%. The intuition: syndication converts a loan that could dent a bank's capital into one that merely nicks it, and it lets a bank serve a giant client it could never bankroll alone. The borrower gets size; every lender gets diversification.

![Horizontal bar chart splitting a 500 million dollar syndicated loan across the lead bank and three participant banks](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-9.png)

### Bilateral, club, and syndicated: three sizes of deal

Not every corporate loan is a sprawling syndicate. The market has three sizes, and which one a borrower gets says a lot about its size and the deal's economics. A **bilateral** loan is one bank lending to one company — the simplest arrangement, used for smaller companies and smaller facilities, where one bank can comfortably hold the whole exposure. A **club deal** is a small group of banks — typically three to six relationship banks — that each take a roughly equal slice, arranged informally without a full marketing process; club deals suit mid-sized borrowers who want to spread the loan across their core banks without the cost and disclosure of a broad syndication. A full **syndicated** loan is the broad-distribution version we have been describing, with a lead arranger marketing the deal to many lenders, used for the largest facilities. As a borrower grows, it tends to graduate from bilateral to club to syndicated — and the loan documentation, the fee structure, and the number of relationships it must manage grow with it. The mechanics of sharing risk are the same in all three; only the number of lenders and the formality of the process change.

### Underwritten versus best-efforts, and why arranging is a real risk

There are two ways to syndicate. In a **best-efforts** syndication, the arranger promises only to *try* to raise the money; if it falls short, the borrower gets less than it wanted. In an **underwritten** syndication, the arranger guarantees the full amount and takes the risk of selling it down later. Underwriting pays a fatter fee precisely because it carries *syndication risk*: if the market turns between the commitment and the sell-down, the arranger is stuck holding the bag.

This is not a hypothetical. In the second half of 2007, banks had underwritten a wave of leveraged buyout loans on aggressive terms. When the credit market froze, investors refused to buy the slices, and the arrangers were left holding an estimated couple of hundred billion dollars of "hung" loans — loans they had committed to fund but could not sell except at a loss. They eventually sold them at deep discounts, taking real losses, which is one reason the leveraged-loan market seized up so violently early in the crisis. The lesson banks relearned: underwriting a loan is selling insurance on your own ability to find buyers, and that ability vanishes exactly when you need it.

### The syndication timeline, week by week

It helps to see syndication as a process with a clock. In a typical deal, the borrower runs a *beauty parade*, inviting several banks to pitch for the mandate; each proposes a structure, a price, and how much it is willing to underwrite. The winner — the lead arranger — signs a *commitment letter* and a *term sheet* with the borrower, locking the broad terms. Then comes the *information memorandum* (the "bank book"), a detailed package describing the borrower, the deal, and the risks, sent under confidentiality to potential lenders. The arranger hosts a *bank meeting* (often now a call) to present the deal, then opens a *syndication period* — usually a few weeks — during which lenders submit their commitments. If demand exceeds the loan size, the deal is *oversubscribed* and the arranger can *flex* the terms tighter (lower the spread, the borrower's win); if demand falls short, the arranger may have to *flex* the other way (raise the spread to attract buyers, or, in an underwriting, eat the unsold piece). Finally the documents are signed, the loan *closes*, and the agent takes over.

The flex provisions are the pressure valve that makes underwriting survivable. A *market-flex* clause lets the arranger move the price within a pre-agreed range to clear the market, so an arranger that misjudged demand can still sell the loan rather than hold it. When markets are calm, deals price tight and flex toward the borrower; when markets wobble, flex moves toward the lenders, and in a real freeze even maximum flex is not enough — which is exactly how the 2007 hung loans happened. The timeline, in other words, contains the seed of syndication risk: there is always a gap between the day the arranger commits and the day it has actually sold the loan, and that gap is where the market can turn against it.

### Why the agent role matters more than it looks

The agent bank is the unglamorous centre of a syndicated loan. It is the single point of contact for the borrower, the collector and distributor of every payment, the keeper of the covenant calculations, and — crucially — the coordinator if the loan goes bad. When a syndicated borrower starts to wobble, the lenders do not act individually; they act through the agent and according to the contract, which specifies what fraction of lenders (often a "majority", two-thirds by value) must agree to waive a covenant, accelerate the loan, or restructure. This matters because syndicates can fracture: a bank that wants its money back will push to enforce, while a bank that wants to preserve the relationship will push to waive. The agency mechanics and the voting thresholds are what keep a syndicate from descending into a free-for-all the moment trouble appears.

## Covenants: the tripwires that let a bank act early

If syndication is how banks limit the *size* of a loss, covenants are how they limit the *timing* — they let a lender act before a deteriorating borrower turns into a defaulted one. This is the most misunderstood part of corporate lending, so we will build it carefully.

A loan's biggest enemy is not a sudden default; it is a slow decline that the lender cannot do anything about until it is too late. A company's earnings sag, it borrows more to paper over the gap, its leverage creeps up, and by the time it actually misses an interest payment, its assets are worth far less than the debt and recovery is poor. Covenants exist to interrupt that slide. They are *promises* the borrower makes — and tests it must keep passing — such that breaking one gives the lender a contractual right to step in *while there is still value to protect*.

### What a breach actually does

Here is the single most important thing to understand, because films and headlines get it wrong: a covenant breach does **not** automatically seize the company or call the loan. Breaching a financial covenant is a *technical default* (also called an event of default), which gives the lenders a menu of *rights* — but the lenders almost always choose to *negotiate* rather than detonate. A breach is leverage, not a guillotine.

![Before and after diagram showing a loan with covenant intact versus a covenant breach and the bank rights it triggers](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-4.png)

When a covenant is breached, the lenders can typically do one of three things. They can **waive** it — agree to overlook the breach, usually in exchange for a *waiver fee* and sometimes a higher spread going forward. They can **amend** the loan — reset the covenant to a looser level (an "amend and extend"), again for a fee and a price bump. Or, if the situation is dire, they can **accelerate** — declare the entire loan immediately due and enforce against any collateral. In practice, waivers and amendments are overwhelmingly the common outcome, because acceleration usually pushes the borrower into bankruptcy, which is slow, expensive, and often recovers less than a negotiated fix. The breach's real value is that it forces the borrower back to the table on the lender's terms.

#### Worked example: a leverage covenant breach (Debt/EBITDA ≤ 4×)

The most common financial covenant is a cap on **leverage**, measured as the ratio of total debt to **EBITDA** — *earnings before interest, taxes, depreciation, and amortisation*, a rough proxy for the operating cash a company generates. A loan might require:

$$\frac{\text{Total debt}}{\text{EBITDA}} \le 4.0\times$$

Suppose at signing the company has \$320 million of debt and \$100 million of EBITDA:

$$\frac{\$320\text{m}}{\$100\text{m}} = 3.2\times \quad (\text{comfortably below } 4.0\times)$$

A year later, a recession hits. EBITDA falls to \$80 million and, to cover a cash shortfall, the company has drawn its revolver and now carries \$368 million of debt:

$$\frac{\$368\text{m}}{\$80\text{m}} = 4.6\times \quad (\text{above the } 4.0\times \text{ limit — breach})$$

The company is still paying its interest on time. But because the leverage covenant is tested *every quarter*, the moment the ratio prints 4.6x the company is in technical default. The lenders did not have to wait for a missed payment; the covenant gave them a tripwire that fired while EBITDA was merely soft, not gone. Now they negotiate. A common outcome: the lenders waive the breach, raise the spread by 100 basis points, charge a waiver fee, and require the company to cut costs and pay down debt to get back under 4.0x within two quarters. The intuition: the covenant converted a quiet deterioration into an event the bank could price and control, months before it would have become a loss.

Why is 4.0x such a common dividing line? Because each extra "turn" of leverage — each extra multiple of debt to EBITDA — raises the chance of default by *more* than the turn before it. A company at 2x leverage can absorb a bad year and still cover its debt; a company at 6x has almost no room before a single soft quarter pushes it toward insolvency. The relationship is convex, not linear, which is exactly why lenders draw the maintenance line where they do and why the spread climbs so steeply for highly leveraged borrowers. The chart below sketches that convexity: the curve barely rises through low leverage, then accelerates sharply past the typical 4.0x limit into the leveraged-loan zone.

![Line chart showing default probability rising convexly with leverage measured as debt to EBITDA](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-7.png)

The convex shape is the mathematical heart of leveraged finance. It tells you why a private-equity sponsor pushing leverage from 5x to 6x to squeeze out a higher return on equity is taking on far more than 20% extra default risk — and why a lender that agrees to it demands a much wider spread, tighter covenants (if it can get them), and a senior secured claim on the assets. The same convexity is why a recession is so dangerous for leveraged borrowers: a 20% fall in EBITDA does not raise leverage by 20%, it raises the *ratio* sharply because the denominator shrank, pushing a cluster of borrowers over their covenant lines and into the lender's office all at once.

### Maintenance versus incurrence: the distinction that defines the loan

Here is the distinction that separates a tightly controlled loan from a loose one, and it has reshaped the entire leveraged-loan market over the past fifteen years.

A **maintenance covenant** is tested *continuously* — typically every quarter — whether or not the company does anything. The leverage cap above is a maintenance covenant: the company must *maintain* Debt/EBITDA below 4.0x at all times, and a soft quarter alone can trip it. Maintenance covenants give lenders an early, automatic warning system.

An **incurrence covenant** is tested *only when the company takes a specific action* — most often, only when it tries to *incur* more debt or pay a dividend. An incurrence-based leverage test might say: "the company may borrow more only if, after the new borrowing, Debt/EBITDA stays below 5.0x." If the company does nothing, the covenant is never tested, no matter how badly earnings fall. Incurrence covenants are far more borrower-friendly because a declining company that simply sits still never trips them.

![Comparison matrix of covenant types maintenance versus incurrence across leverage coverage and liquidity](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-8.png)

The shift from maintenance to incurrence covenants is the story of the modern leveraged-loan market. Loans with few or no maintenance covenants are called **covenant-lite** (or "cov-lite"). Before the 2008 crisis, cov-lite loans were a small fraction of the leveraged-loan market; by the early 2020s, the overwhelming majority of new leveraged loans — well over 80% in the U.S. — were cov-lite. This happened because, in a world awash with money chasing yield, borrowers (and the private-equity sponsors behind them) had the bargaining power to strip out the maintenance tests that let lenders intervene early. The consequence is real: in a cov-lite world, lenders lose their early-warning tripwire and often cannot act until the company actually misses a payment — by which point recovery is lower. Cov-lite does not make loans default more often, but it tends to make them recover *less* when they do, because the lender got to the table later. When you read that a deal is "cov-lite", read it as "the lender gave up the right to act early in exchange for winning the mandate".

### The three families of financial covenant

Beyond leverage, financial covenants come in three families, and a well-structured loan usually has one of each:

- **Leverage covenants** cap debt relative to earnings (Debt/EBITDA). They answer: *is the company taking on more debt than its earnings can support?*
- **Coverage covenants** require earnings to comfortably cover fixed charges — for example, *interest coverage*, EBITDA divided by interest expense, must stay above 3.0x. They answer: *can the company actually afford its interest bill?*
- **Liquidity covenants** require the company to hold a minimum cushion of cash or undrawn credit. They answer: *can the company survive a few bad months without running out of money?*

Each covenant is a different lens on the same question — will this company still be able to pay us? — and together they let a lender catch trouble whether it shows up as too much debt, too little earnings, or too little cash. A well-built loan also defines exactly *how* EBITDA is calculated, because borrowers have learned to inflate it with optimistic "add-backs" — projected cost savings or one-off charges that flatter the number and make leverage look lower than it really is. The negotiation over the EBITDA definition is one of the most consequential in the whole document: a generous add-back regime can quietly turn a 4.5x company into a covenant-compliant 3.9x one on paper, which is why experienced lenders scrutinise *adjusted* EBITDA as hard as the covenant ratio itself. The covenant is only as honest as the number it tests.

## The relationship is the real product

Step back and ask why a bank would do all this — structure facilities, syndicate risk, negotiate covenants — for a loan whose interest spread, after funding costs and capital charges, often yields a thin single-digit return on the capital tied up. The answer is that the loan is not the product. The **relationship** is.

A large company is a fountain of fee income. Over a few years it will likely raise a bond (underwriting fees), hedge its interest-rate and currency exposure (markets revenue), run its global cash through transaction-banking products (steady, sticky fees and cheap deposits), and eventually buy or sell a business (advisory fees that can dwarf years of loan interest). The bank that wins the lending relationship — especially the revolver and the agency role — is first in line for all of it. The loan is the cost of admission to a much larger room.

![Pipeline diagram of the life of a corporate loan from origination through arranging syndication monitoring to the ongoing relationship](/imgs/blogs/corporate-and-commercial-lending-term-loans-revolvers-and-syndication-6.png)

This is why banks will sometimes lend at prices that look like they barely cover risk and capital. The lending desk and the broader bank operate on a *share-of-wallet* logic: a corporate loan that returns, say, 8% on its capital in isolation might return 20% once you credit it with the bond mandate, the hedging flow, and the cash-management deposits the relationship brings in. The corporate banker's job is not to maximise the return on the loan; it is to maximise the return on the *relationship*, and the loan is the lever that opens it.

It also explains a behaviour that puzzles outsiders: banks fighting to lend to companies that could easily borrow more cheaply elsewhere, and companies keeping relationships with banks whose loan rates are not the lowest available. Both sides are pricing the relationship, not the loan. A company wants a bank that will actually fund its revolver in a crisis and lead its next bond; a bank wants a foot in the door of every fee stream the company will generate for a decade. The interest rate is a rounding error in that calculation.

There is a quieter benefit hiding inside the relationship, and it is the one this series cares about most: deposits. A large corporate client runs its cash through its relationship bank — payroll, supplier payments, collections, sweep accounts — and that operating cash sits as **deposits** on the bank's balance sheet, often at a low interest rate. Those deposits are exactly the cheap, sticky funding that makes a bank a profitable maturity-transformation machine in the first place. So a corporate loan can be worth making even at a thin lending margin if it brings in a flow of low-cost operating deposits, because those deposits fund *other*, more profitable, lending. The loan and the deposit are two ends of the same relationship, and the banker who sees only the loan margin is reading half the page.

This is also why losing a relationship hurts a bank far more than losing a single loan. When a company moves its primary banking to a competitor, it does not just refinance one facility; it takes the revolver, the cash management, the deposits, the hedging flow, and the inside track on the next bond and the next acquisition with it. The cost of that loss is measured in years of fee income, not in the spread on one loan. Relationship banking is sticky on purpose: the whole architecture — the revolver as anchor, the agency role, the share-of-wallet pricing — is designed to make the relationship expensive to leave.

## Common misconceptions

**"A bigger loan means a bigger bank took more risk."** No — a bigger loan usually means *more banks shared the risk*. The whole point of syndication is that a \$2 billion loan can be spread across thirty lenders so no single one holds a dangerous amount. The bank whose name is on the press release as lead arranger may end up holding less than 10% of the loan after the sell-down. Size of the headline figure tells you almost nothing about any one bank's exposure.

**"Breaching a covenant means the bank takes the company."** Almost never. A covenant breach is a *technical default* that gives lenders the *right* to act, but the overwhelming response is to negotiate — a waiver or an amendment, in exchange for a fee and a higher spread. Acceleration and enforcement are last resorts, because they usually trigger bankruptcy and recover less than a negotiated fix. The covenant's value is the leverage it creates, not the seizure it threatens.

**"A revolver is free until you use it."** No — you pay a **commitment fee** on the undrawn portion the entire time the facility is open, often 0.25% to 0.50% a year. On a \$100 million revolver left undrawn, that is \$250,000 to \$500,000 a year for the *option* to borrow. The bank earns it because it must keep the money and the regulatory capital available whether or not you ever draw. The revolver is rent on a promise, not free money.

**"Cov-lite loans default more often, which is why they are dangerous."** The evidence does not support the first half. Cov-lite loans do not clearly default *more often* than covenanted ones; the danger is that when they do default, lenders typically recover *less*, because the absence of maintenance covenants meant they could not intervene until the company actually ran out of cash. The risk is in the *recovery*, not the frequency — a subtler and more important point.

**"The benchmark rate is what the borrower's credit quality determines."** Backwards. The benchmark (SOFR) is the same for every borrower in the economy — it is the cost of money, set by policy and the repo market. The borrower's credit quality determines the **spread** added on top. A AAA company and a CCC company both pay SOFR; the AAA pays SOFR + 100 and the CCC pays SOFR + 700. When credit conditions deteriorate, spreads widen even if the benchmark is flat — that is the market repricing risk, not the central bank moving rates.

**"The lead arranger keeps the biggest slice of the loan."** Often the opposite. The arranger's goal is usually to *distribute* the risk, not hoard it — its money is in the arranging and agency *fees*, not in holding the paper. On a heavily distributed deal the lead may keep only a token amount, sometimes called *skin in the game* to reassure other lenders, while selling the bulk to participants and institutional buyers. The bank that did the most work and earned the most fees can be the one holding the least risk when the dust settles — which is the entire economic point of being an arranger rather than just a lender.

## How it shows up in real banks

**The 2007 hung-loan freeze.** In the buyout boom of 2006–07, banks underwrote leveraged loans on aggressive, borrower-friendly terms, confident they could sell every slice. When the credit market seized in mid-2007, investors stopped buying, and arrangers were left holding an estimated couple of hundred billion dollars of "hung" loans they had committed to fund but could not syndicate. They eventually offloaded them at deep discounts, booking real losses. The episode is the textbook illustration of *syndication risk*: underwriting a loan is a promise to find buyers, and that promise is hardest to keep exactly when markets are falling.

**The March 2020 revolver run.** When the pandemic froze markets in March 2020, hundreds of investment-grade companies — airlines, carmakers, retailers — drew down their revolving credit facilities all at once to hoard cash, pulling well over \$200 billion from their banks in a matter of weeks. Banks had sold these revolvers as cheap liquidity backstops, never expecting so many to be claimed simultaneously. The episode showed that a committed revolver is a written liquidity-insurance policy that gets exercised precisely when the bank's own funding is most stretched — a maturity-transformation trap hiding inside an off-balance-sheet promise.

**The rise of cov-lite and the recovery question.** Through the 2010s, cheap money and yield-hungry investors let leveraged borrowers strip maintenance covenants out of new loans, until cov-lite became the market standard — over 80% of U.S. leveraged-loan issuance by the early 2020s. When defaults rose in 2023–24, the consequence appeared in the recovery data: senior secured leveraged loans, which historically recovered around 70 cents on the dollar, recovered noticeably less in the cov-lite era, because lenders reached the restructuring table later and with weaker rights. The structural change in covenants quietly changed the loss content of the entire asset class.

**The syndicated agent in a workout.** When a large syndicated borrower files for bankruptcy, the agent bank becomes the syndicate's hub. Take a leveraged retailer that breaches its covenants, draws its revolver to the maximum, then files: the agent must coordinate dozens of lenders with different incentives — some banks wanting to enforce and exit, some institutional funds happy to convert debt to equity and own the restructured company. The contract's voting thresholds (often two-thirds by value to amend, all-lender consent for the most sensitive changes) decide who controls the outcome. The agency role and the intercreditor terms, negotiated years earlier in calm markets, determine who gets paid first when the money runs out.

**The EBITDA add-back drift of the 2010s.** Through the long credit boom, the definition of EBITDA in leveraged-loan documents drifted steadily in borrowers' favour. Sponsors negotiated ever more generous *add-backs* — letting companies count hoped-for future cost savings as if already realised — so that a company's reported leverage could understate its true leverage by a full turn or more. When defaults rose in 2023–24, several borrowers that had looked comfortably covenant-compliant on adjusted EBITDA turned out, on actual cash earnings, to have been over their real limits for years. The episode is a reminder that a covenant is a contract about a *number*, and whoever controls how the number is defined controls how much protection the covenant really provides.

**Relationship lending and the league tables.** Banks publish, and obsess over, *league tables* ranking who arranged the most loan volume. The reason is not vanity: a high ranking attracts mandates, and each lending mandate is a door to the borrower's fee wallet. A bank may lead a low-margin syndicated loan for a blue-chip company specifically to climb the league table and position itself for that company's next bond issue or acquisition advisory — fees that can exceed a decade of the loan's interest. The loan is loss-leader marketing for the relationship, and the league table is the scoreboard for who is winning the relationships.

## The takeaway: read the structure, not the headline

Corporate lending looks, from the outside, like banks handing money to companies. Read it that way and almost everything important is invisible. The size of the loan tells you nothing about any one bank's risk, because the loan is sliced across a syndicate. The interest rate tells you nothing about the bank's motive, because the loan is the cost of admission to a relationship whose fees dwarf the interest. The absence of a missed payment tells you nothing about the borrower's health, because the covenants — if there are any — are the early-warning system, and their absence (cov-lite) means the warning never comes until it is too late.

So when you look at a corporate loan — as an investor reading a bank's loan book, a journalist parsing a deal, or a borrower negotiating a facility — read the *structure*. Ask: how is the risk shared (who is the arranger, who are the participants, who is the agent)? How is it priced (what is the spread over the benchmark, and what does that spread say about the bank's view of default risk)? And how is it controlled (are there maintenance covenants that fire early, or is it cov-lite, leaving the lender blind until the cash runs out)? Those three questions — sharing, pricing, control — are corporate lending. The dollar figure on the front page is the least informative number in the whole deal.

And tie it back to the spine that runs through this entire series: a bank survives only as long as its thin equity cushion absorbs losses faster than they arrive. Corporate lending is dangerous precisely because corporate loans are large and lumpy — one failed borrower can carve a real chunk out of that cushion. Syndication is the answer to size, spreading any single loss across many balance sheets. Covenants are the answer to timing, letting a lender shrink the loss by acting early. And the relationship model is the answer to profitability, because the loan alone rarely earns enough to justify the risk to that precious equity. Watch how a bank shares, prices, and controls its corporate loans, and you are watching whether it understands the one trade its life depends on.

## Further reading & cross-links

- [The Lending Business: How a Bank Underwrites a Loan End to End](/blog/trading/banking/the-lending-business-how-a-bank-underwrites-a-loan-end-to-end) — the full origination-to-collection pipeline that corporate lending plugs into.
- [Loan Pricing: Cost of Funds, Risk Premium, and the Capital Charge](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge) — the deep dive on how the spread over the benchmark is actually built, including RAROC.
- [Securitization: How Banks Turn Loans into Securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — what happens to syndicated and leveraged loans after origination, including CLOs that buy the slices.
- [Bank Capital and Leverage: Why Equity Is the Thin Cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — why a large single-borrower loss is so dangerous to a bank, and why syndication exists.
- [Inside an Investment Bank: How They Make Money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the wider fee-income machine that the corporate-lending relationship feeds into.
