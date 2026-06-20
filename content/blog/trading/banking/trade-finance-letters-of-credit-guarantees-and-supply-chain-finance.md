---
title: "Trade Finance: Letters of Credit, Guarantees, and Supply-Chain Finance"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank stands between a buyer and a seller who don't trust each other — letters of credit, documentary collections, guarantees, and supply-chain finance, built from zero."
tags: ["banking", "trade-finance", "letter-of-credit", "documentary-collection", "bank-guarantee", "standby-letter-of-credit", "factoring", "reverse-factoring", "supply-chain-finance"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Trade finance is the business of a bank renting out its own creditworthiness so two strangers in two countries can trade without either one going first into the dark.
>
> - A **letter of credit (LC)** swaps the question "do I trust this foreign company?" for "do I trust this big bank?" — the bank promises to pay the seller against documents, not against the goods, so the deal turns on paperwork, not faith.
> - A **documentary collection** is the cheaper, weaker cousin: the bank moves the documents and holds them hostage for payment, but it does **not** promise to pay if the buyer walks away.
> - A **bank guarantee / standby LC** is a promise to pay only if something goes *wrong* — a safety net that, by existing, lets the deal happen and usually never gets called.
> - **Supply-chain finance** (factoring and reverse factoring) is the bank lending against the *invoice* after the goods ship — turning a 90-day "I'll pay you later" into cash today, priced off whoever has the better credit.
> - The one number to remember: a confirmed LC on a shipment typically costs the importer roughly **0.75%–1.5% per quarter** of the shipment value in bank fees — the price of buying away counterparty risk.

In December 2008, as the financial crisis froze the world's credit markets, something quietly terrifying happened to physical trade. Ships full of iron ore, soybeans, and steel sat idle in ports — not because nobody wanted the cargo, but because the **letters of credit that normally financed the voyage had stopped being issued**. The Baltic Dry Index, which tracks the cost of shipping bulk goods by sea, collapsed by more than 90% in a few months. A Brazilian exporter would load a vessel only if a bank promised to pay; the buyer's bank, hoarding cash and unsure which counterparties were solvent, would not promise. So the soybeans stayed on the dock. Global trade is built on a layer of bank promises most people never see, and in 2008 the world got a glimpse of what happens when that layer cracks.

That invisible layer is **trade finance**, and it solves a problem older than banking itself: a seller in Vietnam and a buyer in Germany have never met, are 9,000 km apart, fall under different legal systems, and each one is being asked to go first. The seller is afraid to ship goods to a stranger who might never pay. The buyer is afraid to wire money to a stranger who might never ship. Without something in the middle, the trade simply does not happen — and most of the world's commerce *is* between strangers who will never meet.

The diagram below is the mental model for the whole post: trade finance is a bank standing in the gap, lending its own creditworthiness to bridge a trust that does not exist between the two parties. The bank is not buying the goods or taking a view on the price; it is **renting out its name**. Everything else — the document flow, the fee schedule, the guarantee, the factoring discount — is mechanics on top of that one idea.

![Bank stands in the middle of an international trade between a distrustful buyer and seller using a letter of credit](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-1.png)

## Foundations: the trust gap and the instruments that bridge it

Before any mechanics, let's define every term from zero. If you have never read a trade-finance document in your life, this section is where you start; a practitioner can skim it.

**The trust gap.** Take the simplest possible cross-border deal. An importer (the **buyer**) in one country agrees to buy \$500,000 of machinery from an exporter (the **seller**) in another. Now ask: who pays first, and who ships first? If the seller ships first and the buyer never pays, the seller has lost \$500,000 of goods sitting in a foreign port. If the buyer pays first and the seller never ships, the buyer has lost \$500,000 of cash to a company it can't easily sue. Each side wants the *other* one to go first. That standoff is the **trust gap** — the structural reason international trade needs a middleman. Domestically you might extend credit because you can sue, you share a legal system, and you've heard of the firm. Across borders, none of that holds.

**Counterparty risk.** The risk that the other party to a deal — your *counterparty* — fails to do what they promised (pay, ship, perform). Trade finance is, at its heart, a set of techniques for moving counterparty risk *off* a company you don't trust and *onto* a bank you do.

**Letter of credit (LC),** also called a *documentary credit*. A written promise by a bank — the *issuing bank* — to pay the seller a fixed sum, **provided the seller presents a specified set of documents** (typically the bill of lading, invoice, insurance certificate, inspection certificate) that prove the goods were shipped as agreed. The crucial word is *documents*. The bank never inspects the cargo. It pays against paper. This swaps the seller's question from "will this unknown buyer pay me?" to "will this large bank honor its own written promise?" — a far easier question.

**Bill of lading (B/L).** The single most important document in the LC world. It is issued by the shipping carrier and does three things at once: it is a **receipt** that the goods were loaded, a **contract** of carriage, and — critically — a **document of title**. Whoever legally holds the original bill of lading controls the goods; the carrier will release the cargo only to that holder. This is why the LC can be safe: the bank can hold the bill of lading and thereby hold the goods themselves as security until it is paid.

**Documentary collection.** A cheaper service in which the seller's bank forwards the shipping documents to the buyer's bank with instructions: "release these documents to the buyer **only against payment** (or against the buyer's signed promise to pay)." The bank acts as an *escrow agent for paper*, but — and this is the whole difference from an LC — **the bank does not promise to pay**. If the buyer refuses, the bank simply hands the documents back. The seller's protection is that the buyer can't get the goods without paying; but the seller can still get stuck with goods abandoned in a foreign port.

**Bank guarantee.** An independent written promise by a bank to pay a sum to a *beneficiary* if the bank's customer fails to perform some obligation. A guarantee pays out only when something goes *wrong* (a contractor abandons a project, a buyer fails to pay). It is a safety net, not a payment mechanism. In most of the world it's called a guarantee; in the United States the same economic role is filled by the **standby letter of credit**.

**Standby letter of credit (SBLC).** A letter of credit structured to function as a guarantee. A normal ("commercial") LC is the *primary* way the seller gets paid — it's expected to be drawn on every deal. A *standby* LC is expected **never** to be drawn — it pays only if the underlying deal defaults. Same legal machinery, opposite intent.

**Factoring.** The seller sells its unpaid invoices ("receivables") to a bank or finance company (the *factor*) at a discount, in exchange for cash today. The seller initiates it to fix its own cash flow. Pricing is driven by the *buyer's* credit, because the buyer is who ultimately pays the invoice.

**Reverse factoring (supply-chain finance, SCF).** The opposite initiation. A large, creditworthy *buyer* sets up a program so that *its suppliers* can get paid early by the buyer's bank, at a low rate that reflects the **buyer's** strong credit rather than the supplier's weak one. The buyer benefits (it can stretch its own payment terms), and the supplier benefits (cheap early cash). The bank benefits (low-risk lending against a blue-chip name).

**Open account vs cash-in-advance.** The two endpoints of the trust spectrum. *Open account* means the seller ships and just trusts the buyer to pay later — all risk on the seller, used only when trust is high. *Cash-in-advance* means the buyer pays before anything ships — all risk on the buyer. LCs and collections live in between, splitting the risk.

Here is the single instinct that ties trade finance to this series' spine — **a bank is a leveraged, confidence-funded machine that survives on trust.** In lending, the bank rents out its *money*. In trade finance, it rents out its *name and creditworthiness* — its trust — for a fee, often without putting cash out the door at all until and unless a document is presented. It is one of the highest-return, lowest-balance-sheet things a bank does. That's why transaction and trade banking is so prized: it earns fees on the bank's reputation, the cheapest asset it owns.

## Trade without a bank: why both sides freeze

Let's make the trust gap concrete before we solve it. Strip the bank out entirely and watch the deal die.

Imagine the Vietnamese exporter "Mekong Furniture" agrees to sell \$200,000 of teak dining sets to "Hanseatic Imports" in Hamburg. They've emailed; they've never met. Now they have to choose a settlement method, and every choice gored one of them:

- **Cash in advance.** Hanseatic wires \$200,000 before anything ships. Mekong is delighted; Hanseatic is exposed to a total loss if Mekong ships nothing, ships junk, or simply vanishes. A German importer is not going to send \$200,000 into the dark to a small Vietnamese firm it has never dealt with.
- **Open account.** Mekong ships first and invoices for payment in 60 days. Hanseatic is delighted; Mekong is exposed to a total loss if Hanseatic refuses to pay once the goods have arrived (and good luck suing in a German court from Vietnam). A small exporter cannot afford to gamble its whole quarter's production on a stranger's promise.

So the safe choice for *each* party is the catastrophic choice for the *other*. The before-and-after figure shows the deadlock and the way out. On the left, with no intermediary, both parties stare at the gap and the deal stalls. On the right, the LC drops a bank into the middle: now Mekong relies on a bank's promise (not Hanseatic's), and Hanseatic only releases payment against documents that prove the goods shipped (not on Mekong's word). The bank has converted a problem of mutual distrust into two separate, manageable problems of *trusting a bank*.

![Trade deadlock with no bank versus a bank bridging the gap with a letter of credit](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-2.png)

The genius is that the bank's promise is **abstracted from the goods**. The bank does not need to know whether teak is good teak; it only needs documents to match. That abstraction is what makes the system scale: a bank in Hamburg can confidently finance furniture, soybeans, and turbine blades without knowing anything about furniture, soybeans, or turbines. It only knows paper.

> This independence is also the system's sharpest edge. Because the bank pays against *documents*, not against the *actual goods*, a buyer can receive a container of bricks and still be obligated to pay if the seller's documents were technically perfect. The LC protects against the *risk of non-payment*, not against the *risk of fraud or bad goods* — a distinction we'll return to under misconceptions.

## How a letter of credit actually flows

Now the mechanics. A commercial LC involves up to four parties and a precise sequence. Let's name them, then walk the flow.

- **Applicant** = the buyer (importer). Asks its bank to issue the LC.
- **Issuing bank** = the buyer's bank. Issues the LC and carries the obligation to pay.
- **Advising bank** = a bank in the seller's country that authenticates the LC and passes it to the seller. (It just advises; it does not promise to pay.)
- **Beneficiary** = the seller (exporter). Gets paid under the LC.
- **Confirming bank** (optional) = a bank — often the advising bank — that adds its *own* promise to pay on top of the issuing bank's. We'll come back to confirmation.

The cover diagram traces the canonical flow. Read it left-to-right, then back: the application and credit move out from the buyer's side, the goods and documents move in from the seller's side, and the money flows back along the document chain. The numbered steps:

1. **Contract.** Buyer and seller agree the sale and that payment will be by LC.
2. **Application.** The buyer applies to its bank (the issuing bank) for an LC, posting collateral or using a credit line.
3. **Issuance.** The issuing bank issues the LC and sends it (via SWIFT, the interbank messaging network) to a bank in the seller's country.
4. **Advising.** The advising bank authenticates the LC and notifies the seller: "a bank has promised to pay you if you ship and present these documents."
5. **Shipment.** Now — and only now — the seller ships the goods, because it has a bank's promise in hand. The carrier issues the bill of lading.
6. **Presentation.** The seller hands the shipping documents to the advising/confirming bank.
7. **Examination & payment.** The bank examines the documents against the LC's terms. If they comply, the seller gets paid. The documents travel to the issuing bank, which reimburses, and hands the bill of lading to the buyer — who can now collect the goods.

![Letter of credit document flow from buyer to issuing bank to advising bank to seller and back](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-3.png)

Notice the ordering. The seller ships at step 5 *only because* a bank promised at step 4. The buyer pays at step 7 *only because* documents proving shipment arrived. Neither party ever had to go first into the dark; the bank's promise sequenced the whole dance. This is also why the LC is sometimes called a "documentary credit" — the credit is conditional on documents, full stop.

### Inside document examination: where deals actually break

The romantic part of an LC is the bank's promise. The *operational* reality is a checker at a desk comparing two stacks of paper against a rulebook, and this is where most LCs hit trouble. The global rulebook is the **UCP 600** — the Uniform Customs and Practice for Documentary Credits, published by the International Chamber of Commerce (ICC). Every LC says it is "subject to UCP 600," and that single line imports a whole code of how documents must be examined.

The bank has a defined window — **five banking days** under UCP 600 — to examine a presentation and decide: pay, or refuse for *discrepancies*. The examiner is not checking whether the deal is good; she is checking whether the documents are *internally consistent and match the LC's terms*. Does the invoice description of the goods match the LC word-for-word? Is the bill of lading dated within the shipment window? Does the insured amount cover the required percentage? Are there as many original copies as the LC demands? Is the LC still within its expiry and its presentation period (typically 21 days after shipment unless stated otherwise)? A single mismatch — a port named "Haiphong" on one document and "Hai Phong" on another, an amount over the LC's tolerance, a missing signature — is a *discrepancy*, and a discrepancy gives the issuing bank a lawful reason to refuse payment.

When documents are discrepant, the bank does not simply pay or reject in a vacuum. It typically contacts the buyer (the applicant): "the documents have these discrepancies — do you waive them and authorize payment?" Often the buyer waives, because it wants the goods and the discrepancy is cosmetic. But the buyer now holds leverage it shouldn't: in a falling market, a buyer that no longer wants the goods can *refuse to waive* a trivial discrepancy and walk away legally, leaving the seller exactly where it would be with no LC at all. This is the seller's deepest vulnerability — the "guaranteed" payment is only guaranteed if the seller's paperwork is flawless. Experienced exporters treat document preparation as a discipline equal to making the goods, and many pay a freight forwarder or their bank to pre-check documents before presentation.

### The parties and what each one actually guarantees

It is easy to blur the four banks together. They are not the same, and confusing them is how importers get burned. The graph figure lays out who promises what.

![Parties in a letter of credit and what each one guarantees](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-4.png)

The load-bearing distinctions:

- The **issuing bank** is the only party (other than a confirming bank) that *promises to pay*. Its promise is as good as its own balance sheet. If you're the seller and the issuing bank is a shaky institution in a country with capital controls, that promise is worth less than it looks.
- The **advising bank** promises *nothing about payment*. It only certifies the LC is genuine. Sellers sometimes mistake "my local bank advised the LC" for "my local bank will pay me." It will not, unless it also *confirmed*.
- The **confirming bank** adds its own independent promise. Now the seller has *two* banks on the hook: if the issuing bank fails to pay, the confirming bank pays anyway and chases the issuing bank itself. This is what a seller buys when the issuing bank or its country is risky — it's the upgrade from "trust a foreign bank" to "trust my own bank."

#### Worked example: the all-in cost of an LC on a shipment

Let's price a real one. Mekong Furniture ships \$200,000 of teak sets to Hanseatic Imports under a confirmed LC, with payment due 90 days after shipment (a *usance* LC — one that pays on a delayed date rather than at sight). Fees are quoted per quarter (per 90-day period) and split between the two sides. Representative rates:

- **Issuance fee** (charged to the buyer by the issuing bank): 0.75% per quarter of \$200,000 = \$1,500.
- **Advising fee** (charged to the seller by the advising bank): a flat \$150.
- **Confirmation fee** (charged to the seller, because the seller wants the extra promise): 1.0% per quarter of \$200,000 = \$2,000. This is the price of the confirming bank's risk.
- **Document examination / negotiation fee** (seller): a flat \$250.

Buyer's all-in cost: **\$1,500**, or **0.75%** of the shipment. Seller's all-in cost: \$150 + \$2,000 + \$250 = **\$2,400**, or **1.2%** of the shipment. Combined, the trade carried roughly **\$3,900 of bank fees on \$200,000 of goods — about 1.95%** for a single 90-day shipment.

Is that expensive? Compared to a wire transfer (a few dollars), enormously. Compared to the alternative — a \$200,000 total loss if the counterparty defaults — it is cheap insurance. The fee is the *price of buying away counterparty risk*. The intuition: trade finance turns a binary catastrophe (lose everything or not) into a small, certain cost (a couple of percent), and that conversion is the entire value proposition.

### The variants that make the LC do more work

The plain commercial LC is the base case. Once you understand it, a family of variants falls out naturally, each solving a specific real-world friction. You don't need to memorize the jargon — you need to see that each variant is the same trust-renting machine bent to a new shape.

- **Sight LC vs usance (time) LC.** A *sight* LC pays the seller the moment compliant documents are presented. A *usance* LC pays on a future date — "90 days after the bill of lading date" — which gives the buyer time to receive and resell the goods before paying. The usance LC is, in effect, the issuing bank extending the buyer trade credit; the seller can still get cash early by asking a bank to *discount* the future payment (pay, say, \$197,000 now for the \$200,000 due in 90 days, keeping the \$3,000 as interest). That discounting of an accepted usance draft is its own product — a *banker's acceptance*, a centuries-old money-market instrument that turns a trade promise into a tradeable IOU.
- **Confirmed vs unconfirmed LC.** Covered above: confirmation adds a second bank's promise. The seller pays for it when it doesn't trust the issuing bank or its country.
- **Transferable LC.** Lets the seller (the first beneficiary) transfer part or all of the credit to a *second* beneficiary — usually because the seller is really a middleman/trader who buys the goods from an actual manufacturer. The trader can transfer the LC to its own supplier so the supplier ships directly and gets paid, while the trader keeps the margin. This is how commodity traders finance deals without using their own cash.
- **Back-to-back LC.** When a transferable LC isn't available, the middleman uses the incoming LC from its buyer as *collateral* to have its own bank issue a *second*, separate LC to its supplier. Two distinct LCs, stacked — riskier for the bank (it must manage two sets of documents that have to line up) and so reserved for trusted clients.
- **Revolving LC.** For repeat shipments under one long-term contract, a revolving LC automatically reinstates after each drawdown, so the parties don't re-paper an LC for every container. It cuts cost and friction on a steady relationship.
- **Red-clause LC.** Includes a special clause letting the seller draw a *pre-shipment advance* against the LC — cash to buy raw materials before the goods are even made. The buyer is effectively pre-financing its own supplier, used in agriculture and seasonal trades.

The takeaway from this zoo of variants is not the names; it is that the LC is a *configurable promise*. Once a bank has agreed to stand in the middle, the parties can tune *when* payment happens (sight vs usance), *who* it flows to (transferable, back-to-back), *how often* (revolving), and *how early* (red-clause) — each tuning shifting a sliver of risk or cash-flow timing for a corresponding fee.

#### Worked example: discounting a usance LC into cash today

Mekong ships under a **180-day usance LC** for \$200,000 — meaning the issuing bank will pay \$200,000 in 180 days, not today. Mekong needs cash now, so it asks the confirming bank to *discount* the accepted draft. The bank quotes a discount rate of **6% per year**.

Discount charge: 6% × (180/365) × \$200,000 ≈ **\$5,918**. The bank pays Mekong **\$200,000 − \$5,918 = \$194,082** today and waits 180 days to collect the full \$200,000 from the issuing bank. Mekong has converted a 6-month promise into immediate cash for a known cost; the bank has lent against a *bank's* promise (the issuing bank's), which is why the rate is low — it isn't taking Mekong's credit risk, it's taking the issuing bank's. The intuition: a usance LC plus discounting equals "the seller gets paid now, the buyer pays later, and a bank bridges the gap" — the same maturity-transformation trick that defines the whole institution, applied to a single shipment.

### Documentary collection: the cheaper, weaker cousin

If the parties trust each other a bit more — say they've traded before, or the country risk is low — they can skip the LC's expensive bank *promise* and use a **documentary collection**, where the bank only moves the *paper*.

There are two flavors, and the difference is exactly how much the buyer is trusted:

- **Documents against payment (D/P), also "cash against documents."** The buyer's bank releases the shipping documents (including the bill of lading the buyer needs to collect the goods) **only when the buyer pays**. The seller is reasonably protected: no payment, no documents, no goods.
- **Documents against acceptance (D/A).** The bank releases the documents when the buyer *signs* a promise to pay later (a "bill of exchange" the buyer "accepts"). Now the buyer gets the goods *before* paying — riskier for the seller, because a signature is not cash.

The critical thing to internalize: in a collection, **no bank guarantees payment**. The banks are couriers with a rulebook. If the buyer simply refuses the goods in a D/P, the seller is left with cargo stranded in a foreign port and a choice between shipping it home, selling it locally at a fire-sale price, or abandoning it. A collection is cheaper than an LC — often a flat fee of \$50–\$150 per side rather than 1%+ — precisely because the bank is selling a *service*, not lending its *creditworthiness*. You get what you pay for: less risk transfer.

## The risk ladder: which method, and who carries the danger

There are four main ways to settle a cross-border trade, and they form a clean ladder from "all risk on the buyer" to "all risk on the seller." Choosing among them is the first real decision in any deal, and it is almost entirely a negotiation about *trust* and *bargaining power*. The matrix figure lays out the four methods against the risk each party bears.

![Cash in advance versus letter of credit versus documentary collection versus open account risk matrix](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-5.png)

Reading the ladder from the seller's point of view, safest to riskiest: **cash-in-advance** (seller has the money already), **confirmed LC** (two banks promise), **unconfirmed LC** (one bank promises), **D/P collection** (bank holds documents hostage but won't pay), **D/A collection** (buyer gets goods on a signature), **open account** (pure trust). From the buyer's point of view the ladder is exactly reversed: open account is the buyer's dream and cash-in-advance is the buyer's nightmare.

Who wins the negotiation? Bargaining power decides. A small exporter selling a commodity to a giant retailer will be forced onto open account — the buyer can find another supplier in a day, so the supplier eats the risk and waits 90 days for payment. A specialized exporter selling unique turbine blades to a buyer in a volatile country can demand a confirmed LC or even cash-in-advance. As a trading relationship matures and trust builds, deals tend to *slide down the ladder* from LCs toward open account, shedding bank fees as the parties learn they don't need the middleman anymore. The bank's trade-finance revenue is, in a sense, a tax on *new* and *low-trust* relationships — which is why emerging-market and first-time trade is where the fee pool is richest.

#### Worked example: comparing the cost of the four methods on one shipment

Take Mekong's \$200,000 shipment and price the four methods, all-in, for a single 90-day cycle:

- **Cash-in-advance:** ~\$30 wire fee. Cheapest by far — but only available if the buyer will pre-pay, which it won't for a stranger.
- **Open account:** ~\$30 wire fee at the end — but the seller carries 100% of the default risk for 90 days, an *un-priced* exposure of up to \$200,000.
- **Documentary collection (D/P):** roughly \$100 + \$100 = ~\$200 in bank handling fees. The bank holds documents but does not pay; the seller still risks a refused shipment.
- **Confirmed LC:** ~\$3,900 (from the worked example above). The most expensive, and the only one where a bank *promises* the seller gets paid.

The intuition: the price of each method is almost exactly the price of the *risk transfer it provides*. Open account looks free but hides a \$200,000 tail risk; the confirmed LC costs ~2% but eliminates that tail. There is no free lunch — you either pay the bank a fee or you keep the risk yourself.

## Bank guarantees and standby LCs: paying only when things go wrong

So far every instrument has been about getting the *seller paid* for goods. Guarantees are different: they are promises that pay only when someone *fails to perform*. They are the safety nets that let a deal happen at all — and the well-designed ones almost never pay out.

The everyday analogy is a **security deposit** on an apartment. The landlord doesn't expect to keep it; it sits there as a promise that *if* you trash the place, money is available to fix it. Its mere existence makes the landlord comfortable renting to a stranger. A bank guarantee is the institutional version: a bank holds a promise so that a project owner is comfortable hiring a contractor it doesn't know.

The main types, by what triggers them:

- **Bid bond (tender guarantee).** When a company bids on a big contract, the owner demands a bid bond — typically **1%–5%** of the bid value. If the bidder wins but then refuses to sign the contract, the bond pays the owner. It stops frivolous bidding.
- **Performance bond (performance guarantee).** Once the contract is signed, the contractor posts a performance bond — typically **5%–10%** of the contract value. If the contractor abandons the job or builds it wrong, the bond pays the owner the money to fix it.
- **Advance-payment guarantee.** If the owner pays the contractor money up front (say a 20% mobilization advance to buy materials), the owner demands a guarantee for that amount. If the contractor takes the cash and vanishes, the guarantee returns it.
- **Standby letter of credit (SBLC).** The US-flavored, all-purpose version. It pays the beneficiary if the bank's customer defaults on essentially any specified obligation — paying an invoice, completing a project, repaying a loan. It is "standing by" in case of default.

![Guarantee and standby bond types and the failure that triggers each one](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-8.png)

The economically important feature of most international guarantees is that they are **on-demand** (also "first-demand"). The beneficiary can call the guarantee by simply *stating* that the customer failed, often without proving it. This makes the guarantee powerful and dangerous: it pays first and argues later. A contractor in a dispute can watch the owner pull the full performance bond on a contested claim, and only then sue to get it back. The reverse risk — an **unfair calling** — is real, which is why contractors fight hard over guarantee wording and why some jurisdictions (and the URDG 758 rulebook the ICC publishes) try to constrain abusive demands.

There is a second axis that matters enormously to the contractor: **on-demand vs conditional**. An *on-demand* guarantee pays the beneficiary on a bare written statement of default, no proof required — fast, beneficiary-friendly, and the global default for cross-border work. A *conditional* (or "suretyship") guarantee pays only after the beneficiary actually proves the loss, often through a court or an independent expert. Contractors vastly prefer conditional guarantees because they can't be drained on a contested claim; owners prefer on-demand because they get cash immediately. The choice is a pure allocation of dispute risk: with on-demand, the contractor sues *after* the money has already moved (an uphill fight); with conditional, the owner must establish the loss *before* a cent leaves the bank. Which one a contract uses is often the single most negotiated line in the whole guarantee, because it decides who holds the cash while the lawyers argue.

For the bank, a guarantee is a *contingent liability*: it has promised money it will probably never pay, so it earns a fee for carrying the risk without funding it. A guarantee fee typically runs **0.5%–2% per year** of the guaranteed amount, depending on the customer's credit and the tenor. It's a high-return product because, statistically, most guarantees expire unused — the bank collects fees on a promise that rarely costs it anything.

#### Worked example: a performance bond being called

A contractor wins a \$10 million infrastructure contract and posts a **10% performance bond = \$1,000,000**, issued by its bank to the project owner. Two scenarios:

- **The normal case (95%+ of the time).** The contractor finishes the job. The bond expires unused. The contractor paid a fee of, say, **1.5% per year** on \$1,000,000 = **\$15,000/year**. For a two-year project, \$30,000 total — the price of being allowed to bid at all. The bank earned \$30,000 and paid out nothing.
- **The bad case.** Eighteen months in, the contractor goes bankrupt and walks off the site, 70% complete. The owner calls the bond. The bank pays the owner the full **\$1,000,000** within days (on-demand). The bank now has a \$1,000,000 *claim against the bankrupt contractor* — and stands in line with every other creditor, likely recovering cents on the dollar. This is why the bank required the contractor to post collateral or have strong credit before issuing the bond: when a guarantee is called, the bank's "fee business" instantly becomes a "lending business" with a defaulted borrower.

The intuition: a guarantee is a bet by the bank that its customer will perform. The fee is the premium. When the bet is lost — the guarantee is *called* — the bank's contingent liability converts into a real loan to a counterparty who just defaulted. The whole risk discipline of trade finance is making sure that conversion almost never happens, and that when it does, there's collateral behind it.

## Supply-chain finance: lending against the invoice

Now we move from de-risking the *shipment* to financing the *gap between shipping and getting paid*. This is where the biggest money — and the most modern growth — in trade finance lives.

The core problem: even after goods ship safely, the seller often has to wait. Large buyers impose long payment terms — 60, 90, even 120 days — because stretching payables is free working capital for them. But for the supplier, a 90-day wait is a 90-day hole in its cash flow. It shipped the goods, paid its workers and materials, and now has a \$200,000 *receivable* (an IOU) that it can't spend for three months. Supply-chain finance turns that frozen IOU into cash today.

There are two ways to do it, and the difference — who initiates, and whose credit prices the deal — is everything. The pipeline figure contrasts them.

![Factoring initiated by the seller versus reverse factoring initiated by the buyer](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-6.png)

### Factoring: the seller sells its invoices

**Factoring** is seller-driven. The supplier has a pile of unpaid invoices and sells them to a *factor* (a bank or specialist) for cash now, at a discount. The factor then collects the full amount from the buyer when it falls due, keeping the discount as its return. Two sub-types:

- **Recourse factoring.** If the buyer never pays, the factor can claw the money back from the *seller*. The seller keeps the credit risk; the factor is really just lending against the invoice. Cheaper.
- **Non-recourse factoring.** If the buyer never pays (because it went bankrupt), the loss is the *factor's*. The seller has truly sold the risk away. More expensive — the extra cost is essentially credit insurance.

Crucially, the price of factoring is set mostly by the **buyer's** creditworthiness, because the buyer is who actually pays the invoice. A small supplier with weak credit can still factor cheaply *if* its invoices are owed by a blue-chip buyer. The supplier is, in effect, borrowing against someone else's good name.

#### Worked example: the discount and the effective cost of factoring

Mekong Furniture has a \$200,000 invoice owed by a creditworthy buyer, payable in **90 days**, but Mekong needs cash now to fund its next production run. It factors the invoice. The factor offers:

- An **advance rate** of 90% — it pays \$180,000 today and holds back a 10% reserve (\$20,000).
- A **discount fee** of 2% of face value = \$4,000.
- When the buyer pays the full \$200,000 in 90 days, the factor returns the \$20,000 reserve minus the \$4,000 fee, so Mekong ultimately nets \$200,000 − \$4,000 = **\$196,000**.

What did the cash *cost*? Mekong gave up \$4,000 to get \$180,000–\$196,000 about 90 days early. Take the fee on the advanced amount: \$4,000 / \$180,000 = 2.22% for 90 days. Annualize it: 2.22% × (365 / 90) ≈ **9.0% per year**. That is the *effective annual cost* of the financing.

Is 9% a lot? If Mekong's only alternative is an unsecured working-capital loan at 12%, factoring is cheaper *and* it cleans up the balance sheet (the receivable is gone). If Mekong could borrow at 6% from its own bank, factoring is the more expensive option, justified only by speed or by offloading the credit risk. The intuition: factoring is a loan dressed up as a sale, and its true price is the discount fee *annualized over how early you got the cash* — always compute the annualized rate, because a "2% fee" on a 30-day invoice is an eye-watering 24% per year.

### Reverse factoring: the buyer rescues the supplier

**Reverse factoring** flips the initiator. Here a large, financially strong **buyer** sets up a program with its bank: any of the buyer's *approved* invoices can be paid early to the supplier by the bank, at a financing rate based on the **buyer's** (excellent) credit, not the supplier's (weaker) credit. The buyer still pays the bank on the original due date.

Why does everyone like this?

- **The supplier** gets paid early, at a rate far lower than it could ever borrow on its own — because the bank is taking the *buyer's* blue-chip risk, not the supplier's. A supplier who'd borrow at 12% on its own might get financed at 5% through a strong buyer's program.
- **The buyer** can *extend* its own payment terms (say from 30 to 90 days), keeping the cash longer as free working capital, *while its suppliers are no worse off* because they can take the early-payment option. It looks like generosity; it's working-capital optimization.
- **The bank** lends against a single, well-understood, investment-grade buyer instead of underwriting hundreds of small suppliers one by one. Low risk, scalable, sticky.

#### Worked example: reverse factoring saving the supplier money

A small supplier, "Delta Components," ships \$100,000 of parts to a large investment-grade manufacturer on 90-day terms. Delta is cash-tight and would have to borrow at **12% per year** on its own.

- **Without reverse factoring,** Delta needs the \$100,000 for 90 days, so it borrows: 12% × (90/365) × \$100,000 ≈ **\$2,959** in interest. Delta's effective cost of waiting: ~\$2,960.
- **With the manufacturer's reverse-factoring program,** the bank pays Delta early at a rate tied to the *manufacturer's* credit, say **5% per year**, minus a small fee. Delta's cost for the same 90 days: 5% × (90/365) × \$100,000 ≈ **\$1,233**.

Delta saves about **\$2,959 − \$1,233 = \$1,726** on a single \$100,000 invoice — a ~58% reduction in financing cost — *purely because the bank is pricing off the big buyer's credit instead of Delta's*. Now scale that across hundreds of suppliers and dozens of invoices a year, and reverse factoring becomes a genuine competitive advantage for a supply chain. The intuition: reverse factoring is a mechanism for a strong company to *lend its low borrowing cost to its weaker suppliers*, with a bank in the middle taking a thin, safe spread. It is the trust-renting business of trade finance, scaled to an entire supply chain.

### Forfaiting and the longer-horizon cousins

Factoring handles the short, revolving stream of trade receivables — invoices due in 30, 60, 90 days. For *big-ticket, long-dated* deals — exporting \$20 million of turbines to be paid over five years — the analogous product is **forfaiting**. A forfaiter (a bank or specialist) buys the medium-to-long-term receivable from the exporter outright, **without recourse**, usually backed by a bank guarantee (an *aval*) from the importer's bank. The exporter gets the full present value in cash and walks away with zero credit, country, or currency risk; the forfaiter holds the paper to maturity or trades it on. Forfaiting is what lets a manufacturer in one country sell capital equipment on multi-year credit to a buyer in a risky country *without* carrying that risk on its own books for years — the export turns into a clean cash sale, and a bank absorbs the long tail.

Sitting above all of this is **export credit insurance** and the world's **export credit agencies (ECAs)** — government-backed bodies like the US EXIM Bank, UK Export Finance, or Euler Hermes in Germany. They insure exporters and banks against the political and commercial risk of foreign buyers defaulting, which is what makes a private bank willing to finance a sale into a volatile country at all. The pattern recurs at every layer: the original parties don't trust each other, so a bank steps in; the bank doesn't want to carry the long-dated foreign risk, so an insurer or a government agency steps in behind it. Trust is *laddered* — passed up a chain of ever-stronger balance sheets until it lands somewhere it can sit safely.

A caution, because finance is never free of risk: reverse-factoring programs can be *abused*. If a buyer uses them to stretch payment terms aggressively and books the financing in a way that hides growing supplier debt, it can mask a deteriorating balance sheet. The 2018 collapse of UK outsourcer **Carillion** and the 2021 implosion of **Greensill Capital** both involved supply-chain finance that had quietly become a giant, opaque borrowing machine. The mechanism is sound; like all leverage, it is dangerous when it's hidden.

## How trade finance fits the bank's economics

Step back and see why banks love this business. The data chart below shows where a representative bank's fees land — and trade finance is concentrated in the fee-rich, capital-light corner.

![Bank fee revenue from a letter of credit rises with shipment value](/imgs/blogs/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance-7.png)

A commercial loan ties up the bank's scarce equity capital and earns a *spread* over years. A letter of credit or guarantee, by contrast, is a **contingent** commitment — until a document is presented or a guarantee is called, the bank has not lent a cent. It earns a fee on its *reputation*, the cheapest input it owns. The chart shows the fee scaling roughly linearly with shipment size (the ~0.75% issuance fee per quarter): a \$1,000,000 shipment generates ~\$7,500, a \$5,000,000 shipment ~\$37,500, all on a promise the bank usually never has to fund.

There's a subtlety the spine of this series demands we flag. Trade finance *looks* off-balance-sheet and riskless, but it is not. A confirmed LC is a real obligation; a called guarantee becomes a real loan to a defaulted counterparty; a usance LC the bank *discounts* (pays the seller early and waits to be reimbursed) is real lending. Basel rules force banks to hold capital against these contingent exposures via a *credit conversion factor* — turning, say, 20%–50% of the face value into risk-weighted assets. The 2008 freeze was exactly the moment banks discovered that their "safe" trade book was a web of cross-bank promises that could all go bad at once if a major issuing bank failed. Confidence is the raw material, and confidence is exactly what evaporates in a crisis.

#### Worked example: the bank's return on a trade-finance commitment

A bank issues a \$1,000,000 standby LC for a corporate client, fee **1.5% per year = \$15,000/year**. Under Basel, a standby LC carries a credit conversion factor (CCF); for a performance-type standby, roughly **50%**, so the off-balance-sheet \$1,000,000 becomes \$500,000 of credit exposure. If the client risk-weights at, say, 100%, that's **\$500,000 of risk-weighted assets (RWA)**. At an 8% equity requirement (the realistic base case for a commercial bank, implying about 12.5× leverage), the bank must hold about **\$40,000 of equity** against this commitment.

Return on that equity: \$15,000 fee / \$40,000 equity = **37.5%** — *if* the standby is never called. That eye-popping return on a tiny equity sliver is exactly why trade finance is so prized: it monetizes the bank's name with very little capital tied up. The intuition: trade finance is one of the highest return-on-equity activities a bank runs *as long as nothing is ever called* — and the entire art of the trade-finance desk is making sure that holds, because a single called guarantee on a defaulted client can wipe out years of fees.

## Common misconceptions

**"A letter of credit guarantees I'll receive good goods."** No. An LC pays against *documents*, not goods. If the seller ships a container of bricks but presents a flawless bill of lading and invoice, the bank must pay, and the buyer's recourse is to sue the seller for fraud — separately, after the money is gone. The LC removes *non-payment* risk, not *non-performance* or *fraud* risk. This "independence principle" is the LC's strength (banks don't need to know the goods) and its trap (a buyer must still vet the seller and insist on an independent **inspection certificate** as a required document).

**"An advising bank will pay me."** No. The advising bank only authenticates the LC. Only the *issuing* bank (and a *confirming* bank, if one exists) promises payment. A seller who relaxes because "my local bank handled the LC" has misread the document — that bank pays nothing unless it added a confirmation. The fix is to *get the LC confirmed* if you don't trust the issuing bank or its country.

**"Documentary collection is just a cheap letter of credit."** No — it's a fundamentally weaker instrument. In a collection *no bank promises to pay*; the banks are couriers holding documents hostage. If the buyer walks away from a D/P shipment, the seller is stuck with goods in a foreign port. The price difference (a flat fee vs ~1%) exactly reflects the difference in risk transfer: you pay less because you get less protection.

**"A discrepancy is a technicality the bank will overlook."** No — and this one costs exporters real money. Banks examine LC documents with brutal literalism: a misspelled company name, a bill of lading dated one day after the shipment window, an amount that's \$200,000.00 on the invoice but \$200,000 on the LC — any *discrepancy* lets the issuing bank lawfully refuse to pay until it's fixed or the buyer waives it. Industry studies have long found that **well over half** of first-presentation LC documents are rejected for discrepancies. The seller's protection is only as good as its paperwork; a "guaranteed" payment can stall for weeks over a comma.

**"Factoring and reverse factoring are the same thing in reverse."** Only mechanically. The economic substance differs: factoring is the *seller* selling its own risk (priced off the buyer's credit because the seller is desperate); reverse factoring is the *buyer* extending its own *cheap* credit to lift its suppliers (priced off the buyer's strong name, often with the buyer benefiting via longer terms). One is a small firm fixing a cash crunch; the other is a large firm optimizing its working capital. Conflating them hides who's really benefiting — and who's really on the hook if the program is misused.

## How it shows up in real banks

**The 2008 trade-finance freeze.** As described in the opening, the most visceral example of trade finance's importance was its *absence*. When interbank trust collapsed in late 2008, issuing banks couldn't be sure confirming banks were solvent, and vice versa, so LCs stopped flowing. The World Trade Organization and the G20 scrambled in 2009 to set up trade-finance support programs precisely because they recognized that a credit freeze in the banking layer was strangling *physical* trade in goods. The lesson: trade is built on a chain of bank promises, and the chain is only as strong as confidence in its weakest bank — the same fragility that defines the whole institution.

**Greensill Capital, 2021.** Greensill built a multi-billion-dollar business on reverse factoring (it preferred the term "supply-chain finance"), packaging the early-payment receivables into investment funds. The model worked until it didn't: too much exposure was concentrated in a few clients (notably the GFG Alliance steel group), the receivables included "prospective" invoices for trades that hadn't happened, and when the credit insurance behind the funds was pulled, the whole structure collapsed, taking Credit Suisse's \$10 billion supply-chain funds down with it. The mechanism — financing real invoices off a strong buyer's credit — is sound. Greensill's sin was financing *unreal* invoices and hiding concentration. It is the clearest modern warning that supply-chain finance is leverage, and hidden leverage breaks.

**Carillion, 2018.** The UK construction and outsourcing giant used reverse factoring (its "Early Payment Facility") so heavily that roughly £500 million of what was economically *debt to banks* sat on its books looking like ordinary trade payables. When Carillion collapsed, analysts and a parliamentary inquiry concluded the program had masked the true depth of its borrowing for years. The episode pushed accounting standard-setters to demand far clearer disclosure of supply-chain finance — proof that an instrument designed to *reduce* risk can amplify it when it's used to hide leverage.

**The standby LC as the backbone of US corporate guarantees.** Because US banks were historically restricted from issuing pure guarantees, the **standby letter of credit** evolved to fill the role and is now ubiquitous: it backs commercial-paper programs, insurance obligations, construction contracts, and lease deposits. A landlord leasing a floor to a young company often demands an SBLC instead of cash, so the tenant's bank — not the tenant — stands behind the lease. The total face value of standby LCs outstanding at US banks runs into the hundreds of billions of dollars, almost all of it a promise that will quietly expire unused. It is the purest expression of the trade-finance idea: a bank renting its name so a deal between two parties who don't fully trust each other can happen.

**Discrepancy economics at scale.** A large trade-finance operations center processes thousands of document presentations a day, and the discrepancy rate is the silent killer of the business's reputation. Banks invest heavily in document-checking software and experienced examiners because a wrongly-paid discrepant document (the bank pays when it shouldn't have, and the buyer refuses to reimburse) is a direct loss, while a wrongly-rejected clean document infuriates a valuable client. The whole back-office discipline of trade finance is, in the end, about getting paper exactly right — because in this business the paper *is* the goods.

## The takeaway / How to use this

The durable mental model is this: **trade finance is the business of renting out a bank's creditworthiness so that two parties who can't trust each other can transact anyway.** Everything in this post is a variation on that one move.

- A **letter of credit** rents the bank's promise to the *seller*, so the seller will ship before being paid.
- A **confirmation** rents a *second* bank's promise, so the seller doesn't have to trust a risky foreign issuing bank or its country.
- A **documentary collection** rents only the bank's *neutrality as a document-holder* — cheaper, weaker, no promise to pay.
- A **guarantee or standby LC** rents the bank's promise to *cover a failure*, so a deal that needs a safety net can proceed and usually never tests the net.
- **Factoring and reverse factoring** rent the bank's *balance sheet against an invoice*, converting a frozen IOU into cash and pricing it off whoever has the better credit.

If you ever read a company's annual report and see large "off-balance-sheet commitments," "contingent liabilities," or "supply-chain finance arrangements," you now know what they are and why they matter: promises that earn fat fees on thin capital while nothing goes wrong, and that convert into real, sometimes ruinous, exposure when something does. That is the same fragile, leveraged, confidence-funded trade that defines a bank itself — only here the bank is lending its *trust* instead of its *money*. The fee is the price of that trust; the risk is what happens when the trust is misplaced and the promise is finally called.

For a reader: when you next buy something that crossed an ocean to reach you — coffee, a phone, a car part — there was almost certainly a bank somewhere standing in the gap between a seller and a buyer who never met, turning mutual suspicion into a clean transaction for about one or two percent. That quiet, capital-light, trust-renting business is one of the oldest and most profitable things a bank does, and one of the first to vanish when confidence breaks.

## Further reading & cross-links

- [The payments business: how money actually moves between banks](/blog/trading/banking/cross-border-payments-correspondent-banking-and-how-swift-really-works) — the SWIFT messaging and correspondent-banking plumbing that carries the LC and the cross-border payment leg.
- [Corporate and commercial lending: term loans, revolvers, and syndication](/blog/trading/banking/corporate-and-commercial-lending-term-loans-revolvers-and-syndication) — the funded-lending side that a called guarantee or a discounted usance LC quietly turns into.
- [Cash management and transaction banking for corporates](/blog/trading/banking/cash-management-and-transaction-banking-for-corporates) — the broader sticky, fee-rich transaction-banking franchise that trade finance sits inside.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — how fee-based, capital-light businesses fit a bank's overall revenue mix.
- [Corporate bonds: lending to companies](/blog/trading/fixed-income/corporate-bonds-lending-to-companies) — the credit-risk lens (default, recovery, recourse) that underlies every guarantee and factoring decision.
