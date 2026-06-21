---
title: "Fintech vs Banks: Disruption, Partnership, or Quiet Absorption"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why fintechs won the interface — payments, lending UX, FX, brokerage — but banks kept the balance sheet, and why most fintechs end up partnering with, or absorbed by, the banks they set out to replace."
tags: ["banking", "fintech", "bnpl", "neobanks", "banking-as-a-service", "embedded-finance", "payments", "deposit-franchise", "bank-charter", "disruption"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Fintechs won the interface; banks kept the balance sheet. The slick app, the one-tap checkout, the zero-fee FX, the buy-now-pay-later button — those were the parts of banking that were easy to make better, and fintechs made them dramatically better. But the parts that actually make money and carry risk — holding insured deposits, owning a charter, lending off a regulated balance sheet, being trusted with people's savings — stayed with the banks. That is why most fintechs end up renting a bank's charter, partnering with a bank, or being bought by one.
>
> - Fintech genuinely disrupted the **experience** of payments, lending origination, FX, and brokerage. It barely dented the **economics** underneath, which still run through a chartered bank's balance sheet.
> - A fintech without deposits funds itself near the policy rate. A bank funds about **71%** of its book with deposits at well under 1%, so its blended cost of funds is around **1.9%** while a deposit-less fintech pays around **5.5%**. That 3.6-point gap is the whole game.
> - Buy-now-pay-later looked like free money — a flat merchant fee, no interest charged to the shopper. Then the loss rate climbed through a downturn, the fee stayed flat, and the math flipped from profit to loss.
> - The one number to remember: a bank's deposit franchise lets it fund at roughly **1.9%** against a fintech's ~5.5%. Until a fintech solves that, it does not have a bank — it has a front end for one.

In 2021, it genuinely looked like the banks were finished. A neobank called Chime had more checking accounts than most US regional banks. A payments company called Stripe was, on paper, worth more than Goldman Sachs. A buy-now-pay-later firm called Klarna was valued at \$46 billion. Robinhood had made stock trading feel like a video game and pulled in millions of young accounts. The narrative was everywhere and it was tidy: software was eating banking, the dinosaurs were too slow, and a generation that would "never set foot in a branch" was going to bank entirely through apps built by people in hoodies.

Then 2022 and 2023 happened. Interest rates went up. Klarna's valuation fell roughly 85%, to about \$6.7 billion, in a single funding round. Dozens of banking-as-a-service partnerships blew up over compliance failures, and one of the plumbing companies in the middle, Synapse, collapsed in 2024 and left ordinary people locked out of their own money for months. Meanwhile, when Silicon Valley Bank failed in March 2023, where did the panicked deposits run *to*? Not to a fintech. They ran to JPMorgan — the biggest, oldest, most boringly regulated bank in America. The "dinosaur" took in tens of billions of dollars in a weekend.

The figure below is the mental model for the whole post, and it is the entire argument in one picture: function by function, fintech captured the interface — the screen, the experience, the brand — while the bank kept the balance sheet — the deposits, the charter, the capital, the trust. Hold that split in your head and almost everything about the fintech-versus-banks story stops being confusing.

![Matrix of banking functions showing fintech won the interface and banks kept the balance sheet](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-1.png)

This is the spine of the whole series, viewed from a new angle. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short (deposits), lends long (loans), earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive. Fintech attacked that machine from the outside — and discovered that the easy, visible, profitable-looking parts (the interface) were not where the machine's power actually lived. The power lived in the boring middle: a license to hold insured money cheaply, and the trust that keeps that money from running. You cannot disrupt that with a nicer app. You can only rent it, partner with it, or be bought by it.

## Foundations: fintech, disruption, the interface, the balance sheet, and the two moats

Let's build every term from zero. By the end of this section you will know what "fintech" actually means, the difference between genuinely disrupting a business and just improving it, what we mean by "the interface" versus "the balance sheet," what buy-now-pay-later is, and the two moats — the deposit moat and the regulatory moat — that explain why the banks held.

### What "fintech" actually means

*Fintech* is just short for "financial technology," and like most buzzwords it has been stretched to mean almost anything. For our purposes, a fintech is **a technology company that delivers a financial service — payments, lending, saving, investing, currency exchange — usually through software, usually without holding a banking license itself.** That last clause is the load-bearing one. The defining feature of the classic fintech wave (roughly 2010 to 2021) is that these were *not banks*. They were companies that built a beautiful front door onto financial services and, behind that door, quietly relied on an actual licensed bank to do the regulated, money-holding part.

A *neobank* (also called a challenger bank) is a fintech that offers what looks like a full bank account — a debit card, a balance, direct deposit of your paycheck — through an app, with no branches. Examples include Chime and Cash App in the US, Revolut and Monzo and N26 in Europe. We cover their unit economics in detail in the sibling post [digital banking and the neobank business model](/blog/trading/banking/digital-banking-and-the-neobank-business-model); here, the key fact is that most neobanks, especially the American ones, do not actually hold a banking license. They sit on top of a partner bank.

### Disruption vs sustaining innovation — the distinction that decides everything

The word "disruption" gets thrown at any new app, but it has a precise meaning, and it is the single most useful idea for understanding this whole story. The distinction comes from Clayton Christensen, a Harvard professor who studied why big, well-run companies get killed by smaller upstarts.

A *sustaining innovation* makes an existing product better for the customers a business already has. A faster processor in a laptop, a higher-resolution camera in a phone — these are sustaining. Incumbents are usually very good at sustaining innovations, because they have the customers, the engineers, and the incentive to keep improving.

A *disruptive innovation* is different. It starts at the bottom of the market — cheaper, simpler, often initially worse on the metrics the incumbent cares about — serving customers the incumbent ignores or overcharges. Then it gets better over time and climbs upward, until one day it is good enough to take the incumbent's best customers too. The reason incumbents lose to disruption is not stupidity; it is that the disruptor enters through a door the incumbent has no reason to defend.

Here is the crucial point for banking. **Fintech was, mostly, a sustaining innovation dressed up as a disruptive one.** It made the *experience* of using financial services much better — that is real and valuable. But it did not change the underlying *cost structure or business model* of finance, because it could not get rid of the two things that actually make a bank a bank: the deposit franchise and the regulatory license. A truly disruptive entrant would have found a way to do banking *without* those. Almost none did. They improved the front door and then knocked on the bank's back door asking to use its balance sheet.

### The interface vs the balance sheet — the central frame

This is the frame the whole post hangs on, so let's nail it down with a clean analogy and then make it precise.

Think of a financial product as a two-layer cake. The top layer is the **interface**: everything the customer sees and touches. The app, the brand, the signup flow, the one-tap checkout, the notifications, the customer support chat, the marketing. This layer is about *experience* and *acquisition* — getting a customer and making them happy.

The bottom layer is the **balance sheet**: everything the customer does not see. Where the money actually sits, who legally owes it back, who is on the hook if a loan defaults, who holds the regulatory capital, who answers to the regulator. This layer is about *funding, risk, and license* — and it is where almost all the money and almost all the danger live.

A *balance sheet*, recall from the series, is just the two-sided list of what an entity owns (assets — its loans and securities) and what it owes (liabilities — its deposits and borrowings), with the gap between them being equity. To "have a balance sheet" in banking means you actually hold the money: the deposit is your liability, the loan is your asset, and you carry the consequences of both. The figure below splits the value chain along exactly this line.

![Before and after columns showing fintech owns the interface layers and the bank owns the balance sheet layers](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-2.png)

The whole thesis of this post fits in one sentence: **fintech won the top layer of the cake and banks kept the bottom layer — and the bottom layer is where the money is.**

### What buy-now-pay-later (BNPL) is

Because BNPL is the most-hyped piece of fintech lending and we will use it as a running case, define it now. *Buy-now-pay-later* is a short-term installment loan offered at the checkout: you buy a \$200 jacket and, instead of paying \$200 today, you pay four installments of \$50, typically over six weeks, with no interest charged to you. It feels free to the shopper. So who pays? The merchant. The BNPL provider charges the store a fee — often 4% to 6% of the purchase — far more than a normal card's fee, because BNPL reliably lifts how much people buy. The provider keeps that merchant fee and takes the risk that you do not pay your installments. We will see later why "feels free, merchant pays" is a business that looks wonderful in good times and turns ugly in a downturn.

### The deposit moat — why cheap money is the franchise

Now the first of the two moats. A *moat*, borrowing Warren Buffett's term, is a durable advantage that protects a business from competitors. A bank's first moat is its **deposit franchise**: the pile of cheap, sticky money that ordinary people and businesses leave in checking and savings accounts, paying the bank almost nothing.

Why is this a moat and not just a feature a fintech could copy? Because deposits are *insured* (in the US, up to \$250,000 per depositor per bank by the FDIC), and only a licensed bank can take insured deposits. That insurance is why your money stays put even when you are nervous — and that stickiness is what lets a bank pay you 0.5% while earning 5% on the money. A fintech with no license cannot take an insured deposit. It can hold your "balance," but legally that balance is sitting in a real bank somewhere, and the real bank is capturing the cheap-funding advantage. We go deep on this in [retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise). For now, hold this: cheap insured deposits are the cheapest funding in all of finance, and you can only get them with a charter.

### The regulatory moat — the charter and the compliance machine

The second moat is the **regulatory moat**, and it has two parts. The first is the *charter* — the banking license itself. A charter is a permission, granted by a government, to take deposits and make loans as a bank. It is extraordinarily hard to get: it takes years, a mountain of capital, and a willingness to be examined and second-guessed by regulators forever after. In the US, only a tiny handful of genuinely new bank charters have been granted in any recent year.

The second part is the *compliance machine* — the expensive, unglamorous apparatus a licensed bank must run: know-your-customer (KYC) checks to verify who its customers are, anti-money-laundering (AML) monitoring to catch dirty money, sanctions screening, capital and liquidity reporting, and regular supervisory exams. This is a moat precisely because it is a cost and a hassle. Fintechs entered banking *to avoid* this machine — that was a lot of their speed advantage. But you cannot hold regulated money without it, which means the moment a fintech wants to touch the balance sheet, it has to either build the machine (slow, costly) or rent a bank that already has one. We treat the API-and-platform side of this in [open banking, banking-as-a-service, and embedded finance](/blog/trading/banking/open-banking-apis-banking-as-a-service-and-embedded-finance).

### A short history: unbundling, then rebundling

One more piece of context makes the whole arc legible. For most of the twentieth century, a bank was a *bundle*: it took your deposits, ran your payments, gave you a loan, sold you a mortgage, held your savings, and converted your currency — all under one roof, the branch. The bundle was convenient and the bank made money on every piece, including the pieces it did badly, because you had nowhere else to go.

Fintech's first act, roughly 2010 to 2021, was *unbundling*. Each app took one slice of the bundle and did it brilliantly: Venmo took payments, Robinhood took trading, Wise took FX, Affirm took point-of-sale lending, Chime took the checking account. The pitch was always the same — "the banks do ten things badly; we do one thing perfectly." And for the customer it worked: the experience of each slice got dramatically better.

Fintech's second act, from roughly 2022 onward, has been *rebundling* — and the surprise is which direction it ran. The fintechs, having unbundled the bank, tried to bundle back up (a neobank adding lending, a payments app adding savings) and discovered that every new slice they added pulled them toward the balance sheet, the charter, and the wall. Meanwhile the banks, stung and now awake, rebundled the *good* fintech ideas back into their own apps. The bundle is reassembling — but the reassembly happens on top of a balance sheet, which means it happens on the banks' terms, whether the front end says "bank" or says "app." Keep this arc in mind: the rest of the post is the story of the second act.

With those pieces defined — fintech, disruption, interface vs balance sheet, BNPL, the deposit moat, the regulatory moat, and the unbundle-then-rebundle arc — we can now tell the real story.

## Where fintech actually won

Let's give fintech its due, because the wins were real and large. In four areas, fintechs took something banks did badly and made it dramatically better. Notice the pattern as we go: every win is an *interface* win.

### Payments: the slick front end on someone else's rails

Payments is where fintech's wins are most visible. Before fintech, paying a friend back meant cash, a check, or a clunky bank transfer that took days. Paying online meant typing a 16-digit card number into a form that broke half the time. Accepting card payments as a small merchant meant a frightening contract with a "merchant acquirer" and a card reader that cost hundreds of dollars.

Fintech fixed all of that at the interface. Venmo and Cash App made paying a friend a one-tap social act. Stripe turned "accept payments online" from a months-long integration into seven lines of code. Square (now Block) put a card reader in the pocket of every food-truck owner and farmers-market seller. PayPal made checkout a single login. These are genuine, enormous improvements, and they captured real economics — Stripe and Block are large, valuable companies.

But look underneath. None of these companies replaced the payment *rails* — the underlying networks that actually move money between banks (the card networks, the ACH batch system, the real-time settlement systems). When you Venmo a friend, the money still moves over bank rails; Venmo is a beautiful skin on top. When Stripe processes a card payment, the interchange fee still flows to a card-issuing *bank*. We cover those rails in the series posts on the [cards business](/blog/trading/banking/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split) and domestic payment rails. The fintech took the interface and the customer relationship. The bank kept the rail and a slice of every transaction. Fintech won the checkout; it did not win the clearing.

### Lending UX and BNPL: fast origination, someone else's risk

Lending is the second arena. Banks were slow and opaque at lending. A small-business loan could take weeks of paperwork; a personal loan meant a branch visit; getting approved felt like pleading. Fintechs like SoFi, Upstart, and Affirm made *origination* — the process of finding a borrower, deciding to lend, and disbursing the money — fast, online, and friendly. Affirm and Klarna built BNPL into the checkout so smoothly that "loan" never enters the customer's mind. We cover the underlying products in [consumer lending: mortgages, cards, auto, and personal loans](/blog/trading/banking/consumer-lending-mortgages-cards-auto-and-personal-loans).

This is a real interface win. But ask the balance-sheet question: who *funds* the loan, and who eats the loss if it defaults? In the early "marketplace lending" model, the fintech (like the original LendingClub) just matched borrowers to investors and took a fee — it held nothing. When fintechs wanted to keep the loans, they discovered they needed funding, and funding without deposits is expensive. So they either sold the loans to banks and funds, or partnered with a bank to originate them. The fintech captured the *origination experience*; the *funding and the risk* landed on a balance sheet that was, more often than not, a bank's.

### FX and remittances: crushing a fat, lazy fee

The third clear win, and arguably fintech's cleanest, is foreign exchange and cross-border money transfer. Banks charged outrageous, hidden markups on currency conversion — you would "pay no fee" but get an exchange rate 3% to 5% worse than the real one, and the bank pocketed the difference. Sending money to family abroad through Western Union could cost 7% or more.

Wise (formerly TransferWise) and Revolut blew this up by showing customers the real mid-market rate and charging a small, transparent fee. The savings for ordinary people were huge. Here, fintech did not just win the interface — it genuinely destroyed an economic rent the banks had been collecting lazily for decades. This is the one place where fintech took real money *off the banks' table*, not just the experience. The bank did not "keep the balance sheet" here in any comforting way; it simply lost a fee it should never have been charging.

### Embedded finance: putting the bank inside someone else's app

There is a fourth-and-a-half win worth naming on its own, because it is where the most recent fintech energy went: *embedded finance*. The idea is that any company — a ride-hailing app, an e-commerce platform, an accounting tool — can offer financial products (a payment, a loan, a debit card, insurance) right inside its own app, without sending the customer off to a bank. The driver gets paid instantly inside the driver app; the small merchant gets a working-capital loan inside the platform it already sells on; the software tool offers its users a branded card.

This is a genuine interface win, and a clever one, because it puts the financial product at the exact moment of need — the *point of context* — where it converts far better than a separate bank visit ever could. Shopify Capital lending money to a store the moment it sees that store's sales is a much better offer than a cold bank loan application, because Shopify already knows the store's revenue down to the dollar. But ask the balance-sheet question one more time and the answer is identical to every other win: the loan that Shopify originates is funded and held by a partner — historically a bank or a credit fund. The platform owns the context and the customer; the balance sheet sits underneath, with someone who has a charter and capital. Embedded finance is the interface win taken to its logical extreme — the bank vanishes entirely from view — and it changes *nothing* about who funds the loan and carries the risk. The more invisible the bank becomes, the more completely it still owns the part that matters.

### Brokerage UX: making investing feel like an app

Fourth, stock trading. Robinhood made buying a share feel like sending a text — no commission, no minimum, a clean app. It pulled in a generation of new investors. Commission-free trading then forced the entire industry — Schwab, Fidelity, every incumbent — to drop commissions too. That is a real, lasting change to the customer experience and to industry pricing.

But again, look beneath. Robinhood does not custody most of its own securities settlement infrastructure from scratch; the plumbing of clearing and custody still runs through regulated institutions, and a meaningful chunk of Robinhood's revenue came from "payment for order flow" — selling its customers' orders to market-makers — and from interest on customer cash, which depends on, you guessed it, parking that cash somewhere it earns the policy rate. The interface was revolutionary. The underlying market structure was not replaced.

#### Worked example: a fintech's cost of funds without a deposit franchise

Here is the single most important number in this whole story, worked out in dollars. Suppose two companies both want to lend \$1,000 to a borrower at a 7% interest rate. One is a bank; one is a deposit-less fintech.

The bank funds itself the way every bank in this series does. From the data, a typical large bank's funding mix is about 71% deposits, with the rest a blend of wholesale borrowing, long-term debt, and equity. Say its deposits cost it just 0.5% and the non-deposit ~29% costs it about 5.2% (near the policy rate). Its blended cost of funds is:

$$\text{Bank CoF} = 0.71 \times 0.5\% + 0.29 \times 5.2\% \approx 0.35\% + 1.51\% = 1.86\%$$

So the bank borrows the \$1,000 at about **1.86%** — call it \$18.60 a year — and lends it at 7%, earning \$70. Its gross spread is \$70 − \$18.60 = **\$51.40** per \$1,000.

The fintech has no deposits. It must fund that \$1,000 from venture capital, from selling bonds, or from a warehouse line — all priced near or above the policy rate. Call its cost of funds **5.5%**, or \$55 a year. It lends at the same 7%, earning \$70. Its gross spread is \$70 − \$55 = **\$15** per \$1,000.

The bank makes \$51.40; the fintech makes \$15 — on the *identical loan, at the identical rate, to the identical borrower*. The bank's advantage is not skill; it is the deposit franchise. The intuition: without cheap deposits, a lender is bringing a knife to a gunfight, and no amount of slick UX closes a 3.6-percentage-point funding gap.

![Bar chart comparing a deposit funded bank cost of funds versus a fintech with no deposits](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-3.png)

## Where fintech didn't win — the balance sheet held

Now the other half. For every interface fintech captured, there was a balance-sheet function it could not take. These are the parts that look boring and turn out to be the entire source of durable profit and the entire reason banks survived.

### Deposits stayed with the banks

The deepest non-win is deposits. A fintech can show you a balance in an app, but it almost never *holds* that balance as an insured deposit, because it has no charter. When you put money into a US neobank, that money is "swept" into a partner bank that holds it as an FDIC-insured deposit. The neobank earns a cut, but the *deposit relationship* — and the cheap-funding superpower that comes with it — sits at the bank.

This matters enormously in a panic. Insured deposits are sticky because they are guaranteed; that stickiness is what makes them cheap. A fintech balance that is "kind of like a bank account but technically isn't" is exactly the thing people flee in a scare. We saw this twice over: when SVB failed, money fled *to* the most regulated banks; and when the BaaS plumbing firm Synapse collapsed in 2024, customers of fintechs that relied on it found their money frozen because the question "which bank actually holds this, and is it insured to *this* person?" suddenly had no clean answer. The deposit moat is not a marketing slogan; it is a legal structure that decides who gets to keep your money and who you trust to hold it.

### The regulated lending and the balance sheet held

The second non-win is regulated, balance-sheet lending. Holding a loan to maturity means funding it, reserving capital against it, provisioning for expected losses (under the IFRS 9 / CECL rules covered elsewhere in the series), and answering to a regulator about all of it. That is expensive and slow — and it is exactly what fintechs were built to avoid. So when a fintech actually wants to lend off its own book, it runs straight into the cost of funds wall from the worked example above, plus a capital charge it is not set up to carry.

The result is that the *risky, regulated, capital-intensive* part of lending kept gravitating to balance sheets that were built for it: banks, and increasingly the non-bank credit funds of the shadow-banking world (see [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market)). The fintech kept the friendly origination screen. Someone with a real balance sheet kept the loan and the risk.

### Trust held — the slowest, deepest moat

The third non-win is the hardest to quantify and the most important: trust. People hand a bank their life savings on the belief that the money will be there tomorrow. That belief is built on decades of the bank not failing, on deposit insurance, on the sense that "the regulator is watching this place." A two-year-old app with a clever name does not have that, and it cannot manufacture it with a marketing budget. Trust is earned slowly and lost instantly — which is the entire spine of this series. A bank lives and dies on confidence; an unproven fintech starts with very little of it, and one frozen-funds headline destroys what little it has.

#### Worked example: the deposit-franchise advantage in annual profit

Let's size the deposit moat at the level of a whole institution, in round numbers, to feel how big it is.

Take a bank with \$100 billion of deposits, of which 71% (\$71 billion) is cheap checking-and-savings money costing 0.5%, and the rest is more expensive. Now take a fintech that has acquired the same \$100 billion of customer "balances" but holds none of them as deposits — it just routes them to a partner bank.

On that \$71 billion of cheap deposits, the bank's advantage over funding the same money at the 5.2% policy rate is:

$$\$71\text{bn} \times (5.2\% - 0.5\%) = \$71\text{bn} \times 4.7\% \approx \$3.3\text{ billion per year}$$

That \$3.3 billion is, roughly, the annual value the *deposit franchise alone* throws off — money that exists only because a chartered, insured bank can hold sticky deposits cheaply. The fintech that gathered the same customers captures a referral fee on a sliver of it. The intuition: the deposit franchise is not a line item, it is *the* franchise — and it is structurally unavailable to anything without a charter.

## Why fintechs partner, get absorbed, or fail

So if fintech wins the interface but banks keep the balance sheet, what happens to the fintech? It hits a wall. The wall is the moment the fintech needs the thing it does not have: a balance sheet to hold deposits or loans, a charter to do it legally, cheap funding to compete on price, and a compliance machine to satisfy the regulator. From that wall, there are exactly four exits, and the figure below names them.

![Graph of four outcomes for a fintech that hits the wall disrupt partner get absorbed or fail](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-4.png)

### Exit 1: Disrupt — get your own charter (rare and costly)

The "true disruption" path is to stop renting and become a bank yourself — get a charter, hold your own deposits, build your own compliance machine. A few have tried. SoFi acquired a small bank in 2022 to get a national charter. Varo got a US national bank charter in 2020. In Europe, Revolut spent years and many millions chasing a UK banking license and only secured it (with restrictions) in 2024; Monzo and Starling got theirs earlier.

But notice what "winning" looks like here: the fintech becomes a bank. It takes on the capital requirements, the regulatory exams, the compliance cost, the slow careful culture — everything it was founded to escape. This path is rare because it is brutally expensive and slow, and because the moment you have a charter, your cost advantage over banks largely evaporates: you now *are* one. It is less "disrupting the banks" than "joining them."

### Exit 2: Partner — rent a bank's charter via banking-as-a-service

The most common path by far is to partner: keep being a fintech, and rent a chartered bank's balance sheet through *banking-as-a-service* (BaaS). The fintech provides the app, the brand, and the customer; a partner bank provides the charter, holds the insured deposits, and carries the regulated functions. A layer of middleware companies (Synapse, Unit, Treasury Prime, and others) sits between them, translating the fintech's app calls into the bank's core systems. The figure below traces the flow.

![Pipeline showing a fintech app on top of banking as a service middleware and a partner bank holding the deposit](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-6.png)

This is how most US neobanks operate. Chime is not a bank; it partners with The Bancorp Bank and Stride Bank. Cash App's banking features run through partner banks. The customer thinks they "bank with Chime." Legally, they bank with a small chartered bank in the background, and Chime is the interface. We dig into the mechanics of this layer in [open banking, banking-as-a-service, and embedded finance](/blog/trading/banking/open-banking-apis-banking-as-a-service-and-embedded-finance).

The catch — and the Synapse collapse of 2024 made this brutally concrete — is that the partnership splits responsibility in a way regulators eventually hate. When the middleware fails or the records don't reconcile, the customer's money can be stranded between a fintech that doesn't hold it and a bank that isn't sure which dollars belong to whom. Regulators responded by cracking down hard on BaaS partner banks in 2023 and 2024, issuing enforcement actions and effectively making the "rent-a-charter" model slower and more expensive. The wall got higher.

### Exit 3: Get absorbed — sold to a bank

The third path is acquisition: a bank or a larger incumbent simply buys the fintech, paying for the interface and the customers and bolting them onto its own balance sheet. The bank gets the modern technology and the young customers it struggled to build; the fintech founders get an exit; and the economics quietly move back to where the balance sheet is. Goldman Sachs bought the installment lender GreenSky (and later sold it at a loss, a cautionary tale of its own). Visa tried to buy Plaid for \$5.3 billion (blocked by regulators) and the data-aggregation Plaid stayed independent. JPMorgan, Capital One, and others have made a steady stream of fintech acquisitions. "Quiet absorption" is exactly the right phrase: the disruptor becomes a product line inside the incumbent.

### Exit 4: Fail — run out of cash before solving the wall

The fourth path is simply to fail. Many fintechs were funded on the assumption that customer growth would eventually turn into profit. But customer acquisition is expensive, the unit economics on an interface-only business are thin, and when cheap venture money dried up in 2022, a lot of fintechs discovered they had a beautiful front end and no path to funding the back end. The neobank failure rate is high; many BaaS-dependent fintechs folded when their partner bank or middleware blew up. Failing to clear the wall is the default outcome, not the exception.

#### Worked example: who keeps the spread in a bank-fintech partnership

Let's follow a single dollar of customer deposit through a BaaS partnership and see who keeps what. The headline finding: the partner bank keeps the lion's share, because the partner bank owns the charter and carries the risk.

Suppose a customer parks \$1,000 in a neobank app. The neobank sweeps it to its partner bank, which holds it as an insured deposit and invests it at the 5% policy rate, paying the customer maybe 0.5%. The *gross spread* on that dollar is about 5% − 0% (the neobank often pays the customer almost nothing on the basic balance), but let's use a conservative 5 percentage points of gross spread on the \$1,000, i.e. \$50 a year, and split it the way these deals typically work:

- **Partner bank keeps ~3.4 points** = \$34. It owns the charter, holds the insured deposit, carries the compliance and the regulatory capital. It does the risky, regulated work, so it keeps most of the reward.
- **Fintech gets ~1.2 points** = \$12. This is its referral/program share for bringing the customer and running the app.
- **BaaS middleware takes ~0.4 points** = \$4. The platform fee for connecting the two.

The fintech brought the customer, built the product, and runs the experience — and keeps under a quarter of the economics. The bank does the unglamorous regulated part and keeps the majority. The intuition: in a partnership, economics follow the balance sheet, not the brand. Whoever holds the deposit and the risk holds the money.

![Horizontal bar chart of how the spread splits between the partner bank the fintech and the BaaS platform](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-7.png)

### Why the wall is rising, not falling

It would be natural to assume the wall gets lower over time — that as fintechs mature, charters get easier and partnerships get cheaper. The opposite happened. Three forces have been pushing the wall *up* since around 2022.

First, **rates rose**, and the cost-of-funds gap from the worked example went from a footnote to a chasm. When the policy rate was near zero, a fintech funding at 1% and a bank funding at 0.3% were not that far apart; the deposit moat was quietly there but not decisive. When the policy rate jumped above 5%, a deposit-less fintech's funding cost exploded while a bank's barely moved (because deposit beta is low — banks pass through only part of a hike). The moat went from a few tenths of a point to several full points, and a lot of fintech business models that worked at zero rates simply stopped working.

Second, **regulators tightened the BaaS model**. After a wave of compliance failures at partner banks and the Synapse collapse, US regulators issued enforcement actions against several BaaS partner banks and made clear that the bank — not the fintech — is responsible for compliance on accounts it holds. That raised the cost and slowed the speed of the rent-a-charter path, the very path most fintechs depend on. The middleware that made partnerships fast became a liability when it made compliance murky.

Third, **the banks learned**. The incumbents that looked like dinosaurs in 2015 spent the next decade building better apps, buying fintechs, and copying the good ideas. Zelle (a bank-owned consortium) blunted Venmo; banks dropped trading commissions to match Robinhood; mobile banking apps got genuinely good. The interface gap that fintechs exploited narrowed, because a sustaining innovation is exactly the kind of thing incumbents can eventually match once they take it seriously. Disruption theory predicts that incumbents *lose* to true disruption; it also predicts they *win* against sustaining innovation, and that is what happened.

#### Worked example: the neobank unit economics that don't add up

Let's see why most interface-only neobanks struggle to make money, in per-customer dollars. A neobank's revenue on a basic free account comes from a few thin streams: interchange (a slice of the card fee when the customer swipes), a referral share of the deposit spread from its partner bank, and maybe some out-of-network or overdraft fees.

Say a typical active customer keeps a \$1,500 balance and spends \$1,000 a month on the card. Annual revenue might be:

- Interchange on \$12,000 of annual spend at, say, 0.8% net to the neobank = **\$96**.
- Deposit referral share on the \$1,500 balance: even at a generous 1.2-point share (from our partnership split) = \$1,500 × 1.2% = **\$18**.
- Other fees, conservatively = **\$20**.

Total revenue per customer per year ≈ **\$134**. Now the cost. Customer acquisition cost (CAC) — the marketing spend to win one customer — runs anywhere from \$30 to \$200 for a neobank, and a large share of signups never fund the account or go dormant. Say a blended, all-in CAC across all signups of **\$120** per *active* customer, plus ongoing servicing and fraud costs of, say, **\$40** a year. That is \$160 of cost against \$134 of revenue in year one, and a thin **−\$26** even before counting the dormant accounts that cost money to acquire and earn nothing.

The neobank only makes money if the customer stays for years (amortizing the CAC) and deepens the relationship (more spend, more balance, a loan). Many don't. The intuition: an interface-only bank lives or dies on retention and cross-sell, because each customer is barely profitable on the thin slice of economics the balance-sheet owner leaves it — which is exactly why the data on neobanks shows most of them losing money, and why the durable ones are racing to add lending and to deepen, not just widen, their base.

## Who actually captures the economics

Step back and the pattern across all four exits is the same: economics flow to whoever holds the balance sheet and the risk. The fintech captures the *customer relationship* and a thin slice; the bank captures the *funding spread* and the bulk. This is why, after a decade of "banks are finished" headlines, the biggest banks are more profitable than ever and the fintech sector is, on the whole, still struggling to make money on the parts that touch the balance sheet.

There is a useful way to see it: fintech unbundled the bank — it pulled payments, lending, FX, and investing out of the one-stop branch and made each a great standalone app — and then, hitting the funding and charter wall, it *rebundled* around banks. The unbundling improved the customer's experience enormously. The rebundling returned the economics to the institutions that own the balance sheet. The customer is the clear winner; the question of whether the fintech or the bank wins comes down to who ends up holding the money, and that is almost always the bank.

### The valuation tell: interface multiples vs balance-sheet multiples

You can read this split right off the stock market, and it is the cleanest evidence of all. Markets value interface businesses and balance-sheet businesses completely differently, and the gap tells you what investors think is durable.

#### Worked example: valuing the interface vs valuing the balance sheet

Consider how the market prices the two kinds of business.

A **balance-sheet business** — a bank — is valued on a *price-to-book* (P/B) and *price-to-earnings* (P/E) basis, because its value is its equity capital working at some return. A solid US bank trades around 1 to 1.5 times book value and maybe 10 to 12 times earnings. A bank earning \$10 billion might be worth \$100–\$130 billion. The valuation is anchored to a real, regulated capital base and a steady ~1% return on assets — durable, but capped, because leverage and risk are regulated.

An **interface business** — a payments network or a fast-growing fintech — is valued on *revenue multiples*, because investors are paying for growth and for a capital-light, high-margin stream of fees. At the 2021 peak, fintechs were priced at 20, 30, even 50 times revenue. Stripe was valued at \$95 billion; Klarna at \$46 billion; on a pure earnings basis those numbers were almost unjustifiable.

Now watch what happens to each through the 2022–2023 rate shock. The bank's valuation wobbles but holds — its earnings are real and its franchise is durable. The interface valuations *collapse*: Klarna fell ~85% to \$6.7 billion; many public fintechs fell 70–90% from their peaks. Why the asymmetry? Because a revenue multiple is a bet that today's growth becomes tomorrow's durable profit — and the moment investors doubted that fintechs could ever capture balance-sheet economics, the bet unwound. The intuition: the market will pay a dream multiple for an interface, but it only pays a hard multiple for a balance sheet — and when the dream meets the funding wall, the dream multiple is the one that gets repriced.

## The BNPL credit reality

Buy-now-pay-later deserves its own section, because it is the purest example of an interface business that looked like free money and ran straight into balance-sheet reality. BNPL is interface genius: it removes the word "loan" from a loan, lifts how much shoppers buy, and charges the merchant a fat fee for the privilege. In a booming economy with low unemployment, the losses are tiny and the merchant fees roll in. It looks like the best business in finance.

But BNPL is *still lending*, and lending always has a credit cycle. The provider keeps a flat fee — say 4% of every purchase — regardless of conditions. The loss rate, though, is anything but flat. BNPL's whole pitch is frictionless approval, which means thin credit checks, which means weaker borrowers get in. When unemployment rises and household budgets tighten, those borrowers stop paying, and the loss rate climbs. The fee stays at 4%; the losses go from 1.5% to 4% to 6% of purchase value. The chart below shows the squeeze.

![Bar chart of BNPL flat take rate versus a rising loss rate through a downturn](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-5.png)

#### Worked example: the BNPL take-rate vs loss-rate, and where it flips

Let's run the BNPL math on \$1,000 of purchases through two states of the world.

**Good times.** The provider charges the merchant a 4% fee, so it collects \$40 on the \$1,000 of sales. Its losses (shoppers who never pay their installments) run at 1.5%, or \$15. It also has funding and operating costs — say it must fund the \$1,000 of receivables for six weeks at a 5.5% annual rate (no deposits, remember), which is roughly \$1,000 × 5.5% × (6/52) ≈ \$6.30, plus operating costs of, say, \$10. So:

$$\text{Profit} = \$40 \text{ (fee)} - \$15 \text{ (losses)} - \$6.30 \text{ (funding)} - \$10 \text{ (ops)} = +\$8.70$$

A modest profit. Now a downturn.

**Bad times.** The merchant fee is still 4% = \$40 (the provider cannot raise it; merchants would leave). But losses climb to 6% = \$60. Funding gets more expensive too — risk-averse lenders demand more, call it 7% annualized = \$8. Operating costs hold at \$10. So:

$$\text{Profit} = \$40 - \$60 - \$8 - \$10 = -\$38$$

The same business, the same fee, goes from +\$8.70 to −\$38 per \$1,000 — a swing driven almost entirely by the loss rate, which the interface does nothing to control. The intuition: BNPL monetizes the *interface* (a merchant fee for a better checkout) but carries *balance-sheet risk* (the loans can default), and in a downturn the balance-sheet risk wins. This is exactly why Klarna's losses ballooned and its valuation cratered, and why much of the BNPL industry ended up needing bank-style funding and bank-style underwriting — i.e., ended up becoming more like the banks it was supposed to disrupt.

## Common misconceptions

**"Fintechs disrupted the banks."** Mostly false, in the precise sense of "disrupt." Fintechs disrupted the *experience* of banking — and that is real and valuable — but they left the underlying *business model* (cheap insured deposits funding regulated loans) almost untouched. The proof is in the funding gap: a deposit-less fintech funds at ~5.5% while a bank funds at ~1.9%, a 3.6-point disadvantage that no app can close. Improving the front door is not the same as replacing the building.

**"Neobanks are banks."** Usually false in the US. Most American neobanks hold no charter and no insured deposits; they sit on a partner bank. When you read "FDIC-insured" on a neobank's site, the insurance comes from the *partner bank*, and the pass-through only works if the records reconcile — which the 2024 Synapse collapse showed can fail. The neobank owns the app; the bank owns the deposit.

**"BNPL is free, so it must be cheap for the provider too."** False. BNPL is free to the *shopper* and paid by the *merchant*, but the provider is still making loans and carrying default risk. As the worked example showed, a flat ~4% merchant fee can be swamped by a loss rate that climbs from 1.5% to 6% in a downturn, flipping a small profit into a real loss. "Feels free" is a feature of the interface, not the economics.

**"Banks are slow dinosaurs that will be replaced."** False on the evidence. Through the very rate shock that crushed fintech valuations (2022–2023), the largest banks earned record profits, and panicked deposits ran *toward* the biggest, most-regulated banks, not away from them. The dinosaur metaphor confuses "bad at apps" (true, and partly fixable by buying a fintech) with "bad at being a bank" (false — banks are extremely good at the balance-sheet business that actually makes money).

**"Whoever owns the customer relationship owns the profit."** False in banking. Owning the customer relationship lets a fintech charge a referral fee, but the profit follows the *balance sheet*, not the brand. In a typical partnership the bank that holds the deposit keeps the majority of the spread (~3.4 of ~5 points in our example) while the fintech that owns the customer keeps a slice (~1.2 points). The relationship is worth something; the balance sheet is worth more.

## How it shows up in real banks

**Goldman Sachs and the Apple Card / GreenSky retreat (2016–2024).** Goldman, the ultimate balance-sheet institution, tried to *become* a fintech-flavored consumer bank with its Marcus brand, the Apple Card partnership, and the 2021 acquisition of installment lender GreenSky for ~\$2.2 billion. It learned the lesson from the other direction: building a consumer interface and underwriting consumer credit is hard and loss-heavy if you don't have the deposit base and the consumer-risk machine. Goldman's consumer push lost billions, and it sold GreenSky in 2024 at a loss and pulled back from the Apple Card. The lesson cuts both ways: the interface is genuinely hard, and a balance-sheet champion is not automatically good at the front end.

**The Synapse collapse and the BaaS reckoning (2024).** Synapse was a middleware company sitting between fintechs and partner banks in the banking-as-a-service stack. When it failed in 2024, the reconciliation between what fintechs told customers and what partner banks actually held broke down, and an estimated tens of thousands of end-customers were locked out of their funds — some permanently short, because the ledgers did not add up. It was the clearest possible demonstration that "the fintech shows you a balance but the bank holds the money" is not a detail; it is *the* structural fact, and when the link between interface and balance sheet breaks, the customer discovers the hard way which side actually has their money. US regulators tightened the screws on BaaS partner banks through 2023–2024 in response.

**Klarna's round-trip from \$46bn to \$6.7bn and back toward profit (2021–2024).** Klarna, the BNPL champion, was valued at \$45.6 billion in mid-2021 on an interface-business revenue multiple. As rates rose and credit losses climbed, its valuation collapsed ~85% to \$6.7 billion in a 2022 raise. To survive, Klarna did exactly what the wall predicts: it tightened underwriting, leaned on a banking license it holds in Europe, and pushed toward genuine profitability on a more bank-like footing. The valuation round-trip is the interface-multiple-meets-balance-sheet-reality story in one company.

**The deposit flight to JPMorgan in March 2023.** When SVB and then First Republic wobbled, where did the money go? From the data, the panic moved deposits *to* the largest, most-regulated banks — JPMorgan reportedly took in tens of billions in days. (We cover the run mechanics in [the finance one-pager on SVB and Credit Suisse 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).) No fintech was a haven; the haven was a charter and an implicit too-big-to-fail backstop. In the moment that matters most — a panic — trust and insurance, the bank's deepest moats, are the only things that count.

**Revolut's multi-year charter chase (2021–2024).** Revolut, one of the most successful neobanks in the world with tens of millions of users, spent roughly three years and enormous effort obtaining a UK banking license, finally granted (with restrictions) in 2024. The length of that journey is itself the lesson: even a fintech with a huge customer base and deep pockets needs *years* to get the charter that would let it hold its own deposits. The regulatory moat is real, measured in years and hundreds of millions of dollars, and it is exactly why the partner-bank path is so much more common than the get-your-own-charter path.

**Visa's blocked Plaid acquisition (2020).** Visa agreed to buy Plaid, the data-aggregation layer that connects fintech apps to bank accounts, for \$5.3 billion in 2020 — a classic "absorb the threat" move. US antitrust regulators sued to block it, arguing Visa was buying a nascent competitor, and the deal was abandoned in 2021. It shows the absorption path is not always open: when the incumbent is dominant enough, regulators step in, and the fintech stays independent — though Plaid still ultimately functions as plumbing *between* fintechs and the banks that hold the accounts.

## The takeaway / How to use this

Here is the durable mental model to carry out of this post. When you see a new fintech — a payments app, a lending product, a slick neobank — ask one question: **does it own a balance sheet, or just an interface?** That single question predicts almost everything about who will capture the economics and whether the business is durable.

If it owns only the interface, it is a wonderful product and a fragile business. It will improve customers' lives, win awards, raise money at a revenue multiple — and then hit the wall, where it needs cheap funding it does not have, a charter it cannot easily get, and a compliance machine it was built to avoid. From that wall it will partner (rent a bank), get absorbed (sold to a bank), occasionally disrupt (become a bank, and thereby lose its advantage), or fail. The economics, in every case but the rarest, flow back to whoever holds the deposit and the risk.

If it owns a balance sheet — a charter, insured deposits, regulated capital — it is, whatever it calls itself, a bank. It will be valued like a bank, regulated like a bank, and constrained like a bank. Its advantage is the cheapest funding in finance and the deepest trust moat in finance; its constraint is that it can never grow at software speed, because the balance sheet and the regulator will not let it.

This is the spine of the series stated through fintech's failed assault on it. A bank is a leveraged, confidence-funded maturity-transformation machine. Fintech attacked the *visible* parts of that machine — the funding interface, the lending interface, the payments interface — and made them beautiful. But the *power* of the machine was never in the interface. It was in the right to hold insured money cheaply (the deposit moat) and the trust that keeps that money from running (the confidence). You cannot disrupt a moat made of cheap deposits and slow-earned trust with a better signup flow. You can only build a front end for it — and then negotiate over who keeps the spread, knowing that the side holding the balance sheet holds the cards.

The figure below is the closing thought: the four moats a fintech keeps coming back to a bank for. Win the interface all you like. These four are what made banks the senior partner in nearly every fintech story of the last fifteen years.

![Graph of the four bank moats cheap deposits the charter trust and the compliance machine](/imgs/blogs/fintech-vs-banks-disruption-partnership-or-quiet-absorption-8.png)

The practical use of all this: as a customer, enjoy the interface — it is genuinely better, and fintech competition forced banks to improve too. But know where your money actually sits and whether it is *your* deposit at an insured bank or a *balance* at an app routing it somewhere. As someone reading or running a bank, stop fearing the interface and start owning it: buy the front end, partner with the apps, modernize the experience — and never, ever give up the deposit franchise or the charter, because that is the part no one can take.

## Further reading & cross-links

- [Open banking, banking-as-a-service, and embedded finance](/blog/trading/banking/open-banking-apis-banking-as-a-service-and-embedded-finance) — the API-and-platform mechanics of how a fintech plugs into a bank's charter.
- [Digital banking and the neobank business model](/blog/trading/banking/digital-banking-and-the-neobank-business-model) — the unit economics of the interface-only bank and why most don't make money.
- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — the deposit moat in full, and the CASA ratio and deposit beta behind the cost-of-funds gap.
- [Consumer lending: mortgages, cards, auto, and personal loans](/blog/trading/banking/consumer-lending-mortgages-cards-auto-and-personal-loans) — the retail loan products that BNPL and fintech lenders re-skinned.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — where non-bank credit balance sheets sit when the lending leaves the bank.
- [SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the runs that proved, in a panic, that trust and a charter are the only havens.

*This is educational, not financial advice.*
