---
title: "Cross-Border Payments: Correspondent Banking and How SWIFT Really Works"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How an international payment actually moves — the correspondent-banking chain, SWIFT as a messaging network that never touches your money, nostro and vostro accounts, the FX leg, the fees and lifting charges, and the reforms that are slowly fixing it."
tags: ["banking", "cross-border-payments", "correspondent-banking", "swift", "nostro-vostro", "foreign-exchange", "payments", "iso-20022", "swift-gpi", "transaction-banking"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A cross-border payment is not money flying down a wire. It is a *secure message* (usually sent over SWIFT) instructing a chain of banks to move value across accounts they already hold for each other. SWIFT carries the instruction; the money settles, hop by hop, in *correspondent accounts*. That structure is why an international transfer is slow, opaque, and expensive while a domestic one is instant and nearly free.
>
> - **SWIFT moves messages, not money.** It is a member-owned cooperative running a secure messaging network. It never holds, converts, or settles a single dollar.
> - The money walks a **correspondent-banking chain**: your bank → its correspondent → an FX bank → the beneficiary's correspondent → the beneficiary's bank. Each link is a real account relationship, and **each hop shaves a fee** (the *lifting charge*).
> - A **nostro** ("our account with you") and a **vostro** ("your account with us") are the *same account* seen from two sides. All cross-border settlement is bookkeeping across these accounts — nothing crosses a border.
> - The biggest hidden cost is usually the **FX spread**, not the visible wire fee. On a small payment, all-in cost can run **around 6%** of the amount sent — versus near zero for a domestic instant rail.
> - The one number to remember: send \$10,000 abroad through three correspondents and a 1.5% FX margin, and the payee receives about **\$9,785**. The headline amount and the received amount are not the same number, and the gap can take **days** to fully reveal itself.

You pay 4% interest on your savings and the screen says the transfer to your supplier in Germany will be "instant." You hit send. Three days later, your supplier emails: the money arrived, but it is short by \$215, and nobody can tell either of you exactly where the missing money went. Welcome to cross-border payments — the part of banking that, in 2026, can still feel like mailing cash in 1975.

Here is the strange thing. Domestically, money moves beautifully. In dozens of countries you can now send money to a stranger's phone number and it lands in their account in under ten seconds, for free, at 3am on a Sunday. But the moment that payment has to cross a currency or a national border, it falls off a cliff: it slows to days, sprouts fees you cannot see in advance, and disappears into a black box where neither you nor your bank can say where it is. Why? The answer is not that banks are lazy or that the technology is missing. The answer is *structural* — it is baked into how the global banking system is wired together. And once you understand that wiring, every frustrating thing about an international payment suddenly makes sense.

The diagram below is the mental model for the whole post: a payment from the US to Europe does not travel in a straight line. It hops a chain of banks, each of which holds an account for the next, while a SWIFT message races ahead carrying the *instruction*. The money and the message travel on completely separate tracks. Hold that picture — almost everything that follows is a consequence of it.

![A cross-border payment hopping sender bank, correspondents, an FX bank, and beneficiary bank with SWIFT messages alongside](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-1.png)

This post is the deep mechanics. For the *geopolitics* of SWIFT — sanctions, the weaponization of payments, what it means to be "cut off from SWIFT" — see the companion piece on [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments). Here we stay with the mechanics and ask: when you send money abroad, what *physically happens*, account by account, message by message, fee by fee?

## Foundations: how money actually crosses a border

Before any mechanics, let's build the vocabulary from zero. There are exactly five ideas you need, and every one of them is simpler than it sounds.

### Money does not move; ownership of a claim moves

Start with the most counterintuitive fact, because everything else rests on it. When you "send money," no physical thing travels. What you actually have in a bank account is not cash sitting in a drawer with your name on it — it is a *claim* on the bank, a promise that the bank owes you that amount. A *deposit* is a liability of the bank to you. (We unpack this fully in the post on [reading a bank balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity), but the one-line version is: your asset is the bank's IOU.)

So a "payment" is really an act of bookkeeping: the bank reduces what it owes you and increases what it owes someone else (or owes another bank). Within one bank that is trivial — debit one account, credit another, done. Between two banks in the same country, they settle through a shared *clearing* and *settlement* system, often the central bank, where both banks hold accounts. We cover that domestic plumbing in [the payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) and the specific domestic rails in [domestic payment rails: RTGS, ACH, card networks, and instant payments](/blog/trading/banking/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments).

The trouble starts when the two banks do *not* share a settlement system — because they are in different countries, regulated by different authorities, dealing in different currencies. There is no global central bank where every bank on Earth holds an account. So how do they settle? They use each other.

### The correspondent bank — a bank's bank abroad

A *correspondent bank* is a bank that holds an account for another bank and performs services — chiefly payments and settlement — on its behalf in a market or a currency where the first bank has no direct presence.

Think of it this way. A mid-sized US bank wants to be able to pay people in euros. It cannot open a branch in every European country and join every European settlement system — that would cost a fortune and take years of licensing. Instead, it strikes up a *correspondent relationship* with a large European bank (say, a big German bank) that already has euros and is already plugged into the European settlement system. The US bank opens an account with the German bank, funds it with euros, and now — whenever it needs to pay a euro recipient — it sends an instruction to the German bank: "please pay €X to this person out of my account with you." The German bank does the local leg. The US bank never has to touch the European plumbing directly.

The correspondent is, in plain terms, a bank's *bank abroad*. It is a wholesale, bank-to-bank relationship, the foundation of the entire international payments system. The intuition is the same as a traveler who doesn't speak the local language hiring a local fixer who does: you don't go to the foreign market yourself, you pay someone who is already there and trusted.

### Nostro and vostro — the same account, two names

Here is the term that trips up everyone, and it is genuinely just bookkeeping jargon for the two ends of one account.

When our US bank opens an account at the German bank, the US bank, looking at *its own* books, calls that "our account, held over there, in their currency." The Italian-rooted banking word for *our* is **nostro** — so the US bank records it as a **nostro account** ("our money, parked with you, abroad").

The German bank, looking at the *same* account on *its* books, sees "an account that belongs to you, sitting here with me." The word for *your* is **vostro** — so the German bank records it as a **vostro account** ("your money, here with us").

One account. Two perspectives. The US bank's nostro *is* the German bank's vostro. (You will sometimes hear the German bank also call it a *loro* account — "their" account — when talking about it to a third party, but nostro/vostro is the pair to remember.) The figure below makes the symmetry concrete.

![Bank A nostro account and Bank B vostro account shown as one ledger relationship](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-2.png)

Why does this matter so much? Because *all* cross-border settlement happens by debiting and crediting these accounts. No euros are shipped across the Atlantic. The German bank simply reduces the balance in the US bank's account (its vostro) and pays the local recipient from its own euro pool. The "transfer" is two ledger entries on the German bank's book. The border is never crossed by money — only by a message.

### SWIFT — the messaging network everyone misnames

Now the most misunderstood institution in finance. SWIFT stands for the *Society for Worldwide Interbank Financial Telecommunication*. It is a **cooperative, owned by its member banks**, founded in 1973 and headquartered in Belgium, and it runs a **secure financial messaging network**. Roughly 11,000-plus institutions across 200-plus countries are connected, and on a busy day the network carries on the order of tens of millions of messages.

Read that definition again and notice what is *not* in it. SWIFT does not hold money. SWIFT does not have accounts for the public. SWIFT does not convert currencies. SWIFT does not settle anything. **SWIFT is a messenger.** When your bank "sends money via SWIFT," what it actually sends is a standardized, authenticated *instruction* — a message that says, in a rigid machine-readable format, "pay this amount, in this currency, to this beneficiary, debit my account with you." The classic message type for a single customer credit transfer is the **MT103**; bank-to-bank settlement instructions ride on the **MT202**. (The newer ISO 20022 equivalent of an MT103 is a message called `pacs.008` — more on the format upgrade later.)

The cleanest analogy: SWIFT is like a hyper-secure, standardized postal and email service *for banks*. You can mail a cheque, and the cheque is a piece of paper that instructs a bank to pay — but the postal service that delivered it never had your money. SWIFT is that postal service, except the "letters" are tamper-proof, authenticated, and follow a format so strict that a computer on the other end can act on them automatically. The money still lives in the banks' accounts the whole time.

This single misconception — "SWIFT moves money" — causes more confusion than any other idea in payments, so we will return to it explicitly in the misconceptions section. The before-and-after below states it plainly.

![Side by side comparison of the myth that SWIFT moves money versus the reality that it sends messages](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-3.png)

It helps to be precise about *what an MT103 actually contains*, because seeing the fields makes it obvious that this is a letter and not a payment. The message carries: a transaction reference; the value date and the currency and amount; the *ordering customer* (you, the sender) with name and account; the *beneficiary customer* (your supplier) with name and IBAN; the chain of banks involved (sender's bank, intermediary banks, the beneficiary's bank, each by BIC); a "details of charges" code that says whether fees come out of the payment or are billed separately; and a free-text "remittance information" field for an invoice number or note. Nowhere in that message is there any *money* — there is only a set of instructions and identifiers that tells a receiving bank exactly which account to debit, which to credit, and how to label the entry. A bank reading the message acts on it by adjusting balances in accounts it already controls. The message is the cause; the ledger entries are the effect; SWIFT only delivered the cause.

One more nuance that matters later: SWIFT is *not the only* financial messaging system. Many countries run their own domestic messaging and settlement systems, and there are competing or alternative cross-border messaging arrangements (some built specifically to reduce dependence on SWIFT). But SWIFT remains the dominant, near-universal standard for cross-border interbank messaging — which is precisely why disconnecting a bank from it is such a powerful lever, and why the global migration of *its* message formats to ISO 20022 is such a big deal. When we say "via SWIFT" in this post, we mean the standard correspondent-banking message flow that most international payments still use.

### The FX leg and the lifting charge

Two last terms and we have the whole toolkit.

The **FX leg** is the currency conversion. If you send US dollars and your recipient is to be paid in euros, somewhere on the chain a bank converts USD into EUR. *Foreign exchange* (FX) is just the act of swapping one currency for another at a rate. The rate the bank gives you is almost never the "real" mid-market rate you'd see on a financial site; the bank builds in a margin — the *spread* — which is the gap between the rate at which it will buy a currency and the rate at which it will sell it. That spread is profit for the bank and cost for you, and on a retail payment it is frequently the *largest* single charge — larger than any visible fee.

A **lifting charge** (also called a *lifting fee* or *correspondent charge*) is the small fee a correspondent bank deducts for processing a payment that passes through it. Picture each correspondent in the chain reaching into the envelope as it passes and taking a small handling fee — that is a lifting charge. Crucially, these are often deducted *from the payment amount itself* rather than billed separately, which is exactly why the recipient gets less than the sender sent and why nobody can tell you the total in advance.

That's the foundation. A claim, not cash; a correspondent who is your bank abroad; a nostro/vostro account that is one account with two names; SWIFT the messenger; an FX leg with a spread; and a lifting charge at every hop. Now let's watch a real payment move through all of it.

## The anatomy of one cross-border payment, step by step

Let's send money. You are in New York; you bank with a mid-sized US bank we'll call **Sender Bank**. You want to pay €9,000-ish worth to a supplier in Germany who banks at **Beneficiary Bank**. You'll fund it with \$10,000. Here is what actually happens, in order.

**Step 1 — You instruct your bank.** You give Sender Bank the beneficiary's name, account number (in Europe, an *IBAN* — International Bank Account Number), the beneficiary bank's identifier (a *BIC*, explained below), the amount, and the currency. Sender Bank charges you an outward-wire fee — say **\$30** — and debits your account \$10,000.

**Step 2 — Sender Bank finds the route.** Sender Bank does not have a direct account with Beneficiary Bank in Germany; they have no relationship. So Sender Bank uses its *correspondent network*. It holds a USD account arrangement and a relationship with a large global bank — call it **USD Correspondent** — that can reach Europe. Sender Bank composes a SWIFT MT103 message: pay this beneficiary, debit our account. It sends the message into the SWIFT network, which delivers it, authenticated and intact, to USD Correspondent.

**Step 3 — The USD correspondent processes and screens.** USD Correspondent receives the message. It debits Sender Bank's account on its books (Sender Bank's nostro shrinks), takes a **lifting charge** of, say, **\$15**, and runs the payment through *compliance screening* — checking the names and the destination against sanctions lists and anti-money-laundering (AML) filters. If a name fuzzily matches a sanctioned entity, the payment can be frozen for manual review for hours or days. Assume it clears.

**Step 4 — The FX leg.** The payment needs to become euros. USD Correspondent (or a dedicated FX bank in the chain) converts the dollars to euros at its rate. Suppose the true mid-market rate is such that \$1 buys €0.92, but the bank applies a **1.5% margin**, so it actually gives you a worse rate. On the amount being converted, that margin is worth about **\$150** of value — quietly skimmed inside the exchange rate, never itemized as a "fee."

**Step 5 — Into Europe.** The now-euro payment travels (again as a SWIFT message and a ledger movement) to **EUR Correspondent**, the bank that holds a euro relationship with Beneficiary Bank — or directly into the European settlement system. EUR Correspondent credits Beneficiary Bank's account on its books (Beneficiary Bank's vostro/nostro grows), and takes its own lifting charge of, say, **€10 (about \$12)**.

**Step 6 — The final credit.** Beneficiary Bank receives the funds in its account at EUR Correspondent, applies a small inward-credit fee — say **€8 (about \$8)** — and finally credits the supplier's account. The supplier sees the money.

Notice what happened to the money at each border: *nothing crossed*. USD Correspondent debited Sender Bank's USD account and handed dollars into the FX desk; the FX desk produced euros; EUR Correspondent credited Beneficiary Bank's euro account. Every "transfer" was a debit on one ledger matched by a credit on another. The euros that reached Germany were already *in Germany* — they were EUR Correspondent's euros — and the dollars that left New York stayed in the US correspondent system. Money is local; only the *claims* and the *messages* moved internationally.

It is worth dwelling on the compliance screening in Step 3, because for a beginner it is the least visible and most consequential part. Every bank that handles the payment is legally obliged to check it against *sanctions lists* (registers of individuals, entities, and countries that banks are forbidden to deal with) and to apply *anti-money-laundering* (AML) rules designed to catch funds tied to crime. The screening is automated and fuzzy: a system compares the names and details on the payment against the lists, and because lists hold transliterated names, aliases, and partial matches, the system deliberately errs toward catching too much. A beneficiary named, say, "Ali Hassan" might fuzzily match a sanctioned "Ali Hasan," triggering a *false positive* — the payment is pulled aside for a human to investigate. That human review can take hours or a full business day, and it happens *independently at every bank on the chain*. So a three-hop payment is screened three separate times, and the probability that *at least one* screen produces a false-positive hold rises with each hop. This is why a longer chain is not just more expensive (more lifting charges) but also slower and riskier (more screening checkpoints, more chances to be stopped). Compliance is not a side process bolted on; it is woven through every link, and it is one of the dominant reasons a payment that should take minutes can take a week.

#### Worked example: the three-correspondent fee stack-up

Let's total it up. You send **\$10,000**. Here is every deduction along the chain:

- Sender Bank outward-wire fee: **−\$30**
- USD Correspondent lifting charge: **−\$15**
- FX spread (1.5% margin on the converted amount): **−\$150**
- EUR Correspondent lifting charge (€10): **−\$12**
- Beneficiary Bank inward-credit fee (€8): **−\$8**

Total deducted: \$30 + \$15 + \$150 + \$12 + \$8 = **\$215**.

Net received by the supplier: \$10,000 − \$215 = **\$9,785**.

So the supplier gets **\$9,785** worth of euros against your \$10,000 send — a **2.15%** all-in cost. The intuition to carry away: the headline "\$10,000" and the received "\$9,785" are different numbers, the gap is split across five separate hands, and the single biggest bite (\$150 of \$215, about 70%) was the *invisible* FX spread, not any fee with a name. The chart below shows the same money getting whittled down stage by stage.

![Waterfall of a 10000 USD payment reduced by each correspondent fee and the FX spread down to 9785 received](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-4.png)

## BIC, IBAN, and how a message finds the right bank

For the routing to work, every bank needs a globally unique address. That address is the **BIC** — the *Bank Identifier Code*, also called a *SWIFT code*. A BIC is 8 or 11 characters: 4 for the institution (e.g. `DEUT` for Deutsche Bank), 2 for the country (`DE` for Germany), 2 for the location, and an optional 3 for a specific branch. When your bank addresses an MT103, the BIC tells the SWIFT network exactly which institution to deliver the message to. It is the equivalent of an email address for a bank.

The **IBAN** identifies the specific *account* (and embeds the country and a checksum so a typo is caught before the payment is sent). Together, the BIC routes the message to the right bank and the IBAN tells that bank which customer account to credit. In the US, banks more commonly use a *routing number* (ABA) domestically and a BIC for the international leg.

Here is the part beginners miss: the sending bank often does **not** know the full chain in advance. It knows the beneficiary's bank (from the BIC) and its own correspondents, but the *intermediary* banks in between are chosen along the way by each correspondent based on its own relationships. That is precisely why the total cost is unknowable up front — the sender literally cannot enumerate every hand the payment will pass through until it has passed through them. This routing opacity is the root cause of the whole "where is my money?" problem, and it is what the reforms (later) attack head-on.

There is a subtlety in *who pays the charges* that explains the mysterious shortfall. The MT103 carries a "details of charges" instruction with three options. **OUR** means the sender pays all charges, so the beneficiary receives the full amount — the cleanest but priciest for the sender. **BEN** means the beneficiary bears all charges, deducted from the payment as it travels, so the recipient gets the least. **SHA** ("shared") — by far the most common default — means the sender pays its own bank's fee and the beneficiary bears the *intermediary* lifting charges. Under SHA, every correspondent in the middle quietly deducts its lifting charge from the principal as the payment passes, which is exactly why our supplier received \$9,785 instead of \$10,000: the chain helped itself along the way. If you want your recipient to get the exact amount, you must instruct **OUR** and accept a higher, often unpredictable, total bill. Most people never choose, so they get SHA, and then they are surprised by the shortfall. The choice of charge code is one of the few cost levers a sender actually controls.

#### Worked example: the FX spread leg in isolation

Let's isolate the FX leg, because it is the cost people most consistently underestimate. Suppose the *mid-market* rate — the fair midpoint between buying and selling — is **1 USD = 0.9200 EUR**. You convert \$10,000 at the mid: you would get **€9,200**.

But the bank quotes you its *customer* rate with a 1.5% margin baked in. Instead of 0.9200, it gives you **0.9062** (that's 0.9200 × (1 − 0.015)). Now your \$10,000 buys only:

\$10,000 × 0.9062 = **€9,062**.

The difference is €9,200 − €9,062 = **€138**, which at the mid rate is worth about **\$150**. That \$150 never appears on any statement as a "fee." It is hidden in the exchange rate itself. The bank will happily tell you "we charge no commission on FX" — and that is technically true, because the profit is in the *spread*, not a commission line. The intuition: when a money-transfer service advertises "zero fees," always ask what rate they use, because the spread is where the real charge lives. A wide spread on a "free" transfer can cost you far more than a small flat fee on a transfer that converts at the mid rate.

## The journey takes days, and here's exactly why

A domestic instant payment settles in seconds. A correspondent-banking wire can take **one to five business days**. People assume this is some computer being slow. It is not — the computers are fast. The delay is a stack of *human and institutional* frictions, and each one is a real reason.

**Time zones.** A payment leaving New York at 4pm hits the German correspondent after its business day has closed. The euro leg simply cannot process until the next morning in Frankfurt. One time-zone gap costs a calendar day even though zero "work" was waiting.

**Cut-off times.** Every bank has a daily *cut-off* — a deadline after which a payment is processed the *next* business day rather than today. Miss the USD correspondent's 2pm cut-off and your payment, however urgent, waits until tomorrow's batch.

**Sequential processing.** Each hop must complete before the next can begin: the USD correspondent has to debit, screen, and convert before it hands the payment to the EUR correspondent. The hops are a *relay race*, not parallel lanes. Three hops with one delay each can stack into three lost days.

**Compliance screening.** Every bank on the chain independently screens the payment against sanctions and AML lists. A "false positive" — a beneficiary whose name resembles a flagged entity — triggers a manual review that can hold the payment for a day or more *at any single hop*. Because the checks are repeated at each bank, the chances of at least one stop rise with the number of hops.

**Weekends and holidays.** Banking days exclude weekends and national holidays, which differ by country. A payment that crosses a Friday in one country and a different country's Monday holiday can sit idle for four days while no one is doing anything wrong.

The timeline below traces one wire across its multi-day life.

![Timeline of a cross-border wire over several days passing through correspondents time zones and screening](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-6.png)

#### Worked example: days-to-settle, hop by hop

Let's count the days for our New York → Frankfurt payment, sent Thursday 4pm New York time.

- **Thursday (Day 0):** You submit. Sender Bank composes the MT103 and sends it. New York is winding down; Frankfurt is already closed (it is 10pm there). Nothing further can happen in Europe today.
- **Friday (Day 1):** USD Correspondent processes during US hours, debits Sender Bank, screens, runs the FX leg. It hands off to EUR Correspondent — but by then Frankfurt's cut-off has passed.
- **Monday (Day 2 of business, calendar Day 4):** The weekend intervenes. Frankfurt opens Monday; EUR Correspondent processes the euro leg, credits Beneficiary Bank's account.
- **Tuesday (Day 3 of business):** Beneficiary Bank applies its inward fee and credits the supplier. Money visible.

That is **two business days of actual work** stretched across **five calendar days** by one time-zone gap, one missed cut-off, and one weekend — with *zero* error or compliance hold. Add a single false-positive sanctions hit and you are at a week. The intuition: cross-border slowness is rarely a single broken thing; it is a dozen small, legitimate frictions compounding. Fix one and you save hours, not days. To save days you have to attack the *whole chain* at once — which is exactly what the reforms try to do.

## Domestic versus cross-border: why the same money behaves so differently

The sharpest way to feel how strange the cross-border world is, is to stand it next to the domestic world. Inside a single country, you can send money to a stranger in under ten seconds, for free, around the clock. Across a border, the same act becomes a multi-day, multi-fee ordeal. *Same money, same banks, same customer* — the only thing that changed is that a currency or a border got involved. Why such a chasm?

The answer is that domestic payments have something cross-border payments fundamentally lack: a **single shared settlement venue with a single trusted operator**. Within a country, every bank holds an account at the central bank, and the central bank's books are the one true ledger where final settlement happens. Bank A pays Bank B by moving central-bank money from A's account to B's account — one debit, one credit, on one ledger, with *finality* (the payment cannot be reversed). There is no chain. There is no FX. There is one hop, one operator, one currency, one set of business hours. We unpack exactly how this works in [domestic payment rails](/blog/trading/banking/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments) — RTGS systems for big payments, instant rails for retail — but the structural point is simple: domestic payments settle on a *hub*, cross-border payments settle on a *chain*.

A hub is fast because every participant connects directly to the center. A chain is slow because value has to relay through a series of bilateral relationships, each in its own jurisdiction, currency, and time zone. There is no global central bank, no single ledger where every bank on Earth holds an account, so cross-border payments cannot use a hub. They are stuck on the chain — and the chain is where the cost, the delay, and the opacity all come from. Every reform you will read about is, at bottom, an attempt to make a chain behave a little more like a hub: tracking it end to end, speeding the hops, standardizing the data, or (in the case of stablecoins) replacing the chain with a single shared blockchain ledger that *acts* like a hub.

The cost gap is just as dramatic as the speed gap. A domestic instant payment is essentially free to the user — fractions of a cent in real cost. A cross-border payment, especially a small one, can run several percent of the amount sent once you fold in the lifting charges and the FX spread. The chart below puts the two side by side, and the difference is not subtle.

![Bar chart comparing the all-in cost of a domestic instant payment versus a cross-border correspondent wire](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-7.png)

#### Worked example: a 200-dollar remittance, domestic versus cross-border

Imagine you need to send \$200 to someone. First, domestically, over an instant rail: the transfer is free, lands in seconds, and the recipient gets the full **\$200**. All-in cost: effectively **\$0**, or about **0.1%** once you account for the tiny real processing cost the bank absorbs.

Now send the same \$200 *across a border* through the correspondent chain. A flat outward fee of, say, \$5, plus lifting charges down the chain of about \$5 more, plus a 1.5% FX spread (1.5% × \$200 = \$3) gives a total cost of \$5 + \$5 + \$3 = **\$13** — about **6.5%** of the amount sent, right in the band the World Bank reports as the global remittance average. The recipient gets roughly **\$187** of the \$200 you sent.

The intuition is brutal and important: on small amounts, the *fixed* per-hop fees dominate, so the percentage cost explodes precisely for the people who can least afford it — migrant workers sending small sums home. A \$200 transfer costing \$13 is the same structure as our \$10,000 transfer costing \$215, but at small size the fixed fees swamp everything. Domestic money is free because it rides a hub; cross-border money is expensive because it rides a chain, and the chain charges a toll at every link.

## The four jobs running inside every wire

Step back and it becomes clear that a cross-border payment is not one process but **four** running simultaneously, and any one of them can be the bottleneck. Separating them is the single most useful mental upgrade for understanding why a given payment was slow or expensive.

1. **Messaging** — getting the instruction from bank to bank, accurately and securely. This is SWIFT's job. The failure mode is *bad or incomplete data*: a malformed message, a missing field, a beneficiary name that doesn't match — the message gets rejected or held.
2. **Settlement** — actually moving the value across the correspondent accounts. This is the correspondent banks' job. The cost here is the *lifting charge* at each hop, and the delay is *sequential processing*.
3. **FX conversion** — turning one currency into another. This is an FX bank's job. The cost is the *spread*, the largest hidden charge on most retail payments.
4. **Compliance** — screening for sanctions and money laundering. This is *every* bank's job, independently, on every hop. The cost is *delay* when a payment is held for review.

Crucially, these four are *decoupled*. The message (SWIFT) can arrive in seconds while the settlement (correspondent accounts) takes days. The FX can be locked in at one point while compliance holds the payment at another. When a payment is "stuck," diagnosing it means asking *which of the four layers* is the problem — and the answer is usually settlement (a hop didn't process) or compliance (a hold), not messaging. The matrix below lays out the four layers and where each one hides its cost or delay.

![Matrix of the four layers of a cross-border payment messaging settlement FX and compliance with who runs each and where cost hides](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-5.png)

This four-layer view also explains a subtlety: "cutting a country off from SWIFT" disables the *messaging* layer, not the others. The accounts and the correspondent relationships still exist — but without the standardized message to instruct them, banks fall back to slow, manual, bilateral methods (phone, telex, ad-hoc channels). It is like disabling a country's email while leaving its bank accounts intact: the accounts work, but coordinating across them becomes painfully slow and error-prone. That distinction is the entire mechanism behind SWIFT-based sanctions, which the [SWIFT geopolitics post](/blog/trading/finance/swift-and-the-weaponization-of-payments) explores in full.

## Why correspondent banking is shrinking — the de-risking problem

Here is a trend that surprises people: the global correspondent-banking network has been *shrinking* for over a decade. The number of active correspondent relationships has fallen substantially since around 2011, even as the *value* of payments has grown. Fewer banks are willing to be everyone's bank abroad.

The reason is **de-risking**. A correspondent bank that processes payments for a smaller "respondent" bank in a higher-risk jurisdiction is on the hook for compliance: if a laundered or sanctioned payment slips through *its* pipes, the correspondent — not the respondent — can face enormous fines and reputational damage. (The AML scandals that drove this fear — HSBC, Danske Bank's Estonian branch, and others — are covered in the conduct track of this series.) Faced with a small respondent in a risky country that generates modest fees but unbounded compliance liability, large banks increasingly just *exit the relationship*. It is cheaper to walk away than to monitor it.

The second-order effect is severe and uncomfortable. Whole regions — parts of the Caribbean, Africa, the Pacific, and remittance-dependent economies — have found themselves with *fewer and fewer* paths into the global dollar and euro systems. A migrant worker trying to send \$200 home to a country that has lost its correspondent links may find the payment routed through more hops (each adding cost), or find the cheapest channel has simply vanished. De-risking is a vivid example of how a *regulatory incentive* (punish the correspondent for the respondent's sins) reshapes the *physical topology* of the payment network — and not always in the direction policymakers intended. It is also a quiet driver of the move toward fintech rails and stablecoins, which promise to route around the shrinking chain.

#### Worked example: when fewer hops becomes more hops

Suppose a small bank in a Pacific island nation used to reach the US through *one* direct correspondent. Its remittance customers paid a single \$15 lifting charge plus a 1% FX spread on a \$200 send — total cost about \$15 + \$2 = **\$17**, or **8.5%**. (Remittances are brutal on small amounts: the fixed fee dominates.)

Now its direct correspondent de-risks and exits. The only remaining path runs through *two* intermediary banks. Each takes a \$12 lifting charge, and the FX spread widens to 1.5% because the smaller flow gets a worse rate. New cost: \$12 + \$12 + (1.5% × \$200) = \$24 + \$3 = **\$27**, or **13.5%** of the \$200 sent.

So the *same* \$200 remittance went from costing \$17 to costing \$27 — a 59% jump — purely because a relationship disappeared and the payment had to detour through more hands. The intuition: in correspondent banking, *access* is itself a cost. The fewer banks willing to carry your payments, the longer and pricier every chain becomes — and the people hurt most are the ones sending the smallest amounts.

## How the spread, the fees, and the leverage spine connect

It is worth tying the payment business back to the spine of this whole series: a bank is a *leveraged, confidence-funded maturity-transformation machine* that lives on a spread. Cross-border payments are a different kind of spread business, but a spread business nonetheless.

A correspondent bank earns three things from running these pipes. First, the **lifting charges** — small per-transaction fees that, across millions of payments, add up to a meaningful fee-income line. Second, the **FX spread** — by far the richest, because converting currency at a margin is high-volume, low-capital, and recurring. Third, and most strategically, the **deposits** the business generates: every nostro account a respondent bank funds at the correspondent is a *deposit* sitting on the correspondent's balance sheet. Those are sticky, low-cost wholesale deposits — exactly the cheap funding that, as the [retail deposits post](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) explains, is the whole franchise.

This is why the global transaction-banking and FX business is so prized by the largest banks: it is *fee income plus cheap funding plus FX margin*, all at once, and it requires comparatively little of the precious equity capital that lending consumes. It is the opposite of the risky maturity-transformation trade. No depositor run, no credit losses, no duration gap — just a toll booth on the world's money. A big universal bank that dominates correspondent flows is, in effect, collecting rent on the plumbing. That rent is durable, which is exactly why the network has been slow to change despite decades of complaints — the incumbents are paid handsomely to keep it roughly as-is.

There is also a *network-effect* moat protecting that rent. A correspondent's value to a respondent bank is proportional to how many markets, currencies, and onward relationships it can reach. The biggest global banks can reach almost everywhere, so they win the most respondent relationships, which gives them the most flow, which gives them the best FX pricing and the cheapest unit costs, which makes them an even more attractive correspondent. Smaller banks cannot match the reach, so they themselves become respondents of the giants rather than competitors. The result is a market that naturally concentrates into a handful of dominant global correspondents — a structure that is efficient for the giants and the source of pricing power that no amount of customer grumbling has dislodged. When you wonder why cross-border fees have been so sticky for so long, the network effect is a large part of the answer: the people who could compete the fees away mostly cannot reach far enough to try.

## Common misconceptions

Cross-border payments attract more confident-but-wrong beliefs than almost any topic in banking. Here are the ones worth correcting, each with the number that settles it.

**"SWIFT moves my money."** No — and this is the big one. SWIFT is a *messaging cooperative*. It never holds, converts, or settles money. It transmits the *instruction*; the money settles in correspondent accounts entirely outside SWIFT. The proof is in the structure: your \$10,000 was debited and credited across nostro/vostro accounts at correspondent banks, while SWIFT only carried the MT103 message that told those banks what to do. If SWIFT vanished tomorrow, the *accounts* and *balances* would all still exist — banks would just have to instruct each other by slower means. SWIFT is the email, not the bank.

**"The transfer is free / I'm only paying the wire fee."** The visible wire fee (\$30 in our example) is usually the *smallest* cost. The largest is the **FX spread** — \$150 of the \$215 total, about 70%, hidden inside the exchange rate where no statement itemizes it. "No-commission" FX is a real phrase and a real trap: the bank makes its money on the spread, not a commission. Always compare the *rate you got* against the mid-market rate to see the true cost.

**"It's slow because the technology is old."** Partly, but mostly no. The messages move in seconds. The delay is *time zones, cut-off times, sequential hop-by-hop processing, repeated compliance screening, and weekends* — institutional frictions, not bandwidth. Our payment took five calendar days with zero errors; the technology was never the bottleneck. This is why simply "modernizing the software" never fully fixed cross-border speed — you have to attack the chain structure and the operating-hours problem too.

**"My bank knows exactly where my money is and what it'll cost."** Often it does not. The sending bank picks its first correspondent, but *intermediary* banks are chosen down the chain by each correspondent. The sender frequently cannot enumerate the full route or the total lifting charges in advance — which is precisely why the recipient can receive an unexpected, unexplained shortfall. The whole point of the SWIFT gpi tracker (next section) was to fix this specific blindness.

**"A nostro and a vostro are two different accounts."** They are *one* account viewed from two sides. The US bank's nostro ("our money over there") is the German bank's vostro ("your money here"). Every dollar of settlement is a debit on one ledger matched by a credit on the same account from the other bank's perspective. Confusing them as separate accounts is the single most common exam mistake in transaction banking.

**"Cutting a bank off from SWIFT freezes all its money."** No — it disables the bank's ability to *send and receive standardized payment instructions* over the dominant network. The bank's existing accounts, balances, and correspondent relationships still exist. What it loses is the easy, automated way to instruct payments across borders; it must fall back to slow, manual, bilateral channels. The effect is severe — like disabling a company's email while leaving its bank account intact — but it is a *messaging* sanction, not a seizure of funds. Understanding this distinction is the whole key to reading what a "SWIFT ban" actually does, which is why it deserves its own treatment in the [geopolitics post](/blog/trading/finance/swift-and-the-weaponization-of-payments).

**"Crypto and stablecoins have already made this obsolete."** Not yet, and the reason is instructive. The hard part of cross-border payment was never the *technology of moving a number between ledgers* — it was the *trust, compliance, and on-and-off-ramps* into real currencies. A stablecoin can settle value in minutes on a shared ledger, sidestepping the correspondent chain — but the recipient usually still needs to convert it into local currency to pay rent, which reintroduces an FX leg and a regulated off-ramp. Stablecoins genuinely attack the *settlement and speed* layers; they do not magically erase the *FX* and *compliance* layers. They are a serious challenge to the chain, not a finished replacement of it — which is exactly why incumbents are also racing to fix gpi and ISO 20022 rather than conceding the field.

## How it shows up in real banks and the real world

Let's ground all of this in named, concrete reality — episodes and facts you can verify, with the mechanism from this post visible in each.

**SWIFT cut-offs as a sanctions tool (2012, 2022).** In 2012, under pressure from the EU and US, SWIFT disconnected designated Iranian banks from its network. In 2022, following the invasion of Ukraine, a number of Russian banks were similarly removed. In both cases the *messaging* layer was severed — but the banks' accounts and correspondent balances still existed. The practical effect was that those banks had to fall back on slow, manual, bilateral channels to instruct payments, which dramatically raised friction and cost without literally "freezing" every account. This is the four-layer model in action: kill messaging, and settlement becomes agonizing even though the accounts are intact. The full geopolitics is in the [SWIFT weaponization post](/blog/trading/finance/swift-and-the-weaponization-of-payments).

**The remittance cost problem (ongoing).** The World Bank has long tracked the global average cost of sending remittances, and for years it has hovered around **6%** of the amount sent — far above the UN's Sustainable Development Goal target of 3%, and a multiple of what a domestic transfer costs. For a migrant worker sending \$200 home, a 6% cost is \$12 lost on every transfer, money that does not reach the family. The mechanism is exactly the fee-stack and FX-spread chain we walked through, made worse for small amounts by fixed per-hop charges and by de-risking thinning the cheap routes. This is the single biggest human cost of correspondent banking's structure, and it is the explicit target of the reforms.

**De-risking and the shrinking network (2011 onward).** Industry and Financial Stability Board data show the number of active correspondent relationships falling year after year since around 2011, even as transaction volumes rose. Regulators have publicly worried that some economies are being cut off from the global financial system entirely. The driver is the asymmetry described above: the correspondent bears the compliance liability for the respondent's customers, so it exits marginal relationships. The 1MDB, Danske, and HSBC AML cases — each a multi-billion-dollar fine or near-failure — are why every compliance officer would rather lose a small client than risk a headline.

**SVB and the speed of a modern run (2023).** This isn't a cross-border story directly, but it sharpens a key contrast. In March 2023, depositors tried to pull about **\$42 billion** from Silicon Valley Bank in a single day, with roughly another \$100 billion queued — a *digital* run at a speed no 20th-century bank ever faced. Domestic money now moves at the speed of an app. Cross-border money, by contrast, still moves at the speed of a relay race across time zones. The gap between how fast money can *flee* (instantly, domestically) and how slowly it can *travel* (days, internationally) is one of the structural stresses of modern banking, and a reason regulators care about cross-border speed as a stability issue, not just a convenience one. (The SVB anatomy is in the [SVB and Credit Suisse 2023 post](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**The fintech and stablecoin end-run (2020s).** Companies like Wise built a business almost entirely on attacking the FX-spread misconception: by netting flows internally and converting at or near the mid-market rate, they undercut the hidden spread that traditional correspondents rely on. Meanwhile, dollar stablecoins (covered in [stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar)) promise to move value across borders in minutes by skipping the correspondent chain entirely — settling on a blockchain ledger instead of nostro/vostro accounts. Whether or not stablecoins ultimately win, their *value proposition is defined by the pain points in this post*: slow, opaque, spread-heavy correspondent banking. The incumbents are not standing still — which brings us to the reforms.

## The reforms: SWIFT gpi and ISO 20022

For decades, the frustrations were tolerated because there was no alternative. Now there is competitive pressure, and two reforms — one operational, one a data-format upgrade — are genuinely changing the experience without throwing away the correspondent model.

**SWIFT gpi (global payments innovation), launched 2017.** gpi is not a new network; it is a *rulebook and a tracker* layered on top of the existing one. Banks that join gpi commit to same-day use of funds (within the receiving bank's business day), transparent and disclosed fees, and unaltered transfer of remittance information. The killer feature is the **UETR** — a *Unique End-to-end Transaction Reference*, a tracking number attached to every payment. For the first time, a bank (and increasingly the customer) can *track a payment end to end*, like a parcel, seeing exactly which correspondent has it right now and what each one charged. gpi attacked the "where is my money?" blindness head-on. The result: the majority of gpi payments are now credited within minutes to a few hours, and a large share within seconds, rather than days.

**ISO 20022 — the data upgrade.** The old MT messages (like MT103) carry *thin*, loosely structured data: short free-text fields where a beneficiary's name and address get truncated and mangled. ISO 20022 is a richer, structured, XML-based messaging standard (the credit-transfer message is `pacs.008`, the equivalent of the old MT103). It carries far more *structured* information — properly separated name, address, purpose-of-payment, and reference fields — which means compliance systems can parse it cleanly instead of guessing. Fewer false-positive sanctions hits, fewer manual holds, faster straight-through processing. The global migration to ISO 20022 for cross-border payments has been underway with a multi-year transition period, and it is one of the largest plumbing changes the industry has ever attempted.

The crucial point: **neither reform replaces the correspondent chain.** gpi and ISO 20022 keep nostro/vostro settlement and the hop-by-hop structure intact. They make it *traceable* (gpi's UETR), *faster* (gpi's same-day rules and cut-off coordination), and *cleaner* (ISO 20022's structured data clears compliance with fewer stops). They are an upgrade to the road, not a new road. The before-and-after below contrasts the legacy flow with the reformed one.

![Comparison of the legacy opaque cross-border flow versus the gpi and ISO 20022 tracked and structured flow](/imgs/blogs/cross-border-payments-correspondent-banking-and-how-swift-really-works-8.png)

#### Worked example: total cost and time before and after gpi

Take our \$10,000 New York → Frankfurt payment and compare the legacy experience with a gpi-plus-ISO-20022 experience.

*Legacy:*

- Visible fees + FX spread: **\$215** (from the earlier worked example), 70% of it the invisible spread.
- Time: **5 calendar days**, with no tracking — "send and pray."
- Transparency: the supplier discovers the \$215 shortfall only after the money arrives.

*Reformed (gpi + ISO 20022):*

- Fees are *disclosed up front* under gpi rules, so the supplier knows to expect roughly **\$215** before sending — same money, but no surprise.
- Cleaner ISO 20022 data means the payment is far less likely to trip a false-positive compliance hold, removing the biggest source of multi-day delay.
- A UETR lets both parties *track* the payment hop by hop in near real time.
- Settlement: with same-day gpi rules and fewer holds, the payment is more likely credited **within hours**, not days — many gpi payments clear in seconds.

The intuition: the reforms barely touched the \$215 of *cost* (the spread is still there, and that requires real FX competition to fix — which is what fintechs supply). What they transformed was the *time* and the *opacity*. You still pay roughly the same toll, but now you can see the road, you know the toll in advance, and the trip is hours instead of days. Fixing transparency and speed turned out to be far easier than fixing cost, because cost is where the incumbents earn their rent.

## The takeaway: read every international payment as a chain, not a wire

If you remember one reframe from this post, make it this: **an international payment is not a wire, it is a chain of bookkeeping entries set in motion by a message.** That single shift dissolves nearly every mystery about cross-border money.

Why is it slow? Because a chain processes one link at a time, across time zones, cut-offs, weekends, and a compliance check at every link. Why is it expensive? Because every link in the chain takes a toll, and the biggest toll — the FX spread — is hidden inside the exchange rate where no statement names it. Why is it opaque? Because the sending bank doesn't even know all the links in advance; intermediaries are chosen along the way. Why does "cutting a country off SWIFT" hurt without freezing its accounts? Because SWIFT is only the *messaging* link; sever it and the other links still exist but become painfully manual. And why are the reforms helping with speed and transparency far more than cost? Because gpi and ISO 20022 upgrade the *road* — tracking, structured data, same-day rules — while leaving the *toll booths* (the correspondents and their FX spreads) exactly where they are.

For how you actually use money, the practical lessons fall out immediately. When you send money abroad, the number that matters is not the advertised fee — it is the *rate you receive against the mid-market rate*, because that spread is usually the real cost. Ask for the all-in figure, not the headline. If speed matters, ask whether your bank is on SWIFT gpi and whether you can get a tracking reference. And recognize that the fintechs and stablecoins competing for your transfer are all, in their different ways, selling you an escape from one specific feature of this post: the spread-heavy, multi-hop correspondent chain. They win exactly where it is weakest.

For how you read a *bank*, the lesson is about the business model. Correspondent banking and transaction services are the part of a bank that earns fees and cheap deposits without the leverage, duration, and credit risk that define the dangerous side of banking. It is the toll-booth business — durable, capital-light, and quietly enormous. When you analyze a big universal bank, the strength of its transaction-banking and FX franchise tells you how much of its profit comes from this safe, sticky rent versus the risky maturity-transformation trade at the heart of the series spine. The plumbing is not glamorous. But the bank that owns the plumbing gets paid every time the world moves money — and the world moves money about thirty million times a day.

## Further reading and cross-links

- [The payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — the foundational view of clearing, settlement, and correspondent banking that this post drills into for the cross-border case.
- [Domestic payment rails: RTGS, ACH, card networks, and instant payments](/blog/trading/banking/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments) — the inside-one-country plumbing whose speed and cost stand in stark contrast to the cross-border chain here.
- [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments) — the geopolitics of the messaging layer: sanctions, cut-offs, and what it really means to remove a bank from SWIFT.
- [Trade finance: letters of credit, guarantees, and supply-chain finance](/blog/trading/banking/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance) — the other half of cross-border banking, where banks de-risk international *trade* rather than just move the money.
- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — why the sticky, low-cost deposits that correspondent and transaction banking generate are so prized.
- [SVB and Credit Suisse 2023: bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the speed-of-money contrast: how fast domestic deposits can flee versus how slowly cross-border money travels.
- [Stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) — the most direct technological challenge to the correspondent-banking chain described here.

*This is educational material about how the payment system works, not financial advice.*
