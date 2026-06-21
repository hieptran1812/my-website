---
title: "Core Banking Systems: The Engine Behind Every Transaction"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "What the core banking system actually is — the ledger of record behind every account, why some banks still post overnight in batch, and why replacing a decades-old core terrifies even the biggest banks."
tags: ["banking", "core-banking", "ledger", "batch-processing", "real-time", "legacy-systems", "cobol", "core-migration", "cloud-native", "fintech"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The core banking system is the bank's ledger of record: the single, authoritative list of every account and every posting to it. Everything else — the app, the ATM, the card network, the regulator's reports — is just a window onto that one ledger.
>
> - A bank does not "have your money" in a vault with your name on it. It has a **number in a database row**, and the core banking system is the database whose number is legally the truth. Every other system must agree with it or be reconciled to it.
> - Many large banks still run a **batch core**: postings queue up during the day and are applied in one **overnight run**, so your intraday balance is an educated estimate, not the truth. Modern cores post in **real time**, second by second.
> - Cores are old. A large share of the world's biggest banks still run **mainframe COBOL** systems written in the 1970s–90s, because the code works, nobody fully understands it anymore, and replacing it is terrifying.
> - The one number to remember: when TSB botched its core migration in **April 2018**, it locked roughly **1.9 million customers** out of their accounts, paid out about **£32.7 million** in redress and costs, and the CEO lost his job. That is why core migrations are the most feared projects in banking.

In the early hours of Sunday, 22 April 2018, the UK bank TSB threw a switch. Over a long weekend it would move the records of about five million customers off a core system rented from its former parent, Lloyds, and onto a brand-new platform built by its Spanish owner, Sabadell. The plan was a "big bang": shut everything down Friday night, copy the entire ledger across, and reopen Monday on the new core. By Monday morning, customers logging into the app saw other people's accounts. Balances were wrong. Payments vanished or duplicated. Some businesses could not pay staff. The outage dragged on for weeks, the regulator opened an investigation, and the chief executive resigned by September. The post-mortem ran to hundreds of pages.

What broke was not the app, not the website, not a marketing database. What broke was the **engine** — the system that holds the official record of who has how much money. When that engine stutters, every single thing a bank does stutters with it, because every single thing a bank does is, underneath, a read or a write to that one ledger. This post is about that engine: what a core banking system is, how it records a transaction, why some banks still process overnight while others post instantly, why the technology underneath is often older than the engineers maintaining it, and why swapping it out is one of the highest-stakes projects a bank will ever attempt.

![Core banking system at the center with channels feeding it and the general ledger and reporting downstream](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-1.png)

The diagram above is the mental model for the whole post. In the middle sits the **core banking system** — the ledger of record. Around the edges sit the **channels**: the branch teller, the mobile app, the ATM and card networks, the open-banking APIs that fintechs plug into. Every one of those channels does the same thing — it asks the core to read a balance or to write a posting. Downstream, the core feeds the bank's **general ledger** (its company-wide accounting) and its **regulatory and risk reporting**. The whole bank is, structurally, a fan of windows onto one ledger. Get the ledger right and everything works; corrupt the ledger and nothing does.

## Foundations: the ledger of record and the language of posting

Before any of the clever parts, we need a handful of plain definitions. None of them require finance background — they are bookkeeping ideas that are five centuries old, dressed up in modern hardware.

### What is a ledger, and what does "of record" mean?

A **ledger** is just a list of accounts and the running balance of each. Your grandmother's notebook tracking who owes the corner shop how much is a ledger. A bank's ledger is the same idea at enormous scale: one row per account, a balance on each row, and a history of every change.

The phrase **"of record"** is the load-bearing part. It means: when two systems disagree about your balance, *this* one wins. It is the legal, authoritative truth. If your mobile app shows \$1,000 and the core shows \$900, you have \$900 — the app is simply out of date or wrong, and it gets corrected to match the core, never the other way around. The **core banking system** is the software that owns the ledger of record for a bank's deposit and loan accounts. It is sometimes literally called the "system of record," abbreviated SoR.

This is worth sitting with, because it overturns a common picture. People assume their money sits somewhere physical with their name on it. It does not. Your "money in the bank" is a **liability the bank owes you**, recorded as a number in a row of the core's database. The cash itself has been lent out (that is the bank's whole business — borrow short from depositors, lend long to borrowers, as the rest of this series keeps coming back to). What makes the number *real* is that the core is the agreed source of truth, backed by law and regulation. The engine is not a vault. It is a notebook that everyone has agreed to believe.

### What is a posting?

A **posting** is a single entry written to the ledger that changes a balance. "Add \$500 to account A" is a posting. "Subtract \$500 from account B" is a posting. The verb bankers use is *to post* a transaction.

Crucially, postings come in matched pairs, because of **double-entry bookkeeping** — the 500-year-old rule that every transaction touches at least two accounts and the changes must sum to zero. If you transfer \$500 to a friend, the core does not write one entry; it writes two: a **debit** (a subtraction) on your account and a **credit** (an addition) on theirs. The two entries are equal and opposite, so the bank's books always balance. (In banker's language the words *debit* and *credit* have a precise technical sense that can be the reverse of everyday usage, but for a customer deposit account the simple reading holds: a debit reduces your balance, a credit increases it.)

Why insist on pairs? Because it makes errors visible. If a single posting ever fails to have its matching opposite, the ledger no longer sums to zero, and that imbalance is a giant red flag that something is broken. Double-entry is the oldest error-detection system in finance, and the core enforces it on every transaction, millions of times a day.

### What is batch, and what is real-time?

These two words describe *when* the core does the work.

**Real-time** means the core applies a posting the instant it arrives. You tap to pay, and within a fraction of a second your balance has actually changed in the ledger of record.

**Batch** means the core collects postings into a pile during the day and applies them all at once in a single scheduled run — almost always overnight. This run is called the **end-of-day** (EOD) processing, or simply *the batch*. During the day, the bank tracks your activity in a separate, provisional "memo" balance so it can stop you overspending, but the *real* posting to the ledger of record does not happen until the nightly batch.

If that sounds antique, it is — and we will spend a whole section on why banks still do it. For now, hold the contrast: a batch core knows the truth only once a night; a real-time core knows it continuously.

### What is the general ledger, and how is it different from the core?

The **core** holds *customer* accounts — your checking account, your savings account, your mortgage balance. The **general ledger** (GL) holds the *bank's own* accounting: total deposits, total loans, interest income, expenses, the entries that roll up into the bank's financial statements. The core feeds the GL: at end of day, the millions of individual customer postings are summarized into a handful of GL entries ("total deposits rose by \$40 million today"). If you want the bank-wide view those numbers produce, this series covers it in [reading a bank balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity). The point here is that the core is the *retail* truth and the GL is the *corporate* truth, and the two must reconcile — a recurring theme below.

With those five terms — ledger of record, posting, double-entry, batch vs real-time, general ledger — we have enough vocabulary to open the engine up.

## The account and ledger model: what the core actually stores

Strip a core banking system down to its skeleton and there are three things: **accounts**, a **ledger of postings**, and the **rules** that turn a request into postings. Everything else — interest, fees, statements, holds, overdrafts — is built on top of those three.

### Accounts are rows; balances are derived from postings

The simplest mental version of a core is a giant table with one row per account: an account number, an owner, a product type (checking, savings, term deposit, loan), and a balance. The subtlety is where the balance comes from. In a well-built core, the balance is not the *primary* fact — the **stream of postings is**. The balance is what you get when you add up every posting on the account from the day it opened. This is the same principle behind a database's [write-ahead log](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability): the immutable list of changes is the truth, and the current state is a replay of that list.

Storing the postings rather than just the balance is what makes a bank auditable. You can always answer "how did this balance get to be \$1,732.18?" by walking the history. A core that only stored the latest balance would have no way to prove it, and no way to recover if the number got corrupted.

#### Worked example: how a balance is built from postings

You open an account and over a week the core records these postings:

- Open with a deposit: **+\$1,000.00**
- Card purchase at a shop: **−\$42.50**
- Salary credit: **+\$2,400.00**
- ATM withdrawal: **−\$200.00**
- Monthly account fee: **−\$5.00**

The balance is simply the sum: \$1,000.00 − \$42.50 + \$2,400.00 − \$200.00 − \$5.00 = **\$3,152.50**. The core does not "store \$3,152.50 and edit it" as the main fact; it stores the five postings, and \$3,152.50 is the answer to adding them up. If a regulator or a customer disputes the balance, the bank points at the five lines. **The intuition: in a core, the balance is a conclusion, not an input — the postings are the evidence.**

### Every transaction is a balanced set of postings

Now the double-entry rule from Foundations does real work. When money moves, the core writes the entries on *both* sides so the books stay balanced. Internally, a transfer between two customers of the same bank is the cleanest case: one customer's account is debited, the other's is credited, and the bank's own position has not changed at all — it still owes the same total to its depositors, just split differently between them.

When money leaves the bank entirely — say you pay someone at a different bank — the second leg of the entry lands on one of the bank's own internal accounts (a settlement or "nostro" account) rather than on another customer. The customer leg and the bank-internal leg still net to zero. Actually moving the funds between the two banks is **settlement**, a separate step handled by the payment rails, which this series covers in [the payments business](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks). The core's job is to get the *bookkeeping* right and to flag the funds for settlement; it is not itself the wire that carries the cash.

![Pipeline of a transaction posting: authorize, debit payer, credit payee, update balances, settle](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-2.png)

The pipeline above traces a single \$500 transfer through the core. First **authorize**: check the payer actually has \$500 available and is within any limits. Then **debit the payer** by \$500 and **credit the payee** by \$500 — the matched pair. Then **update balances** so both accounts reflect the change. Finally **settle** if the money is leaving the bank. Notice the colors: the debit is red (money out of one account) and the credit is green (money into the other), the sign convention this series uses everywhere.

#### Worked example: a posting that debits and credits two accounts

You send your landlord \$500. Both of you bank at the same institution. Before the transfer: you have \$3,152.50, your landlord has \$8,000.00. The core writes two postings against the same transaction reference:

- Debit your account: **−\$500.00** → your new balance \$2,652.50
- Credit landlord's account: **+\$500.00** → their new balance \$8,500.00

Sum of the two postings: −\$500.00 + \$500.00 = **\$0.00**. The bank's total deposit liability is unchanged — it owed \$11,152.50 across the two of you before, and \$11,152.50 after; only the split moved. The transaction is *balanced*, and because it nets to zero the core can prove it did not create or destroy money. **The intuition: a transfer inside a bank is pure bookkeeping — no cash moves, two numbers just swap a bit of their size.**

### Holds, available balance, and why your balance has two numbers

One more piece of the model explains a daily frustration. Your account often shows two figures: a **ledger balance** (the posted truth) and an **available balance** (what you can actually spend right now). They differ because of **holds** — amounts earmarked but not yet posted. When you check into a hotel, the hotel asks your bank to *authorize* a hold (say \$300) against your card. The core has not posted a \$300 debit; it has reduced your *available* balance by \$300 while leaving the *ledger* balance untouched, pending the real charge. This split — authorize now, post later — is at the heart of the batch-versus-real-time question, because in a batch world *everything* works a bit like a hold during the day.

### Idempotency: why a posting must happen exactly once

There is a property of the posting model that customers never see but that engineers obsess over: a posting must apply **exactly once**, even when the message asking for it is sent twice. Networks drop packets and retry; a channel that does not hear "done" will re-send the request. If the core posted a \$500 debit twice because the request arrived twice, it would silently steal \$500 from a customer. So every posting carries a unique transaction reference, and the core records that it has already processed that reference. A second arrival of the same reference is recognized and ignored — applied zero further times. This property is called **idempotency**, and it is the difference between a ledger you can trust and one that drifts every time the network hiccups. It is also why "I got charged twice" is a real bug worth a post-mortem rather than a shrug: in a correct core it should be impossible, so when it happens, the exactly-once guarantee has failed somewhere in the chain between the channel and the ledger.

### Reconciliation: proving the ledger still sums to the truth

A bank does not simply trust that its postings are correct — it *proves* it, every single day, through **reconciliation**: the process of checking that two independently-kept records of the same thing agree. The core reconciles in several directions at once. It checks that the day's customer postings sum to the movements in the general ledger (the retail truth ties to the corporate truth). It checks that the funds it expected to receive and send through the payment networks match what the networks actually cleared (the **nostro reconciliation** — agreeing the bank's record of its account at another bank against that other bank's statement). And it checks internal control accounts — the temporary "suspense" accounts where a posting waits when its other leg is delayed — back down to zero, because a suspense account that does not clear is a transaction that got lost.

When a reconciliation *breaks* — when the two records do not agree by even a cent — the bank has a problem it must chase down before it can close the books with confidence. A persistent break is how frauds and bugs are caught: the famous Barings collapse hid losses in an unreconciled error account (the "88888" account) for years, a scandal this series covers separately. The lesson for the core is that double-entry plus daily reconciliation is the immune system of the ledger. Double-entry makes every transaction balance; reconciliation proves the whole pile still balances against the outside world. A core that posts beautifully but reconciles poorly is a core you cannot trust, because you have no proof its truth is still the truth.

## Batch versus real-time: the two clocks a bank can run on

Here is the design choice that most shapes how a bank's technology feels to a customer. Does the core post transactions the instant they happen, or does it queue them and post them all overnight?

### The batch core: knowing the truth once a night

For decades, the only practical answer was batch. Computers were expensive and slow, and the cheap way to use them was to let work pile up and then crunch it all in one efficient pass when nothing else was competing for the machine — overnight, when branches were closed. So the classic core works like this: all day, transactions are captured and validated, but they are written to a queue rather than posted to the ledger of record. A separate, provisional **memo balance** is maintained so the bank can decide whether to approve a card swipe or an ATM withdrawal. Then, after a daily **cutoff** (often around 6 p.m. or after the last clearing exchange), the bank "closes the books" and runs the batch: it posts the day's queued transactions, accrues interest, applies fees, generates statements, and reconciles to the general ledger. By morning, the ledger of record reflects yesterday, and a new day's queue begins.

![Batch core posts overnight so daytime balances are estimates, real-time core posts each transaction instantly](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-3.png)

The before-and-after above contrasts the two. On the batch side (left, in caution colors), the daytime balance is a memo estimate, postings queue in a file all day, the end-of-day batch posts everything at once, and the *real* balance is only known the next morning. On the real-time side (right, in green and blue), the balance is live and accurate now, each posting hits the ledger instantly, there is no nightly cutoff, and the real balance is known every second.

The batch model is the reason for a string of behaviors you have probably noticed. Why does a deposit "clear" the next business day? Because it posts in tonight's batch. Why do transactions sometimes appear in a different order than you made them, occasionally pushing you into an overdraft you did not expect? Because the batch can be configured to post the day's debits in a chosen order (notoriously, largest-first in some past cases, which maximized overdraft fees). Why is there "no banking on weekends"? Because the batch runs on business days and the books do not close on a Sunday. The customer experiences these as quirks; they are direct fingerprints of a batch core.

#### Worked example: the batch window and a stale balance

A batch bank cuts off at 18:00. At 09:00 you have a ledger balance of \$2,652.50. During the day you receive a \$1,000 salary credit (captured at 11:00) and spend \$300 on a card (captured at 15:00). All day, the ledger of record still says **\$2,652.50** — none of it has posted. Your *memo/available* balance moves to \$2,652.50 + \$1,000 − \$300 = \$3,352.50 so the bank can authorize your spending. But if at 16:00 you ask "what is my balance," the system of record literally still holds yesterday's \$2,652.50. Only in tonight's batch, after the 18:00 cutoff, do the two postings hit the ledger, and tomorrow morning the ledger of record reads **\$3,352.50**. For about 12 hours, the bank's official truth was 27% lower than your real spendable position. **The intuition: under a batch core, your "balance" during the day is a forecast the bank is making about tonight's posting run — accurate enough to trade on, but not the legal truth yet.**

### The real-time core: continuous truth

A **real-time core** collapses that gap. Each posting hits the ledger of record the moment it is authorized; the memo balance and the ledger balance are the same number because there is no queue. There is no nightly cutoff that freezes the books — interest can accrue continuously, statements can be produced on demand, and the balance you see *is* the truth, not a forecast.

This is not free. Real-time posting means the core must handle a high, spiky rate of individual writes correctly and durably, all day, with no quiet overnight window to catch up. It is a far harder engineering problem than crunching a tidy file at 2 a.m., which is exactly why it took cheap, fast, horizontally-scalable computing to make real-time cores mainstream. The payoff is that the bank's truth and the customer's experience finally match: when a neobank shows you "you just spent \$4.20 on coffee" with a push notification two seconds later, that is a real-time core posting and confirming the entry in the moment.

### "Real-time" still does not mean "settled"

A vital subtlety: a real-time core can update *your* balance instantly even when the *money* has not finished moving between banks. Posting (the bookkeeping) and settlement (the actual transfer of funds across institutions) are different events. Instant-payment schemes give the customer real-time posting and near-instant settlement, but card networks famously do not — the merchant gets authorized in seconds, while settlement between the banks can take a day or more. So "I saw it instantly in my app" tells you the *core* is real-time; it does not tell you the cash has cleared. That distinction is the whole subject of [the payments business](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks); here we only need to know the core can run faster than the rails beneath it.

## A closer look at the end-of-day batch cycle

Even banks moving toward real-time keep a nightly cycle for the things that genuinely happen once a day — and understanding what runs in the dark explains a lot about how banks behave. The batch is not one job; it is a long, dependency-ordered chain of jobs, and if one link fails the whole chain can stall.

![End-of-day batch cycle timeline from cutoff through interest, statements, reconciliation, and reopen](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-4.png)

The timeline above walks a representative cycle. At the **18:00 cutoff** the books close for the day — no new postings join today's run. Around **20:00** the bank exchanges **clearing files** with the payment networks: the inbound file of payments to credit customers, the outbound file of payments to debit them. By **22:00** the core runs **interest accrual and fees** — calculating a day's interest on every deposit and loan, applying monthly charges. Around **01:00** it generates **statements and reports**. By **04:00** it **reconciles to the general ledger**, proving the millions of customer postings sum to the bank-wide figures. By **06:00** it **reopens** with live balances for the new day. Miss the window — say a job hangs at 03:00 — and the bank may open late, with stale balances, which customers experience as an outage.

#### Worked example: why the batch window is a hard deadline

Suppose a mid-size bank processes 8 million postings in its nightly batch, and its processing rate is about 1,000 postings per second. The arithmetic: 8,000,000 ÷ 1,000 = 8,000 seconds ≈ **2 hours and 13 minutes** of pure posting time. Add interest accrual, statement generation, and reconciliation, and the full batch needs perhaps 5 hours. If the cutoff is 18:00 and branches must reopen with accurate balances by 06:00, the bank has a 12-hour window for a 5-hour job — comfortable. But suppose volumes grow 50% (to 12 million postings) and a hardware fault halves throughput to 500/second for part of the run. Posting time alone becomes 12,000,000 ÷ 500 = 24,000 seconds ≈ **6 hours and 40 minutes**, and the full batch blows past the 06:00 deadline. The bank opens late on stale data. **The intuition: a batch core is a nightly race against the clock, and growth or a glitch can turn a comfortable margin into a missed open — which is exactly why banks dread anything that lengthens the batch.**

## The legacy problem: why your bank runs on code older than its engineers

Here is the fact that surprises people most. The core banking systems running the largest banks on Earth are, in many cases, decades old, written in **COBOL** (a programming language from 1959) and running on **mainframes** (room-filling, fantastically reliable IBM computers whose lineage goes back to the 1960s). This is not a fringe phenomenon. By widely cited industry estimates, a large majority of the world's biggest banks still run their cores on mainframe COBOL stacks, and an enormous share of daily transaction volume — by some counts the bulk of ATM swipes and card transactions worldwide — touches COBOL at some point.

![Illustrative share of large-bank cores by technology era, most still on mainframe COBOL](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-5.png)

The chart above is an illustrative split (not precise data — labeled as such in the source line) of where large-bank cores run by technology era. The dominant amber bar is the **1970s–90s mainframe COBOL** stack; a middle band were rewritten on mid-range platforms in the 2000s; and only a small green slice are genuinely **cloud-native cores** built since around 2015. The exact percentages vary by survey, but every credible survey tells the same story: the old stuff still runs most of the money.

### Why is the core so old?

It is tempting to call this incompetence. It is closer to the opposite. There are hard, rational reasons banks still run forty-year-old cores:

**It works, and it almost never fails.** A mainframe COBOL core that has been hardened over four decades is astonishingly reliable — uptime measured in "minutes of downtime per year." That reliability was paid for in the hard currency of decades of bug-fixing. A shiny new system has not yet earned that trust, and a bank's tolerance for an outage on its ledger of record is close to zero.

**It is load-bearing and tightly coupled.** Over forty years, thousands of other systems — branch terminals, ATMs, card processors, regulatory reports, fraud checks, statement printers — have been wired into the core. Each connection encodes assumptions about how the core behaves. Replacing the core means re-wiring all of them, and any one missed wire is a production incident. This is the [shared-data and distributed-monolith trap](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) that software teams know well, except the shared database here is the legal record of millions of people's money.

**Nobody fully understands it anymore.** The engineers who wrote the original code have retired or died. The business rules — how a particular fee is calculated, what edge case that odd branch handles — survive only in the code itself, which has been patched by hundreds of hands over decades. The code *is* the documentation, and the documentation is in COBOL. Re-implementing it means first reverse-engineering it, with the certainty that you will miss something.

**The COBOL talent has aged out.** The pool of engineers fluent in COBOL and mainframe operations is shrinking as a generation retires, which paradoxically makes the old core both harder to maintain *and* harder to safely replace, because you need that scarce expertise to do either.

#### Worked example: the cost of maintaining versus replacing a legacy core

A regional bank runs a 35-year-old COBOL core. It spends, say, \$40 million a year keeping it alive — mainframe licensing, a specialist COBOL team, vendor support, and the slow, expensive process of bolting on each new feature. Replacing it with a modern core is quoted as a \$300 million, five-year program. The bank does the math: continuing as-is costs \$40m × 5 = **\$200 million** over the same five years, with zero migration risk and the lights staying on. The replacement costs **\$300 million** *plus* the (very real) risk of a TSB-style disaster that could cost tens of millions more in redress and a chunk of the customer base. On a pure five-year cash basis, *not* migrating is \$100 million cheaper and far less risky — so the board defers, again. The catch is the *later* term: maintenance cost rises as talent gets scarcer, and the inability to ship modern features quietly erodes the franchise. **The intuition: legacy cores survive because the next year of "do nothing" is always cheaper and safer than the next year of "replace it" — right up until the day it isn't.**

This is the trap. Each year, deferral is the locally rational choice, and so the core ages another year. The bill is real but deferred; the risk of the cure is immediate and vivid. Multiply that across an industry and you get the world we have: trillions of dollars run on code from the disco era.

### The middle path: hollowing out the core without replacing it

Because both extremes — keep the legacy core forever, or rip it out in one go — are painful, most large banks have settled on a middle path: leave the old core running as the ledger of record, but wrap it and slowly drain its responsibilities. The pattern has a few common moves.

The first is to put an **API layer** in front of the legacy core, so new channels and partners talk to a clean, modern interface while the old COBOL keeps doing the actual posting behind it. The customers and the fintech partners never touch the mainframe directly; they see a tidy API, and the ugliness is contained. The second is to **lift functions out** of the monolithic core one at a time — move the product-configuration logic, or the interest-calculation engine, or the statements, onto modern systems that call back into the core only for the authoritative balance. Over years, the core is gradually *hollowed out*: it still owns the ledger, but more and more of the surrounding intelligence has moved to systems that are cheap to change. The third is the **coexistence run** — stand up a modern core beside the legacy one and migrate products or customer segments across in deliberate waves, with both ledgers reconciled against each other until the bank trusts the new one enough to retire the old.

None of these is fast or cheap, and all of them spend years living with *two* systems and the reconciliation burden that creates. But they share one virtue that the board cares about above all else: at no point is the entire franchise riding on a single weekend cutover. The blast radius of any one step is small. This is the same instinct that drove the [strangler-fig migration pattern](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) in software — wrap the old system, divert work piece by piece, shrink the legacy until it can be switched off without anyone noticing. Applied to a bank, it turns the terrifying, all-at-once core migration into a long series of survivable small ones. The price is time and dual-running cost; the prize is never appearing in a headline like TSB's.

## Why a core migration terrifies banks: the TSB disaster

If running an old core is the slow, quiet risk, *replacing* it is the loud, sudden one. A core migration is the open-heart surgery of banking: you have to swap the organ that everything else depends on, and the patient cannot afford to be unconscious for long. The reason it is so dangerous is structural.

![Core migration risk graph: channels, payments, and reporting all depend on the new core and a fault becomes customer harm](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-8.png)

The graph above shows why. When the **new core goes live**, three things immediately depend on it being correct: **logins and the mobile app**, **payments and cards**, and **balances and statements**. A fault in any of them converges on the same place — **locked-out customers, fines, and lost trust**. There is no part of the bank you can quarantine from a bad core, because the core is what the whole bank reads and writes. A bug that would be a minor incident in a peripheral system becomes a front-page event when it is in the ledger of record.

### The two migration strategies, and their distinct fears

There are broadly two ways to move:

**Big bang.** Freeze the old core, copy all the data to the new one over a maintenance window (usually a weekend), and reopen on the new system. The appeal is that you run only one core afterward — no expensive period of keeping two systems in sync. The terror is that there is no going back gracefully: if Monday morning is broken, millions of customers are broken at once, and rolling back after real transactions have posted to the new core is itself a fraught data-recovery exercise. TSB went big bang.

**Phased / parallel run.** Move customers or products in waves, often running the old and new cores side by side and reconciling them until you trust the new one, then cutting over gradually. This is the banking cousin of the [strangler-fig pattern](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) software teams use to retire a monolith: you wrap the old system, route new traffic to the new one piece by piece, and shrink the old one until it is gone. It contains the blast radius — a problem hits one wave, not everyone — but it is slower, costs more (you pay for two cores at once), and the data-synchronization between old and new is its own deep source of bugs. The closest software analogue for the data side is a [zero-downtime schema migration](/blog/software-development/database/zero-downtime-schema-migrations): change the underlying records while the system keeps serving live traffic, never with a hard stop.

### What actually went wrong at TSB

The April 2018 TSB migration is the case study every bank technologist now studies, because it was a big bang that detonated. TSB moved roughly five million customers' records onto the new Sabadell-built platform over a weekend. When it reopened, the failures were exactly the converging kind the graph predicts: customers were locked out of online and mobile banking; some saw *other people's* account details; balances and transactions were wrong or missing; payments failed or duplicated; and the call centers and branches were overwhelmed because their tools depended on the same broken core. The disruption affected around **1.9 million** customers and dragged on for weeks rather than hours. An independent investigation pointed to inadequate testing of the new platform at full production scale and a "big bang" cutover with insufficient ability to cope when things went wrong. TSB ultimately reported costs and redress on the order of **£330 million** related to the migration (post-migration costs, customer redress, fraud, and foregone income), its CEO resigned, and years later the regulators fined the bank about **£48.65 million** for the operational-resilience and management failings. (Different figures get quoted for different slices — the redress-and-remediation line was reported around £32.7 million, the all-in cost around £330 million — but the direction is unambiguous: a botched core migration is a nine-figure event.)

#### Worked example: sizing a core-migration project's cost and risk

Frame a big-bank migration as a decision under risk. A bank budgets a **\$300 million**, five-year core replacement. It estimates a **15% probability** of a "serious incident" on cutover (a TSB-class event). It models the cost of such an incident — redress, fines, fraud, remediation, and lost customers — at **\$350 million**, anchored on the TSB experience. The *risk-adjusted* expected cost of the project is therefore the certain build cost plus the probability-weighted disaster cost: \$300m + (0.15 × \$350m) = \$300m + **\$52.5m** = **\$352.5 million**. Now compare two ways to spend part of that budget. Big bang keeps the build cheaper but leaves the 15% tail. A phased/parallel run might add \$80 million in dual-running and reconciliation cost but cut the serious-incident probability to 4%: \$380m + (0.04 × \$350m) = \$380m + **\$14m** = **\$394 million** expected — *more* expected cost, but with the catastrophic tail slashed by nearly three-quarters. **The intuition: a board choosing a migration strategy is really pricing a tail risk — and after TSB, many decided that paying tens of millions extra to make the disaster scenario four times less likely is the cheap option.**

That worked example is the whole psychology of core migration in one sum. The expected costs of the two strategies can be close; what differs is the *shape* of the risk. Big bang is cheaper on average and ruinous in the tail. Phased is dearer on average and survivable in the tail. A bank that has watched a peer make headlines for a broken core will pay a lot to move the disaster from "possible" to "improbable."

### Why downtime, specifically, is so expensive

The migration tail is dominated by **downtime** — hours when the core cannot serve. Putting a number on an hour of dead core explains why banks are so cautious.

![Illustrative cost of one hour of core downtime rising sharply with bank size](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-7.png)

The chart above is illustrative (labeled in the source) but order-of-magnitude realistic: an hour of full core downtime might cost a community bank tens of thousands of dollars, a regional bank under a million, a national bank several million, and a global G-SIB on the order of **\$12 million an hour** once you add up failed payments, lost fee and interest income, SLA penalties, fraud that slips through while controls are down, emergency staffing, remediation, and the slow bleed of customers who decide they cannot trust the bank with their money. The cost scales super-linearly with size because the giants run more transactions, more obligations, and more reputational exposure per minute.

#### Worked example: the cost of an hour of core downtime

Take a national bank whose all-in downtime cost is **\$4.5 million per hour**. The TSB-style outage was not measured in hours but effectively in days of degraded service. Even a comparatively contained incident of, say, **6 hours** of full core unavailability would cost 6 × \$4.5m = **\$27 million** in direct terms alone — before any regulatory fine, before any customer who closes their account and tells ten friends. Stretch that to a multi-day TSB-scale event and you are quickly in the hundreds of millions, which is exactly where TSB landed. Set that against the **\$300 million** project budget from earlier, and a single bad cutover can wipe out a meaningful fraction of the whole program's value. **The intuition: the core is the one system where downtime is priced in millions per hour, so the entire migration is engineered around the single goal of never being down — and a strategy that risks days of downtime is gambling the project's whole economics.**

## The modern cloud-native core: Thought Machine, Mambu, and the new model

If legacy cores are the problem and migrations are the danger, the answer the last decade produced is a new kind of core built for the cloud. Vendors like **Thought Machine** (its platform is called Vault Core) and **Mambu** offer cores that differ from the old mainframe model in a few deep ways.

**Cloud-native and horizontally scalable.** Instead of one enormous, vertically-scaled mainframe, a modern core runs on commodity cloud infrastructure and scales out by adding machines. This is what makes real-time posting at high volume economical — you are no longer rationing a single expensive computer's overnight window.

**Real-time first.** These cores are designed to post continuously, not in a nightly batch. Batch becomes the exception (for things that genuinely happen once a day) rather than the rule.

**Configurable products as code.** The old core baked product rules deep into COBOL; changing a fee or launching a new savings product could take months. Modern cores let the bank define products in a higher-level, sandboxed way — Thought Machine calls these "smart contracts" for financial products (not blockchain; just configurable logic) — so a new product can ship in weeks. This is the feature that legacy banks most envy, because product velocity is competitive survival.

**API-first.** A modern core exposes clean APIs so channels and fintech partners can plug in without bespoke wiring, which is the foundation of open banking and banking-as-a-service.

This is exactly what most **neobanks** (app-only banks with no branches) run on, and it is a big part of why they can ship features fast and show you a real-time balance — a model this series covers in [digital banking and the neobank business model](/blog/trading/banking/digital-banking-and-the-neobank-business-model). It is also why some incumbents are spinning up a brand-new bank on a cloud-native core *alongside* their legacy bank (JPMorgan's UK Chase, Goldman's Marcus, NatWest's Mettle, and others), migrating customers over time rather than attempting a big-bang swap of the mothership. Build the new core empty, fill it gradually, and you sidestep the worst of the migration tail.

### The honest limits of the new cores

Calm and honest, as this series insists: a cloud-native core is not a free lunch. It has not yet survived four decades of every conceivable edge case, so its reliability is *promised* rather than *proven* at the scale of a 50-million-customer incumbent. Running on the public cloud introduces a dependence on a handful of cloud providers — a concentration risk regulators now scrutinize. And configurable products as code is powerful precisely because it lets you change the ledger's behavior quickly, which is also a way to introduce a subtle, fast-moving bug into the system of record. The new cores trade *aged reliability* for *speed and flexibility*. For a neobank starting from zero, that trade is obviously worth it. For an incumbent with a working forty-year-old core and millions of customers, it is the hardest decision the technology team will ever frame for the board.

## Build versus buy: who should own the engine

A bank that decides to modernize faces one more fork: should it **buy** a packaged core from a vendor (Temenos, FIS, Finastra, Thought Machine, Mambu), **build** its own from scratch, or **keep** the legacy core and bolt on around it? Each path trades the same four things differently.

![Matrix comparing keep legacy, buy a core, and build your own across cost, risk, speed, and control](/imgs/blogs/core-banking-systems-the-engine-behind-every-transaction-6.png)

The matrix above lays out the trade. Reading down the columns: **keeping legacy** has low cost now (but high cost later), no migration risk (nothing moves), slow speed to ship (you can only bolt things on), and leaves you locked in to an aging vendor. **Buying a core** carries a high upfront cost (license plus the migration project) and high migration risk, but fast speed to ship (the vendor already built it) and limited control (you live within the vendor's roadmap). **Building your own** is the most expensive (years of engineering) and the highest risk (new and untested), is slow to ship initially, but gives you full control — you own the roadmap and owe no vendor.

The practical answer for most banks is **buy**, because the core is not where a bank differentiates — customers do not choose a bank for its posting engine — and reinventing a ledger of record is a way to spend a fortune re-learning lessons the vendors already learned. The famous exceptions are the giants and the digital-native challengers with the engineering depth to build (or heavily customize) a core as a genuine competitive asset. A handful of the world's largest banks run substantially in-house cores; most everyone else buys, and the entire neobank wave is built on bought cloud-native cores plus their own front-ends.

#### Worked example: the build-versus-buy break-even

A mid-size bank compares two five-year plans. **Buy**: \$60 million license and implementation, plus \$8 million a year to run and maintain it, so \$60m + (5 × \$8m) = \$60m + \$40m = **\$100 million** over five years, live in about 18 months. **Build**: \$25 million a year for a 40-engineer team over five years = **\$125 million**, live in about 3 years if all goes well — and "all goes well" is doing heavy lifting in a from-scratch ledger of record. Build is \$25 million dearer, ships 18 months later, and carries more execution risk; its only advantage is total control of the roadmap. For a bank whose strategy does not hinge on a uniquely clever core, **buy wins** — the \$25 million and the 18 months are better spent on the products customers actually choose the bank for. **The intuition: build only the core if the core itself is your edge; otherwise the engine is a utility you should rent, and spend your scarce engineering on the windows, not the ledger.**

## Common misconceptions

**"The bank keeps my money in an account with my name on it."** No. Your balance is a number in a row of the core's database, recording a debt the bank owes you; the actual cash has largely been lent out. What makes the number real is that the core is the agreed, law-backed ledger of record — not a vault. A bank with \$2 trillion in deposits does not have \$2 trillion in cash; it has \$2 trillion of postings on a ledger and a thin slice of actual reserves, which is the whole maturity-transformation trade this series keeps returning to.

**"When the app shows my balance, that's my money, in real time."** Often not. If the bank runs a batch core, the app may be showing a *memo* balance — a provisional figure that has not yet posted to the ledger of record. The legal truth is updated in tonight's batch. This is why a transaction can "disappear and reappear," why a deposit "clears" the next day, and why two of your devices can briefly disagree.

**"Modern banks all run modern technology."** A large majority of the biggest banks still run mainframe COBOL cores written in the 1970s–90s, and a huge share of global card and ATM traffic still touches COBOL. The app on your phone may be brand new; the ledger it talks to may be older than your phone's operating system by forty years.

**"Replacing the core is just a big IT project."** It is the riskiest project a bank runs, because the core is the one system everything else depends on. TSB's 2018 migration affected around 1.9 million customers, cost on the order of £330 million all-in, drew a regulatory fine later put at about £48.65 million, and cost the CEO his job. A failed core migration is an existential, board-level event, not a delayed IT ticket.

**"Real-time core means my payment has actually settled."** No. A real-time core posts the *bookkeeping* instantly, but settlement — the actual movement of funds between banks — is a separate step that can still take a day or more, especially on card rails. "Instant in the app" tells you the core is fast; it does not tell you the cash has cleared.

## How it shows up in real banks

**TSB, April 2018 — the migration that became the cautionary tale.** TSB attempted a big-bang weekend cutover of around five million customers from a Lloyds-rented core to a new Sabadell-built platform. It reopened broken: customers locked out, some seeing strangers' accounts, payments failing or duplicating, balances wrong. About 1.9 million customers were affected; the disruption lasted weeks. The all-in cost was reported around £330 million (including roughly £32.7 million of direct redress and remediation in one accounting), the CEO resigned within months, and regulators later fined the bank about £48.65 million for the operational-resilience failings. Every figure in this post about migration risk traces back to the simple fact that this is what a big bang can do.

**The incumbents still on mainframes.** When you use a card issued by one of the world's largest retail banks, there is a strong chance the authorization and the eventual posting touch a COBOL program on an IBM mainframe that has been running, with patches, since before the engineers maintaining it were born. This is not hidden — it shows up in banks' technology disclosures and in the persistent demand for mainframe and COBOL specialists. The reliability is genuinely excellent; the cost is rigidity, slow product launches, and an ever-scarcer talent pool.

**The neobanks born real-time.** App-only banks such as the cloud-native challengers launched in the 2010s never had a legacy core to carry, so they ran real-time, cloud-native cores from day one. That is why they could show you a balance that updates the instant you spend, push a notification two seconds later, and ship a new feature in weeks rather than quarters. The economics of those banks — high feature velocity but a hard road to profitability — are a separate story this series tells in [digital banking and the neobank business model](/blog/trading/banking/digital-banking-and-the-neobank-business-model); the technology enabler is the modern core.

**The incumbents building a "new bank" on the side.** Faced with the migration tail, several giants chose to start a fresh, cloud-native bank alongside the legacy mothership — JPMorgan's UK Chase, Goldman's Marcus, and others — building the new core empty and filling it with new customers over time. It is the strangler-fig move applied to a whole institution: grow the new core, shrink the old one, and never bet the entire franchise on a single weekend cutover.

**The batch fingerprint on your everyday banking.** Every time a deposit "clears next business day," a Friday-evening transfer lands Monday, an overdraft appears because debits posted in an unexpected order, or your balance briefly disagrees across devices, you are seeing a batch core's nightly rhythm leak through to the surface. The ledger of record told the truth at 6 a.m.; everything in between was a forecast.

## The takeaway / How to use this

Strip away the COBOL and the cloud, and a core banking system is one idea: a ledger of record that everyone has agreed to believe. The bank's whole business — borrow short from depositors, lend long to borrowers, earn the spread, survive on confidence and a thin equity cushion — is recorded, line by line, in that ledger. The core is where the maturity-transformation machine keeps its books. That is why it matters so much, and why messing with it is so dangerous: corrupt the ledger and you have not broken a feature, you have broken the bank's grip on the truth about who owns what, which is the only thing standing between an orderly bank and a panic.

So here is how to *read* a bank through its core. When you hear a bank still runs batch, you have learned something real: its truth is a day old, its product launches are slow, and its technology risk is the quiet, deferred kind that compounds every year the board says "not yet." When you hear a bank runs a real-time, cloud-native core, you have learned the opposite: it can move fast and show customers the truth in the moment, but it is trading proven reliability for promised reliability, and it carries a concentration on a handful of cloud providers. When you hear a bank is *migrating* its core, sit up — that is the single highest-stakes project it will run this decade, and the difference between a big-bang and a phased approach is the difference between betting the franchise on one weekend and merely spending a fortune to de-risk the tail. And when your own app shows you a balance, you now know whether to trust it as the truth or read it as a forecast, depending on which clock the engine behind it runs on.

The engine is invisible by design — a good core is one you never think about. But it is the most consequential piece of technology in the entire institution, because everything else is just a window onto it. The next time a bank makes headlines for an outage, look past the app and the website to the ledger of record underneath. That is where the bank actually lives, and where it can actually die.

## Further reading & cross-links

- [The payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — the rails and settlement step the core hands off to once a transaction is posted.
- [Reading a bank balance sheet: assets, liabilities, and equity](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) — the bank-wide view the core's millions of postings roll up into.
- [Digital banking and the neobank business model](/blog/trading/banking/digital-banking-and-the-neobank-business-model) — the banks built real-time on cloud-native cores from day one, and their unit economics.
- [Bank treasury and asset-liability management](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit) — how the balance sheet the core records is steered for liquidity and rate risk.
- [Write-ahead logging: how databases guarantee durability](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability) — the software principle behind storing postings, not just balances, in the ledger of record.
- [Zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — the engineering pattern behind moving a live ledger without a hard stop.
- [The strangler-fig pattern: migrating a monolith](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) — the phased-cutover idea banks borrow to retire a legacy core safely.
