---
title: "The Payments Business: How Money Actually Moves Between Banks"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A from-scratch guide to the invisible plumbing of payments — clearing, settlement, correspondent banking, nostro and vostro accounts, central-bank RTGS, and netting — and why this low-risk fee business is one banks fight to own."
tags: ["banking", "payments", "clearing", "settlement", "correspondent-banking", "nostro-vostro", "rtgs", "netting", "transaction-banking", "fee-income"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When you "send money" to another bank, no money actually travels; instead, two banks agree who owes whom (clearing), the central bank moves reserves between their accounts (settlement), and at some precise moment the transfer becomes irreversible (finality). That plumbing is invisible, enormous, and — because it earns fees and float without taking much credit risk — one of the most prized businesses in banking.
>
> - A "payment" is really three separate jobs: **clearing** agrees the debt, **settlement** moves the real money across the central bank, and **finality** is the legal moment it can't be undone.
> - Banks that can't reach a foreign currency directly **rent access** through a **correspondent bank**, holding money in **nostro** ("our account at them") and **vostro** ("their account at us") accounts.
> - **Netting** collapses a huge tangle of gross obligations into one small number per bank — in our worked example, \$410 million of gross flows settles as just \$20 million of net reserves, a 95% cut in the liquidity a bank must find.
> - The one number to remember: payments is a **fee-and-float** business with low credit risk, which is exactly why every bank, card network, and fintech is fighting to own the rails.

On the morning of March 10, 2023, the people running the U.S. payment system watched something that almost never happens at that speed. Silicon Valley Bank's customers had tried to pull \$42 billion out the day before, and another roughly \$100 billion was queued to leave. Every one of those withdrawals was an instruction to move money *out of SVB and into some other bank*. But here is the thing most people never think about: when \$42 billion "leaves" a bank, it does not get loaded onto a truck. Nothing physical moves at all. What moves is a number — the balance in SVB's account at the Federal Reserve — and a matching number rises in the accounts of all the banks receiving the money. The run was a run on that ledger.

That ledger, and the machinery that updates it, is the subject of this post. Most people picture a payment as money flying from your account to someone else's, like an email with cash inside. The reality is stranger and far more interesting: your money mostly stays put, the banks keep a running tally of who owes whom, and a few times a day (or, for big payments, instantly) the central bank trues up those tallies by moving reserves. Understanding that one mechanism explains everything from why an international wire takes three days and loses you \$40, to why instant payments are such a big deal, to why a payment failure can take down a bank faster than a bad loan ever could.

![How one payment crosses the banking system from payer bank through clearing and central-bank settlement to payee bank](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-1.png)

The diagram above is the mental model for this entire article. Follow a single \$1,000 payment left to right: it leaves the payer, the payer's bank (Bank A) debits the payer and now *owes* the money, a clearing system matches and nets the instruction, the central bank moves \$1,000 of reserves from Bank A to Bank B, and only then does Bank B credit the payee. The money the payee receives is not the payer's original dollars — it is a fresh \$1,000 that Bank B now holds, backed by \$1,000 more of reserves at the central bank. We will unpack every box. But keep this shape in your head: **clearing carries the message, settlement moves the real money.** They are not the same step, and confusing them is the single most common mistake people make about payments.

This connects to the spine that runs through this whole banking series: a bank is a leveraged, confidence-funded machine that borrows short and lends long. Payments sit at the very heart of that machine, because the deposits a bank holds *are* its short-term funding, and the payment system is how those deposits move, stay, or flee. A bank that owns the payment relationship owns cheap, sticky deposits — the franchise. A bank that loses control of its payment flows can be drained in hours. Payments look like boring plumbing. They are actually where a bank's funding lives or dies.

## Foundations: the words people use as if they were the same thing

Before we go anywhere, we need to define a small vocabulary precisely, because in everyday speech these words are used interchangeably and in the payment world they mean very different things. Read this section slowly; the rest of the post depends on it.

**A deposit is a debt the bank owes you.** This is the foundation under the foundation. When you have \$1,000 "in the bank," you do not have \$1,000 sitting in a drawer with your name on it. You have an IOU: the bank owes you \$1,000 and will pay it on demand. Your money *is* the bank's promise. So when you "send" \$1,000 to a friend at another bank, what you are really doing is asking your bank to stop owing *you* \$1,000 and arrange for the friend's bank to start owing *your friend* \$1,000. The total amount of money in the world does not change. What changes is *who owes whom*.

**Reserves are the banks' own money at the central bank.** Just as you keep your money at a commercial bank, commercial banks keep *their* money at the central bank — the Federal Reserve in the U.S., the European Central Bank in the euro area, the State Bank of Vietnam in Vietnam. These balances are called **reserves** (or "central-bank money"). Reserves are the only kind of money banks use to settle with each other, because they are the one form of money that is final and risk-free: a balance at the central bank cannot bounce, and the central bank cannot go bust in its own currency. When we say "the money actually moved," we almost always mean *reserves changed hands at the central bank.*

It helps to keep two layers of money straight from the start, because the whole payment system lives in the relationship between them. **Commercial-bank money** is the deposit in your account — an IOU from a private bank, and only as good as that bank. **Central-bank money** is reserves — an IOU from the central bank, the safest money there is. You and I transact in commercial-bank money; banks settle with each other in central-bank money. A payment between two banks is essentially the moment commercial-bank money (a deposit) gets *backed* by a movement of central-bank money (reserves). That two-layer structure is why a bank can fail and your deposit can be at risk, while the reserves that settle the system never bounce — and it's the reason proposals like central-bank digital currency, which would give the public direct access to central-bank money, are such a big deal for the deposit-funded banking model.

**Clearing is agreeing who owes whom.** *Clearing* is the process of exchanging payment instructions between banks, matching them, and calculating the resulting obligations. After clearing, everyone knows the score — "Bank A owes Bank B \$2.4 million net today" — but no actual money has moved yet. Clearing is the bookkeeping. Take two friends who go out all week and keep a tab of who paid for what; clearing is the moment they sit down and add it all up.

**Settlement is actually paying the debt.** *Settlement* is the moment the obligation computed during clearing is discharged with real money — reserves move from the debtor bank to the creditor bank at the central bank. Settlement is the cash changing hands. In the two-friends example, settlement is one friend finally handing the other a \$20 note to square the tab.

**Finality is the point of no return.** *Settlement finality* is the precise legal moment after which a payment cannot be reversed, even if a bank in the chain fails one second later. Before finality, a payment is a promise that could still unwind; after finality, it is done, full stop. Finality is not a thing that "moves" — it is a status, defined by law and by the rules of the settlement system. It matters enormously, because the gap between "instruction sent" and "payment final" is exactly the window in which things go catastrophically wrong.

**A correspondent bank is a bank you rent access through.** No bank has an account in every country or every currency. When your bank needs to make a payment in a currency or country where it has no presence, it uses a *correspondent bank* — a larger bank that *does* have an account at that country's central bank and agrees to make and receive payments on your bank's behalf. Your bank holds money at the correspondent, and the correspondent does the local plumbing. It is, almost literally, renting a desk in a country where you have no office.

**Nostro and vostro are two views of the same account.** These two intimidating Latin words name nothing more than perspective. A *nostro* account ("ours" in Latin) is the account *your* bank holds *at* a correspondent bank, in the correspondent's currency — "our money, over at their place." A *vostro* account ("yours") is the exact same account seen from the correspondent's side — "their money, sitting here with us." One account, two names, depending on whose books you're reading. If your Vietnamese bank holds \$5 million at a U.S. correspondent, that balance is a *nostro* account on your bank's books and a *vostro* account on the U.S. bank's books. Same \$5 million.

**Netting is settling only the difference.** *Netting* is the practice of offsetting mutual obligations so that, instead of every claim being paid in full, each party pays or receives only its net balance. If A owes B \$100 and B owes A \$90, netting means A simply pays B \$10. Netting is the single most important efficiency trick in the entire payment system, and we will spend a full section and a worked example on it.

**RTGS is the central bank settling each big payment instantly.** *Real-Time Gross Settlement* is a system run by a central bank in which large payments are settled one at a time, in full ("gross," meaning no netting), the instant they are submitted, in final central-bank money. Fedwire in the U.S., TARGET2 in the euro area, and CHAPS in the U.K. are RTGS systems. RTGS is how the big, urgent, must-be-final-now payments move.

That is the whole dictionary. Everything below is just these nine ideas interacting. Now let's watch them work.

## The trick: your money doesn't move, the ledger does

Let's nail down the central illusion with the simplest possible case. You bank at Bank A. Your friend banks at Bank B. You send your friend \$1,000.

Here is what actually happens, step by step:

1. You instruct Bank A to pay \$1,000 to your friend at Bank B.
2. Bank A reduces your deposit balance by \$1,000. You now have \$1,000 less. But that \$1,000 is *not* sent anywhere yet — Bank A is simply now holding \$1,000 it no longer owes you, and it has incurred an obligation to get \$1,000 to Bank B.
3. The instruction goes through a clearing system, which records that **Bank A owes Bank B \$1,000.**
4. At settlement, the central bank moves \$1,000 of reserves from Bank A's reserve account to Bank B's reserve account. *This* is the only "money" that physically (well, electronically) moved, and it moved between two accounts at the central bank — not between you and your friend.
5. Bank B, now \$1,000 richer in reserves, increases your friend's deposit balance by \$1,000.

Notice what just happened to the bank balance sheets. Bank A's assets (reserves) fell \$1,000 and its liabilities (your deposit) fell \$1,000 — its balance sheet shrank. Bank B's assets (reserves) rose \$1,000 and its liabilities (your friend's deposit) rose \$1,000 — its balance sheet grew. The deposit didn't *travel*; it was *destroyed* at Bank A and *created* at Bank B, with a matching reserve transfer to back it. (If you want the deeper story of how deposits get created and destroyed, see [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).)

This is why a "payment" is really three jobs, not one — and the diagram below pulls them apart so you never confuse them again.

![Matrix comparing clearing settlement and finality across what they do what moves who runs them and when risk ends](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-4.png)

Read the columns left to right. **Clearing** matches and totals the claims; nothing but messages move; a clearing house or scheme runs it; and the risk is *not* over — you only have a promise to pay. **Settlement** moves reserves across the central bank; the central bank runs it; and the risk is *almost* over. **Finality** is purely a legal status — nothing moves, the law and the system's rules confer it, and *now* the payee is paid for good. The whole drama of payments lives in the journey across those three columns, and especially in how long it takes to reach the green box on the right.

#### Worked example: a single domestic payment, traced through reserves

Suppose Bank A starts the day with \$50 million of reserves at the central bank and \$50 million of customer deposits (we'll ignore everything else for clarity). You, a customer, hold \$1,000 of that \$50 million in deposits. You send \$1,000 to your friend at Bank B.

- Before: Bank A has reserves \$50,000,000 and deposits \$50,000,000.
- Step 2: Bank A debits your deposit. Deposits drop to \$49,999,000. Reserves are still \$50,000,000 — Bank A is momentarily "over-funded" by \$1,000, which is exactly the \$1,000 it now owes Bank B.
- Step 4 (settlement): the central bank moves \$1,000 of reserves from A to B. Bank A's reserves fall to \$49,999,000.
- After: Bank A has reserves \$49,999,000 and deposits \$49,999,000 — balanced again, just \$1,000 smaller on both sides. Bank B has \$1,000 more reserves and \$1,000 more deposits.

The lesson in one sentence: **a payment is a swap of a deposit liability for a reserve asset at the sending bank, mirrored at the receiving bank — your money was never an object, only an entry.**

## Correspondent banking: how a bank reaches a country it isn't in

The single-payment story above quietly assumed both banks have accounts at the *same* central bank. That's true for two American banks — they both settle at the Fed. But what if you bank in Vietnam and want to pay a supplier in Germany? Your Vietnamese bank has no account at the Federal Reserve, no account at the ECB, no way to put reserves into a German bank directly. It is not a member of those systems. So how does the money get there?

The answer is **correspondent banking**: your bank holds money at a bigger bank that *does* have the access, and that bigger bank does the plumbing on your behalf. This is the layer that makes cross-border payments possible at all, and it's where the nostro/vostro vocabulary earns its keep.

![Graph of correspondent banking showing a small bank reaching a foreign currency through a correspondent and nostro vostro accounts](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-2.png)

Trace the chain in the figure. Small Bank A in Vietnam holds Vietnamese dong but needs to pay in U.S. dollars. It holds a dollar account at a U.S. correspondent bank — that account is Bank A's *nostro* ("our dollars, over there") and the correspondent's *vostro* ("their dollars, here with us"). The U.S. correspondent has a reserve account at the Federal Reserve, so it can settle dollars. On the other end, Small Bank B in Germany also reaches dollars through *its* own U.S. correspondent. To move dollars from A to B, A instructs its correspondent to pay B's correspondent; the two correspondents settle across the Fed; and B's correspondent credits B. Four banks, two of them just renting access, one central bank doing the actual settlement. The deeper mechanics of how the messaging works across borders — and how SWIFT fits in — are the subject of a dedicated sibling post; here we just need the shape.

The beauty and the curse of this arrangement are the same thing: it's a *chain*. Each link is a bank trusting the bank before it, holding a balance, taking a fee, and adding a day. The chain is why an international payment is slow, opaque, and expensive in a way a domestic payment isn't — and why a single bank deciding to "de-risk" and drop a correspondent relationship can cut an entire country off from the dollar.

One subtlety that trips people up: in correspondent banking, the **message** and the **money** travel on completely different tracks. The *instruction* to pay — "credit \$200,000 to this beneficiary" — flies between banks as a standardized message (over the SWIFT network, in practice). But the *money* doesn't ride along with that message. The message is just a request; the money moves only when the holding bank actually debits one account and credits another, and ultimately when reserves settle at the central bank. This is why a cross-border payment can show as "sent" in your app while the funds sit untouched for two days: the message arrived instantly, but the settlement and the chain of intermediary bookkeeping had not yet caught up. Separating "the instruction" from "the value" is the single most clarifying idea in cross-border payments.

There's also a question of *which currency you can settle in directly*, and it explains the dollar's outsized role. A bank can only put final money into another bank's hands in a currency it has a settlement account for. Almost every bank in the world can settle dollars (through a U.S. correspondent and ultimately the Fed) and euros (through TARGET2), so cross-border trade overwhelmingly routes through those two currencies even when neither party is American or European — a Brazilian importer paying an Indonesian exporter will often settle in dollars, because that's the currency both sides' banks can reliably reach. The correspondent chain, in other words, isn't just plumbing; it's *why* a handful of currencies dominate global commerce.

#### Worked example: a nostro account and a correspondent fee

Your Vietnamese bank keeps a \$5,000,000 nostro balance at its U.S. correspondent so it can fund dollar payments. A corporate client asks your bank to send \$200,000 to a U.S. supplier.

- Your bank instructs the correspondent to pay \$200,000 out of the nostro. The nostro balance falls from \$5,000,000 to \$4,800,000.
- The correspondent charges a **lifting fee** — a flat charge for handling the payment — of, say, \$25, plus your own bank adds a transfer fee of \$15 and an FX margin when it converts the client's dong into the dollars that funded the nostro.
- If the FX margin is 0.5% on the \$200,000, that's another \$1,000 of revenue captured on the conversion (the client gets a slightly worse exchange rate than the mid-market price).
- The client thinks they paid "a \$15 fee." They actually paid \$15 + \$25 + \$1,000 = **\$1,040** all-in, most of it invisible.

The lesson in one sentence: **in correspondent banking the visible fee is the small part — the lifting fees along the chain and the FX margin on the currency leg are where the real money is, which is why the business is so profitable and so hard for a customer to price.** (Multiply this opacity by geopolitics and you get [the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments), where access to the dollar chain becomes a tool of statecraft.)

## Netting: the trick that shrinks an ocean of payments to a puddle

Now we get to the most beautiful idea in the whole system. On any given day, thousands of banks send each other millions of payments. If every single payment had to settle in full, one at a time, banks would need to hold staggering amounts of reserves just to fund the gross flow — money that does nothing but slosh back and forth. **Netting** is how the system avoids that.

The intuition is the two-friends tab again, scaled up. Take three friends — Ann, Bob, and Carl — who pay for each other's lunches all week. By Friday, Ann owes Bob \$100 and Bob owes Ann \$90; Bob owes Carl \$70 and Carl owes Bob \$60; Ann owes Carl \$50 and Carl owes Ann \$40. If they paid every IOU in full, \$410 of cash would change hands. That's absurd — most of it just cancels out. Instead they net: each person's gains and losses are summed, and only the net difference is settled. Banks do exactly this, every day, in clearing systems built for the purpose.

![Before and after diagram showing gross settlement paying every claim in full versus net settlement of one balance per bank](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-3.png)

The before/after figure makes the savings concrete. On the left (gross), every obligation is paid in full and \$410 of reserves must move. On the right (net), the same trades collapse to a single balance per bank — and as we'll compute in a moment, only \$20 of reserves actually needs to move. The red-to-green flip is the whole point: netting takes a frightening gross number and turns it into a tiny net one.

#### Worked example: a three-bank netting cycle, gross to net

Let's do the arithmetic the diagram is built on, treating the friends as banks A, B, and C with amounts in millions.

The six gross obligations for the day:

- A owes B \$100m, B owes A \$90m
- B owes C \$70m, C owes B \$60m
- A owes C \$50m, C owes A \$40m

**Gross total that would move if settled one-by-one:** \$100 + \$90 + \$70 + \$60 + \$50 + \$40 = **\$410 million.**

Now compute each bank's net position (what it receives minus what it pays):

- **Bank A:** pays 100 + 50 = 150; receives 90 + 40 = 130. Net = 130 − 150 = **−\$20m (A pays \$20m).**
- **Bank B:** pays 90 + 70 = 160; receives 100 + 60 = 160. Net = 160 − 160 = **\$0 (B is flat).**
- **Bank C:** pays 60 + 40 = 100; receives 70 + 50 = 120. Net = 120 − 100 = **+\$20m (C receives \$20m).**

The net positions must always sum to zero — A's −\$20m plus C's +\$20m plus B's \$0 = \$0. The check confirms the arithmetic.

**Reserves that actually move under netting:** A sends \$20m, C receives \$20m, B does nothing. **\$20 million** changes hands instead of \$410 million.

The reduction: 1 − (20 / 410) = **95.1%.** Netting cut the reserves the system needed to find by more than nineteen-twentieths.

The lesson in one sentence: **netting doesn't change who ends up with what — it just deletes the offsetting flows, slashing the reserves banks must hold to keep the system running, which is why almost all retail payments settle net rather than gross.** This is the same chart, drawn as bars:

![Grouped bar chart of three banks comparing gross amount paid out versus net amount to fund after netting](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-5.png)

The bars show each bank's gross outflow in red and its net funding need in green. Bank A's gross \$150m collapses to a \$20m net; Bank B's gross \$160m collapses to *zero*; Bank C pays \$100m gross but on net *receives*, so its funding need is zero too. Across the system, \$410m of gross becomes \$20m of net. That green-versus-red gap is, quite literally, the liquidity that netting saves banks from having to hold.

There is a catch, and it's the reason netting isn't used for everything. Between the moment of netting and the moment of net settlement, the obligations are *unsettled* — they're just promises. If a bank fails in that window, the survivors are exposed to the failed bank's net position, and unwinding the netting can ripple through everyone else. That risk — **settlement risk**, sometimes called **Herstatt risk** after the 1974 collapse of a German bank that took payments and failed before settling its side — is exactly why the biggest, most urgent payments are *not* netted at all. They go through RTGS.

### Two kinds of settlement risk, and the two ways the system kills them

It's worth being precise about *what* can go wrong in that unsettled window, because the entire design of modern payment systems is a response to two specific dangers, and the fixes are different.

The first danger is **principal risk** — the risk of losing the whole amount, not just a price movement. In a foreign-exchange trade you pay one currency and receive another, and the two legs settle in different countries and time zones. If you've paid your dollars but the counterparty fails before paying your euros, you've lost the *entire* dollar amount, not a fraction. That is exactly what happened in 1974: banks paid Herstatt their Deutsche marks in Frankfurt in the morning and were due to receive dollars in New York that afternoon, but Herstatt was shut down in between. The dollars never came. The fix the industry eventually built is **payment-versus-payment (PvP)**: a mechanism (CLS Bank, launched in 2002) that releases *neither* leg of an FX trade until *both* are funded, so neither side can be left holding only the leg it already paid. PvP doesn't reduce the amount that settles; it removes the *timing gap* that let one side pay and the other vanish.

The second danger is **replacement-cost risk** — the milder cost of having to redo a failed trade at a worse price. If a counterparty fails before settling, you haven't lost the principal (you never paid it), but you now have to replace the deal in the market, possibly at a loss if prices moved. Netting systems handle this by requiring members to post **collateral** and pre-fund a **default fund**, so a failed member's losses are covered by its own and the group's posted resources rather than spreading to survivors. The same logic governs securities settlement, where the analogous fix is **delivery-versus-payment (DvP)**: the security and the cash change hands simultaneously, so you never deliver the bond without receiving the cash or vice versa.

Both fixes share one idea: **make the two halves of a transaction conditional on each other, so neither completes alone.** That is the entire art of settlement design — shrinking, and ideally eliminating, the window in which one party has paid and the other hasn't. The closer a system gets to "both legs or neither," the safer it is, and the more liquidity it tends to demand. Once again, safety and liquidity trade off against each other.

## RTGS versus deferred net settlement: the core trade-off

Every payment system in the world is built on one fundamental choice: settle each payment instantly and in full (gross), or batch them up and settle the net later (deferred). The first is **Real-Time Gross Settlement (RTGS)**; the second is **Deferred Net Settlement (DNS)**. They are two answers to the same question — how do we balance speed and finality against liquidity and cost? — and a modern banking system runs both, side by side, for different jobs.

![Matrix comparing RTGS real time gross settlement and deferred net settlement across timing liquidity risk and use](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-6.png)

The matrix lays out the trade-off cleanly. **RTGS** settles each payment now, one at a time, in final central-bank money — so settlement risk is near zero, but the bank needs the *full* amount of reserves on hand for every payment, which is expensive. It's used for large, urgent, must-be-final-now payments: Fedwire, TARGET2, CHAPS. **Deferred net settlement** batches payments, nets them, and settles the small net balance later (often once or twice a day) — so the liquidity need is tiny, but a bank can fail between batching and settlement, creating real settlement risk. It's used for the high-volume, low-value retail stream: ACH transfers, card batches, paper checks.

The reason both exist is that they fail in opposite ways. RTGS never leaves a bank exposed to a counterparty overnight, but it can seize up if banks hoard reserves and refuse to send the first payment, waiting for incoming funds (a "gridlock" — modern RTGS systems include clever liquidity-saving mechanisms that net queued payments to break it). DNS sips liquidity but stacks up settlement risk all day, which is tolerable for a \$50 grocery payment and unthinkable for a \$500 million bond purchase. So the system sorts payments by size and urgency: the few enormous ones go gross and final immediately; the billions of tiny ones go net and final later.

#### Worked example: the liquidity cost of gross versus net

Take a mid-sized bank that, over one day, sends \$8 billion of payments and receives \$7.5 billion of payments across the system.

- **Under pure RTGS (gross):** to send every payment the instant it's due, the bank might need to fund up to \$8 billion of outgoing reserves before its \$7.5 billion of incoming arrives — in the worst case it must hold or borrow the full gross outflow intraday. Even with smart queuing, its peak intraday reserve need runs into the billions. Reserves are not free: holding \$1 billion of idle reserves instead of lending it at, say, a 2% net spread costs the bank \$20 million a year in foregone income.
- **Under DNS (net):** the bank's net position for the day is \$8.0bn out − \$7.5bn in = **\$0.5 billion to fund.** It needs to find half a billion, not eight billion — a sixteenth as much.

The lesson in one sentence: **netting buys cheap liquidity at the price of intraday risk, and gross settlement buys safety at the price of expensive idle reserves — every payment rail is just a different point on that one trade-off.** (The companion post on [domestic payment rails](/blog/trading/banking/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments) walks rail-by-rail through where each lands on this spectrum.)

## Float: the money that earns interest while it's in flight

There's a quiet source of profit hiding in every payment, and once you see it you can't unsee it: **float.** Float is money that has left one party but not yet arrived at the other — money "in transit" that, for a moment, belongs to no one in particular and sits in a bank's hands earning interest.

The intuition is simple. Suppose a payment takes two days to clear. On day one, your bank has debited you \$1,000 — that money is gone from your account. But Bank B hasn't credited your friend yet. For those two days, \$1,000 is sitting somewhere in the chain, and whoever holds it can earn interest on it. Multiply by billions of payments a day and float becomes a serious line of revenue, especially when interest rates are high. Float is one big reason banks have historically been in no hurry to make payments faster: every day a payment is "in transit" is a day of free interest for someone in the chain.

![Graph showing where fees and float accrue along the payment chain at the sending bank correspondent and network](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-8.png)

The figure follows \$1,000 down the chain and marks every place value is skimmed off. The sending bank takes a transfer fee plus an FX margin and earns float while it holds the funds; the correspondent takes a lifting fee and earns its own float in transit; the scheme or network takes a per-message clearing fee. By the time the payment "lands," the payee receives less than \$1,000 — the difference is the chain's revenue. Notice that float (the amber boxes) is earned *in parallel* with the fees: the same money that's being charged a fee is also sitting around earning interest. That double-dip — a fee on the transaction *and* interest on the balance — is what makes payments such an attractive business.

#### Worked example: float income on payment volume

Suppose a transaction bank processes \$50 billion of payments a year, and on average each dollar spends 1.5 days in transit before it settles to the destination. The bank earns the overnight risk-free rate, say 4.5% annually, on whatever balance is "in flight" at any moment.

- Average float balance = (annual volume × average days in transit) ÷ 365 = (\$50,000,000,000 × 1.5) ÷ 365 = **\$205.5 million** sitting in transit at any time.
- Annual float income = \$205.5 million × 4.5% = **\$9.25 million per year**, earned purely on money passing through.

Now watch what happens when payments speed up. If an instant-payment rail cuts the average transit time from 1.5 days to near zero, that \$9.25 million of float income largely *disappears.*

The lesson in one sentence: **float is interest earned on money that's merely passing through, so a bank's payment profit is partly a tax on slowness — which is exactly why faster rails are simultaneously great for customers and a threat to the banks that built the slow ones.**

Float has a long and slightly notorious history. In the U.S. check-clearing system before electronic processing, the gap between when a check was deposited and when it cleared could be days, and banks earned billions of dollars a year on the resulting float. Some institutions even engineered delays — routing checks through distant clearing points to stretch the transit time — until the Check 21 Act of 2004 let banks exchange digital images of checks and collapsed clearing from days to roughly one day. Float income shrank accordingly. The same dynamic is replaying now at the level of interbank and cross-border payments: every regulatory push toward faster settlement is, viewed from the bank's income statement, a push to shrink float. This is the quiet tension under the whole "instant payments" debate — customers want their money now, and "now" is precisely when float stops earning.

### Intraday liquidity: the cost of being ready to pay

There's a mirror image of float that sits on the *cost* side of the payment business: **intraday liquidity.** To settle payments throughout the day, a bank must hold enough reserves (or arrange enough intraday credit from the central bank) to fund its outgoing payments *before* its incoming payments arrive. Payments don't arrive evenly — a bank might owe \$3 billion at 10am but not receive its big inflows until 2pm — so it has to be funded for the peak gap, not the daily average. Holding that buffer of idle reserves is a real cost: every dollar parked to cover an intraday timing mismatch is a dollar not earning a loan spread. This is why a bank's treasury manages its payment timing as carefully as its balance sheet — releasing payments in a sequence that uses incoming funds to cover outgoing ones, queuing non-urgent payments, and tapping central-bank intraday credit (usually collateralized and free or cheap within the day) to bridge the gaps. The float earned on slow payments and the cost of intraday liquidity are two sides of the same coin: both are about *when* money is where, not how much of it there is.

## Where the fees come from: the low-risk business banks fight to own

Step back and ask the question that explains the whole industry's behavior: *why do banks and fintechs and card networks fight so hard over payments?* The answer is that payments are an unusually attractive business — and to see why, you have to compare them to the bank's *other* business, lending.

Lending makes money on the **spread** between what a bank pays for deposits and what it earns on loans, but it carries **credit risk**: borrowers default, and in a bad year provisions for loan losses can wipe out a year's profit. Payments, by contrast, make money on **fees and float** and carry almost no credit risk — the bank is moving the customer's own money, not lending out its own. A payment that clears is just revenue; there's no five-year tail of default risk hanging over it. The income is fee-based, recurring, and capital-light: regulators require far less capital to back a payments business than a loan book, because there's far less that can go wrong. (This is the mirror image of the [investment bank's](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) risk-heavy trading income — payments are the *boring, durable* counterpart.)

And there's a second prize, even bigger than the fees: payments generate **cheap, sticky deposits.** A company that runs its payroll, collections, and supplier payments through your bank leaves large operating balances sitting in its accounts — balances the bank funds itself with, at near-zero interest. Owning a customer's payment flows means owning their deposit balances, and (as the post on [retail deposits](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) argues) cheap deposits are the entire franchise. This is why banks will run a payments business at a thin direct margin: the deposits it drags in are worth more than the payment fees themselves.

To see the fee mechanics concretely, look at how a card payment — one of the most lucrative corners of payments — splits its take. The figure shows where the merchant's fee on a credit-card sale actually goes.

![Stacked bar showing how a card merchant fee splits between interchange network fee and acquirer markup](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-9.png)

On a representative U.S. credit-card sale, the merchant pays a *merchant discount rate* of about 2.3% of the transaction. The biggest slice — about 1.75% — is **interchange**, which flows to the bank that *issued* the customer's card. A thin sliver (about 0.13%) is the **network fee** to Visa or Mastercard for running the rails. The rest (about 0.42%) is the **acquirer markup**, the margin taken by the bank that serves the merchant. Three different players, three slices, all on a payment that carries essentially no credit risk to the acquiring side. The deep economics of this four-party model are a post of their own; the point here is simply that *every* hop in a payment chain is a place to charge a fee.

#### Worked example: payment-fee revenue on volume

Suppose a bank's transaction-banking arm processes \$120 billion of corporate payments a year and earns an average all-in fee of 8 basis points (0.08%) across wires, ACH, FX, and card flows.

- A *basis point* is one hundredth of a percent — 0.01% — so 8 basis points is 0.08%.
- Fee revenue = \$120,000,000,000 × 0.08% = \$120,000,000,000 × 0.0008 = **\$96 million per year.**
- Now add float: if average balances of \$3 billion sit in operating accounts at a 4% net funding benefit, that's another \$3,000,000,000 × 4% = **\$120 million per year** of value from the cheap deposits the payment relationship drags in.

So the payment relationship is worth roughly \$96 million + \$120 million = **\$216 million** — and barely half of it is the fees customers see. The deposit franchise is the larger half.

The lesson in one sentence: **payments earn a small, low-risk fee on enormous volume *and* generate the cheap deposits a bank funds itself with, which is why the payments business is worth far more than its visible fee line and why everyone wants to own the rails.**

## The scale of the machine: who actually runs the rails

It's worth pausing to feel the size of the entities at the center of this system. The banks that operate as the world's correspondents — the ones with reserve accounts at every major central bank, the ones every smaller bank routes its dollars and euros through — are giants. Their balance sheets are larger than most countries' economies.

![Horizontal bar chart of the largest banks by total assets in trillions of dollars](/imgs/blogs/the-payments-business-how-money-actually-moves-between-banks-7.png)

The chart ranks the world's largest banks by total assets. The Chinese state banks — ICBC at about \$6.3 trillion, Agricultural Bank of China, China Construction Bank, Bank of China — top the list, followed by the U.S. and European money-center banks: JPMorgan Chase around \$4.0 trillion, Bank of America, HSBC, BNP Paribas, Citigroup, Wells Fargo. These are the banks that *can* settle directly at the major central banks, hold the vostro accounts of thousands of smaller banks, and clear the bulk of the world's cross-currency payments. When a Vietnamese or Kenyan or Argentine bank needs dollars, it is almost certainly reaching them through one of these names.

This concentration is a feature and a danger at once. It's a feature because settlement *needs* a small number of trusted, well-capitalized hubs — a payment system where everyone settles with everyone is unmanageable. It's a danger because it means a handful of banks sit at choke points: if one of them de-risks an entire region (drops correspondent relationships to avoid compliance costs), or is cut off from a currency for geopolitical reasons, whole countries can lose access to global payments. The plumbing is centralized, and centralized plumbing has chokepoints.

The concentration also has an economic logic that makes it self-reinforcing: payments are a **network-effect business.** A correspondent bank that already clears for a thousand smaller banks is more valuable to the next small bank than one that clears for ten, because it can reach more counterparties in one hop. A card network that already connects millions of merchants and cardholders is nearly impossible for a new entrant to dislodge, because the value of joining is the number of people already there. The same logic favors the incumbents in clearing systems, where each new member makes membership more useful for everyone else. This is why payments tend toward a few dominant rails rather than a fragmented market — and why fintechs that want into payments usually find it easier to *ride* an incumbent's rails (and split the fees) than to build their own. The economics push relentlessly toward scale, which is exactly why the same names appear at the center of payments decade after decade.

#### Worked example: why a payment failure scares a bank more than a bad loan

Tie this back to the spine of the series. A bank borrows short (deposits) and lends long (loans), surviving only as long as depositors trust it. Compare two bad days at a bank.

- **A bad loan:** a \$100 million corporate loan defaults. With a 60% recovery, the loss is \$40 million. Painful, but the bank books a provision, absorbs it against the year's profit, and lives. The damage is *bounded and slow.*
- **A payment failure:** the bank misses an intraday settlement obligation in RTGS — it owes \$2 billion at 2pm and can't fund it. Other banks instantly see it can't pay. They stop sending it payments. Its incoming flows dry up, which makes its next obligation harder to fund, and within hours counterparties are pulling lines. The damage is *unbounded and fast* — a liquidity spiral, not a loss.

This is precisely the shape of the SVB story we opened with: not a slow credit problem but a fast *outflow* problem, where the payment system was the channel through which \$42 billion tried to leave in a day.

The lesson in one sentence: **credit losses are bounded and slow, but payment and settlement failures are fast and self-reinforcing — which is why a bank's treasury watches its intraday reserve position far more nervously than its loan book, and why control of payment flows is, ultimately, control of survival.**

## Common misconceptions

**"When I send money, my money travels to the other person."** No money travels. Your bank destroys a deposit it owed you and the recipient's bank creates a deposit it owes the recipient, with a transfer of central-bank reserves to back the change. The total money supply is unchanged; only *who owes whom* changes. In our worked example, the payee's \$1,000 is brand-new money created by Bank B, not your original dollars relocated.

**"Clearing and settlement are the same thing."** They are two distinct steps separated in time, and the gap between them is where settlement risk lives. Clearing computes "Bank A owes Bank B \$2.4 million net today"; settlement is the later moment reserves actually move to discharge it. A payment can be cleared but not yet settled — agreed but not yet paid — and if a bank fails in that window, the survivors are exposed. The 1974 Herstatt collapse, where a bank took payments and failed before settling its side, is the textbook case.

**"Faster payments are obviously better, so banks should have built them years ago."** Faster payments are better for customers, but they destroy two things banks like: **float** (interest on money in transit, which near-instant settlement eliminates) and **the deposit-stickiness of slowness** (when payments are instant, money is easier to move out — including out of *your* bank in a run). In our float example, cutting transit time from 1.5 days to zero erased \$9.25 million of annual income. Banks' historical foot-dragging on instant payments was not incompetence; it was self-interest, and regulators often had to *push* the rails into existence.

**"An international wire is expensive because moving money across borders is technically hard."** The technical part is cheap. The cost is the *chain*: each correspondent bank in the path takes a lifting fee, holds a balance, adds a day, and the currency conversion carries an FX margin. In our worked example a "\$15" transfer actually cost \$1,040 once the correspondent fee and the 0.5% FX margin were counted. The expense is the number of hands the money passes through, each taking a slice — not the physics.

**"Payments are low-margin plumbing, not a real business."** The direct fee margin is thin, but payments throw off *cheap, sticky deposits* the bank funds itself with — and that deposit value usually dwarfs the fee line. In our transaction-banking example, the \$96 million of fees was the *smaller* half of a \$216 million relationship; the \$120 million of deposit-funding value was the bigger prize. Payments aren't low-value; their value is just mostly hidden in the funding base.

## How it shows up in real banks

**The 1974 Herstatt failure and the birth of settlement-risk discipline.** Bankhaus Herstatt, a German bank, was taking in Deutsche marks against U.S. dollars it was due to pay later the same day. German regulators shut it down in the afternoon — after it had *received* the marks but *before* it sent the dollars. Counterparties on the other side of the ocean were left with a cleared-but-unsettled claim and no money. The episode named "Herstatt risk" — the cross-currency settlement risk created by time-zone gaps between the two legs of an FX trade — and eventually produced CLS Bank in 2002, which settles both legs of an FX trade simultaneously ("payment versus payment") so neither side can be left exposed. It is the clearest real-world proof that *clearing is not settlement*, and that the gap between them can be fatal.

**SVB, March 2023: a run is a payment event.** When Silicon Valley Bank failed, the proximate cause people remember is the duration losses on its bond portfolio. But the *mechanism* of failure was payments: on March 9, customers instructed \$42 billion to leave, each instruction a payment out of SVB's account at the Fed and into other banks. Another roughly \$100 billion was queued for March 10. No bank has \$140 billion of reserves on hand; SVB couldn't fund the outflow, and the FDIC stepped in before the second day's payments could settle. The lesson treasuries took: in a digital, instant-payment world, a deposit run is a *settlement* event that can drain a bank in hours, not the days-long queue outside a branch from older bank runs. (The full ALM story is in [SVB and Credit Suisse 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**De-risking and the dropping of correspondent relationships.** Since roughly 2012, large correspondent banks have been *cutting* relationships with smaller banks in higher-risk regions — the Caribbean, parts of Africa, the Pacific — because the compliance cost of monitoring those flows for money-laundering and sanctions exceeds the fee income. The Financial Stability Board has tracked a steady decline in the number of active correspondent relationships worldwide. The effect is that entire countries can find themselves struggling to send or receive dollars, not because they're sanctioned, but because no big bank finds it worth the regulatory risk to bank them. It's a vivid demonstration that the payment system runs on *voluntary* correspondent links, and those links can be withdrawn.

**The 2019 repo spike: when settlement liquidity ran short.** In September 2019, the overnight repo rate — the rate banks pay to borrow cash against safe collateral, the grease of the settlement system — spiked from about 2% to nearly 10% in a day. One trigger was a collision of corporate tax payments and Treasury settlements that drained reserves out of the banking system faster than expected, leaving banks short of the reserves they needed to settle. The Fed had to inject tens of billions to calm it. The episode showed that even the U.S. settlement system can run short of the one thing it depends on — reserves — and that the plumbing's smooth functioning is not guaranteed. (The repo mechanics are covered in [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).)

**CHIPS and the netting efficiency of the dollar's plumbing.** Most large-value dollar payments between banks don't actually settle one-by-one across Fedwire; they run through CHIPS (the Clearing House Interbank Payments System), a netting system that settles roughly \$1.8 trillion of payments a day on a tiny fraction of that in actual reserve movement, by netting throughout the day and settling the residual. CHIPS is the netting worked example from this post, operating at national scale: it lets the dollar system move trillions while requiring banks to fund only a sliver of net positions. It's the single best illustration of why netting isn't a nice-to-have — it's what makes a payment system of that size physically possible.

**The instant-payments shift: FedNow, UPI, and the squeeze on float.** India's UPI rail moves billions of small payments a month, instantly and at near-zero cost; the U.S. launched FedNow in 2023; the euro area, the U.K., Brazil (Pix), and others have their own instant rails. As these spread, the slow-payment float that banks quietly earned for decades shrinks, and the deposit-stickiness that came from money being hard to move erodes. Banks are responding by competing on the *services* wrapped around payments — fraud protection, data, integration — because the float and the friction that used to be free money are disappearing. It is the clearest sign that the economics laid out in this post are actively in flux.

## The takeaway: read a bank through its payment flows

If you remember one thing from this post, make it the separation of the three jobs: **clearing agrees the debt, settlement moves the money, finality locks it.** Almost every confusion about payments — why wires are slow, why netting matters, why a run is so fast, why Herstatt happened — dissolves once you stop treating "a payment" as a single instantaneous act and start seeing it as a journey across those three stages, with risk living in the gaps between them.

For reading a bank as a business, the payments lens gives you three durable instincts. First, **follow the deposits, not just the fees.** A bank's payment franchise is worth most for the cheap, sticky operating balances it drags in, not the visible fee line — so when you see a bank with a deep transaction-banking business, you're looking at a cheap funding base, which (per the series spine) is the heart of a bank's profitability and resilience. Second, **watch the intraday liquidity, not just the capital.** A bank can be perfectly solvent and still die in an afternoon if it can't fund its settlement obligations as deposits flee through the payment system — solvency is about losses, but payments are about *speed*, and speed is what kills. Third, **respect the chokepoints.** The world's payments run through a handful of giant correspondents and a few central-bank settlement systems; that concentration is what makes the system efficient and what makes it fragile, and it's why control of the rails is a strategic prize fought over by banks, networks, fintechs, and governments alike.

The deepest point is this: payments look like the most boring part of banking — plumbing, back-office, plumbing again. But the plumbing is where a bank's funding lives, where its survival is tested by the hour, and where one of its most profitable, lowest-risk businesses sits. A bank is a confidence-funded machine; the payment system is the artery that confidence flows through. Watch the artery, and you'll see the health of the machine before it shows up anywhere else.

## Further reading & cross-links

- [Domestic payment rails: RTGS, ACH, card networks, and instant payments](/blog/trading/banking/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments) — rail by rail through the speed-cost-finality trade-off introduced here.
- [Cross-border payments: correspondent banking and how SWIFT really works](/blog/trading/banking/cross-border-payments-correspondent-banking-and-how-swift-really-works) — the messaging layer and the FX leg of the chain we sketched.
- [Cash management and transaction banking for corporates](/blog/trading/banking/cash-management-and-transaction-banking-for-corporates) — how the sticky, cheap deposits payments generate become the franchise.
- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — why owning the payment relationship is really about owning the deposits.
- [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments) — what happens when access to the correspondent chain becomes a tool of statecraft.
- [How money is created: banks, central banks, and the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) — the system view of the deposit-creation step we traced here.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the overnight funding market that keeps the settlement system supplied with reserves.

*This is educational, not financial advice.*
