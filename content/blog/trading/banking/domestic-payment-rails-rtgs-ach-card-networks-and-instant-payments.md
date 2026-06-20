---
title: "Domestic Payment Rails: RTGS, ACH, Card Networks, and Instant Payments"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A from-zero tour of the four ways money moves inside a country — real-time gross settlement, ACH batch netting, card networks, and instant rails — and why each one trades speed, cost, value-size, and finality differently."
tags: ["banking", "payments", "rtgs", "ach", "card-networks", "instant-payments", "fedwire", "fednow", "settlement-finality", "payment-rails"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — There is no single "payment system" inside a country; there are several rails, and each one is built for a different job because no rail can be fast, cheap, large, and final all at once.
>
> - **RTGS** (real-time gross settlement, e.g. Fedwire) moves each payment one at a time, in central-bank money, with instant finality — the choice for big, urgent, can't-be-reversed transfers. It is fast and final but costs a few dollars an item and runs only in banking hours.
> - **ACH** (automated clearing house) is the opposite design: it queues thousands of small payments, nets them down, and settles only the *difference* between banks once or twice a day. That makes it astonishingly cheap (cents per item) but slow (often one to two days) — the rail for payroll, bills, and direct debits.
> - **Card networks** sit on a third axis: the *authorization* is instant at the till, but the *money* settles days later through the issuing and acquiring banks, the merchant pays a percentage-of-sale fee, and the payment is **reversible** (chargebacks). Convenience and consumer protection, paid for in basis points.
> - **Instant rails** (FedNow, the UK's Faster Payments, India's UPI) are the new design that tries to break the trade-off: seconds, 24/7, irrevocable, and cheap — but value-capped, which is why they complement rather than replace RTGS and ACH.
> - The one number to remember: on the same set of payments, **gross settlement (RTGS) can move ten times more central-bank reserves than net settlement (ACH)** — netting is how the system moves trillions on a thin sliver of actual reserves.

In the early hours of a normal weekday, almost nothing visible happens in finance — and yet, inside the central bank's computers, several trillion dollars change hands before most people have had coffee. A pension fund settles a bond purchase. A bank repays an overnight loan to another bank. A company wires the down-payment on a factory. None of these moves through the same plumbing as the \$4.30 you tap for a coffee, or the rent that leaves your account on the first of the month, or the paycheck that lands on Friday. They travel on entirely different *rails*.

That word — **rail** — is the one to hold onto. A payment rail is a shared track that banks agree to use to move money between themselves on a customer's behalf. Just as a country has highways for trucks, rail lines for freight, and footpaths for pedestrians — each tuned to a different load, speed, and cost — a country has several payment rails, each tuned to a different *kind* of payment. The mistake almost everyone makes is to imagine "the payment system" as one thing. It is not. It is a small handful of systems with sharply different personalities, and a bank's treasury, an app developer, and a corporate treasurer all spend real effort choosing which rail to put a given payment on.

The figure below is the mental model for this whole post: four rails, scored on the four dimensions that actually matter — speed, cost per item, the typical value they carry, and finality (whether the payment can be undone). Read it as a menu of trade-offs, not a ranking. There is no "best" rail, only a best rail *for a given payment*.

![Matrix comparing RTGS, ACH, card networks, and instant rails on speed, cost, value, and finality](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-1.png)

This is the operations-level companion to the system view of how money moves between banks. If [the payments business as a whole](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) is the story of clearing, settlement, and the correspondent accounts that connect banks, this post zooms into the specific tracks money runs on *within one country* — and why the design of each track is a deliberate answer to one unavoidable tension. That tension traces straight back to this series' spine. A bank is a leveraged, confidence-funded machine; payments are where that confidence is tested thousands of times a second, because every payment is a promise that the money is real and will not be clawed back. How a rail handles that promise — instantly and irrevocably, or slowly and reversibly — is the whole story.

## Foundations: the words you need before we touch a rail

Before we compare the rails, we have to define the handful of ideas every one of them is built from. A practitioner can skim this section; a newcomer cannot proceed without it, because the entire post is built on these six terms.

**The corner-shop analogy.** Start with two corner shops on the same street, Shop A and Shop B, that constantly run errands for each other's customers. A customer of Shop A buys something and asks Shop A to "pay Shop B \$10 for me." There are two completely different ways the shops can square up. They could walk \$10 in cash next door *the moment* each errand happens — that is **gross settlement**: every transaction paid in full, individually, immediately. Or they could keep a tally on a chalkboard all day — "A owes B \$10 here, B owes A \$7 there" — and at closing time settle only the *net difference*, say \$3, with a single walk next door. That is **net settlement**. Hold onto this example. RTGS is "walk the cash next door every time"; ACH is "keep a chalkboard tally and settle the difference at closing." Everything below is this shop, with precise vocabulary.

### Clearing vs settlement — two different verbs

People use "clearing" and "settlement" loosely, but they are distinct steps and the difference is the heart of payments.

- **Clearing** is the *information* step: the banks exchange the instructions — who is paying whom, how much, into which account — and agree on what is owed. Clearing is messages. No money has actually moved yet.
- **Settlement** is the *money* step: the actual transfer of funds that discharges the obligation, so that the paying bank is genuinely poorer and the receiving bank is genuinely richer. Settlement is final.

A useful way to keep them straight: clearing is the order, settlement is the payment. When you order a coffee, the order (clearing) and the moment your card is charged (settlement) are different events — and on some rails they are separated by *days*. The defining difference between the rails is largely how far apart they push clearing and settlement, and what they let happen in the gap between.

### Central-bank money vs commercial-bank money

This distinction is subtle but it explains why RTGS exists at all. The money in your checking account is **commercial-bank money** — a promise from your bank to pay you. It is only as good as the bank. The money banks hold in their accounts *at the central bank* — their **reserves** — is **central-bank money**, the safest money there is, because the central bank cannot run out of its own currency.

When two banks settle a payment, the cleanest, most final way is to move central-bank money between their reserve accounts: one bank's reserves go down, the other's go up, on the central bank's own books. There is no residual credit risk, because central-bank money carries none. RTGS settles in central-bank money, which is exactly why it is so final and so trusted for large amounts. Many other rails settle in central-bank money too, but only at the *net*, and only at certain times — which is where credit risk sneaks in.

### Finality — the property everything hinges on

**Finality** means a payment cannot be reversed or undone. Once it is final, the money is the recipient's, full stop — even if the sender goes bankrupt the next minute, even if it was a mistake, even if it was fraud. Legal finality and practical finality matter enormously:

- An RTGS payment is final *the instant it settles*. The receiver can spend it immediately with zero risk that it vanishes.
- An ACH payment becomes final only after its settlement window completes, and some ACH items can be *returned* for days afterward (insufficient funds, unauthorized debit).
- A card payment is, by design, *not* final for a long time — it can be **charged back** weeks or months later if the cardholder disputes it. That reversibility is a feature for consumers and a cost for merchants.
- An instant-rail payment is final in *seconds and irrevocable* — closer to handing over cash than any other electronic rail.

Finality is the property that determines what a payment is *for*. You would never settle a \$50 million property purchase on a rail where the money can be clawed back; you would never demand bullet-proof instant finality for a \$30 utility bill where a one-day delay is harmless and the saving is real.

### Push vs pull

One more axis. A **push** payment is initiated by the payer: "I send money to you." The payer pulls the trigger, so the payer is in control and the receiver can't move money out of the payer's account without permission. RTGS and instant rails are push: the money leaves *because the sender said so*.

A **pull** payment is initiated by the payee: "I take money from you, with your prior authorization." A direct debit (your gym charging your account monthly) and a card payment (the merchant's bank reaching into your account) are pull. Pull is convenient — you set it up once and forget it — but it is structurally riskier, because someone other than the account holder is reaching in. That risk is exactly why pull rails (ACH debits, cards) come with **return** and **chargeback** rights: the safety valve that lets you undo an unauthorized pull. Push rails generally don't need that valve, which is one reason they can offer cleaner finality.

With those six ideas — clearing vs settlement, central-bank money, gross vs net, finality, push vs pull — we can now take the four rails one at a time.

## RTGS: pay every transaction in full, one by one, right now

Real-time gross settlement is the heavyweight rail. The name says exactly what it does. **Real-time**: the payment settles within seconds of being submitted, not at the end of the day. **Gross**: each payment is settled individually and in full — no batching, no netting, no waiting for offsetting payments. **Settlement**: the money moves on the central bank's books, in central-bank money, with immediate finality.

In the United States this rail is **Fedwire**, run by the Federal Reserve. The euro area has **TARGET2** (now consolidated as T2), the UK has **CHAPS**, and almost every country has an equivalent. The mechanism is simple and brutal in its cleanliness: when Bank A wants to pay Bank B \$10 million, Bank A submits the instruction, the central bank checks that Bank A has \$10 million of reserves (or available intraday credit), then debits \$10 million from Bank A's reserve account and credits \$10 million to Bank B's reserve account. Done. Final. The figure below traces that flow.

![Pipeline showing an RTGS payment debiting the sender reserve account and crediting the receiver with immediate finality](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-2.png)

Because each payment is settled in full and immediately, **there is no settlement risk** — the moment the credit posts, Bank B has central-bank money it can use right away, and nothing about Bank A's later fortunes can take it back. This is the rail's superpower and the reason it carries the highest-value, most time-critical payments: the cash legs of securities trades, interbank loans, large corporate payments, and the settlement of *other* payment systems (the net positions of ACH and card networks ultimately settle across RTGS).

But that superpower has a price. Gross settlement is **liquidity-hungry**. If you settle every payment in full, you need enough reserves on hand at every moment to cover the gross flow — even if money is flowing back to you a minute later. Take a bank that sends \$100 million at 9:00 and is due to receive \$100 million at 9:05. Under gross settlement it must have \$100 million of reserves ready at 9:00, because it cannot net the incoming against the outgoing. That is real money tied up, and across the system it is enormous. Central banks address this by extending **intraday credit** (free or cheap overdrafts on reserve accounts during the day, repaid by close) and by clever queuing algorithms that try to offset payments where they can. Still, the liquidity demand of gross settlement is the central design cost of RTGS, and it is precisely why the system did not put *every* payment on this rail.

RTGS is also constrained by **operating hours**. Fedwire runs on business days, roughly 21 hours a day in its current extended schedule, but it closes — there is a cutoff, and it does not run on weekends or holidays. So "real-time" means "real-time, while the rail is open." A wire submitted after the cutoff waits for the next business day. And it costs real money: a Fedwire transfer runs on the order of a few dollars in network fees, and banks typically charge customers \$15 to \$35 for an outbound wire. That is trivial on a \$10 million payment and absurd on a \$30 one — which is the whole point of having other rails.

#### Worked example: the cost and speed of an RTGS payment

Suppose a law firm must send \$2,000,000 to complete a real-estate closing this afternoon, and the money absolutely cannot be reversed once the seller's lawyer confirms receipt. The firm instructs its bank to send a Fedwire.

- **Speed.** The bank submits the wire at 2:00 p.m. The Fed debits the firm's bank's reserves and credits the seller's bank's reserves within seconds. By 2:00 p.m. and change, the seller's bank has \$2,000,000 of central-bank money. It posts to the seller's account immediately. The closing proceeds.
- **Finality.** The instant the Fed posts the credit, the payment is final. If the firm's bank failed at 2:01 p.m., it would not matter to the seller — the money is already central-bank money in the seller's bank.
- **Cost.** The bank charges the firm a \$30 outbound-wire fee. On a \$2,000,000 payment that is 0.0015% — a rounding error.

Now flip the numbers. If that same \$30 fee were charged on a \$30 utility bill, it would be a **100%** surcharge — you would pay twice. **The intuition:** RTGS is priced and engineered for payments where the value is so large, and the need for instant finality so absolute, that a few dollars and a few seconds of liquidity are nothing. Put a small everyday bill on it and the economics are insane. That is the gap the next rail fills.

### The hidden machinery: intraday liquidity and queuing

It is worth pausing on the liquidity problem because it is the least visible and most consequential part of how RTGS actually runs. Because every payment settles in full and immediately, a bank's reserve account behaves like a checking account that must never go negative for a payment it wants to make — the central bank will simply reject (or queue) a payment the bank can't fund at that instant. So a bank's payments operations team spends the whole day watching a single number: how much central-bank money is available right now, against the queue of payments it needs to send.

Two mechanisms keep this manageable. First, **intraday credit**: the central bank lets banks run an overdraft on their reserve account *during* the day, as long as it is repaid before close. The Fed extends this credit largely free up to collateralized limits, which lets a bank send a \$100 million payment at 9:00 even though it won't receive its own \$100 million until 9:05 — the intraday overdraft bridges the five-minute gap. Without intraday credit, gross settlement would demand banks pre-stockpile reserves to cover their entire gross outflow, which would be ruinously expensive.

Second, **queuing and offsetting algorithms**: modern RTGS systems don't always settle a payment the literal instant it arrives. If Bank A wants to send \$50 million but is short on reserves, and Bank B has a \$48 million payment queued *to* Bank A, the system can hold both and settle them together, so each bank only needs to fund the \$2 million net at that moment. This "liquidity-saving mechanism" quietly borrows a little of ACH's netting logic to ease gross settlement's liquidity hunger — without giving up the per-payment finality. It is a reminder that the four rails are not as cleanly separated in their internals as the menu suggests; each borrows tricks from the others where it can.

The reason any of this matters to a reader of bank balance sheets: a bank can be perfectly *solvent* — its assets exceed its liabilities — and still fail to make a payment because it is short of *reserves at that minute*. Payments are a liquidity problem, not a solvency one, and the RTGS rail is where that distinction becomes concrete and daily. A bank that mismanages its intraday liquidity can find itself unable to honor a wire it has already promised, which is a reputational catastrophe even if the bank is fundamentally healthy.

## ACH: queue everything, net it down, settle the difference

The automated clearing house is RTGS's mirror image. Where RTGS optimizes for speed and finality at the cost of liquidity and fees, ACH optimizes for *cost and volume* at the expense of speed. It is the rail behind your paycheck, your mortgage payment, your utility direct debits, your tax refund, and the vast majority of recurring, low-value, non-urgent payments in the economy.

Here is the design. Instead of settling each payment as it arrives, ACH **batches**. Banks collect payment instructions all day and hand them to an ACH operator (in the US, the Federal Reserve's FedACH and a private operator, The Clearing House's EPN) as files. The operator sorts the instructions, computes how much each bank owes every other bank across *all* the payments in the batch, and then settles only the **net** position of each bank — usually across Fedwire, in central-bank money — at fixed times during the day. The figure below traces that batch-and-net flow.

![Pipeline showing ACH collecting payments into a batch, netting each bank, and settling the net at a window](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-3.png)

The genius — and it really is a piece of genius that predates computers — is **multilateral netting**. If Bank A is sending \$10 million to Bank B's customers across thousands of payments, and Bank B is sending \$9.6 million to Bank A's customers, the two banks don't move \$19.6 million between them. They move \$0.4 million — the net. Across the whole system, billions of dollars of customer payments collapse into a much smaller pile of actual interbank reserve movements. That is why ACH can move staggering volumes on a thin base of reserves, and why it is so cheap.

ACH comes in two flavors that map exactly onto our push/pull distinction. An **ACH credit** is a push: the payer's bank sends money to the payee (your employer pushing your salary into your account). An **ACH debit** is a pull: the payee's bank reaches into the payer's account, with prior authorization (your utility company pulling your bill). Both ride the same batch-and-net plumbing, but the direction of initiation matters for risk — which is why ACH debits can be **returned** (for insufficient funds, for "unauthorized," for "account closed") for a period after settlement. That return right is the safety valve on a pull rail.

The cost of all this is **speed**. Traditional ACH settles on a one-to-two business-day timeline: a payment submitted Monday might post Wednesday. The US has added **Same Day ACH** (multiple settlement windows in a single day, with a per-item value cap), which compresses the timeline but is still batch-based and still bound to business hours and windows. ACH does not run on weekends. And finality is softer than RTGS: a payment is settled when its window completes, but the return window means an ACH debit isn't *truly* unclawbackable for days. This is fine for the payments ACH carries — nobody needs their salary to be irrevocable to the second — but it is exactly why you would never close a property sale on ACH.

There is a useful piece of vocabulary that demystifies how a bank actually handles an ACH item. The bank that *starts* a payment on behalf of its customer is the **originating depository financial institution (ODFI)**; the bank that *receives* it for its customer is the **receiving depository financial institution (RDFI)**. When your employer's bank pushes your salary, it is the ODFI and your bank is the RDFI; when your gym pulls your monthly fee, your gym's bank is the ODFI (it originates the debit) and your bank is the RDFI. The same two roles, ODFI and RDFI, describe every ACH payment — and the legal rules about who bears the risk of an unauthorized or returned item are written in terms of them. The reason an ACH debit can be returned for "unauthorized" up to 60 days later for consumers is precisely that the RDFI's customer must be protected from a pull they never agreed to. That long return window is the safety valve that makes pull-on-ACH tolerable, and it is also why ACH cannot offer the bullet-proof same-second finality of a push wire.

It also pays to see *where the credit risk lives* in ACH. When an employer originates payroll on Monday for Wednesday settlement, the employee's bank (the RDFI) often makes the money available to the employee on the morning the credit posts, *before* the interbank settlement is fully irrevocable. The RDFI is extending a small, well-understood credit, betting the originating side will be good for it. Multiply that across millions of payments and the system is shot through with tiny, brief credit exposures — which is exactly the netting-gap risk we will name again below, just at the level of customer accounts rather than interbank balances. ACH's cheapness is purchased partly with this web of small, trusting credits, and it works because the participants are regulated banks that almost always settle.

#### Worked example: batch netting, and why the chalkboard beats the cash-walk

Three banks — A, B, C — process a day of customer payments. The gross flows are:

- A's customers pay B's customers \$8 million; B's customers pay A's \$7.5 million.
- B's customers pay C's customers \$5 million; C's customers pay B's \$4 million.
- A's customers pay C's customers \$3 million; C's customers pay A's \$2.8 million.

Under **gross settlement (the RTGS way)**, every one of these moves in full. Total reserves shuffled across the day: \$8 + \$7.5 + \$5 + \$4 + \$3 + \$2.8 = **\$30.3 million** of central-bank money in motion, and each bank must hold enough reserves to cover its outgoing flows in real time.

Under **net settlement (the ACH way)**, we tally each bank's net:

- Bank A: receives 7.5 + 2.8 = 10.3, pays 8 + 3 = 11.0 → net **pays \$0.7 million**.
- Bank B: receives 8 + 4 = 12.0, pays 7.5 + 5 = 12.5 → net **pays \$0.5 million**.
- Bank C: receives 5 + 3 = 8.0, pays 4 + 2.8 = 6.8 → net **receives \$1.2 million**.

The whole day collapses to A paying \$0.7m and B paying \$0.5m into C — **\$1.2 million** of reserves actually moved, versus \$30.3 million gross. That is a **25-to-1 reduction** in the central-bank money the system has to touch. **The intuition:** netting is how the payment system moves the volume of an entire economy on a sliver of real reserves — but it works only because banks are willing to wait until the window to settle, and to trust each other during the gap. That waiting and that trust are the price of cheapness.

The figure below shows the same idea on a smaller, cleaner set of three payments — gross moves \$30 million, netting moves \$4 million — so you can see at a glance how much less money net settlement actually shifts.

![Before and after comparison showing RTGS moving 30 million of reserves gross versus ACH moving 4 million net](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-4.png)

There is a sting in the tail of netting, and it is worth naming because it caused real crises. During the gap between clearing and settlement, the banks are extending each other credit. If Bank B is due to pay its net \$0.5 million at the window and *fails before the window*, the other banks were counting on funds that won't arrive. This is **settlement risk** (sometimes called **Herstatt risk**, after a German bank whose 1974 failure mid-settlement froze the system). Modern net-settlement systems blunt this with collateral, loss-sharing agreements, and caps, but the risk is intrinsic to netting and is the deep reason high-value payments use gross settlement instead. Speed and finality, or cost and liquidity-efficiency — you genuinely cannot have all four, and ACH made the cost-and-efficiency choice on purpose.

## Card networks: instant authorization, delayed settlement, reversible money

Card networks are a different animal again, and the easiest to misunderstand because the experience is so seamless that the plumbing is invisible. When you tap a card, it *feels* like the money moves instantly. It does not. What happens instantly is **authorization** — a yes/no answer to "does this account have the funds and is the card valid?" The actual movement of money settles days later, through a chain of banks and a network in the middle.

To see it, you need the **four-party model**. There are four parties to a card transaction: the **cardholder** (you), the **issuer** (your bank, which issued the card and holds your account), the **merchant** (the shop), and the **acquirer** (the merchant's bank, which "acquires" the transaction on the merchant's behalf). The card network — Visa, Mastercard — sits in the middle as the switch that routes messages and sets the rules. The full economics of this model — issuing, acquiring, interchange, and how the merchant fee is split — are a whole subject in their own right, covered in [the cards business](/blog/trading/banking/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split); here we care about it strictly as a *rail*: how fast, how cheap, how final.

The flow has two clocks. The **authorization clock** runs in seconds: tap → acquirer → network → issuer checks funds → "approved" comes back → the merchant's terminal beeps. The **settlement clock** runs in days: at end of day the merchant batches its approved transactions, the acquirer submits them to the network, the network nets the positions among issuers and acquirers, and the actual funds settle (typically across an RTGS or net-settlement rail) over the following one to three business days. The merchant gets paid days after the customer walked out with the goods. So a card payment is *clearing-first, settlement-much-later* — the widest gap between the two verbs of any rail.

Two consequences flow from that gap, and they define the card rail's personality.

First, **cost**. The merchant does not get the full sale price. A slice is taken out as the **merchant discount rate (MDR)** — the all-in fee the merchant pays to accept the card. The biggest piece of the MDR is **interchange**, paid by the acquirer to the issuer (it is what funds your rewards points and the issuer's fraud losses); the rest is the network's fee and the acquirer's markup. Because this is a *percentage of the sale*, not a flat fee, cards are expensive for large payments and tolerable for small ones — the exact inverse of RTGS's economics. The figure below shows where roughly \$2.30 of fees on a \$100 credit-card sale goes, next to the few cents an ACH debit would cost.

![Stacked bar showing a card sale costs about 2.3 percent in fees while an ACH debit costs about 0.03 percent](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-9.png)

Second, **reversibility**. Because a card payment is a *pull* (the merchant's side reaches into your account) and because settlement lags authorization, the system bakes in a powerful consumer-protection mechanism: the **chargeback**. If you dispute a charge — fraud, goods never arrived, duplicate billing — you can ask your issuer to reverse it, and the money is pulled back from the merchant weeks or even months later. This is the opposite of finality, and it is *deliberate*. The whole value proposition of a card to a consumer is that the money is not final: if something goes wrong, you have recourse. That same property is a cost and a risk to the merchant, who can lose both the goods and the money.

#### Worked example: the cost of a card payment vs an ACH payment

A merchant makes two \$100 sales. One customer pays by credit card; the other pays by ACH (say, a bank-transfer checkout option).

- **Card.** The merchant discount rate is about 2.3% all-in, split roughly as interchange 1.75%, network fee 0.13%, acquirer markup 0.42%. On \$100 that is about \$2.30 in fees. The merchant nets **\$97.70**, and receives it two business days later. And the sale is reversible: a chargeback could pull the \$100 back, leaving the merchant out the goods and a chargeback fee.
- **ACH.** The fee is a flat few cents — call it \$0.03. The merchant nets **\$99.97**, but waits one to two days and accepts that the payment could be *returned* if the customer's account lacks funds.

The card costs the merchant about **77 times more** in fees than the ACH (\$2.30 vs \$0.03) on the same \$100. Scale it up: a business doing \$10 million a year on cards pays roughly \$230,000 in fees; on ACH it would pay roughly \$3,000. **The intuition:** the card fee is not a tax — it buys the merchant guaranteed customer convenience, instant authorization, fraud-loss transfer to the issuer, and access to customers who *only* carry cards. Whether that is worth 2.3% depends entirely on the margin of the business, which is why some merchants surcharge cards and others bury the cost in their prices.

Notice what the card rail is *not* good at: large payments (the percentage fee becomes brutal) and payments that must be final (the chargeback right makes them inherently reversible). You would not buy a house with a credit card even if a seller accepted one — the 2.3% fee on a \$500,000 home is \$11,500, and the chargeback risk to the seller is unacceptable. Cards are the rail of the till and the checkout page: small, frequent, consumer-initiated, convenience-first.

## Instant rails: the attempt to break the trade-off

For decades the menu was just those three: RTGS (fast, final, expensive, big), ACH (cheap, slow, small), cards (convenient, reversible, percentage-priced). Each made one hard choice. The newest rail tries to refuse the choice entirely. **Instant payment rails** aim to deliver settlement in *seconds*, *24 hours a day, 7 days a week*, with *immediate finality* and *low, flat cost*. In the US this is **FedNow** (launched by the Federal Reserve in July 2023) and the private-sector **RTP** network from The Clearing House. The UK pioneered the model with **Faster Payments** in 2008; the most dramatic example anywhere is India's **UPI** (Unified Payments Interface), which moves billions of transactions a month.

How do they pull off "fast and final and cheap" when we just spent three sections arguing you can't have all of them? The trick is a hybrid design. Most instant rails clear each payment *individually and in real time* (like RTGS — so the receiver sees final, irrevocable funds in seconds) but settle the interbank money on a *deferred net or prefunded* basis (borrowing from ACH's efficiency). To make the credit to the receiver safe *before* the interbank money settles, banks **prefund** a balance or post collateral that backs the payments. So the customer experience is instant and final; the interbank settlement is managed in the background against prefunded positions. That hybrid is how a small, low-value payment can be both irrevocable in seconds and cheap to run.

The figure below scatters the four rails on the two dimensions that drive rail choice — speed and cost per item — on a log cost axis so you can see the spread. The prized corner is bottom-right: fast *and* cheap, where instant rails sit.

![Scatter plot positioning RTGS, ACH, card, and instant rails by settlement speed and cost per item](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-5.png)

If instant rails are fast, cheap, and final, why haven't they swallowed everything? Three reasons, and each one explains why the older rails persist.

1. **Value caps.** Instant rails carry a per-transaction limit (FedNow launched with a \$500,000 cap, later raised; many systems cap far lower). Because settlement is backed by prefunded balances and irrevocable finality means no chargeback safety net, the systems deliberately limit single-payment size. A \$50 million interbank loan still belongs on RTGS.
2. **Irrevocability cuts both ways.** Instant finality is wonderful for legitimate payments and terrible for fraud victims. If you are tricked into sending an instant payment to a scammer, the money is *gone* in seconds with no chargeback. This "authorized push payment" fraud is the dark side of instant rails and a live regulatory battle, especially in the UK. Cards' reversibility, the thing that makes them "slow to finality," is exactly the consumer protection instant rails lack.
3. **Adoption and reach.** A rail is only useful if both banks are on it. RTGS and ACH reach essentially every bank in the country; instant rails are still building coverage. UPI succeeded so spectacularly partly because India mandated and standardized it; FedNow is opt-in and still ramping.

#### Worked example: instant-payment settlement, seconds and final

It's Saturday night. You owe a friend \$80 for concert tickets and want to settle it now, not Monday.

- On **ACH**, your payment can't even start until Monday (no weekend processing) and would post Tuesday or Wednesday. Cost to you: usually free, but useless tonight.
- On a **wire**, you can't — RTGS is closed on weekends, and a \$30 fee on \$80 is absurd anyway.
- On a **card**, you can't pay a person directly; cards are a merchant rail.
- On an **instant rail** (via a banking app or a service built on FedNow/RTP), you initiate the \$80 at 9:47 p.m. Saturday. Within seconds your friend's bank shows \$80 of *final, spendable* money. The interbank settlement happens in the background against your bank's prefunded position. Cost: a few cents to your bank, often free to you.

The instant rail did in seconds, at near-zero cost, on a Saturday night, what no other rail could do at all. **The intuition:** instant rails are not "RTGS for small amounts" or "fast ACH" — they are a genuinely new design that breaks the speed/cost/finality trade-off *within a value cap*, by separating the instant, final customer experience from the deferred, prefunded interbank settlement. That is why they complement the old rails for everyday small payments rather than replacing the rails built for very large or very protected ones.

It is worth being precise about how the prefunding makes instant finality *safe*. On FedNow, a participating bank holds a balance in a dedicated settlement account at the Fed (or shares one through a correspondent). When the bank sends an instant payment, the Fed checks that the sending bank's settlement balance covers it and moves the funds between the two banks' settlement balances *as part of the same transaction* that credits the receiver — so the interbank settlement actually is immediate and in central-bank money, not deferred. That makes FedNow more like a 24/7 RTGS for small payments than like ACH. RTP, the private network, uses a shared prefunded pool at the Fed and settles positions against it. Either way, the bank must keep its settlement balance topped up around the clock — including weekends and the middle of the night — which is the new liquidity-management burden that 24/7 rails impose. A bank can no longer assume its reserves can "rest" overnight; an instant rail means a customer can drain a settlement balance at 3 a.m. on a Sunday, and the bank's treasury has to have planned for it.

This is the quiet revolution instant rails represent for bank operations. For a century, the payment system slept — it ran in business hours, paused on weekends, and let banks square up their books overnight. Instant rails end that. Money now moves continuously, which means a bank's intraday liquidity management becomes an *all-day, every-day* discipline rather than a business-hours one. The technology to send a payment in two seconds turned out to be the easy part; rebuilding a bank's liquidity, fraud-monitoring, and operations functions to run continuously is the hard part, and it is why even technically launched instant rails take years to feel ubiquitous.

## Value vs volume: why the rails split the work the way they do

Step back and look at the whole system through one lens: **value vs volume**. The single most illuminating fact about domestic payment rails is that the rail carrying the most *money* is almost never the rail carrying the most *payments* — and vice versa. The figure below makes the inverse relationship visible.

![Grouped bar chart showing RTGS wires carry most value but few payments while cards carry most payments but little value](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-6.png)

RTGS-type wires are a tiny fraction of the *number* of payments in an economy — a fraction of a percent — yet they carry the overwhelming majority of the *value*, because each one is huge. Cards are the opposite: an enormous share of the *count* (every coffee, every tap) but a tiny share of the *value*, because each one is small. ACH sits in the middle on both axes — lots of payments (payroll, bills) of medium size. This is not an accident of history; it is the system organizing itself by the economics we just walked through. Expensive, liquidity-hungry, perfectly final RTGS naturally attracts the few payments where those properties are worth paying for. Cheap, reversible, percentage-priced cards naturally attract the billions of tiny payments where convenience is everything and the value is small. The rails *specialize*, and the specialization is exactly what their design trade-offs predict.

This value/volume split has a practical consequence that matters to anyone running a business or a bank: **a rail that is cheap per item can still be your biggest cost line, and a rail that is expensive per item can be trivial.** A retailer doing millions of small card sales pays a fortune in interchange even though each fee is small, because the volume is colossal. A corporate treasury making a handful of huge supplier wires pays almost nothing in fees relative to the sums moved, even though each wire fee looks "expensive." Cost-per-item is the wrong lens; *cost as a fraction of value moved* is the right one — and on that lens, cards are pricey, RTGS is cheap, and ACH is nearly free.

## Timing: when, during a day, each rail actually settles

Speed is not just "how long does one payment take" — it is also *when in the day and week the rail is even open*. This matters more than newcomers expect, because a payment that is "instant on the rail" can still be slow to the recipient if the rail is closed. The figure below sketches when, across a single day, each rail settles.

![Timeline showing instant rails settle around the clock, RTGS within operating hours, and ACH on delayed batches](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-7.png)

The pattern: instant rails settle at any time, including the middle of the night and weekends, because they were built 24/7 from the start. RTGS settles continuously *but only within its operating window* on business days — a wire at 3 a.m. or on Sunday simply waits. ACH settles at a small number of fixed windows during business days, so a payment's actual arrival depends on which window it caught and whether it landed before or after a cutoff. This is why "I sent it Friday afternoon" can mean the money arrives Tuesday: it missed Friday's last ACH window, and the rail didn't run over the weekend.

For a bank's treasury, these timing rhythms are a daily operational reality. Reserves must be in the right place before each RTGS cutoff; ACH net positions must be funded before each settlement window; intraday liquidity must be managed so the bank never finds itself unable to make a payment it is committed to. The move to 24/7 instant rails is quietly forcing banks to rethink decades of "the payment system sleeps at night" assumptions — money now moves on Sunday at 2 a.m., and a bank's liquidity has to be ready for it.

## Choosing a rail: the decision in practice

So how does anyone actually pick? In practice the choice falls out of three questions: **how large is the payment, how urgent is it, and who is being paid?** The decision tree below shows the logic.

![Decision graph choosing a payment rail by value, urgency, and counterparty](/imgs/blogs/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments-8.png)

- **Very large and time-critical** (a property closing, an interbank loan, a securities settlement) → **RTGS**. The fee is irrelevant, the instant finality is essential, the liquidity cost is acceptable for the importance of the payment.
- **Small or mid-sized and needs to clear now** (a person-to-person transfer, an urgent supplier payment on a weekend) → **instant rail**, if both parties' banks are on it and the amount is under the cap.
- **Bulk and not urgent** (payroll, recurring bills, tax refunds, mass disbursements) → **ACH**. The one-to-two-day timeline is fine, and the per-item cost of pennies is decisive when you are running thousands or millions of payments.
- **Buying at a shop or online as a consumer** → **card network**, because that is the rail merchants accept, it offers instant authorization and chargeback protection, and the per-payment fee is small relative to a small purchase.

A useful comparison table pulls the whole menu together:

| Rail | Speed | Cost per item | Typical value | Final? | Push/Pull | Classic use |
|---|---|---|---|---|---|---|
| **RTGS** (Fedwire) | Seconds, banking hours | High (a few \$) | Very large | Immediate, irrevocable | Push | Property, interbank, securities |
| **ACH** (batch) | 1–2 days (or same-day window) | Very low (cents) | Small–mid | After window; returnable | Push or pull | Payroll, bills, direct debit |
| **Card network** | Auth instant; settle in days | Mid (% of sale) | Small | Reversible (chargeback) | Pull | Retail, e-commerce |
| **Instant rail** (FedNow) | Seconds, 24/7 | Low (cents–dimes) | Small–mid, capped | Immediate, irrevocable | Push | P2P, urgent small payments |

Read the table and the underlying truth jumps out: **every rail is the best at something and the worst at something else.** RTGS is the most final and the most expensive. ACH is the cheapest and the slowest. Cards are the most convenient and the only reversible one. Instant rails are the fastest-with-reach but capped and unforgiving of error. A mature economy runs all four because it has all four kinds of payment.

## Common misconceptions

**"Faster is always better, so instant rails will replace everything."** No. Instant finality is a liability for large payments (no chargeback if you're defrauded, no netting efficiency for the system) and for consumer protection (cards' slow-to-final design is what lets you dispute a charge). A \$50 million interbank settlement deliberately uses RTGS over an instant rail despite both being "instant," because RTGS settles in central-bank money at gross with no value cap, while instant rails are capped and prefunded. Speed is one dimension of four; the right rail optimizes the *combination*, and for big or protected payments that is not the fastest one.

**"When I tap my card, the money leaves my account immediately."** It doesn't. What happens instantly is *authorization* — a hold or a check that funds exist. The actual settlement runs through the four-party chain over one to three business days, which is why a pending card charge can take days to "post" and can even disappear if the merchant never captures it. The gap between authorization and settlement is days wide, and it is the room in which chargebacks live.

**"Netting is just an accounting trick — the same money moves either way."** Profoundly not. On the worked example above, gross settlement moved \$30.3 million of central-bank reserves; netting moved \$1.2 million. That is a 25-fold reduction in the actual reserves the system must touch. Netting is how an economy moves the value of millions of payments on a thin base of reserves — the trick is real, it saves enormous liquidity, and its only cost is the settlement risk in the gap before the window completes.

**"ACH and wires are the same thing — they're both bank transfers."** They are opposite designs. A wire (RTGS) is gross, real-time, final, expensive, and best for large urgent payments. ACH is batched, netted, one-to-two days, cheap, returnable, and best for bulk small payments. Sending payroll by wire would cost a fortune; closing a house by ACH would expose the seller to a returned payment. The word "transfer" hides two completely different machines.

**"A cheaper rail per payment is always cheaper overall."** Not for your actual bill. Cards charge a small fee per item but you make millions of card sales, so interchange can be a merchant's single biggest cost line. Wires charge a large fee per item but you make few of them on huge sums, so the fee is trivial as a share of value. The right metric is *cost as a fraction of value moved*, on which cards are the expensive rail and RTGS the cheap one — the exact opposite of the per-item ranking.

## How it shows up in real banks

**Fedwire and the daily liquidity dance.** Every business day, US banks move trillions of dollars across Fedwire — far more than the total reserves in the system — because reserves *recirculate*: a dollar received at 10:00 funds a payment sent at 10:05. Banks rely on the Fed's intraday credit to bridge timing mismatches, and the Fed's queuing logic tries to offset opposing payments to economize on liquidity. When liquidity gets tight (as in the September 2019 repo spike), the gross-settlement machine strains: banks become reluctant to pay early because they're hoarding reserves, payments queue up, and the system can gridlock. Gross settlement's liquidity hunger is not a textbook curiosity — it is a daily operational reality that occasionally bites.

**Same Day ACH and the squeeze on the batch.** For decades US ACH was strictly next-day or two-day. Faced with faster rails abroad and customer demand, the industry rolled out Same Day ACH (2016 onward), adding intraday settlement windows and progressively raising the per-payment cap (to \$1,000,000 by 2022). It is a revealing compromise: the system kept the cheap, netted batch architecture but compressed the timing. It shows how a legacy rail evolves — not by abandoning its design, but by adding windows within it. Even so, Same Day ACH is still batched, still business-hours, and still returnable, which is why it didn't make FedNow redundant.

**FedNow's slow build, July 2023 onward.** When the Federal Reserve launched FedNow, headlines treated it as the arrival of instant US payments. The reality on the ground was a slow ramp: a few dozen banks at launch, growing into the hundreds, because a rail is only useful when *both* banks are on it. Meanwhile The Clearing House's RTP had been live since 2017 with broad large-bank coverage. The lesson banks took: building the rail is the easy part; achieving ubiquity — the moment when you can assume the other side is reachable — is the hard, multi-year part, and until then instant payments coexist awkwardly with the old rails.

**UPI and what mandated standardization can do.** India's Unified Payments Interface is the world's most successful instant rail by volume — billions of transactions a month, a huge share of them tiny everyday payments that would elsewhere be cash. UPI worked because it was standardized and pushed hard by a central institution, made interoperable across apps, and engineered to be effectively free to users. It is the proof-of-concept that instant rails *can* dominate small-value payments when reach and cost are solved — and a reminder that the US, EU, and others are years behind on ubiquity despite having the technology.

**The card-fee tug-of-war.** Because card interchange is a percentage of every sale and the volume is colossal, it is one of the most fought-over numbers in commerce. The EU capped consumer-card interchange (0.2% on debit, 0.3% on credit) in 2015; the US Durbin Amendment capped debit interchange for large issuers in 2011; merchants, networks, and regulators have litigated the rest for years. Every basis point of interchange is billions of dollars moving between merchants and issuers. The card rail's percentage pricing — the thing that makes it the wrong rail for large payments — is precisely what makes its economics a permanent battleground.

**Settlement risk and why high value never went net.** The 1974 collapse of Bankhaus Herstatt mid-settlement — it had taken in payments but failed before paying out the other legs — gave the industry a permanent scar and a name for the risk in the netting gap. The response over decades was to push the highest-value, most systemically important payments onto gross, real-time, central-bank-money settlement (RTGS), where there *is* no gap. It is the clearest case of design following a disaster: the entire existence of RTGS as the high-value rail is, in part, an answer to what net settlement can do when a bank fails at the wrong moment.

## The takeaway: a rail is a frozen answer to a trade-off

The single most useful thing to carry away from this post is that **there is no neutral way to move money — every rail is a deliberate, frozen answer to a trade-off you cannot escape.** You can have instant finality, or you can have liquidity-efficient netting, but the same payment cannot have both. You can have a flat trivial fee, or you can have percentage-priced consumer protection, but not on the same rail. You can have 24/7 reach, or you can have an unlimited value ceiling, but the systems that offer one tend to limit the other. Every rail picked a corner of that space and built itself there: RTGS chose finality and size; ACH chose cost and efficiency; cards chose convenience and reversibility; instant rails chose speed and reach within a cap.

Once you see the rails this way, a lot of finance stops being mysterious. You understand why your salary takes a day but your tap is instant; why a house is bought by wire and not by card; why a scammer wants you on an instant rail (no chargeback) and a legitimate retailer is happy on cards (the chargeback protects the buyer); why a central bank obsesses over intraday liquidity (gross settlement is liquidity-hungry by design); why netting can move an economy's payments on a sliver of reserves (and why that sliver carries a hidden settlement risk). The rails are not interchangeable pipes. They are specialized tools, and knowing which tool fits which payment is most of payments literacy.

And it ties straight back to the spine of this whole series. A bank is a leveraged, confidence-funded machine, and a payment is a moment of pure confidence: a promise that the money is real and will arrive. Each rail is a different *engineering of that promise* — RTGS makes it ironclad and instant for the payments where breaking it would be catastrophic; ACH makes it cheap and good-enough for the payments where a day's delay is harmless; cards make it conditional and reversible so the consumer is protected; instant rails make it instant and absolute within a safe size. The choice of rail is, at bottom, a choice about *how strong the promise needs to be* — and how much you are willing to pay, in money, time, liquidity, or risk, to make it that strong.

The next time you tap, wire, get paid, or pay a bill, you are quietly choosing a rail — and now you know what you are choosing between.

## Further reading & cross-links

- [The payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — the system-level view of clearing, settlement, and correspondent banking that these rails sit inside.
- [The cards business: issuing, acquiring, interchange, and the MDR split](/blog/trading/banking/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split) — the full economics of the four-party card model and where every basis point of the merchant fee goes.
- [Cross-border payments: correspondent banking and how SWIFT really works](/blog/trading/banking/cross-border-payments-correspondent-banking-and-how-swift-really-works) — what happens when a payment has to leave the country and cross between currencies and jurisdictions.
- [Central bank digital currencies (CBDCs)](/blog/trading/finance/central-bank-digital-currencies-cbdc) — the proposed next rail: central-bank money in retail hands, and what it would do to the rails described here.
