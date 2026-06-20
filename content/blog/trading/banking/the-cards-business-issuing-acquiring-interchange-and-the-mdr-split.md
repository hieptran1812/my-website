---
title: "The Cards Business: Issuing, Acquiring, Interchange and the MDR Split"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a card payment really works — the four-party model, the authorization-clearing-settlement chain, where the merchant discount rate goes, and why interchange is the most contested fee in finance."
tags: ["banking", "payments", "credit-cards", "interchange", "merchant-discount-rate", "acquiring", "card-issuing", "visa-mastercard", "durbin-amendment", "card-economics"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A card payment is a four-party trade with the network (Visa or Mastercard) sitting in the middle as a switch: the cardholder's bank (the *issuer*) and the merchant's bank (the *acquirer*) never deal directly, they meet through the network. The merchant pays a *merchant discount rate* (MDR) of roughly 2.3% of every credit-card sale, and the biggest slice of that — the *interchange*, about 1.75% — flows from the merchant's bank to the cardholder's bank.
>
> - **The merchant pays for the whole system.** On a \$100 credit-card sale, the merchant keeps about \$97.70. Of the \$2.30 that vanishes, \$1.75 is interchange (to the issuer), \$0.13 is the network's fee, and \$0.42 is the acquirer's markup.
> - **Interchange is the engine of the card business.** It is the single largest revenue line for many card issuers and the thing that pays for your cashback and your airline points. Richer rewards cards carry higher interchange, which is why a merchant pays more when you tap a premium card.
> - **An issuer makes money three ways:** interchange (per swipe), interest (on balances you carry), and fees (late, annual, cash-advance). For a typical credit-card business, *interest* on revolving balances is the biggest line, with interchange second.
> - **The one number to remember:** about **1.75%** — the interchange a US merchant pays the cardholder's bank on a typical credit-card sale. That single percentage is why card fees are fought over in courtrooms, legislatures, and regulators' offices on three continents.

In 2024, the largest US merchants paid an estimated \$170 billion in card-acceptance fees — more than they paid for electricity, more than most of them paid in rent. A coffee shop selling a \$5 latte handed roughly 14 cents of it to a chain of banks the customer never sees and the owner can't negotiate with. A supermarket running on a 2% net margin watched a 2.3% card fee quietly turn a thin profit on a card sale into something closer to a wash. And the strangest part is that almost none of this fee goes to the company whose name is on the card. Visa and Mastercard — the brands everyone blames — keep only a sliver. The lion's share flows to a bank you've probably never thought about: the one that issued the card sitting in your wallet.

This is the cards business, and it is one of the most profitable, most misunderstood, and most litigated corners of banking. It looks, from the outside, like a simple convenience: you tap, it works, money moves. Underneath is a four-party machine with a private toll booth in the middle, a fee structure that has been challenged all the way to the Supreme Court, and an economic logic that explains why your bank is so eager to mail you a card with "2% cashback on everything" — and who is really paying for that cashback. (This is educational, not financial advice; we are explaining how the plumbing works, not telling anyone to swipe more or less.)

The diagram below is the mental model for everything that follows: four parties, with the network as the switch in the middle. The cardholder and the merchant — the two people who actually agreed on a price — never touch each other's money. Their two banks meet through the network instead. Hold that picture; every fee, every dispute, and every regulator's intervention is a fight over how value moves across the links in this one diagram.

![The four party card model with the network in the middle](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-1.png)

This post is the operations-level deep dive into how a bank runs cards. It connects to the spine of this whole series — a bank is a leveraged, confidence-funded maturity-transformation machine that borrows short and lends long — because a credit card is one of the purest expressions of that trade: the issuer lends you money the instant you tap, funds it with cheap deposits, and earns a spread plus a per-swipe toll. We'll build every term from zero, walk a single \$100 sale through the entire chain, take apart where the money goes, show how an issuer actually earns its living, and end on why interchange is the most contested number in payments.

## Foundations: the four parties, the three moves, and the fee everyone fights over

Before we can follow the money, we need a small, precise vocabulary. Each of these words is doing real work in the machine, so we'll define it from zero — a plain-English explanation first, the industry term second.

**The cardholder.** That's you. You hold a card — a plastic (or digital) token that lets you pay with someone else's money for a moment and settle up later (for a credit card) or instantly draw your own money (for a debit card). The cardholder is the *demand* side: you decide to pay by card instead of cash.

**The merchant.** The shop, restaurant, website, or vending machine that sells you something and wants to be paid. The merchant is the *supply* side. Crucially, the merchant does not get the full price you paid — it gets the price minus a fee, which is the whole subject of this post.

**The issuer.** This is the bank that gave you the card — the *issuing bank*, or *issuer* for short. Chase, Capital One, your local credit union: whoever's name and logo sit on the card. The issuer is your bank in this transaction. When you tap, the issuer is the one that either lends you the money (credit) or moves it out of your account (debit). The issuer carries the risk that you won't pay it back, and in exchange it collects most of the fee.

**The acquirer.** This is the merchant's bank — the *acquiring bank*, or *acquirer*. It's the institution that signed the merchant up to accept cards, gives it the card terminal or the online payment gateway, and *acquires* the transaction on the merchant's behalf, depositing the money into the merchant's account. (In modern practice, a *payment processor* like Stripe, Square, or Adyen often sits in front of the acquirer doing the technical work, but economically it plays the acquirer's role. We'll treat them together.)

**The network.** Visa, Mastercard, American Express, Discover. The network is the *switch* in the middle — the private infrastructure and rulebook that routes a transaction from the acquirer to the right issuer and back, and that sets the fees. Here is the single most important and most counter-intuitive fact in the whole business: **Visa and Mastercard do not issue most of the cards that carry their logos, and they do not lend you any money.** They are networks, not lenders. Thousands of banks issue Visa-branded cards; Visa just runs the rails and the rulebook and takes a small toll per transaction. (American Express and Discover are the exceptions — they are *closed-loop* networks that are usually both the network and the issuer, which is a different model we'll come back to.)

Those are the parties. Now the three moves that turn a tap into money in the merchant's account.

**Authorization.** The instant you tap, a message races from the terminal, through the acquirer, across the network, to your issuer, which checks: is this card real, not stolen, not frozen, and does it have enough credit or balance? It answers yes or no in about two seconds. *Authorization is a promise, not a payment.* No money has moved yet. The issuer has merely set aside (a *hold*) the amount and told the merchant: go ahead, you'll be paid.

**Clearing.** Later — usually that night or the next day — the merchant batches up all the day's approved transactions and sends them through the network for *clearing*: sorting and matching every transaction to the correct issuer, computing who owes whom, and applying the fees. Clearing is the accounting step that says exactly how much each issuer must pay each acquirer.

**Settlement.** Finally, the actual money moves. Each issuer pays the network's settlement system, the network pays each acquirer, and the acquirer credits the merchant's account — minus the fee. *Settlement is when value actually changes hands*, typically a day or two after you tapped. This is why a refund can take days to appear: the original money already traveled the whole chain, and reversing it has to travel back.

**Interchange.** This is the fee at the center of everything. *Interchange* is the amount the acquirer (merchant's bank) pays the issuer (your bank) on every transaction — set by the network, paid out of the merchant's pocket. On a typical US credit-card purchase it's about 1.75% of the sale. Interchange is not the network's fee and it is not the merchant's bank's fee; it is a transfer from the merchant's side to the cardholder's side, and it is the reason your bank wants you to have its card.

**The merchant discount rate (MDR).** This is the *all-in* fee the merchant actually pays to accept a card, expressed as a percentage of the sale. It bundles three things: the interchange (which goes to the issuer), the network's *assessment* fee (which the network keeps), and the acquirer's own markup (which the acquirer keeps for its service and risk). A representative US credit-card MDR is about 2.3%. The merchant sees one number — "you'll pay 2.3% to take cards" — but that number splits three ways.

**Assessment.** A small fee — usually around 0.13% of the sale — that the network (Visa/Mastercard) charges for using its rails. This is what the network itself actually earns per transaction. It's tiny per swipe but enormous in aggregate, because the networks process hundreds of billions of transactions.

An everyday way to hold all of this together: think of a card payment like sending a registered letter between two people who use different post offices. You (the cardholder) drop your payment at your local branch (your issuer). The recipient (the merchant) collects mail at their own branch (the acquirer). Neither of you walks to the other's post office. Instead, a national postal system (the network) carries the letter between the two branches, charges a small handling fee for the route, and enforces the rules about what counts as valid mail. The twist that makes cards different from the post office is that *your* branch — the issuer — gets paid by *their* branch for the privilege of delivering the payment, because your branch is the one that fronted the money and vouched for you. That backwards-seeming payment is interchange. Once you see the card system as "two post offices and a postal network, where the sender's post office gets paid," the whole fee structure stops being mysterious.

That's the whole dictionary, plus the analogy that ties it together. With those nine terms, you can now read every line of the machine. Let's put numbers on it.

## Where the money goes: the MDR split on a real sale

The cleanest way to understand the cards business is to follow one transaction all the way through. So let's run a \$100 credit-card sale and watch the \$2.30 fee split apart.

When you buy \$100 of goods with a typical US rewards credit card, the merchant doesn't receive \$100. It receives roughly \$97.70. The missing \$2.30 is the merchant discount rate, and the chart below shows exactly how it splits.

![Stacked bar showing the merchant discount rate split into interchange network fee and acquirer markup](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-2.png)

The three slices are: interchange of 1.75% (to the issuer), the network assessment of 0.13% (to Visa/Mastercard), and the acquirer's markup of 0.42% (to the merchant's bank or processor). Add them up: 1.75 + 0.13 + 0.42 = 2.30%. That's the MDR.

Notice the proportions, because they're the whole story. **More than three-quarters of the fee — 1.75 out of 2.30 — is interchange, and interchange goes to the bank that issued your card.** The network everyone blames (Visa, Mastercard) keeps only 0.13%, about one-eighteenth of the total. The merchant's own bank keeps 0.42%. The cardholder's bank — the one with no direct relationship to the merchant at all — takes the biggest cut.

#### Worked example: where ~\$100 of a card sale's MDR goes

Let's make it concrete with the curated split. You buy \$100 of groceries and tap a rewards credit card.

- **Sale:** \$100.00. The customer is charged the full \$100.00 — the fee is invisible to the buyer.
- **Interchange (to the issuer):** 1.75% × \$100 = **\$1.75**. This leaves the acquirer and lands in your bank's pocket.
- **Network assessment (to Visa/Mastercard):** 0.13% × \$100 = **\$0.13**. The network's cut for running the switch.
- **Acquirer markup (to the merchant's bank/processor):** 0.42% × \$100 = **\$0.42**. Pays for the terminal, the gateway, fraud tools, and the acquirer's risk.
- **Total MDR:** \$1.75 + \$0.13 + \$0.42 = **\$2.30**.
- **Merchant nets:** \$100.00 − \$2.30 = **\$97.70**.

The intuition: on a card sale, the merchant is silently selling at a 2.3% discount, and the single largest beneficiary is a bank it has never met — the one that issued the customer's card.

The next chart shows the same arithmetic as a "who keeps what" picture on the full \$100. It makes the asymmetry visceral: the merchant keeps the overwhelming majority, but the slice it loses is concentrated in the issuer's hands.

![Horizontal bar of who keeps what on a 100 dollar card sale](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-7.png)

Two things matter here. First, **2.3% sounds small until you put it against a thin retail margin.** A grocery store earning a 2% net profit margin makes \$2 of profit on a \$100 cash sale — but on a \$100 card sale it loses \$2.30 to fees, wiping out the entire profit and then some. That's why some merchants surcharge card payments, offer cash discounts, or steer you toward debit. Second, **the network is not where the money is.** If you wanted to "fix" card fees by squeezing Visa and Mastercard's 0.13%, you'd barely move the merchant's bill. The number that matters is interchange, and interchange is set by the network but paid to thousands of issuing banks. That structural fact — the network sets a fee it doesn't keep — is the root of every interchange fight, and we'll return to it.

The figure below shows the same split as a small taxonomy: one number the merchant pays (the MDR), branching into three streams that flow to three different parties.

![Tree showing the merchant discount rate splitting into interchange network and acquirer fees](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-9.png)

## Authorization, clearing, settlement: how a tap becomes money

We've seen *where* the money goes. Now let's see *how* and *when* it actually moves, because the timing explains a lot of card behavior — why a "pending" charge isn't final, why refunds are slow, and why the issuer is taking real risk the moment you tap.

A card transaction happens in stages, and the gap between "the card said yes" and "the money arrived" is where the bank's whole risk and float live. The pipeline below lays out the five steps.

![Pipeline of authorization capture clearing settlement and billing for a card payment](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-3.png)

**Step 1 — Authorization (about two seconds).** You tap. The terminal sends an authorization request through the acquirer, across the network, to your issuer. The issuer checks the card is valid, not reported stolen, and has available credit (or balance). It runs fraud scoring in milliseconds. If everything passes, it returns an *approval code* and places a *hold* on the amount. No money has moved. The merchant has a promise.

**Step 2 — Capture (end of day).** The merchant *captures* the authorized transactions — confirms the final amounts (which can differ from the authorization; think of a restaurant tip added after the card is run, or a hotel finalizing a bill). Captured transactions are bundled into a *batch*.

**Step 3 — Clearing (overnight to next day).** The acquirer submits the batch to the network, which *clears* it: routing each transaction to the right issuer, validating it, applying interchange and fees, and calculating the net positions — who owes whom and how much. Clearing is the moment interchange is computed and deducted.

**Step 4 — Settlement (T+1 to T+2).** The money moves. Issuers fund their net obligations into the network's settlement bank; the network pays out to acquirers; the acquirer credits the merchant — net of the MDR. The merchant finally has the cash, minus its fee. This is "T+1" or "T+2" in the trade: one or two business days after the sale.

**Step 5 — Billing (later).** Separately, the issuer bills *you*. For a debit card, the money already left your account at authorization-ish timing. For a credit card, the issuer has fronted the money and now waits — it puts the charge on your statement, and either you pay in full (and the issuer earned only the interchange) or you carry a balance (and the issuer starts charging interest, which is where the real money is).

Here is the crucial insight buried in this chain: **the issuer takes on credit risk and funds the transaction at authorization, but doesn't get paid back by you until billing — possibly a month later, possibly never.** The instant you tap a credit card, your bank has effectively made you a tiny, instant, unsecured loan. That is the maturity-transformation spine of this whole series showing up in miniature: the issuer lends long (your revolving balance) funded by short, cheap money (its deposits), earns the spread plus interchange, and survives only as long as it prices the credit risk correctly. A card portfolio is a lending book wearing a payments costume.

#### Worked example: the issuer's float and risk on one credit-card swipe

You buy a \$1,000 TV on a credit card on the 1st of the month. Walk the issuer's position:

- **Day 0:** Issuer authorizes and, at settlement (Day 1–2), pays the network \$1,000 minus its share — effectively the issuer is out \$1,000 to fund your purchase, having collected \$17.50 of interchange (1.75% × \$1,000).
- **Day 1–25:** The issuer has lent you \$1,000 for free during the *grace period*. It funds that \$1,000 from its own deposits, which might cost it, say, 2% annualized — roughly \$1.37 of funding cost over 25 days (\$1,000 × 2% × 25/365).
- **If you pay in full on the due date:** The issuer earned \$17.50 of interchange, paid about \$1.37 in funding cost, and ate its operating and rewards costs. On a single transaction, interchange minus costs is a thin but real profit — and it only works at massive scale.
- **If you carry the balance:** Now the issuer charges you, say, 22% APR. On \$1,000 carried for a year, that's \$220 of interest — *12.6 times* the interchange. This is why issuers are not heartbroken when you don't pay in full.

The intuition: interchange pays the issuer for the *payment*; interest pays it for the *loan*. The whole credit-card business model is a bet that enough cardholders will turn the free payment into a paid loan.

## How an issuer actually makes money

Most people think the card business is "the swipe fee business." It isn't, quite. Interchange is real and large, but for a credit-card issuer it's usually the *second*-biggest line. The biggest is interest on the balances people carry month to month. The chart below shows the typical revenue mix for a US credit-card issuer.

![Stacked bar of how a card issuer makes money from interest interchange and fees](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-4.png)

The three engines of issuer revenue:

**1. Interest (the biggest, for credit).** When you carry a balance past the grace period, the issuer charges interest — often 18% to 30% APR. Because a large share of cardholders *revolve* (carry a balance), interest is typically the dominant revenue line for a credit-card business, on the order of 70% of revenue. This is pure maturity transformation: the issuer borrows at deposit rates and lends to you at card rates, pocketing an enormous spread. Note that this line is *zero* for someone who pays in full every month — the famous "deadbeat" customer, in industry slang, who costs the issuer money on funding and rewards and gives back only interchange.

**2. Interchange (per swipe).** The 1.75%-ish on every purchase, paid by merchants. This is steady, low-risk fee income — the issuer earns it whether or not you carry a balance, and it's the part that funds rewards. For a credit issuer it's typically the second line, around 15–20% of revenue. For a *debit* issuer (where there's no interest, because debit isn't a loan), interchange is nearly the entire revenue.

**3. Fees (the rest).** Late fees, annual fees on premium cards, cash-advance fees, foreign-transaction fees, over-limit fees. Individually small, collectively meaningful — often 10–15% of revenue. Late fees in particular have been a regulatory target, because they fall hardest on customers already struggling.

#### Worked example: issuer revenue = interchange + interest + fees

Let's build a single cardholder's annual economics for the issuer. Suppose you spend \$2,000 a month on the card (\$24,000/year), carry an average revolving balance of \$3,000, and trip one late fee.

- **Interchange:** 1.75% × \$24,000 of spend = **\$420** a year.
- **Interest:** 22% APR on a \$3,000 average balance = 0.22 × \$3,000 = **\$660** a year.
- **Fees:** one \$35 late fee + a \$95 annual fee = **\$130**.
- **Gross revenue from you:** \$420 + \$660 + \$130 = **\$1,210**.

Now the issuer's costs on you: rewards (say 1.5% of \$24,000 spend = \$360), funding the \$3,000 balance at ~2% (\$60), expected credit loss (say 4% of the \$3,000 balance = \$120), and operating/servicing cost (say \$80). Total cost ≈ \$620.

- **Net profit from you:** \$1,210 − \$620 = **\$590** a year.

The intuition: the revolving balance — that \$660 of interest — is what turns you from a break-even payments customer into a \$590-a-year profit center. Interchange alone (\$420) barely covers your rewards (\$360); it's the *interest* that makes the issuer rich, which is exactly why issuers compete so hard to put a card with rich rewards in your wallet.

## The economics of a rewards card: who actually pays for your points

Here's a question that sounds naive but isn't: if your card gives you 2% cashback on everything, who pays for that 2%? It can't be the bank out of pure generosity. Follow the money and the answer is precise and a little uncomfortable: **rewards are funded primarily by interchange, which is funded by merchants, which is funded by everyone's prices — including people who pay cash.**

The logic chain works like this. A premium rewards card carries a *higher* interchange rate than a plain card, because the network sets richer interchange tiers for cards that drive more spending. The issuer collects that higher interchange and hands a chunk of it back to you as points or cashback. The merchant pays the higher interchange. And because merchants generally can't (or won't) charge card users a different price than cash users, they bake the average card cost into the shelf price for everyone. So the cash-paying customer subsidizes the points-earning customer.

#### Worked example: a rewards card's economics for the issuer

You have a premium card with 2% cashback and a \$95 annual fee. You spend \$30,000 a year and always pay in full (so no interest). The issuer's premium card earns a higher interchange — say 2.1% instead of 1.75%.

- **Interchange:** 2.1% × \$30,000 = **\$630**.
- **Annual fee:** **\$95**.
- **Gross revenue:** \$630 + \$95 = **\$725**.
- **Rewards paid to you:** 2% × \$30,000 = **\$600**.
- **Funding + servicing:** say **\$70**.
- **Net to issuer:** \$725 − \$600 − \$70 = **\$55**.

The intuition: even with you paying in full and collecting \$600 of cashback, the issuer still squeaks out a small profit — because the *merchant's* higher interchange (\$630) more than covers your rewards. The merchant funds your points; the issuer's profit is the thin difference plus your annual fee. This is why "free" rewards aren't free: someone in the chain pays, and it's the seller, passed on to all buyers.

This also explains a behavior that puzzles people: why does a merchant pay *more* when you tap a fancy metal card than a basic one? Because premium cards carry premium interchange. The merchant is, in effect, paying for your lounge access. It's also why the largest merchants — Costco, Amazon, big airlines — negotiate ferociously, sometimes refusing certain high-cost cards or cutting exclusive deals, and why the EU and others capped interchange specifically to stop this rewards-funded escalation.

It's worth naming the one escape valve merchants have: pricing. Where network rules and local law allow it, a merchant can offer a *cash discount* (a lower price for cash) or a *surcharge* (an explicit add-on for paying by credit), nudging the cost back onto the card user who triggered it. For decades the networks' "no-surcharge" and anti-steering rules blocked exactly this, which is why merchant lawsuits and regulators have spent so much energy on the *rules* and not just the *rate* — the rules determine whether the cost can be made visible to the cardholder at all. When you see a gas station post two prices, cash and credit, you're looking at the one place the invisible interchange becomes visible. The fight over interchange is therefore also a fight over whether the fee is allowed to stay hidden inside the shelf price, where the cash buyer quietly subsidizes the points buyer.

## Debit versus credit: the same swipe, a very different fee

A debit card and a credit card feel identical at the terminal — you tap, it works. Economically, they could not be more different, and the difference shows up most sharply in interchange. The matrix below lays out the contrast.

![Matrix comparing debit credit and premium rewards cards on interchange and risk](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-6.png)

**Debit** draws money directly from your checking account. There's no loan, no credit risk, no grace period to fund — the issuer is just moving your own money. In the US, *regulated* debit interchange (for large banks) is capped by law at roughly **0.05% plus 22 cents** per transaction under the Durbin Amendment (part of the 2010 Dodd-Frank Act). On a \$100 debit sale that's about 27 cents — a fraction of the \$1.75 a credit card charges. Because debit can't earn interest (it's not a loan), interchange is nearly the issuer's entire revenue on it, which is exactly why banks fought the Durbin cap so hard.

**Credit** is a loan. The issuer fronts the money, carries default risk, funds a grace period, and pays for rewards — so its interchange is far higher (about 1.75%) and uncapped in the US. The merchant pays roughly 6–7 times more to accept a credit card than a regulated debit card.

**Premium rewards credit** sits at the top: interchange of 2.0–2.7%, the richest rewards, and the highest cost to the merchant. The fancier the card, the bigger the bite.

#### Worked example: debit vs credit interchange on the same \$100 sale

A customer buys \$100 of goods. Compare what the merchant pays depending on the card.

- **Regulated debit card:** interchange ≈ 0.05% × \$100 + \$0.22 = \$0.05 + \$0.22 = **\$0.27**. Add the network fee and acquirer markup (say \$0.13 + \$0.42), and the merchant's total cost is roughly **\$0.82**, so it nets about \$99.18.
- **Standard credit card:** interchange = 1.75% × \$100 = **\$1.75**. Total MDR \$2.30, merchant nets **\$97.70**.
- **Premium rewards credit card:** interchange ≈ 2.4% × \$100 = **\$2.40**. Total MDR roughly \$2.95, merchant nets about **\$97.05**.

The intuition: the merchant pays about **\$0.82 to take debit, \$2.30 to take standard credit, and \$2.95 to take a premium card** on the identical \$100 sale. The fee tracks the loan and the rewards, not the goods — which is why merchants love debit, tolerate basic credit, and quietly resent the metal cards.

The Durbin cap is the cleanest natural experiment in interchange. When it took effect in 2011, regulated debit interchange fell by roughly half overnight, transferring billions a year from large banks to merchants. The banks' response is instructive: they cut debit rewards (debit cashback programs largely vanished), raised checking-account fees, and pushed customers toward credit cards (which Durbin didn't cap). The fee didn't disappear; it moved. That's the recurring lesson of payment economics — squeeze the fee in one place and it pops up in another.

## Closed-loop versus open-loop: why Amex is different

We've been describing the *open-loop* model — Visa and Mastercard — where the network is separate from the thousands of banks that issue cards and the thousands that acquire merchants. There's a second model worth understanding because it changes the economics entirely.

In a *closed-loop* network — classically American Express and Discover — the same company is the network *and* the issuer *and* often the acquirer. Amex issues its own cards to cardholders and signs up merchants directly. There's no interchange transfer between two separate banks, because there aren't two separate banks — Amex is both sides. Instead, Amex charges merchants a single, typically *higher* discount rate (historically 2.5–3.5%) and keeps the whole thing.

The trade-off is clean. The closed loop lets Amex capture more economics per transaction and control the customer experience end to end, which funds its famously rich rewards and lets it target affluent, high-spending cardholders. The cost is acceptance: because Amex charges merchants more, fewer merchants take it, especially small ones. (Amex has spent years narrowing this acceptance gap precisely because it limits where its premium cardholders can spend.) The open-loop networks chose the opposite: take a tiny network fee, let banks compete to issue and acquire, and win on ubiquity — Visa and Mastercard are accepted almost everywhere on earth because the model scales.

#### Worked example: closed-loop vs open-loop economics on a \$100 sale

Same \$100 purchase, two network models:

- **Open loop (Visa/Mastercard):** Merchant pays \$2.30 MDR. It splits three ways — \$1.75 to the issuer, \$0.13 to the network, \$0.42 to the acquirer. No single party keeps it all.
- **Closed loop (Amex):** Merchant pays, say, \$3.00 discount rate. Amex keeps essentially all of it — it's the network, issuer, and (often) acquirer combined — minus its own card costs and rewards.

The intuition: Amex earns *more per transaction* but on *fewer transactions* (lower acceptance). Visa earns a sliver per transaction on *vastly more* transactions. Both are excellent businesses; they just sit at opposite ends of the price-vs-ubiquity trade-off. The four-party split exists specifically because the open-loop model needs a way to pay the issuer who took the credit risk — that payment *is* interchange.

## The network's own business: a toll booth on a global highway

It's worth pausing on the network's economics, because the network is the part of the chain people understand least. Visa and Mastercard keep only the ~0.13% assessment per transaction — a tiny slice — and yet they are among the most profitable companies in the world, with operating margins north of 60%. How? The answer is the purest example of a *scale toll-booth* business in finance.

A network doesn't lend, doesn't carry credit risk, doesn't hold deposits, and doesn't fund any transaction. It runs software and a rulebook. Its costs are largely fixed — the data centers, the fraud systems, the brand — and its revenue grows almost one-for-one with transaction *volume*. So once the rails are built, every incremental transaction is nearly pure profit. Process a few hundred billion transactions a year at 0.13% each on an average ticket, and that sliver compounds into tens of billions of dollars of revenue at extraordinary margins.

The network earns in a few ways, all small per swipe and gigantic in aggregate:

- **Assessment / service fees** — the ~0.13% of volume for carrying the transaction. This is the core toll.
- **Data-processing fees** — a small flat fee per transaction *switched*, regardless of size. This rewards the network for ubiquity and high transaction counts (lots of small taps).
- **Cross-border fees** — a richer fee when a transaction crosses currencies or borders, which is far more lucrative than a domestic swipe. This is why the networks love international travel and e-commerce.
- **Value-added services** — fraud scoring, tokenization, analytics, consulting — an increasingly large slice as raw switching commoditizes.

Crucially, the network sits on *both* sides of a two-sided market. It must keep merchants accepting the card (acceptance) *and* keep banks issuing the card (issuance). Interchange is the lever it uses to balance the two: set it high to attract issuers, but not so high that merchants revolt and drop acceptance. That balancing act is exactly what makes interchange so politically fraught, which is the next section.

#### Worked example: the network's revenue on a single \$100 sale

Same \$100 credit-card purchase. What does the *network* actually earn?

- **Assessment:** 0.13% × \$100 = **\$0.13**.
- **Data-processing fee:** a flat amount per transaction, say **\$0.02** (roughly, regardless of ticket size).
- **Network total per swipe:** about **\$0.15** — versus the issuer's \$1.75 of interchange.

Now scale it: at, say, 200 billion transactions a year averaging \$60 each, even \$0.15 a swipe is \$30 billion of revenue, against costs that barely rise with each extra transaction.

The intuition: the network's genius isn't a big cut — it's a *tiny* cut on an *unimaginable* number of transactions, with costs that don't grow. The issuer takes the risk and the big slice; the network takes the toll and the scale. This is why payment networks are valued like software companies, not banks.

## Chargebacks and fraud: the risk hiding inside the fee

We've been treating a card transaction as if it always works. It doesn't. Cards get stolen, customers dispute charges, goods don't arrive, and fraudsters exploit every seam. Managing all of that is a core, expensive part of the card business — and it's a major reason the fees exist in the first place. The 0.42% acquirer markup and a chunk of the 1.75% interchange are, in part, an insurance premium against things going wrong.

The central mechanism is the *chargeback*. A chargeback is the cardholder's right to dispute a transaction and force a reversal — "I never made this purchase," "the item never arrived," "this is a duplicate charge." When a cardholder disputes, the issuer can claw the money back from the merchant through the network, reversing the original flow. The merchant loses the sale *and* typically pays a chargeback fee on top. This is the card's great consumer-protection feature, and it's also a real cost and risk borne mostly by the merchant and the acquirer.

Here's the risk allocation that makes the system work, and it's not symmetric:

- **The issuer** carries *cardholder* fraud risk — if your card is stolen and used, you're generally not liable (the issuer eats the loss, or recovers it from the merchant via chargeback). This protection is a feature interchange pays for.
- **The merchant** carries *transaction* risk in card-not-present (online) sales — if a fraudster uses a stolen card number online, the merchant usually eats the chargeback, because there was no physical card to verify. This is why online merchants invest so heavily in fraud screening.
- **The acquirer** carries *merchant* risk — if a merchant goes bankrupt with a pile of unfulfilled orders (think an airline that collapses before flights are flown), the customers all charge back, and if the merchant can't pay, the *acquirer* is on the hook. This is why acquirers underwrite merchants like lenders, hold reserves against risky ones, and charge more to high-risk sectors.

This last point is underappreciated: **the acquirer's 0.42% isn't just a service fee, it's a risk premium.** Acquiring a travel agency or a ticketing site — businesses that take money long before delivering the service — is genuinely risky, because a single collapse can leave the acquirer holding millions in chargebacks. That's why high-risk merchants pay much higher discount rates, post reserves, or get declined entirely.

#### Worked example: a chargeback on a \$200 online order

A fraudster uses a stolen card number to buy a \$200 gadget from an online merchant. Walk the loss:

- The merchant ships the \$200 gadget, having netted about **\$195.40** after a 2.3% MDR.
- The real cardholder spots the charge and disputes it. The issuer files a chargeback.
- The money is reversed: the merchant loses the **\$200** sale, eats the **\$4.60** of fees it already paid, *and* pays a chargeback fee of, say, **\$15**.
- Total merchant loss: \$200 (goods) + \$4.60 (fees) + \$15 (chargeback fee) = **\$219.60** — more than the sale itself, plus the shipping and the lost inventory.

The intuition: for card-not-present fraud, the merchant — not the issuer or the customer — usually eats the loss, which is why online merchants treat fraud screening as life-or-death and why "fraud" is one of the biggest hidden line items in the card economy. The fee structure is, in large part, the price of running this insurance scheme at scale.

## Why interchange is the most contested fee in finance

Now we get to the heart of it. Interchange isn't just a fee; it's a fee with a structural design flaw that guarantees conflict. The figure below maps the fight.

![Graph showing why interchange is contested with merchant issuer cardholder and regulator incentives](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-8.png)

The problem is this: **the network sets interchange, but the network doesn't pay it or keep it.** The merchant pays it; the issuer keeps it. So the network has every incentive to set interchange *high* — because high interchange attracts banks to issue its cards (more revenue for them) rather than a rival's. This is a competitive dynamic that runs *backwards* from a normal market. Usually, competition pushes prices down. In card networks, Visa and Mastercard compete for *issuing banks*, and they win banks by offering *higher* interchange — so network competition pushes the merchant's price *up*. Economists call this a "reverse competition" or "must-take cards" problem.

Each party's incentive points a different way:

- **Merchants want interchange low.** It's pure cost, and they can't refuse cards without losing sales — they're "must-take." They've spent decades suing, lobbying, and surcharging to push it down.
- **Issuers want interchange high.** It's a major revenue line and it funds the rewards that win customers. Any cut to interchange hits issuer profits directly.
- **Cardholders are conflicted.** They love the rewards interchange funds, but they pay for it invisibly in higher prices on everything they buy — including with cash.
- **Cash and non-card buyers lose unambiguously.** They pay the higher shelf prices that bake in card costs but get none of the rewards. It's a regressive transfer from people who don't use premium cards to people who do.

Because the market can't resolve this on its own (the merchant can't negotiate with thousands of issuers, and the network won't cut a fee it gives away to its bank partners), **regulators have repeatedly stepped in.** The European Union capped credit-card interchange at 0.3% and debit at 0.2% in 2015 — roughly a *fifth* of typical US rates. The US capped debit (Durbin, 2011) but left credit uncapped. Australia, Canada, and others have negotiated or mandated cuts. And in the US, the long-running merchant litigation against Visa and Mastercard has produced multibillion-dollar settlements over the years, with fights over the rules continuing to this day.

#### Worked example: the stakes of a 1% interchange change

Why is a fraction of a percent worth this much warfare? Scale.

- US card purchase volume runs in the trillions of dollars a year — call it roughly \$10 trillion across credit and debit.
- A **1 percentage point** change in average interchange on that volume = 1% × \$10 trillion = **\$100 billion a year** moving between merchants and banks.
- Even the EU's cut from ~1.5% to 0.3% on credit — a 1.2-point drop — redirected an estimated tens of billions of euros a year from banks to merchants (and, in theory, to consumers via prices).

The intuition: interchange is fought over so fiercely because it's a tiny percentage of an astronomically large number. Move the toll by a few basis points and tens of billions of dollars shift between two of the most powerful lobbies in the economy. That's why this fee, more than almost any other in banking, lives in courtrooms and legislatures rather than just in pricing committees.

## A sale from gross to net: the merchant's-eye view

Step back from the banks and look at the transaction the way the merchant experiences it. The merchant agreed to sell a \$100 item. The customer paid \$100. But the merchant's bank account receives \$97.70. The before/after below shows that gap.

![Before and after of a 100 dollar sale showing the merchant nets 97 dollars 70](/imgs/blogs/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split-5.png)

This is why card acceptance is a genuine business decision, not a given. A merchant accepting cards is trading a slice of every sale for the benefits cards bring: more customers (people spend more on credit than cash), higher average tickets, no cash-handling risk or theft, faster checkout, and the simple fact that many customers carry no cash at all. For most merchants the trade is worth it — the extra sales more than cover the 2.3% — which is exactly why cards became must-take. But for low-margin, high-volume businesses (grocery, gas, restaurants), that 2.3% is a serious cost line, and they fight it through surcharges, cash discounts, debit steering, and lobbying.

There's a subtle second-order effect here that ties back to the spine of this series. Card acceptance generates *deposits* for the acquiring bank — the merchant's daily card receipts flow into an account at the acquirer, creating sticky, low-cost transaction balances. So the acquirer isn't just earning its 0.42% markup; it's also gathering cheap funding it can lend out. The merchant's payment business and the bank's lending business are connected through the deposit base. A bank that does a merchant's acquiring tends to do its lending, its payroll, its cash management — the whole relationship — and the card-acceptance fee is partly a loss leader for that broader, deposit-rich relationship. That is the maturity-transformation machine again: even the payments business exists, in part, to feed the cheap-deposit engine that funds the bank's long-term loans.

## Common misconceptions

**"Visa and Mastercard make all that money from card fees."** No — they keep the smallest slice. On a 2.3% MDR, the network's assessment is about 0.13%, roughly one-eighteenth of the fee. The bulk — the 1.75% interchange — goes to the *issuing bank*, the one that gave the customer the card. Visa and Mastercard are wildly profitable, but their money comes from the sheer *volume* of transactions × a tiny per-swipe fee, not from a big cut of each one. Blaming the networks for high card fees mostly misses where the money actually flows.

**"Rewards are free money from the bank."** No — they're funded by interchange, which is paid by merchants, which is built into prices everyone pays. A premium card's 2% cashback is largely the merchant's higher interchange handed back to you. The net effect is a transfer from cash-and-basic-card users (who pay the higher prices but earn no rewards) to premium-card users. The bank is an intermediary, not a benefactor.

**"Debit and credit cost the merchant about the same."** Not even close. In the US, a regulated debit transaction costs the merchant roughly \$0.82 all-in on \$100, while a standard credit transaction costs \$2.30 and a premium card nearly \$3.00. The merchant pays several times more for credit — which is why some steer you to debit and why the Durbin debit cap was such a big deal.

**"The merchant gets paid instantly when I tap."** No. The tap triggers an *authorization* — a promise — in about two seconds, but the actual money settles a day or two later, net of fees. This is why "pending" charges can change or disappear, why refunds take days, and why the issuer is carrying real risk and funding the transaction in the gap between tap and settlement.

**"If we just regulated Visa and Mastercard, card fees would fall."** Capping the *network's* 0.13% would barely move the merchant's bill, because the fee is mostly *interchange* — set by the network but paid to thousands of issuers. That's exactly why effective regulation (the EU caps, Durbin) targets *interchange* directly, not the network's assessment. The structural target has to be the 1.75%, not the 0.13%.

## How it shows up in real banks

**The Durbin Amendment and the great fee migration (2011).** When the US capped regulated debit interchange at roughly \$0.05 + \$0.22 in October 2011, large banks lost an estimated \$6–8 billion a year of debit interchange. The response was textbook fee-balloon behavior: banks killed debit rewards programs, introduced or raised monthly checking-account maintenance fees, and aggressively marketed credit cards (uncapped). Merchants got the fee cut, but studies disagreed on whether prices fell for consumers — much of the gain seemed to stay with merchants. The lesson for reading a bank: a regulatory cap on one fee line tends to migrate, not vanish, so watch the *whole* fee P&L, not one line.

**The EU interchange cap (2015) and the rewards reset.** The EU's Interchange Fee Regulation capped consumer credit interchange at 0.3% and debit at 0.2% across the bloc. The effect on issuer economics was dramatic: with interchange slashed to a fifth of US levels, European card rewards became far stingier than American ones, and issuers leaned harder on annual fees and interest. This is why a US traveler's rewards card is so much richer than a typical European card — the interchange that funds rewards is regulated down across the Atlantic. For a bank analyst, EU card portfolios simply earn less interchange per dollar of spend, structurally.

**The merchant litigation and the multibillion-dollar settlements.** US merchants have litigated against Visa, Mastercard, and issuing banks for decades over interchange-setting and the network rules (like "honor all cards" and anti-steering rules that stop merchants surcharging or steering customers to cheaper cards). The fights have produced some of the largest antitrust settlements in US history and recurring proposed settlements over the rules. The economic stakes — roughly \$100 billion would move for each 1-point change in average interchange on US volume — are why this is fought at the scale of national policy, not pricing committees.

**The processor land-grab: Stripe, Square, Adyen, and the acquirer's markup.** The 0.42% acquirer markup is where modern fintech competition concentrated. Stripe, Square (Block), Adyen, and others rebuilt acquiring with clean software, instant onboarding, and bundled tools — competing on the acquirer's slice (and bundling extra services) rather than the interchange they can't change. Notice the structure: fintechs disrupted the *acquiring* end (the 0.42%) because interchange (1.75%) is fixed by the networks and protected by the issuers. The fee they could attack was the only one set in a competitive market. The deep dive on these players lives in [fintech disruptors: Stripe, PayPal, Ant](/blog/trading/finance/fintech-disruptors-stripe-paypal-ant).

**Costco and the cost of being "must-take."** For years, Costco famously accepted only one credit-card network in its US warehouses, switching its exclusive deal from Amex to Visa (with Citi as issuer) in 2016 — a contract worth so much that both networks fought hard for it. The logic is pure interchange economics: a merchant with enormous volume and razor-thin margins (Costco's net margin sits around 2–3%, and it makes much of its profit from membership fees, not markups) cannot absorb high card fees, so it uses its scale to extract the lowest possible acceptance cost — even at the price of telling its members which card to bring. It's the clearest real-world demonstration that interchange is a negotiated cost of doing business, and that scale is the only leverage a merchant has against a fee it can't otherwise refuse.

**Tokenization and the quiet rebuild of the rails.** When you add a card to a phone wallet, the network doesn't store your real card number on the device — it issues a *token*, a stand-in number that's useless if stolen because it's locked to your device. This tokenization service is one of the network's growing "value-added" revenue lines, and it changed the risk math: card-present-grade security came to online and in-app payments, cutting some fraud. For a bank, tokenization is both a fraud-reduction tool (fewer chargebacks) and a reason the networks keep a grip on the ecosystem — the token vault is theirs. It's a reminder that the networks are not static toll booths; they reinvest the toll into services that deepen their lock on both sides of the market.

**A debit-heavy small bank vs a credit-card monoline.** Compare two issuers. A community bank issuing mostly debit cards earns nearly all its card revenue from the (capped) debit interchange — small, steady, no credit risk, no interest. A credit-card specialist (a "monoline" like a Capital One or a Synchrony) earns the bulk of its card revenue from *interest* on revolving balances, with interchange second and fees third — and it carries real credit risk, so its earnings swing with the credit cycle. When unemployment rises and cardholders default, the monoline's charge-offs spike and profits crater; the debit-only bank barely notices. The same "card business" is two completely different risk profiles depending on whether it's a payment business (debit) or a lending business (credit). For the product-by-product retail lending view, see [consumer lending: mortgages, cards, auto and personal loans](/blog/trading/banking/consumer-lending-mortgages-cards-auto-and-personal-loans).

## The takeaway / How to use this

If you remember one thing about the cards business, make it this: **a card payment is a lending transaction disguised as a payment, and interchange is the toll that pays the lender.** The four-party model exists for one reason — to route a payment between a cardholder and a merchant who bank at different institutions — and interchange exists to compensate the cardholder's bank for the credit risk, the funding, and the rewards it provides. Everything else is detail hung on that frame.

So here's how to actually use this knowledge.

**When you read a bank's results, separate the two card businesses.** Debit interchange is a stable, low-risk payment fee, capped in many places, that barely moves with the economy. Credit-card revenue is dominated by *interest*, which means it's a lending book — it earns a fat spread in good times and bleeds charge-offs in a recession. A bank that calls cards "fee income" is hiding a credit-cycle bet. The single most useful question to ask of any card portfolio is: *how much of this revenue is interchange (a fee) versus interest (a loan)?* The answer tells you whether you're looking at a payments business or a leveraged lender.

**When you wonder who pays for your rewards, trace the interchange.** Your points come from the higher interchange your premium card charges, paid by merchants, baked into prices for everyone. It's a real benefit if you're a heavy spender on a premium card — and a real cost if you pay cash or use a basic card and never see the rewards. Knowing this won't change whether you swipe, but it should change how you read the phrase "rewards card."

**When you see a fight over card fees — in the news, in a courtroom, in a legislature — know what's actually being fought over.** It's almost never the network's 0.13%. It's the interchange, the 1.75%, set by a network that doesn't pay it, paid by merchants who can't refuse cards, kept by thousands of banks. That structural mismatch — the price-setter isn't the price-payer or the price-keeper — is why this one fee can't settle in a market and keeps ending up in front of regulators on three continents.

And the connection back to the spine of this whole series is exact. A credit card is the maturity-transformation machine in its purest, smallest form: the issuer lends you money the instant you tap, funds it with cheap short-term deposits, earns the spread plus a per-swipe toll, and lives or dies on whether it priced your credit risk right. The plastic in your wallet is a tiny, instant, unsecured loan with a payments wrapper — and the entire elaborate machine of issuers, acquirers, networks, and interchange exists to make that loan happen ten thousand times a second and get everyone paid.

## Further reading & cross-links

- [The payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — the clearing, settlement, and correspondent-banking plumbing that sits under every transfer, of which cards are one rail.
- [Domestic payment rails: RTGS, ACH, card networks and instant payments](/blog/trading/banking/domestic-payment-rails-rtgs-ach-card-networks-and-instant-payments) — where card networks fit among the other ways money moves, and the speed-cost-finality trade-offs.
- [Consumer lending: mortgages, cards, auto and personal loans](/blog/trading/banking/consumer-lending-mortgages-cards-auto-and-personal-loans) — the credit-card *lending* book product by product: yields, losses, and behavior through a cycle.
- [Fintech disruptors: Stripe, PayPal, Ant](/blog/trading/finance/fintech-disruptors-stripe-paypal-ant) — how modern processors attacked the acquiring slice of the card stack and rebuilt merchant payments.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the other side of how banks earn fees, for contrast with the steady, high-volume economics of cards.
