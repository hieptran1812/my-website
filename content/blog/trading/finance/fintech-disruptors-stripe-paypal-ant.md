---
title: "Fintech Disruptors: How Stripe, PayPal, and Ant Routed Around the Banks"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How a generation of software companies peeled payments, lending, and wallets off the traditional bank, built some of the most valuable financial firms on earth, and ran headlong into the regulators who guard the old system."
tags: ["fintech", "payments", "stripe", "paypal", "ant-group", "alipay", "nubank", "klarna", "buy-now-pay-later", "neobanks", "interchange", "financial-regulation"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Fintech firms did not replace banks so much as *unbundle* them: they peeled off payments, lending, and wallets into faster, software-first products, and in doing so built some of the most valuable financial companies in the world while colliding with the regulators who protect the old system.
>
> - A *bank* bundles many functions (holding deposits, moving money, lending, issuing cards) under one roof; fintech took those functions apart and rebuilt each one as a standalone app.
> - The money is hidden in plumbing most people never see: every card swipe is split among a *processor*, the *card networks*, and the customer's *issuing bank*, and fintechs make money by inserting themselves into that split.
> - Stripe sells payments as a developer API, PayPal and Block run digital wallets, Ant and Tencent built billion-user super-apps, Nubank and Revolut are banks without branches, and Klarna and Affirm turned the checkout button into a tiny loan.
> - Their revenue comes from a handful of repeating mechanisms: a *take rate* on volume, *interchange* on cards, *float* on money in transit, and interest on lending.
> - When fintech got big enough to threaten the system, regulators reacted hard: China halted Ant's record IPO days before listing, and the buy-now-pay-later boom of 2021 turned into a sharp bust.
> - Being valuable is not the same as being profitable: many of these firms grew enormous before they made a dollar, and some still have not.

Here is a number that ought to stop you: in late 2020, a Chinese payments company called Ant Group was days away from the largest stock-market debut in human history, raising around \$37 billion at a valuation north of \$300 billion. Then, with everything printed and the listing scheduled, regulators in Beijing simply switched it off. The company that had taught a billion people to pay for noodles and movie tickets with a phone was, overnight, told it could not go public. The most valuable fintech on the planet was halted by the state in 48 hours.

The diagram above is the mental model for this entire piece: a traditional bank is a bundle of many separate jobs stacked into one institution, and the story of fintech is the story of software companies peeling those jobs off — one by one — and rebuilding each as a faster, cheaper, app-shaped product. They did not storm the bank's front door. They walked around it.

![A traditional bundled bank on the left versus an unbundled fintech stack on the right](/imgs/blogs/fintech-disruptors-stripe-paypal-ant-2.png)

By the end of this piece you will understand exactly what each of these firms does, how the plumbing of a card payment actually works, where the money is buried, why "routing around the banks" was such a powerful idea, and why the same scale that made these companies valuable also made them collide with the regulators who guard the financial system. We will do all of it from zero — no finance background assumed — and ground every claim in real dollars.

## Foundations: the plumbing nobody sees

Before any of the disruptors make sense, you need to understand the system they disrupted. The trouble is that this system is mostly invisible. When you tap a card and the terminal beeps, it feels instantaneous and free. It is neither. Behind that beep is a relay race among four or five parties, each taking a small cut, settling over a day or two. Let us build the whole thing up from the everyday act of buying a coffee.

### Payment rails and card networks

A **payment rail** is just a shared set of pipes and rules for moving money between accounts. There are several, and they are old. *Cash* is the simplest rail — you hand over paper and it is done. *Bank transfers* (called ACH in the US, SEPA in Europe, or by local names elsewhere) move money directly between bank accounts but slowly, often taking a day or more, in batches. And then there are **card networks** — Visa and Mastercard being the giants — which are the rails almost every plastic or phone payment rides on.

A card network is not a bank and does not hold your money. It is a *switchboard*: a set of rules, a messaging system, and a settlement process that lets thousands of banks talk to each other so that a card issued by a bank in Tokyo works in a shop in Lisbon. Visa and Mastercard make money by charging tiny fees for every transaction that crosses their rails, and because billions of transactions cross every day, "tiny" compounds into one of the most profitable businesses on earth.

The crucial thing to hold in your head is that *the card network is the road, not the car*. The banks are the cars. The merchant's shop is the destination. And the network charges a toll.

### Interchange fees: the toll booth inside every swipe

Here is the single most important fact in payments, and almost nobody outside the industry knows it: **the merchant pays a fee on every card sale, and the biggest slice of that fee goes to the bank that issued the customer's card.** That slice is called **interchange**.

Glossed plainly: *interchange* is a fee, set by the card network but paid to the customer's bank, that the merchant's side hands over on every card transaction. In the US it runs roughly 1.5% to 2.5% on credit cards (lower on debit, after regulation capped it). In Europe regulators capped it far lower, around 0.2% to 0.3%. The merchant never sees interchange as a separate line; it is baked into the total fee they pay their processor.

Why does the issuing bank get paid for *your* purchase? Because the bank took the risk of giving you a card, fronted the money, and runs the rewards program. Those airline miles and 2% cashback you enjoy are funded, in large part, by interchange. The merchant, in effect, subsidizes your rewards.

This is the first piece of the puzzle. The fee on a card payment is real, it is substantial, and it is split among parties the customer never thinks about. Fintech firms got rich by understanding this split better than anyone and inserting themselves into it.

### A payment processor and a payment gateway

When a merchant wants to accept cards, they cannot just plug into Visa directly. They need intermediaries. Two terms matter, and they are routinely confused.

A **payment gateway** is the software that captures the card details at the moment of sale — the digital equivalent of the card-reader slot — and securely passes them onward. A **payment processor** (often the same company today) is the party that actually shepherds the transaction through the networks and the banks: it asks the issuing bank "is this card good for \$100?", gets a yes or no, and then handles the settlement that moves the money a day or two later. The bank that ultimately receives the funds on the merchant's behalf is the **acquirer** (or *acquiring bank*).

For decades this was a miserable experience. A small business wanting to accept cards online had to negotiate with a bank, fill out paper forms, wait weeks for a "merchant account", and wire up clunky gateway software. The whole stack was hostile to anyone without a procurement department. Hold that thought — it is the exact pain Stripe was built to kill.

It is worth pausing on the fact that a card payment happens in two distinct steps, separated in time, because that gap is where some of the money is made. The first step is **authorization**: at the instant of sale, the processor asks the issuing bank "is this card valid and is there \$100 of available credit?", and the bank places a *hold* on the funds and answers yes or no in well under a second. The customer walks away with the goods. But the money has not actually moved yet. The second step, **settlement and clearing**, happens hours or a day or two later, in a batch, when the funds are genuinely transferred from the issuing bank through the network to the acquirer and finally to the merchant's account. In between, the money is *in transit* — and money in transit, sitting somewhere overnight, earns interest for whoever is holding it. That interest is the *float* we will meet again and again. Most consumers never register that the beep at the terminal and the money landing in the merchant's bank are two separate events with a day-long gap between them, but that gap is a quiet profit center.

![The payments value chain shown as four stacked layers from bank rails up to a developer API](/imgs/blogs/fintech-disruptors-stripe-paypal-ant-6.png)

The figure above is the payments value chain as a stack. At the bottom are the **rails** — Visa, Mastercard, the banking system — which fintechs almost never own. Above that is **acquiring and settlement**, the boring bank function of actually paying the merchant. Above that is the **gateway and processor** layer that authorizes and routes each payment and screens for fraud. And at the top is the **developer-facing API and checkout** that the merchant actually touches. The pattern that defines fintech is this: they captured the top layer — the part humans and developers see — while renting the rails beneath. They became the friendly face on top of plumbing they did not have to build.

### A neobank, a wallet, BNPL, and the super-app

Four more terms, each naming a category we will meet repeatedly.

A **digital wallet** is an app that stores a balance or links your cards and lets you pay or send money without re-entering details. PayPal is the original; Alipay and WeChat Pay are the giants. A wallet sits *on top of* the rails and abstracts them away.

A **neobank** is a bank-like app with no branches. It offers a checking account, a debit card, often savings and lending — but it is built mobile-first, frequently runs on top of a partner bank's license, and serves customers the incumbents ignored. Nubank in Brazil and Revolut in the UK are the archetypes.

**Buy-now-pay-later (BNPL)** is a way of splitting a purchase into a few interest-free instalments — say four payments of \$50 on a \$200 jacket — financed at the point of sale. The lender (Klarna, Affirm, Afterpay) pays the merchant in full immediately, charges the merchant a fee for the privilege, and collects the instalments from the shopper over weeks. It is, mechanically, a tiny short-term loan dressed up as a checkout button.

A **super-app** is a single app that bundles payments, messaging, shopping, lending, investing, and dozens of other services into one place. The Chinese giants Alipay (from Ant) and WeChat Pay (from Tencent) are the defining examples, and we will spend real time on why they took a shape the West never replicated.

### The unbundling of the bank

Now we can state the thesis precisely. A traditional bank is a **bundle**: it takes your deposits, runs your checking account, issues your debit and credit cards, moves your money, lends to you, and sells you investments — all under one regulated roof, mostly through branches, mostly slowly. For a century that bundle was a moat, because assembling all those functions was hard and expensive.

The internet and the smartphone dissolved the moat. Suddenly a startup could take *one* slice of the bundle — just payments, just the checking account, just the loan at checkout — and do that single thing dramatically better: faster to sign up, cheaper, prettier, available at 2 a.m. on a phone. This is **unbundling**: attacking the bank not as a whole but function by function, where each attacker is small enough to be ignored at first and focused enough to win its slice.

![Branching diagram showing the bundled bank fanning out into payments, wallets, accounts, and credit fintechs](/imgs/blogs/fintech-disruptors-stripe-paypal-ant-4.png)

The figure above shows the fan-out. The bundled bank in the center is peeled into payments (Stripe, Square), wallets (PayPal, Alipay), accounts (Nubank, Revolut), and credit (Klarna, Affirm) — and every one of those pieces ultimately funnels back into the same handful of money-making mechanisms: take rates and float. That is the whole game in one image. Keep it in mind, because once you see fintech as unbundling, every company in this piece slots neatly into one of those four branches.

#### Worked example: the \$100 swipe

Let us follow a single coffee-shop sale and watch the cut get split, because the arithmetic is the foundation of everything that follows.

A customer buys \$100 of goods with a US credit card. Here is what happens to that \$100:

```
Customer pays:                      $100.00
Interchange (to issuing bank):       -$1.70   (~1.7%, set by network, paid to card issuer)
Network scheme fee (Visa/MC):        -$0.13   (~0.13%, the toll on the rail)
Processor markup (e.g. Stripe):      -$1.07   (the processor's own margin)
                                    --------
Total merchant fee:                  -$2.90   (≈ 2.9% of the sale)
Merchant receives (settled):         $97.10
```

So on a \$100 sale the merchant nets about \$97.10. Of the \$2.90 in fees, the largest slice (~\$1.70) flows to the *customer's* bank as interchange, a sliver (~\$0.13) to Visa or Mastercard for use of the rail, and the rest (~\$1.07) to the processor. Three different companies just got paid out of one coffee, and the customer noticed none of it.

The intuition: a card payment is not one transfer but a relay, and at every hand-off someone clips a coupon — which is precisely why owning a hand-off is so lucrative.

![A one hundred dollar card payment relayed across cardholder, processor, network, and issuing bank](/imgs/blogs/fintech-disruptors-stripe-paypal-ant-1.png)

The figure above traces that exact relay. The cardholder pays \$100 to the acquirer/processor, which routes the authorization through the card network to the issuing bank; the issuer takes its interchange, the network takes its scheme fee, the fees total about \$2.90, and the merchant is settled the remaining \$97.10. Notice who is colored as the intermediary (lavender — the network and the issuing bank) and who is the merchant winning the net amount (green). This single picture is the substrate every fintech in this article is standing on.

## The disruptors, one at a time

With the plumbing understood, we can meet the companies and see exactly which slice each one attacked and how it made money. We will go category by category, and for each we will be concrete about the mechanism and the dollars.

![A matrix of Stripe, PayPal, Ant, Nubank, and Klarna by niche, revenue model, and scale](/imgs/blogs/fintech-disruptors-stripe-paypal-ant-3.png)

The matrix above is the cheat sheet for the whole section: five firms, three columns — their core niche, how they actually make money, and rough scale. Read down the "main revenue" column and you will see the four mechanisms from the foundations recurring: take rate, transaction fees, lending, interchange, and float. The figures (volumes, account counts, user numbers) are approximate and as of recent reporting; treat them as orders of magnitude, not precise current values.

### Stripe: developer-first payments infrastructure

In 2010, two Irish brothers, Patrick and John Collison, looked at the misery of accepting payments online and asked a simple question: what if a developer could start accepting cards in an afternoon, with a few lines of code, instead of weeks of bank paperwork? Stripe was the answer. Its famous early pitch was that you could integrate payments by pasting in seven lines of code.

Mechanically, Stripe is a *payment processor and gateway* (the top two layers of our stack) sold as an **API** — an application programming interface, meaning a set of code endpoints another program can call. A developer sends Stripe a request that says "charge this card \$100", and Stripe handles everything beneath: the gateway capture, the authorization through Visa or Mastercard, the fraud screening, the settlement to the merchant's bank, even the compliance paperwork. The merchant never touches a bank or fills out a merchant-account form. Stripe abstracted the entire ugly stack into a clean software product.

How does Stripe make money? With a **take rate** — glossed as the fraction of every dollar of payment volume that the company keeps. Stripe's headline price in the US is famous: **2.9% + \$0.30 per transaction** for standard online card payments. That \$0.30 fixed component matters enormously for small payments and matters little for large ones, which shapes who Stripe is cheap or expensive for.

Stripe does not keep the whole 2.9%. Recall the \$100 swipe: most of that fee is interchange and network fees Stripe must pass through to the issuing bank and Visa. Stripe's own margin is the markup on top — the ~\$1.07 in our example. Stripe's genius was not a cheaper price; for years it was *more* expensive than negotiating directly with a bank. Its genius was that it made the price *not matter* by making integration trivial. Developers chose Stripe because it saved them weeks of work, and businesses follow their developers.

#### Worked example: Stripe's take rate on a SaaS subscription

Suppose a software startup runs \$1,000,000 of customer payments through Stripe in a month, all standard online card charges, across 5,000 individual transactions averaging \$200 each.

```
Volume:                          $1,000,000
Percentage fee (2.9%):              $29,000
Fixed fee ($0.30 x 5,000 txns):      $1,500
                                  ---------
Total Stripe charge:                $30,500   (≈ 3.05% effective)
```

The startup pays Stripe about \$30,500 — an effective rate of 3.05%, slightly above the headline 2.9% because of the per-transaction \$0.30 adding up across 5,000 charges. Of that \$30,500, the bulk flows through to issuing banks and networks as interchange and scheme fees; Stripe's retained margin is the thinner slice on top. But because Stripe processes well over \$1 trillion of volume a year (approximate, as of recent reporting), even a thin margin on a vast river of money is an enormous business.

The intuition: Stripe's product is not low price but low friction, and it monetizes friction-removal by taking a small, relentless percentage of an immense and growing flow.

There is a deeper strategic reason Stripe aimed at developers rather than at finance departments. In a modern company, it is increasingly the engineers who decide which tools the business uses, because they are the ones building the product. If you make the engineers' lives easy — clean documentation, code that works on the first try, sane error messages — they will reach for your tool by default, and the purchasing decision follows the integration rather than preceding it. This is the classic "bottom-up" software-distribution playbook applied to money. Stripe also relentlessly expanded *upward* from raw payments into adjacent jobs that sit on top of the same flow: subscription billing, invoicing, marketplace payouts, fraud detection, sales-tax calculation, even helping companies incorporate. Each of these is another thin take on the same river of payment volume, and each one makes a customer who already uses Stripe for payments more likely to stay. The strategy is to own the entire financial *operating system* of an internet business, not just the checkout button — which, not coincidentally, is the same unbundle-then-rebundle pattern we will see Ant attempt at the scale of a nation.

### PayPal and Block/Square: digital wallets and merchant tools

PayPal is the elder statesman of fintech, founded in 1998 (partly out of a company called Confinity, with a young Elon Musk's X.com merging in) to let people send money by email. Its breakthrough was riding eBay: in a marketplace of strangers, PayPal let a buyer pay a seller they would never meet, and PayPal sat in the middle holding the trust. eBay bought it in 2002.

A **digital wallet** like PayPal makes money the same essential way a processor does — a fee on transaction volume — but it owns something the processor does not: the *consumer relationship*. You have a PayPal account, a PayPal balance, a PayPal login. That account is the moat. When you check out with PayPal, the merchant pays PayPal a fee (often around 2.9% + a fixed amount, mirroring card economics), and PayPal handles the rest. PayPal also earns **float** — interest on money sitting in user balances and in transit — and increasingly from lending and other services.

Block (formerly Square), founded by Jack Dorsey in 2009, attacked a different slice: the *small physical merchant*. Square's first product was a tiny white card reader that plugged into a phone's headphone jack, letting a food-truck owner or a farmers-market vendor take cards with no bank relationship and no fixed terminal. Square charged a flat, simple rate (around 2.6%-2.75% + a small fixed fee for a swipe). It then expanded upward into a full small-business toolkit — payroll, loans, inventory — and sideways into consumers with Cash App, a peer-to-peer wallet that also sells stock and bitcoin. Block's two halves (Square for sellers, Cash App for individuals) are each a different unbundled slice of the old bank.

The pattern across PayPal and Block: own a relationship the bank was too slow or too snobbish to serve — strangers on eBay, vendors at a market, the unbanked on Cash App — and monetize the payments that relationship generates.

### Ant Group/Alipay and Tencent/WeChat Pay: China's super-apps

To understand why China produced something the West did not — the super-app — you have to understand that China largely *skipped* the credit-card era. Where the US spent decades wiring up card terminals, hundreds of millions of Chinese consumers went straight from cash to the smartphone. There was no entrenched card network to route around; the road itself was being paved fresh, and two tech companies paved it.

**Alipay** began in 2004 as an escrow service for Taobao, Alibaba's eBay-like marketplace: a buyer's money sat with Alipay until the goods arrived, solving the same stranger-trust problem PayPal solved for eBay. From that beachhead it became the default way to pay for *everything* in China via a QR code — you scan a code at a street stall, the money moves instantly between Alipay balances, and crucially *the card networks are bypassed entirely*. No Visa, no interchange, no merchant terminal. Alipay spun out into **Ant Group**, which layered on a money-market fund (Yu'e Bao, at one point the largest in the world), consumer lending (Huabei and Jiebei), insurance, and credit scoring.

**WeChat Pay**, from Tencent, did the same thing but starting from a messaging app a billion people already used daily. Paying a friend became as easy as sending a text. The famous "red envelope" feature — sending small cash gifts in chat during Lunar New Year — virally onboarded hundreds of millions of users to payments in a single holiday.

Both companies make money in ways that go far beyond a thin payment fee. The payment is almost a loss-leader; the profit is in **lending** (using transaction data to underwrite tiny loans at scale), in **float and fund management** (parking idle balances in money-market funds and earning the spread), and in being the *distribution channel* for every other financial product. This is why Ant was valued at over \$300 billion: it was not a payments company, it was a financial-services operating system for a billion people. And that, precisely, is what alarmed the state.

The data advantage here is hard to overstate and is the real engine of the super-app. A traditional bank deciding whether to lend you \$500 has a thin file: your declared income, a credit score, maybe a few years of account history. Alipay, by contrast, could see what you bought, where, when, how reliably you repaid the last micro-loan, who you transacted with, and how your spending trended week to week — all in real time, across hundreds of millions of people. That firehose of behavioral data let Ant underwrite tiny loans (sometimes equivalent to a few dollars) profitably and almost instantly, at a scale and cost no branch-based bank could match. The payments business was, in effect, a sensor network that fed the lending business. This is the purest illustration of the entire fintech thesis: the payment is the hook that generates the data, and the data is what makes the lending and the rest of the bundle profitable. It is also why "just a payments app" turned out to be a misnomer — and why a regulator, looking at the same picture, saw a systemically important lender hiding inside a consumer app.

### Nubank and Revolut: neobanks

A **neobank** attacks the most basic slice of all — the checking account and the debit/credit card — by deleting the branch.

**Nubank**, founded in Brazil in 2013, is the standout. Brazilian banking was famously concentrated, expensive, and exclusionary: a handful of big banks charged high fees, demanded paperwork, and left a huge share of the population either unbanked or badly served. Nubank launched with a single product — a no-fee purple credit card managed entirely from an app — and grew by word of mouth into one of the largest digital banks in the world, serving roughly 100 million-plus customers across Brazil, Mexico, and Colombia (approximate, as of recent reporting). It went public in 2021 at a valuation around \$45 billion, briefly worth more than some of the incumbent banks it was disrupting.

**Revolut**, founded in the UK in 2015, started as a way to spend abroad without the punishing foreign-exchange fees banks charged, then bolted on accounts, stock trading, crypto, and more across Europe and beyond.

How does a neobank make money on a "free" account? Two main ways. First, **interchange**: every time you swipe the neobank's debit or credit card, the neobank (as the issuer, or sharing in the issuer's economics) collects interchange from the merchant's side. Multiply a small per-swipe amount by tens of millions of customers swiping daily and it adds up. Second, **lending and float**: the neobank lends out deposits, offers credit cards and personal loans, and earns the spread between what it pays depositors and what it charges borrowers — the classic bank business, just with a far lower cost base because there are no branches.

#### Worked example: a neobank's unit economics per customer

Consider a simplified neobank customer who spends \$1,000 a month on the neobank's debit card and keeps a \$500 average balance.

```
Interchange on $1,000/mo spend (~1.0% blended debit):   $10.00 / month
Net interest on $500 balance + small loan book (~3%):    $1.25 / month
                                                        ----------
Gross revenue per customer:                             $11.25 / month  = $135 / year
Cost to serve (app, support, fraud) per customer:        -$60 / year (illustrative)
                                                        ----------
Contribution per customer per year:                      ~$75
```

So a single active customer might throw off on the order of \$135 a year in revenue and perhaps \$75 in contribution after the cost of serving them — but only *if* they are active. A customer who signs up, gets the free card, and never uses it is pure cost. This is why neobanks obsess over "engagement" and "primary account" status: their entire model depends on each cheap-to-acquire customer actually transacting. The brutal arithmetic of the model is that profit is the difference between a tiny per-customer margin and a tiny per-customer cost, multiplied by tens of millions of people — which only works at enormous scale, and only if customers are active.

The intuition: a neobank is a bet that software lets you serve a customer so cheaply that even a few dollars of monthly interchange and interest turns a profit — but the bet pays off only at massive scale with genuinely active users.

### Klarna and Affirm: buy-now-pay-later

BNPL is the cleverest unbundling of all, because it disguises a *loan* as a *feature*. When you reach checkout and see "4 interest-free payments of \$50", a lender (Klarna, Affirm, Afterpay) is offering to pay the merchant your full \$200 right now and let you pay it back over six weeks.

Why would a merchant want this? Because BNPL demonstrably increases sales — shoppers buy more, and bigger, when the pain of payment is spread out. So the merchant happily pays BNPL a fee, typically much higher than card interchange — often **4% to 6%** of the purchase — for the bump in conversion and basket size. That merchant fee is the core of the BNPL business. On the classic "pay in 4, interest-free" product, the consumer pays no interest, so the merchant fee plus late fees plus the **float** (the lender holds and recycles money) is the revenue. On longer BNPL plans (Affirm in particular offers 6-, 12-, or 36-month financing), the lender also charges the consumer interest, like a traditional installment loan.

It is worth being precise about how Klarna and Affirm differ, because they represent two flavors of the same idea. Klarna, founded in Sweden in 2005, built its business around the short, interest-free "pay in 4" product and around being a shopping-and-discovery app in its own right — its revenue leans heavily on merchant fees, late fees, and float, with relatively less consumer interest. Affirm, founded in the US in 2012 by PayPal co-founder Max Levchin, leaned earlier and harder into *longer* installment loans (6, 12, even 36 months) that explicitly charge the consumer interest, and made a selling point of *never* charging late fees, positioning itself as the transparent alternative to credit cards. So Klarna is closer to "a merchant-funded short-term split", and Affirm is closer to "a transparent point-of-sale installment lender". Both are lending; they just place the cost on different parties and over different horizons.

The model is seductive and dangerous in equal measure. It is seductive because the lender gets paid by the merchant up front and the consumer gets an interest-free split. It is dangerous because the lender is, fundamentally, extending unsecured credit to consumers — often young, often without a thick credit history — and if those consumers stop paying, the losses land squarely on the lender. BNPL is a lending business wearing the costume of a payment button, and lending businesses live and die on credit losses. The whole sector is therefore exquisitely sensitive to the economy: in good times, defaults are low and the thin margins hold; in a downturn, defaults rise across the entire loan book at once, and because the margin on each sale was only a few dollars to begin with, even a modest rise in losses can flip the business from profit to loss almost overnight.

#### Worked example: BNPL on a \$200 purchase

A shopper buys a \$200 jacket using a "pay in 4, interest-free" plan from a BNPL provider.

```
Merchant's side:
  Sale price:                              $200.00
  BNPL merchant fee (~5%):                  -$10.00
  Merchant receives immediately:           $190.00

Consumer's side (4 payments of $50, 0% interest):
  Today:   pay $50
  +2 wks:  pay $50
  +4 wks:  pay $50
  +6 wks:  pay $50   -> total paid $200, no interest IF on time

BNPL provider's economics on this one sale:
  Merchant fee collected:                   +$10.00
  Funding cost (advance $200 for ~6 wks @ 8% APR):  -$1.85
  Expected credit loss (say 2% default x $200):     -$4.00
                                            ----------
  Approx gross profit per on-time sale:      ~$4.15
```

So the BNPL provider makes roughly \$4 on a \$200 sale *if the consumer pays on time*. The merchant gives up \$10 (5%) — far more than the ~2.9% they would pay on a card — but accepts it because BNPL lifts both conversion and average order size. The consumer, if disciplined, pays nothing extra. But notice how thin the provider's margin is and how much of it is eaten by the assumed 2% credit loss: if defaults double to 4%, that \$4 profit becomes a \$4 loss. BNPL profitability is a knife-edge that swings entirely on whether consumers keep paying — which is exactly what went wrong in the bust we will come to.

The intuition: BNPL shifts the cost of credit onto the merchant up front and bets that consumers repay on time; the whole model is a wager on credit losses staying low.

## How they make money: the four mechanisms

Step back and notice that across all these companies, the revenue comes from just four recurring mechanisms. Once you can name them, you can decode any fintech's business model in a sentence.

1. **Take rate** — a percentage of payment volume the company keeps. Stripe's ~2.9% + \$0.30 is the canonical example. The number to watch is not the headline rate but the *net* take rate after passing through interchange and network fees, which is far smaller.

2. **Interchange** — the fee the issuing side earns on every card swipe. Neobanks and wallets that issue their own cards capture this. In the US it is meaningful (often 1%-2%); in Europe, regulators capped it near 0.2%-0.3%, which is precisely why European neobank economics are harder.

3. **Float** — interest earned on money the company is holding or moving on someone else's behalf. A wallet balance, a BNPL advance, settlement money in transit overnight — all of it earns interest before it reaches its destination, and at scale that interest is real money. Float quietly became more valuable for everyone in 2022-2023 as interest rates rose.

4. **Lending** — the oldest financial business of all. Use the transaction data you already have to underwrite loans (consumer credit, BNPL, merchant cash advances) and earn the spread between funding cost and the rate you charge. This is where the *most* money is — and the most risk.

The decisive insight is that payments are often the *hook*, not the profit. Payments get you the customer, the relationship, and above all the **data**. The profit frequently comes later, from lending and float layered on top of that relationship. Ant is the purest expression of this: it gave away cheap payments to a billion people and made its money lending to and managing the savings of those same people. That is also exactly why regulators decided Ant was not a tech company but a bank — and banks, they insisted, must be regulated like banks.

#### Worked example: how float quietly prints money

Float is the most invisible of the four mechanisms, so let us make it concrete. Suppose a wallet or processor holds, on average, \$2 billion of customer balances and in-transit settlement money on any given day — money that belongs to users and merchants but sits in the company's accounts for a day or two before reaching its destination.

```
Average money held (float):          $2,000,000,000
Interest the company earns on it:    ~5% per year (e.g. parked in short-term Treasuries)
Annual float income:                 $2,000,000,000 x 5% = $100,000,000
```

That is \$100 million a year earned not by charging anyone a fee, but simply by holding other people's money for a short while at a time when safe interest rates are around 5%. The company did no extra work and took no credit risk; it just sat on the float. Crucially, this number swings violently with interest rates: in 2021, when safe rates were near zero, that same \$2 billion of float earned almost nothing, so float income was a rounding error. By 2023, with rates near 5%, float had become a major profit line for wallets and brokers alike. This is why the rise in interest rates was simultaneously a disaster for fintech *lenders* (their funding got expensive) and a windfall for fintech *holders of float* (their idle balances started paying).

The intuition: float is interest earned on money that is merely passing through your hands, so it costs nothing to produce but scales with both your volume and the level of interest rates — which is why the 2022-2023 rate surge reshuffled who in fintech was winning.

## The regulation collision

For a long time, fintech enjoyed a quiet privilege: it was treated as *software*, not as *finance*. Software is lightly regulated and moves fast. Banking is heavily regulated and moves slowly, because a bank failure can take down an economy. As fintechs grew from cute apps into systemically important conduits for trillions of dollars, that gap became untenable, and the regulators moved in. Three collisions tell the story.

![Timeline of fintech milestones from 1998 to 2022, with the 2020 Ant IPO halt marked in red](/imgs/blogs/fintech-disruptors-stripe-paypal-ant-5.png)

The timeline above lays out the arc: PayPal in 1998, Alipay in 2004, Stripe in 2010, the PayPal-eBay split in 2015, the Ant IPO halt in 2020 (marked in red as the rupture), Nubank's blockbuster IPO in 2021, and the BNPL bust in 2022. Read it as two decades of unbundling cresting into a regulatory reckoning. The 2020 halt is deliberately the visual pivot — it is the moment the era's optimism met the era's limits.

### The Ant IPO halt (2020)

The defining collision. In November 2020, Ant Group was set to list simultaneously in Shanghai and Hong Kong in what would have been a roughly \$37 billion offering — the largest IPO ever — at a valuation above \$300 billion. Days before, after a now-famous speech in which Ant's founder Jack Ma publicly criticized Chinese financial regulators as having a "pawnshop mentality", Beijing suspended the listing.

The official rationale was substantive, not merely punitive. Ant had built an enormous *lending* business by originating loans and then selling the credit risk on to partner banks while keeping the fees — meaning Ant earned like a lender but held almost none of a lender's capital cushion against losses. Regulators argued this was a bank-sized risk masquerading as a tech platform, and they were not wrong about the mechanism. In the aftermath, Ant was forced to restructure into a financial holding company subject to bank-style capital rules, its lending was reined in, and its valuation was slashed — by some later estimates to well under half its pre-halt peak. The most valuable fintech on earth had been re-classified, overnight, as the bank it always functionally was.

#### Worked example: Ant's valuation, before and after

The arithmetic of what regulation did to Ant's worth.

```
Pre-halt (Nov 2020):
  IPO valuation:                   ~$315 billion
  Implied as a "tech platform" multiple on fee revenue

Post-halt / restructured:
  Reclassified as financial holding co, bank-style capital rules
  Lending capped, growth slowed
  Later internal/secondary valuations:   ~$80-150 billion (approximate, varied by source)

Value erased by reclassification:        well over $150 billion
```

In round numbers, treating Ant as a tech company versus as a regulated lender was worth on the order of \$150 billion or more in valuation. Nothing about the underlying cash flows changed in those 48 hours — what changed was the *category* the market and the regulators agreed to put Ant in. (These figures are approximate and as of various points after 2020; private and secondary valuations vary widely.)

The intuition: a fintech's valuation depends enormously on whether the market believes it is a lightly regulated tech platform or a heavily regulated bank, and a regulator can move it from one bucket to the other with a single ruling.

### BNPL scrutiny

As BNPL exploded in 2020-2021, regulators in the US, UK, and Australia grew uneasy for concrete reasons. BNPL sat in a gap: it was credit, but because the classic "pay in 4" product charged no interest and ran short, it often escaped the disclosure rules, affordability checks, and credit-reporting that govern credit cards and personal loans. That meant a consumer could stack multiple BNPL plans across multiple providers, with none of them seeing the others, building up hidden debt no credit bureau could observe. Regulators worried about over-indebtedness among young and lower-income users, about thin disclosures, and about late fees. The UK moved to bring BNPL under formal consumer-credit regulation; the US consumer regulator pushed to treat large BNPL providers more like credit-card issuers, including dispute rights. The "feature, not a loan" framing was, regulators decided, exactly the problem.

### Neobank-charter fights

Neobanks face a structural bind: to do real banking — hold insured deposits, lend on their own book — you need a banking *charter*, the license that comes with the heaviest regulation and capital requirements. Many neobanks avoid the charter by partnering with a small chartered bank that holds the deposits while the neobank runs the app and the relationship (the "banking-as-a-service" model). This is faster but fragile: regulators have increasingly scrutinized these partnerships, and several blew up when the middleware connecting fintech apps to partner banks failed, freezing customers out of their own money. Some neobanks (including in the US) have pursued their own charters to escape the dependency, only to find the application process slow and grueling — the very moat the incumbents enjoy. Routing around the bank works beautifully until the function you are performing *is* banking, at which point the regulator insists you become one.

There is a subtle but important consumer-protection issue buried in the banking-as-a-service arrangement. When a customer keeps money in a neobank app, the deposit insurance that makes a bank safe (in the US, FDIC insurance up to a limit) attaches to the *partner bank's* account, not to the app. As long as the records are clean — the partner bank knows exactly which dollars belong to which app customer — that protection flows through. But if the middleware company keeps sloppy records, or fails, the chain can break, and customers can find their insured money inaccessible while administrators painstakingly reconstruct who is owed what. This is not a hypothetical: when a prominent middleware provider collapsed, hundreds of thousands of consumers across many fintech apps had their funds frozen for months because nobody could immediately prove which balances were theirs. The lesson regulators drew, and are still acting on, is that pushing the customer relationship into a lightly regulated app while parking the actual money and the actual insurance one or two layers away creates exactly the kind of opacity that bank regulation exists to prevent.

## Profitability: valuable is not the same as profitable

A theme runs under all of this that deserves its own section, because it is the most common misunderstanding about fintech. These companies became staggeringly *valuable* — measured by what investors would pay for them — long before, and sometimes without, becoming *profitable* — measured by whether they actually earned more than they spent.

The fintech playbook, especially in the 2010s era of cheap money, was to grow users and volume at almost any cost, on the theory that scale and the eventual layering of high-margin lending and float would deliver profit later. Sometimes it worked: PayPal is solidly profitable; Nubank turned profitable at scale; Stripe is widely believed to be profitable on its core processing. Sometimes it took a brutally long time: Block's two halves have run hot and cold. And sometimes the profit never convincingly arrived, which is exactly what the BNPL bust exposed.

When interest rates rose sharply in 2022, the music stopped. Capital became expensive, growth-at-all-costs fell out of fashion, and investors abruptly demanded a path to profit. Fintechs that had been valued as hyper-growth tech were repriced as what their cash flows said they were. Klarna, valued at around \$46 billion in mid-2021, raised money again in 2022 at roughly \$6.7 billion — an 85% cut — not because its business had collapsed but because the market stopped paying tech multiples for a lending business with rising credit losses and high funding costs. The lesson, written in red, was that for a fintech doing lending, profitability is gated by two things it does not fully control: the cost of money and the rate of customer defaults.

It helps to see why the *type* of fintech determines how hard profitability is. A pure processor like Stripe has it relatively easy: it takes a fee on volume, holds little risk on its own balance sheet, and its costs scale slower than its revenue, so once volume is large enough the margins are durable. A neobank has it harder: it must acquire customers cheaply, get them to actually transact, and earn enough interchange and net interest to cover the cost of serving them — a real but achievable bar at scale, as Nubank proved. A BNPL lender has it hardest of all, because its core product is unsecured consumer credit, and that means it carries the two risks no software company wants: *funding risk* (it must borrow money to lend it out, and that borrowing gets expensive exactly when the economy weakens) and *credit risk* (its borrowers default exactly when the economy weakens). Those two risks arrive together, which is why a BNPL book can look pristine for years and then deteriorate sharply all at once. The general rule that emerges is intuitive once stated: the more a fintech's profit comes from *fees on flow*, the more robust it is; the more it comes from *lending*, the more it behaves like — and must be capitalized like — a bank.

## Common misconceptions

**"Fintechs replaced the banks."** Mostly false. Almost every fintech in this piece still *rides on* the banking system: Stripe settles through banks and runs on Visa and Mastercard rails; neobanks often hold deposits at a partner bank; even Alipay touches the banking system at the edges. They unbundled the bank's *customer-facing functions*, but the regulated plumbing underneath is still largely the banks'. Fintech is a layer on top, not a replacement beneath.

**"The 2.9% Stripe charges is Stripe's profit."** False. The bulk of that 2.9% is interchange and network fees Stripe must pass through to the issuing bank and Visa or Mastercard. Stripe's own margin is the markup on top, a fraction of the headline rate. The same is true for nearly every processor and wallet.

**"Buy-now-pay-later is free money."** Misleading. For a disciplined consumer who pays on time, "pay in 4" can genuinely be interest-free — but the cost is paid by the merchant (often 4%-6%), and the model relies on a meaningful share of users paying late (fees) or, on longer plans, paying interest. BNPL is not free; it is a loan whose cost has been rearranged so the shopper feels it least.

**"Super-apps failed in the West because Western consumers don't want them."** Too simple. They failed largely because the West already had entrenched, working systems — credit cards, established banks, a regulatory structure that compartmentalizes banking, payments, and tech — that left no green field. China's super-apps grew in the *absence* of an entrenched card system and under a different regulatory regime, not because Chinese consumers are uniquely fond of bundling.

**"Neobanks are cheaper because they have lower margins."** Not the real reason. Neobanks can offer free accounts because they have a radically lower *cost base* (no branches, no legacy systems) and because they monetize through interchange, float, and lending rather than account fees. The "free" account is a customer-acquisition tool, not charity, and it only pays off if the customer is active.

**"A high valuation means the company is making money."** False, and the single most expensive misconception. Valuation reflects what investors expect a company to be worth in the future; profitability reflects what it earns today. Many fintechs commanded huge valuations while losing money, and the 2022 repricing was the market belatedly insisting on the difference.

## How it shows up in real markets

These are not abstractions; they were front-page financial events. A handful of named episodes anchor the whole story.

**Ant's record IPO, halted (November 2020).** The largest IPO ever, roughly \$37 billion at a \$300 billion-plus valuation, suspended by Chinese regulators days before listing after the founder publicly criticized the regulators. It remains the single starkest demonstration that a fintech's fate can turn entirely on whether the state treats it as a tech platform or a bank.

**PayPal's spin-out from eBay (2015).** eBay had bought PayPal in 2002; by 2015, under pressure from activist investors, it spun PayPal back out as an independent public company. The logic was that PayPal had outgrown its origin as eBay's checkout button and was worth more free to serve the whole web. PayPal's market value as an independent company went on to dwarf eBay's — the payments layer turned out to be worth more than the marketplace that birthed it.

**Stripe's valuation swings.** Stripe became one of the most valuable private companies in the world, peaking at a roughly \$95 billion valuation in early 2021 amid the pandemic e-commerce boom. As markets repriced growth in 2022-2023, a later internal funding round marked it down to around \$50 billion before it recovered toward \$65-70 billion in subsequent rounds (approximate, as of recent reporting). Stripe never went public through any of this, and its swings became a barometer for how the market valued private fintech in general.

**Nubank's Latin-America scale and IPO (2021).** Nubank went public in late 2021 at a valuation around \$45 billion, at one point worth more than Brazil's largest traditional bank — a startling statement that a branchless app a few years old could be worth more than a century-old institution. It has since grown to over 100 million customers and turned profitable, one of the clearest proofs that the neobank model can work at scale in the right market.

**The BNPL boom and bust (2021-2022).** BNPL volumes exploded during the pandemic e-commerce surge; valuations soared. Then rates rose, credit losses ticked up, and the model's thin margins inverted. Klarna's roughly 85% valuation cut (from ~\$46 billion to ~\$6.7 billion) in 2022 was the emblem of the bust. The businesses did not vanish, but they were violently repriced from "tech" to "lender", and the episode permanently changed how investors think about fintech profitability.

**The SVB collapse and the banking-as-a-service plumbing (2023 onward).** When Silicon Valley Bank failed in 2023 and, separately, when fintech-middleware providers connecting apps to partner banks ran into trouble, thousands of fintech customers discovered that "your money is in a neobank" actually meant "your money is at a partner bank you've never heard of, reached through software that just broke." It was a hard, public lesson in how much of fintech still rests on bank plumbing and how that dependency can become a single point of failure.

**The European interchange cap and the neobank squeeze (2015 onward).** When the European Union capped interchange at roughly 0.2% on debit and 0.3% on credit, it dramatically improved life for merchants — but it also quietly kneecapped one of the main revenue mechanisms a neobank relies on. A US neobank earning over 1% of interchange on every swipe has a fundamentally fatter per-customer margin than a European one earning a fifth of that on the same swipe. This single regulatory difference is a large part of why so many European neobanks struggled to reach profitability while serving similar numbers of customers, and why several leaned hard into subscription tiers, foreign-exchange fees, trading, and crypto to make up the gap. It is a clean demonstration that fintech economics are not laws of nature; they are downstream of exactly where a regulator decides to set a fee cap.

![A tree of the four fintech categories — payments, neobanks, BNPL, and super-apps — with example firms as leaves](/imgs/blogs/fintech-disruptors-stripe-paypal-ant-7.png)

The figure above is the taxonomy to take away: four categories — payments, neobanks, BNPL/lending, and super-apps — each attacking a different function the bank once bundled, with the real firms hanging off as leaves. If you can place a new fintech you read about into one of these four buckets and then ask "which of the four money mechanisms does it run on, and is it being treated as tech or as a bank?", you will understand it faster than most of the headlines do.

## When this matters to you, and further reading

This matters to you in three concrete ways. First, as a *consumer*: every "free" fintech product is monetizing you somewhere — through interchange on your swipes, float on your balance, a fee charged to a merchant who quietly prices it into what you pay, or a loan you may not realize you are taking. None of this is sinister, but it is worth seeing clearly, especially with BNPL, where the interest-free framing can mask genuine debt. Second, as a *small merchant or builder*: the difference between a 2.9% processor, a 0.3% European interchange regime, and a 5% BNPL fee is the difference between a viable margin and a dead one, and understanding the plumbing lets you negotiate it. Third, as someone who reads the *financial news*: when you see a fintech valuation swing wildly or a regulator step in, you will now recognize the underlying mechanic — a company being repriced between "tech" and "bank", or a function being dragged from light regulation into heavy.

For further reading, walk outward from here. To understand the institutions fintech is unbundling, see the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions). To understand the deposit-and-lending machinery a neobank is imitating, see [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier). To see another corner of finance where software firms inserted themselves between you and the market, read about [market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading). And to see where the *state* is now building its own answer to private digital payments — the logical endgame of the Ant story — read about [central bank digital currencies](/blog/trading/finance/central-bank-digital-currencies-cbdc).

The throughline is simple and worth holding onto. Banks bundled many financial jobs into one slow, regulated institution. Software let a generation of companies peel those jobs off one at a time and do each one faster — and that worked spectacularly, right up until the moment a peeled-off job grew large enough to *be* banking, at which point the oldest rule in finance reasserted itself: if you take people's money and lend it back out at scale, sooner or later the regulator is going to treat you exactly like the bank you were trying to route around.
