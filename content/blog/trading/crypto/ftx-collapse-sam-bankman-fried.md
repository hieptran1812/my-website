---
title: "The FTX Collapse: How Sam Bankman-Fried Turned Customer Deposits Into the Biggest Crypto Fraud"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A from-scratch walkthrough of how FTX secretly funneled billions of customer dollars to its sister trading firm Alameda, propped the whole thing up with a token it printed itself, and collapsed in nine days when one tweet started a bank run."
tags: ["ftx", "sam-bankman-fried", "alameda-research", "crypto-exchange", "fraud", "commingling", "bank-run", "ftt-token", "case-study", "crypto-regulation", "custody", "effective-altruism"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — FTX looked like the respectable face of crypto until it imploded in days in November 2022, revealing that its sister trading firm Alameda had been secretly funded with billions of dollars of customer deposits — the oldest fraud in finance dressed up in new technology.
>
> - FTX was a crypto exchange, a place where customers deposited money to buy and sell coins; an exchange's first duty is to keep customer money separate from its own and ready to return on demand. FTX broke that duty.
> - Its founder, Sam Bankman-Fried, also owned a trading firm called Alameda Research. Customer deposits were funneled to Alameda, which spent and gambled them — roughly \$8 billion went missing this way.
> - The glue was FTT, a token FTX created out of thin air, marked at billions of dollars, and used as collateral to borrow real money — circular value backing real debt.
> - It unraveled when a leaked balance sheet showed Alameda was stuffed with FTT, a rival announced it would dump its FTT, the token crashed, and customers rushed to withdraw — a classic bank run that found the vault empty.
> - FTX filed for bankruptcy on November 11, 2022; Bankman-Fried was arrested, convicted of fraud in November 2023, and sentenced to 25 years in prison in 2024.
> - The durable lesson: new technology does not invent new frauds. Commingling, self-dealing, and pretend collateral are centuries old — blockchain just gave them a fresh coat of paint and a charismatic spokesman.

In the spring of 2022, FTX was the company crypto pointed to when it wanted to look grown-up. Its founder testified before Congress. Its logo sat on an NBA arena and on the jerseys of a Formula 1 team. Tom Brady, Gisele Bündchen, and Larry David starred in its ads. Sequoia Capital, one of the most respected venture firms in the world, published a glowing profile of its founder. The company was valued at \$32 billion, and its founder was worth, on paper, more than \$20 billion before he turned thirty. He spoke earnestly about giving nearly all of it away.

By the second week of November 2022, all of it was gone. Customers could not withdraw their money. The company filed for bankruptcy. And the missing money — roughly \$8 billion of it — turned out to have been quietly handed to the founder's own trading firm, which had spent it. The man who had been the trustworthy face of an untrustworthy industry was, it emerged, running one of the largest financial frauds in history.

The diagram above is the mental model: a respectable-looking exchange on top, a hidden pipe underneath carrying customer money to an affiliated gambler, and a token the company printed itself holding the whole structure up until one push knocked it over.

![Timeline of the FTX collapse from November 2 to November 11, 2022](/imgs/blogs/ftx-collapse-sam-bankman-fried-1.png)

We are going to build this up from zero. If you have never bought a crypto coin, never read a balance sheet, and could not define "collateral" on the spot, that is completely fine. By the end you will understand what a crypto exchange is and the one duty it owes its customers; what it means to *commingle* funds and why it is forbidden; what a trading firm is and why an exchange owning one is a conflict; what a token is, how a company can issue its own, and why using your own token as collateral is a house of cards; what a *bank run* is and why it is the moment a hidden hole gets exposed. Then we will watch all of it break, day by day, and dissect exactly how the money disappeared.

## Foundations: how a crypto exchange is supposed to work

Before we can see the fraud, we need the honest version clearly in mind. Almost every part of the FTX story is a violation of a rule that exists for a good reason, so let us lay down the rules first.

### What a crypto exchange is

A **crypto exchange** is a business where people buy and sell cryptocurrencies — digital coins like Bitcoin and Ethereum. The largest exchanges, including FTX, Binance, and Coinbase, are **centralized exchanges**: a single company runs them, the way a bank runs your checking account. You sign up, you send the company some money (dollars by bank wire, or crypto from another wallet), and that balance shows up as a number on your account page. When you want to trade, you click buy or sell, and the company updates the numbers.

Here is the crucial subtlety, and it is the hinge of this entire story. When your FTX account said you had "\$10,000," that number was not your money sitting in a labeled box with your name on it. It was a **liability** — a promise. It meant *FTX owes you \$10,000 and will give it back when you ask*. The actual dollars and coins went into FTX's own accounts, pooled together. The whole system runs on trust that the company is holding enough real assets to pay everyone back if they all ask at once. A centralized exchange is, in this respect, very much like a bank. (For a fuller treatment of how these companies are built, see [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase).)

### Custody, segregation, and the one rule that matters

Because your balance is a promise, the single most important thing an exchange can do is **hold the assets that back the promise**. The technical word for holding someone else's assets on their behalf is **custody**. A custodian's job is to keep the assets safe and available — not to invest them, not to lend them, not to spend them.

The rule that makes custody trustworthy is **segregation**: customer assets must be kept *separate* from the company's own money, in their own accounts, untouched. Your \$10,000 is supposed to live in a customer account, ring-fenced, so that even if the exchange's own business goes bankrupt tomorrow, your money is still there waiting for you. A properly run exchange should be able to return every customer's balance at any time, because it never touched that money in the first place.

The opposite of segregation is **commingling**: mixing customer money together with the company's own funds, so that the line between "money we are holding for customers" and "money we can spend" disappears. Once funds are commingled, an exchange can quietly dip into customer deposits to cover its own bets, and no customer can tell, because the account page still shows the same friendly number. Commingling is the original sin of this story. It is illegal for exactly this reason, and it is centuries older than crypto.

### A proprietary trading firm

The second character in our story is a different kind of company entirely: a **proprietary trading firm**, or "prop shop." This is a firm that trades financial assets with its *own* money to make a profit. It is not holding anything for customers; it is a gambler at the table, betting its own chips. Alameda Research, the firm at the center of the FTX fraud, was a crypto prop shop. It made bets on whether coins would rise or fall, ran trading strategies, made loans, and invested in startups.

There is nothing wrong with a prop shop on its own. The problem — the conflict at the heart of FTX — arises when the *same people* own both an exchange (which holds customers' money) and a prop shop (which gambles money). The exchange knows things its customers do not: who is about to buy or sell, where the big orders sit, when a coin is about to be listed. An affiliated prop shop could trade on that. And, far worse, the exchange is sitting on a giant pool of customer money that the prop shop would love to borrow. The wall between the two must be absolute. At FTX, there was no wall.

### A native exchange token

The third character is a thing FTX created: a **token** called **FTT**. A token is a digital unit issued on a blockchain — a coupon or a poker chip a company can print at will. Many exchanges issue their own **native token** and give holders perks: lower trading fees, a share of revenue, voting rights. Binance has BNB; FTX had FTT.

Here is the part to hold onto. A company can decide how many tokens to create. FTX created about 350 million FTT and kept most of them. The *price* of a token is set by trading on the open market — but if a company holds the vast majority of the supply and only a small fraction ever trades, the market price is thin and easy to move. FTX could, in effect, point at the market price of FTT, multiply it by the huge pile it held, and declare that pile worth billions of dollars. We will see why that is a dangerous illusion.

To make that illusion concrete, you need one more idea: the difference between **float** and **market cap**. The *float* is the portion of a token actually available to trade — the coins not locked up by the issuer. The *market cap* (market capitalization) is the price per token multiplied by the *total* supply, including the locked-up portion. When the float is tiny relative to total supply, the market cap is a wildly misleading number, because the price was set by a handful of trades on the small float, but it gets applied to the whole pile as if you could sell it all at that price. You cannot. If you tried to sell the locked pile into the thin market, the price would collapse long before you finished. This gap between "marked value" (price times your pile) and "achievable value" (what you could actually sell for) is where the FTT illusion lived. It is the same trap as a thinly traded penny stock: the screen says your holding is worth a fortune, but the first serious sell order would prove otherwise.

### Collateral, and the trap of circular collateral

**Collateral** is something valuable you pledge to a lender so they will lend to you. If you borrow \$100 and post a \$150 asset as collateral, the lender is protected: if you do not repay, they seize and sell the asset. Collateral is the safety net under nearly all lending. The whole point is that the collateral is *independently valuable* — worth what it is worth regardless of whether you repay.

Now suppose the collateral is a token *you printed yourself*, whose price depends on confidence in *your own business*. That is **circular collateral**, and it is a trap. You issue FTT, you say it is worth \$5 billion, you borrow real cash against it. But the moment people doubt your business, the token's price falls, and the collateral evaporates exactly when the lender needs it most. The value was never independent; it was a reflection of confidence in you, and confidence is the one thing that disappears in a crisis. We will build a worked example of this shortly, because it is the mechanical core of the FTX failure.

### A balance-sheet hole, and a bank run

Two final terms tie it all together.

A **balance-sheet hole** is the gap between what a company owes and what it actually has. If FTX owed customers \$16 billion but held only \$8 billion of assets it could actually sell, there is an \$8 billion hole. A company can operate for years with a hidden hole as long as everyone keeps their money parked and nobody demands it back all at once.

A **bank run** is what happens when they do. If enough customers lose confidence and rush to withdraw simultaneously, even a healthy bank can struggle — because no bank keeps every customer's money in cash; it is lent out or invested. A run is a coordination event: each customer wants to be first in line before the cash runs out, so the rush feeds on itself. The fear of a run can cause a run, which is why runs are so dangerous and so fast.

But there is a critical distinction between a *solvent* institution and an *insolvent* one in a run, and FTX sits firmly on the wrong side of it. A solvent institution facing a run has a **liquidity** problem: it owns enough good assets to cover everyone, but those assets are not all in cash *right this second* — they are in loans or bonds that take time to sell. Given time, or an emergency loan against those good assets, it pays everyone in full. An *insolvent* institution facing a run has a **solvency** problem: even if you gave it all the time in the world to sell everything it owns, it still could not cover what it owes, because the assets are simply worth less than the obligations. A liquidity problem is a timing inconvenience; a solvency problem is a hole.

FTX was insolvent, not merely illiquid. For a fraudulent exchange with a real hole, a run is fatal and *revealing*: it forces an instant test of whether the assets exist, and the answer comes back no. The run did not *cause* FTX's insolvency. The hole was already there. The run simply pulled back the curtain. Keep the liquidity-versus-solvency distinction handy — SBF's defense leaned hard on calling the collapse a liquidity crunch, and seeing through that framing is most of understanding what really happened.

With those building blocks — exchange and custody, segregation versus commingling, the prop-shop conflict, the native token, circular collateral, the hole, and the run — we can now meet the people and watch the machine get built.

## The setup: a golden boy, two companies, and one token

### Sam Bankman-Fried and the halo

Sam Bankman-Fried — universally abbreviated to **SBF** — was a former trader at the quantitative firm Jane Street. In 2017 he founded **Alameda Research**, a crypto trading firm, and made early money on an arbitrage: Bitcoin traded at a higher price in Japan than in the United States, so Alameda bought in the US and sold in Japan, pocketing the difference. In 2019 he founded **FTX**, a crypto exchange, with Alameda as its first big customer and market maker (a market maker is a firm that constantly offers to buy and sell, providing liquidity so that ordinary users can trade easily).

What made FTX extraordinary was not the technology but the *image*. SBF cultivated a persona of rumpled, brilliant sincerity: cargo shorts and a T-shirt at conferences, a beanbag at his desk, video games during investor meetings. He gave long, fluent interviews about market mechanics. And he wrapped the whole enterprise in a moral mission called **effective altruism** — the idea that one should earn as much money as possible in order to give it away to the most effective causes. SBF said he was getting rich only to donate nearly all of it. This was, in retrospect, the most effective part of the marketing: who would suspect a fraud from a man in cargo shorts who promised to give his fortune to charity?

The halo was reinforced from every direction. FTX bought the naming rights to the Miami Heat's arena. It ran a Super Bowl ad with Larry David. It signed Tom Brady, Steph Curry, and other athletes as ambassadors. SBF became one of the largest political donors in the United States, giving heavily to candidates and meeting with regulators, positioning himself as the responsible adult who wanted *sensible* crypto rules. Blue-chip investors — Sequoia, SoftBank, Singapore's Temasek, the Ontario Teachers' Pension Plan — poured in money at a \$32 billion valuation. The respectability was real to everyone looking from outside. It was the disguise.

### Two companies that were never really separate

On paper, FTX and Alameda were distinct companies with distinct jobs: FTX the exchange held customers' money; Alameda the prop shop traded its own. In reality they were run by the same small circle of people, mostly living together in a luxury apartment in the Bahamas, and the money flowed between them with no real wall. Alameda was run by Caroline Ellison, who was at times SBF's romantic partner. The "two companies" framing was the conflict of interest we defined earlier, made concrete: the firm holding the customer money and the firm gambling it were, functionally, one operation.

It is worth being precise about what a *real* corporate separation requires, because FTX failed every part of it. Genuinely separate companies have separate boards that can say no to each other, independent finance teams, arm's-length contracts reviewed by lawyers, and audited books. FTX had none of this in any meaningful form. The bankruptcy filing described an absence of a functioning board, no independent risk function, corporate funds and personal funds intermingled, and decisions made in group chats. There was no minority shareholder to object, no outside director to ask hard questions, no independent auditor of the kind that polices a public company. When the same handful of people control both the entity holding billions in customer money and the entity that wants to borrow it, and no one outside that circle can review or veto the transfers, the legal fiction of two companies is just that — a fiction. The separation that protects customers is not a logo on a different building; it is governance, and FTX had effectively none.

### FTT: the token that held it all up

Then there was FTT. FTX issued FTT in 2019 and used it the way exchanges use native tokens — fee discounts, perks — but it also became the central asset on *Alameda's* balance sheet. Alameda received huge allocations of FTT, marked them at the market price, and treated that paper value as real wealth it could borrow against. Because FTX held most of the FTT supply off the market, the traded price was thin and supportable, and the marked value was enormous — billions of dollars.

This is the circular collateral we defined, and it is worth pausing on because it is the engine of the fraud.

![Graph of the circular structure linking FTX, the FTT token, and Alameda](/imgs/blogs/ftx-collapse-sam-bankman-fried-2.png)

The diagram above traces the loop. FTX mints FTT. FTT is handed to Alameda and marked at billions on Alameda's books. Alameda posts that FTT as collateral to borrow real cash — including, as we will see, customer cash. The borrowed money funds illiquid bets, venture investments, real estate, and political donations. The "asset" anchoring the whole structure is a token FTX printed and whose price FTX's own behavior propped up. Pull confidence out from under it and there is nothing there. Here is exactly how that paper wealth was conjured.

#### Worked example: FTT as circular collateral

Suppose you start a company and print 100 tokens, calling them XYZ. You keep 90 and sell 10 on an exchange. A handful of buyers, excited about your company, bid the price of those 10 tokens up to \$10 each.

Now look at what you can claim. The "market price" is \$10 per token. You hold 90 tokens. So you declare your holding worth 90 x \$10 = \$900. On paper, you just created \$900 of wealth out of nothing but 90 coupons you printed and a thin market on the other 10.

Next, you walk into a lender and say: "I have \$900 of XYZ tokens. Lend me \$400 in real cash, and I will post the tokens as collateral." The lender, seeing a \$900 asset backing a \$400 loan, agrees. You now have \$400 of *real* money — money you can spend on anything — backed by tokens you minted for free.

Here is the trap. The \$10 price held only because few tokens trade and everyone believes in your company. Suppose confidence cracks and the price falls to \$1. Your 90 tokens are now "worth" \$90 — but you owe \$400. And if you tried to actually *sell* 90 tokens into a market that only ever absorbed 10 at a time, the price would collapse toward zero before you sold a fraction of them. The collateral was never \$900. It was a number that existed only as long as nobody tested it.

Scale this up: FTX held hundreds of millions of FTT, marked at a price north of \$20, implying many billions of dollars. Alameda borrowed real money against it. The intuition: collateral you printed yourself is worth a lot exactly until someone needs it to be — and then it is worth almost nothing.

### A balance sheet stuffed with its own tokens

The natural consequence of all this was an Alameda balance sheet that was, in substance, a pile of FTX-linked tokens dressed up as diversified wealth. Its largest assets were FTT, plus other thinly traded tokens that FTX had a hand in — Solana-ecosystem tokens, a token called SRM (Serum), and similar. These were marked at market prices that only existed because the float was tiny. Against this pile of self-referential paper, Alameda had borrowed billions in real liabilities. As long as nobody added up the assets honestly or tried to sell them, the balance sheet "balanced." It was a stick of dynamite waiting for a match. In early November 2022, someone published the balance sheet, and a rival lit the match.

## The blow-up: nine days in November 2022

The collapse was astonishingly fast. A company valued at \$32 billion went to bankruptcy in nine days. Here is the sequence.

### November 2: the leaked balance sheet

On November 2, 2022, the crypto news outlet **CoinDesk** published a report based on a leaked document: Alameda Research's balance sheet. It showed that the bulk of Alameda's roughly \$14.6 billion in assets was FTT and other FTX-associated tokens — about \$5.8 billion of it tied to FTT directly, plus large positions in Solana and other thin tokens. The revelation was simple and damning: the trading firm sitting next to the exchange was not backed by cash or blue-chip holdings. It was backed by the exchange's own coupon.

For anyone who understood circular collateral, this was a flashing red light. If Alameda's "wealth" was mostly FTT, and FTT's price depended on confidence in FTX, then a wobble in confidence could unwind everything at once. The market noticed, but it took a second event to turn a worry into a stampede.

### November 6: CZ pulls the pin

Binance, the largest crypto exchange in the world, was an early investor in FTX and had been paid out partly in FTT — it still held a large position, on the order of \$580 million. On November 6, Binance's founder, **Changpeng Zhao** (known as **CZ**), announced on Twitter that, citing "recent revelations," Binance would sell its entire FTT holding. He framed it as risk management; FTX and Binance were rivals, and the effect was to publicly signal that the most powerful player in crypto had no confidence in FTT.

This was the match. A large, public, motivated seller announcing it will dump a thinly traded token is a near-guarantee the price will fall — and if the price falls, the collateral under Alameda's loans evaporates, and the doubts from the CoinDesk report become a crisis. Alameda's Caroline Ellison publicly offered to buy CZ's FTT at \$22 to defend the price, which only advertised how fragile that price was.

### November 7-8: the token crashes and the run begins

FTT, which had traded around \$22-25, began to fall hard. As it dropped, two things happened at once. The collateral under Alameda's borrowings shrank, deepening the hole. And FTX's customers — now thoroughly alarmed — rushed to pull their money out. This was the bank run.

![Pipeline showing how customer deposits were routed to Alameda](/imgs/blogs/ftx-collapse-sam-bankman-fried-3.png)

The trouble was that the cash to satisfy those withdrawals was not there, because — as the pipeline above shows and as we will dissect in detail — customer deposits had been flowing to Alameda for a long time. Over roughly three days, customers tried to withdraw on the order of \$5 billion. FTX could meet the early withdrawals from whatever liquid funds it had, but the well ran dry fast. Let us put numbers on what a run does to a firm with a hidden hole.

#### Worked example: a bank run when withdrawals exceed liquid assets

Consider a simplified exchange that owes its customers \$16 billion in total. A sound exchange would hold \$16 billion in liquid, sellable assets against that — every dollar of deposits matched by a dollar it can actually return.

Now suppose this exchange secretly lent \$8 billion of those deposits to an affiliated firm, which spent it. The exchange now holds only \$8 billion of liquid assets against \$16 billion of obligations. On a calm day this is invisible: most customers leave their balances parked, so daily withdrawals might be a few hundred million, easily covered.

A run breaks that calm. Suppose customers, panicking, demand \$5 billion back in three days. The first chunk of withdrawals — say the first \$3 billion — gets paid from the liquid assets, and everything looks fine; the account pages still work. But the liquid pile is only \$8 billion and shrinking, while the queue of people wanting out is growing past it. To pay the next \$2 billion, the exchange would need to *sell* its remaining assets — and its remaining assets are mostly the affiliated firm's IOU and a pile of its own illiquid token, neither of which can be sold for cash in a hurry. The exchange freezes withdrawals because it physically cannot pay. The instant it freezes, the \$8 billion hole is no longer hidden; it is the headline.

The intuition: a run does not create the loss — the loss was already there in the missing \$8 billion. The run just forces everyone to find out on the same afternoon.

### November 8-9: the Binance rescue that wasn't

On November 8, FTX halted customer withdrawals — the unmistakable sign of an exchange that cannot pay. The same day, in a stunning reversal, CZ announced that Binance had signed a non-binding **letter of intent** (a preliminary, non-committal agreement) to *acquire* FTX and bail it out. For about a day, it looked like FTX might be rescued.

Then Binance's team looked at the books. On November 9, Binance walked away, citing the size of the hole and reports of mishandled customer funds and pending investigations. There would be no rescue. The non-binding letter of intent had given Binance a look inside, and what it saw was unsurvivable.

### November 11: bankruptcy

On November 11, 2022, FTX, Alameda, and roughly 130 affiliated entities filed for **Chapter 11 bankruptcy** in the United States. SBF resigned as CEO. He was replaced by **John J. Ray III**, the restructuring expert who had overseen the Enron bankruptcy. Ray's first court filing was scathing: "Never in my career have I seen such a complete failure of corporate controls and such a complete absence of trustworthy financial information." There were no proper books, no list of who worked there, no reliable accounting. Corporate funds had been used to buy homes for employees and to fund personal purchases. The respectable face of crypto had no face behind it.

SBF was arrested in the Bahamas in December 2022, extradited to the United States, and tried in New York. In November 2023, a jury convicted him on seven counts of fraud and conspiracy. In March 2024, he was sentenced to **25 years in prison**. Caroline Ellison and other lieutenants pleaded guilty and cooperated.

## The mechanism dissected: exactly how the money disappeared

The timeline tells you *that* FTX collapsed. To understand *why* the money was gone — not merely locked up, but spent — we have to dissect four interlocking mechanical failures. Each on its own is a known fraud. Stacked together, they produced the largest crypto collapse to date.

### Mechanism 1: commingling — customer deposits routed to Alameda

The foundational crime was the one we defined first: commingling. Customer money was not segregated. It was, by multiple routes, available to Alameda.

The most concrete route involved bank accounts. In FTX's early years, before it had its own banking relationships, customers who wanted to wire US dollars to fund their FTX accounts were sometimes told to wire the money to a bank account belonging to **Alameda** — operating under a subsidiary innocuously named **North Dimension**. So a customer's dollars physically landed in Alameda's account. FTX would credit the customer's balance on the platform (an IOU), but the actual cash was now sitting in the trading firm's account, where it could be — and was — spent. Over time this amounted to billions of dollars.

The second, deeper route was a software exemption inside FTX itself. We will come to that under Mechanism 3. The combined effect is captured in the pipeline figure above: customer dollars in at the left, an account credit shown to the customer, and the real money flowing right into Alameda and out into bets and purchases.

#### Worked example: the ~\$8 billion customer-funds hole

Let us trace where the \$8 billion went, in round numbers, to make it concrete. (These are illustrative figures consistent with the bankruptcy estimates; the exact accounting was a mess by design.)

Start with the obligation. FTX owed customers something on the order of \$8 billion in liquid claims it could not meet — that is the figure most often cited for the shortfall, and we will use it as the size of the hole.

Now the uses. Alameda had borrowed and spent customer money across several buckets. It made leveraged trading bets, many of which lost money badly in the 2022 crypto downturn (the collapse of the Terra/Luna stablecoin and the failures of crypto lenders that year — see [Three Arrows Capital and the lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) — blew large holes in Alameda's positions). It made venture investments in startups and other crypto projects, many illiquid. It bought real estate in the Bahamas. It funded political donations and endorsement deals. And it lent money out, including to FTX insiders.

Add it up: a firm that had quietly drawn down billions of dollars of customer money and converted it into a portfolio of losing trades, locked-up investments, property, and spending. When the run came, FTX needed to turn assets back into \$8 billion of cash. But the assets were no longer cash — they were Alameda's scattered, mostly illiquid, partly worthless positions, plus an IOU from Alameda to FTX that Alameda could not honor. The \$8 billion was not in a vault waiting; it had been transformed into things that could not be sold for anywhere near \$8 billion in a hurry.

The intuition: the hole was not an accounting glitch or a temporary liquidity squeeze. The money had genuinely left the building, converted into assets worth a fraction of what was owed.

### Mechanism 2: FTT as self-issued, circular collateral

We already built the circular-collateral worked example; here is its role in the failure. Alameda's ability to borrow — from third parties and from FTX itself — rested heavily on FTT and other FTX-linked tokens being marked at huge values. As long as FTT held its price, the books appeared to "balance": billions in token assets against billions in loans.

This created a fatal feedback loop. FTT's price held up partly *because* FTX and Alameda were healthy and propping it. When confidence cracked on November 6, FTT's price fell, which shrank the collateral, which made Alameda and FTX look insolvent, which cracked confidence further, which drove the price down more. Circular collateral does not just fail to protect the lender; it actively *accelerates* the collapse, because the asset and the borrower fall together. A normal collateral — a government bond, say — holds its value when the borrower gets into trouble. Self-issued collateral does the opposite.

Let us watch the number itself fall.

#### Worked example: FTT falling from ~\$25 to ~\$1

Before the crisis, FTT traded around \$25. Suppose, for round numbers, that FTX and Alameda together held 300 million FTT. At \$25, that pile was marked at 300,000,000 x \$25 = \$7.5 billion. Against assets like that, you can borrow a great deal of real money and call your balance sheet solid.

Now the price falls. When CZ announced Binance would dump its FTT, the price slid from about \$22 toward \$3 within a couple of days, and after the bankruptcy it fell under \$1.50 and kept dropping toward roughly \$1. Re-mark the same 300 million tokens:

- At \$22: 300,000,000 x \$22 = \$6.6 billion.
- At \$3: 300,000,000 x \$3 = \$0.9 billion.
- At \$1: 300,000,000 x \$1 = \$0.3 billion.

In a matter of days, the marked value of that pile fell from \$7.5 billion to a few hundred million — a drop of roughly \$7 billion in "wealth" that had never been real, because you could never have sold 300 million tokens into that thin market without crushing the price to near zero anyway. The loans that the \$7.5 billion supposedly secured were still owed in full.

The intuition: marking a huge, thinly traded token holding at the last small-trade price is a fiction that holds only in calm markets; the moment you need the value, it is gone, and the debt remains.

### Mechanism 3: the absence of real custody and the secret backdoor

A properly run exchange cannot lend customer money to an affiliate even if it wants to, because the customer assets are segregated and the systems enforce it. FTX had the opposite. Investigators found that FTX's software contained a special exemption for Alameda: where every other account had its borrowing capped by its collateral and would be automatically liquidated if it ran too negative, Alameda's account was exempted from those limits. It could draw a virtually unlimited negative balance — effectively borrowing from the pool of customer funds — without triggering the protections that applied to everyone else.

This is the deepest violation. It means the commingling was not an accident or a one-time emergency; it was *engineered into the platform*. The "custody" FTX offered was a fiction. There was no segregation in the code. Customer assets and the affiliated gambler's credit line were wired into the same system, with the gambler given a key to the vault.

#### Worked example: the conflict of an affiliated market maker trading against customers

Set aside the borrowing for a moment and consider the plain conflict of an exchange owning the biggest trader on its own platform.

Suppose a customer places a large order to buy a coin at the market price. The exchange's matching engine sees this order a fraction of a second before anyone else. An affiliated trading firm with a privileged connection — Alameda was FTX's main market maker and had unusually deep access — could, in principle, buy that coin first, let the customer's order push the price up, and sell into it. The customer pays a worse price; the affiliate pockets the difference. This is the textbook abuse that the wall between exchange and trader is meant to prevent.

Put numbers on it. A customer wants to buy 1,000 units of a coin trading at \$100, expecting to pay about \$100,000. An affiliate that front-runs the order buys first, nudging the price to \$101 by the time the customer fills, so the customer pays \$101,000 — \$1,000 worse. Repeated across thousands of orders a day, a privileged affiliated trader can extract a steady, invisible tax from the very customers the exchange is supposed to serve.

Whether or not FTX systematically did this is harder to prove than the commingling, but the *structure* made it possible, and that is the point: an exchange should never own the largest trader on its own venue, because the exchange holds information and infrastructure that the trader can exploit against the people whose money it is also holding. The intuition: when the house also plays at the table and can see everyone's cards, the players cannot win a fair game.

### Putting the four mechanisms together: the gap between claim and reality

Stack the four mechanisms and you get the difference between the balance sheet FTX implied and the one it actually had.

![Before-after comparison of FTX's claimed assets versus its real assets](/imgs/blogs/ftx-collapse-sam-bankman-fried-4.png)

On the left is what customers and the public reasonably assumed: customer dollars held one-to-one in safe, liquid reserves, with Alameda a fully separate firm gambling its own money. On the right is what was actually there when the auditors of the bankruptcy looked: the "reserves" were largely FTT and other self-issued, illiquid tokens; a giant receivable from Alameda that Alameda could never repay; and customer cash that had already been spent. The two pictures could not be reconciled, and the gap was the hole. Let us look at the hole directly.

![Stack diagram of the roughly eight billion dollar balance-sheet hole](/imgs/blogs/ftx-collapse-sam-bankman-fried-5.png)

The stack above is the anatomy of the shortfall: at the top, the real dollars owed to customers; below it, the "backing" that was supposed to cover those dollars — an uncollectible Alameda IOU and a pile of illiquid FTX-linked tokens; and at the bottom, the net shortfall, the simple fact that the cash had already been spent and could not be conjured back. A hole is not a complicated thing once you stop pretending the IOU and the tokens are worth their marked values. It is just: owed more than you have.

## The aftermath: recoveries, a credibility crater, and a regulatory push

### What happened to the people

The legal aftermath was swift by the standards of financial fraud. SBF was convicted in November 2023 and sentenced to 25 years in prison in March 2024, along with an order to forfeit billions. Caroline Ellison (Alameda's CEO), Gary Wang (FTX's co-founder and chief technology officer, who built the backdoor), and Nishad Singh (FTX's director of engineering) all pleaded guilty and testified against him. Ryan Salame, another executive, pleaded guilty to campaign-finance and money-transmitting charges. The inner circle was dismantled.

### What happened to the money

Here the story takes a strange turn that is widely misunderstood. The FTX bankruptcy estate, under John Ray, spent the next two years clawing back assets, selling Alameda's venture investments, and pursuing recoveries. Crucially, some of those venture bets — most famously a large stake in the AI company Anthropic — *rose* enormously in value after the collapse, and the price of Bitcoin and other crypto recovered from the 2022 lows.

The result: by 2024, the estate announced that it expected to repay essentially all customer claims in full, plus interest, in dollars. This sounds like a happy ending and is often reported as one, but it deserves a sharp caveat. Customers are being repaid in *dollars* based on the *November 2022 value* of their holdings — not their crypto back. Someone who held one Bitcoin worth roughly \$16,000 in November 2022 gets paid the dollar value of that, even though Bitcoin later traded far higher; they do not get their Bitcoin, which would have been worth much more. And the "full recovery" depended on luck — a few venture bets and a crypto rally — not on the assets having been there. Had Anthropic flopped and crypto stayed depressed, the hole would have stayed a hole. A fraud does not become not-a-fraud because the receiver got lucky cleaning up after it.

### What happened to crypto's credibility

The reputational damage was severe and broad. FTX had been the industry's respectable ambassador; its fraud handed every crypto skeptic their strongest argument. The collapse also spread contagion: firms and funds that had money on FTX or were owed money by Alameda took losses, deepening the "crypto winter" of 2022-2023 that had already been triggered by the Terra/Luna and Three Arrows collapses earlier that year.

### What happened to the rules

FTX accelerated a regulatory push that was already building. Regulators and lawmakers, embarrassed that the loudest voice for "sensible regulation" had been running a fraud, focused on the obvious lessons: customer assets at exchanges must be genuinely segregated and provably held; affiliated trading arms create unacceptable conflicts; and "trust me" is not an auditing standard. The episode strengthened the case for **proof of reserves**, for rules separating custody from trading, and for treating large crypto exchanges with the same custody and conflict standards as regulated financial institutions.

Proof of reserves deserves a word, because it is the direct technical answer to the FTX failure and it has real limits. The idea is that an exchange cryptographically demonstrates two things: that it controls wallets holding at least as much crypto as it owes customers (the *assets* side), and the total of what it owes customers (the *liabilities* side, often proven with a cryptographic structure called a Merkle tree that lets each customer verify their balance is included without revealing everyone's balances). If assets are greater than or equal to liabilities, the exchange is at least solvent in what it can show. The catch is that proof of reserves does not, by itself, prove there are no *hidden* liabilities — an exchange could borrow coins to pass the check on the day of the snapshot, or owe money off-chain that the proof does not capture. It is a meaningful improvement over "trust me," but it is a floor, not a guarantee; a full picture still needs an independent audit of both sides. Implementation varies by country and is still evolving, but the direction of travel — exchanges must *prove* segregation rather than promise it — is the durable regulatory legacy of FTX.

## Common misconceptions

### "It was a sophisticated, novel crypto crime that no one could have caught."

It was the opposite of novel. Strip away the tokens and the blockchain and you are left with three of the oldest frauds in finance: taking customer money you were supposed to safeguard (commingling and misappropriation), lending it to yourself through an affiliate (self-dealing), and pretending worthless paper is valuable collateral. The technology was new; the crime was ancient. And it *was* catchable — the leaked balance sheet showed the FTT concentration in plain numbers, and skeptics had been warning about the FTX-Alameda relationship for a while.

### "FTX failed because of a bank run / because CZ attacked it."

The run and CZ's tweet were the *trigger*, not the *cause*. A solvent exchange survives a run, because it can return everyone's money — that is the whole point of segregation. FTX could not survive the run because the money was already gone; roughly \$8 billion of customer funds had been spent before November 2022. Blaming the run is like blaming the smoke alarm for the fire. CZ's announcement lit the match, but the building was already soaked in gasoline.

### "Everyone got their money back, so it wasn't really fraud / no one was hurt."

Two errors here. First, the recovery is in November-2022 dollars, not the customers' actual crypto, so holders of assets that later rose were materially harmed even if "made whole" on paper. Second, the recovery happened *despite* the fraud, through windfall gains on a few investments and a crypto rebound — not because the assets existed. Fraud is defined by the conduct (deceiving people and misusing their money), not by whether a receiver later got lucky. A jury convicted on the conduct.

### "SBF was a well-meaning genius who made risky bets that went wrong."

The "I made mistakes but didn't commit fraud" defense was exactly the argument the trial rejected. The evidence — the software backdoor exempting Alameda from borrowing limits, customer funds routed to Alameda's bank accounts, lieutenants testifying to knowing misappropriation — described intentional deception, not honest risk-taking gone bad. Risk that loses money is unfortunate; spending customers' money without telling them is a crime. The line between them is consent, and customers never consented to funding Alameda.

### "Effective altruism caused the fraud."

The charitable mission was *cover*, not cause. The instinct it exploited is a real and important one: we lower our guard around people who appear to be sacrificing for the good of others. But a stated intention to give money away is not a control, not an audit, and not a wall between an exchange and its affiliated trading firm. Plenty of effective-altruism-minded people run honest organizations. The lesson is about *verification*, not philosophy: a halo — charitable, celebrity, political, or technological — is a reason to look harder, not a reason to look away.

### "Crypto's transparency / the blockchain should have prevented this."

A common belief is that because crypto is "on-chain" and transparent, a fraud like this is impossible. But FTX was a *centralized* exchange: customer balances lived in FTX's private internal database, not on a public blockchain. What FTX did with the pooled funds — wire them to Alameda, exempt Alameda from limits — happened in private bank accounts and private software, invisible to any blockchain. Centralized custody reintroduces exactly the trust problem that decentralized systems were meant to remove. The blockchain was never watching the part that mattered.

## How it echoes in other markets

FTX feels unprecedented only if you have not read financial history. The same wiring appears again and again.

### Bernie Madoff (2008)

The closest parallel by *structure* is Bernie Madoff's Ponzi scheme. Madoff, like SBF, was a trusted, respected industry figure — a former NASDAQ chairman. His firm both managed clients' money *and* ran its own broker-dealer that supposedly executed the trades, with no independent check, the same fusion of roles that should never coexist. And the assets behind the account statements simply did not exist; clients were shown fabricated returns. When redemptions surged in the 2008 crisis — a run — the absence of real assets was exposed. FTX is Madoff with a token and an exchange instead of fictional stock trades. For the full mechanics of how that one worked, see [the Madoff Ponzi scheme](/blog/trading/finance/madoff-ponzi-scheme).

### Enron (2001)

Enron is the closest parallel by *self-dealing and pretend value*. Enron used affiliated entities — special purpose vehicles run by its own CFO — to move debt off its books and book fake profits, and it marked illiquid, hard-to-value contracts at optimistic estimates to manufacture earnings. Replace "special purpose vehicle run by the CFO" with "Alameda run by SBF's circle," and "mark-to-market on a 20-year energy contract" with "mark a self-issued token at its thin-market price," and the playbook is the same: use a captive affiliate and unverifiable valuations to make a fragile enterprise look solid. It is no accident that the same restructuring expert, John Ray III, cleaned up both. See [the Enron accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud) for how marking the unmarkable works.

### Mt. Gox (2014)

The original crypto-exchange catastrophe. Mt. Gox once handled most of the world's Bitcoin trading and then collapsed, losing hundreds of thousands of Bitcoins belonging to customers — a mix of hacking and chaotic, untrustworthy custody. The lesson FTX failed to learn was already a decade old: a centralized exchange that does not provably segregate and safeguard customer assets is a single point of catastrophic failure, no matter how big it looks.

### Celsius and the crypto lenders (2022)

In the same brutal year, the crypto lender Celsius (and others like BlockFi and Voyager) froze customer withdrawals and went bankrupt. The pattern rhymes with FTX: companies that took in customer crypto promising safety and yield, then quietly deployed it into risky, illiquid bets, and could not return it when customers asked. The specific frauds differ, but the shared sin is taking custody of customer money and then treating it as the firm's own to gamble with — commingling under another name.

### Three Arrows Capital and the contagion of 2022

Three Arrows Capital was a giant crypto hedge fund that borrowed heavily and blew up when the Terra/Luna stablecoin collapsed in May 2022, taking down the lenders that had financed it. It is part of the FTX story directly: the 2022 contagion blew holes in Alameda's positions, which is part of why Alameda needed to keep dipping into FTX customer funds. The broader echo is leverage plus illiquid collateral plus interlocking exposures — the same ingredients, a different blast radius. See [Three Arrows Capital and the lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion).

### The "it was just risk management" framing

Watch for the rhetorical move SBF and his defenders used, because it recurs across blowups: recasting fraud as a *risk-management failure*. "We took on too much risk." "Our controls weren't mature." "It was a liquidity crunch, not insolvency." This framing is seductive because risk-taking is legitimate and failures of risk management are unfortunate-but-not-criminal. The tell is *whose money* was at risk and *whether they knew*. Risking your own capital and losing is business. Risking customers' capital that you told them was safe, without their knowledge, is theft dressed as a spreadsheet. Whenever a collapse is explained purely in the language of "risk," ask: risk to whom, with whose money, disclosed to whom?

The full pattern is easiest to see side by side.

![Matrix comparing FTX's red flags with the Enron and Madoff frauds](/imgs/blogs/ftx-collapse-sam-bankman-fried-6.png)

The matrix above lines up the three structural red flags — an affiliated insider entwined with the main entity, the absence of a real independent auditor, and assets whose value cannot be independently verified — across FTX, Enron, and Madoff. In every column, all three boxes are filled. That is the signature of this family of fraud, and it is visible long before the collapse if anyone insists on checking rather than trusting.

## The picture that ties it together: the run as the moment of truth

We have covered the wiring; one last figure shows why the end came so fast once it started.

![Graph of the bank run that exposed the FTX balance-sheet hole](/imgs/blogs/ftx-collapse-sam-bankman-fried-7.png)

The graph above traces the chain reaction. Two forces fed the panic at once: FTT's price collapsing from around \$22 toward \$3, and CZ's public warning that Binance would dump its holding. Both drove customers to fear insolvency, which drove the withdrawal rush of roughly \$5 billion in days, which drained the liquid assets, which forced FTX to freeze withdrawals — and the freeze *was* the public confession that it was insolvent. The structure took years to build and nine days to expose. That is the nature of a hidden hole: it is patient, and then it is sudden.

## When this matters to you, and further reading

You will probably never run a crypto exchange or audit one. So why does this matter beyond a dramatic story?

First, because the structure of the FTX fraud is a template you can now recognize anywhere money is held on your behalf. Whenever you hand assets to an intermediary — an exchange, a broker, a lender, a "high-yield" platform — there is exactly one question that matters more than the brand, the celebrity endorsements, the regulatory testimony, or the charitable mission: *are my assets segregated and provably there, or am I trusting a promise?* If the answer is "trust me," you are exposed to every mechanism in this post. The cargo shorts and the giving pledge were not evidence; they were the disguise.

Second, because the FTX collapse is a clean demonstration that **technology does not abolish the need for trust and verification — it relocates it.** Crypto promised to remove trusted intermediaries. But the moment users handed their coins to a centralized exchange for convenience, they re-created the exact trust relationship — and the exact vulnerability — that the technology was supposed to eliminate. The lesson is not "crypto is a scam" or "crypto is the future." It is narrower and more useful: a centralized custodian is a centralized custodian, whether it holds dollars or Bitcoin, and the old rules about segregation, conflicts of interest, and independent verification apply with full force.

Third, because the red flags are legible in advance if you look. A token a company issued itself, marked at billions and used as collateral. An affiliated trading firm with privileged access. An absence of a serious, independent auditor. A balance sheet that is one asset class deep. Any one of these is a warning; all of them together, as in the matrix above, is the signature of the family of fraud that runs from Enron through Madoff to FTX.

To go deeper, three companion pieces build out the surrounding terrain. For how centralized exchanges are supposed to work — and the proof-of-reserves practices that FTX's collapse pushed into the mainstream — see [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase). For the prior-generation fraud whose structure FTX most resembles, see [the Madoff Ponzi scheme](/blog/trading/finance/madoff-ponzi-scheme). And for the leverage-and-contagion blowups that battered Alameda's balance sheet on the way down, see [Three Arrows Capital and the crypto lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion).

The single sentence to carry away: FTX was not undone by a clever new crypto exploit. It was undone by the oldest move in finance — spending the money you were trusted to keep — and the only thing the blockchain added was a fresh stage and a more convincing actor.
