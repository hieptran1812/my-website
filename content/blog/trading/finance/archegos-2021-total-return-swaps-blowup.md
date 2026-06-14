---
title: "Archegos: The Family Office That Vaporized 10 Billion Dollars of Bank Capital"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How one secretive family office used total-return swaps spread across several banks to build a giant hidden bet that, when it cracked, cost the banks about ten billion dollars."
tags: ["archegos", "total-return-swaps", "prime-brokerage", "leverage", "credit-suisse", "bill-hwang", "counterparty-risk", "fire-sale", "case-study", "risk-management"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Archegos used total-return swaps spread across several prime brokers to build a giant, hidden, concentrated, leveraged bet that no single bank could fully see, so when it turned, the brokers' panicked race to liquidate cost them about \$10 billion.
>
> - A family office run by Bill Hwang turned roughly \$10 billion of capital into more than \$50 billion of stock exposure, mostly through derivatives, and almost nobody outside the firm knew.
> - The core mechanism: a total-return swap lets you control a stock without owning it, so the position never shows up in public filings, and by splitting the swaps across five or six banks Hwang made sure no counterparty saw the whole picture.
> - The scale: when the bet unwound in late March 2021, banks lost roughly \$10 billion in a single week, with Credit Suisse alone losing about \$5.5 billion and Nomura about \$2.9 billion.
> - Who paid: the banks ate the losses, several risk officers were fired, Credit Suisse's reputation took a wound it never recovered from, and Bill Hwang was convicted of fraud and market manipulation in 2024.
> - The durable lesson: leverage you cannot see is the most dangerous kind, and a market built so that no single lender knows the total size of a borrower's bet will, eventually, be surprised by the size of a borrower's bet.

On the morning of Friday, March 26, 2021, traders at Goldman Sachs and Morgan Stanley began quietly selling enormous blocks of a handful of stocks. ViacomCBS. Discovery. A clutch of Chinese internet names. Tens of billions of dollars of shares hit the market over two days, far more than normal trading could absorb, and the prices of those stocks fell off a cliff. By the time the dust settled the following week, a fund almost nobody outside Wall Street had ever heard of had ceased to exist, and several of the world's largest banks were tallying losses that ran into the billions. Credit Suisse alone would book a loss of about \$5.5 billion. The total across all the banks was somewhere near \$10 billion.

The fund was Archegos Capital Management. It was not a hedge fund in the ordinary sense; it was a family office, which means it managed only the personal money of one man, Bill Hwang. It had no outside clients to answer to. It filed almost nothing with regulators. And in the space of a year it had built one of the largest concentrated equity positions in the world while remaining, to the broader market, essentially invisible. The diagram above is the mental model: a position assembled over many months, then forced out in a single catastrophic week once one of its stocks cracked.

![Timeline of the late March 2021 Archegos unwind](/imgs/blogs/archegos-2021-total-return-swaps-blowup-1.png)

What makes Archegos such a clean and instructive disaster is that none of the pieces were exotic. There was no clever new product, no fraud against investors in the usual sense, no Ponzi scheme paying old investors with new money. There was a perfectly ordinary derivative called a total-return swap, a perfectly ordinary service called prime brokerage, and a perfectly ordinary appetite for leverage. The danger was entirely in the arrangement: the way these ordinary pieces were combined so that the true size of the bet was hidden from everyone who might have stopped it. This is a story about visibility. It is about how a market can be full of sophisticated, well-paid risk managers and still fail to see a \$50 billion position forming right in front of them, because the system was built so that each of them only ever saw a slice.

By the end of this article you will understand exactly how a total-return swap works, why it hides ownership, why splitting one across many banks hides leverage, and why, once the music stopped, the rational move for each bank was the very thing that turned a bad situation into a catastrophe. We will build every concept from zero, ground each one in dollar arithmetic, and then watch the whole machine break in slow motion.

## Foundations: the building blocks of a hidden bet

Before we can dissect what broke, we need a shared vocabulary. Every term in this section is something the rest of the story turns on. If you already know what a total-return swap is, skim; if you do not, read carefully, because the entire disaster lives in the gap between what these instruments appear to do and what they actually do.

### A family office, and why nobody watches it

A **family office** is a private firm that manages the wealth of a single family or individual. The Walton family (Walmart), the Koch brothers, Michael Dell, and George Soros all run family offices. The defining legal feature is that a family office has no outside investors. It is not managing your retirement money or a pension fund's money; it is managing only the principals' own money.

This matters enormously, because most of the financial regulation that constrains a normal investment manager exists to protect *outside* investors. A hedge fund that takes money from pensions and endowments must register with the Securities and Exchange Commission (the SEC, the main US securities regulator), file disclosures, and submit to examinations. A family office, because it has no clients to protect, is largely exempt. After the 2008 crisis, the Dodd-Frank Act tightened the rules on investment advisers, but it carved out an explicit exemption for "family offices." So a firm like Archegos could manage tens of billions of dollars and yet face almost none of the disclosure and oversight a public mutual fund faces. The regulators' implicit assumption was: if you are only risking your own money, you are only hurting yourself.

That assumption turns out to be wrong the moment you add leverage borrowed from someone else. When Archegos blew up, Bill Hwang lost his own fortune, yes. But the people who lost the most were the banks that had lent him the firepower, and through them, their shareholders and ultimately the wider market.

### Leverage: borrowing to make a bet bigger

**Leverage** is the use of borrowed money to increase the size of a position relative to your own capital. If you have \$10 and you buy \$10 of stock, you are unleveraged. If you have \$10, borrow another \$40, and buy \$50 of stock, you have 5x leverage: your exposure is five times your equity.

Leverage is a magnifier in both directions. A 10% rise in a \$50 position is a \$5 gain, which on \$10 of your own equity is a 50% return. Wonderful. But a 10% fall is a \$5 loss, which is a 50% loss of your equity. And a 20% fall wipes you out entirely, because \$10 of losses against \$10 of equity leaves nothing. The higher the leverage, the smaller the price move needed to destroy you. This single fact is the gravitational center of the entire Archegos story, and we will return to it with exact numbers.

### A prime broker: the bank that arms the fund

A **prime broker** is a bank that provides the plumbing a fund needs to trade at scale. The prime brokerage divisions of Goldman Sachs, Morgan Stanley, Credit Suisse, Nomura, and UBS do several things at once: they lend money and securities to the fund, they hold the fund's assets, they clear and settle its trades, and crucially, they are the counterparty on its derivatives. Prime brokerage is a lucrative, competitive business. Banks fight for the biggest, most active clients because those clients pay fat fees in financing charges and commissions.

That competition matters. A fund like Archegos was a prize client, and the banks that served it had a strong incentive to keep it happy, extend it generous terms, and not ask too many awkward questions. Hold that thought.

#### Worked example: why the banks loved this client

The financing economics explain why every prime broker fought for Archegos despite the risk.

- Suppose a bank fronts \$8 billion of the \$10 billion of stock under its swaps with Archegos (Archegos posts \$2 billion of margin).
- The bank charges a financing spread on the money it fronts — say 1.5% a year on \$8 billion.
- That is \$120 million a year in financing revenue from one client, plus execution commissions on every trade the bank does to build and adjust the position.
- Across the desk, a single relationship throwing off \$100-200 million a year is a star account, and the banker running it is rewarded for growing it, not shrinking it.

The intuition: the fees were large, certain, and immediate, while the tail risk was small-looking, uncertain, and deferred — exactly the asymmetry that makes institutions lean toward more exposure, not less.

### A total-return swap: owning the gains without owning the shares

Now the central instrument. A **total-return swap** (TRS) — closely related to a **contract for difference** (CFD) — is a derivative contract between you and a bank in which the bank agrees to pay you the entire return of some underlying asset (say, a stock), and you agree to pay the bank a financing fee plus any losses. "Total return" means everything: price appreciation and dividends.

Here is the key move. *The bank, not you, buys and holds the actual shares.* The bank is the legal owner of record. You never appear on the share register. You have simply contracted to receive whatever economic outcome the shares produce. If the stock goes up \$10, the bank pays you \$10. If it goes down \$10, you pay the bank \$10. Economically you are fully exposed to the stock — you have all the upside and all the downside — but legally you own nothing.

Why would anyone do this rather than just buying the stock? Three reasons, and all three matter here. First, **leverage**: you only post a fraction of the value as collateral (the margin), so a small amount of cash controls a large position. Second, **financing convenience**: the bank handles the buying, holding, and borrowing. Third, and most importantly for our story, **invisibility**: because you do not own the shares, you do not have to tell anyone you control them.

![Pipeline of how a total-return swap works](/imgs/blogs/archegos-2021-total-return-swaps-blowup-3.png)

#### Worked example: one total-return swap, dollar by dollar

Let me make the swap concrete. Suppose you want exposure to \$100 of a stock and your bank requires 20% margin on this swap.

- You post \$20 of collateral (the margin) to the bank.
- The bank goes into the market and buys \$100 of the stock. The bank pays the full \$100; \$20 came from your margin, \$80 is effectively the bank's money at risk.
- The bank holds the \$100 of shares on its own books. On the public share register, the bank is the owner.
- The swap contract says: the bank pays you the price return of that \$100 of stock, and you pay the bank a financing fee on the \$80 it fronted.
- If the stock rises 10% to \$110, the bank's shares are worth \$110; under the swap the bank owes you the \$10 gain. Your \$20 of margin has earned \$10, a 50% return before fees.
- If the stock falls 10% to \$90, you owe the bank \$10. Your \$20 margin is now effectively \$10. The bank will ask you to top it back up.

The intuition: you put up \$20 and you are riding \$100 of stock, while your name appears nowhere. You have 5x leverage and total invisibility, in one contract.

### Margin and the margin call

The **margin** is the collateral you post to back the swap. The bank sets it to cover the loss it might suffer if the position moves against you before it can close out. A **margin call** is the demand the bank makes when your collateral falls below the required level: "Post more cash, now, or we close your position." Margin calls are the trigger mechanism of nearly every leveraged blow-up. As long as you can meet them, you survive. The day you cannot, the bank seizes the collateral and liquidates — and that liquidation is what turns your problem into the market's problem.

### Why a swap hides ownership: the disclosure gap

Here is the legal machinery that made Archegos invisible. US securities law requires public disclosure of large *ownership* stakes:

- **Schedule 13D / 13G**: anyone who acquires beneficial ownership of more than 5% of a public company's voting shares must file with the SEC within days, naming themselves and the size of their stake.
- **Form 13F**: large institutional investment managers must file quarterly a list of the US equities they hold.

Both rules hang on the word *ownership*. They are about who owns the shares. But under a total-return swap, *the bank* owns the shares. Archegos owned a contract referencing the shares. The prevailing interpretation at the time was that swap exposure did not trigger 13D ownership disclosure. So Archegos could have the full economic exposure of a 10%, 15%, even 20% stake in a company and never file a single ownership report. The market — other investors, the companies themselves, index providers — had no idea a giant was standing behind the stock.

> [!note]
> This is the single most important sentence in the article. Owning the shares would have forced disclosure; owning a swap on the shares did not. The swap was, functionally, a cloaking device. Everything else — the leverage, the concentration, the eventual fire sale — was built on top of that cloak.

### Position concentration

**Concentration** simply means how much of your money is in how few bets. A diversified portfolio spreads risk across hundreds of names; a concentrated one piles it into a handful. Concentration multiplies the danger of leverage, because the thing that could wipe you out is now a move in one or two stocks, not the average of a thousand. Archegos was both extremely leveraged *and* extremely concentrated — perhaps a dozen names — which is the worst possible combination. We will quantify this.

### Counterparty risk

**Counterparty risk** is the risk that the other side of your contract fails to pay what they owe. When a bank writes a total-return swap to a fund, the bank is exposed to the fund: if the stock falls and the fund owes the bank money, the bank is counting on the fund to pay. If the fund cannot, the bank is left holding the actual shares it bought to hedge, and it must sell them — possibly at a loss, possibly into a falling market. Counterparty risk is the channel through which Archegos's losses became the banks' losses.

### A fire sale

A **fire sale** is the forced sale of an asset at a deep discount because the seller has no choice and no time. When a bank seizes a defaulted client's collateral and must dump it fast, it cannot wait for a good price; it sells into whatever bid exists. If several banks are doing this at once, in the same few stocks, the selling pressure crushes the price, which makes the remaining holdings worth even less — a feedback loop. The Archegos liquidation was one of the most dramatic fire sales in modern equity-market history.

With these building blocks in hand — family office, leverage, prime broker, total-return swap, margin, the disclosure gap, concentration, counterparty risk, and fire sale — we can now assemble the machine and watch it run.

## The setup: how Bill Hwang built a hidden colossus

To understand Archegos you have to understand the man, because Archegos was, almost literally, one man's conviction expressed at enormous scale.

### Bill Hwang and the Tiger lineage

Sung Kook "Bill" Hwang was a **Tiger cub** — one of the protégés of Julian Robertson, the legendary investor whose Tiger Management spawned a whole generation of fund managers. Hwang launched his own fund, **Tiger Asia Management**, in 2001, focusing on Asian equities. For a while it was very successful, growing to several billion dollars.

But there is a crucial blemish in this history. In 2012, Tiger Asia pleaded guilty to wire fraud in connection with insider trading in Chinese bank stocks and settled related civil charges with the SEC. Hwang was barred from managing public money and paid tens of millions in penalties. This is not a minor footnote. It means that when Hwang re-emerged, the regulatory and reputational record on him was not clean. He could no longer easily run a hedge fund taking outside money. So he converted his operation into a **family office** — managing only his own capital — and named it **Archegos**, a Greek word meaning "leader" or "author" (it appears in the New Testament referring to Christ; Hwang was devoutly religious). The family-office structure was, in part, a way around his disqualification from managing other people's money. It also meant he answered to no one and disclosed almost nothing.

### From a few billion to a hidden fifty

Starting with somewhere around \$200 million to \$500 million of his own remaining wealth after the Tiger Asia penalties, Hwang ran an extraordinarily aggressive, concentrated strategy. By 2020-2021, riding the post-pandemic bull market and his own large, self-reinforcing bets, Archegos's capital had grown to roughly **\$10 billion**. That alone would have made it a sizeable fund. But \$10 billion of equity was only the visible tip.

Through total-return swaps written by a roster of prime brokers — Credit Suisse, Nomura, Morgan Stanley, Goldman Sachs, UBS, Deutsche Bank, Mitsubishi UFJ and others — Hwang controlled total equity exposure estimated at **\$50 billion or more**, perhaps as high as \$100 billion at gross notional including both swaps and some directly held positions. The exposure was piled into a remarkably short list of names: **ViacomCBS** (now Paramount), **Discovery**, and several Chinese ADRs (American Depositary Receipts — shares of Chinese companies that trade on US exchanges) such as **Baidu**, **Tencent Music**, **GSX Techedu**, **Vipshop**, and **Farfetch**.

![Before-after of Archegos equity versus swap exposure](/imgs/blogs/archegos-2021-total-return-swaps-blowup-4.png)

#### Worked example: turning \$10 billion into \$50 billion of exposure

Let me show the leverage arithmetic at the level of the whole firm.

- Archegos equity: approximately \$10 billion.
- Target leverage: roughly 5x.
- Total exposure: \$10 billion x 5 = \$50 billion.

To get \$50 billion of exposure from \$10 billion of equity, Archegos needed the banks to front the other \$40 billion. It did this not by borrowing \$40 billion in cash (which would have looked alarming) but by entering swaps. On a swap requiring, say, 15-20% margin, \$10 billion of margin supports \$50-66 billion of notional exposure. The bank fronts the rest because it holds the shares as its hedge and holds your margin as protection.

The intuition: with 5x leverage, a 20% move in the underlying stocks is a 100% move in your equity — it either doubles you or destroys you. Archegos chose to live on that knife's edge across a dozen highly correlated, volatile names.

### Why no single bank saw the danger

This is the part that still astonishes practitioners. Each prime broker knew it had written swaps to Archegos. Each knew, roughly, how big *its own* book with Archegos was. Credit Suisse's exposure to Archegos was on the order of \$10 billion of underlying. Morgan Stanley's was similar. Nomura's was several billion. But **no single bank knew the others' exposures.** There was no shared registry, no aggregator, no public filing, because the swaps did not trigger disclosure.

![Graph of one position split across five brokers](/imgs/blogs/archegos-2021-total-return-swaps-blowup-2.png)

So each bank, looking at its own slice, saw a large but seemingly manageable relationship with a wealthy, sophisticated client who paid well and had a strong recent track record. What each bank could not see was that five or six other banks were doing the exact same thing on the exact same stocks. The total leverage and the total concentration were emergent properties of the whole system, invisible from any single vantage point. It was a coordination failure dressed up as competition: every bank competing for Archegos's business, none of them comparing notes.

### Why the bet looked smart

It is easy, after the fact, to call this reckless. But for most of 2020 and early 2021, it looked like genius. The stocks Hwang chose surged. ViacomCBS more than tripled from late 2020 into March 2021, partly on its own merits (a streaming-service relaunch) and partly — this is important — because Archegos's own relentless buying through the swaps was pushing it up. The Chinese ADRs ran hard too. Archegos's returns were spectacular, its capital compounding, its banks delighted. A leveraged, concentrated bet that is going up looks like conviction rewarded. Right until it does not.

## The blow-up, step by step

Now the machine runs in reverse. The chronology is fast — the core of it spans about five trading days in late March 2021 — but each step is a clean illustration of the mechanisms we built earlier.

### Monday-Tuesday: the trigger

On Monday, March 22, ViacomCBS, which had been on a tear, announced a \$3 billion stock offering — it wanted to sell new shares to raise cash. New share issuance dilutes existing holders, and the stock was already looking stretched after tripling. The market reaction was negative. ViacomCBS shares, which had peaked around \$100, began to slide. Over Tuesday and Wednesday the slide accelerated; by midweek the stock was down roughly 30% from its peak.

For a normal, unleveraged investor, a 30% drop is painful but survivable. For Archegos, sitting on perhaps 5x leverage with an enormous concentrated position in this exact stock (and in correlated names that also fell), it was lethal.

#### Worked example: the 10% drop that wipes the equity

Let me isolate the leverage math on a single slice of the book to show why a moderate stock move was fatal.

- Suppose Archegos has \$2 billion of margin posted against \$10 billion of ViacomCBS-and-similar exposure with one bank (5x leverage).
- ViacomCBS and its correlated names fall 10%. The \$10 billion of exposure loses \$1 billion.
- That \$1 billion loss comes straight out of the \$2 billion of margin, halving it to \$1 billion.
- The bank, to maintain 5x on a now-\$9 billion position, wants about \$1.8 billion of margin. Archegos is short. The bank issues a margin call.
- Now imagine the drop is 20%, not 10%. The loss is \$2 billion — the entire margin is gone, and the bank is exposed to further losses with no cushion.

The intuition: at 5x leverage, a 20% adverse move erases 100% of your equity. The stocks did move roughly that much, across the whole concentrated book, in a matter of days. There was no cushion left.

### Wednesday-Thursday: the calls Archegos cannot meet

As the stocks fell, every prime broker's risk system flashed at once. Each bank's margin model said the same thing: Archegos's collateral is no longer enough; issue a margin call. The calls went out — collectively, billions of dollars of demands for fresh collateral, due immediately.

Archegos could not meet them. Its \$10 billion of equity was already fully deployed as margin across the banks; there was no spare cash to post. When a leveraged fund's losses exceed its ability to post more collateral, it has defaulted in all but name. By Thursday, March 25, the banks understood that Archegos was insolvent and could not save itself.

### Thursday night: the banks finally compare notes

On the evening of March 25, the prime brokers held an emergency call with Archegos. For the first time, the banks began to grasp the full picture — that they were not the only counterparty, that the same handful of stocks were pledged across the entire street, and that the total position dwarfed anything they had individually modeled. Archegos reportedly asked for an orderly, coordinated wind-down: sell everything slowly, share the pain, avoid crashing the stocks.

This is the pivotal moment. An orderly, coordinated unwind would have been better for everyone *collectively*. But it required every bank to trust every other bank to hold back. And no bank could be sure the others would. This is a textbook prisoner's dilemma, and it resolved exactly the way prisoner's dilemmas resolve.

### Friday onward: the race to the exit

![Graph of the liquidation race among the brokers](/imgs/blogs/archegos-2021-total-return-swaps-blowup-7.png)

On Friday, March 26, Goldman Sachs and Morgan Stanley broke ranks. They did not wait for coordination. They seized the collateral (the shares the banks themselves were holding as hedges) and began dumping enormous blocks into the market. Goldman alone reportedly sold around \$10 billion of stock over Friday and the following days. Morgan Stanley sold billions more, some of it pre-placed quietly with other investors before the news broke.

Once two banks were selling, the rest had no choice. If you are holding \$10 billion of ViacomCBS as your hedge against a defaulted swap, and you can see Goldman flooding the market with the same stock, every minute you wait the price falls further and your eventual recovery shrinks. So Credit Suisse, Nomura, UBS, and the others piled in — but later, and into a market already cratering.

The stocks collapsed. ViacomCBS and Discovery each fell roughly 50% in a matter of days. The Chinese ADRs were savaged. The very concentration that had powered Archegos's gains now powered its losses: because the position was packed into so few names, the forced selling overwhelmed the natural liquidity in those names and the prices went into free fall.

By the end of the following week, Archegos was gone. Bill Hwang's personal fortune, once estimated in the many billions, was largely wiped out. And the banks tallied their losses.

#### Worked example: the block-trade fire sale

Let me quantify why dumping the position crashed it.

- ViacomCBS traded perhaps \$1-2 billion of shares on a normal day.
- The banks collectively needed to sell roughly \$20 billion of ViacomCBS and a few correlated names within days.
- That is more than ten times a normal day's volume, concentrated into a window of hours, with every seller knowing the others are selling.
- To find buyers for that much stock that fast, the sellers had to offer steep discounts — block trades were placed well below the prevailing price.
- Each discounted block reset the market price lower, which forced the next block to be sold even lower. The price gapped down 50%.

The intuition: liquidity is not a fixed property of a stock; it evaporates exactly when everyone needs it. A position you can build quietly over a year you cannot exit quietly in a day, and the cost of that asymmetry is the fire-sale discount.

## The mechanism dissected: why it broke

We have the narrative. Now the depth — the second-order structure that turned a bad bet into a \$10 billion systemic event. Three mechanisms compounded: hidden leverage, extreme concentration, and the liquidation race. Underneath all three sat the disclosure gap.

### Mechanism one: leverage hidden by fragmentation

Leverage by itself is dangerous but visible. A bank lending you \$40 against \$10 of equity knows it has lent you \$40. What made Archegos's leverage uniquely dangerous was that it was *fragmented* across counterparties so that the aggregate was invisible.

![Stack of the Archegos leverage](/imgs/blogs/archegos-2021-total-return-swaps-blowup-6.png)

Think of it this way. Each bank ran a risk model on its own book. Credit Suisse's model said: "We have written Archegos swaps on \$10 billion of underlying; Archegos has posted adequate margin; in a stress scenario where these stocks fall 20%, we might lose a few hundred million, which is uncomfortable but survivable." That model was not wrong about Credit Suisse's slice. It was wrong about the *world*, because it assumed Credit Suisse could exit its position into a functioning market. It could not, because five other banks were trying to exit the same names at the same instant.

The hidden leverage poisoned every bank's risk model in the same way: each model assumed the bank's own selling would not move the market much, which is true for a slice and catastrophically false for the whole. The fragmentation meant the *total* leverage — the thing that actually determined the fire-sale severity — appeared on no one's risk report. The system as a whole was running at perhaps 5x leverage on a hyper-concentrated book, and there was no entity, public or private, that knew it.

#### Worked example: the same loss seen two ways

- Credit Suisse's own view, ex ante: \$10 billion of Archegos exposure, 20% stress = \$2 billion gross move, mostly covered by margin, manageable.
- The system's reality, ex post: \$50+ billion of exposure across all banks in the same names; when forced to sell, the market gapped 50%, not 20%, because the selling itself moved the price; Credit Suisse, selling last, recovered far less than its margin assumed.
- Credit Suisse's realized loss: about \$5.5 billion — far beyond its own stress estimate.

The intuition: a risk model that ignores what your competitors are doing in the same trade is not a risk model, it is a fair-weather forecast.

### Mechanism two: concentration turned correlation to one

A diversified leveraged book can survive a shock because its many positions do not all fall together. Archegos's book could not, because it was concentrated in a handful of names that were, in the crisis, perfectly correlated — they fell together because the *same forced seller* was dumping all of them at once.

This is a subtle and important point. The stocks were not fundamentally correlated in normal times (ViacomCBS and a Chinese tutoring company have little in common). But in the unwind they became correlated, because the common factor driving all their prices down was the Archegos liquidation itself. Concentration plus forced selling manufactures correlation. The diversification Archegos thought it had — different sectors, different geographies — was an illusion the moment a single seller was forced to liquidate everything simultaneously.

### Mechanism three: the prisoner's dilemma of liquidation

The liquidation race deserves its own dissection because it is so often misunderstood. When Archegos defaulted, the banks faced a collective-action problem. Consider it as a game between two banks, each holding a large block of the same stock as collateral:

- If both sell slowly and cooperatively, they each recover, say, 90 cents on the dollar. Best collective outcome.
- If one sells fast while the other holds, the fast seller recovers 90 cents (it got out before the crash) and the slow holder recovers 50 cents (it sold into the crash the first seller caused). 
- If both sell fast, they crash the stock together and both recover 60 cents.

Look at the incentives for either bank. Whatever the other bank does, you are better off selling fast: if they hold, you get 90 instead of being the sucker; if they sell, you get 60 instead of 50. So both sell fast, both get 60, even though both holding would have gotten them 90. The cooperative outcome is unstable because no one can trust the others to hold, and the cost of being the last seller is ruinous.

This is why Goldman and Morgan Stanley selling first was not villainy — it was the rational, even fiduciary, response to a situation where holding back meant absorbing other banks' losses. And it is why the banks that hesitated, hoping for an orderly process, paid the most. Credit Suisse and Nomura, slower to act, sold into the worst of the crash.

### The disclosure gap underneath it all

Every one of these mechanisms required the position to be hidden. Had Archegos been forced to file 13D ownership reports as its swap exposure crossed 5%, 10%, 15% in ViacomCBS and the Chinese ADRs, several things would have changed. Other investors would have seen the concentration and grown wary. The banks, seeing the public filings, would have understood they were not alone and would have demanded more margin or capped their exposure. The companies themselves would have known a single leveraged holder dominated their float. The fire sale might still have happened, but its scale would have been known and priced. The disclosure gap was not a side detail; it was the enabling condition for everything else.

It is worth pausing on how large the hidden stakes were. Archegos's swap exposure to ViacomCBS at its peak represented something on the order of 10% or more of the entire company — a holding that, owned outright, would have made Archegos one of the largest shareholders of a major media company and a mandatory 13D filer many times over. The same was true across several of the Chinese ADRs, where Archegos's swap-driven buying had become a dominant share of the daily trading volume. By controlling these stakes through swaps rather than shares, Hwang kept his name off every register, every filing, every list of major holders. The companies' investor-relations teams did not know who their biggest economic owner was. Index funds rebalancing those stocks had no idea a single forced seller could appear. The cloak was total, and it held right up to the moment the banks were forced to throw it off by selling the underlying shares into the open market — the one act that finally made the position public, at the worst possible price.

## The aftermath: who paid and what changed

### The bank losses

The losses landed unevenly, and the pattern tells the story of the liquidation race precisely.

![Matrix of prime brokers, exposure, exit, and loss](/imgs/blogs/archegos-2021-total-return-swaps-blowup-5.png)

- **Goldman Sachs**: sold first and fastest; reported negligible losses, possibly even a small gain on the unwind.
- **Morgan Stanley**: sold early, including pre-placed blocks; lost roughly \$900 million.
- **Wells Fargo, Mizuho, Mitsubishi UFJ, Deutsche Bank**: smaller exposures and quicker exits; modest or minimal losses (Deutsche Bank reportedly hedged out with little loss).
- **UBS**: lost about \$770 million.
- **Nomura**: lost about \$2.9 billion.
- **Credit Suisse**: lost about \$5.5 billion — by far the worst.

The total across all counterparties came to roughly \$10 billion. The ordering is not random. The banks that prioritized speed over coordination — Goldman, Morgan Stanley — protected themselves. The banks with weaker risk culture and slower internal decision-making — Credit Suisse above all — were left holding the bag.

#### Worked example: Credit Suisse's \$5.5 billion against its capital

The loss is abstract until you size it against the bank.

- Credit Suisse's loss from Archegos: about \$5.5 billion.
- For comparison, Credit Suisse's total net income in a good year was on the order of \$3-4 billion.
- So the Archegos loss alone exceeded an entire year's profit for the bank, in a single event, from a single client.
- The bank's Common Equity Tier 1 capital (the core regulatory cushion) was on the order of \$45 billion; \$5.5 billion is more than 10% of that cushion gone to one mistake.

The intuition: a counterparty loss that consumes more than a year of profit and a tenth of your core capital is not a trading mishap, it is an existential governance failure — and it was treated as one.

### The internal reckoning

At Credit Suisse, an independent investigation (the "Paul, Weiss report") found a litany of failures: the prime brokerage business had let Archegos's exposure balloon while holding far too little margin, risk officers' warnings were ignored or under-resourced, and the firm had a culture that prioritized fees from a big client over prudent limits. Several senior executives, including the head of the investment bank and the chief risk officer, left. Coming on top of a separate scandal that same month (the collapse of Greensill Capital, supply-chain finance funds Credit Suisse had marketed), Archegos shattered confidence in Credit Suisse's risk management.

### Regulatory response: closing the disclosure gap

The episode prompted regulators to revisit the swap-disclosure gap. The SEC proposed and advanced rules to require more disclosure of large security-based swap positions, aiming to ensure that someone building a dominant economic stake through swaps could no longer remain entirely invisible. Regulators and the Financial Stability Board also pushed prime brokers to improve their handling of concentrated, leveraged single-client exposures — to demand more margin, to stress-test for the possibility that they were not the only counterparty, and to share risk information where possible. Bank prime-brokerage divisions tightened terms with family offices and concentrated funds. The broad lesson the industry took: a wealthy, sophisticated client with a clean-looking individual relationship can still be a systemic risk if the same client is doing the same thing at every bank on the street.

### Credit Suisse's road to 2023

Archegos did not kill Credit Suisse on its own, but it was a major waypoint on the road to the bank's demise. The reputational damage, the capital hit, the management churn, and the demonstration that the bank could not control its own risk all eroded confidence. Over the following two years a series of further scandals, losses, and client outflows hollowed the bank out. In March 2023, amid a broader banking scare, Credit Suisse suffered a fatal run on deposits and was forced into an emergency rescue takeover by its rival UBS, ending 167 years of independent history. You can read the fuller version of that collapse in the companion piece on the [2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs). Archegos was the moment the market started to seriously doubt whether Credit Suisse knew what it was doing.

### Bill Hwang's conviction

The legal reckoning for Hwang came later. In April 2022, US prosecutors charged him with securities fraud, racketeering, and market manipulation, alleging that he had deliberately used the swaps both to hide his positions and to artificially inflate the prices of the stocks he held (his buying pushed prices up, which let him borrow more, which let him buy more). In July 2024, a jury convicted Hwang on most counts. In late 2024 he was sentenced to 18 years in prison. The conviction reframed Archegos: not merely a risk-management failure by the banks, but a deliberate scheme by Hwang to deceive his counterparties and manipulate the market.

## Common misconceptions

Archegos is widely misremembered. Here are the most common errors, each corrected with the mechanism that explains why it is wrong.

### Misconception 1: "Swaps are just gambling with no real-world effect"

It is tempting to think that because Archegos never owned the actual shares, its swaps were a private side-bet that did not touch the real market. This is exactly backwards. When you enter a total-return swap, the bank on the other side **buys the actual stock to hedge itself**. The bank does not want to be on the hook for the stock's return without offsetting it, so it goes into the open market and buys the shares. So Archegos's swaps caused tens of billions of dollars of real buying in ViacomCBS and the other names — buying that pushed their prices up. And when the swaps unwound, that same real stock had to be sold, crashing the prices. The swaps were not a sideshow; they were the hidden engine driving the real share prices both up and down. A derivative whose hedge is a real purchase is not a side-bet at all.

### Misconception 2: "The banks didn't know they were lending Archegos this money"

The banks knew perfectly well they had each extended large, leveraged swap exposure to Archegos. What they did not know was the *total* across all banks. Each bank consented to its own slice with eyes open; the failure was not ignorance of their own lending but ignorance of the aggregate, combined with too-generous margin terms. The distinction matters: this was not a fraud that tricked banks into lending money they did not realize they were lending. It was a structural blind spot — no bank could see the others — compounded by banks competing so hard for a lucrative client that they under-priced the risk of their own slice. (Hwang's later conviction added a layer of active deception about position sizes, but the structural blindness existed independent of any lie.)

### Misconception 3: "A family office is too small to matter"

The word "family office" sounds small and private — a wealthy individual managing personal money. The assumption baked into the regulatory exemption was that such an entity could only hurt itself. Archegos demolished that assumption. With \$10 billion of equity leveraged 5x into \$50+ billion of concentrated exposure, a single family office moved individual stock prices, then triggered \$10 billion of losses at globally systemic banks, contributed to the eventual collapse of a 167-year-old institution, and prompted new SEC rulemaking. Size is not just headcount or client count; it is leverage times concentration. A small entity with enough leverage and concentration is, functionally, a large and dangerous one.

### Misconception 4: "The banks that sold first behaved badly"

There is a moral intuition that Goldman and Morgan Stanley "ran for the exit" and "left the others holding the bag," and that this was somehow unfair or predatory. But in a default situation, each bank is holding collateral it is entitled to sell to recover what it is owed. As the prisoner's-dilemma analysis showed, once any bank might sell, the rational and even fiduciarily required move for every bank is to sell promptly, because the cost of selling last is ruinous. The first sellers were not behaving badly; they were behaving rationally under a structure that made cooperation impossible to sustain. The blame belongs to the structure (the hidden leverage, the absence of a coordinated workout mechanism), not to the banks that read the structure correctly and acted on it. Punishing the fast sellers would only teach everyone to sell even faster next time.

### Misconception 5: "This was a 2008-style systemic crisis"

Archegos was a serious event, but it was not 2008. In 2008 the losses were spread across the whole financial system, tied to mortgage assets held everywhere, and they froze the entire credit market. Archegos's losses were concentrated in a handful of prime brokers and a handful of stocks; the broader market barely noticed, and there was no systemic freeze. The lesson of Archegos is narrower and in some ways more unsettling: the system absorbed this one, but only because it happened to involve a manageable set of counterparties and liquid large-cap stocks. A larger or more interconnected version — more leverage, less liquid underlyings, more banks — could have been far worse, and the disclosure gap that enabled it was a general feature, not a one-off.

### Misconception 6: "Better risk models would have prevented it"

More sophisticated quantitative models at each bank would not have caught this, because the missing information was not modelable from any single bank's data. The danger lived in the *aggregate* across banks, which no bank's model had access to. You cannot model your way out of a blind spot whose cause is missing data, not flawed math. The fix is structural — disclosure, information sharing, aggregate limits — not a better risk equation. This is a recurring theme in financial blow-ups: the failure is usually not that the math was wrong, but that the math was fed a picture missing the one variable that mattered.

## How it echoes in other markets

Archegos is one instance of a pattern that recurs whenever hidden leverage meets forced selling. Here are other episodes that share its mechanism, so the lesson generalizes beyond this single story.

### Long-Term Capital Management (1998)

LTCM was a hedge fund run by Nobel laureates that built a balance sheet of about \$125 billion on roughly \$5 billion of equity, plus derivative notional exceeding a trillion dollars — leverage that, like Archegos's, was spread across many bank counterparties so that no single one saw the whole. When Russia defaulted in August 1998, LTCM's correlated bets all moved against it at once, margin calls cascaded, and a fund that could not be quietly unwound threatened to take the street down with it. The Federal Reserve organized a \$3.6 billion bank consortium to recapitalize and unwind it in an orderly way. The echo is exact: hidden, fragmented leverage; extreme reliance on positions staying liquid; a forced unwind that the system could not absorb gracefully. The difference is that LTCM was deemed systemic enough to require a coordinated bailout, whereas Archegos's banks were left to eat their own losses.

### The 1998 flight to liquidity

Beyond LTCM specifically, the autumn of 1998 demonstrated the general phenomenon that liquidity vanishes precisely when leveraged players need to sell. The "flight to liquidity" — investors dumping anything risky and crowding into safe, liquid assets — meant that leveraged positions could not be exited at anything near their marked value. Archegos's fire sale was a compressed, single-fund version of the same dynamic: the assumption that you can always sell at a price close to the last quote is the assumption that kills leveraged players, over and over.

### The London Whale (2012)

JPMorgan's Chief Investment Office built an enormous, concentrated position in credit derivatives that grew so large relative to the market that the trader (Bruno Iksil, nicknamed the "London Whale") could no longer move it without moving the price against himself. When the position turned, JPMorgan lost over \$6 billion. The shared mechanism is concentration: a position so large relative to the available liquidity that exiting it is itself the thing that crushes the price. Archegos was concentration imposed by a small entity on liquid stocks; the Whale was concentration imposed by a giant bank on a thinner derivatives market. Both learned that size relative to liquidity, not size in absolute dollars, is what determines whether you can get out.

### Leveraged-ETF and volatility-product unwinds (2018, "Volmageddon")

In February 2018, a spike in market volatility triggered the collapse of products that were short volatility — they were structured so that a sharp rise in volatility forced them to buy volatility to rebalance, which pushed volatility higher, which forced more buying. The XIV exchange-traded note lost almost all its value in a day. The echo is the feedback loop: a structure where the forced response to a price move amplifies that very move. Archegos's fire sale had the same self-reinforcing quality — selling to meet margin crashed the price, which triggered more margin calls, which forced more selling.

### The Treasury basis trade scares (2020 and beyond)

Hedge funds run a "basis trade" that exploits a tiny price gap between Treasury bonds and Treasury futures, made meaningful only by enormous leverage obtained through the repo market — and, as with Archegos, spread across many counterparties so the aggregate is hard to see. In March 2020, as the pandemic shock hit, this trade blew up: forced deleveraging in the world's most important bond market required the Federal Reserve to intervene massively to restore order. Regulators have repeatedly warned that the sheer size and hidden leverage of the basis trade make it a candidate for the next Archegos-style accident, just in a far more systemically important market. The mechanism is identical: hidden, fragmented, enormous leverage on a position that everyone assumes is safe because the underlying gap is small — until a shock forces a simultaneous unwind.

### Amaranth Advisors (2006)

Amaranth was a hedge fund that built a massive, concentrated bet on natural-gas futures spreads, controlled through positions so large that they dominated the market. When the spreads moved against it, the fund lost about \$6 billion in a week and collapsed. The echo is concentration plus leverage in a market too thin to absorb the exit. Like Archegos, Amaranth had grown a position whose size was its undoing: the bigger the bet, the worse the price you get when forced to close it.

### Nick Leeson and Barings (1995)

A single trader, Nick Leeson, accumulated enormous unauthorized leveraged bets on Japanese stock-index futures while hiding the true size of his position in a secret account. When the market moved against him after the Kobe earthquake, the losses — about \$1.4 billion — exceeded the entire capital of Barings Bank, a 233-year-old institution, and destroyed it overnight. The echo to Archegos is the hidden-position theme in its purest form: a leveraged bet whose true size was concealed from the people who could have stopped it, revealed only when the losses forced it into the open. Barings shows the same lesson at the level of one rogue trader that Archegos shows at the level of one family office spread across a dozen banks: it is not the bet that kills you, it is the bet you could not see.

### Melvin Capital and the 2021 short squeeze

In a strange mirror-image happening at almost the same time as Archegos, the hedge fund Melvin Capital was destroyed in early 2021 by a short squeeze in GameStop and other "meme stocks." Melvin had a large, leveraged *short* position; a coordinated buying wave forced it to cover at enormous losses. The mechanism rhymes: a concentrated leveraged position, a forced unwind, and losses amplified by the unwind itself. You can read the mechanics of that episode and how leveraged funds operate in the piece on [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20). The common thread across Melvin and Archegos is that 2021's cheap money and concentrated positioning made several funds fragile to the same kind of forced-unwind shock, in opposite directions.

## When this matters to you, and further reading

You will probably never write a total-return swap or run a family office. So why does Archegos matter to you?

First, because it is the cleanest available lesson in **hidden leverage**, and hidden leverage is the recurring cause of financial accidents. Whenever you read that some fund, bank, or product "blew up," the question to ask is not "what did they bet on?" but "how much did they borrow, and could anyone see it?" The instrument changes — repo, swaps, futures, options — but the failure is almost always leverage that was invisible until it was forced into the open. Train yourself to look for the leverage and to ask who can see the total.

Second, because it teaches the difference between **risk you can see and risk you cannot.** Each Archegos bank had competent risk managers running reasonable models. They were not stupid; they were blind, because the system was built so that the dangerous variable — the aggregate position — appeared on no one's screen. When you evaluate any system, financial or otherwise, ask what it is structurally incapable of seeing. That blind spot, not the visible risks everyone is already managing, is where the next failure will come from.

Third, because it shows how **competition can manufacture systemic risk.** The banks were not colluding; they were competing fiercely for a prized client, each cutting the others' margins and asking fewer questions to win the business. That competition is what let the aggregate exposure balloon. Healthy competition between firms can, in aggregate, produce a deeply unhealthy outcome for the system — a reminder that what is individually rational and what is collectively safe are not the same thing.

Fourth, because the **liquidation dynamics generalize far beyond finance.** The prisoner's dilemma of the banks racing to sell is the same structure as a bank run, a stampede for an exit, a sell-off in any crowded trade. Whenever an outcome depends on everyone holding back but no one can trust the others to hold back, expect the panic. Understanding why Goldman selling first was rational, not villainous, is understanding why panics happen and why "just stay calm, everyone" is not a stable strategy.

If you want to go deeper into the institutions and mechanics behind this story, these companion pieces build out the surrounding world:

- [How hedge funds work: leverage, 2-and-20, and the search for alpha](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20) — the economics of leveraged funds and why they reach for borrowed firepower.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — what prime brokerage is and why banks compete so hard for clients like Archegos.
- [SVB and Credit Suisse, 2023: the year the bank runs came back](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — where Credit Suisse's road, which Archegos helped start, finally ended.
- [A field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) — how family offices, hedge funds, banks, and regulators fit together in the larger map.

The enduring image of Archegos is a man building, brick by invisible brick, the largest concentrated stock position in the world, with no one able to see the whole structure rising — until one stock cracked, and the structure came down in a single week, burying \$10 billion of other people's capital under it. The instruments were ordinary. The arrangement was lethal. And the lesson is permanent: leverage you cannot see is the most dangerous leverage there is, and any market built so that no one knows the total size of the bets will, sooner or later, be astonished by the size of the bets.
