---
title: "Bookbuilding and Price Discovery: How the IPO Price Is Set"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How a bank turns hundreds of scattered investor orders into one IPO price — the demand curve, the allocation decision, why IPOs pop on day one, and the auctions and direct listings that tried to fix it."
tags: ["capital-markets", "ipo", "bookbuilding", "price-discovery", "underwriting", "primary-market", "underpricing", "dutch-auction", "direct-listing", "spac"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — An IPO price is not a calculation; it is the output of a controlled auction called **bookbuilding**, in which a bank collects investor orders, draws a demand curve, and picks the price by hand.
>
> - The **book** is a ledger of investor orders — how many shares each wants and up to what price. Stacked from low price to high, those orders form a downward **demand curve**, and the deal "clears" where cumulative demand meets the number of shares offered.
> - The bank does not just read the curve — it judges the **quality** of demand (sticky long-only funds vs. flippers), and it keeps **discretion over allocation**, which is why hot IPOs go to favoured clients and why "spinning" and "laddering" drew regulators.
> - IPOs are **systematically underpriced**: the median US IPO has popped double digits on day one for decades. That pop is real money the company leaves on the table — a `\$500M` raise that pops 30% handed `\$150M` to first-day buyers.
> - The one fact to remember: the IPO price is set to *clear the book with room to spare*, not to capture the last dollar — which is exactly why the stock usually jumps when it opens.

On the morning of December 11, 1980, Apple Computer went public at `\$22` a share. By the time the stock opened for trading it was at `\$29`, and it closed near `\$29` too — a roughly 32% jump in a single day on a deal that raised about `\$100M`. The bankers who priced it at `\$22` were not incompetent. They knew, more or less, that the stock would trade higher. They priced it there on purpose. Four decades later, on the morning of September 14, 2021, the eyewear company Warby Parker went public — but it skipped the bankers' price entirely and let the New York Stock Exchange open the stock wherever buyers and sellers met. Two companies, two completely different machines for answering the same question: *what is one share of this company worth at the exact moment it starts trading?*

That question has no clean answer, because before the first trade there is no price — only opinions. A capital market's whole job is to **turn savings into long-term investment**, and the primary market is the engine that *creates* the security in the first place: it manufactures a brand-new claim on a company's future and sells it to savers for cash the company can spend on factories, software, and people. But manufacturing a security raises a problem a factory never faces — you have to invent its price from nothing, with hundreds of buyers who will all lie a little about how much they want it. The mechanism markets evolved to solve this is **bookbuilding**, and understanding it is understanding how price discovery actually works when there is no market yet to discover a price from.

This post is about that mechanism: how investor demand becomes a single number, who decides it, why the number is almost always set a little too low, and what the alternatives — auctions, direct listings, SPACs — change about the answer.

![Bookbuilding demand curve mapping cumulative demand to a clearing price](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-1.png)

## Foundations: what a "book" is and why the price has to be discovered

Start with the everyday version. Suppose you are selling one rare concert ticket and ten friends want it. You could name a price and hope. Or you could ask each friend privately, *"how much would you pay?"* — and write their answers on a sheet of paper, sorted from highest to lowest. That sheet is a **book**. Reading down it, you learn not just the top bid but the *shape* of demand: maybe two friends would pay `\$200`, four would pay `\$120`, and the rest only `\$60`. Now you can set a price knowing exactly how many buyers you keep and how many you lose at each level.

An IPO is the same exercise scaled up by a factor of a thousand, with one twist: you are not selling one ticket, you are selling *millions of identical shares* at a *single uniform price*. Everybody who gets shares pays the same price — the **clearing price** — no matter what they said they would pay. So the bank's job is to find the one price at which it can sell all the shares on offer to buyers it trusts, with demand to spare.

A few terms have to be nailed down before any of this makes sense.

- A **security** is a claim you can sell — here, a share of stock, which is a slice of ownership in the company plus a vote and a claim on future profits. (What a share is *worth* in fundamental terms — discounted cash flows, multiples — is the job of [equity research](/blog/trading/equity-research/dcf-valuation-projecting-and-discounting-free-cash-flow); this post is about how a *price* gets *set* on the day, which is a different thing. Investors disagree about value; the IPO price is where their disagreement gets resolved into one number.)
- The **primary market** is where securities are *created and sold for the first time* to raise capital for the issuer. The **secondary market** is where those same shares change hands between investors afterward, with no new money reaching the company. An IPO — initial public offering — is the canonical primary-market event: the company sells new shares (and/or existing holders sell theirs), money flows in, and the very next moment the shares begin trading in the secondary market. This post lives squarely in the primary market.
- The **issuer** is the company raising money. The **underwriter** or **bookrunner** is the investment bank that organizes the sale, sets up the book, and — critically — often *buys the shares from the company and resells them*, taking the risk in between. (Who bears that risk, and how the syndicate of banks splits it, is its own deep topic — see the sibling post on [underwriting and the syndicate](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk).)
- An **indication of interest**, or **IOI**, is a non-binding statement from an investor: *"we'd buy roughly this many shares at roughly this price."* IOIs are the raw material of the book. They are not legally binding orders, which matters — investors can and do inflate them, and reading through that inflation is half the bookrunner's skill.

The reason the price has to be *discovered* rather than *calculated* is that the people who know what the shares are worth — the institutional investors who will hold them — have no incentive to tell the truth. If you ask an investor "how much would you pay?", the honest answer hurts them: a high answer means they pay more. So they shade their answers down, and the bookrunner has to extract the real demand curve from a chorus of strategically understated IOIs. This is the central tension of the whole exercise, and it explains almost everything that follows, including the persistent underpricing.

### Why the primary market needs the secondary market to even function

It is worth pausing on *why anyone buys an IPO at all*, because the answer is the spine of this whole series. A capital market is a machine that turns savings into long-term investment. A saver hands over cash today; a company turns that cash into factories, code, and hiring that pay off over years or decades. But almost no saver will lock their money into a 20-year project if they cannot get it back when they need it. The thing that resolves this — that makes a saver willing to fund a multi-decade venture — is the **secondary market**: the promise that the share they buy in the IPO can be sold to someone else tomorrow morning at a market price.

This is not a footnote to bookbuilding; it is its precondition. When a fund manager decides how much to bid in the book and at what price, the single most important thing they are assessing is *aftermarket liquidity*: will there be a deep, two-sided market in this stock the moment it lists, so I can adjust my position, take profit, or cut a loss? A company with thin expected liquidity — a small float, an obscure sector — gets a thinner book and a lower price, because investors demand a discount for being stuck. A company with a famous brand and an expected gusher of trading volume gets a fat book at a full price. In other words, **the bookrunner is not really selling a company; it is selling a claim that the secondary market will keep liquid.** The entire pricing exercise is downstream of how confident the buyers are that they can get out. That is why an IPO window slams shut the instant the secondary market gets scared: when liquidity dries up, the promise that underwrites every IOI evaporates, and the book cannot be built at any price.

Here is the mental model for how the scattered IOIs become one price. The whole process is a funnel.

![Pipeline from the price range through the order book to one clearing price and allocation](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-2.png)

The deal starts with a **price range** printed in a preliminary prospectus, runs through a **roadshow** where management pitches investors, collects **IOIs** into the **book**, and ends in two decisions the bank makes by hand: the **price** and the **allocation**. Everything in this post is an expansion of one of those boxes.

## The book: how a bookrunner records demand

When a company decides to go public, it hires one or more investment banks as bookrunners and files a preliminary prospectus — known in the trade as the **red herring**, after the red disclaimer printed down its side warning that the document is not yet final. The red herring contains everything about the company except the final price; instead it carries a **price range**, say `\$18` to `\$22` a share. That range is the bank's opening hypothesis about where demand sits, built from comparable companies, the company's financials, and pre-marketing conversations with a handful of big investors. The range is deliberately a range, not a point — it is an invitation to the market to push back.

Then comes the **roadshow**: a one-to-two-week tour where the CEO and CFO present to institutional investors in back-to-back meetings — mutual funds, pension funds, hedge funds, sovereign wealth funds. The roadshow's real purpose is not persuasion; it is *measurement*. As investors leave each meeting, the bank's salesforce calls them and asks for their IOI. Those IOIs flow into the book.

### What an order in the book actually contains

A serious IOI has three parts: a **size** (how many shares or how many dollars), a **price** (up to what level the investor will pay), and increasingly, a **type** that says how price-sensitive that demand is. This is where bookbuilding borrows the vocabulary of the order book in the secondary market — though the meanings shift a little.

- A **limit order** (the price-sensitive kind) says *"I'll buy 2 million shares at `\$20` or below, but not a penny more."* This is gold to the bookrunner, because it reveals a point on the true demand curve. If the price comes in at `\$22`, this investor is out; at `\$20` they are in for 2 million. Stack enough limit orders and you can see the whole curve.
- A **market order** (the price-insensitive kind) says *"I'll buy 2 million shares at whatever price you set."* This sounds like strong demand, and naïve readers treat it that way. But it carries almost no information — it tells the bookrunner nothing about *where* demand thins out. A book full of market orders is a book the bank is flying blind on. Worse, price-insensitive orders are exactly what a flipper submits when they only care about getting an allocation in a hot deal, not about the price.
- A **step order** (sometimes called a scaled or strip order) is the richest of all: *"I'll buy 3 million at `\$18`, but only 2 million at `\$20`, and just 1 million at `\$22`."* A single step order maps one investor's entire personal demand curve. The biggest, most sophisticated funds submit step orders precisely because they reveal how the investor's interest fades as the price climbs — and good bookrunners reward that honesty with better allocations.

The matrix below lays out how each order type — and each kind of investor — feeds the bookrunner's read on demand.

![Matrix of IPO order types and investor quality against the demand signal they give](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-3.png)

### The quality of demand: long-only money vs. fast money

Here is the thing a spreadsheet cannot capture and the single most important judgment in the whole process. **Not all demand is equal.** A bookrunner does not want to simply sell to the highest bidders; it wants to sell to investors who will *hold* the stock, support it, and come back for the next deal. The book is sorted not just by price but, implicitly, by *quality*.

- **Long-only funds** — the big mutual funds and pension managers who buy a stock to hold it for years — are the prize. Their demand is "sticky": they are not going to dump the shares on the open. An order from a respected long-only fund is *price-forming*, because the rest of the market reads their participation as a vote of confidence. Bookrunners over-allocate to these accounts and under-allocate to everyone else, and they will happily price the deal a touch lower to keep the long-only books happy.
- **Fast money** — hedge funds and proprietary traders looking to "flip" the stock for the first-day pop and sell into the open — inflates the book without adding quality. A hedge fund might put in a `\$50M` market order on a `\$300M` deal not because it wants `\$50M` of stock but because it knows it will be cut back to `\$5M`, and it wants `\$5M` of a stock it plans to sell within an hour. This is rational for them and corrosive for the bookrunner's read on real demand.

Distinguishing the two is the bookrunner's craft. A book that looks "10× oversubscribed" might be 3× real long-only demand and 7× flippers padding their orders. The art is to discount the padding and find the price at which the *sticky* demand still covers the deal.

#### Worked example: building the book and finding the clearing price

Say a company is selling **20 million** new shares, with a red-herring range of `\$18`–`\$22`. After the roadshow the book looks like this, as cumulative limit orders:

- At `\$22` or below: investors want **12 million** shares.
- At `\$20` or below: investors want **37 million** shares (the 12M from `\$22` plus 25M more who only step in once the price drops to `\$20`).
- At `\$18` or below: investors want **77 million** shares (another 40M appear at the bottom of the range).

Now find the **clearing price** — the highest price at which cumulative demand still covers the 20 million shares on offer:

- At `\$22`: demand is 12M < 20M offered → **not covered**. Price the deal here and 8M shares go unsold. Too high.
- At `\$20`: demand is 37M > 20M offered → **covered, with 1.85× oversubscription** (37 ÷ 20). Comfortable.
- At `\$18`: demand is 77M > 20M → covered 3.85×, but you are leaving price on the table.

The deal clears at **`\$20`**: the highest price that fully covers the offering with a healthy cushion. The company raises **20M × `\$20` = `\$400M`** (before fees). The bookrunner won't push to `\$22` even though *some* demand exists there, because at `\$22` the book is under-covered and the stock would likely break below the offer price on day one — the cardinal sin of an IPO. *The clearing price is the highest price the book covers comfortably, not the highest price anyone will pay.*

This is exactly the curve in Figure 1 at the top of the post: cumulative demand on the horizontal axis, price on the vertical, and the clearing price where the staircase crosses the 20-million-share supply line.

### The shape of the curve tells you more than the clearing price

A bookrunner reads not just *where* the curve crosses the supply line but *how steep it is there*. Two books can both clear at `\$20` and behave completely differently afterward.

- A **steep** curve at the clearing price — demand falling away fast just above `\$20` — means the price is fragile. Push it a dollar higher and large chunks of demand vanish; a little bad news after listing and there are few buyers waiting underneath to catch the stock. The bank prices conservatively into a steep curve and keeps a deeper cushion.
- A **flat** curve — lots of demand stacked at `\$19`, `\$20`, and `\$21` alike — means robust, price-insensitive interest across a band. The bank can price near the top of the band with confidence, because even if the stock dips, a wall of demand sits just below the offer ready to absorb the shares. A flat, deep curve is what lets a bank price aggressively without fear of a broken deal.

This is why the *count* of oversubscription ("the deal was 10× covered") is a crude summary. A deal 10× covered by flippers with market orders is far more fragile than a deal 3× covered by long-only funds with limit orders stacked tightly around the price. The bookrunner is pricing the *quality and shape* of the curve, not its headline multiple — which is exactly why the whole apparatus needs human judgment rather than a clearing algorithm.

### Bookrunner, co-managers, and who actually controls the book

One more structural point, because it bears on the allocation discretion below. A typical IPO is run by a **syndicate** of banks, but they are not equals. One or two **lead bookrunners** physically hold the book — they see every order, they talk to every major investor, and they make the pricing and allocation recommendations. The other banks in the syndicate — **co-managers** and junior bookrunners — get their names on the cover, share in the fee, provide research coverage, and bring some of their own clients, but they often see only a partial view of the book and have little say in allocation. This is why winning the **lead-left** position (the leftmost, most senior name on the prospectus cover) is so fiercely contested among banks: the lead controls the information and the discretion, which is where both the prestige and the power sit. The detailed economics of who in the syndicate takes the underwriting risk and how the spread is split is the subject of the [underwriting](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk) post; here, what matters is that "the bookrunner" who reads the demand curve and decides the price is a specific, powerful seat — not the whole syndicate.

## Setting the price: from the range to the final number

The range in the red herring is a starting bid, not a commitment. As the book builds, the bankers watch the demand curve take shape and *revise* the range — and how they revise it is itself a signal the whole market reads.

### Revising the range up or down

If the book is filling fast with high-quality orders well above the midpoint, the bank files an amended prospectus **raising** the range — say from `\$18`–`\$22` to `\$23`–`\$26`. This is a loud public signal that the deal is hot, and it tends to *attract more demand* (investors hate to miss a hot deal), which is part of why bankers do it. If the book is thin or priced at the low end, the bank **lowers** the range or, in a weak market, pulls the deal entirely. There is a well-documented asymmetry here that goes by the name the **partial adjustment phenomenon**: IPOs whose price is revised *upward* during bookbuilding tend to have the *biggest* first-day pops. That sounds backwards — if demand was so strong they raised the price, why does it still pop? The answer is at the heart of the underpricing puzzle, and we will get to it.

The word *partial* is the key. When demand comes in strong, the bank moves the price up — but only *part* of the way toward where the demand actually sits. It deliberately stops short. Suppose the book reveals that real, sticky demand would clear at `\$26` on a deal whose range was `\$18`–`\$22`. A bank that adjusts *fully* would price at `\$26` and capture every dollar for the issuer. A bank that adjusts *partially* prices at, say, `\$24` — above the range (so the issuer is delighted) but below the true clearing level (so the stock still pops to `\$26`). That leftover `\$2`-per-share pop is not an accident. It is the payment, and the bigger the upward revision, the bigger the demand that was revealed, the bigger the promised reward — hence the biggest pops cluster on the most upward-revised deals.

#### Worked example: a 10× oversubscribed deal and the allocation haircut

Suppose the `\$400M` deal above (20M shares at `\$20`) ends up **10× oversubscribed** — the book holds orders for 200 million shares against 20 million available. You are a mid-sized fund and you put in an order for **`\$1,000,000`** worth of stock, i.e. 50,000 shares at `\$20`.

In a naïve pro-rata world you would get 1/10th of your order: 5,000 shares, or **`\$100,000`** worth. But allocation is not pro-rata — it is discretionary. The bookrunner ranks accounts by quality. If you are a sticky long-only fund the bank values, you might get **`\$250,000`** (25,000 shares) — far more than your pro-rata share. If you are flagged as a flipper, you might get **`\$20,000`** (1,000 shares) or zero. The same `\$1M` order produces wildly different fills depending on *who you are*. *Oversubscription does not just set the price; it hands the bank a rationing decision worth real money — and the bank spends that scarcity on relationships.*

### Oversubscription, the greenshoe, and the final pricing call

A healthy IPO is oversubscribed — the book holds orders for several times the shares on offer. Oversubscription is what gives the bank confidence the stock won't break its offer price, and it is the raw material for allocation discretion. To manage it, almost every IPO carries a **greenshoe** (formally an over-allotment option, named after the Green Shoe Manufacturing Company that first used it): the underwriters can sell up to ~15% *more* shares than planned, and they have an option to buy those extra shares from the company at the offer price. The greenshoe lets the bank meet stronger-than-expected demand and, through a mechanism called the **stabilization bid**, support the stock if it sags after listing. (The mechanics of the greenshoe and stabilization sit with [underwriting](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk); here it matters only as a tool for absorbing oversubscription.)

On **pricing night** — usually the evening before the first trade — the lead bankers, the company's board, and sometimes its existing investors gather (often by phone) and make the final call. They look at the book, the quality of the demand, the tone of the market that day, and they pick a number. It is a negotiation, not a formula. The company wants the highest price; the bankers want a price that pops enough to reward the buyers they need for the *next* deal. The number they settle on is the IPO price, and the next morning the shares start trading.

## The allocation decision: discretion, favours, and the abuses that drew regulation

The single most underappreciated power the bookrunner holds is not setting the price — it is deciding **who gets the shares**. In an oversubscribed deal, allocation is a gift: the recipient gets to buy a stock that is about to jump. The bank decides, account by account, who receives that gift. This discretion is defensible in theory — rewarding sticky, price-forming investors makes future deals work — but it sits one short step away from outright corruption, and the history of IPO allocation is a history of that step being taken.

![Graph of allocation discretion branching into favoured clients, spinning and laddering, and the regulation that followed](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-4.png)

### Why hot deals go to favoured clients

The clean justification: bookrunners allocate to investors who provide *price discovery* and *aftermarket support*. A fund that submits an honest, detailed step order, that shows up for every deal, and that holds rather than flips is doing the bookrunner a service. Rewarding it with generous allocations on hot deals is how the bank pays for that service in a currency that never shows up on an invoice. Seen this way, allocation is the grease that makes the whole bookbuilding machine run: investors tell the truth about demand because honesty earns allocations, and allocations are worth telling the truth for.

The problem is that the same discretion can be sold for things that have nothing to do with price discovery.

### Spinning and laddering: when the favour becomes a kickback

Two specific abuses got names and got regulated:

- **Spinning** is allocating hot IPO shares to the *personal* brokerage accounts of executives at companies the bank wants future business from. The executive gets handed a near-guaranteed first-day profit; the bank gets the executive's company's next M&A mandate or follow-on offering. It is a bribe paid in IPO allocations. The practice was rampant during the dot-com boom — investigations after 2000 found bank executives keeping lists of corporate VIPs and steering allocations to them — and it was a central charge in the 2003 **Global Settlement** between regulators and ten major Wall Street banks.
- **Laddering** is tying an allocation to a promise: *you can have shares in the IPO, but only if you agree to buy more in the open market afterward at higher prices.* This manufactures artificial aftermarket demand, props up the stock's pop, and lets early allocants sell into the engineered rally. It is straightforward market manipulation dressed up as allocation policy.

The regulatory response hardened over the 2000s. **FINRA Rule 5130** restricts allocating IPOs to industry insiders (brokers, fund managers' own accounts) who could front-run the public. **FINRA Rule 5131** specifically bans spinning — allocating to executive officers and directors of companies in a position to send the bank investment-banking business — and bans quid-pro-quo allocations and certain "flipping" penalties. The Global Settlement reshaped how research and banking interact. None of this eliminated discretion; bookrunners still allocate by hand and still favour their best clients. It drew a line around the *kinds* of favours that are legal.

#### Worked example: what the allocation is worth on a hot deal

Take a deal priced at `\$20` that pops 30% to `\$26` on day one. A favoured fund that receives an allocation of **500,000 shares** has, on paper, made:

- 500,000 × (`\$26` − `\$20`) = 500,000 × `\$6` = **`\$3,000,000`** of first-day gain.

That `\$3M` is the value of the *allocation itself*, before the fund has done any analysis or taken any meaningful risk — it is the gift the bookrunner handed out. Multiply that across dozens of hot deals a year and you see why allocation discretion is the most valuable, most fought-over, and most abused lever in the primary market. *The reason allocation gets regulated is simple arithmetic: in an underpriced IPO, the right to buy at the offer price is worth millions, and someone is deciding who gets it.*

That phrase — *underpriced IPO* — is the hinge of the whole subject. Why is there a `\$6` gap to hand out at all? Why doesn't the bank just price the deal at `\$26` and keep the `\$3M` for the company? That is the underpricing puzzle, and it is one of the most studied anomalies in all of finance.

## The underpricing puzzle: why IPOs pop on day one

Here is the empirical fact that launched a thousand academic papers: across decades and across countries, IPOs *systematically* close their first trading day above their offer price. In the US the median first-day return — "the pop" — has run in the double digits for most of the last forty years, spiking to 40%+ in frothy periods like 1999–2000 and 2020–2021. This is not a few lucky deals; it is the central tendency. Companies are, on average, selling their shares for meaningfully less than the market will immediately pay.

![Median first-day IPO return by year showing the pop over time](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-5.png)

The pop is not free. Every dollar of first-day jump is a dollar the company *could* have raised and did not. Economists call it **money left on the table**, and it is large.

#### Worked example: the pop as money left on the table

A company raises **`\$500M`** by selling 25 million new shares at **`\$20`** each. The stock pops **30%** on day one to **`\$26`**.

- Money left on the table = shares sold × (first-day price − offer price) = 25,000,000 × (`\$26` − `\$20`) = 25,000,000 × `\$6` = **`\$150,000,000`**.

The company raised `\$500M` but *could* have raised `\$650M` for the same 25 million shares if it had priced at `\$26`. That `\$150M` went straight to the first-day buyers — the institutions the bookrunner allocated to. To put the magnitude in perspective: the underwriting fee on a `\$500M` IPO is typically around 7%, or `\$35M`. The money left on the table — `\$150M` — was *more than four times the explicit fee*. The biggest cost of going public is usually not the bankers' fee; it is the underpricing. *The pop is a transfer from the company's existing owners to the new buyers, and it dwarfs the line-item fees everyone complains about.*

So why does it happen? Why do issuers tolerate it, deal after deal? There are several explanations, and the honest answer is that they all contribute.

### Explanation 1: the winner's curse and information asymmetry

This is the foundational theory (Rock, 1986), and it is the most important to understand. Split the IPO buyers into two camps: **informed** investors who can tell a good deal from a bad one, and **uninformed** investors who cannot. Now follow what happens to an uninformed investor across many deals:

- On **good deals** (underpriced, will pop), the informed investors pile in, the deal is oversubscribed, and the uninformed investor gets *rationed* — a small allocation of the good stuff.
- On **bad deals** (overpriced, will sink), the informed investors stay away, so the uninformed investor's order is filled *in full* — they get all of the bad stuff they asked for.

This is the **winner's curse**: an uninformed investor systematically ends up with a small slice of the winners and a full helping of the losers. Across all deals, that adverse selection would leave them with negative returns — so they would simply stop buying IPOs. But the primary market *needs* uninformed money to clear the deals. The only way to keep uninformed investors in the game is to underprice IPOs *on average* by enough that even their cursed, adversely-selected portfolio earns a fair return. The pop is the compensation the market pays uninformed capital for the winner's curse it suffers. Underpricing is not a bug; it is the price of keeping the buyer base broad enough to absorb the bad deals along with the good.

### Explanation 2: bookbuilding theory — paying for the truth

The winner's curse explains *average* underpricing. **Bookbuilding theory** (Benveniste and Spindt, 1989) explains the *pattern* — why the deals that get revised *up* still pop the most. Recall the central tension: investors have no incentive to reveal honestly how much they would pay, because honesty raises the price they pay. The bank needs that honesty to set the price well. How do you buy truthful information from someone who is hurt by telling the truth?

You pay them. Specifically, the bank promises: *the more bullish and detailed your IOI, the better your allocation will be — and we'll price the deal low enough that a good allocation is worth having.* An investor who reveals strong demand drives the price up (bad for them on the shares they get) but earns a bigger allocation of an underpriced stock (good for them). The bank deliberately *partially adjusts* the price upward — moving it toward the strong demand but not all the way — so that the investors who revealed the strong demand are left with a profit. That residual profit is the payment for their information. This is exactly why upward-revised deals pop the most: the upward revision is the *signal* that investors revealed strong demand, and the pop is the *reward* the bank promised them for doing so. Underpricing here is a deliberate payment for price discovery, baked into the mechanism.

### Explanation 3: the issuer's tolerance and the banker's incentives

Why do company founders and existing owners — who are giving away that `\$150M` — put up with it? A few reasons, none flattering and all real:

- **The owners are usually selling only a slice.** A founder who keeps 80% of the company after the IPO cares far more about the price of their *remaining* stake than about squeezing the last dollar out of the 20% they sold. A strong pop creates buzz, analyst coverage, and a rising stock — which helps the 80% they still hold. Leaving money on the table on the sold shares can look like a marketing expense.
- **Risk aversion and the fear of a broken deal.** A deal priced too aggressively that *breaks* below its offer price is a disaster — embarrassing, damaging to the stock's reputation, hard to recover from. Pricing conservatively buys insurance against that catastrophe. Issuers, advised by bankers who hate broken deals, choose the safe side.
- **The bankers' incentives are not aligned with the issuer.** This is the uncomfortable part. The bookrunner earns a fee that is a *percentage of proceeds* — so a higher price means a bigger fee, which argues for pricing high. But the bookrunner *also* has a roomful of buy-side clients it sells dozens of deals to every year, and those clients want IPOs to pop. The bank's franchise depends on keeping its best buy-side accounts happy far more than on squeezing one issuer for an extra `\$10M` of fee. So the bank's true incentive tilts toward *underpricing* — toward handing its repeat-customer institutions a reliable profit, at the one-time issuer's expense. The issuer shows up once; the institutions show up every week.

Put those three together and underpricing becomes overdetermined: the winner's curse requires it on average, bookbuilding theory builds it into the information-extraction mechanism, founders tolerate it because they keep most of the stock and fear a broken deal, and the bankers quietly prefer it because their bread is buttered on the buy side. No single actor is villainous; the *structure* produces the pop.

### The lockup: why the pop is not pure profit for everyone

There is a mechanism that complicates the "money left on the table" story, and it is worth naming because it shapes who actually captures the pop. Almost every IPO comes with a **lockup**: insiders — founders, employees, pre-IPO investors — agree not to sell their shares for a set period after listing, typically 90 to 180 days. The point is to prevent a flood of insider selling from crushing the freshly-listed stock while the market is still figuring out its price. The consequence for our story is sharp: the people who *give away* the money-left-on-the-table (the existing owners, on the shares the company sold) cannot themselves capture the day-one pop, because they are locked up. The day-one pop accrues to the *new* buyers — the institutions the bookrunner allocated to — who face no lockup on their IPO allocation and can flip into the opening jump. So the pop is doubly a transfer: from the company and its locked-up insiders, *to* the unlocked new institutional buyers. This is one more reason allocation discretion is so valuable and so fought over: an allocation is not just a chance to buy a popping stock, it is an *unlocked* chance, while the insiders who created the company watch the first-day gain from the sidelines.

### Going public is more expensive than the fee suggests

Stacking up the costs of an IPO makes the underpricing point concrete. On a `\$500M` raise, an issuer pays roughly: a 7% underwriting spread (`\$35M`); legal, accounting, and listing fees (often `\$5M`–`\$10M`); and the underpricing itself (`\$150M` in our 30%-pop example). The largest cost by far — bigger than every explicit fee combined and several times over — is the invisible one: the gap between the offer price and where the stock actually trades. Founders and the financial press fixate on the 7% spread because it appears on an invoice. The `\$150M` does not appear anywhere; it is simply value that the company's owners transferred to new buyers by selling too cheap. Understanding bookbuilding is, more than anything, understanding *where that `\$150M` goes and why the system is built to send it there*.

## Alternatives to bookbuilding: auctions, direct listings, and SPACs

If bookbuilding leaves so much money on the table, why not replace it? People have tried. Each alternative changes *who sets the price* and *whether new capital is raised*, and each has its own reason for staying rare.

![Matrix comparing bookbuilding, Dutch auction, direct listing and SPAC on who sets the price and whether new capital is raised](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-6.png)

### The Dutch auction: let the market clear the price

A **Dutch auction** removes the bank's discretion and lets a transparent auction set the price. Investors submit sealed bids — a quantity and a price. The shares are filled from the highest bid down until they run out, and *everyone pays the same clearing price*: the lowest accepted bid (the price at which the last share sells). In principle this captures the full demand curve and prices the deal exactly at market-clearing, leaving little money on the table — the auction, not a banker, finds the price.

Google's 2004 IPO is the famous case. Google ran a modified Dutch auction explicitly to reduce underpricing and to democratize access (anyone could bid, not just favoured institutions). It priced at `\$85` and still popped about 18% on day one — less than a typical hot tech deal of the era, but not zero. The auction worked, more or less, and yet auctions never took over. Why?

- **Bankers dislike them** — they strip out the allocation discretion that is the bank's most valuable lever, and they shrink the underwriting role to little more than logistics.
- **The winner's curse bites bidders harder.** In an auction with no bank to ration, sophisticated bidders fear that if they win their *full* bid, it is because the deal is overpriced (the bad-deal scenario). So they shade their bids down to protect against the curse — which can *lower* the clearing price and reintroduce underpricing through the back door.
- **No information-gathering mechanism.** Bookbuilding's whole apparatus for extracting honest demand (allocations as payment for truth) vanishes; the auction just takes bids at face value, which can be noisier.
- **Issuers feared the unknown.** Founders watching their once-in-a-lifetime listing preferred the bankers' reassuring hand-holding to an open auction's uncertainty, especially the risk of a weak clearing price.

So the Dutch auction stayed a curiosity. It solves underpricing in theory and partially in practice, but it removes the features that make the players want to participate.

#### Worked example: a Dutch-auction clearing price

A company auctions **10 million** shares. The sealed bids come in (cumulative, highest price first):

- 3 million shares bid at `\$24` or higher.
- 7 million shares bid at `\$22` or higher (the 3M at `\$24` plus 4M more willing at `\$22`).
- 14 million shares bid at `\$20` or higher (another 7M appear at `\$20`).

Fill from the top down until the 10 million shares are gone. At `\$22`, cumulative demand is 7 million — not enough. At `\$20`, cumulative demand is 14 million — more than the 10 million on offer. So the auction clears at **`\$20`**: the lowest price needed to sell all 10 million shares. Every winning bidder — including the ones who bid `\$24` — pays the single clearing price of `\$20`. The company raises 10M × `\$20` = **`\$200M`**, and there is no banker deciding who gets what. *A Dutch auction makes the demand curve do the pricing directly, which is its strength and, for the people who profit from discretion, its weakness.*

### Direct listings: no new capital, a reference price, pure secondary trading

A **direct listing** is radically simpler: the company lists its existing shares on an exchange and lets them trade, *without selling any new shares and without raising any new capital*. There is no offer price and no allocation — existing shareholders (employees, early investors) simply become free to sell, and public buyers meet them on the exchange. Instead of an IPO price, the exchange publishes a non-binding **reference price** the night before — a rough anchor based on recent private-market trades — and the stock opens wherever the opening auction on the exchange clears.

Spotify pioneered this in April 2018, and Slack and Coinbase followed. The appeal: no underpricing (there is no offer price to be popped *from* — the first trade *is* the price), no lockup forcing insiders to wait, no banker allocation, and far lower fees. The catch is decisive: a direct listing **raises no capital for the company**. It is a way to *get liquidity for existing holders*, not a way to fund the business. That makes it viable only for companies that are already cash-rich and famous enough not to need the IPO's marketing machine — and not at all for the typical growth company whose whole point in going public is to raise money. (The SEC has since allowed "direct listings with a capital raise" that bolt a primary sale onto the mechanism, but adoption has been slow.) A direct listing is best understood as the secondary market opening for business in a security that skipped the primary-market issuance step entirely.

### SPACs: price discovery by negotiation

A **SPAC** — special purpose acquisition company — is a third route, and a stranger one. A SPAC is a shell company with no business: it raises cash in its *own* IPO (usually at the round number of `\$10` a unit), parks the money in a trust, and then goes looking for a private company to merge with. When it finds one, the private company effectively goes public by merging into the listed shell — a "de-SPAC." The price is discovered not by a book or an auction but by **bilateral negotiation** between the SPAC's sponsors and the target company, the way a private M&A deal is priced, with a valuation agreed in the merger terms.

SPACs surged in 2020–2021 as a faster, more flexible route to public markets — a target could negotiate a valuation privately and skip the roadshow gauntlet. But the price-discovery quality is weak: the valuation is set by two parties with aligned incentives to agree on a high number (the sponsor gets paid only if the deal closes), and crucially, SPAC investors can **redeem** their `\$10` back from the trust rather than stay in the deal. When redemptions are heavy — as they were across most 2021 SPACs once the froth turned — the company ends up with far less cash than the headline trust value, and the post-merger stock often craters. SPACs proved that you *can* take a company public without bookbuilding, but they also proved why genuine price discovery matters: a negotiated price between aligned insiders, with an escape hatch for everyone else, is not the same as a market clearing.

## Common misconceptions

**"The IPO price is the company's fair value."** No. The IPO price is the *clearing price of a controlled auction on a single night*, deliberately set below where the bankers expect the stock to trade so the book clears with room to spare. The first-day pop is direct evidence the offer price was *not* the market-clearing value. Fair value is what equity research estimates; the IPO price is a transaction price engineered to make the deal work.

**"A bigger pop means a more successful IPO."** For the *buyers* who got an allocation, yes. For the *company*, a giant pop is a giant transfer of its money to those buyers — `\$150M` in the worked example above. A "successful" IPO from the issuer's standpoint is one that prices high *and* trades up modestly; a 100% pop means the deal was mispriced by half and the founders gave away an enormous sum. The financial press celebrates pops; CFOs should grimace at them.

**"Allocation is fair and pro-rata."** It is neither. On a 10× oversubscribed deal, allocation is entirely discretionary — the bookrunner decides account by account, favouring sticky long-only clients and its franchise relationships. Two identical `\$1M` orders can receive `\$250,000` and `\$20,000` based purely on *who* is asking. The discretion is the point, and it is what spinning and laddering abused.

**"Market orders show the strongest demand."** The opposite. A price-insensitive market order tells the bookrunner nothing about *where* demand thins out, and it is exactly what flippers submit to grab an allocation they intend to sell into the open. The most valuable order in the book is a detailed *limit* or *step* order from a long-only fund — it reveals a real point on the demand curve and signals sticky, price-forming money.

**"Dutch auctions and direct listings eliminate underpricing, so they should win."** They reduce the *cost* of underpricing but remove the *features* that make the players participate — the allocation discretion bankers prize, the information-extraction mechanism that prices the deal well, and the hand-holding nervous founders want. And direct listings raise no new capital at all. Mechanisms persist not because they are efficient but because they are *incentive-compatible for everyone whose cooperation they need*. Bookbuilding's "inefficiency" is the glue.

## How it shows up in real markets

**Google, August 2004 — the auction that proved the point.** Google deliberately ran a modified Dutch auction to cut underpricing and broaden access, even cutting its expected range from `\$108`–`\$135` down to `\$85` as demand came in softer than hoped and regulatory hiccups piled up. It priced at `\$85`, raised about `\$1.7bn`, and still rose roughly 18% on the first day. The lesson cut both ways: the auction *did* compress the pop relative to a typical hot tech deal, vindicating the theory — but it also showed that even an auction does not drive the pop to zero (bidders shade for the winner's curse), and the messy process scared other issuers off the auction route. Twenty years later, near-zero IPOs use it.

**The dot-com spinning scandals and the 2003 Global Settlement.** During 1999–2000, with first-day pops routinely exceeding 50% and sometimes topping 100%, the right to an IPO allocation was worth a fortune, and banks used it as currency. Investigations revealed banks steering hot allocations into the personal accounts of telecom and tech executives whose companies the banks wanted as banking clients — textbook spinning — and tying allocations to aftermarket buying — laddering. The fallout produced the `\$1.4bn` Global Settlement with ten major banks, the build-out of FINRA Rules 5130 and 5131, and a permanent tightening of how allocation discretion may be used. The discretion survived; the kickbacks were outlawed.

**Spotify, April 2018 — the direct listing arrives.** Spotify skipped bookbuilding entirely. The NYSE published a reference price of `\$132`; the stock opened, after an extended opening auction, around `\$165` and closed near `\$149`. No new shares were sold, no capital was raised, no banker allocated anything — existing holders simply became free to sell. Spotify could do this because it was famous, cash-generative, and did not *need* IPO proceeds. It demonstrated a real alternative for a narrow class of companies, and Slack and Coinbase followed the template.

**The 2021 boom and the 2022 bust.** The IPO window is brutally cyclical, and bookbuilding's dependence on hot demand makes it so. 2021 saw a historic surge — US traditional-IPO proceeds spiked and the deal count roughly doubled — riding the same froth that produced 30%+ median pops and a SPAC mania. Then in 2022 the window slammed shut: rising rates and falling valuations cut US IPO proceeds to a trickle, and SPAC after SPAC saw mass redemptions that left de-SPACed companies with a fraction of their headline cash and a collapsing stock. The episode is the clearest illustration of the series' spine: **secondary-market conditions govern primary-market issuance.** When the secondary market is buoyant and liquid, books fill, deals price, and companies raise capital; when it sours, the primary market goes dark — nobody underwrites a deal they cannot sell.

![US IPO proceeds by year showing the 2021 boom and 2022 collapse](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-7.png)

The deal *count* tells the same story as the dollar proceeds — the number of companies that can get a book built collapses and rebounds with the window, because bookbuilding only works when there is hot, liquid demand to build a book *from*.

![Number of US IPOs by year showing the cyclical issuance window](/imgs/blogs/bookbuilding-and-price-discovery-how-the-ipo-price-is-set-8.png)

Both charts make the same point in two units: issuance is not steady. It comes in windows that open when the secondary market is hungry and slam shut when it is fearful. A bank can build the most beautiful book in the world, but if the secondary market won't absorb the shares the next morning, there is no deal to price.

## The takeaway: a price is a negotiated truce, not a calculation

The deepest thing to take from bookbuilding is that the IPO price — the most important number in a company's financial life — is not computed. It is *negotiated, signalled, and engineered*, by a banker reading a book of strategically dishonest orders and choosing a number designed to clear the deal with a deliberate cushion. Every odd feature of the process — the underpricing, the allocation discretion, the favours, the auctions that never caught on — falls out of one fact: **before the first trade there is no price, so someone has to manufacture one from opinions, and everyone whose opinion you need has a reason to shade it.**

That reframes how to read every IPO headline. When a stock "pops 40% on its debut," that is not a triumph for the company — it is the company having handed roughly 40% of the day's float value to the institutions the bookrunner chose to favour, in exchange for their cooperation in building the book and their willingness to buy the next deal. When a company runs an auction or a direct listing, it is choosing to keep more of that value at the cost of the bank's reassuring machine. And when the IPO window slams shut in a downturn, it is the secondary market reminding the primary market who is really in charge: **you can only issue a security if someone is confident they can sell it tomorrow.** The entire bookbuilding apparatus — the roadshow, the book, the allocation, even the underpricing — exists to convince that someone, the night before the first trade, that the answer is yes.

That is the primary market's core trick, and bookbuilding is how the trick is performed: it does not find the true price; it finds a price low enough that the book clears, the buyers profit, and the secondary market opens with a willing crowd. The "money left on the table" is not a flaw in the machine. It is the fee the primary market pays the secondary market for showing up.

So the next time you read that a company "raised `\$500M` in its IPO and the stock soared 30%," translate it: the company raised `\$500M`, handed roughly `\$150M` of additional value to the institutions the bank chose, and bought itself a liquid, willing crowd of secondary-market buyers in return. Whether that was a triumph or a giveaway depends entirely on whose seat you are sitting in — and bookbuilding is the negotiation that decides the split. Understanding it is understanding the precise moment a private idea becomes a public, tradeable price, and who pays what for the privilege of being there when it does.

## Further reading & cross-links

- [The IPO process end to end: from mandate to first trade](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade) — the full timeline this post zooms into; bookbuilding is the pricing stage of that arc.
- [Underwriting and the syndicate: who takes the risk](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk) — who buys the shares from the company, who bears the risk between pricing and selling, and how the greenshoe and stabilization work.
- [Beyond the IPO: follow-ons, rights issues, and private placements](/blog/trading/capital-markets/beyond-the-ipo-follow-ons-rights-issues-and-private-placements) — how companies raise capital *after* they are already public, where price discovery is easier because a market price already exists.
- [How a price is made: discovery, arbitrage, and efficiency](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency) — the secondary-market sequel: how prices keep getting discovered every second once trading begins.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the bookrunner's business model, including why the buy-side relationship outweighs any single issuer's fee.
