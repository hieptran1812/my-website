---
title: "How a Bond Is Issued: Auctions, Syndication, and the Deal"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How governments auction debt and how companies syndicate it — competitive bids, stop-out yields, order books, and the new-issue concession that gets a bond sold."
tags: ["capital-markets", "primary-market", "bond-issuance", "treasury-auctions", "syndication", "new-issue-concession", "primary-dealers", "order-book", "fixed-income", "underwriting"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A bond is sold into existence one of two ways: governments mostly **auction** it (bidders compete on yield, and a single clearing price falls out of the demand), while companies and sovereigns mostly **syndicate** it (banks build an order book, tighten the price as demand shows up, then allocate).
>
> - In a **single-price auction**, every winner pays the same **stop-out yield** — the highest yield needed to sell the whole issue — and the **bid-to-cover ratio** tells you how hungry demand was.
> - In a **syndicated deal**, the price walks from wide **initial price talk (IPT)** down to a tight final spread as the book fills; a 3x-covered book might tighten 20bp from where it started.
> - A new bond almost always prices a few basis points **cheap** to the issuer's existing curve — the **new-issue concession** — because investors need a reward to swap out of bonds they already own.
> - **One number to remember:** the US Treasury issued roughly **\$23 trillion** of debt in 2023 — more than every corporate, mortgage, muni, and asset-backed bond *combined*. The auction machine is the single biggest funding event on earth, run on autopilot, every week.

On a normal Wednesday at 1:00 p.m. New York time, the United States borrows tens of billions of dollars in about ninety seconds. There is no roadshow, no press conference, no banker working the phones. A clock on the Treasury's auction system ticks to zero, the system locks the bids that have been submitted, sorts them, and prints a single number: the yield at which the auction "stopped." Within two minutes that number is on every trading desk in the world, and the world's largest borrower has refinanced a chunk of the national debt before most people are back from lunch.

That quiet, repeatable, almost boring efficiency is the whole point. The same week, a few blocks away, a different kind of debt sale is unfolding with the opposite texture: a company wants to borrow a billion dollars for ten years, and a small army of bankers spends the morning on the phone, sending messages to a hundred investors, nudging a price tighter and tighter as orders pile up, until at lunchtime they "launch" the deal at a spread that did not exist when the day started. One issuance is a machine; the other is a negotiation. This post is about both — and about why each method exists, who bears the risk in each, and what the price of a brand-new bond is really telling you.

This is a post in a series about **capital markets** — the machine that turns savings into long-term investment. That machine has two engines: a **primary market** that *creates* securities to raise capital, and a **secondary market** that *trades* them to provide liquidity. Bond issuance is the primary market doing its job. And the deep secret that makes it all work shows up here in a very concrete way: nobody bids in a Treasury auction, and nobody puts an order into a corporate book, unless they are confident they can *sell that bond tomorrow morning*. Secondary-market liquidity is the precondition for primary issuance. We will keep returning to that.

A note on scope. This post owns the **mechanics of selling debt** — how the auction clears, how the book is built, how the price is set. It does **not** re-derive how a bond is *priced* in the first place (coupons, yield, duration, the discount math). For that, see [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance). We will lean on those ideas; we will not re-prove them.

![Single-price auction with bids stacked into a staircase and the size line landing on the stop-out yield](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-1.png)

## Foundations: what "issuing a bond" actually means

Start with the everyday version. Suppose you want to borrow \$10,000 from friends to open a coffee cart, and you promise to pay it back in three years with a little interest each year. You could do this two ways. You could go to one friend who has \$10,000 and negotiate the whole thing privately. Or you could chop the loan into a hundred \$100 slips of paper — each one a promise "I will pay the holder of this slip \$100 in three years, plus \$5 a year until then" — and sell the slips to a hundred different people. Each slip is a **bond**: a tradeable IOU. The act of creating those slips and selling them for cash is **issuance**.

A few definitions we will use throughout, built from zero:

- A **bond** is a security that represents a *loan*. The buyer (the lender, the investor) hands over cash today; the issuer (the borrower) promises a stream of fixed payments — usually periodic **coupons** plus the **face value** (also called **par**, typically \$1,000 per bond) at **maturity**. Unlike a stock, a bond does not give you ownership; it gives you a contractual claim to be repaid. (Why a company might choose to sell bonds rather than shares is the subject of [debt vs equity](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital).)
- The **issuer** is whoever is borrowing: a government (the US Treasury, the UK's DMO, Vietnam's State Treasury), a company (Apple, a utility, a bank), or a supranational (the World Bank).
- The **primary market** is where the bond is *born* — sold by the issuer to its first owners, with the cash going *to the issuer*. The **secondary market** is where those first owners later trade the bond among themselves, with cash going from one investor to another and *nothing* going to the issuer. This post lives almost entirely in the primary market.
- **Yield** is the return an investor earns if they buy the bond at its current price and hold it. The crucial fact for issuance: price and yield move in *opposite* directions. A higher price means a lower yield; a lower price means a higher yield. So when you read "the auction stopped at a higher yield than expected," that means the bonds sold for a *lower* price than expected — a slightly worse outcome for the issuer.
- A **basis point (bp)** is one hundredth of a percentage point. 0.01% = 1bp. 0.50% = 50bp. Bond people speak almost entirely in basis points because the differences that matter are tiny: a 5bp move on a 10-year Treasury changes the price by roughly 40 cents per \$100 of face. We will use bp constantly.

The reason issuance is its own craft — rather than just "post a price and see who buys" — is that the issuer faces a genuine dilemma every single time. Price the bond too *cheap* (too high a yield) and you have left money on the table: you are paying more interest than you needed to, for decades. Price it too *rich* (too low a yield) and nobody buys, the deal fails, your name is mud with investors, and your next deal costs you more. The entire apparatus of auctions and syndication exists to solve that one problem: **discover the highest price (lowest yield) at which the whole issue will actually clear.** Hold that thought; it is the thread through everything below.

One more foundational point, because it shapes how a bond can be sold: a bond is not merely a yield and a maturity, it is a *legal contract*, and that contract is called the **indenture**. The indenture is the (often hundreds of pages long) document that spells out every promise the issuer is making — the coupon, the payment dates, the maturity, the seniority of the claim, and the **covenants**: the rules the borrower agrees to live by while the bond is outstanding. Covenants come in two flavors. *Incurrence* covenants bite only when the issuer takes an action (for example, "you may not take on new debt that pushes leverage above 4x"); *maintenance* covenants must be satisfied at all times (for example, "you must keep interest coverage above 2x, tested every quarter"), and are far stricter — they are common in bank loans and rare in public bonds. Because no investor reads the whole indenture and no investor trusts the issuer to police itself, the contract names an independent **trustee** (a specialized bank department) to hold the deal documents, collect and pass on the coupon payments, monitor the covenants, and — if the issuer trips a covenant or misses a payment — declare an **event of default** and act for the bondholders as a group. The standardization of these terms is exactly what lets a stranger buy the bond in the secondary market without re-negotiating anything: the contract travels with the security. When a deal is announced as "10-year senior unsecured," that single phrase is shorthand for a whole indenture the buyer already understands.

![Tree of three bond-issuance channels: auction for governments, syndication for corporates and sovereigns, and private or programmatic routes](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-3.png)

There are, broadly, three channels, shown above. **Auctions** are how governments sell the bulk of their debt: standardized, frequent, mechanical. **Syndication** is how companies and most sovereigns-in-foreign-currency sell theirs: bespoke, relationship-driven, a single deal at a time. And a set of **private and programmatic routes** — private placements, medium-term-note programs, shelf registrations — handle everything that does not fit the first two. We will take them in that order, spend the most time on the first two, and end on what the price of a new bond is really telling you.

## Government bond auctions: how the world's biggest borrower sells debt

Governments are *serial* issuers. The US Treasury does not raise money once a year with a big splashy deal — it raises money almost every business day, rolling over maturing debt and funding the deficit in a relentless stream. In 2023 the Treasury issued on the order of \$23 trillion of securities (gross, counting the constant rollover of short-term bills). Look at how that dwarfs everything else in the US bond market:

![US bond issuance by type in 2023, with Treasury far larger than corporate, mortgage, muni, agency, and ABS](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-2.png)

When you issue that much, that often, you cannot afford a bespoke negotiation each time. You need a *protocol* — a fixed, transparent, repeatable procedure that anyone can participate in and that produces a fair price without a banker in the middle. That protocol is the **auction**.

### The published calendar and the standing army that must show up

The first thing to understand is that a Treasury auction is *boring on purpose*. Weeks ahead, the Treasury publishes a **financing calendar**: which securities (bills, notes, bonds, TIPS) will be auctioned, on which days, in roughly what size. There are no surprises about *when* — surprises happen only in the *price*. This predictability is itself a feature: it lets the market prepare, it spreads demand evenly, and it removes the temptation for the issuer to "time the market," which serial borrowers cannot do anyway.

The second thing is the **primary dealers**. These are a couple of dozen large banks and broker-dealers (roughly 24 in the US, names like JPMorgan, Citigroup, Goldman Sachs, Nomura, Barclays) that the New York Fed designates as its trading counterparties. In exchange for that status — and the business that comes with it — a primary dealer takes on an *obligation*: it must bid in **every** Treasury auction, for at least its **pro-rata share** of the offering, at a "reasonable" price. They are the buyers of last resort. Even if real-money demand is soft on a given day, the dealers are contractually on the hook to absorb the paper. This is the structural reason a Treasury auction essentially *cannot fail*: there is always a floor of obligated bidders underneath it.

![Treasury auction cycle pipeline from calendar to when-issued trading to dealer bidding to the single-price auction to stop-out and settlement](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-8.png)

Between the announcement and the auction, the bond trades before it even exists. This is **when-issued (WI)** trading — a "grey market" in the not-yet-auctioned security. Dealers and investors buy and sell the bond on a *when-issued* basis (settlement contingent on the auction actually happening), and the WI price gives everyone a real-time read on where demand is. By auction time, the market already has a tight consensus on the likely yield; the auction's job is to *confirm and crystallize* it, not to discover it from a blank slate. The WI market is the secondary-market-makes-primary-possible principle in miniature: people will commit to buy a bond that does not yet exist precisely because they know they can trade it the instant it does.

### Competitive vs non-competitive bids

There are two kinds of bid you can submit.

A **competitive bid** says: "I want \$X of this bond, but *only if the yield is at least Y*." You are stating both a size and a price (expressed as a yield). You are a price-sensitive, professional buyer — a dealer, a hedge fund, a large asset manager — and you are willing to walk away if the auction prices too rich for you.

A **non-competitive bid** says: "I want \$X of this bond, and I will take whatever yield the auction produces." You are not trying to call the price; you just want the bonds. This is the channel for small and retail buyers (and for some foreign central banks). In the US, non-competitive bids are capped (currently \$10 million per bidder per auction) and — this is the kind part — they are filled *first*, in full, at the auction's clearing yield, right off the top of the offering. You never have to worry about being "too aggressive" and getting shut out; you simply receive the same yield the big professional bidders set.

#### Worked example: a non-competitive bid getting filled

Suppose the Treasury is auctioning a 10-year note and you, a retail investor, submit a non-competitive bid for \$5,000 of face value through TreasuryDirect. You do not specify a yield. When the auction closes, the competitive bidding (which we will walk through next) sets a clearing yield of, say, **4.26%**. Your bid is filled in full: you get \$5,000 of face value of the new note, and your yield is exactly 4.26% — the same yield JPMorgan's billion-dollar competitive bid received. If the note carries a 4.25% coupon, you would pay just a hair under par for it (a 4.26% yield on a 4.25% coupon means a tiny discount to the \$5,000 face, on the order of \$5,000 × (1 − 0.0008) ≈ \$4,996). **The intuition:** non-competitive bidders are passengers, not drivers — they ride along at whatever fare the professional bidders negotiate, and they never get bumped.

### The single-price (Dutch) auction and the stop-out yield

Now the heart of it. How does the competitive bidding actually set the price?

Picture every competitive bid as a rung on a ladder, sorted from the *lowest* yield (the most aggressive bidder, willing to accept the least return, i.e. willing to pay the most) up to the *highest* yield (the cheapest bidder, demanding the most return). The Treasury fills bids starting from the bottom of that ladder — accepting the lowest-yield bids first, because those are the *best* deals for the taxpayer — and keeps climbing the ladder, accepting bid after bid, until the cumulative amount accepted equals the size of the offering. The yield of the *last* bid it needs to accept to sell the whole issue is the **stop-out yield** (also called the *high yield*, because it is the highest accepted yield). The auction "stops" there.

Here is the crucial design choice. In the modern US system, *every* winning bidder — including the aggressive ones who bid at much lower yields — pays the *same* price, corresponding to that single stop-out yield. That is the **single-price**, or **uniform-price**, or **Dutch** auction. The figure at the very top of this post shows it: the bids stack into a staircase, the size line slides in from the right, and wherever it lands sets one clearing yield that everyone pays.

This was not always how it worked. Until the early 1990s, the Treasury ran a **multiple-price** (also called *discriminatory* or *pay-your-bid*) auction: each winning bidder paid the price corresponding to *their own bid*. If you bid aggressively at a low yield, you paid a high price; the cheap bidders at the stop-out paid less. That sounds like it should be *better* for the issuer — why give the aggressive bidders the same good deal as the marginal one? The answer is the **winner's curse**, and it is the single most important idea in auction design.

#### Worked example: a competitive Treasury auction, stacked to the stop-out

Let us run a concrete \$40 billion 10-year note auction. First, \$2 billion of non-competitive bids come in; they are filled off the top, leaving \$38 billion for the competitive bidders. Now the competitive bids, sorted from most aggressive (lowest yield) up:

| Bid yield | Amount bid | Cumulative competitive | Status |
|---|---|---|---|
| 4.20% | \$10bn | \$10bn | filled |
| 4.23% | \$12bn | \$22bn | filled |
| 4.25% | \$14bn | \$36bn | filled |
| 4.26% | \$8bn | \$44bn | **stop-out** (partial) |
| 4.28% | \$10bn | \$54bn | rejected |

We need \$38 billion of competitive bids filled. Walking up the ladder: 4.20% (\$10bn), 4.23% (\$12bn), 4.25% (\$14bn) get us to \$36bn — still \$2bn short. The next rung, 4.26%, has \$8bn bid but we only need \$2bn more, so those 4.26% bidders are filled at a **prorated 25%** (\$2bn / \$8bn). The auction stops at **4.26%** — that is the stop-out yield, and *every* winner, including the 4.20% bidder, gets the bonds at a 4.26% yield. The 4.28% bidder gets nothing.

Now the **bid-to-cover ratio**: total bids received divided by the amount on offer. Total bids = \$2bn (non-comp) + \$10 + \$12 + \$14 + \$8 + \$10 = \$56bn, against a \$40bn offering. Bid-to-cover = \$56bn / \$40bn = **1.40x**. **The intuition:** the stop-out yield is the worst price the issuer accepts to sell everything, and the bid-to-cover ratio (1.40x here, meaning \$1.40 of demand for every \$1 sold) measures how much appetite was standing behind that price — the higher the cover, the stronger the auction.

### Why single-price beats multiple-price: the winner's curse

In a pay-your-bid auction, every bidder faces a nasty asymmetry. If you bid aggressively (a low yield, a high price) and win, you might have *overpaid* — you won precisely because you bid more than everyone else thought the bond was worth. That is the "winner's curse": winning is itself bad news about your bid. Rational bidders protect themselves by **shading** their bids — bidding a little less aggressively than they truly value the bond, to leave a cushion. Everyone shades, so the auction clears at a *worse* price for the issuer than it should.

The single-price auction kills the curse. Because you pay the *clearing* price regardless of how aggressively you bid, there is no penalty for bidding your true value — you can bid exactly what you think the bond is worth, secure that you will never pay more than the marginal clearing price. Less shading means more aggressive bidding means a *better* price for the taxpayer. The US Treasury studied this carefully through the 1990s (after the 1991 Salomon Brothers bid-rigging scandal exposed how gameable the old system was) and converted fully to single-price auctions by 1998. Most major sovereign issuers have followed. The lesson generalizes far beyond bonds: when you want honest bids, make the price uniform.

### Who ends up holding it: dealers, directs, and indirects

About two minutes after the auction closes, the Treasury releases the results — and the part the whole market pounces on is not just the stop-out yield, but the **allotment**: how the bonds were divided among three categories of bidder. **Primary dealers** are the obligated banks we just met. **Direct bidders** are large institutions (asset managers, hedge funds, pension funds) that bid for their own account directly through the Treasury's system rather than routing through a dealer. **Indirect bidders** place their bids through a dealer but on behalf of someone else — a category that captures most foreign demand, including the foreign central banks that bid through the New York Fed. Those three buckets always sum to 100% of the competitive award.

Why does anyone care who got the bonds, as long as they sold? Because the split is the cleanest read on the *quality* of demand. A high **indirect** share signals that real foreign and institutional money showed up hungry and took the paper down — a strong, "sticky" auction whose buyers are unlikely to dump the bonds next week. A high **dealer** share is the opposite warning sign: it means the obligated dealers were left holding the bag, mopping up supply that end investors did not want at the clearing yield. Dealers do not want to sit on inventory, so a high dealer takedown often precedes a few days of the bond drifting cheaper as they work it off. So when a strategist says an auction was "weak," they rarely mean it failed to sell — Treasury auctions essentially never fail — they mean it stopped at a tail, on a soft bid-to-cover, *and* left an outsized chunk with the dealers. Read together, those three numbers turn a routine government financing into a real-time poll on global appetite for US debt.

### Tails: reading a weak auction

When market participants are watching an auction, they compare the stop-out yield to the **when-issued yield** that prevailed *just before* the auction closed. If the auction stops at a *higher* yield than the WI market expected, that gap is called a **tail** — the auction "tailed" by, say, 2bp. A tail means demand was softer than the grey market thought: bidders demanded a little extra yield (a little lower price) to absorb the supply. A "through" (or "stopping through") is the opposite — the auction stops at a *lower* yield than WI, signaling unexpectedly strong demand.

Tails are a real-time sentiment gauge for the entire bond market. A string of tailing Treasury auctions in late 2023 — alongside a low bid-to-cover and a rising share going to the dealers (who must mop up what real-money buyers leave behind) — was read across markets as a warning that investors were getting full on US duration just as the deficit was ballooning. The same yields that every new corporate bond prices off were, in those moments, being set in real time by how the auctions cleared. We will look at that level next.

### The level it all hangs off: the Treasury curve

Every bond on earth is priced *relative to* a risk-free benchmark, and for dollar bonds that benchmark is the US Treasury curve. The auction yields we have been discussing *are* that benchmark being set. Here is the 2-year and 10-year over the rate-hiking cycle:

![Two-year and ten-year US Treasury yields from 2020 to 2026 showing the rate-hiking cycle](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-5.png)

When a company issues a 10-year bond, it does not pick a yield out of the air. It takes the 10-year Treasury yield (here, around 4.5% in mid-2026) and adds a **credit spread** — extra yield to compensate for the chance the company defaults. A solid investment-grade firm might pay the 10-year Treasury plus 130bp; a riskier high-yield name might pay Treasury plus 400bp. The auction sets the floor; the syndicate sets the spread on top. (Why the curve has the shape it does, and what it predicts, is [the yield curve's own story](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — we are only using its *level* here.)

## Corporate and sovereign syndicated deals: how a company sells a billion dollars of debt

A company is not a serial issuer in the Treasury sense. It might come to market a few times a year, in different maturities and sizes, and each deal is a one-off event that has to be *sold* to investors who were not necessarily expecting it. There is no standing army of dealers obliged to bid. There is no published calendar that the whole world watches. So the company cannot just run an auction and trust a clearing price to fall out — it needs intermediaries who *know the buyers*, who can gauge demand before committing to a price, and who will *guarantee* the company gets its money even if demand disappoints. Those intermediaries are the **syndicate** of investment banks, and the process is **syndication**.

(The mechanics of *who bears the risk* in a syndicate — firm-commitment vs best-efforts underwriting, the gross spread, the greenshoe — are the subject of [underwriting and the syndicate](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk). Here we focus on the *price-discovery* process: how the deal goes from an idea to a printed bond.)

### The mandate and the announcement

It begins with the **mandate**: the issuer chooses a small group of banks to run the deal. The lead bank (or banks) is the **bookrunner** — the one that actually builds the order book and sets the price; others may be **co-managers** in supporting roles. Winning the mandate is fiercely competitive, because it is lucrative and prestigious; banks pitch for it with views on timing, structure, and where they think the deal can price. (How a bank makes money from this seat is covered in [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).)

On the morning of the deal — issuers like to launch early so the deal can price and the order book can be worked through US, European, and sometimes Asian hours — the bookrunner sends out the **announcement**: "ABC Corp, \$1bn benchmark, 10-year senior unsecured, expected ratings Baa2 / BBB, books open." This is the starting gun.

![A syndicated bond deal timeline from mandate through announcement, IPT, books open, guidance, launch, and allocation](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-4.png)

### Initial price talk, the book, guidance, and tightening

Here is where syndication differs most sharply from an auction. The bank does not announce a *price*; it announces a *range*, deliberately set wide and cheap, and then watches demand pull it in. The sequence:

1. **Initial price talk (IPT)**, also called *initial guidance*: the bank floats a starting spread, set conservatively wide — say "Treasuries + 150bp area." The word "area" matters: it is a soft, generous number designed to *attract* orders, not to be the final price. Pricing IPT wide is intentional; it is the opening offer in a negotiation, and a wide IPT gives the deal somewhere to go.
2. **Building the book.** Investors who like the bond at +150 area place orders with the bookrunner: "I'll take \$50mm." The bank aggregates these into the **order book** — the running tally of demand at each price level. Crucially, the bank can *see* the demand building in real time while the issuer is still on the hook for nothing. This is private price discovery: the issuer learns what the market will bear before committing.
3. **Guidance and tightening.** As orders pour in and the book grows past the deal size — \$2bn of orders for a \$1bn deal, then \$3bn — the bank revises the talk *tighter*: "guidance: +135bp (+/- 5)." A bigger book means the bank can demand a higher price (lower spread) and still sell the whole thing. Investors who want the bond have to decide whether they will stay in at the tighter level. Some drop out; if enough stay, the book is still covered, and the bank can tighten further.
4. **Launch and final pricing.** When the book is comfortably oversubscribed at a level the issuer is happy with, the bank **launches** the deal at the final spread — "+130bp, launched" — sets the actual coupon, and closes the books. The whole arc from announcement to launch is usually a single morning.

### The investor call, anchor orders, and the padded book

Two practical details make the book-building above work in the real world. The first is *marketing*. For a plain investment-grade deal from a well-known issuer, marketing might be nothing more than an **investor call** — a recorded call or a few headlines — and the deal launches the same morning it is announced. For a riskier high-yield deal, a debut issuer, or a complex structure, the syndicate runs a proper **roadshow**: one to three days of group calls and one-on-one meetings in which management walks investors through the credit story before any order is taken. The rougher the credit, the more selling it needs — which is the same principle that governs equity IPOs, just dialed to the lower-drama register of a bond.

The second detail is that the bookrunner usually has the deal half-sold before the public announcement. In the hours before launch, the bank quietly lines up **anchor orders** (sometimes called cornerstone orders) — large, credible commitments from a handful of trusted real-money accounts. Announcing into a book that already has, say, \$400 million of anchor demand against a \$1 billion target creates instant momentum and emboldens everyone else to pile in. It is the order-book equivalent of seeding a tip jar.

But there is an open secret that every bookrunner manages around: **the book is padded.** Investors know that hot deals get scaled back at allocation, so they deliberately *over-order* — an account that genuinely wants \$50 million might put in for \$150 million, expecting to be cut. A "\$3 billion book" on a \$1 billion deal therefore overstates true demand, and a good syndicate desk *discounts* the inflated orders when judging how far it can tighten. Push the spread too tight on a padded book and the inflated orders evaporate, leaving the deal under-covered at launch — an embarrassment that can force the bank to widen back out or, worse, eat unsold bonds under its firm commitment. Reading a book honestly — separating real, sticky demand from reflexive padding — is precisely the skill an issuer is paying the bookrunner for.

#### Worked example: a \$1B deal, 3x covered, tightening 20bp

ABC Corp wants \$1 billion of 10-year money. With the 10-year Treasury at 4.50%, the bank opens with **IPT at Treasuries + 150bp**, implying an all-in yield of about 6.00%. Orders flow in: by mid-morning the book is at \$3 billion — **3x oversubscribed**. With three dollars of demand for every dollar on offer, the bank tightens. It moves to **guidance at +135bp**, and although a few price-sensitive accounts drop out, the book stays above \$2.5bn. The bank tightens once more and **launches at +130bp**, a final yield of about 5.80%. The deal tightened **20bp** from IPT (150 → 130), and the book finished around 2.5x covered after the dropouts.

What did that 20bp save the issuer? On \$1 billion of 10-year debt, 20bp is 0.20% × \$1,000,000,000 = **\$2,000,000 of interest per year**, or roughly **\$20 million** over the ten-year life of the bonds (ignoring discounting). **The intuition:** the oversubscribed book is not vanity — every turn of cover the bookrunner converts into a tighter spread is real, recurring interest the company never has to pay, which is exactly why the issuer hired a bank that could generate the demand.

### Allocation: who actually gets the bonds

When a \$1bn deal has a \$2.5bn book, someone has to *not* get what they asked for. **Allocation** is the bookrunner deciding which orders to fill and by how much, and it is more art than arithmetic. The bank does not simply prorate. It favors:

- **High-quality, long-term holders** ("real money" — pension funds, insurers, big asset managers) over fast-money accounts likely to flip the bond on day one. The issuer wants a stable shareholder-of-debt base, not a churn.
- **Accounts that supported the deal early**, when the book was uncertain and orders were genuinely useful, over latecomers who piled in once the deal was obviously covered.
- **The issuer's relationships** and the syndicate's own franchise clients.

This discretion is a feature, not a bug: it is *why* an issuer pays a bank rather than running a blind auction. The bank's knowledge of *who the buyers are* lets it place the bonds in strong hands, which makes the bond trade better in the secondary market, which makes the *next* deal easier to sell. Once again: the primary market is selling a claim whose value depends on its secondary-market behavior.

### The new-issue concession: why a new bond prices cheap

Now the subtlest and most important idea in the whole post. A new bond almost never prices *flat* to where the issuer's existing bonds trade. It prices a few basis points **cheap** — a slightly higher yield, a slightly lower price — than the issuer's own secondary-market curve would imply. That give-up is the **new-issue concession** (NIC, sometimes "new-issue premium").

Why would an issuer leave money on the table on purpose? Because investors have a choice. ABC Corp already has bonds trading in the secondary market at, say, a 4.50% yield. If the new bond is offered at *exactly* 4.50%, why would an investor buy it? They could just buy the existing bond. To pull money *out of* the bonds investors already own (or out of a competitor's bonds), the new issue has to offer a small *reward* — a few extra basis points. That reward is the concession. It is the price of getting attention and clearing the book.

![Before and after of pricing flat to the curve versus adding a five basis point concession, and what it does to the order book](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-6.png)

The size of the concession is itself a market signal. In calm, demand-heavy markets, concessions shrink toward zero — issuers can price flat to their curve, or even *through* it (a *negative* concession), because investors are scrambling for paper. In volatile or supply-heavy markets, concessions widen to 10, 20, even 30+bp, because investors need a bigger inducement to take down risk. Watching new-issue concessions is one of the cleanest reads on how hungry the credit market is on any given day.

#### Worked example: the arithmetic of a 5bp concession

ABC Corp's existing 10-year bonds trade at a 4.50% yield. The bookrunner prices the new 10-year issue at a **5bp concession** — a 4.55% yield. What does 5bp cost the issuer, and what does it cost in price terms?

In *yield* terms, the cost is direct: 5bp on \$1 billion is 0.05% × \$1,000,000,000 = **\$500,000 of extra interest per year**, about **\$5 million** over the bond's life (undiscounted). In *price* terms, the concession shows up as a lower issue price. The **DV01** (dollar value of a 1bp move) of a 10-year bond is roughly \$0.08 per \$100 of face, so a 5bp concession is about 5 × \$0.08 = **\$0.40 per \$100**, i.e. the bond is issued at roughly \$99.60 instead of \$100.00. On \$1bn face, that is a \$4 million lower upfront take. **The intuition:** the concession is a small, deliberate discount the issuer pays so the bond is *visibly* cheaper than what is already out there — and because a well-conceded bond usually trades *up* in the days after pricing, that pop is, in effect, a gift the issuer hands to its first investors to keep them coming back.

That last point closes the loop with the auction. A tailing Treasury auction and a fat new-issue concession are the *same phenomenon* in two different selling mechanisms: both are the market demanding extra yield (a lower price) to absorb supply. The auction reveals it as a stop-out above the when-issued yield; the syndicate reveals it as a wider concession. In both cases the issuer is discovering, in real time, the highest price at which the whole issue clears — which is exactly the problem we said issuance exists to solve.

## Other routes: private placements, MTN programs, and shelves

Auctions and syndicated public deals are the headline acts, but a large share of debt is raised through quieter channels that trade *speed and flexibility* for *breadth of distribution*.

A **private placement** sells the bond directly to a small number of large, sophisticated investors (insurers, big asset managers) *without* a public offering and without full public registration. In the US, the legal machinery is **Regulation D** (which exempts certain private offerings from registration) and **Rule 144A** (which lets those privately-placed securities be resold among **qualified institutional buyers (QIBs)** — institutions with at least \$100 million in investments). A 144A deal can be syndicated and booked much like a public deal, but it skips the slow, expensive SEC-registration process, so it is faster to market and is the standard route for high-yield issuance and for foreign issuers selling into the US. The trade-off: a narrower buyer base (only QIBs) and slightly less secondary-market liquidity, for which investors usually demand a few extra basis points.

#### Worked example: a 144A high-yield deal vs a public deal

A mid-size company rated BB wants \$500 million of 7-year money quickly. A fully SEC-registered public deal might take weeks of preparation; a **144A-for-life** deal (sold only to QIBs, never registered) can be announced and priced in days. The cost is liquidity: because the bonds can only trade among QIBs, investors price in a small liquidity premium — say **+15bp** versus an otherwise-identical registered bond. On \$500 million, 15bp is 0.15% × \$500,000,000 = **\$750,000 a year** in extra interest. **The intuition:** the issuer is paying roughly three-quarters of a million dollars a year for the privilege of raising money in days instead of weeks and skipping the registration grind — a trade that makes sense when speed or confidentiality is worth more than the cheapest possible coupon.

A **medium-term-note (MTN) program** is standing infrastructure for issuers who tap the market *constantly* in small, varied amounts. Instead of documenting each bond from scratch, the issuer sets up a program once — a master legal framework and a maximum aggregate size — and can then issue notes off it on short notice, in bespoke sizes, maturities, and even currencies, often *reverse-inquiry* (an investor calls and says "I want \$25mm of your paper at 8 years," and the issuer prints it to order). MTNs turn issuance from an event into a utility.

**Shelf registration** is the public-market cousin of the MTN idea. Under SEC Rule 415, a frequent issuer files one big registration statement that "shelves" a large amount of securities it *may* sell over the next few years. When it wants to issue, it can do a **takedown** off the shelf almost instantly — no fresh registration, just a short pricing supplement. The largest, most creditworthy issuers get **automatic shelf registration** (a Well-Known Seasoned Issuer, or WKSI, can issue essentially on demand). Shelves are why a blue-chip company can go from "we'd like to raise \$2bn" to "priced" in a single morning: the heavy legal lifting was done in advance.

The common thread across all three: they exist to *reduce the friction and time* of issuance for repeat borrowers. The public syndicated deal is the full-dress version; private placements, MTNs, and shelves are the express lanes.

## The role of credit ratings: where a bond prices before it prices

We have repeatedly said a corporate bond prices at "Treasury plus a spread." The single biggest determinant of that spread — before the order book even opens — is the issuer's **credit rating**. A rating, from an agency like Moody's, S&P, or Fitch, is a graded opinion on how likely the issuer is to pay back. The grades run from the pristine (AAA / Aaa) down through **investment grade** (BBB-/Baa3 and above) into **high yield** or "junk" (BB+/Ba1 and below).

The rating matters for issuance in three concrete ways. First, it sets the *starting* spread: an A-rated issuer simply prices tighter than a BB-rated one, full stop, and the IPT will reflect that from the first message. Second, it determines *who can even buy* the bond — many institutional mandates are legally or contractually restricted to investment-grade paper, so crossing the BBB-/junk line dramatically shrinks or expands the buyer base, which feeds straight back into the spread. Third, **rating triggers** at the edges (a "fallen angel" downgraded from BBB to BB, or a "rising star" upgraded the other way) cause forced selling or buying that moves spreads sharply. (The mechanics of *how* ratings are assigned, and the conflicts of interest in the issuer-pays model, are their own topic — we are only using ratings here as the input that sets the price.)

The practical upshot for issuance: before a deal is even announced, the bookrunner and issuer have a tight estimate of where it should price, built from (1) the Treasury benchmark for that maturity, (2) the issuer's rating and where similarly-rated peers trade, and (3) the current new-issue concession environment. The order-book process then *confirms or adjusts* that estimate against live demand. The auction and the syndicate are, in the end, two different machines for the same job: turning a credit's rating and the prevailing rate level into a single clearing price that the whole issue will actually sell at.

## Common misconceptions

**"A government auction is just the government posting a price and people buying."** No — the government posts a *size and a date*, not a price. The price (yield) is *discovered* by the competitive bidding and crystallized as the stop-out. The Treasury is a price-*taker* at its own auctions; it accepts whatever clearing yield the demand produces (it does not set a reserve or reject the auction for being too cheap). That is the opposite of how most people assume a government "selling bonds" works.

**"In a single-price auction, the aggressive bidders overpay."** Exactly backwards. In a single-price auction *everyone pays the same clearing price*, so the aggressive bidder who bid at 4.20% pays the 4.26% stop-out price just like the marginal bidder — they get a *better* deal than they bid for. It was the *old* multiple-price system where aggressive bidders paid their own (higher) price. Single-price was adopted *precisely* to remove that penalty and encourage honest, aggressive bidding.

**"An oversubscribed deal means investors will make easy money."** Not reliably. A book that is 3x covered does not mean the bond is cheap — it usually means the bookrunner will *tighten* the price until the book is just comfortably covered, capturing that demand for the issuer. The investor's "reward" is the small new-issue concession (a few bp, often worth a fraction of a point of price), not a windfall. Genuine mispricings get arbitraged away in the book-building itself.

**"The new-issue concession is a banker's trick to underprice the deal."** The concession is real, but it is the issuer's *rational choice*, not a swindle. Without a few basis points of reward, investors have no reason to swap out of bonds they already own and into the new one. The concession is the cost of pulling money off the sidelines and into *this* deal — and in hot markets it shrinks to nothing or goes negative, which would be impossible if it were merely a fee in disguise.

**"Bid-to-cover is the only thing that matters in an auction."** It is one signal, not the signal. A high bid-to-cover with a big *tail* (a stop-out well above the when-issued yield) is a weak auction dressed up — lots of low-ball bids inflating the cover while real demand was thin. Strong auctions stop *through* or near WI *and* have healthy cover *and* place a high share with end investors rather than dealers. You read all three together.

**"The underwriting bank decides what yield the bond pays."** The bank runs the *process*, but the market sets the *price*. A bookrunner that simply declared a spread would either leave the issuer overpaying (if it picked a number that was too cheap) or be stuck with unsold bonds (if it picked one that was too rich). What the bank actually contributes is *distribution and judgment*: it knows who the buyers are, it builds and reads the order book, and it discounts the padding to find the tightest spread that real demand will hold. The clearing yield is discovered from that demand, not dictated. The same is true of the issuer's credit rating — the agencies grade the borrower, but it is investors, bidding their cash, who decide how many basis points that grade is worth this week. Every actor in the chain influences the price; none of them simply sets it.

## How it shows up in real markets

**The Salomon Brothers scandal and the birth of single-price (1991–1998).** In 1991, Salomon Brothers was caught submitting bids in the names of clients (without authorization) to corner Treasury auctions and squeeze the when-issued market. The scandal — which nearly destroyed the firm and forced out John Gutfreund — exposed how gameable the old multiple-price auction was. In response, the Treasury ran a multi-year experiment, auctioning 2- and 5-year notes under the single-price method starting in 1992 and finding it produced *better* prices with *less* gaming. By November 1998 it converted *all* marketable securities to single-price auctions. Every Treasury auction since then is the system you saw in this post's first figure — and the reform traces directly to a bid-rigging scandal.

**Apple's return to the bond market (2013 onward).** When Apple — sitting on a mountain of cash — wanted to fund buybacks and dividends without repatriating overseas profits, it issued bonds. Its September 2013 deal was a \$17 billion syndicated, multi-tranche, shelf-registered offering — at the time the largest corporate bond sale in history. The order book reportedly exceeded \$50 billion, several times the deal size, and the bonds priced at tight spreads with a thin concession because AA-rated Apple was exactly the kind of scarce, high-quality paper that "real money" accounts fight to be allocated. It is the textbook case of how a strong rating plus a deep book lets a syndicate tighten aggressively and price near the issuer's own curve.

**The weak Treasury auctions of 2023.** Through the second half of 2023, as the deficit widened and the Fed kept rates high, a run of Treasury auctions *tailed* — stopping at yields above the when-issued level — with soft bid-to-cover ratios and rising dealer takedowns (the obligated buyers absorbing what real-money investors left behind). A particularly weak 30-year auction in August 2023 and several in October helped push the 10-year toward 5%, the highest since 2007. Markets read these auctions, in real time, as a referendum on how much US duration the world wanted to hold — a vivid reminder that the "boring" auction machine is also a continuous, public vote on the price of money. Recall the level it set, in the 2Y/10Y chart above: those are not abstractions, they are auction clearing yields.

**March 2020: when the primary market slammed shut, then reopened on demand.** In the first weeks of the pandemic, investors dumped everything for cash, and the corporate bond *primary* market simply stopped — for about two weeks in March 2020, almost no new investment-grade deals could price at any sane spread, and high-yield issuance went to zero. No company could raise long-term money precisely because no investor was confident they could sell the bond on again. Then, on March 23, 2020, the Federal Reserve announced it would backstop *new* corporate issuance (the Primary Market Corporate Credit Facility) alongside secondary-market purchases. The mere announcement — before the facility bought a single bond — restored enough confidence in future liquidity that the primary market exploded back to life: investment-grade issuers printed a record of more than \$1 trillion of new bonds in the following months, including blockbuster deals from the likes of Boeing and Oracle. It is the cleanest natural experiment in this whole series: pull the expectation of secondary liquidity, and issuance dies within days; restore it, and the machine roars back literally overnight. The auction calendar and the syndicate desk are only the visible plumbing; the water that runs through them is investor confidence that the bond will trade tomorrow.

**Structured and securitized issuance across the cycle.** Not all bonds are plain corporate or government debt. A large slice of issuance is *structured* — pools of loans repackaged into asset-backed securities (ABS) and collateralized loan obligations (CLOs), sold in tranches via syndication. Watch how that machine behaves across the credit cycle:

![Structured-bond issuance from asset-backed securities and CLOs by year, collapsing in 2008 and 2009](/imgs/blogs/how-a-bond-is-issued-auctions-syndication-and-the-deal-7.png)

The collapse in 2008–2009 is the issuance machine *seizing*: when investors stopped trusting the secondary-market value of these bonds, the primary market for them shut almost overnight. It is the spine of this whole series stated in the negative — when the secondary market cannot promise liquidity, *primary issuance dies*. (The mechanics of building these structures live in [securitization](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities); we mention them here only as a third issuance channel.)

**Vietnam and emerging-market sovereign issuance.** Smaller and emerging sovereigns mostly *syndicate* their international (typically dollar- or euro-denominated) bonds rather than auction them, precisely because they lack a captive primary-dealer base and a deep domestic buyer pool for foreign-currency paper. They hire global banks, build a book among international investors, and pay a concession reflecting their rating and the appetite for emerging-market risk that week. Domestically, Vietnam's State Treasury *does* auction local-currency government bonds through the Hanoi exchange to a panel of banks and insurers — the same single-price logic, at national scale. The split (auction at home where you have a captive base, syndicate abroad where you must be sold) is the clearest illustration of *why two issuance methods coexist*: it depends entirely on whether the buyers will come to you or must be brought to you.

## The takeaway: a new bond's price is a confession

Step back. We have looked at two very different selling machines — the mechanical Treasury auction and the relationship-driven syndicated deal — plus the express lanes of private placements, MTNs, and shelves. Underneath the differences, they do one identical thing: they *discover the highest price at which a whole new issue will clear*, and they make that price public.

That is why a new bond's price is worth reading like a confession. A Treasury auction that *tails* and a corporate deal that needs a *fat concession* are telling you the same thing in two dialects: right now, this week, the market demands extra yield to absorb supply. An auction that stops *through* its when-issued level and a deal that tightens 20bp on a 3x book are also telling you the same thing: paper is scarce and buyers are hungry. The stop-out yield, the bid-to-cover, the tail, the concession, the cover ratio of the book — these are not back-office trivia. They are the real-time price of money being set, deal by deal, auction by auction.

And it all rests on the secret at the center of this series. No primary dealer would be obligated to bid, no when-issued grey market would exist, no real-money account would anchor a corporate book, if any of them feared they could not *sell the bond tomorrow morning*. The auction and the syndicate are elaborate machines for *creating* securities — but they only run because a secondary market stands ready to *trade* them. When that confidence holds, the world's largest borrower refinances itself in ninety seconds on a Wednesday. When it fails — as it did for structured credit in 2008 — the machine stops cold. Issuance is the visible event; liquidity is the invisible permission slip that makes it possible.

The next time you see a headline that "the Treasury auction was weak" or "the corporate deal priced with a chunky concession," you will know exactly what is being said: somewhere in a quiet, automated, almost invisible process, the price of borrowed money just moved — and somebody had to be paid a little more to lend it.

## Further reading & cross-links

- [Debt vs equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — why an issuer chooses to sell bonds rather than shares in the first place.
- [Underwriting and the syndicate: who takes the risk](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk) — firm-commitment vs best-efforts, the gross spread, the greenshoe, and how the banks get paid for the deal mechanics we described.
- [Money market vs capital market: where short meets long](/blog/trading/capital-markets/money-market-vs-capital-market-where-short-meets-long) — how short-dated T-bills and commercial paper differ from the long-dated notes and bonds issued here.
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — the benchmark *level* every bond prices off, and what its shape predicts (the pricing math we deliberately did not re-derive).
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the bookrunner's seat, the franchise, and why mandates are worth fighting for.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the structured-issuance channel we touched on, in full.
