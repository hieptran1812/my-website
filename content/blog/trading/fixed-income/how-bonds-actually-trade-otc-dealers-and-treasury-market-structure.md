---
title: "How bonds actually trade: OTC, dealers, and Treasury market structure"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into why bonds trade over the counter through dealers instead of on a central exchange, why they are far less liquid than stocks, how the Treasury market is plumbed together, and why liquidity vanishes in a crisis until the Fed steps in."
tags: ["fixed-income", "bonds", "market-microstructure", "dealers", "liquidity", "treasury-market", "bid-ask-spread", "repo", "primary-dealers", "us-treasuries", "trace"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — bonds do not trade on a central exchange like stocks; they trade *over the counter* through a web of dealers, and that one structural fact explains almost everything strange about how the bond market behaves.
> - There is **no single price and no single tape** for a bond. You ask a dealer for a quote, the dealer buys from you at the **bid** and sells to you at the higher **ask**, and the gap — the **bid-ask spread** — is the dealer's pay for holding the bond on its own balance sheet.
> - Bonds are **far less liquid than stocks** for a simple reason: one company has **one** stock but can have **dozens** of separate bonds, each a different size, coupon, and maturity, so trading is fragmented across thousands of barely-traded line items.
> - The newest Treasury — the **on-the-run** issue — soaks up almost all the trading and is super-liquid; the **off-the-run** bonds that came before it trade thinly and yield a few basis points more, a measurable **liquidity premium**.
> - Liquidity is not constant. It is **cheapest when you do not need it and most expensive when you do**: in a panic, dealer balance sheets jam, spreads blow out, and even Treasuries can become hard to sell — which is exactly what happened in **March 2020** until the Fed pledged unlimited buying.
> - Knowing the plumbing is not academic. A round-trip trade in a thin bond can cost you **1–2% in spread alone** — more than a year of its yield — and the difference between a liquid and an illiquid market is the difference between selling at a fair price and selling at a fire-sale one.

Here is a fact that surprises almost everyone when they first hear it. You can buy a share of Apple right now, in one second, at a price you can see on your screen, and that price is the same for everyone on Earth at that instant. But if you wanted to buy an Apple *bond* — a loan to the same company — there is no screen showing one price, no exchange matching you to a seller, and the price you get might be meaningfully different from the price the investor next to you gets for the identical bond at the identical moment. Same company. Completely different machinery.

That difference is not an accident or an oversight. It is the deliberate, centuries-old structure of the bond market, and it is the subject of this post. Stocks trade on **exchanges** — central, lit marketplaces where every order meets every other order in one book. Bonds trade **over the counter** (OTC): privately, through dealers, one negotiation at a time. Once you understand *why* bonds work this way, a whole list of mysteries dissolves — why bonds are so much less liquid than stocks, why the same bond can have several prices, why the "10-year Treasury yield" everyone quotes is really the yield on one specific super-traded bond and not the others, and why, in a crisis, the safest market in the world can briefly stop working until a central bank rides to the rescue.

![A side-by-side comparison showing on the left stocks trading on one central exchange where buyers and sellers all route to a single order book, and on the right bonds trading over the counter through a connected web of dealers each quoting their own price to an investor](/imgs/blogs/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure-1.png)

The diagram above is the mental model for the entire post — keep it in your head as we go. On the left is the stock market: one company has one stock, every order flows into one central order book, and there is one price and one public record of every trade (the "tape"). On the right is the bond market: there is no central book. Instead there is a network of **dealers** — banks and trading firms that stand ready to buy and sell — and when you want to trade, you contact several of them, collect their quotes, and deal with whoever offers the best one. The dealer in the middle is the defining character of this story. By the end, you will be able to trace a single bond trade through that web, put a dollar figure on what it costs, and understand why that cost balloons in exactly the moments you would most want to trade. (This is educational, not financial advice; the goal is to understand the machine, not to tell you what to do with your money.)

## Foundations: the building blocks you need first

Let's assemble the vocabulary from zero. Some of this overlaps with [the rest of the bond series](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction); if a term already feels familiar, skim it, but don't skip, because the whole post lives in how these pieces fit together.

**A bond is a loan you can trade.** When you buy a bond, you lend money to an **issuer** — a government, a company, a city — in exchange for a schedule of payments: periodic interest (the **coupon**) and the return of your money (the **face value**, or **par**, usually \$1,000 or quoted as \$100) on a fixed date (the **maturity**). The key word here is *trade*: a bond is not a savings account you wait out. You can sell it to someone else before it matures, at whatever price the market will bear that day. This post is entirely about *how* that selling and buying actually happens.

**A market is just a place where buyers meet sellers — but the "place" can take very different shapes.** The two shapes that matter for us are the **exchange** and the **over-the-counter** market. An *exchange* is a single, central venue — the New York Stock Exchange, Nasdaq — where every buy order and every sell order for a security lands in one shared list called the **order book**, and the venue automatically matches the highest bid to the lowest offer. *Over the counter* means there is no central venue at all: trades are arranged privately, usually through an intermediary called a dealer, and each trade is its own little negotiation.

**A dealer (or market maker) is a firm that trades for its own account and quotes you two prices.** This is the most important definition in the post, so go slowly. A dealer is not a matchmaker who finds you a counterparty and takes a fee — that's a **broker**. A dealer is a *principal*: it buys the bond *from* you onto its own books, or sells the bond *to* you out of its own inventory, taking the other side of your trade itself. To do that, it quotes two prices at once: a **bid** (the price at which it will *buy* from you) and an **ask**, also called the **offer** (the higher price at which it will *sell* to you).

**The bid-ask spread is the gap between those two prices, and it is the dealer's pay.** If a dealer is willing to buy a bond from you at \$98.50 and sell the same bond to someone else at \$99.00, the **bid-ask spread** is \$0.50 per \$100 of face value. That half-dollar is what the dealer earns for standing in the middle, holding inventory, and bearing the risk that prices move against it before it can offload the bond. The spread is the *price of immediacy* — the cost of being able to trade right now instead of waiting to find a natural counterparty. Throughout this post, when something gets "less liquid," what concretely happens is *the spread gets wider*.

**Liquidity is how easily you can trade without moving the price.** A *liquid* market is one where you can buy or sell a large amount quickly, at a price close to the last trade, with a tight spread. An *illiquid* market is the opposite: trading is slow, the spread is wide, and a decent-sized order shoves the price against you. Liquidity is not a fixed property of a bond — it changes by the hour, and it is the central variable of this whole post.

**A basis point** is one hundredth of a percent — 0.01%. Bond people quote tiny differences in yield in basis points ("bps"): a bond that yields 4.26% instead of 4.20% yields "6 bps more." We'll use basis points constantly because the liquidity differences between bonds are often just a handful of them.

**On-the-run vs off-the-run.** When the US Treasury auctions a new 10-year note, that brand-new bond becomes the **on-the-run** 10-year — the current, freshly-issued benchmark. The moment the *next* 10-year is auctioned a few weeks or months later, the old one becomes **off-the-run**: still a perfectly good bond, just no longer the headline issue. As we'll see, almost all the trading piles into the on-the-run, which makes it the most liquid bond on Earth and leaves the off-the-runs trading more thinly.

**Primary dealers** are the roughly two dozen big banks and trading firms officially designated to trade directly with the Federal Reserve and obligated to bid at every Treasury auction. They are the wholesale backbone of the Treasury market — the firms through which new government debt first enters the world.

**The repo market** (short for *repurchase agreement*) is the overnight lending market that funds the dealers. A dealer holding a billion dollars of bonds in inventory doesn't pay for them with its own cash; it borrows against them overnight, using the bonds themselves as collateral, in the **repo** market. Repo is the plumbing under the plumbing — the cheap, short-term funding that lets dealers carry large inventories. When repo seizes up, dealers can't hold inventory, and liquidity dries up everywhere.

With those in hand, here's the one sentence that motivates everything: **bonds trade over the counter through dealers who profit on the bid-ask spread, and because each issuer has many bonds, that liquidity is thin and fragile — so the spread you pay, and the danger that it explodes in a crisis, is the hidden cost at the heart of fixed income.** Now let's build it up, piece by piece.

## Why bonds trade over the counter and stocks don't

Start with the puzzle itself. Apple has *one* common stock. Every share is identical and interchangeable; there are billions of them, and millions trade every day. That homogeneity is exactly what makes a central exchange work: pile every order for that one identical thing into one book, and you get a deep, continuous market with a single price.

Now look at Apple's *debt*. A large company like Apple might have a dozen or more separate bonds outstanding at once — a 3-year note from one offering, a 5-year and a 10-year from another, a 30-year from a third, each with its own coupon, its own maturity date, its own size, its own fine-print covenants. The US government is even more extreme: the Treasury has *hundreds* of distinct securities outstanding, a new one minted at every auction. Each of those bonds is its own little security with its own little market.

That is the core reason bonds trade OTC: **fragmentation.** A central exchange thrives on concentration — one instrument, many traders. The bond market is the opposite — many instruments, each with relatively few traders. If you tried to run a central order book for every one of a company's bonds, most of those books would sit nearly empty most of the time, with no buyer and no seller present at the same moment. A continuous lit market needs continuous two-sided interest, and the typical bond simply doesn't have it. Many bonds don't trade *at all* on a given day.

The dealer model solves this. Instead of waiting for a buyer and a seller to show up simultaneously for the same obscure bond, a dealer stands permanently in the middle: it will buy your bond *now*, even if there's no other buyer in sight, by taking it into its own inventory and waiting — for hours, days, or weeks — until it finds someone to sell it to. The dealer is a *liquidity buffer*. It absorbs the timing mismatch between buyers and sellers, and it charges for that service through the spread. Without dealers, most bonds would be nearly untradeable; with them, you can almost always trade, you just pay the spread for the privilege.

There's a second reason rooted in *trade size*. Bond trades are often huge — millions or tens of millions of dollars at once, dwarfing a typical stock trade. Dumping a \$20 million order into a thin public order book would crater the price the instant it hit. The OTC model lets a big institution negotiate a large block privately with a dealer, off the public screen, without telegraphing its intentions to the whole market and getting front-run. Privacy and size go together: the people who trade bonds are mostly large institutions moving large amounts, and they prefer the discretion of a negotiated trade.

#### Worked example: why one stock beats many bonds for liquidity

*Setup.* Imagine a fictional company, **Northwind Corp**, which we'll use throughout. Northwind has one common stock and, say, eight separate bond issues outstanding: a 2026 note, a 2028, a 2030, a 2033, a 2035, a 2040, a 2045, and a 2052.

*Step 1 — count the stock's daily liquidity.* All of Northwind's equity investors who want to trade today — buyers and sellers alike — converge on *one* instrument. Suppose 4,000 investors want to trade Northwind stock today. All 4,000 meet in one order book. That's a deep, liquid market: plenty of buyers for every seller.

*Step 2 — count the bonds' daily liquidity.* Now suppose 800 investors want to trade Northwind *bonds* today — already far fewer, because bonds are bought to hold, not to flip. But those 800 are split across eight different issues. That's an average of **100 interested traders per bond**, and on any given day a particular issue might have *three* people interested, or *zero*.

*Step 3 — see why an exchange fails here.* A central order book for the 2040 bond, with maybe two buyers and one seller present this hour, is a ghost town. There's no continuous price, no depth. You can't reliably trade. So instead, a dealer quotes a bid and an ask for that 2040 bond and stands ready to take either side — and *that* is why it trades OTC.

*Step 4 — the takeaway.* *One identical, heavily-traded stock concentrates liquidity into a single deep pool; many small, hold-to-maturity bonds shatter the same investor interest into dozens of shallow puddles — which is precisely why stocks live on exchanges and bonds live with dealers.*

## The dealer in the middle: bid, ask, and the spread as pay

Let's zoom all the way in on a single trade and watch the dealer earn its keep. This is the engine of the whole market, and it's worth seeing in dollars.

![A diagram of the dealer model: an investor sells a bond to a dealer at the bid price of ninety-eight dollars fifty, the dealer holds it in inventory on its own balance sheet, then sells it to another investor at the ask price of ninety-nine dollars, and the fifty-cent gap is labeled as the spread that pays the dealer for warehousing the bond](/imgs/blogs/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure-2.png)

Read the figure left to right. On the left, you want to *sell* a bond. The dealer buys it from you at the **bid** — say \$98.50 per \$100 of face. The dealer now *owns* the bond; it sits in the dealer's inventory, on the dealer's own balance sheet, funded by the dealer's own (mostly borrowed) money. The dealer is now exposed: if rates rise before it can sell, the bond loses value and the dealer eats the loss. On the right, eventually another investor wants to *buy* that bond, and the dealer sells it to them at the **ask** — \$99.00. The dealer pocketed the difference: \$0.50 per \$100, the bid-ask spread.

That spread is not a fee skimmed off the top; it's the dealer's *entire business model.* The dealer makes money by buying low (at the bid) and selling high (at the ask), over and over, across thousands of bonds. For that, it provides a genuine service: **immediacy.** You didn't have to wait days or weeks to find a natural buyer for your bond; the dealer gave you cash *now* and took on the job of finding the next holder. The spread is the price of that immediacy, and it compensates the dealer for three real costs: the risk that prices move while it holds the bond (**inventory risk**), the cost of funding the inventory (**carry**), and the risk that the person selling to it knows something it doesn't (**adverse selection**).

Notice what the spread means for *you*, the round-tripper. If you buy a bond at the ask and immediately sell it back at the bid, you lose the full spread without anything happening to the bond's value at all. You bought at \$99.00 and can only sell at \$98.50 — a 0.5% loss baked in the instant you trade. This is why bonds are not for rapid flipping: every round trip costs you the spread, and in thin bonds that spread can dwarf a year's worth of interest.

#### Worked example: the cost of a round trip in a corporate bond

*Setup.* You buy \$100,000 face value of a Northwind Corp bond. The dealer quotes a bid of \$98.50 and an ask of \$99.00 per \$100 of face.

*Step 1 — what you pay to buy.* You buy at the ask: \$99.00 per \$100. On \$100,000 face, that's \$100,000 × (99.00 / 100) = **\$99,000**.

*Step 2 — what you'd get if you sold immediately.* You sell at the bid: \$98.50 per \$100. That's \$100,000 × (98.50 / 100) = **\$98,500**.

*Step 3 — the round-trip cost.* You spent \$99,000 and could recover \$98,500. The difference is \$99,000 − \$98,500 = **\$500**. That \$500 is the bid-ask spread, paid to the dealer, and it vanished the moment you traded — the bond itself did nothing.

*Step 4 — compare it to the yield.* Suppose this bond yields about 5% a year, so \$100,000 face throws off roughly \$5,000 of interest annually. The \$500 round-trip cost is **a tenth of a full year's interest**, gone in one buy-and-sell. If you traded this bond in and out four times a year chasing small moves, you'd burn 40% of your annual income on spreads.

*Step 5 — the takeaway.* *The bid-ask spread is invisible on any statement, but it is a real, immediate cost; in a moderately liquid corporate bond it can eat a tenth of a year's yield per round trip, which is why bonds reward patience and punish churn.*

### Why the spread is wider than it looks

The headline spread — \$0.50 in our example — is the cost of a *clean, normal-sized* trade in a *reasonably liquid* bond. The reality is often worse, for three reasons.

First, **size moves the spread.** A dealer will quote you a tight spread for a "round lot" (a standard institutional size, often \$1 million face). Ask to trade much larger, and the dealer widens the quote, because a bigger position is riskier and harder to offload. Ask to trade much *smaller* — say a retail-sized \$10,000 — and you also get a worse price, because small trades cost the dealer almost as much effort to handle but earn it less.

Second, **the quote you see may not be the quote you get.** In a stock, the displayed bid and ask are firm and public. In a bond, until you actually ask a specific dealer for a price on a specific size, you often *don't know* the real spread. You have to go shopping — ping several dealers — and the act of shopping itself can move the market against you if dealers sense you're a forced seller.

Third, **the spread is two prices set by humans (or algorithms) who can see you coming.** If a dealer suspects you *must* sell — you're a fund facing redemptions, say — it can quote you a lower bid, knowing you have no choice. That's the adverse-selection cost made personal. The spread isn't just a mechanical markup; it reflects what the dealer infers about *why* you're trading.

## The centerpiece: liquidity dries up exactly when you need it

Now we reach the most important idea in the whole post — the one that separates people who understand the bond market from people who've only read about it. **Bond liquidity is not constant. It is procyclical: abundant and cheap in calm markets, and scarce and ruinously expensive in stress.** Put more sharply: liquidity is cheapest when you don't need it and most expensive when you do.

![An illustrative chart with market volatility on the horizontal axis and the bid-ask spread on the vertical axis, showing a curve that stays low and flat in calm markets then bends sharply upward as volatility rises into a panic, with the March 2020 region marked where the spread blows out roughly tenfold, and a note explaining that dealers have finite balance sheets so they widen quotes and pull back when stress hits](/imgs/blogs/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure-3.png)

The chart above is the centerpiece, and it's illustrative — the exact numbers are stylized, but the shape is real and well documented. The horizontal axis is market volatility, the rough fear gauge of how violently prices are moving (think the VIX for stocks or the MOVE index for bonds). The vertical axis is the bid-ask spread a dealer quotes. In calm markets — the flat left portion — a Treasury trades on a spread of a fraction of a cent, and even a corporate bond trades cheaply. But as volatility climbs, the curve doesn't rise gently; it bends upward, faster and faster, until in a true panic the spread can blow out by a factor of ten or more.

Why this brutal nonlinearity? Because of the dealer's balance sheet. A dealer can only hold so much inventory — it has finite capital and finite borrowing capacity. In normal times, it happily warehouses bonds, funded cheaply in repo, and quotes tight spreads to win business. But when volatility spikes, three things happen at once. The bonds in inventory are losing value fast, so the dealer is already nursing losses. Its risk managers cut the amount it's allowed to hold, so it has *less* room to take on new inventory. And its funding gets more expensive or disappears, because repo lenders also get nervous. The dealer's response is the only rational one: **widen the spread dramatically and quote for smaller size**, to discourage trades it can't absorb and to get paid more for the ones it does take. From the seller's side, that looks like liquidity *vanishing* — the bond you could sell for \$99 yesterday now bids \$94, if a dealer will quote it at all.

This is the deepest, most counterintuitive truth about bond liquidity, and it has a precise consequence: **the value of being able to sell is highest precisely when selling is hardest.** A bond fund hit by redemptions in a crisis must sell into exactly this widening-spread environment, crystallizing losses far beyond what the bonds' fundamentals justify. The structure that makes bonds tradeable in good times — dealers absorbing the timing mismatch — depends on dealers *willing and able* to absorb risk, and in a panic they are neither.

#### Worked example: the spread blowing out in a panic

*Setup.* You hold \$1,000,000 face of a liquid investment-grade Northwind bond. In calm markets, the dealer quotes a spread of about 15 cents per \$100 (a half-spread of ~7.5 cents). A crisis hits and volatility spikes.

*Step 1 — the calm-market round-trip cost.* At a 15-cent spread, a round trip on \$1,000,000 face costs \$1,000,000 × (0.15 / 100) = **\$1,500**. Annoying but minor against a bond throwing off perhaps \$50,000 a year in interest.

*Step 2 — the crisis spread.* In the panic, the spread blows out roughly tenfold, to about \$1.50 per \$100. The dealer is barely willing to quote, and only for a fraction of your size.

*Step 3 — the crisis round-trip cost.* At a \$1.50 spread, the round trip now costs \$1,000,000 × (1.50 / 100) = **\$15,000** — ten times the calm-market cost, and a third of the bond's *entire annual interest*, just to trade once.

*Step 4 — the trap.* The cruel part is *when* this happens. You don't widen the spread; the market does, and it does so exactly when you may be forced to sell — to meet redemptions, to cut risk, to raise cash. The spread is widest at the worst possible moment.

*Step 5 — the takeaway.* *Bond liquidity is a fair-weather friend: the cost of trading can multiply tenfold in a crisis, so the liquidity you blithely assumed in calm markets is the very thing that evaporates when a stressed seller needs it most.*

## On-the-run vs off-the-run: even Treasuries aren't equally liquid

If any bond market should be uniformly liquid, it's US Treasuries — the deepest, most-traded bond market in the world, backed by the government that prints the currency. And yet even here, liquidity is wildly uneven, and the reason is a beautiful little case study in how trading concentrates.

![A side-by-side comparison of an on-the-run Treasury and an off-the-run Treasury showing that the on-the-run is the just-auctioned ten-year with deep liquidity and a tiny spread yielding four point two zero percent, while the off-the-run was issued months ago with thinner trading and a wider spread yielding four point two six percent, a six basis point liquidity premium, and a banner at the bottom stating that the only difference is how easily each can be traded](/imgs/blogs/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure-4.png)

The figure lays out the comparison. The **on-the-run** 10-year is the most recently auctioned 10-year note. By convention and by habit, it's where almost all 10-year trading flows: traders quote it, hedge with it, benchmark off it. That concentration makes it phenomenally liquid — the spread is a fraction of a cent, and you can trade enormous size instantly. The **off-the-run** 10-year is a note auctioned some months earlier, with a slightly different coupon and maturity date. It's the same credit (the US government) and almost the same maturity, but trading has moved on to the newer issue, so it trades more thinly, on a wider spread.

Here's the elegant part: because the off-the-run is harder to trade, investors demand to be *paid* for holding it — a few extra basis points of yield. That extra yield is the **liquidity premium**, and it's directly observable. Two bonds, identical credit, nearly identical maturity, differ in yield by several basis points purely because one is easier to sell than the other. It's the cleanest demonstration in all of finance that *liquidity has a price*.

This also explains a subtlety in how people quote "the 10-year yield." When a news anchor says "the 10-year Treasury yield is 4.20%," they mean the *on-the-run* 10-year — that one specific, super-traded bond. The off-the-runs, with their slightly higher yields, are not what gets quoted, even though there are far more of them. The benchmark *is* the most liquid bond, by definition.

#### Worked example: harvesting the liquidity premium

*Setup.* The on-the-run 10-year Treasury yields **4.20%**. An off-the-run 10-year — same government, maturing just a couple of months earlier — yields **4.26%**. You plan to hold to maturity and don't expect to trade.

*Step 1 — the yield pickup.* The off-the-run yields 4.26% − 4.20% = **6 basis points** more. On \$1,000,000 face, that's an extra \$1,000,000 × 0.0006 = **\$600 a year**, every year you hold it.

*Step 2 — what you give up.* If you ever need to sell *before* maturity, the off-the-run will cost you a wider spread — maybe a few extra cents per \$100. If you sell once, you might pay back a year or two of that pickup in the wider spread.

*Step 3 — when it's worth it.* If you're a genuine buy-and-hold investor — a pension matching a liability, an insurer — you never pay the exit spread, so the 6 bps is pure extra income. That's why such investors deliberately hold off-the-runs: they're being paid for liquidity they don't need.

*Step 4 — the takeaway.* *The liquidity premium is real money you can choose to collect by holding the less-tradeable bond — a free lunch if and only if you truly never need to sell, and a trap if you might.*

## The plumbing of the Treasury market

We've been talking about "dealers" as if there's just one layer of them. In the Treasury market specifically, the structure is richer and worth understanding in full, because this is the market that anchors every other interest rate on the planet — the [risk-free benchmark of the world](/blog/trading/fixed-income/us-treasuries-the-risk-free-benchmark-of-the-world).

![A network diagram of the Treasury market showing the US Treasury issuing new debt at auction to primary dealers, the repo market funding the dealers overnight, the dealers trading among themselves through interdealer brokers and electronic platforms like BrokerTec and Tradeweb, and the debt flowing out to institutions like pensions and foreign central banks and to households through TreasuryDirect and bond funds](/imgs/blogs/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure-5.png)

Trace the figure top to bottom — it's the life cycle of a Treasury bond. Let's walk each layer.

**The Treasury issues new debt at auction.** The US government finances its deficits by selling bonds, and it does so through regular **auctions**. It doesn't sell to the public directly at scale; it sells wholesale to the **primary dealers** (and a few others), who bid for the new bonds. This is the "primary market" — where bonds are *created*. (For how the size and pace of issuance moves yields, see [deficits, debt, and bond supply](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields).)

**Primary dealers are the obligated wholesalers.** There are roughly two dozen of them — the big global banks and trading firms. In exchange for the privilege of trading directly with the Federal Reserve, they take on an obligation: they *must* bid at every Treasury auction, ensuring the government can always sell its debt. They are the firms through which new debt enters the system, and the firms the Fed deals with when it conducts monetary policy. They are, in effect, the franchised plumbers of the government bond market.

**The repo market funds the dealers.** A primary dealer that just bought billions of new bonds at auction doesn't pay cash; it borrows against the bonds overnight in the **repo** market — selling them tonight with an agreement to buy them back tomorrow at a tiny markup that is, in effect, an overnight interest rate. Repo is the dealers' working capital. It lets them carry huge inventories on a sliver of their own money. When repo is cheap and plentiful, dealers warehouse freely and liquidity is deep; when repo seizes (as it briefly did in September 2019, and again in March 2020), dealers can't fund inventory and liquidity dries up across the entire market.

**Dealers trade with each other through interdealer brokers and electronic platforms.** Having bought bonds, dealers need to lay off and rebalance inventory among themselves. They do this in the **interdealer market**, historically through **interdealer brokers** (IDBs) — middlemen who let dealers trade with each other *anonymously*, so no dealer tips its hand about its position. Increasingly, this happens on **electronic platforms** like BrokerTec and Tradeweb, where trades are clicked or executed by algorithms at machine speed. This interdealer layer is where the on-the-run's razor-thin spread is set — it's the most competitive, fastest-moving corner of the market.

**The debt flows out to end investors.** Finally, the bonds reach the people who actually want to *own* them: institutions (pension funds, insurers, mutual funds, and foreign central banks, who together are [the global demand for safe income](/blog/trading/fixed-income/who-buys-bonds-the-global-demand-for-safe-income)) and households (directly through TreasuryDirect, or indirectly through bond funds and ETFs). This is the "secondary market" — where existing bonds change hands after issuance, and where everything we've discussed about dealers and spreads plays out day to day.

#### Worked example: a new Treasury's journey from auction to your fund

*Setup.* The Treasury auctions \$40 billion of a new 10-year note. Follow \$1,000,000 of it.

*Step 1 — the auction.* A primary dealer bids and wins \$1,000,000 of the new note at a price implying a 4.20% yield. The dealer now owns it — and it's the on-the-run 10-year.

*Step 2 — the funding.* The dealer doesn't tie up \$1,000,000 of its own cash. It repos the note overnight, borrowing, say, \$999,000 against it at the overnight repo rate and posting the bond as collateral. Its own capital at risk is a thin sliver.

*Step 3 — the distribution.* Over the next days, the dealer sells pieces of its position — some to a pension fund through a direct quote, some to other dealers on BrokerTec, some into a bond ETF that needs the new benchmark. Each sale earns the dealer a tiny spread.

*Step 4 — your slice.* You own shares of a Treasury bond ETF. The fund bought a piece of this exact note. You now indirectly own a fraction of a bond that, two weeks ago, existed only as a line in a Treasury auction — having passed through a primary dealer, the repo market, and an electronic platform to reach you.

*Step 5 — the takeaway.* *Every Treasury you own traveled a specific pipeline — auction to primary dealer, funded in repo, distributed through brokers and platforms to end investors — and understanding that pipeline is understanding why the market is deep in calm times and fragile when any one of those pipes clogs.*

## When the world's safest market broke: March 2020

Everything above — the procyclical liquidity, the dealer balance-sheet constraint, the repo plumbing — came together in a single, terrifying week in March 2020, when the Treasury market itself, the bedrock of global finance, briefly stopped working. It's the most important market-structure event of recent decades, and it's the perfect case study because it shows every mechanism in this post failing at once.

![An illustrative timeline of March 2020 showing Treasury market stress climbing through early March as a dash for cash forced everyone to sell, peaking around the twentieth, then collapsing back to calm after March twenty-third when the Fed pledged unlimited purchases, with a note explaining that foreign central banks, funds, and leveraged trades all dumped Treasuries at once and dealer balance sheets jammed](/imgs/blogs/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure-6.png)

The figure tells the arc. In early March 2020, as the pandemic shut down the world, investors did something that, on paper, made no sense: they sold *Treasuries* — the safe asset you're supposed to *buy* in a panic. The reason was a **dash for cash.** Everyone needed dollars at once — companies drawing credit lines, funds meeting redemptions, foreign central banks defending their currencies, leveraged investors getting margin-called. Treasuries are the most liquid thing you own, so they're what you sell first when you need cash *now*. The selling was indiscriminate and enormous.

On the other side, the dealers — the firms whose job is to absorb selling — couldn't. The sheer volume overwhelmed their balance sheets. Post-2008 regulations had made it more expensive for banks to hold large bond inventories, so they had less spare capacity to begin with. As prices gapped around, their risk limits tightened. Repo funding got strained. The result: bid-ask spreads on even on-the-run Treasuries blew out to many times normal, the off-the-runs became nearly untradeable, and the price differences between bonds that *should* trade in lockstep became bizarre. For a few days, the deepest market in the world had the liquidity of a backwater.

Then, on March 23, 2020, the Federal Reserve ended it with one sentence. It announced it would buy Treasuries (and agency mortgage bonds) in **unlimited** quantities — "in the amounts needed." The Fed became the buyer of last resort, the infinitely deep balance sheet that dealers no longer were. The effect was almost immediate: with a buyer of unlimited size standing behind the market, sellers could sell again, spreads collapsed back toward normal, and function returned within days. (This is the "QE" tool in [the central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance), deployed here not to stimulate the economy but to repair the market's plumbing — and it's why the Fed's role in [how central banks use bonds](/blog/trading/fixed-income/how-central-banks-use-bonds-qe-qt-and-the-plumbing) is the ultimate backstop of bond liquidity.)

#### Worked example: the dash-for-cash math

*Setup.* Suppose, in a normal week, the Treasury market's dealers can comfortably absorb \$50 billion of net selling without spreads widening much. Then a dash for cash hits.

*Step 1 — the surge.* In one week, net selling jumps to, say, \$250 billion as everyone raises cash at once — five times the comfortable capacity. (These are illustrative round numbers to show the mechanism, not exact figures.)

*Step 2 — the dealer's bind.* The dealers can't take \$250 billion onto balance sheets that can hold a fraction of that. Their only levers are price and size: drop the bid sharply and quote for tiny amounts. Spreads on even the on-the-run gap from a fraction of a cent to many cents.

*Step 3 — the spiral.* Falling prices trigger more margin calls and more forced selling, which the dealers can absorb even less, which drops prices further. The market is eating itself.

*Step 4 — the backstop.* The Fed's balance sheet is, for practical purposes, infinite. By pledging to buy "in the amounts needed," it converts the \$250 billion of selling from "more than dealers can hold" into "absorbed by the central bank." The spiral stops because the supply of buying is now unlimited.

*Step 5 — the takeaway.* *Dealer liquidity is finite and procyclical, so when selling overwhelms it the market can seize even in Treasuries; the only balance sheet large enough to backstop the whole system in a true panic is the central bank's, which is why the Fed is the ultimate liquidity provider of last resort.*

## TRACE, transparency, and the rise of electronic trading

The OTC market has a structural weakness for the small investor: opacity. If you can't see a public tape of prices, how do you know the dealer's quote is fair? For decades, you mostly didn't — and dealers profited handsomely from that information gap, especially against retail and smaller buyers.

That changed in 2002, when US regulators introduced **TRACE** (the Trade Reporting and Compliance Engine), which requires dealers to *report* their corporate-bond trades shortly after they happen. TRACE didn't turn bonds into exchange-traded stocks — there's still no central order book — but it created something almost as valuable: a *post-trade tape*. After a trade, the price (and, for smaller trades, the size) becomes public. Now an investor can look up where a bond *actually traded* recently and judge whether a dealer's quote is reasonable.

The effect was measurable and large: studies found that **bid-ask spreads on corporate bonds fell substantially after TRACE**, especially for retail-sized trades and for the kinds of bonds where the information gap had been widest. Transparency narrowed the dealer's edge. It's a clean illustration that a chunk of the spread had been pure information rent — money dealers earned simply because the customer couldn't see the fair price.

The second great change is **electronification.** For most of history, trading a bond meant a phone call: you'd ring several dealers, ask for quotes, and deal by voice. Today, a growing share of bond trading happens on electronic platforms, where you can request quotes from many dealers at once with a click, or trade directly against streamed prices. In the most liquid corners — on-the-run Treasuries especially — trading is now substantially **algorithmic**, executed by machines at sub-second speed, much like equities. The further you go toward illiquid bonds (small high-yield issues, odd municipals), the more the old voice-and-relationship model persists, because there simply aren't enough trades to automate. The market is electronifying from the liquid end inward.

Both forces — transparency and electronification — push the same direction: tighter spreads, lower costs for end investors, and a slow erosion of the dealer's traditional information advantage. But neither has abolished the core structure. There is still no central exchange; there are still dealers; liquidity is still procyclical; and in a crisis, the electronic platforms can simply go quiet as dealers pull their quotes, just as the phones once went unanswered.

#### Worked example: what TRACE saved a small investor

*Setup.* You're a retail investor buying \$25,000 face of a Northwind corporate bond. The bond's fair mid-price (halfway between a fair bid and ask) is \$100.

*Step 1 — the pre-TRACE world.* With no public tape, the dealer quotes you an ask of \$101.00 — a full point above fair value — betting you can't check. You pay \$25,000 × (101 / 100) = **\$25,250**, an extra \$250 over fair value, hidden in the price.

*Step 2 — the post-TRACE world.* Now you look up TRACE and see the bond traded at \$100.10 an hour ago in institutional size. Armed with that, you push back, or take your order to a platform that shows competing quotes. The dealer, knowing you can see the tape, quotes \$100.30. You pay \$25,000 × (100.30 / 100) = **\$25,075**.

*Step 3 — the saving.* Transparency saved you \$25,250 − \$25,075 = **\$175** on a single \$25,000 trade — the portion of the old spread that was pure information rent.

*Step 4 — the takeaway.* *Post-trade transparency doesn't change the OTC structure, but by letting you see where bonds actually trade it strips out the part of the spread that was just the dealer charging for your ignorance — which is why TRACE measurably narrowed retail bond costs.*

## Putting a number on it: the round-trip cost across the liquidity spectrum

Let's consolidate everything into one practical table, because the single most useful thing this post can give you is a feel for *how much trading actually costs* across the spectrum from a super-liquid Treasury to a thin junk bond.

![A comparison table showing the round-trip trading cost on one million dollars of face value for four bonds: an on-the-run ten-year Treasury costs about one hundred dollars or zero point zero one percent and trades instantly, an off-the-run Treasury costs about four hundred dollars, a liquid investment-grade corporate costs about three thousand dollars or zero point three percent, and a small high-yield bond costs about fifteen thousand dollars or one point five percent and may take days to trade](/imgs/blogs/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure-7.png)

The table makes the whole argument visceral. The same \$1,000,000 traded round-trip costs about **\$100** in the on-the-run Treasury — a rounding error — and about **\$15,000** in a small high-yield bond, a 150-fold difference, with the off-the-run Treasury and the investment-grade corporate sitting in between. The cost scales directly with illiquidity, exactly as the dealer model predicts: the harder a bond is for a dealer to warehouse and offload, the wider it quotes, and the more you pay to cross the spread.

Two lessons fall out of this. First, **the asset you're trading determines your cost far more than your skill.** A clumsy trader in on-the-run Treasuries pays less than a brilliant one in junk bonds. Second, **illiquidity is a tax on changing your mind.** The thin high-yield bond might yield 8% versus the Treasury's 4%, but if you have to sell it in a bad month, the 1.5% spread eats a fifth of a year's extra yield in one trade — and far more if the spread blows out in stress. The yield premium on illiquid bonds is partly *compensation* for exactly this trading cost.

#### Worked example: when the spread eats the yield advantage

*Setup.* You're choosing between the on-the-run 10-year Treasury yielding **4.0%** and a small high-yield bond yielding **8.0%** — a tempting 4-percentage-point pickup. You're not sure how long you'll hold.

*Step 1 — the headline pickup.* On \$1,000,000, the high-yield bond pays \$80,000 a year versus the Treasury's \$40,000 — an extra **\$40,000** annually. Looks like an easy win.

*Step 2 — the entry cost.* But the high-yield bond's round-trip spread is ~1.5%, or ~0.75% each way. Buying \$1,000,000 face costs you about **\$7,500** in spread on the way in alone.

*Step 3 — the exit, in a bad month.* If you have to sell during stress and the spread has doubled to 3%, your exit costs ~1.5%, or **\$15,000**. Your total round-trip trading cost is \$7,500 + \$15,000 = **\$22,500** — more than *half* of one year's extra yield, gone to spreads.

*Step 4 — the horizon test.* If you hold the high-yield bond for five years, the \$22,500 of trading cost spreads across \$200,000 of extra income — a fine trade. If you're flipped out after eight months, you may have paid more in spread than you earned in extra yield. The yield premium only pays off if your *holding period* is long enough to amortize the trading cost.

*Step 5 — the takeaway.* *A juicy yield on an illiquid bond is not free money; a chunk of it is rent for the wide spread you'll pay to get in and out, so the illiquidity premium only becomes real profit if you hold long enough to outlast the cost of trading it.*

## Common misconceptions

**"There's one price for a bond, like there is for a stock."** No. Because bonds trade OTC across many dealers, there is no single official price at a given instant. Different dealers quote different bids and asks; large and small trades get different prices; the on-the-run and off-the-run versions of "the same" maturity trade at different yields. The "price" of a bond is a fuzzy region, not a point — which is exactly why post-trade tools like TRACE matter so much.

**"Treasuries are perfectly liquid, always."** They're the *most* liquid bonds in the world, but liquidity is concentrated in the on-the-run issues and is procyclical even there. The off-the-runs trade more thinly every single day, and in a true crisis — March 2020 being the proof — even on-the-run Treasuries can become hard to trade until a central bank backstops the market. "Risk-free" refers to credit risk (you'll get paid back), not to liquidity risk (you can always sell at a fair price).

**"The bid-ask spread is a small, fixed cost I can ignore."** It's neither small in illiquid bonds nor fixed. A round trip in a thin bond can cost 1–2% — more than a year of its yield — and that spread *widens dramatically in stress*, exactly when you might be forced to trade. The spread is the single largest hidden cost of bond investing for anyone who trades rather than holds to maturity.

**"Dealers are just middlemen taking a cut."** A broker is a middleman taking a cut. A dealer is a principal that *takes the other side of your trade onto its own balance sheet* and bears real risk — the bond could fall before it resells. The spread isn't a toll; it's payment for the dealer turning your bond into instant cash and warehousing the risk until it finds the next buyer. That service is most valuable, and most fragile, precisely in bad markets.

**"Electronic trading has made bonds work like stocks."** Electronification has tightened spreads and sped up the liquid end of the market, but it hasn't created a central exchange or abolished the dealer model. The illiquid bulk of the market still trades by negotiation, and even the electronic platforms can go dark in a panic when dealers pull their quotes. The plumbing is more modern, but its shape — and its fragility — is the same.

**"More transparency would fix everything."** Transparency (TRACE) measurably helped, especially for small investors, by stripping out information rent. But there's a genuine trade-off: full, instant, pre-trade transparency on *large* trades could actually *hurt* liquidity, because a dealer who must publicly reveal that it just bought a huge block can be front-run before it offloads, so it would quote a worse price for big trades to begin with. The market deliberately keeps large-trade reporting delayed and capped for this reason. Opacity isn't purely predatory; some of it is what makes large-block trading possible at all.

## How it shows up in real markets

**The September 2019 repo spike.** In mid-September 2019, the overnight repo rate — the cost for dealers to fund their bond inventory — suddenly spiked from around 2% to nearly 10% intraday, as a confluence of corporate tax payments and Treasury settlements drained cash from the system. Because repo is the plumbing under the dealers, the spike threatened their ability to carry inventory and hinted at fragility in the whole structure. The Fed responded by injecting cash through repo operations and later expanding its balance sheet. The episode was a dress rehearsal for March 2020: a reminder that the dealer model rests on cheap, reliable overnight funding, and that when that funding wobbles, liquidity everywhere is at risk.

**March 2020 and the birth of "market functioning" QE.** As detailed above, the dash for cash in March 2020 broke the Treasury market until the Fed pledged unlimited purchases on March 23. What's notable for market structure is that this was QE deployed for a *new* purpose: not to lower long-term rates and stimulate the economy (its 2008–2014 role), but to *repair the plumbing* — to be the buyer of last resort when dealer balance sheets were overwhelmed. It permanently established the Fed as the ultimate backstop of bond-market liquidity, with consequences for how investors price liquidity risk: if the central bank will always step in, how much should you fear illiquidity? (That's the moral-hazard debate behind every modern liquidity crisis.)

**The 2022 UK gilt crisis and forced selling.** In late September 2022, a UK budget announcement sent long-dated government bond (gilt) yields soaring. UK pension funds running "liability-driven investment" strategies faced collateral calls and were forced to sell gilts into a falling market — classic procyclical selling into vanishing liquidity, a feedback loop that threatened to become a doom spiral. The Bank of England intervened with emergency gilt purchases to stop it. It was March 2020's mechanism in a different market: forced sellers, overwhelmed dealers, spiraling prices, and a central bank stepping in as the only balance sheet large enough to absorb the selling.

**The on-the-run/off-the-run trade and Long-Term Capital Management.** The liquidity premium between on-the-run and off-the-run Treasuries is so reliable that hedge funds trade it: buy the cheap (higher-yielding) off-the-run, sell short the rich on-the-run, and earn the spread as the two converge. The famous hedge fund LTCM did exactly this, at enormous leverage, in the 1990s. It worked beautifully — until the 1998 Russian default triggered a flight to quality that *widened* the on-the-run/off-the-run spread instead of narrowing it, as panicked investors paid up for the most liquid bond. LTCM's leveraged convergence bet blew up, requiring a Fed-organized bailout. The lesson is pure market structure: the liquidity premium is normally stable, but in a crisis the *demand for liquidity itself* spikes, and bets that assume it will stay calm can be lethal.

**Corporate bond ETFs and the liquidity mismatch.** Bond ETFs let you trade a basket of bonds on a stock exchange, intraday, with the liquidity of equities — even though the *underlying* bonds trade OTC and slowly. In calm markets this works seamlessly. In March 2020, some bond ETFs briefly traded at large *discounts* to the stated value of their holdings — because the ETF's exchange price reflected real-time liquidity conditions, while the underlying bonds' "official" prices were stale (those bonds simply weren't trading). It was a vivid demonstration that you cannot conjure liquidity that isn't there: wrapping illiquid bonds in a liquid exchange-traded shell doesn't make the bonds themselves liquid, it just relocates where the stress shows up.

## When this matters to you, and further reading

If you own bonds only through a fund and never trade them yourself, the dealer and the spread are still working on your behalf every day — inside the fund, every purchase and sale crosses a spread, and that cost quietly drags on your returns, more so in higher-yield and emerging-market funds where the bonds are thinner. If you ever buy individual bonds, the single most valuable habit you can build is to *check the recent traded price* (via TRACE for corporates) before accepting any dealer's quote, and to favor on-the-run and liquid issues unless you're a genuine buy-and-hold investor who can harvest the liquidity premium without ever paying the exit cost.

The deeper takeaway is about *fragility*. The bond market's dealer structure is a marvel that makes thousands of barely-traded securities tradeable in normal times — but it rests on dealers willing and able to warehouse risk, funded cheaply overnight, in calm conditions. Strip away any of those, as a crisis does, and liquidity evaporates exactly when it's most needed. That's why the Fed has become the market's ultimate backstop, and why "liquidity risk" deserves a seat right next to credit risk and interest-rate risk in how you think about any bond.

To go deeper from here: see [credit risk and the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) for the other great risk of corporate bonds, [from the ten-year yield to your mortgage](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates) for how the prices set in this market reach your wallet, and [PIMCO and the bond market](/blog/trading/finance/pimco-and-the-bond-market) for how the largest bond investors actually navigate this OTC world. For the macro backdrop of who issues all this debt and why it moves yields, see [deficits, debt, and bond supply](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields).
