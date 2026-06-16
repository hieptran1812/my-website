---
title: "Treasury Auctions and Refunding: When Supply Is the Event"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Most traders watch demand-side data, but the supply of government bonds is itself a market-moving event — a weak auction with a tail can send yields jumping, and the quarterly refunding announcement reprices the whole curve."
tags: ["event-trading", "macro", "treasury-auctions", "bond-supply", "refunding", "qra", "yields", "bonds", "issuance", "quantitative-tightening", "fixed-income"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Markets obsess over *demand-side* data — inflation, jobs, the Fed — but the **supply** of government bonds is itself a scheduled, market-moving event. When the U.S. Treasury sells more debt than buyers want at the going price, yields have to rise until someone bites.
>
> - A **Treasury auction** sells a fixed pile of bonds to the highest bidders. The clearing yield — the **stop-out yield** — versus the market's pre-auction guess (the **when-issued yield**) is the verdict on demand. A weak auction "**tails**": it stops *above* the when-issued yield because buyers backed away, and yields jump across the curve.
> - The **Quarterly Refunding Announcement (QRA)** is the set-piece: Treasury pre-announces *how much* it will borrow and in *which maturities*. The market reprices the coming supply *before* a single auction happens — the announcement is the event, not the sale.
> - In autumn 2023 nothing about the Fed changed, yet the 10-year yield marched toward **5%** as the market choked on a wave of issuance. Supply moved yields, not the Fed. The trade: read the tail, the bid-to-cover and the indirect take-down, then **follow** the move on a genuinely weak auction or **fade** the knee-jerk on an overreaction.
> - The one number to remember: on **19 October 2023** the 10-year Treasury yield touched roughly **4.99%** — a supply-driven peak with no Fed meeting in sight.

In the autumn of 2023, the U.S. Treasury market did something that confused a lot of people who only watch the Federal Reserve. Between early August and mid-October, the yield on the 10-year Treasury note climbed from about 4.0% to nearly 5.0% — its highest level since 2007. That is an enormous move in the world's most important interest rate, the benchmark off which mortgages, corporate loans and global asset prices are set. And here is the strange part: the Federal Reserve did almost nothing in that window. It held rates steady at its September meeting. Inflation was actually *falling*. The economy was not obviously overheating. By the usual playbook — watch the Fed, watch inflation — yields should have been drifting sideways or down.

Instead they surged. The reason was not on the demand side of the economy at all. It was on the **supply** side of the bond market. The U.S. government was running a deficit of roughly \$1.7 trillion and had to sell an avalanche of new debt to fund it. At the same time, the Fed was *shrinking* its balance sheet — pulling itself out of the market as a buyer. So the market had to absorb a tidal wave of new bonds with one of the biggest buyers stepping back. The only way to clear that much supply is to make the bonds cheaper, and a cheaper bond means a *higher yield*. Yields marched toward 5% not because anyone re-priced the Fed, but because the market was choking on Treasury supply.

![Pipeline of a Treasury auction from when-issued yield to bids to stop-out yield to the tail to a strong or weak verdict](/imgs/blogs/treasury-auctions-and-refunding-supply-as-an-event-1.png)

This post is about the part of the macro calendar almost nobody outside the bond desk pays attention to: the **supply** of government bonds. Most of this series teaches you to trade *demand-side* surprises — a hot inflation print, a weak jobs number, a hawkish Fed. But there is a parallel calendar of **Treasury auctions** and a quarterly **refunding announcement** that can move yields just as hard, and on a 2:00 p.m. clock that has nothing to do with the Fed. By the end you will understand exactly how an auction works, what a "tail" is and why it terrifies a bond trader, what the Quarterly Refunding Announcement is and why it has become one of the most-watched set-pieces in macro, why heavy issuance shoved yields toward 5% in 2023, and how a trader actually reads an auction in the sixty seconds after the results post. The thesis is simple and, once you see it, unforgettable: **supply is an event.**

## Foundations: how a Treasury auction works

Before we can trade the supply, we need a shared vocabulary. The bond desk speaks a dialect of its own — "tails," "stop-outs," "indirects," "bid-to-cover" — and none of it makes sense until you watch a single auction from start to finish. So this section builds the whole machine from zero. If you have no fixed-income background, read it slowly; every later section leans on these definitions.

### A Treasury bond is a loan to the government, sold at auction

When the U.S. government spends more than it collects in taxes — which it does, by a lot — it covers the gap by borrowing. It borrows by selling **Treasury securities**: IOUs that promise to pay the holder a fixed stream of interest and then return the face value at a set date. The short ones (one year or less) are **bills**, which pay no coupon and are sold at a discount. The medium ones (2 to 10 years) are **notes**, and the long ones (20 and 30 years) are **bonds**, both of which pay a fixed **coupon** twice a year. In casual usage everyone calls the whole family "Treasuries" or "bonds," and so will I.

The crucial mechanism is *how* the government sells them: not at a fixed price it sets, but by **auction**. The Treasury announces, "On Wednesday we are selling \$25 billion of new 30-year bonds," and then it lets the market bid for them. Whoever is willing to accept the *lowest yield* — equivalently, pay the *highest price* — gets filled first. This is the single most important fact in the whole post: the government does not dictate the interest rate on its debt. **The market sets it, at auction, every single week.** When you hear that "the 10-year yield is 4.5%," that number was discovered by an auction full of bidders fighting over a fixed pile of bonds. The auction is where the price of government money is actually *made*.

### The when-issued yield is the market's guess before the auction

In the days *before* an auction, the bond does not officially exist yet, but traders already trade it in a forward market called the **when-issued** (WI) market — "when, as and if issued." If everyone expects the new 10-year to clear around a 4.60% yield, the when-issued yield will sit near 4.60%. This is the market's consensus forecast for the auction, exactly analogous to the consensus forecast for a CPI print. It is the bar the auction has to clear. We will measure the auction's strength against this WI yield, so hold onto it: **the when-issued yield is what the auction was "supposed" to clear at.**

### The stop-out yield is the price that clears the supply

When the auction closes (1:00 p.m. Eastern for most coupon auctions), the Treasury sorts every bid from the lowest yield (most aggressive buyer) to the highest yield (most reluctant). It fills the cheap-for-the-buyer bids first and works *up* the yield ladder until the entire \$25 billion is sold. The yield of the *last* bid it has to accept to sell the whole amount is the **stop-out yield** (also called the "high yield" or simply "the stop"). Everyone who bid at or below the stop gets filled; everyone who bid above it goes home empty-handed.

The stop-out yield is the auction's *clearing price*, the single number that summarizes whether demand was strong or weak. If demand was ravenous, bidders fought hard and the auction cleared at a *low* yield (a high price). If demand was thin, the Treasury had to keep walking *up* the yield ladder, accepting less and less aggressive bids, and the auction cleared at a *high* yield. So a high stop-out yield = weak demand; a low one = strong demand.

### The tail is the verdict — and it is the number that moves the market

Now put the two together. The **tail** is the gap between the stop-out yield and the when-issued yield:

> **tail = stop-out yield − when-issued yield**

If the auction clears *above* where the market expected (stop 4.63% vs WI 4.60%), that is a **+3 basis point tail**, and it is *bad news*: the Treasury had to offer a higher yield than the market thought to unload the bonds, which means real buyers backed away. A tail is the bond market's equivalent of a weak appetite at the dinner table — the food had to be marked down to get eaten. When an auction tails badly, yields across the entire curve jump in the seconds after the result, because the auction just revealed that buyers are less willing to lend to the government than everyone assumed. (A **basis point**, or "bp," is one hundredth of a percentage point: 0.01%. Bond people measure everything in bp.)

The opposite of a tail is a **stop-through**: the auction clears *below* the when-issued yield (stop 4.58% vs WI 4.60%, a −2 bp "through"). That is *good news* — demand was so strong that bidders accepted an even lower yield than expected, and yields tend to *fall* after the result. So the sign of the tail is the headline of the auction: positive tail = weak = yields up; negative (through) = strong = yields down.

### Bid-to-cover measures how much demand showed up

The tail tells you the *price* of demand; the **bid-to-cover ratio** tells you the *quantity*. It is simply total bids divided by the amount on offer:

> **bid-to-cover = total dollar amount bid ÷ amount auctioned**

If the Treasury offers \$25 billion and receives \$62.5 billion of bids, the bid-to-cover is 2.5×. A higher ratio means more buyers competing for the same pile, which is healthy. The number alone is meaningless — you compare it to the **trailing average** of the last six to twelve auctions of the same security. A 30-year that normally covers 2.4× but comes in at 2.1× is a soft auction even if 2.1× sounds like plenty. *Versus the average* is the only way to read it.

### Indirect, direct and dealer bidders tell you who showed up

The Treasury also reports *who* bought, split into three buckets, and the mix matters as much as the totals:

- **Indirect bidders** — bids routed through a primary dealer on behalf of someone else, a category dominated by **foreign central banks, foreign accounts and large asset managers**. A high indirect take-down signals strong *real-money* and foreign demand, the stickiest, most price-insensitive buyers. This is the bucket traders watch most.
- **Direct bidders** — domestic institutions (pensions, funds, banks) bidding for their own account directly.
- **Primary dealers** — the ~25 big banks obligated to bid at every auction and absorb whatever is left over. Dealers are the *buyers of last resort*. A high dealer take-down is a *bad* sign: it means real buyers (indirects and directs) didn't show, so the dealers got stuck with the inventory and will have to sell it down later, pressuring prices.

So the read is: high indirects + high cover + a stop-through = a strong auction; low indirects + low cover + a fat tail + dealers stuffed = a weak one.

### Bills, notes and bonds — the maturity ladder

It helps to picture the issuance not as one pile but as a ladder of maturities, because each rung behaves differently. At the bottom are **bills**: 4-, 8-, 13-, 26- and 52-week instruments, sold weekly, in enormous size. Bills are the workhorse of cash management — when the government needs money fast, it sells bills, and money-market funds (which hold trillions in cash) absorb them almost without complaint. Selling bills barely disturbs the long end of the yield curve, because a 13-week bill carries almost no interest-rate risk: it matures so soon that its price hardly moves when yields change.

The middle rungs are **notes** — the 2-, 3-, 5-, 7- and 10-year — auctioned monthly. These carry real duration and are the bread-and-butter of the bond market. The 10-year in particular is the global benchmark, the single most-watched yield on Earth, so its auctions are scrutinized closely.

The top rungs are the **long bonds** — the 20-year and the 30-year — also auctioned monthly but in smaller size, because the natural buyer base for 30-year paper is narrow: pension funds and life insurers who need to match very long-dated liabilities, plus a thin layer of speculative duration buyers. Because that demand base is shallow, the long bond is the most fragile rung — a 30-year auction is the one most likely to tail, and a tail there does the most damage. When you read that "the long end is under pressure," it almost always traces back to the supply-demand imbalance in the 20s and 30s.

The reason this ladder matters for trading is that the Treasury *chooses* where on the ladder to issue. It can fund a given deficit mostly with bills (easy, gentle on the curve) or tilt toward coupons (harder, pushes long yields up). That choice is the QRA's central lever, which is why we now turn to it.

### The Quarterly Refunding Announcement is the supply forecast

All of the above is about a *single* auction. But the market also needs to know the *plan*: how many bonds, in total, is the Treasury going to sell over the next quarter, and in which maturities? That plan is the **Quarterly Refunding Announcement (QRA)** — released by the Treasury four times a year (the first Wednesday of February, May, August and November, at 8:30 a.m. Eastern), alongside the prior Monday's borrowing-estimate release. The QRA tells the market two things that move yields: the **total borrowing need** for the quarter, and the **maturity mix** — how much will be funded with short **bills** versus longer **coupon** notes and bonds.

That mix is the part that detonates. Selling short bills is easy: money-market funds soak them up and they barely touch the long end of the curve. Selling long coupons is hard: it dumps a lot of interest-rate risk (**duration**) onto a market that has to be paid to hold it, so it pushes *long* yields up. When the QRA says "we're shifting toward more long-dated coupon issuance," the long end can sell off on the announcement alone — *before any auction happens.* The QRA is a pure supply event, and since 2023 it has become one of the most-watched set-pieces on the macro calendar.

With the vocabulary in hand, let's go deep — starting with the mechanics of a single auction and building up to why all of this shoved yields toward 5%.

## Auction mechanics: the tail, bid-to-cover, and who buys

Let's slow down and live inside one auction, because the microstructure is where the trade is. Say a 30-year bond auction is scheduled for a Thursday at 1:00 p.m. Eastern. In the days before, the when-issued 30-year has been trading around a 4.60% yield. That is the bar. Every desk on the Street has a view on whether the auction will be "strong" (stop through 4.60%) or "weak" (tail above it), and they have positioned accordingly — short the bond into a feared-weak auction, long it into an expected-strong one. The when-issued yield already contains all of that. Just like a CPI print, the auction result only moves the market to the extent it *surprises* relative to the WI level.

At 1:00 p.m. the bidding window slams shut. Two minutes later, at 1:01–1:02, the Treasury posts the results: the stop-out yield, the bid-to-cover, and the indirect/direct/dealer split. The entire bond market reprices off those numbers in *seconds*. This is a genuine event with a hard clock, exactly like an 8:30 a.m. data release — except it happens at 1:00 p.m. and most equity traders don't even know it's on the calendar.

The single number that hits first is the **tail**. If the stop comes in at 4.63% against a 4.60% when-issued, that +3 bp tail tells everyone that real demand was 3 bp worse than the market assumed. Bonds sell off instantly: the 30-year yield gaps up, and the move ripples down the curve into the 10-year and even the 5-year, because a weak long-bond auction is information about the whole government-bond complex, not just that one bond. Let's put a dollar figure on why traders flinch.

#### Worked example: a 30-year auction tails 3 basis points

Say you run a relative-value book and you are long \$500,000 face value of the 30-year on the assumption the auction would go fine. The auction tails +3 bp and the 30-year yield jumps from 4.60% to 4.63%. How much did that cost you?

The key tool is **DV01** — the dollar value of a one-basis-point move, i.e. how many dollars your position gains or loses when the yield moves 1 bp. A 30-year bond has a very long **duration** (its price is extremely sensitive to yields), so its DV01 is large. For roughly \$500,000 face of the 30-year, the DV01 is about \$1,200 per basis point. So:

- Yield move: **+3 bp** (yields up, prices down — bad for a long).
- DV01: about **\$1,200 per bp** on this \$500,000 position.
- Mark-to-market: 3 bp × \$1,200/bp = **−\$3,600**.

A tiny, almost invisible 3-basis-point tail — three hundredths of one percent — just cost the book **\$3,600** in the two minutes after 1:01 p.m. *Intuition: at the long end, where duration is enormous, even a microscopic tail translates into real money, which is exactly why bond desks hold their breath at 1:00 p.m.*

That is why the tail is the headline. Now layer in the other two numbers. The **bid-to-cover** confirms whether the weak stop was a fluke or a genuine demand problem. A tail *with* a low bid-to-cover (say 2.1× against a 2.5× average) is a real, confirmed demand shortfall — the market should believe it and yields should stay up. A tail with a *normal* bid-to-cover is more ambiguous; sometimes a couple of big accounts simply didn't show, and the move can fade. And the **indirect take-down** tells you whether the missing buyers were the sticky foreign and real-money accounts (a worrying, persistent signal) or just opportunistic dealers. A weak auction where indirects collapsed is a louder, more durable sell signal than one where dealers simply chose not to reach.

The figure below shows the anatomy laid side by side: what a weak auction looks like across all three numbers versus a strong one.

![Before and after comparison of a weak tailing auction versus a strong stop-through auction across tail, cover and indirects](/imgs/blogs/treasury-auctions-and-refunding-supply-as-an-event-3.png)

One more piece of microstructure: the auction's *information* is asymmetric. A *strong* auction (a stop-through, high cover, fat indirects) tends to produce a modest, orderly rally in bonds — "good, demand is fine, move on." A *weak* auction can produce a violent, disorderly selloff, because a tail doesn't just say "this one auction went badly," it raises the terrifying question *who is going to buy all the bonds coming next week, and next month?* The bad auctions matter more than the good ones, which is the same reason markets fall faster than they rise. We will see this asymmetry drive the entire autumn-2023 episode.

### The role of the primary dealers — the shock absorbers

To understand why a tail is genuinely scary, you have to understand the **primary dealers**. These are roughly two dozen large banks (the Goldmans, JPMorgans, Citis and their foreign peers) that have a special relationship with the Treasury and the Fed: in exchange for their privileged role, they are *obligated* to bid at every single auction and to take down whatever the rest of the market doesn't want. They are the system's shock absorbers — the guarantee that every auction clears no matter how weak demand is.

That guarantee is exactly why a weak auction is informative rather than catastrophic. The auction will *always* sell out, because the dealers backstop it. So the question is never "did the auction fail?" (it can't) but "at what *price*, and who got stuck holding it?" When real-money buyers (indirects and directs) step back, the dealers are forced to absorb a bigger slice. But the dealers don't *want* the inventory — they're intermediaries, not investors — so they immediately start hedging and selling it down, which pushes prices lower (yields higher) in the hours and days after the auction. A high dealer take-down is therefore a *leading indicator* of further selling pressure: the bonds are sitting in weak hands that need to offload them. This is why traders watch the dealer share so closely. A 30-year auction where dealers took 35% instead of their usual 20% is a warning that there's a pile of unwanted long bonds about to hit the market.

### The auction clock — a hidden event on the calendar

The thing equity traders miss is that this is a *scheduled* event with a precise clock, every week. A typical coupon auction follows the same rhythm: the Treasury announces the size a week ahead (in the QRA-derived schedule), the when-issued market trades it for days, the bidding window closes at **1:00 p.m. Eastern**, and the results hit the wires at **1:01–1:02 p.m.** In those sixty seconds the entire Treasury curve can lurch. There is a real "auction concession" phenomenon, too: dealers often nudge yields *up* in the morning before a big auction (selling the security to cheapen it) so they can bid for it at a more attractive level, then the market reverses after a strong result. That pre-auction concession and post-auction snap-back is itself a tradeable pattern for those who watch the clock. The lesson is that supply has its own intraday calendar, sitting quietly at 1:00 p.m. while everyone else is watching the 8:30 a.m. data prints.

## Reading a strong versus a weak auction

Now make this operational. When the results post at 1:01 p.m., a trader runs a three-second mental scorecard, and you can too. There are three numbers, each compared against its own benchmark:

1. **The tail (vs. when-issued).** Did it stop *through* (negative, strong) or *tail* (positive, weak)? How big? A 1 bp tail is noise; a 3 bp tail is a real miss; a 5+ bp tail in a long-bond auction is a genuine event that will move the whole curve.
2. **Bid-to-cover (vs. the 6-auction average).** Above average confirms strength; well below average confirms weakness. The level alone is meaningless — only the comparison matters.
3. **Indirect take-down (vs. recent auctions).** High indirects = strong, sticky, foreign/real-money demand. Collapsing indirects = the price-insensitive buyers are stepping away, the most worrying signal of all.

A **strong auction** lights up green on all three: stops through, cover above average, indirects high. The bond rallies (yields fall), and the relief often spills into risk assets — a well-bid Treasury auction removes a tail risk that was hanging over the market. A **weak auction** lights up red: a fat tail, cover below average, indirects soft, dealers stuffed with the leftovers. Yields jump, and if it's a long-bond auction the selloff can drag down stocks too, because higher long-term yields raise the discount rate on every future cash flow in the economy.

#### Worked example: scoring a weak 10-year auction

Suppose a 10-year auction posts these numbers, and you have to grade it in three seconds:

- **Tail:** stop 4.42% vs when-issued 4.38% → **+4 bp tail** (weak; a 4 bp tail on a 10-year is a clear miss).
- **Bid-to-cover:** 2.30× vs a 2.55× trailing average → **below average** (weak; confirms the tail).
- **Indirects:** 62% vs a recent 68% average → **soft** (weak; foreign/real-money demand stepped back).

Three for three weak. The read is unambiguous: real demand fell short, this is not a fluke, and yields should *stay* higher rather than snap back. A trader who was flat going in would lean **short bonds** (positioned for higher yields) immediately, with a stop if the 10-year somehow rallies back through 4.38% (which would invalidate the weak read). *Intuition: when all three numbers agree, the signal is real and you follow it; the dangerous, fade-able auctions are the ones where the three numbers disagree.*

The disagreement case is where the money is for nimble traders. A fat tail with a *strong* bid-to-cover and *high* indirects is a contradiction: plenty of demand showed up (high cover, high indirects), yet the stop came in high. That often means a single mechanical quirk — a large bid placed slightly off-market, an awkwardly timed concession — rather than a genuine demand failure. The knee-jerk selloff on the headline tail can then be *faded*: yields spiked on a number that the internals contradict, and they tend to retrace. This is the bread-and-butter "fade the overreaction" trade, and we'll size one in the playbook.

## The QRA: the quarterly issuance plan as a set-piece event

Step back from the single auction to the *plan*. Four times a year, the Treasury tells the market exactly how much it intends to borrow next quarter and how it will split that borrowing between short bills and long coupons. That is the **Quarterly Refunding Announcement**, and since 2023 it has graduated from a sleepy plumbing release into a genuine macro event that can move the long end of the curve by 10–20 bp in a morning — before a single bond is sold.

![Pipeline of the Quarterly Refunding Announcement from borrowing need to the announcement to the maturity mix to the market repricing the supply](/imgs/blogs/treasury-auctions-and-refunding-supply-as-an-event-5.png)

Why does an *announcement* — not a sale — move yields? Because the market is a discounting machine: it reprices the *expected* supply the instant it learns about it, exactly the way it reprices off a CPI surprise rather than waiting for inflation to physically happen. If the QRA reveals more coupon issuance than expected, the long end sells off *now*, in anticipation of the auctions that will land over the coming weeks. The auctions, when they arrive, are then mostly *already priced* — the QRA was the surprise.

The mechanics run through two knobs. The first is the **total borrowing need**: the bigger the deficit and the more maturing debt that has to be rolled over, the more bonds the Treasury must sell, full stop. The second, subtler knob is the **bill-versus-coupon mix**, and this is the one that detonated in 2023. Short **bills** are easy to place — money-market funds, awash in cash, hoover them up without touching long-term rates. Long **coupons** are hard: each long bond dumps a lot of **duration** (interest-rate risk) onto investors who must be compensated with a higher yield to hold it. So when the Treasury tilts the mix toward more long-dated coupons, it is asking the market to swallow more duration, and the long end gives way.

This is exactly the lever the Treasury pulled in the opposite direction in late 2023, to dramatic effect. After the August 2023 QRA spooked the market with a larger-than-expected slug of coupon issuance — a key ingredient in the run-up toward 5% — the **November 1, 2023 refunding** surprised the market the *other* way: the Treasury announced it would lean more heavily on bills and grow coupon auction sizes by less than feared. The long end rallied hard on the news, and the relief, combined with a dovish-leaning Fed meeting the same week and a soft jobs report two days later, helped mark the *top* in 10-year yields for that cycle. The QRA did not just react to the supply story — it *was* the supply story.

There's actually a two-step rhythm to the refunding, and traders watch both releases. The Monday before the QRA, the Treasury publishes its **marketable borrowing estimate** — a single big number for how much it plans to borrow over the coming quarter. That number alone can move the market: in late July 2023, the borrowing estimate came in roughly \$270 billion *above* the prior projection, an upside supply shock that helped kick off the August selloff before the detailed QRA even landed. Then on Wednesday comes the QRA proper, with the auction-by-auction size schedule and the all-important bill-versus-coupon split. So the supply event is really a one-two punch: the Monday borrowing number sets the *magnitude*, and the Wednesday QRA sets the *composition*.

For traders, the QRA is a calendar event to be treated like any other: there is a consensus (analysts forecast the borrowing estimate and the coupon-size schedule in advance), and the move is driven by the *surprise* against that consensus. A QRA that prints more coupon supply than expected is bearish bonds (yields up); one that leans more on bills than expected is bullish (yields down). We map the trade in the playbook.

One practical note on *where* to watch all this: auction results are published on TreasuryDirect within minutes of the 1:00 p.m. close, and the data wires (Bloomberg, Reuters) push the headline tail, bid-to-cover and the bidder breakdown the instant they post. The QRA and the borrowing estimate come from the Treasury's Office of Debt Management. You don't need a terminal to follow the story — the headline tail on a 10-year or 30-year auction is reported in real time, and the QRA is a public release on a known schedule. The barrier to trading supply isn't access to the data; it's *knowing the calendar exists at all.*

## The 2023 supply-driven selloff: deficits to issuance to yields

Now we can tell the autumn-2023 story properly, because we have all the pieces. The chain runs **deficits → issuance → yields**, and it explains a move that the Fed-watchers couldn't.

Start with the deficit. In fiscal 2023 the U.S. ran a budget gap of roughly \$1.7 trillion — about 6.2% of GDP, an extraordinary number for an economy that was not in recession. Every dollar of that gap had to be borrowed, on top of *rolling over* trillions of existing debt as it matured. Total federal debt had climbed past \$33 trillion. The arithmetic of issuance is brutal and mechanical: a bigger deficit means more bonds to sell, and more bonds to sell — all else equal — means a higher yield to clear them. The chart below shows the supply backdrop that almost nobody outside the bond desk was watching.

![Total federal debt rising with net interest outlays exploding from 2020 to 2025](/imgs/blogs/treasury-auctions-and-refunding-supply-as-an-event-4.png)

Notice the second line on that chart — net interest outlays — because it is the doom-loop hiding inside the supply story. As the debt grows *and* the yield on it rises, the government's interest bill explodes, which *widens* the deficit, which forces *more* issuance, which pushes yields *higher* still. Let's make the scale of that interest bill concrete, because it is the engine of the whole feedback loop.

#### Worked example: the interest bill on the national debt at 5% vs 2%

Take a round \$35 trillion of federal debt (roughly the 2024 level) and ask what the annual interest costs at two different average yields. This is illustrative — the *average* coupon on the existing stock of debt lags the market because old low-coupon bonds are still outstanding — but it shows why rising yields are a supply accelerant.

- At a **2%** average yield: \$35,000,000,000,000 × 0.02 = **\$700 billion** a year.
- At a **5%** average yield: \$35,000,000,000,000 × 0.05 = **\$1,750,000,000,000** — about **\$1.75 trillion** a year.
- The difference: roughly **\$1.05 trillion** of extra interest a year, just from a 3-point rise in the average rate.

That extra \$1.05 trillion doesn't get paid from thin air — it gets *borrowed*, by issuing still more bonds. *Intuition: higher yields make the debt more expensive to service, which forces more issuance, which pushes yields higher — supply and the interest bill chase each other in a loop, and that loop is precisely what the bond market was pricing in autumn 2023.* (In FY2024, net interest actually exceeded the entire national-defense budget for the first time.)

Now overlay the timing. Through August, September and October 2023, three things happened at once. First, the Treasury's August QRA revealed heavier-than-expected coupon issuance — supply was about to flood in. Second, Fitch had downgraded the U.S. credit rating in early August, putting the fiscal trajectory on the front page. Third, the Fed was running **quantitative tightening**, shrinking its balance sheet and pulling itself out of the market as a buyer (more on that next). The result was a near-relentless climb in the 10-year yield from about 4.0% in early August to roughly **4.99% intraday on October 19, 2023** — its highest since 2007. The chart below traces it.

![Line chart of the 10-year US Treasury yield from 2022 to 2024 with the October 2023 peak near 5 percent annotated](/imgs/blogs/treasury-auctions-and-refunding-supply-as-an-event-2.png)

The point worth hammering: across that climb, the Fed funds rate did not move. Inflation was falling — core PCE had come down from its 2022 peak. The thing that *changed* was the supply picture. This is the cleanest real-world demonstration in modern markets that **supply moves yields independently of the Fed and of inflation.**

#### Worked example: riding the 2023 run to 5% in a 10-year position

Suppose in early August 2023 you bought \$1,000,000 face value of the 10-year note at a 4.00% yield, expecting the Fed's pause to cap rates. Then the supply wave hit and the yield ran to roughly 5.00% by mid-October — a **+100 bp** move against you. What did the supply story cost?

- The DV01 of \$1,000,000 face of a 10-year at this yield is about **\$430 per basis point** (a 10-year has roughly 8 years of duration; \$1,000,000 × ~8.6 years × 0.0001 ≈ \$860 of price per bp on a duration basis, but on a clean par-position DV01 convention we use ~\$430/bp for the round figure here).
- Yield move: **+100 bp** (yields up, prices down — painful for a long).
- Mark-to-market: 100 bp × \$430/bp = **−\$43,000**.

A \$1,000,000 position lost about **\$43,000** as yields climbed 100 bp — and not one basis point of that was a Fed hike. *Intuition: a long-only bond investor who ignored the supply calendar and watched only the Fed got run over by an event that wasn't on their radar at all.*

## How supply interacts with the Fed: QT adds to the wave

There is one more actor in the supply story, and it is — ironically — the Fed, but in a role nobody talks about. For years the Fed was the bond market's *whale*: under **quantitative easing (QE)**, it created money and bought trillions of Treasuries, soaking up a huge share of every auction's worth of issuance. A buyer that doesn't care about price — that buys *because policy says to*, not because the yield is attractive — is the best friend a borrower can have. While QE was running, the Treasury could flood the market with bonds and the Fed would quietly absorb a chunk of it, keeping yields down.

**Quantitative tightening (QT)** is the reverse. Starting in 2022, the Fed let its bonds *mature without replacing them*, shrinking its balance sheet from a peak near \$9 trillion toward \$6.6 trillion. That does two things to supply at once. It removes the Fed as a buyer — the price-insensitive whale swims away. And, more subtly, when the Fed's holdings mature, the Treasury has to *re-issue* those bonds to the public to raise the cash to pay the Fed back, so QT effectively *adds* to the net supply the public must absorb. Rising issuance from a fat deficit, plus the Fed exiting as a buyer, plus QT pushing extra supply onto the public — three forces all leaning the same way, opening a **demand gap**. Someone has to absorb the bonds, and the only buyers left are price-*sensitive*: pensions, hedge funds, foreign investors. They will buy — but only at a higher yield. The figure below shows how the pieces combine.

![Graph showing QT removing the Fed as a buyer while issuance rises, opening a demand gap that forces yields higher](/imgs/blogs/treasury-auctions-and-refunding-supply-as-an-event-6.png)

### Term premium: the price tag on too much supply

There's a piece of jargon worth decoding here because it's the cleanest way bond people talk about the supply effect: the **term premium**. The yield on a 10-year bond can be split into two parts. The first is the average short-term rate the market expects the Fed to set over the next ten years — that's the "expectations" component, and it's a pure demand-side, Fed-driven number. The second is the **term premium**: the *extra* yield investors demand, on top of those expected short rates, simply for the risk and inconvenience of locking their money up for ten years instead of rolling overnight. The term premium is where supply lives.

When there are too many bonds and too few price-insensitive buyers, the term premium rises — investors say, "fine, I'll hold all this long paper, but you have to pay me more for it." For most of the 2010s the term premium was actually *negative* (QE was crushing it), which is why huge deficits coexisted with low yields. In 2023 it turned sharply positive, and estimates of the rising term premium accounted for a large share of that climb toward 5%. So when an analyst says "term premium is repricing higher," translate it as: *the market is demanding more yield to absorb the supply.* It's the same supply story in a more technical wrapper, and it's the variable that the Fed cannot directly control — the Fed sets the expectations component via the policy rate, but the term premium is set by the supply-and-demand for duration. That's the deepest reason the autumn-2023 move was a supply event the Fed couldn't have stopped by holding rates.

This is the structural reason supply matters *more now* than it did for most of the post-2008 era. Through the 2010s, the Fed was either buying (QE) or holding a vast portfolio steady, and the deficit, while large, was being partly monetized. The combination that defined 2022–2024 — a 6%-of-GDP deficit *and* an actively shrinking Fed balance sheet — left the public to absorb a record gross supply with the whale gone. When you read that "term premium is rising," this is largely what it means: investors are demanding extra yield to hold long bonds precisely because there are so many of them and so few price-insensitive buyers. The macro-trading series develops the mechanism in depth; here the trader's takeaway is enough: *QT is a supply story wearing a monetary-policy costume, and it stacks on top of fiscal issuance.*

#### Worked example: fading an auction overreaction

Not every supply scare is real, and the contradictions are where nimble traders make money. Suppose a 7-year auction posts a headline **+2 bp tail** that triggers a knee-jerk selloff — the 7-year yield jumps 5 bp in the minute after 1:01 p.m. But the internals contradict the tail: bid-to-cover came in *above* its average and indirects were *strong*. That combination says the demand was actually fine and the tail was a mechanical quirk. You judge the 5 bp spike is overdone and put on a fade.

- You buy \$200,000 face of the 7-year, positioned for the yield to retrace lower (price higher).
- The DV01 on \$200,000 face of a 7-year is about **\$130 per basis point**.
- The yield retraces **5 bp** back toward the when-issued level over the next hour.
- Profit: 5 bp × \$130/bp ≈ **\$650** — and if you'd sized this as a \$430/bp position (about \$660,000 face), the same 5 bp recovery would be about **+\$2,150**.

The fade worked because the headline tail and the internals disagreed, and the internals were right. *Intuition: follow the auction when all three numbers agree and fade it when the headline tail is contradicted by strong cover and indirects — the move that's built on a number the internals don't support is the one that retraces.*

## How it reacted: real episodes

Enough theory — here is the supply story playing out on the tape, with dated numbers.

### The August–October 2023 surge toward 5%

This is the canonical episode and we have already walked through it, but let's pin the reaction sequence. The **August 2, 2023** Fitch downgrade and the **August 2, 2023** QRA (which lifted coupon-issuance guidance more than expected) lit the fuse. Over the following ten weeks the 10-year yield climbed roughly **100 bp**, from about 4.0% to **4.99% intraday on October 19, 2023** — the highest since 2007 — even as the Fed held steady and inflation cooled. The 30-year crossed 5% in the same window. Crucially, several long-dated auctions in that stretch **tailed**, confirming in real time that buyers were balking: a soft 30-year auction in particular added fuel, each weak result reinforcing the "who's going to buy all this?" fear. For a \$1,000,000 10-year long position, that 100 bp move was the roughly **−\$43,000** mark we computed above — a supply-driven loss with no Fed hike behind it.

Then came the reversal, and it was *also* a supply story. The **November 1, 2023 QRA** surprised dovishly — the Treasury leaned more on bills and grew coupon sizes less than feared. The long end rallied hard on the announcement. Combined with a Fed meeting that markets read as the end of hikes and a soft October jobs report on November 3, the 10-year yield fell sharply from its ~5% peak back toward 4.5% within weeks. The QRA marked both the top *and* the turn. If you were short bonds into the November refunding expecting more bad supply news, the dovish surprise was a fast, painful squeeze; if you faded the 5% panic, it was the trade of the quarter.

### Weak long-bond auctions as standalone shocks

Beyond the headline 2023 episode, individual weak auctions have repeatedly jolted the tape on their own. A long-bond (30-year) auction is the most sensitive because the 30-year has the most duration and the thinnest natural demand base — there are only so many pensions and insurers who *need* 30-year paper. When a 30-year auction tails several basis points with weak cover, the 30-year yield can gap up 5–8 bp in minutes, dragging the 10-year and even equities lower as the discount rate on every long-duration asset (growth stocks, especially) ticks up. These are textbook supply events: no economic data, no Fed, just an auction revealing that the buyers weren't there at the assumed price. Conversely, a string of *strong* auctions in late 2023 and 2024 — stops-through with fat indirects — quietly underwrote bond rallies, removing the supply tail risk and letting yields drift down.

The structural backdrop made every one of these auctions matter more: a 6%-of-GDP deficit, debt past \$33 trillion climbing toward \$35 trillion, net interest outlays nearly tripling from \$345 billion (2020) to roughly \$880 billion (2024), and the Fed in QT the whole time. With that much paper to place and the whale gone, the market had no slack — so each auction became a referendum on whether demand could keep up with supply.

### The cross-asset spillover: supply hits more than bonds

A supply-driven rise in long yields doesn't stay in the bond market — it leaks into every asset whose value depends on the discount rate. When the 10-year ran toward 5% in October 2023, the S&P 500 fell roughly 8% from its late-July high into late October, with the damage concentrated in the longest-duration equities: high-growth tech and unprofitable speculative names, whose value sits in far-future cash flows that a higher discount rate slashes hardest. The mechanism is the same one that makes a 30-year bond more rate-sensitive than a 2-year: the further out the cash flows, the more a higher yield hurts. So a supply-driven yield spike is, indirectly, a growth-stock event.

It hit the dollar and emerging markets too. Higher U.S. yields pulled global capital toward Treasuries — why take currency and credit risk in an emerging market when the U.S. government pays you 5% risk-free? The dollar strengthened through the autumn-2023 yield surge, and emerging-market currencies and equities felt the squeeze. Vietnam is a clean example of the channel: as U.S. yields climbed and the dollar firmed in 2023, the **State Bank of Vietnam** faced pressure on the dong (the USD/VND rate drifted from about 23,600 at end-2022 toward 24,300 by end-2023), constraining how much the SBV could cut its own policy rate to support growth. Foreign investors net-sold Vietnamese equities, and the VN-Index — which had recovered to around 1,130 by end-2023 from its 2022 trough near 911 — found its rally capped partly by the gravitational pull of high U.S. yields. The lesson generalizes: when U.S. Treasury supply pushes the world's benchmark risk-free rate up, it raises the bar for every risk asset on the planet, from a Nasdaq growth stock to a Ho Chi Minh City-listed bank. Supply in Washington becomes a headwind in Hanoi.

#### Worked example: the equity drag from a supply-driven yield spike

Suppose you hold \$25,000 of a long-duration tech ETF and the 10-year yield jumps 40 bp over two weeks on a string of weak auctions and a heavy QRA. Long-duration equities have historically moved sharply on rate spikes; say this basket falls 6% on the move (a typical magnitude for high-growth names on a 40 bp rate shock).

- Position: **\$25,000**.
- Move: **−6%** as the discount rate jumps.
- Mark-to-market: \$25,000 × −0.06 = **−\$1,500**.

A bond-supply event you weren't even watching just cost an *equity* position **\$1,500**. *Intuition: the supply calendar isn't only a bond-desk concern — a weak auction raises the discount rate on every future cash flow, so it reaches all the way into your stock portfolio.*

## Common misconceptions

**Myth 1: "Only the Fed moves yields."** This is the big one, and 2023 demolished it. From August to October 2023 the Fed funds rate did not move and inflation *fell*, yet the 10-year yield rose roughly **100 bp** to ~4.99%. The mover was supply — a record issuance wave hitting a market with the Fed exiting as a buyer. The Fed sets the *short* end (the overnight rate); the *long* end is set by the supply-and-demand for duration, and supply is a first-order driver. A \$1,000,000 10-year long lost about **−\$43,000** in that window with zero Fed hikes behind it.

**Myth 2: "An auction is just plumbing; it can't move the market."** A weak auction is *new information* about demand, and the market reprices off it in seconds. A +3 bp tail on a 30-year cost a \$500,000 long position about **−\$3,600** in two minutes in our example. Multiply that across the trillions of duration on the Street and a tail is a genuine, tradeable event — one that happens on a 1:00 p.m. clock most equity traders ignore.

**Myth 3: "A high bid-to-cover means a strong auction."** Not in isolation. A 2.3× cover sounds healthy until you learn the trailing average is 2.6× — then it's a soft auction. Every auction number is read *versus its own benchmark*: the tail vs when-issued, the cover vs the 6-auction average, the indirects vs recent auctions. The level alone tells you nothing.

**Myth 4: "More debt automatically means much higher yields, always."** Supply is one force among several. Japan ran debt above 200% of GDP for years with yields near zero, because the Bank of Japan was an infinite price-insensitive buyer (QE) and domestic demand was deep. Supply pushes yields up *when there is no whale to absorb it* — which is exactly why QT made 2023 different. Supply and demand both matter; the regime decides which dominates.

**Myth 5: "The QRA is the same as an auction."** No — the QRA is the *announcement of the plan*, weeks before the auctions. The market reprices the supply on the *announcement*, so by the time the auctions arrive they're often already in the price. The November 1, 2023 QRA moved the long end on the news, then marked the *top* in yields — the announcement was the event, not the sales.

## The playbook: how to trade supply as an event

Here is the operational map. Treat the supply calendar like any other event: know the consensus, trade the surprise, define the invalidation.

**Know the calendar.** Mark the recurring auctions — the Treasury publishes the schedule weeks ahead. The marquee ones for traders are the **10-year note** and **30-year bond** auctions (usually mid-month, 1:00 p.m. ET), because they carry the most duration. And mark the four **QRA** dates (first Wednesday of February, May, August, November, with the borrowing estimate the prior Monday). These are scheduled events with a hard clock, exactly like CPI or the FOMC.

**Establish the consensus.** For an auction, the consensus is the **when-issued yield** and the trailing averages for bid-to-cover and indirects. For the QRA, it's the analysts' forecast for the quarterly borrowing estimate and the expected coupon-auction-size schedule. You can only trade a surprise if you know what was expected.

![Decision graph for reading an auction: tail, bid-to-cover and indirects branch into follow or fade the yield move](/imgs/blogs/treasury-auctions-and-refunding-supply-as-an-event-7.png)

**The auction scenarios.**

- **Weak auction (fat tail + low cover + soft indirects) → follow.** All three agree on weakness. Lean **short bonds** (positioned for higher yields). In risk assets, a weak long-bond auction is a mild headwind for long-duration equities (growth/tech) as the discount rate rises, and a tailwind for the dollar if the move is orderly. *Invalidation:* yields snap back through the when-issued level within the hour, which says the tail was a fluke.
- **Strong auction (stop-through + high cover + fat indirects) → follow the rally.** Lean **long bonds** (positioned for lower yields). A well-bid auction removes a supply tail risk and is mildly risk-on. *Invalidation:* yields keep rising despite the strong result, which says a bigger force (a hot data print, a hawkish Fed headline) is overriding the supply signal.
- **Contradicted tail (fat tail BUT strong cover + high indirects) → fade.** The internals say demand was fine; the headline tail was mechanical. Fade the knee-jerk selloff — buy the dip in bonds, positioned for the spike to retrace. This is the highest-conviction *fade*. *Invalidation:* yields hold the spike for more than an hour or push higher, meaning the tail was real after all.

**The QRA scenarios.**

- **More coupon supply than expected → bearish bonds.** The long end sells off on the announcement; lean short the long end (30-year, 10-year). This was the August 2023 setup.
- **More bills / less coupon than expected → bullish bonds.** The long end rallies on relief; lean long. This was the November 2023 setup that marked the top.

**Sizing and risk.** Auctions are fast, two-minute events — size for a quick, sharp move and use tight invalidations (the when-issued level for a tail trade). DV01 is your sizing unit: decide how many dollars per basis point you're willing to risk, then back out the face value. From our examples, \$500,000 of the 30-year is ~\$1,200/bp; \$1,000,000 of the 10-year is ~\$430/bp; \$200,000 of the 7-year is ~\$130/bp — pick the instrument that gives you the DV01 you want. For the QRA, the move is bigger and slower (it plays out over the morning and the following auctions), so you can size larger with a wider stop. Above all, **respect the regime**: in a world of fat deficits and QT, the market has no slack, so supply surprises bite harder and trend further than they did in the QE era. When the whale is gone, every auction is a referendum.

## Further reading and cross-links

Supply is the demand-side calendar's quiet twin. To see how it fits the rest of the macro picture, read these companion posts:

- [Treasury issuance: bills, coupons and the liquidity drain](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain) — the plumbing of how issuance pulls cash out of the system and why the bill-vs-coupon mix matters.
- [Deficits, debt and bond supply: why issuance moves yields](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields) — the deficit → issuance → yields chain in full, the mechanism behind this post.
- [Sovereign debt and the bond vigilantes](/blog/trading/macro-trading/sovereign-debt-and-the-bond-vigilantes) — when the market itself disciplines a government's borrowing by demanding higher yields.
- [Reading the yield curve: slope, inversion and recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — how supply at different maturities reshapes the curve's slope.
- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — the founding idea of this series: markets trade the surprise versus consensus, and an auction's tail is exactly that surprise.

The lesson to carry away: most traders stare at the demand-side calendar — inflation, jobs, the Fed — and never glance at the supply calendar that sits right beside it. But the bonds have to be *sold*, every week, at auction, and when the buyers balk the yield jumps. In autumn 2023 nothing about the Fed changed and inflation was falling, yet the 10-year marched to 5% because the market was drowning in supply. Watch the auctions. Read the tail. **Supply is the event.**
