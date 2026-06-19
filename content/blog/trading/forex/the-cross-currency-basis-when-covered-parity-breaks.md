---
title: "The Cross-Currency Basis: When Covered Parity Breaks"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why the rate gap and the forward stopped lining up after 2008 — the cross-currency basis as the market price of a dollar shortage, a real arbitrage that won't close."
tags: ["forex", "currencies", "cross-currency-basis", "covered-interest-parity", "dollar-funding", "fx-swaps", "arbitrage", "balance-sheet", "liquidity", "crisis"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The cross-currency basis is the gap that opens when the FX forward stops matching the interest-rate difference between two currencies; it is the residual left over after covered interest parity (CIP) fails, and it is almost always negative, meaning the rest of the world pays *more* than the textbook says to get its hands on dollars.
>
> - **CIP says** the forward exchange rate is pinned by the two countries' interest rates — there should be no leftover wedge. For decades it held to within a basis point or two.
> - **Since 2008 it does not.** A persistent, mostly negative basis means anyone funding in dollars via the FX swap market pays an extra few-to-many basis points over the "fair" CIP rate. That extra cost *is* the basis.
> - **It is a real arbitrage that won't close**, because the trade that would close it eats a balance-sheet-constrained bank's scarcest resource (leverage capacity), so the riskless profit sits there unharvested.
> - **The number to remember:** in calm markets the EUR/USD basis is around −15 bps; in the 2008 crash the USD/JPY basis hit about **−220 bps**, and in March 2020 the EUR/USD basis blew out to about −85 bps. The basis is the market's live price for a dollar shortage.

On the morning of 17 March 2020, a European pension fund treasurer tried to do the most boring thing in finance: roll a three-month FX swap to keep her dollar hedge in place. She held US bonds, she had hedged the currency, and every quarter she rolled the hedge forward. The screen quoted her a price that, translated into an interest rate, meant she was effectively borrowing dollars at far above the US money-market rate — about 85 basis points above what the difference between euro and dollar interest rates said she *should* pay. Nothing about euro rates had changed. Nothing about dollar rates had changed. The "law" that connects spot, forward, and the two interest rates — the one every textbook calls an arbitrage that cannot be violated — was being violated, in size, in front of her, on a Bloomberg screen.

She was not seeing a glitch. She was seeing the cross-currency basis, and on that particular morning it was screaming that the entire planet was short of dollars at once. The same wedge had appeared in the autumn of 2008, when the USD/JPY version of it gapped out past minus two hundred basis points. It reappeared in the 2011 euro-area crisis. It is, in the calm of a normal Tuesday, a quiet few basis points; in a panic it becomes one of the cleanest stress gauges in all of markets — a thermometer for how badly the world wants the one currency it cannot print.

What makes the cross-currency basis such a satisfying thing to understand is that it sits at the intersection of three ideas that most people meet separately: the no-arbitrage logic that pins forwards to interest rates, the post-2008 reality that bank balance sheets are no longer free, and the brute geopolitical fact that the world runs on a currency only one country can print. The basis is where those three collide and leave a residue you can read off a screen in basis points. Master it and you have a single lens for a surprising amount of modern markets — why a hedged bond yields less than its headline, why funding crises announce themselves in the swap market first, why the Fed has quietly become the central bank for the entire planet, and why "riskless arbitrage" is a phrase that needs an asterisk the size of a balance sheet.

This post is about that wedge. We will build it from zero: what covered interest parity is, why the forward is *supposed* to be pinned by the rate gap, what the "basis" actually measures when that pinning fails, why it persists for years instead of being instantly arbitraged away, and why it explodes in every dollar-funding crisis. The thread that runs through the whole series — *you never own a currency in isolation; what moves it is the gap between two countries' interest rates plus the flow of money across borders* — has a sharp corollary here: when the flow of dollars dries up, even the iron link between rates and the forward bends. The basis is where that bending shows up as a price.

![Cross-currency basis for EUR USD and USD JPY 2008 to 2024 with crisis spikes](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-1.png)

## Foundations: The cross-currency basis

Before we can talk about a *deviation*, we need the thing it deviates from. So start with the four numbers that sit at the heart of all currency pricing, and the law that ties them together.

### Spot, forward, and the two interest rates

A **spot** exchange rate is the price for currency delivered now (technically in two business days). EUR/USD at 1.0800 means one euro costs 1.08 dollars today. Here the **base** currency is the euro (the thing being priced) and the **quote** currency is the dollar (the thing it is priced in) — if that base/quote convention is new to you, the post on [reading an FX quote](/blog/trading/forex/base-quote-pips-and-how-to-read-an-fx-quote) walks through it slowly. A **forward** exchange rate is the price agreed today for currency delivered on a fixed future date — say three months or a year out. The forward is not a forecast; it is a contract price you can lock in right now. (For the full mechanics of forwards and how a swap is just a spot leg plus an offsetting forward leg, see the sibling post on [spot, forward, and swap](/blog/trading/forex/spot-forward-and-swap-the-three-ways-to-trade-a-currency).)

Then there are two **interest rates** — one for each currency. If you hold dollars for a year you earn the dollar rate; if you hold euros you earn the euro rate. Call them \$r_{usd}\$ and \$r_{eur}\$. In our running example, the one-year dollar rate is 5.0% and the one-year euro rate is 3.0%.

The whole edifice rests on a single question: if I have euros today and I will need dollars in a year, I have two ways to get there. I can swap to dollars now and earn the dollar rate for a year, or I can keep euros, earn the euro rate, and lock a forward to convert at year-end. Both routes start with the same euros and end with dollars at the same future date. **If they delivered different amounts of dollars, that difference would be free money** — borrow on the cheap route, lend on the rich one, pocket the gap with no risk. So the forward must adjust until the two routes tie. That requirement is covered interest parity.

### Covered interest parity, stated once and carefully

Covered interest parity (CIP) says the forward rate equals the spot rate scaled by the ratio of the two gross interest rates. With the euro as base and the dollar as quote:

```
F = S * (1 + r_usd) / (1 + r_eur)
```

Plug in the numbers: spot \$S = 1.0800\$, dollar rate 5%, euro rate 3%.

```
F = 1.0800 * (1.05 / 1.03) = 1.0800 * 1.01942 = 1.1010
```

So CIP says the one-year EUR/USD forward must be **1.1010**. The euro trades at a forward *premium* (1.1010 is above the 1.0800 spot) precisely because the dollar carries the higher interest rate: a currency with the higher rate trades at a forward *discount*, the lower-rate currency at a premium, so that the extra interest you earn is exactly given back through a worse exchange rate later. The "covered" in the name means the future exchange rate is locked — covered — by the forward, so there is no currency risk in the comparison. (The *uncovered* cousin, which leaves the rate unhedged and turns out to fail in a completely different way, is the subject of its own post on the [forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle).)

The word to underline is **must**. CIP is not a behavioural theory like "high-rate currencies appreciate." It is a no-arbitrage identity: if the forward is anything other than 1.1010, you can construct a portfolio of spot, forward, and two deposits that prints a riskless profit. For most of the post-1980s era, the market enforced this so tightly that the realised basis sat within one or two basis points of zero. Quants treated CIP the way physicists treat conservation of energy.

It is worth dwelling on *why* the link was so airtight before 2008, because that tells you exactly what changed. The arbitrage that enforces CIP — borrow the cheap currency, swap into the other, lend it, lock the forward back — was, in the pre-crisis world, both riskless *and* nearly free to execute. Banks could expand their balance sheets at will; an extra billion dollars of perfectly hedged assets cost essentially nothing in regulatory capital, because capital was assessed on *risk* and this trade had none. So whenever the smallest wedge appeared, dealers piled in without limit and crushed it back to zero within minutes. The market behaved like a textbook precisely because the assumption the textbook makes — that arbitrageurs face no balance-sheet constraint — was, for those years, very nearly true. CIP held not because of magic but because the cost of enforcing it was zero. When that cost stopped being zero, so did the basis.

### The FX swap is how the world borrows dollars without an interbank loan

One more building block, because the basis lives on it. An **FX swap** is two trades stapled together: you sell euros for dollars at spot today, and simultaneously agree to buy those euros back for dollars at the forward rate on a future date. The net effect is that, for the life of the swap, you have given someone euros and received dollars — you have *borrowed* dollars and *lent* euros, fully collateralised by the currencies themselves. No credit line, no unsecured loan, no trust required: each side is holding the other's currency the whole time. That collateralised quality is exactly why the FX swap market stays open when the unsecured interbank market slams shut in a crisis — and why it becomes the *only* door to dollars when fear takes hold.

The price of that dollar borrowing is baked entirely into the difference between the spot leg and the forward leg, the **forward points**. If the forward points are set by CIP, your synthetic dollar borrowing costs exactly the dollar money-market rate. If the market demands forward points *richer* than CIP — because dollars are scarce — then your synthetic borrowing costs more, and the excess is the basis. So the basis is not some exotic derivative; it is the embedded interest rate on the single most-used funding instrument on the planet. The BIS Triennial Survey puts FX swaps at the largest slice of the entire \$7.5-trillion-a-day FX market, dwarfing spot. When people say "FX is mostly swaps," this is why it matters: most of that volume is the world rolling its dollar funding, and the basis is the spread it pays.

#### Worked example: building the CIP forward, then converting €10,000,000

You hold **€10,000,000** today and you will need dollars in one year. Route A — convert now: €10,000,000 × 1.0800 = **\$10,800,000** today, deposited at 5% grows to \$10,800,000 × 1.05 = **\$11,340,000** in a year. Route B — stay in euros: €10,000,000 at 3% grows to €10,300,000, then converted at the CIP forward 1.1010 gives €10,300,000 × 1.1010 = **\$11,340,300** — the same \$11.34 million, give or take rounding. The two routes tie to the dollar, which is *why* the forward had to be 1.1010 and not, say, 1.0950. **CIP holds when both ways of turning euros into future dollars hand you the identical amount; the forward is the number that forces that tie.**

![Before and after showing covered interest parity holding versus a basis residual](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-2.png)

### The basis: the residual the market leaves behind

Now the punchline. Go to a real screen and read three of those four numbers directly: the spot, the dollar money-market rate, and the euro money-market rate. Then read the *fourth* — the actual traded forward — instead of computing it. Since 2008, the forward you read is not the forward CIP predicts. There is a leftover wedge.

The **cross-currency basis** is that wedge, expressed as the extra interest rate you have to add to one side of the CIP equation to make it balance against the *observed* forward. Write it as \$b\$, conventionally added to the non-dollar (here euro) leg:

```
F_observed = S * (1 + r_usd) / (1 + r_eur + b)
```

If the observed forward is *higher* than the CIP forward — the euro is dearer to buy forward than parity says — then \$b\$ must be **negative** to make the equation fit. A negative basis is the normal state of the world, and it has a brutally concrete meaning: **whoever is using the swap market to turn their currency into dollars is effectively earning a lower euro rate (\$r_{eur} + b\$, with \$b<0\$) than the actual euro deposit rate.** They are accepting a worse deal on their own currency in exchange for getting the dollars. Equivalently, and more usefully: a non-US institution borrowing dollars synthetically through the FX swap market pays the dollar rate *plus the absolute value of the basis*. A basis of −50 bps means you pay roughly half a percent a year more for your synthetic dollars than the US money-market rate — and more than CIP says you should.

So the sign is everything. **Negative basis = the market charges a premium to deliver dollars = a dollar shortage outside the United States.** The basis is not a number floating above the market; it is a price a treasurer pays, in cash, every time she rolls a hedge. The cover chart above shows the whole story at a glance: both the EUR/USD and USD/JPY bases live below zero, drift toward zero in calm years, and crash to deeply negative readings in 2008, 2011, and 2020. Every one of those crashes is a moment the world ran short of dollars.

#### Worked example: the extra cost of swapping €10,000,000 into dollars at a −50 bps basis

Take the same €10,000,000, but now the one-year EUR/USD basis is **−50 bps**. You want one-year dollar funding via the FX swap. Under pure CIP you would pay the 5.0% dollar rate. With the basis, your *all-in* synthetic dollar borrowing cost is the dollar rate plus the basis you give up on the euro side — about 5.0% + 0.50% = **5.50%** on the dollars you raise. On a \$10,800,000 swapped notional, that extra 50 bps is 10,800,000 × 0.0050 = **\$54,000 a year** more than CIP says you should pay. Flip it around: a US fund with spare dollars who swaps them out to a euro borrower *earns* that 50 bps — \$54,000 of riskless pickup on \$10.8m, paid by the dollar-short side. **The basis is a cash transfer from the dollar-needy to the dollar-rich, and −50 bps on €10m is real five-figure money every year.**

### How the basis is actually quoted and read

In practice traders rarely speak in observed-versus-CIP forward levels; they quote the basis directly, in basis points, off two reference curves. The cleanest measure uses **OIS** rates — overnight-indexed-swap rates, which strip out term bank-credit risk and proxy the "pure" risk-free rate in each currency. You compare the cost of borrowing dollars directly at dollar OIS against the cost of borrowing dollars *synthetically* — borrow euros at euro OIS, swap to dollars via the FX swap — and the difference is the cross-currency basis. Because both legs are referenced to near-risk-free OIS curves, what is left over cannot be explained away as one country's banks being riskier than another's; it is the swap-market premium for dollars, full stop. That is the number plotted on the cover chart and the yen-basis chart in this post.

There is also a traded instrument that carries the basis as its own line item: the **cross-currency basis swap**, in which two parties exchange floating-rate payments in two currencies (say, dollar SOFR against euro €STR) over several years, with the basis added explicitly to the non-dollar leg as the market-clearing spread. When a five-year EUR/USD basis swap quotes at −25 bps, that is the market saying: to receive dollars and pay euros floating for five years, you must accept 25 bps less on your euro leg. The short-dated FX-swap basis and the longer-dated basis-swap spread are two windows onto the same dollar-funding premium — one for rolling funding over days and weeks, the other for locking it over years. Both are negative for the same reason, and both blow out together when dollars get scarce.

The two windows also tell you slightly different things. The short-dated FX-swap basis is the live thermometer — it reacts within hours to a funding squeeze and is the one to watch in a crisis. The multi-year basis-swap spread is structural — it reflects the market's view of the *average* cost of the dollar shortage over years, and it is what a long-term hedger or a bond investor actually locks in. A treasurer hedging a ten-year dollar liability cares about the ten-year basis; a money-market desk caught in a panic cares about the one-week. They co-move, but the term structure of the basis — how it slopes from one week to ten years — is itself informative, steepening at the front in acute squeezes and flattening out as the market prices the friction as permanent rather than panic-driven.

## Why CIP broke after 2008

For the entire era when banks could expand their balance sheets almost for free, CIP was airtight. The arbitrage that enforces it — borrow dollars, swap into euros, lend euros, lock the forward back — was riskless and essentially costless, so dealers did it in unlimited size whenever the smallest wedge appeared, and the wedge vanished. Then the 2008 crisis rewrote the cost of doing that trade, and it has never gone back.

### The arbitrage used to be free; now it uses up a scarce resource

The CIP arbitrage is "riskless" in market-risk terms — you are perfectly hedged on rates and FX. But it is *not* free on the balance sheet. To do it you must borrow dollars (a liability), hold a euro asset, and post collateral on the forward. Every one of those positions expands your balance sheet, and after 2008 a bank's balance sheet stopped being free.

Three things changed at once. First, **counterparty risk became real**: in 2008 banks discovered that lending dollars to another bank, even overnight, could mean not getting them back, so the interbank dollar market — the wholesale plumbing through which the world used to source dollars — froze. Second, **regulation put a hard price on balance-sheet size**: the post-crisis leverage ratio (and the supplementary leverage ratio in the US) caps how big a bank's total assets can be relative to its equity, *regardless* of how safe those assets are. A perfectly hedged CIP arbitrage still consumes leverage-ratio capacity, and that capacity now has a real cost of equity attached. Third, **dollar funding itself got scarcer and lumpier**, concentrated in money-market funds and a handful of US banks whose willingness to lend out dollars varies with their own constraints.

The result: the trade that used to be a free lunch now costs the arbitrageur a slice of his most precious resource — room under the leverage cap. He will only do it if the expected pickup exceeds that cost. So the basis can sit at −15, −30, −50 bps indefinitely: that is roughly the price of the balance sheet the trade consumes. CIP didn't get repealed; the *cost of enforcing it* went from zero to positive, and the basis is what fills the gap.

![Pipeline showing dollar demand exceeding supply leading to a negative basis](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-3.png)

### It takes a one-sided imbalance, not just a friction

Balance-sheet cost explains why the basis *can* be non-zero, but not why it is almost always *negative* rather than randomly positive or negative. For that you need a standing, one-directional imbalance in who wants to swap which way — and there is one. The world outside the United States holds a vast stock of dollar-denominated assets and dollar liabilities, far larger than the dollar deposits sitting offshore to fund them. Japanese life insurers buying US Treasuries, European banks holding US mortgage securities, emerging-market corporates with dollar bonds to service, reserve managers running dollar portfolios — all of them are *structurally* trying to source dollars against their own currency. Net, the non-US world is long dollar assets and short dollar funding, and it plugs the gap through the FX swap market.

That standing net demand to receive dollars in swaps is what makes the basis lean negative. If the imbalance were symmetric — as many people wanting to swap *out* of dollars as into them — the balance-sheet friction would widen the bid-offer but leave the mid near zero. It is the *combination* of a real friction (balance sheet is no longer free) and a real imbalance (the world is net short dollar funding) that produces a persistent negative basis. Remove either and it collapses: in a world of free balance sheets, arbitrage closes any imbalance; in a balanced world, friction alone has nothing to push against. The basis is the product of both, which is why it is a *dollar* phenomenon specifically and not a generic feature of every currency pair. The euro, yen, sterling, and franc all show it against the dollar; the cross-rates between two non-dollar currencies show far less, because neither side is the world's reserve currency that everyone is scrambling to fund in.

### Quarter-ends and year-ends: when the cost spikes on a calendar

If the basis were purely about crises, it would be quiet between them. It is not — it has a *calendar*. Watch the EUR/USD or USD/JPY basis into any quarter-end, and especially into year-end, and you will see it lurch more negative in the final days, then snap back in January. Nothing about the world's dollar needs changes on 31 December. What changes is that many banks report their balance-sheet size on that single snapshot date, and some regulators and home supervisors assess leverage ratios on the year-end print. A bank that does not want a fat balance sheet on the reporting date pulls back from balance-sheet-using trades — including the CIP arbitrage — right when the snapshot is taken. With fewer arbitrageurs willing to warehouse the trade across the turn, the basis gaps out.

This is one of the cleanest pieces of evidence that the basis is a *balance-sheet* phenomenon, not a rates or FX phenomenon. Rates and spot do not care what day of the quarter it is; balance-sheet capacity does. The year-end spike is a calendar tax on the people who need dollars over the turn, and its existence is almost a controlled experiment proving the mechanism.

#### Worked example: the year-end turn costs a Japanese bank an extra 30 bps overnight

A Japanese bank funds part of its US-Treasury portfolio by swapping yen into dollars, rolling short FX swaps. In mid-December the one-week USD/JPY basis sits around −35 bps. As the trade rolls across 31 December — the regulatory snapshot — the one-week basis for the turn lurches to roughly −65 bps because so few dealers will lend dollars over the reporting date. On a **\$500,000,000** synthetic-dollar book, the extra 30 bps annualised over a one-week turn is 500,000,000 × 0.0030 × (7/360) ≈ **\$29,000** for one week's roll, versus a few thousand on an ordinary week. **The same trade, the same risk, costs an order of magnitude more for one week a year — because that is the week the balance sheet is being counted.**

## The basis as a dollar-funding stress gauge

Once you understand that a negative basis is the price of getting scarce dollars, the basis becomes a thermometer. In calm markets dollars are plentiful, the queue to borrow them via swaps is short, and the basis hugs zero. When fear hits, everyone with dollar liabilities and non-dollar assets scrambles for dollars at once, the swap market is the only door still open, and the price to get through it — the basis — blows out. The size of the blow-out is a direct read on the severity of the dollar shortage.

### What the gauge reads in calm versus crisis

The contrast is stark and quantitative. End of 2019, a calm year: the three-month EUR/USD basis was around **−15 bps** and the USD/JPY basis around **−40 bps** — small frictions, the ordinary balance-sheet tax. Then March 2020 hit: as the pandemic froze markets and every corporate treasurer, foreign central bank, and leveraged fund reached for dollars simultaneously, the three-month EUR/USD basis gapped to roughly **−85 bps** and the USD/JPY basis to around **−145 bps** within days. The dollar-funding door narrowed and the toll to pass through it multiplied five-fold.

The deepest reading on record came in the autumn of 2008. After Lehman, the offshore dollar market simply stopped: no bank trusted another bank with dollars, and the only way for a non-US institution to get them was to pledge its own currency in a swap and pay whatever was asked. The three-month USD/JPY basis touched around **−220 bps** — meaning a Japanese institution borrowing dollars synthetically paid over two full percentage points a year above the dollar money-market rate, on top of an already elevated rate. That number is the price tag on a global dollar panic.

![Before and after of the basis as a stress gauge calm versus crisis](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-7.png)

### Why the yen basis is the deepest

Across crises, the USD/JPY basis is consistently the most negative of the majors, and that is not random. Japan is the world's great structural dollar borrower: Japanese banks, insurers, and pension funds hold enormous portfolios of US and global assets that they fund, in large part, by swapping yen into dollars. Domestic yen rates have been near zero for decades, so the demand to convert that cheap yen into dollar assets is permanent and one-directional. That standing, structural, *net* demand for dollars-via-swaps presses the yen basis below the others even in calm times — and when a crisis amplifies dollar demand, the side that was already most lopsided gaps out the most.

The chart below isolates the yen basis to make the point: it lives further below zero than EUR/USD in normal years (−40 bps at end-2019 versus −15 bps for EUR/USD) and reaches the deepest crisis prints. The basis is not a single global number; it is currency-specific, and its depth tells you *who* in the world is most chronically short of dollars.

![USD JPY cross-currency basis history with the 2008 and 2020 lows](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-4.png)

### The basis versus the other funding-stress gauges

Markets have a small family of funding-stress thermometers, and the basis is the most international member. The classic domestic gauge is the **TED spread** (and its modern cousins) — the gap between unsecured bank borrowing rates and risk-free rates, which reads how nervous banks are about lending to each other in dollars. The **FRA-OIS spread** does something similar for term funding. What the cross-currency basis adds is the *cross-border* dimension: it specifically prices how hard it is to get dollars when you are *outside* the dollar system and must source them against another currency. In a purely domestic US scare, the TED spread can widen while the basis stays calm. In a global dollar scramble, all of them widen — but the basis is the one that isolates the international, non-US-bank dollar shortage.

That is why policymakers and strategists watch it. The basis cannot be talked down with reassuring words about US banks; it only narrows when actual dollar supply reaches the people who are short, which in a real panic means the Fed's swap lines. So a widening basis is a particularly *honest* gauge: it does not respond to sentiment, only to the physical availability of dollars offshore. When it gaps out, the plumbing — not the mood — has broken, and that is exactly the signal a risk manager wants, undistorted by narrative.

#### Worked example: how negative the basis must be before an arbitrageur is paid to take it

Suppose a US money-market fund has **\$100,000,000** of spare dollars and is considering lending them, via a one-year FX swap, to a euro borrower — earning the dollar rate plus the basis pickup. The trade is riskless on rates and FX, but it consumes balance sheet at the fund's intermediating bank, who charges, say, **20 bps a year** for the leverage capacity it uses — that is \$100,000,000 × 0.0020 = **\$200,000** of cost. With the basis at −15 bps, the gross pickup is \$100,000,000 × 0.0015 = **\$150,000** — *less* than the \$200,000 balance-sheet cost, so the trade loses \$50,000, nobody does it, and the basis stays put. Only when stress pushes the basis past **−20 bps** (a \$200,000 pickup) does the trade clear its costs and arbitrage capital start to lean against it. **The basis is not pulled to zero; it is pulled to the cost of the balance sheet that closing it consumes — roughly 15-25 bps in calm times, far wider when that capacity is rationed.**

### Reading the basis as an early warning, not a lagging tape

Because the basis prices the *physical* availability of dollars rather than sentiment, it tends to move early and decisively when funding genuinely tightens — often before equity indices or credit spreads have fully registered the stress. A treasurer rolling a hedge feels the squeeze in the swap price days before it shows up as a headline. That makes a steadily widening basis a useful tripwire: it says the plumbing is tightening regardless of what the risk narrative is. A practitioner watching for trouble keeps three things on one screen — the three-month basis for the major pairs, the year-end/quarter-end forward turn, and the Fed's swap-line draws — and reads them as a system. Quiet basis, flat turn, no swap draws: the dollar plumbing is calm, whatever the equity market is doing. Widening basis, gapping turn, swap lines lighting up: the plumbing is under strain, and history says that strain front-runs broader stress more often than it lags it.

The caveat is to read the *level relative to its own calm baseline*, not against zero. A EUR/USD basis at −20 bps is unremarkable; the same −20 bps would be a screaming alarm for a cross-rate that normally sits at zero. Each pair has its own resting level set by its structural dollar imbalance — the yen's deeply negative, the euro's modestly so — and the signal is in the *deviation* from that baseline, not the absolute number. Calibrate to each pair's normal, then watch for the gap-out.

## Why the arbitrage won't close

Here is the part that breaks most people's intuition. If a non-US bank borrowing dollars synthetically pays 50 bps over the fair rate, then a US bank with spare dollars is being *offered* 50 bps of riskless return. Riskless. Hedged on rates, hedged on FX, collateralised. Finance is supposed to have armies of people who exist to harvest exactly this. Why does the 50 bps just sit there, year after year, uncollected?

### The trade is riskless on paper, expensive on the balance sheet

The answer is the one we have been circling: "riskless" and "free" are not the same word. The CIP arbitrage carries no *market* risk, but it carries a heavy *balance-sheet* cost, and after the post-crisis leverage rules, balance sheet is the binding constraint at exactly the institutions that could do the trade in size — the global dealer banks.

Walk the chain. A bank that wants to harvest a −50 bps basis must borrow dollars, lend the foreign currency, and post collateral — inflating its total assets. Under the leverage ratio, total assets are capped relative to equity *regardless of risk*, so even this perfectly hedged trade eats capacity that the bank could otherwise use for higher-returning business. The bank's own internal cost of that capacity — the return on equity it demands per unit of balance sheet — sets a floor on the basis it will chase. If the basis is shallower than that floor, the desk passes. Add the calendar effect (nobody wants the trade on a reporting date), counterparty and rollover risk (you are funding a long-dated asset with short swaps that must be rolled, and the roll can gap against you in a panic), and the apparently free 50 bps is, after honest costs, roughly fair pay for a constrained balance sheet. The graph below traces why the money goes unharvested.

![Graph of why the cross-currency basis arbitrage will not close](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-5.png)

### Limits to arbitrage, made concrete

Academic finance has a name for this: **limits to arbitrage**. The classic theory says mispricings persist when the people who could correct them face constraints — capital limits, funding risk, the danger that the misprice widens before it converges. The cross-currency basis is the textbook real-world case. The arbitrageurs are not absent or asleep; they are *constrained*, and the basis settles at the level where the marginal arbitrageur is just indifferent — where the pickup exactly compensates the balance-sheet cost. That equilibrium can be −5 bps in a flush market or −150 bps in a panic when balance sheet is being hoarded, and in both cases the basis is "correctly" priced relative to the cost of the capital that would close it.

This reframes what an arbitrage *is*. The textbook arbitrage assumes a frictionless balance sheet — borrow and lend without limit at the risk-free rate. Real intermediaries cannot. So the basis is not a market failure to be fixed; it is the *price of a real, scarce input* (intermediary balance sheet) showing up in a place the textbook assumed that input was free. The 50 bps is not lying on the sidewalk; it is fenced behind a leverage ratio, and the fence has a price.

There is a deeper irony here that is worth naming. The post-crisis regulations that put a price on bank balance sheets — the leverage ratio, the supplementary leverage ratio, the liquidity rules — were designed to make the financial system safer, and broadly they did. But one side effect was to make the CIP arbitrage costly, which means the regulations themselves are part of *why* the basis no longer closes. A safer banking system is also a system where riskless cross-border mispricings can persist, because the very capacity that would arbitrage them away is now rationed by design. The basis is, in part, the visible price of post-2008 financial regulation — a small, permanent tax that the rest of the world pays for the privilege of funding in a currency it does not control, sitting on top of a banking system deliberately built to take less risk. You cannot wish the basis to zero without unwinding the rules that keep banks safe; the two are the same coin.

### Who could supply the dollars — and why they pull back

If the basis is a price the dollar-short pay, someone must be on the other side collecting it: the dollar-rich. In normal times that is US money-market funds, US banks with excess reserves, and US corporates parking cash. They lend dollars into the swap market — directly or through a dealer — and earn the basis as pickup over plain dollar deposits. The basis is the wage that coaxes those dollar holders to part with their dollars for three months and hold a foreign currency instead. So far, so functional.

The problem is that these suppliers retreat in exactly the conditions where they are most needed. Money-market funds, scarred by 2008 and reformed since, become risk-averse and pull back from lending to anyone they perceive as exposed — which in 2011 meant European banks specifically. Dealers, who intermediate the flow and warehouse the balance-sheet cost, hoard their leverage capacity in a panic because every basis point of it is suddenly precious for their own survival. So the supply of dollars into the swap market is *procyclical*: abundant when nobody needs it, scarce precisely when everyone does. That procyclicality is what turns an ordinary −20 bps friction into a −150 bps crisis reading — not a change in the arbitrage logic, but a collapse in the willingness of the natural dollar suppliers to lend when fear is highest. The Fed's swap lines exist to be the *countercyclical* supplier of last resort, stepping in when the private suppliers all step back at once.

#### Worked example: the riskless profit a balance-sheet-constrained bank still won't take

A dealer sees a one-year EUR/USD basis at **−40 bps** and a clean trade: borrow \$1,000,000,000, swap into euros, lend the euros, lock the forward back — harvesting 40 bps riskless, **\$4,000,000** a year. Tempting. But the \$1bn of new assets consumes leverage-ratio capacity, and the bank's internal charge for balance sheet is 50 bps of assets per year — \$5,000,000 for this \$1bn. After the balance-sheet charge the "riskless" trade *loses* \$1,000,000 a year. The desk declines. The 40 bps survives, visible to everyone, harvestable by no one whose balance sheet is fully priced. **A profit that is riskless in market terms can still be unprofitable after the cost of the balance sheet it consumes — which is exactly why the basis does not close.**

## Common misconceptions

**"A non-zero basis is an arbitrage opportunity — free money."** Only if balance sheet is free, which it is not. The basis sits at the cost of the leverage capacity required to harvest it. At −15 bps with a 20 bps balance-sheet charge, the "arbitrage" loses money; the misprice is fully consistent with rational, constrained intermediaries. The free-money framing is a relic of a frictionless world that ended in 2008.

**"The basis reflects different credit risk between the two countries' rates."** Tempting, but no. The basis is computed against collateralised, near-risk-free reference rates (and shows up even in OIS-based, secured measures), and it appears between two safe sovereigns like the US and Japan. Credit spreads are a separate thing; the basis is overwhelmingly a *balance-sheet and dollar-funding* phenomenon, which is why it spikes on reporting dates that have nothing to do with credit.

**"A negative basis means the dollar is going up."** Related but not the same. The basis measures the *cost of funding* in dollars, not the *direction* of the dollar. The two often move together — a scramble for dollars tends to both push the dollar up and widen the basis (see the dollar-index chart below) — but they are distinct readings. You can have a strong dollar with a calm basis (steady US outperformance) and a wide basis without a soaring dollar (a pure funding squeeze). Read them as two instruments, not one.

**"Central-bank swap lines are a bailout / subsidy."** They are a *price cap*. When the Fed lends dollars to the ECB or BoJ against their currencies, and those central banks on-lend to their local banks, they add dollar supply exactly where the shortage is — capping how negative the basis can go. It is the Fed acting as the world's dollar lender of last resort, not a gift; the swaps are collateralised, short-dated, and have always been repaid. The sibling post on [the global dollar shortage and swap lines](/blog/trading/forex/the-global-dollar-shortage-and-central-bank-swap-lines) details the mechanism.

**"The basis is a tiny technicality only swap traders care about."** In normal times, a few basis points — minor. But it is the cleanest live gauge of global dollar stress, it determines the all-in cost of trillions of dollars of cross-border hedging and funding, and when it blows out it is one of the first and most reliable signals that a funding crisis is underway. Ignoring it is ignoring the price of the most important borrowing the world does.

**"If CIP fails, the textbook is just wrong."** Not wrong — incomplete. CIP is exactly right under its stated assumption that arbitrageurs face a frictionless, unconstrained balance sheet. The post-2008 world simply violates that assumption, so the *identity* still holds once you add the missing term — the basis. Think of the basis as the correction that makes the textbook equation true again in a world with costly balance sheets: \$F = S(1+r_{usd})/(1+r_{eur}+b)\$ holds perfectly, with \$b\$ absorbing every friction. The textbook was not refuted; it was extended, and the extension has a name and a number you can trade.

## How it shows up in real markets

The mechanism earns its keep in three episodes, each a moment the world ran short of dollars and the basis printed the price.

### 2008: the offshore dollar market freezes

After Lehman Brothers failed in September 2008, the unsecured interbank market — where banks lent each other dollars overnight on trust — simply stopped, because trust was gone. Non-US banks holding dollar assets (US mortgages, US bonds) that they had funded with short-term dollar borrowing suddenly could not roll that borrowing. Their only remaining door to dollars was the FX swap market: pledge euros or yen, receive dollars, pay whatever it takes. With every dollar-short bank crowding that one door and almost no one willing to lend dollars out, the price exploded. The three-month USD/JPY basis reached around **−220 bps** and EUR/USD around **−160 bps**. The Fed responded by opening dollar **swap lines** with foreign central banks, which drew a peak of about **\$583 billion** in December 2008 — adding dollar supply directly into the shortage and pulling the basis back. The next chart shows those swap-line draws: each bar is the Fed acting as the planet's dollar lender of last resort.

![Federal Reserve dollar swap line draws by crisis episode](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-6.png)

The 2008 basis blow-out is why the basis stopped being a textbook footnote and became a permanent feature of markets. Before 2008 it was approximately zero; after 2008 it never returned to zero, because the balance-sheet and regulatory costs that the crisis introduced — and that the post-crisis rules then locked in — never went away.

The detail worth sitting with is the *direction* of the squeeze. It was non-US banks that were caught: they had built large dollar asset books during the boom — US mortgages, structured credit, agency debt — and funded them with short-term dollar borrowing in the wholesale market, much of it from US money-market funds. That is a classic maturity-and-currency mismatch: long-dated dollar assets, short-dated dollar funding, and not a domestic dollar deposit base to fall back on. When the wholesale funding evaporated, those banks could not simply borrow dollars at home the way a US bank could; their only recourse was to pledge euros, yen, or sterling and pull dollars out of the swap market. The basis is, in this sense, the price of *not having a domestic dollar deposit base while running a dollar balance sheet* — the structural condition of nearly every large non-US bank.

### 2011: the euro-area crisis and dollar-starved European banks

In the second half of 2011, the euro-area sovereign-debt crisis hit European banks where they were most exposed: their dollar funding. US money-market funds, frightened that European banks were sitting on losses from Greek, Italian, and Spanish sovereign debt, pulled back hard from lending dollars to them — a quiet, fast, wholesale run that never made the front pages the way a deposit run would. European banks, which had financed large dollar asset books with exactly that money-market borrowing, were squeezed precisely as in 2008, and turned to the swap market to plug the hole. The three-month EUR/USD basis widened to around **−150 bps** and the USD/JPY basis to around **−90 bps**.

The episode is a clean illustration that the basis is driven by *who is short of dollars*. In 2011 the dollar-short institutions were European banks specifically, so it was the EUR/USD basis that did the most violent moving while other pairs were comparatively contained — the basis told you, currency by currency, exactly where the funding stress was concentrated. That diagnostic precision is one of the basis's most useful properties: a single risk-off index lumps all stress together, but the menu of currency-specific bases points a finger at *which* part of the world has the funding problem. Coordinated central-bank action — including reactivated and cheapened Fed swap lines announced jointly with five other major central banks at the end of November 2011 — added dollar supply where the shortage was and capped the squeeze, and the basis narrowed almost the moment the announcement crossed the wires.

### March 2020: the dash for cash

The COVID shock in March 2020 produced the broadest dollar scramble of all — not just banks, but corporates drawing down their credit lines en masse to hoard cash, foreign central banks defending their currencies, and leveraged funds facing margin calls, all reaching for dollars at once. This was new: in 2008 and 2011 the squeeze was concentrated in the banking system, but in 2020 the demand for dollars came from every corner of the real economy and the investment world simultaneously. Even US Treasuries — the world's safe asset — were being sold for cash in a "dash for cash" so frantic that the Treasury market itself seized up. The three-month EUR/USD basis gapped to roughly **−85 bps** and USD/JPY to around **−145 bps** in a matter of days.

The Fed's response was faster and bigger than ever, and it had clearly learned the 2008 lesson that the basis is the gauge to manage. It cut the price on its standing swap lines with major central banks and lengthened their tenor, reactivated temporary lines with a wider group of central banks, and introduced the **FIMA repo facility** so foreign official institutions could raise dollars against their Treasury holdings without dumping those Treasuries into a broken market. Swap-line draws peaked near **\$449 billion** by May 2020. Within weeks the basis normalised — a far faster recovery than 2008's grinding months. The speed of both the policy response and the basis's recovery is the clearest demonstration of the basis as a real-time gauge that policymakers now watch and actively manage; the Fed effectively treats the cross-currency basis as a target variable for its role as the world's dollar lender of last resort.

These episodes also show why the basis and the broad dollar tend to travel together. When the world scrambles for dollars, the dollar usually strengthens *and* the basis widens — two faces of the same shortage. The dollar-index chart makes the co-movement visible: the 2022 dollar surge and the 2020 spike both coincide with basis stress, even though, as the misconception above warned, the two are distinct readings you should track separately.

![Dollar index DXY year-end levels with the 2020 and 2022 stress points](/imgs/blogs/the-cross-currency-basis-when-covered-parity-breaks-8.png)

For the deeper plumbing — *why* the world has so many dollar liabilities outside the US in the first place — the post on [eurodollars and the offshore dollar system](/blog/trading/forex/eurodollars-and-the-offshore-dollar-system) is the companion to this one: it explains the offshore dollar stock, and this post explains what its price does when that stock gets scarce.

### What the basis costs a real currency-hedged investor

The basis is not only a crisis siren; it quietly taxes one of the most common positions in global finance — the currency-hedged foreign bond. A euro-based investor who buys US Treasuries and hedges the dollar exposure back to euros does that hedge with a rolling FX swap, and the cost of that swap *includes the basis*. When the basis is negative, the euro investor's hedged dollar yield is lower than the headline US yield by roughly the absolute basis. This single fact has rerouted hundreds of billions of dollars of global investment.

Consider the late-2010s episode when US yields were attractive on paper but the EUR/USD basis sat meaningfully negative and dollar money-market rates had risen. For a euro or yen investor, the *hedged* yield on US Treasuries — after paying away the basis on the rolling hedge — fell to near or even below their domestic bond yield. The trade that looked like free extra yield in the headline number was, after the basis, no better than staying home. Capital that would otherwise have flowed into US bonds did not, purely because the basis made the hedge too expensive. The basis, in other words, is not just a stress gauge; in calm times it is a standing toll that shapes where the world's bond money actually goes.

#### Worked example: the basis quietly erases a euro investor's yield pickup

A euro-based pension fund looks at a US Treasury yielding **4.5%** versus a German Bund yielding **2.4%** — a tempting 2.1% pickup. But it must hedge the dollar back to euros with a rolling FX swap. The swap cost is roughly the US-minus-euro rate gap (which the fund gives back, since the higher dollar rate is offset in the forward) *plus* the basis. With the one-year EUR/USD basis at **−40 bps**, the fund pays an extra 0.40% on its hedge. After hedging, the Treasury's euro-equivalent yield lands near the Bund's — the entire 2.1% headline pickup is consumed by the rate-gap give-back and the 40 bps basis toll. **For a hedged foreign investor, a negative basis is a direct haircut to the yield they actually keep — which is why the basis, not just the headline yield, decides where global bond money flows.**

## The takeaway

The cross-currency basis is the single best answer to a question the textbook says should never come up: *what happens to the iron link between the rate gap and the forward when the world runs short of the currency on one side?* The answer is that the link bends, and the basis is the price of the bend.

Three things to carry away. First, **a negative basis is not a glitch or a free lunch; it is a price** — the market's charge for delivering scarce dollars to those who need them via the swap market, equal to the cost of the balance sheet that closing it would consume. Read it the way you read any other price: it tells you how badly someone wants something. When the basis at −15 bps and the same basis at −150 bps are both "correctly" priced — each at the cost of the balance sheet that would arbitrage it — you have understood that prices in modern markets are set not by some frictionless law but by the cost and availability of the intermediary capital that enforces the law. The basis is the purest example of that truth in all of finance, because the thing being priced is supposed to be free. Second, **the basis is the cleanest live gauge of global dollar stress.** When it sits at −15 bps the plumbing is calm; when it gaps toward −80, −150, −220 bps you are watching a funding crisis in real time, before it shows up in headlines. Watch the basis into quarter-ends and into any risk-off scramble, and watch the Fed's swap-line draws as the counterweight. Third, **it is the canonical limits-to-arbitrage case** — proof that "riskless profit" assumes a frictionless balance sheet that real intermediaries do not have, and that after 2008 the price of that balance sheet became a permanent feature of every cross-border funding cost.

Bring it back to the series' spine. An exchange rate is the relative price of two monies, moved by the gap between two countries' interest rates plus the flow of money across borders. The cross-currency basis is what you see when that flow seizes up: the rate gap is still there, the forward is still quoted, but the *flow of dollars* has a price now, and that price is the basis. When you next see a treasurer paying 50 bps over fair value to roll a hedge, or the yen basis gapping into year-end, you are not looking at a market malfunction. You are looking at the cost of a dollar shortage, quoted to four-decimal precision, by the largest market on Earth.

## Further reading & cross-links

- [Spot, forward, and swap: the three ways to trade a currency](/blog/trading/forex/spot-forward-and-swap-the-three-ways-to-trade-a-currency) — the mechanics of the forward and the FX swap that the basis is measured on.
- [Interest-rate differentials: the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx) — the rate gap that CIP turns into a forward, and that the basis is the deviation from.
- [The global dollar shortage and central-bank swap lines](/blog/trading/forex/the-global-dollar-shortage-and-central-bank-swap-lines) — how the Fed caps the basis by lending dollars to the world.
- [Eurodollars and the offshore dollar system](/blog/trading/forex/eurodollars-and-the-offshore-dollar-system) — why so many dollar liabilities sit outside the US, the stock whose price the basis sets.
- [Uncovered interest parity and the forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle) — the basis's cousin: the unhedged parity that fails for completely different reasons.
- [The dollar as a wrecking ball for emerging markets](/blog/trading/forex/the-dollar-as-a-wrecking-ball-for-emerging-markets) — what a dollar shortage does to the economies most dependent on dollar funding.
