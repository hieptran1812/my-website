---
title: "Amber, Galaxy, and the Asia Market-Maker Landscape: How Time Zones and Regulation Shape Crypto Liquidity"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero guide to the regional structure of crypto liquidity — why depth rotates around the clock, how a global desk hands its book from Asia to London to New York, why an Asia-native token is cheapest to trade at 3am UTC, how regulation decides who can quote what, and why regional price wedges like the kimchi premium survive arbitrage. Featuring Galaxy Digital and Amber Group, with worked dollar examples."
tags: ["crypto", "market-makers", "asia", "galaxy-digital", "amber-group", "liquidity", "market-microstructure", "arbitrage", "crypto-players", "time-zones"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 27
---

> [!important]
> **TL;DR** — "Crypto trades 24/7" is true of the *market* and false of the *market makers*. The firms quoting your prices keep office hours, so liquidity is not constant — it **rotates around the globe with the working day**, and where a token's makers sit determines when it is cheap to trade and when it is not.
>
> - **Depth follows the sun.** Asian desks (Hong Kong, Singapore, Tokyo, Seoul) carry the book from roughly 00:00–08:00 UTC; London and Europe add depth from 07:00; the deepest book of the day is the **EU–US overlap around 13:00–17:00 UTC**; and there is a **"dead zone"** around 21:00–00:00 UTC where spreads are widest and depth thinnest.
> - **A global desk passes its book like a baton.** It does not flatten at 5pm; it hands a live position and a risk limit to the next time zone, so an Asia-built long can be worked out into European depth hours later.
> - **Regulation, not geography, ultimately sets the map.** What a desk is *allowed* to quote — retail access, new-token listings, derivatives, fiat ramps — varies by jurisdiction, and those rules decide where a token's liquidity actually lives.
> - **Two firms, two structures.** **Galaxy Digital** (US-listed, SEC-reporting, diversified into trading, asset management, and data centers) and **Amber Group** (Asia-rooted, private-origin, trading plus wealth) show how the *same* order flow gets monetized through very different corporate shapes once you cross a listing and a time zone.
> - **Regional price wedges survive because plumbing blocks the arbitrage.** The famous "kimchi premium" persists not because traders miss it, but because capital controls and settlement rails stop them from closing it.

Almost every other post in this series treats the crypto market as a single place — one order book, one price, one set of players pushing on it. This post breaks that assumption on purpose, because the single-place model hides something important: **liquidity has a geography and a clock.** The same token does not trade the same way at 03:00 UTC and 15:00 UTC, and the reason is not mysterious — it is that the humans and machines quoting it are asleep in one hemisphere and awake in another.

Two firms anchor the story because they sit at opposite corners of it. **Galaxy Digital** is a US-listed, diversified, disclosure-heavy institution built in the American time zone and regulatory world. **Amber Group** is an Asia-rooted trading house that grew up in the Hong Kong–Singapore corridor with a very different regulatory and cultural surround. Neither is the "biggest" market maker — Wintermute and Jump quote more size — but together they map the *regional* structure of who quotes what, when, and under whose rules. Understanding that structure is the difference between wondering why your altcoin order filled terribly at midnight and knowing exactly why.

## Foundations: crypto never closes, but the desks keep office hours

The single most useful idea in this post is also the most counterintuitive: **a 24/7 market does not mean 24/7 liquidity.** Yes, you can trade Bitcoin at any second of any day. But *quoting* a tight two-sided market — resting real size on both bid and ask, and standing ready to take the other side of a block — is work done by firms with traders, risk managers, and compliance officers who live somewhere and sleep sometime. When their working day ends, their quotes thin out, even though the exchange stays open.

The result is a **liquidity relay** that circles the planet. Follow it around one UTC day:

- **00:00 UTC** — Tokyo and Seoul come online. Asian desks take the book.
- **02:00–06:00 UTC** — Hong Kong, Singapore, and Tokyo are in full swing; the Western world sleeps. Asia *is* the market during these hours.
- **07:00 UTC** — London opens. European desks add depth on top of Asia's.
- **13:00–17:00 UTC** — the **EU–US overlap**: both continents are at their desks. This is the deepest, tightest book of the day.
- **20:00–21:00 UTC** — US close, and a spike of **ETF net-asset-value (NAV) strike and rebalance flow** as US spot-crypto ETFs mark and adjust.
- **21:00–00:00 UTC** — the **dead zone**: America has gone home, Asia has not yet arrived. Spreads are widest, depth thinnest.

![A 24-hour clock of crypto liquidity in UTC, showing depth rotating from Asian desks overnight, through the London open, to the deep EU–US overlap in the afternoon, then thinning into a late-evening dead zone.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-1.webp)

If you have ever placed a market order that filled far worse than you expected, there is a good chance you did it in the dead zone. Nothing was "wrong" — the makers were simply off the desk. Conversely, the reason large, price-sensitive institutional flows so often execute in the 13:00–17:00 UTC window is that this is when the book can absorb them with the least impact. The clock is not decoration; it is a map of where the liquidity is at any given hour, and the smartest execution desks route around it deliberately.

It is worth pausing on *why* makers keep hours at all, since "the algorithm never sleeps" is a common misconception. The quoting engine may run continuously, but the **risk appetite behind it** does not. A market maker's willingness to rest large size depends on a human risk manager being awake to intervene if something breaks, on funding and credit lines being live, and on the ability to hedge in correlated markets that themselves keep hours (traditional futures, FX). Overnight, with the risk desk thin, firms deliberately *cut the size they quote and widen their spreads* — not because they cannot technically quote, but because the cost of being wrong with no one watching is too high. Automated quoting without awake risk oversight is how a fat-finger print or a stale oracle turns into a nine-figure loss, so prudent desks throttle back precisely when supervision is lightest. The dead zone is a risk-management decision, not a technical limitation.

The **ETF NAV spike** around 20:00–21:00 UTC deserves a note too, because it is a newer feature of the post-spot-ETF market. US spot-crypto ETFs strike their net asset value against a reference price near the US close and rebalance creations and redemptions accordingly. That produces a concentrated burst of mechanical buying or selling at a specific time each day — flow that has nothing to do with anyone's view and everything to do with the ETF plumbing. Desks that understand this flow position for it; traders who do not sometimes find the price lurching at the same time each afternoon and invent a narrative for what is really an administrative rebalance.

Weekends and holidays are the same phenomenon in slow motion. With traditional markets closed and many desks skeleton-staffed, weekend crypto liquidity is structurally thinner, which is why sharp, "unexplained" weekend moves are so common: the same order that barely dents a Wednesday-afternoon book can gap the price on a Sunday, simply because far fewer makers are present to absorb it. The clock operates at multiple scales — hour of day, day of week, and holiday calendar — and all of them are really one variable: how many risk-takers are awake and willing.

## Where the quoting desks actually sit

The relay works because the firms are physically clustered in a handful of financial cities, and those clusters keep local office hours. A rough map of who quotes crypto looks like this:

![A map of the cities that house most crypto market makers: Asia-anchored desks in Hong Kong, Singapore, Taipei, Seoul, and Tokyo quoting the 00:00–08:00 UTC window, and Europe/US desks like London-based Wintermute and B2C2 quoting the 07:00–21:00 UTC window.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-4.webp)

- **Asia-anchored desks** carry the core **00:00–08:00 UTC** window. Hong Kong and Singapore host trading houses like Amber, QCP, and Auros; Taipei and Seoul host firms like Kronos and Presto; Tokyo has a set of licensed venues operating under Japan's strict regime.
- **Europe and US desks** carry the core **07:00–21:00 UTC** window. London is a particular hub — Wintermute and B2C2 are London firms — bridging the Asian handover into US hours.

This geography is not an accident of where people happen to like living. It tracks three things: proximity to *demand* (Asia has enormous retail and institutional crypto participation), proximity to *regulatory clarity* (Singapore, Hong Kong, and Japan built licensing regimes early), and proximity to *banking rails* (you cannot run a fiat-settling desk without banks that will hold your money). Where those three overlap, desks cluster; where they do not, liquidity is thin regardless of how many traders want to be there.

The hubs also specialize, and the specialization matters. **Singapore** became a preferred base for institutional trading firms and funds because of its early regulatory engagement and stable banking, so a disproportionate share of professional market making and OTC is domiciled there. **Hong Kong** re-entered the map aggressively with its licensed VATP regime, positioning itself as the gateway for regulated retail access in Asia. **Tokyo** operates under one of the strictest regimes in the world — Japan's FSA vetting means fewer tokens list but the ones that do trade in a highly protected environment, which shapes a distinctive, somewhat insular liquidity pool. **Seoul** is a demand powerhouse: Korean retail participation is enormous, but capital controls wall that demand off from the global market (the source of the kimchi premium we will reach shortly). **Taipei** hosts a cluster of high-frequency and quant firms. The point is that "Asia" is not one market; it is a set of distinct regulatory and cultural pools, each with its own depth profile, and a token's liquidity depends on *which* of them adopted it.

There is a feedback loop here worth naming: makers domicile where the rules let them quote the products their clients want, clients gravitate to where the makers and depth are, and regulators compete to attract both by offering clarity. That loop is why a handful of cities capture the overwhelming majority of professional crypto liquidity while dozens of other financial centers host almost none. Liquidity, like capital, is sticky and self-reinforcing — it pools where it is already pooled.

The practical upshot for you is that **a token's liquidity lives wherever its makers sit.** A token whose community, listing, and makers are Korean will have its deepest, tightest book during Seoul working hours and a miserable one at 22:00 UTC. A US-anchored major like an ETF-linked BTC product trades best in US hours. The token does not know what time it is; its market makers do.

## Passing the book around the world

Here is where the two-sided nature of the business gets elegant. A global desk with offices in multiple time zones does not *close* its position when one office goes home. It **hands the book to the next desk** — a live position, plus a risk limit describing how much that position is allowed to grow or shrink before the incoming traders must act.

![A book-handover timeline: a Hong Kong desk opens flat, ends the Asian session four million dollars long, passes that position and its risk limit to London at 07:00 UTC, which works it down into deeper European liquidity through the afternoon overlap before quotes widen again at 21:00.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-5.webp)

### Worked example 1 — a book handed from Hong Kong to London

Start the clock at **00:00 UTC**. The Hong Kong desk opens **flat** — no position. Through the Asian session it fills a lot of client buys, so by **07:00 UTC** it is **+$4 million long** the token: it has sold to buyers and is now holding the offsetting inventory it must eventually unwind.

At the **07:00 UTC handover**, Hong Kong does not dump that $4 million into a thin Asian book at the end of its day — that would move the price against itself. Instead it passes the **position and a risk limit** to the London desk: "You are +$4M long; keep it under +$6M, work it down when the book is deep." London then sells that inventory *into the EU–US overlap* from **13:00–17:00 UTC**, when depth is greatest and its selling barely moves the price. By **21:00 UTC**, the desk is roughly flat again, and as quotes widen into the dead zone it cuts its size and waits for Tokyo to reopen.

The elegance is that the firm treats the globe as one continuous trading day with three shifts, each handing risk to the next. This is why the largest desks are multi-office by design: a single-time-zone desk is forced to either hold overnight risk through hours when it cannot hedge, or flatten into thin books at a cost. A relay of offices turns the planet's rotation from a problem into an advantage — there is always *somewhere* awake to manage the position.

The flip side is what happens when the relay *breaks*. If the incoming desk is understaffed — a local holiday, a systems outage, a sudden risk-off moment when everyone widens at once — the handover fails, and the position that should have been quietly worked down instead sits exposed through thin hours. This is a real source of the sharp, "out of nowhere" moves that punctuate crypto: not fresh news, but a large inventory that could not be handed off cleanly and had to be dumped or defended into a book too thin to take it. The relay is a strength when it works and a concentrated fragility when it does not, which is why the firms that survive are the ones whose risk limits assume the handover *might* fail rather than assuming it always succeeds. A desk that sizes its overnight book to what the *next* time zone can absorb on a bad day, not a good one, is the desk that is still standing after the bad day arrives.

This is also why single-region desks command a structurally different risk profile from the global relays. An Asia-only desk that builds a large position during its session must either hold it through the hours when it is asleep and cannot react, or unwind it into its own thinning book at the end of the day — both costly. A global firm with the relay simply passes the baton. That advantage is a large part of why the biggest, most durable market makers are multi-continental, and why a purely regional desk tends to specialize in the tokens native to its own session, where it has an informational edge that offsets its time-zone handicap.

## The clock has a price: spread by hour

The relay is not just about *depth*; it shows up directly in the **spread you are quoted**, and it does so differently for different tokens depending on where their makers live.

![A chart of quoted spread by hour of day for two tokens: an Asia-native token is cheapest to trade near 03:00 UTC when its home desks are active, while a US-anchored major is tightest around 15:00 UTC in the EU–US overlap.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-6.webp)

### Worked example 2 — the same trade, two different hours

Suppose you want to buy **$200,000** of an Asia-native token — say a Korean-community token whose makers sit in Seoul. At **03:00 UTC**, its home desks are fully staffed, and the quoted spread is a tight **20 basis points**. Your round-trip cost to cross that spread is roughly 0.20% × $200,000 = **$400**.

Now place the identical order at **21:00 UTC**, in the dead zone, when the Seoul desks have gone home and no one has replaced them. The spread has widened to **80 basis points**. The same $200,000 trade now costs about 0.80% × $200,000 = **$1,600** — **four times as much** — for no reason other than the clock. You bought the identical token in the identical size; you simply demanded liquidity when its providers were asleep.

Flip the token and the pattern inverts. A US-anchored major is tightest in the **13:00–17:00 UTC** overlap and worse at 03:00 UTC when only Asian desks — who may not specialize in it — are quoting. The lesson is not "trade at 3am" or "trade at 3pm"; it is **match your execution to your token's home session.** For a price-insensitive $200 order this is noise. For a $2 million order it is real money, and it is the first thing a competent execution desk checks.

## Two ways to build a crypto trading house

With the geography in hand, the two anchor firms become easy to place — and they are deliberately chosen to be opposites, because the *same* underlying activity (make markets, trade, lend, manage assets) gets wrapped in very different corporate structures once you cross a stock listing and a time zone.

![A comparison of Galaxy Digital, a Nasdaq-listed and SEC-reporting US firm, against Amber Group, a private-origin Asia trading house with a listed wealth arm, across ownership, disclosure, and core business lines.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-2.webp)

**Galaxy Digital** is the American-model firm. Founded by **Mike Novogratz** (a former Goldman Sachs partner and Fortress principal), Galaxy grew into a diversified, publicly traded institution spanning trading and market-making, asset management, investment banking, and — increasingly — **digital infrastructure**, including data-center capacity at its Helios campus in Texas that it has repositioned toward AI/high-performance-computing tenants. Galaxy trades on public markets (it completed a U.S. listing under the ticker **GLXY**), which means it files audited financials and discloses its segments to regulators. That transparency is the defining feature of the American model: you can *read* what it did last quarter.

**Amber Group** is the Asia-model firm. Founded in **2017** by a group of former traders, Amber grew up in the Hong Kong–Singapore corridor as a trading and market-making house that later added a **wealth-management** arm (marketed under the WhaleFin brand). Its origins are private — funded by venture rounds rather than a public listing — with the disclosure that implies: much less is visible from the outside than for a US-listed peer. (Amber's corporate structure has since evolved, including a Nasdaq-listed entity for part of the group; details here are stated as reported, and the point is the *shape*, not a specific quarter's numbers.)

The contrast is the lesson. Both firms monetize order flow. But Galaxy does it inside a US-listed, SEC-reporting, multi-segment public company, while Amber grew up as a private Asia trading house. That difference is not cosmetic — it changes who can invest, what must be disclosed, which regulators have jurisdiction, and, ultimately, how much of the firm's risk-taking the outside world ever sees. When you read about a "crypto market maker," the corporate shell around the trading desk tells you as much about the risk as the trading itself.

The disclosure asymmetry cuts both ways, and it is worth being fair about it. Galaxy's public listing means its bad quarters are *visible* — you can read the losses, the segment breakdown, the equity cushion — which is exactly why we can use it as a teaching example. That visibility is a genuine protection for anyone dealing with it: a counterparty can assess Galaxy's solvency from filings rather than rumor. The private Asia model trades that transparency for speed and flexibility. Amber could raise venture capital, expand into wealth management, and pivot quickly in ways a public company's disclosure obligations slow down — but the outside world, including its own counterparties, sees far less. When the 2022 contagion hit, the market could read Galaxy's exposure in its filings; it had to *guess* at the exposure of the many private Asia desks, and that opacity is itself a risk that gets priced into how much anyone will trade with, or lend to, a firm they cannot see inside.

Amber's own history illustrates the point. Founded in 2017 and backed by prominent venture investors through the 2021 boom, it grew rapidly into trading and wealth management, sponsoring high-profile marketing under the WhaleFin brand. It also weathered the 2022 storm — reported exposure to that year's failures, a subsequent retrenchment, and a corporate restructuring — largely out of public view, with the details emerging in fragments rather than quarterly filings. Whether a given private desk came through such an episode healthy or wounded is often genuinely hard to know from outside, which is precisely why counterparty due diligence in the Asia landscape leans so heavily on relationships, credit lines, and reputation rather than on published numbers. The structure you choose to build your trading house in determines not just your tax and regulatory exposure, but how the rest of the market is able to trust you.

## What is actually inside a diversified firm

The diversified public-company model has a specific, non-obvious risk that is worth making concrete, because it explains a pattern you will see again and again in crypto-firm earnings: **the trading desk can make money while the company as a whole loses it.** The culprit is usually the firm's *own* balance sheet — the coins it holds for itself, marked to market.

![An illustrative segment breakdown of a diversified public crypto firm, showing a profitable trading and asset-management arm and a small data-center profit offset by a large mark-to-market loss on the firm's own token treasury, netting to a group loss.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-3.webp)

### Worked example 3 — a profitable desk, a losing company

Consider the segment structure this figure illustrates (the numbers are illustrative of the *pattern*, chosen to show the mechanism rather than to report any specific quarter):

- **Digital Assets** — the trading, OTC, lending, and asset-management arm — earns an **adjusted gross profit of +$49 million**, serving well over a thousand institutional counterparties across a loan book of more than a billion dollars. The *business* is working.
- **Data Centers** — leasing infrastructure capacity — adds a modest **+$3.1 million**.
- **Treasury & Corporate** — the firm's *own* crypto holdings, marked to market — posts a **−$140 million** loss because the market fell during the quarter.
- **Group total:** roughly **−$88 million** adjusted gross profit and a net loss, against a multi-billion-dollar equity base.

Read that top to bottom and the moral is stark: a firm can run a genuinely profitable market-making and lending operation and still print a large group loss, purely because its treasury of self-held coins got marked down faster than the operating businesses earned. This is the diversified firm's version of the risk the [Alameda post](/blog/trading/crypto-players/alameda-research-the-cautionary-tale) took to its catastrophic extreme: **holding a large directional position in the assets you also trade** means your balance sheet and your business share a fate. Galaxy's real, well-documented **large losses in 2022** — driven by the market crash and its exposure to that year's collapses — are a concrete instance of exactly this dynamic. The difference between a diversified public firm and Alameda is not that the former avoids the risk; it is that disclosure, segregation, and a genuine equity cushion let it *survive* the mark-down instead of being destroyed by it.

For you as an observer, this decomposition is a practical tool. When a crypto firm reports a big loss, do not stop at the headline — ask *which segment* lost the money. A loss concentrated in the treasury or corporate line means the operating business is fine and the firm simply had directional exposure that fell; that is a solvency question about the size of the equity cushion, not a verdict on the business. A loss concentrated in the *trading or lending* segment is far more alarming, because it means the core money-making engine is impaired. The two look identical at the group level and mean opposite things underneath. This is why the diversified, disclosed model is worth understanding even if you never buy the stock: it teaches you to read a crypto institution the way you would read a bank — by segment, by where the risk actually sits — rather than by the single scary number at the top of the press release.

## Regulation decides who can quote what

Underneath the clock and the corporate structures sits the layer that ultimately draws the map: **rules.** What a desk is *allowed* to do — and to whom — varies enormously by jurisdiction, and those permissions decide where a token's depth can even exist.

![A matrix of what a desk is permitted to do by jurisdiction — local retail access, new-token listings, retail derivatives, and fiat on/off ramps — showing that regulation, not trader appetite, determines where liquidity for a given product can live.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-7.webp)

The dimensions that matter most are:

- **Local retail access** — can ordinary residents legally trade on local venues at all? In Hong Kong, for instance, only licensed Virtual Asset Trading Platforms (VATPs) may serve retail, and listings are vetted slowly by the SFC.
- **New-token listing** — how fast and how freely can a venue list a new token? A slow, vetted regime means a hot new token's early liquidity migrates offshore.
- **Retail derivatives** — many jurisdictions restrict or ban leveraged crypto products for retail, pushing that volume to offshore venues.
- **Fiat on/off ramps** — whether local banks will connect to crypto at all, which determines whether real money can enter and exit without friction.

The consequence is that **the same token can have a completely different liquidity profile in two countries**, purely because of what the local rules permit. A token that is freely listable and tradable with leverage in one offshore jurisdiction, but blocked from retail in another, will see its depth concentrate where the rules are permissive. Market makers, in turn, domicile and license themselves to be *able* to quote the products their clients want — which is why the desk map and the regulatory map are, to a first approximation, the same map. Rules decide which firms can quote which product to whom, and that decides where the depth lives.

The regimes themselves span a wide spectrum. At the **permissive-but-vetted** end, Singapore and Hong Kong license firms and platforms, demand real compliance, but allow a broad regulated business to operate. At the **strict-and-protective** end, Japan's FSA runs one of the most conservative regimes: exhaustive token vetting, tight custody rules, and strong retail protections that make it slow to list anything new but very safe once listed. At the **walled-demand** end sits Korea, where enormous retail appetite is deliberately fenced off from the global market by capital controls and real-name banking. And at the **restrictive** end, some large jurisdictions ban retail derivatives outright or bar local exchanges entirely, pushing that volume offshore to venues domiciled specifically to serve it. Each regime is a different answer to the same question — *how much crypto risk should ordinary people be allowed to take, and through whom?* — and each answer redraws the liquidity map for every token exposed to it.

For a market maker, this is not abstract policy; it is the daily determinant of what business it can even book. A desk licensed in Singapore can offer products to institutions that it may not offer to Hong Kong retail; a token hot in an offshore-derivatives venue may be untouchable for a US-facing desk. The firms navigate this by holding multiple licenses across jurisdictions and routing each client to the entity permitted to serve them — a compliance architecture as elaborate as the trading technology, and one of the real barriers to entry that keeps the professional market-making world small.

## Why a regional price wedge survives arbitrage

The most striking consequence of this fragmented, regulated, time-zoned structure is that **the same asset can trade at persistently different prices in different regions** — and the arbitrage that "should" close the gap is blocked, not by a lack of traders who see it, but by plumbing.

![A flow diagram of why a regional premium survives: a local demand shock lifts the domestic price, the obvious arbitrage of buying offshore and selling locally appears, but capital controls, real-name banking rules, and the absence of an offshore settlement rail block the trade, so the wedge persists.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-8.webp)

### Worked example 4 — the kimchi premium that will not close

The classic case is the Korean "kimchi premium." Suppose a wave of domestic demand hits: Korean-won buyers pile in, and the local **BTC/KRW** book lifts to **+2% above** the global price. A first-year finance student sees the trade instantly: **buy BTC offshore at the global price, move it to Korea, sell it for KRW at +2%, pocket the spread.** Do it at scale and the premium collapses. So why does it persist for weeks?

Because every leg of that trade is blocked by real infrastructure:

- **Capital controls** cap how much KRW an individual can remit abroad, so you cannot freely send won out to buy BTC offshore in size.
- **Real-name bank-account rules** tie every crypto account to a verified domestic bank identity, making the large, fast, cross-border flows the arb requires impractical.
- **No offshore KRW settlement rail** exists to move won internationally the way dollars move — so even if you spot the gap, you cannot settle the round trip.

The arbitrage is obvious and the traders are sophisticated; the trade still cannot be done at scale, so the wedge survives. This is the deepest lesson of the regional structure: **price is only "one thing" where capital can freely flow to make it so.** Wherever regulation, banking, or settlement blocks the flow, the "law of one price" quietly fails, and a regional premium or discount becomes a durable feature rather than a fleeting anomaly. Every persistent regional wedge you see is a map of where the plumbing is broken.

The kimchi premium is the most famous case, but it is not the only one, and the others teach the same lesson from different angles. The **"Coinbase premium"** — when Bitcoin trades slightly higher on Coinbase (a US-institutional venue) than on offshore exchanges — is read by traders as a real-time proxy for US institutional demand: when American desks are aggressively buying, the premium turns positive; when Asian selling dominates, it turns negative. It is small and closes quickly precisely *because* the dollar plumbing between Coinbase and offshore venues works well; the fact that it stays small is the mirror image of why the kimchi premium stays large. Historically, the **GBTC discount** — when the Grayscale Bitcoin Trust traded far below the value of the Bitcoin it held — was a wedge of a different kind: it persisted for over a year not because of geography but because the structure had *no redemption mechanism*, so no arbitrageur could convert a discounted share back into the underlying coin to close the gap. When a spot-ETF conversion finally created that mechanism, the discount collapsed. Same principle, different plumbing: a price gap survives exactly as long as the mechanism to close it is blocked, and dies the moment the mechanism opens.

The general skill this builds is to look at any persistent price difference and immediately ask *what specifically prevents the obvious arbitrage?* If the answer is "nothing, it will close in minutes," it is noise. If the answer is a capital control, a missing settlement rail, a redemption lock, or a licensing wall, then the wedge is *structural* and can persist for as long as the barrier does. That question — not the size of the gap, but the durability of the barrier — is what separates a fleeting mispricing from a tradeable, or untradeable, regional feature.

## What it means if you're on the other side

You are unlikely to run a multi-office relay or arbitrage the kimchi premium. So how does the regional structure change what a normal trader should do? It collapses to a simple, powerful habit: **time your order to your token's home session.**

![A three-step guide to timing an order: classify the token as Asia-native or a global major, find the UTC session when its makers are active, and price the timing gap before crossing the spread.](/imgs/blogs/amber-galaxy-and-the-asia-mm-landscape-9.webp)

The three-step read is:

1. **Classify the token.** Is it an **Asia-native** token — a KRW or JPY pair, a Hong Kong or Korean-community listing — or a **global major** like BTC, ETH, or an ETF-linked product? The classification tells you where its makers live.
2. **Find its home session.** Asia-native tokens are best quoted in the **00:00–08:00 UTC** window; global majors are tightest in the **13:00–17:00 UTC** EU–US overlap. Trading a token *outside* its home session means paying the dead-zone spread.
3. **Price the timing gap.** Before you cross the spread, ask what it costs *now* versus in the token's home session (Worked Example 2 showed a 4× difference). For small orders, ignore it. For meaningful size, wait for the session — the saving is often larger than any edge you think you have on direction.

This is the retail-scale version of everything the big desks do industrially. Wintermute passes its book across time zones; you can at least place your order when the book is deep. Galaxy and Amber domicile themselves where the rules let them quote; you can at least trade a token where its liquidity actually lives. The market's 24/7 label invites you to trade at any hour as though all hours were equal. They are not. The liquidity has a geography and a clock, and the single most reliable execution edge available to an ordinary trader is simply to respect both.

The Asia landscape matters to this series because it is where a huge share of real crypto liquidity is born and quoted, in the overnight hours when Western commentary is asleep and "nothing is happening." A great deal is happening — it is just happening in Seoul, Hong Kong, Singapore, and Tokyo, under rules and on a clock most Western traders never learn. The players who move price are not only the named funds and market makers of the other posts; they are also the time zone the trade happens in and the regulator who decides whether it can happen at all.

Hold onto the one habit if you forget everything else: before you send a meaningful order, glance at the clock in UTC and ask whether your token's makers are awake. It costs nothing, it takes a second, and over a year of trading it will save you more than most of the directional cleverness people spend their evenings chasing. The market's geography is invisible on a price chart, but it is priced into every fill you get — and simply respecting it is an edge available to anyone willing to look.

## Sources & further reading

- Galaxy Digital public filings and investor materials (Nasdaq: GLXY) for its segment structure, digital-infrastructure/Helios business, and historical results, including its documented 2022 losses.
- Amber Group public materials and press coverage of its founding (2017), funding rounds, wealth-management (WhaleFin) business, and subsequent corporate restructuring; details stated as reported.
- Reporting and research on the Korean "kimchi premium," capital controls, and real-name crypto banking rules for the persistence of regional price wedges.
- Regulatory references: Hong Kong SFC Virtual Asset Trading Platform regime; Singapore MAS and Japan FSA digital-asset licensing, for the jurisdictional matrix.
- Related posts in this series: [What a Crypto Market Maker Actually Does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), [Cross-Exchange Arbitrage and the Latency Game](/blog/trading/crypto-players/cross-exchange-arbitrage-and-the-latency-game), [GSR, Cumberland and the Established OTC Desks](/blog/trading/crypto-players/gsr-cumberland-and-the-established-otc-desks), and the hub, [Crypto VCs and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers).

*Nothing here is legal or investment advice. Firm details are stated as reported; the segment figures in Worked Example 3 are illustrative of a general pattern, not a specific company's audited results, and dollar figures in the other worked examples are illustrative and rounded to show the mechanics.*
