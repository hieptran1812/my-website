---
title: "How VCs Move Price: Listings, Unlocks, and the Narrative Machine"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "The concrete mechanics by which a crypto venture fund's actions show up in a token's price: getting it listed on a top venue, seeding the narrative that brings demand, and riding the scheduled unlock that transfers cheap insider supply to the public."
tags: ["crypto", "venture-capital", "token-unlocks", "listings", "tokenomics", "fdv", "narrative", "vesting", "crypto-players", "market-structure", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A crypto venture fund rarely "pumps" a token by buying it. It pulls three specific levers that convert cheap, locked private tokens into public exit dollars: it gets the token **listed** on a top venue (a price event), it **seeds the narrative** so demand shows up, and it rides the **unlock** that hands its supply to the public market on a publicly-known schedule.
>
> - A top-tier **listing** is itself a price event: the most-cited study (Messari, 2021) found Coinbase listings added an average of **+91% over five days**, with a huge range and a pop that often faded within weeks.
> - **Narrative is demand.** Research, founder threads, conference stages, paid influencers and media manufacture the attention that becomes net buying — attention *is* the demand curve.
> - An **unlock** is the supply the chart cannot see coming. Insiders buy at seed prices (Solana's seed was **\$0.04** in 2018) and exit into the public market on a schedule you can read in advance.
> - The one number to remember: tokens launched in 2024 averaged a market-cap-to-FDV ratio of about **12%** (Binance Research) — meaning roughly **88% of the supply was still waiting to hit the market**.
> - Your defense is boring and it works: read the cap table, size the next unlock against daily volume, and ask who was paid to tell you the story — *before* you buy.
> - Educational, not financial advice. Every hypothetical token below (we call it ZEPH) uses round made-up numbers; every real figure is dated and sourced at the end.

You have probably watched this happen. A token you had never heard of appears on a major exchange. Within a week there is a research report, three podcast appearances, a founder doing a keynote, and a dozen crypto influencers explaining why this is the future of *something*. The price triples. Six months later it is down 80%, and the same influencers are talking about a different coin.

It is tempting to file this under "crypto is a casino" and move on. But that explanation hides the machine. Behind most of these episodes is a small number of venture funds that bought the token years earlier, at a fraction of the price you can buy it at, and whose job — literally, the thing their investors pay them to do — is to turn those cheap private tokens into public cash. They are very good at it. And the way they do it is not mysterious or illegal; it is a repeatable, three-part playbook that shows up in the price in ways you can learn to see.

The diagram below is the mental model for the whole post. Read it as a machine that takes cheap, locked tokens in on the left and produces public dollars for the fund on the right, with three levers in the middle.

![The VC's three levers: a fund converts cheap, locked private tokens into public exit dollars by listing the token, seeding the narrative, and riding the scheduled unlock.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-1.webp)

The three levers are **list it**, **narrate it**, and **unlock it**. Lever one gets the token onto a venue where millions of people can buy it and where its price re-rates. Lever two manufactures the demand that meets the new supply. Lever three is the scheduled release of the insiders' cheap tokens into that demand. None of these is a secret; the unlock schedule is usually published, the listings are announced, and the research is signed. What most buyers miss is how the three fit together — and that they are usually on the *other* side of the trade. This post builds each lever from zero, grounds it in worked dollar arithmetic, and ends with the checklist that lets you see the machine before you fund it.

This is the price-mechanics capstone of the venture-capital wave of the [Crypto Players](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) series. It assumes no prior finance or crypto knowledge — we define every term. If you want the operating model of the funds themselves, start with [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers), the series hub.

## Foundations: the building blocks

Before the levers, we need a shared vocabulary. If you already know what FDV, float, vesting, and a market maker are, skim this; if not, everything after this section depends on it.

### A token is not a share, and that changes everything

When a company sells shares to a venture fund, those shares are private. The fund cannot sell them to the public until an IPO — a heavily-regulated event with disclosure, underwriters, and (usually) a lock-up that stops insiders dumping for months. A crypto token skips almost all of that. There is no IPO gate, no mandatory disclosure, and the "public market" is just an exchange listing that can happen a year or two after the fund bought in. The result: a crypto venture investor can often exit on a public market *years earlier* than an equity investor could, selling to retail buyers who have no equivalent early access. We covered this structural gap in depth in [Why a Token Is Not a Stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock); here it is simply the ground the whole machine stands on.

### Supply, float, and FDV

A token has a **total supply**: the maximum number of tokens that will ever exist (say, one billion). At any moment, only a fraction of those are actually tradeable — the rest are locked in vesting contracts, team wallets, or a foundation treasury. The tradeable fraction is the **circulating supply**, or **float**.

Two numbers describe the token's size:

- **Circulating market cap** = circulating supply × price. This is the value of the tokens that actually trade.
- **Fully diluted valuation (FDV)** = total supply × price. This is the value of *all* tokens, including the ones that do not exist in the market yet.

When the float is a small slice of total supply, these two numbers diverge wildly — and that gap is the single most important fact in this entire post.

#### Worked example: ZEPH's float and FDV

Let us invent a token, ZEPH, and reuse it throughout. Suppose:

- Total supply: **1,000,000,000** ZEPH (one billion).
- At listing, circulating float: **120,000,000** ZEPH (12% of supply).
- Listing price: **\$2.00**.

Then:

- Circulating market cap = 120M × \$2.00 = **\$240 million**.
- FDV = 1,000M × \$2.00 = **\$2.0 billion**.

The market-cap-to-FDV ratio is \$240M ÷ \$2,000M = **12%**. So 88% of ZEPH's value, by supply, is in tokens that are not trading yet. The price — the \$2.00 — was discovered on a thin 12% slice, but every one of the billion tokens is now marked at that price. The intuition to carry: *a small amount of real buying can set the price of the entire supply, and most of that supply is a queue waiting to sell into you later.*

That 12% figure is not just a convenient hypothetical. Binance Research, in its May 2024 report, found that tokens launched in 2024 had an average market-cap-to-FDV ratio of about **12.3%** — the lowest of the previous three years. We will come back to why that matters, but note it now: our made-up ZEPH is a faithful model of the real 2024 launch playbook.

![How the listing price is set on a sliver of supply: only about 12% of ZEPH's tokens circulate at listing, so the price of the whole supply is discovered on a thin float while roughly 88% stays locked as future sell pressure.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-2.webp)

### Vesting, cliffs, and the unlock schedule

The locked 88% does not stay locked forever. It is released on a **vesting schedule** — a contract that spells out when each cohort's tokens become tradeable. Two shapes matter:

- A **cliff** is a date on which a big batch unlocks all at once. Before the cliff, zero; on the cliff day, a large slug becomes sellable.
- **Linear** (or "continuous") vesting drips tokens out gradually — a little every day or month.

The whole schedule is usually published at launch (in the tokenomics documentation, and on trackers like Tokenomist or DefiLlama). That is the strange and important thing about crypto: the future supply that will land on the market is *public knowledge*. The **overhang** is the market's word for that known, pending supply — the tokens that are contracted to arrive and must find buyers.

### Listing, and a market maker (briefly)

A **listing** is the moment a token becomes available to trade on an exchange. Two flavors:

- A **DEX** (decentralized exchange, like Uniswap) is a smart contract anyone can trade against. Listing is permissionless — you just create a liquidity pool. But the audience and the liquidity are usually small.
- A **CEX** (centralized exchange, like Binance or Coinbase) is a company that decides which tokens it will list. A top-tier CEX listing puts the token in front of tens of millions of users and is a gated, discretionary decision — which is exactly why it is valuable.

A **market maker (MM)** is a firm that continuously quotes both a price to buy and a price to sell, so that other people can always trade. Without one, a freshly listed token has no reliable liquidity. In crypto the MM is often paid in a very particular way — the project *lends* it tokens and gives it *call options* on the price — which we will touch on and which the next wave of this series, starting with [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), unpacks in full. For now, just hold the idea that a listing does not create liquidity by itself; a market maker does.

### Narrative and attention

Finally, the soft one. **Narrative** is the story that explains why a token should be worth something — "this is the L1 for AI agents," "this is the future of decentralized compute." **Attention** (or "mindshare") is how many people are currently thinking and talking about it. The claim we will defend in Lever Two is not vague hand-waving: in a market where a token has no cash flows, *attention is the demand curve*. If nobody is thinking about ZEPH, nobody buys ZEPH. Manufacturing attention is manufacturing demand.

With those blocks in place, here are the three levers.

## 1. Lever one: the listing is a price event

Start with the plain-English version. Imagine you are selling a rare item. You could sell it at a tiny flea market where forty people might see it, or you could put it on a global auction site with tens of millions of buyers. The item is identical. The *price you can get* is not. A listing is the crypto version of moving from the flea market to the global auction — and because the token's price is set by whoever is willing to trade it, moving venues moves the price.

A top-tier listing does three things at once:

1. **Access.** It exposes the token to the exchange's entire user base. ZEPH goes from a few thousand DeFi-native buyers who know how to use a DEX to tens of millions of people who can buy it with a tap.
2. **Liquidity.** The exchange (and its market makers) provide a deep order book, so larger orders can fill without moving the price as violently. Paradoxically, deeper liquidity makes it *easier* for the price to rise, because demand can express itself without immediately exhausting the book.
3. **Legitimacy.** A listing on a respected venue is read as a stamp of approval — "it passed their review." Whether or not that review means much, the *perception* pulls in buyers who would never touch an unlisted token.

![A top-tier listing is itself a price event: it moves a token from a thin DEX pool with a few thousand buyers to a venue with deep liquidity and tens of millions of users, re-rating the price on access and legitimacy.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-3.webp)

The empirical name for this is the **listing pop** — the jump in price around a new listing. The most-cited measurement is a 2021 study by the crypto data firm Messari, which looked at token prices in the first five days after listing on major venues. It found that Coinbase listings produced an average five-day return of about **+91%**, the highest of any venue studied — but with an enormous spread, from roughly −32% to +645% across tokens. A separate cut that excluded a few extreme outliers put the average nearer **+29%**. Both numbers tell the same story two ways: a top listing tends to lift the price, sometimes violently, and the effect is noisy.

Two honest caveats, because this is where retail gets hurt:

- **The pop often fades.** Messari's own analysis noted that tokens without a real revenue model tended to give back most of the listing premium within weeks. The listing is an access event, not a cash-flow event; once the initial rush of new buyers is spent, there is nothing underneath to hold the price.
- **The effect has weakened over time.** The 2021 numbers came from an era when top-venue listings were rarer and more special. As listings became routine and as more tokens launched, the reliable "free pop" shrank. Treat +91% as a historical high-water mark, not a promise.

### The DEX ceiling: why the small venue caps the price

To feel why access matters, look at what a token can *do* before a top listing. On a DEX, the price comes from a liquidity pool — a pot of two assets (say ZEPH and a stablecoin) that a formula prices against each other. The pool's depth sets how much a buy moves the price. A thin pool means every buyer pays a steep, self-inflicted premium.

#### Worked example: the thin-pool ceiling

Suppose ZEPH trades only in a DEX pool holding \$500,000 of ZEPH and \$500,000 of stablecoin (a \$1,000,000 pool). Using the standard constant-product formula that most DEXs use, a **\$50,000** buy — 10% of one side — pushes the price up by roughly 20% and costs the buyer a chunky slice in **slippage** (the gap between the price you expected and the price you actually got). Now try to deploy \$5,000,000 into that same pool: you would move the price by multiples and hand most of your money to the pool as slippage. The pool simply cannot absorb size.

That is the ceiling a listing removes. A top-tier CEX arrives with market makers quoting a deep book, so a \$5,000,000 order can fill near the quoted price. The same demand that would have been strangled by slippage on the DEX can now express itself — which is *why* the price can re-rate upward on a listing rather than just choking. Deeper liquidity is not neutral; it is the road that lets demand travel. The mechanics of pools, books, and slippage are the subject of [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move); the takeaway here is that the fund needs the listing not only for access but because a thin venue physically caps how high its position can be marked.

#### Worked example: what a listing does to the VC's paper value

Our fund — call it Fund A — led ZEPH's seed round and holds **10,000,000** ZEPH (1% of supply), bought at **\$0.10** for a cost of **\$1,000,000**. Before the top-tier listing, ZEPH trades only in a thin DEX pool at **\$2.00**. Fund A's stake is worth 10M × \$2.00 = **\$20 million** on paper — already 20× its cost.

Now ZEPH lists on a top venue and pops a modest **+50%** in the first days, from \$2.00 to \$3.00 (well within the historical range, and deliberately below the +91% average so we are not cherry-picking). What happens?

- Fund A's paper value: 10M × \$3.00 = **\$30 million**. The pop alone added \$10 million to one fund's mark.
- The token's FDV: 1,000M × \$3.00 = **\$3.0 billion**. On paper, the *whole project* re-rated by a billion dollars — on the strength of a few days of listing-driven buying on a thin float.

Notice what did *not* happen: no new cash flows, no product milestone, no change in the locked 88%. The listing moved the venue and the venue moved the price. That is the first lever. The one-sentence intuition: *a listing changes who can buy and how deep the book is, and both of those re-rate the price of the entire supply without a dollar of fundamentals changing.*

### What this costs and when it breaks

The listing lever is not free to the fund. Top venues can demand listing fees, market-making commitments, or a slice of tokens. And the lever breaks when there is no demand waiting on the other side: a listing into silence is a listing that fades in a day. Which is exactly why Lever One is almost never pulled alone. It is pulled *with* Lever Two.

## 2. Lever two: the narrative machine

Here is the uncomfortable core of it. A stock has cash flows: you can argue about the right price by discounting future earnings. Most tokens have no cash flows at all. So what sets the "right" price? In the absence of fundamentals, price is set by supply and demand for the token itself — and demand, for a thing with no earnings, is almost entirely a function of *attention*. How many people are thinking about it, believe in it, and are willing to buy.

That means demand is *manufacturable*. You cannot easily manufacture a company's earnings, but you can absolutely manufacture attention. This is the second lever, and it is the one that most looks like magic from the outside and most looks like a marketing budget from the inside.

![The narrative machine: research, founder threads, conference stage time, paid influencers and media all manufacture attention; attention becomes net buying; buying becomes a rising price that then feeds still more attention.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-4.webp)

A fund seeds narrative through a stack of channels, most of them entirely legitimate on their face:

- **Research and theses.** Large funds publish research — "the case for modular blockchains," "why AI agents need their own chain." When a multi-billion-dollar fund publishes a thesis, it is not just analysis; it is a signal about where that fund's money (and its portfolio companies) are going. For context on scale: a16z's crypto arm raised a **\$4.5 billion** fund in May 2022 — the largest single crypto venture fund on record — and a **\$2.2 billion** fifth fund in May 2026. When capital that size publishes a thesis, markets listen. (Its operating model and policy machine get their own profile in [a16z crypto: the institutional giant](/blog/trading/crypto-players/a16z-crypto-the-institutional-giant).)
- **Founders and podcasts.** The project's founders go on every podcast and post long technical threads. Genuine communication — and also demand generation.
- **Conference stage time.** Funds and the conferences they sponsor decide who gets the keynote. Stage time is mindshare, allocated.
- **Paid influencers (KOLs).** "KOL" means *key opinion leader* — a crypto influencer with a large following. Projects and funds run **KOL allocation rounds**: influencers receive cheap or free tokens (sometimes with a short lock) in exchange for promotion. The disclosure of these arrangements ranges from thin to nonexistent. This is the channel that most directly crosses from "marketing" into "the audience does not know the person talking was paid in the thing they are shilling."
- **Media coverage.** Favorable coverage in the crypto press, sometimes from outlets with their own commercial ties.

Each channel produces attention; attention produces new buyers; new buyers produce a rising price; and — crucially — a rising price produces *more* attention. That last loop is **reflexivity**: the price move becomes its own advertisement. A token going up is the single most effective narrative there is, because "number go up" pulls in buyers who never read the thesis. We wrote about this feedback loop from the order-book side in [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move); here it is the engine that makes a seeded narrative self-sustaining, right up until the supply changes.

### Why the KOL round is the sharpest tool

Of all the channels, the paid-influencer round is the one that most cleanly turns money into demand, so it is worth understanding its mechanics. In a **KOL allocation round**, a set of influencers receive tokens — often at a discount to the public price, sometimes free, occasionally with a short lock — in exchange for promotion. The influencer is now long the token *and* the person telling their audience to buy it. Their incentive is not to be right; it is for their followers to buy, because follower buying lifts the price of the allocation they are holding. The audience, meanwhile, hears a trusted voice express organic-seeming conviction. The disclosure that would let a listener discount the message — "I was given \$100,000 of this token to say this" — is frequently absent or buried. That gap between what the audience believes it is hearing (a tip) and what is actually happening (paid distribution of insider supply) is the mechanism. It is also why the disclosure question is the fifth item on the defense checklist later in this post.

### A small reflexive loop, in numbers

Reflexivity sounds abstract until you watch it compound. Suppose a seeded narrative brings in \$1,000,000 of net buying on day one, and on a thin float that lifts the price 5%. The 5% gain shows up on price trackers, screenshots circulate, and the "number go up" signal pulls in \$1,500,000 the next day — a bigger move on the now-thinner remaining sell-side. That lifts the price another 8%, which draws \$2,000,000 the day after. None of these buyers needed to read the thesis; each was recruited by the *previous* buyers' price impact. The loop is real demand, but it is demand manufactured by demand, not by anything changing about the token — which is exactly why it reverses just as fast when the supply schedule turns the balance the other way.

#### Worked example: narrative demand versus the supply released

Narrative is worth nothing to a fund unless demand shows up faster than supply. So let us put numbers on the balance across ZEPH's first year. Suppose net buying (demand) and newly-unlocked tokens hitting the market (supply, valued in dollars) move through four phases:

| Phase | Net demand (buying) | New supply released | Net pressure | Price effect |
|---|---|---|---|---|
| Listing week | +\$30M / week | \$5M / week | **+\$25M** | Rises hard |
| Months 1–3 (narrative peak) | +\$15M / week | \$8M / week | **+\$7M** | Grinds up |
| Months 4–6 (attention fades) | +\$6M / week | \$12M / week | **−\$6M** | Rolls over |
| Cliff month (first big unlock) | +\$8M / week | \$40M / week | **−\$32M** | Falls hard |

Read the last column top to bottom and you have the life of a narrative-driven token. While demand outruns supply (the first two rows), the price rises and the story looks vindicated. The moment supply outruns demand (the last two rows), the same token falls — and no amount of fresh threads fixes a supply/demand imbalance of that size. The one-sentence intuition: *narrative sets the demand, but the unlock schedule sets the supply, and the price is just the running difference between them.*

![Demand leads, then supply floods: around listing, seeded demand outruns a tiny float and the price rises; months later, scheduled unlocks flip the balance and the overhang grinds the price down.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-5.webp)

### What this costs and when it breaks

Narrative costs real money — research teams, conference sponsorships, KOL allocations, PR. Funds spend it because the return on a successful narrative is enormous: a story that lifts the token 3× multiplies the value of their entire locked position. The lever breaks in two ways. First, narrative fatigue: audiences move to the next thing, and attention is a rival good — every token is competing for the same finite mindshare. Second, and fatally, narrative cannot beat arithmetic. When the supply from an unlock is many multiples of the demand a story can summon, the story loses. Which brings us to the third lever.

## 3. Lever three: the unlock overhang

The first two levers are about *demand* — bringing buyers to the token. The third lever is about *supply* — and it is the one that actually pays the fund. An unlock is the event where the fund's locked tokens become sellable. Everything before it (the listing, the narrative, the price the market discovered on a thin float) exists to create a deep, liquid, attentive market to sell *into*.

The key mechanical fact: because the float is small, an unlock is often huge relative to the tokens that actually trade. This is where the low-float/high-FDV structure stops being an abstraction and starts hurting.

Two definitions make it precise:

- **Unlock as a percentage of float.** If 50 million tokens unlock into a circulating float of 100 million, that unlock is *50% of the float* — it can, in principle, increase the sellable supply by half in a single day.
- **Sell pressure versus daily volume.** Not all unlocked tokens sell immediately, but the ones that do have to be absorbed by the market's daily trading volume. The right way to size an unlock is to compare its dollar value against how much the token trades per day.

#### Worked example: sizing ZEPH's cliff unlock

At ZEPH's first cliff (month 12), the investor and team tranches begin to vest. Suppose the cliff releases **50,000,000** ZEPH at a market price of **\$2.00**, so **\$100 million** of tokens become sellable in one event. Around that time, ZEPH's circulating float is roughly 100–120 million tokens and its daily trading volume is about **\$10 million**.

- **As a share of float:** 50M unlocked against a ~100M float is roughly a **+50% increase in sellable supply** overnight. That is enormous. (For reference, unlock-research shops flag anything above 5% of circulating supply in one event as a serious red flag; above 20% they consider it a near-guarantee of downward pressure.)
- **Against daily volume:** if the whole \$100M tried to sell, that is **ten full days** of the token's entire trading volume — and daily volume is not one-directional, so the true absorption time is longer. Even if only *half* the unlock sells, \$50M into a \$10M/day market is about **five sessions** of one-sided selling.

![One unlock versus the volume that must absorb it: a \$100M cliff equals ten days of a token trading \$10M per day, so even a half-sold unlock is roughly five sessions of one-sided selling — an overhang, not a one-day dump.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-6.webp)

The one-sentence intuition: *an unlock is not a one-day dump you can wait out; it is an overhang that must be absorbed over many sessions, and the thinner the volume, the longer it grinds.* The fund does not have to crash the price to profit — it just has to feed its supply into the demand the first two levers created, patiently, over the unlock window. [Following the money on a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) shows how to read exactly who holds that unlocking supply and when it vests; here the point is that the schedule is knowable in advance, and that knowledge is the retail buyer's best defense.

### Cliff versus linear, and the front-run

The *shape* of the unlock changes how it hits the price. A **cliff** — a single large batch on one date — is a discrete supply shock; the market braces for it, and the price often sags in the days and weeks before as sellers position ahead of it. **Linear** vesting drips supply out continuously; the pressure is gentler per day but relentless, a slow bleed rather than a shock. Many real schedules are both: a cliff, then a linear tail. ZEPH's month-12 cliff is followed by four years of smaller monthly releases, each a fresh, smaller version of the same overhang.

Because the schedule is public, sophisticated traders **front-run** it. Seeing a large cliff on the calendar, they sell or short the token — or simply refuse to buy — for weeks beforehand, so the price weakens *into* the unlock. This is the mechanism behind the confusing "the price barely moved on unlock day" observation: the move already happened during the anticipation window. Front-running the unlock does not save retail; it just moves the damage earlier, to a moment when there is no dramatic headline to warn a casual buyer that supply is coming. One nuance worth holding: front-running is also self-limiting, because if *everyone* sells ahead of a cliff, the cliff day itself can bounce as shorts cover — which is why unlock trading is treacherous and why the honest retail move is usually to avoid the window, not to try to trade it.

### What this costs and when it breaks

The unlock lever's "cost" to the fund is patience and price impact: sell too fast and you crush your own exit; sell too slow and the overhang narrative depresses the price before you are out. The lever "breaks" in the fund's favor when demand is deep enough to absorb the supply quietly — which is the whole reason Levers One and Two came first. It breaks *against* the fund when the market front-runs the unlock: sophisticated traders, seeing a large cliff on the calendar, short the token or step back from buying weeks ahead, so the price sags into the unlock before a single locked token sells.

## 4. The market maker: the quiet fourth hand

The three levers are the fund's. But there is a fourth party whose behavior the fund often arranges and whose actions land directly in the price: the market maker. A full treatment is a whole wave of this series (starting with [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does)); here is the brief version, because it changes how you read the pop.

Recall that a fresh listing has no natural liquidity. The project hires a market maker to quote continuous buy and sell prices so the token is tradeable. In crypto, the standard contract is unusual and important. Rather than paying the MM a simple fee, the project typically **lends** the MM a large batch of tokens to quote with, and grants the MM **call options** on the token — the right to buy tokens at a fixed "strike" price later. A *call option* is simply a contract that pays off if the price rises above the strike: buy at the low strike, sell at the higher market price, keep the difference.

Follow the incentive. The market maker now profits most when the price rises above its option strikes. It is a motivated *seller* of tokens exactly when the price goes up — because a rising price is when its options are worth exercising. So the very party providing the "neutral" liquidity that supports the listing is also holding a large, price-triggered supply of tokens that hits the market on strength. From the retail buyer's chair, this means the deep, reassuring liquidity around a hot new listing is not purely a service; part of it is a counterparty positioned to sell into your enthusiasm. This is why the third item on the defense checklist is "look for the market-maker deal": a plain monthly-fee arrangement is benign, but an option-loaded deal means another large seller appears precisely when the narrative is working.

Note the elegant, uncomfortable symmetry: the fund wants a listing pop to mark up its position and seed the reflexive loop; the market maker's options pay off on that same pop; and both are selling supply into the demand the narrative created. The interests that are *aligned* here are the insiders' with each other — not the insiders' with the public buyer. That structural misalignment is the throughline of the whole series.

## 5. Putting it together: the VC's realized return, timed to the unlock

Now assemble the machine. A fund's *paper* return is set the moment the token lists and re-rates. Its *realized* return — actual dollars in the bank, the only kind its investors can spend — is set by selling into the unlock window. The gap between the two is why timing the sell to the unlock is the whole game.

Let us complete Fund A's round-trip on its 10,000,000 ZEPH.

#### Worked example: from \$1M cost to \$14.5M realized

- **t = 0 (seed):** Fund A buys 10M ZEPH at \$0.10 = **−\$1,000,000** out.
- **TGE / listing:** ZEPH lists at \$2.00. Fund A's stake is marked at 10M × \$2.00 = **\$20 million** — a 20× *paper* gain. But the tokens are locked; not a dollar is realized.
- **The cliff and the unlock window:** as its tokens vest, Fund A sells into the demand the narrative built, across the falling price its own supply helps create:
  - Sell 3M at \$2.00 = **+\$6.0M**
  - Sell 3M at \$1.50 = **+\$4.5M** (six months later; price already softening under the overhang)
  - Sell 4M at \$1.00 = **+\$4.0M** (twelve months later)
- **Total realized:** \$6.0M + \$4.5M + \$4.0M = **\$14,500,000**, against a \$1,000,000 cost.

![The VC round-trip, timed to the unlock: buy 10M ZEPH cheap and locked at \$0.10, mark it at \$2.00 on listing, then realize the return by selling into public demand across the unlock window for \$14.5M total.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-7.webp)

Read the arithmetic carefully, because it contains the whole thesis:

- Fund A never made a 20× return, despite the 20× paper mark. It made a **14.5× gross return** (\$14.5M proceeds on \$1M cost) — still spectacular, but achieved by selling into a *declining* price, because its own supply was part of what pushed the price down.
- The fund made money **whether or not the token "succeeded."** ZEPH went from \$2.00 to \$1.00 over the window — a 50% drawdown that wiped out most public buyers — and Fund A still 14.5×'d. Its profit came from the cost-basis gap (\$0.10 versus \$1–2), not from the token going up.
- The public buyer's return is the mirror image. Someone who bought the listing pop at \$3.00 and held is down two-thirds. The fund's \$14.5M and the public's losses are the *same dollars* moving across the unlock window.

The one-sentence intuition: *a fund's realized return is manufactured by selling cheap locked supply into demand it helped create — so it can win big on a token that lost most of its buyers money.* This is not a bug the fund exploits; it is the structure. Whose interests collide, and why "aligned incentives" marketing rarely survives this arithmetic, is the subject of [Cui Bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto).

## How it shows up in price

We have looked at each lever's price effect in isolation. Stitch them into the single timeline a buyer actually experiences:

1. **Pre-listing quiet.** The token trades thinly on a DEX or not at all. Insiders are fully locked. Price is whatever the small early market says.
2. **The listing pop.** A top-venue listing and a coordinated narrative burst arrive together. Access, liquidity, legitimacy, and a fresh story hit a tiny float. Demand overwhelms the small circulating supply, and the price gaps up — the +29% to +91%-type move, sometimes far more.
3. **The plateau.** The initial rush is spent. New demand slows; early flippers take profit. The price stops rising and starts chopping sideways. Nothing has "broken" — the demand engine has simply run out of fuel while the supply engine has not started.
4. **The overhang builds.** As the first unlocks approach, the market prices in the coming supply. Sophisticated players reduce exposure or short. The price sags *before* the unlock — the "priced-in" effect that fools people into thinking the unlock was harmless when in fact the damage happened during the anticipation.
5. **The unlock and the grind.** The cliff releases supply that is a large fraction of the float. Even a partial sale is many days of volume. The price grinds lower over the unlock window as insider supply is absorbed. Each subsequent linear unlock is a fresh, smaller version of the same pressure.

The tell that ties it together: **the biggest price gains happen when the float is smallest and the narrative is freshest, and the biggest losses happen as the supply schedule catches up.** That is not a coincidence or a market mood; it is the three levers operating in sequence. A retail buyer who shows up for step 2, drawn by the pop and the story, is systematically arriving right before steps 3 through 5.

## Common misconceptions

**"VCs pump tokens by buying them."** Almost never. A fund does not need to buy the token — it already owns a cheap, huge position. Its levers are on the *supply* side (getting listed, controlling the unlock) and the *demand* side (narrative), not on placing buy orders. Watching for VC buy pressure is watching the wrong hand.

**"The unlock is priced in, so it's fine."** Sometimes true, often not, and the nuance matters. A study by the trading firm Keyrock of more than 16,000 unlock events across 40 major tokens found that around 90% generated negative price pressure. Yet other research (a widely-cited 236-event analysis) finds that for *large, liquid* tokens the drop is mostly anticipated and absorbed before the date. Reconcile them like this: "priced in" is a property of deep, efficient markets. A low-float token with thin volume is the opposite of that — there is not enough liquidity for the overhang to be smoothly discounted, so it grinds lower in real time. "Priced in" is a reason to relax only if the book is deep enough to have done the pricing.

**"A listing is a fundamental milestone."** A listing is an access-and-liquidity event, not a cash-flow event. Nothing about the project's revenue, users, or technology changed the day it listed. Treat the pop as a change in *who can trade*, not in what the thing is worth — which is why the pop so often fades.

**"FDV is the real valuation."** FDV multiplies the price by tokens that mostly do not trade and, in many cases, do not yet exist in circulation. It is best understood as a marketing number that makes a project sound large. The number that reflects real capital is the circulating market cap; the gap between them is the overhang you will be selling against.

**"If the VC is still holding, they must believe in it."** A fund holding through a cliff may simply be *locked*, not loyal. "Diamond hands" that cannot open are not conviction. The meaningful question is not whether they hold but whether they *can sell yet* — and what the vesting schedule says they will do next.

**"Narrative is just information; if the thesis is good, the price is justified."** Attention is demand, and demand is not value. A well-funded narrative can lift a token with no users and no revenue precisely because, without cash flows, there is nothing anchoring price to anything but attention. A good story and a good investment are different things, and the machine profits from the confusion between them.

**"Only investor unlocks matter; the team's tokens are safe."** The team and advisor allocation is frequently the single largest cohort — in Arbitrum's March 2024 cliff, the team-and-advisor slice (about 673.5 million tokens) was actually larger than the investor slice (about 438.25 million). And team tokens are usually *grants*: a cost basis of essentially zero. A holder who paid nothing has the most room to sell into any price and still profit. Reading the cap table means watching the team wallets, not just the funds.

**"An airdrop means the community owns it, so it's fair and decentralized."** An airdrop distributes a slice of supply to users, which sounds like fairness, but two things usually remain true: insiders still hold the majority of the tokens, and airdrop recipients — who also paid nothing — tend to sell quickly, becoming the *first* wave of exit liquidity rather than long-term holders. A generous-looking airdrop can coexist with a cap table where the real ownership, and the real overhang, sits with the funds and the team. Fair distribution is a claim to verify on-chain, not to infer from a headline.

## How it shows up in real markets

The hypothetical ZEPH kept the arithmetic clean. Here are real, dated episodes where the mechanics are visible. Contested attributions are flagged as such; the point is the *mechanism*, not an accusation about any single actor's intent.

### 1. The "Coinbase effect" and the listing pop (2021)

In April 2021, Messari analyst Roberto Talamas published the most-cited measurement of the listing pop. Studying tokens' first five trading days after listing across major venues (Coinbase, Binance, Kraken, and others), he found Coinbase listings produced the highest average return — about **+91%** over five days — but with a range from roughly **−32% to +645%**. A stricter cut excluding extreme outliers put the average closer to **+29%**. The same analysis warned that tokens without revenue models tended to surrender most of the premium within weeks. The lesson for the machine: a top-venue listing reliably created a demand shock — the raw material Lever One is designed to harvest — but the effect was noisy and fleeting, and it has weakened as listings have become routine since 2021.

### 2. Solana's seed-to-liquid cost-basis gap (2018–2020)

Solana's earliest investors bought SOL in a seed round on 5 April 2018 at **\$0.04** per token (raising \$3.2M), according to token-sale trackers. Later private rounds priced at roughly **\$0.20–\$0.25**, and the March 2020 public sale cleared near **\$0.22**. SOL later traded in the triple and quadruple digits (reaching an all-time high around **\$294** in January 2025). Whatever one thinks of Solana as a project, the structural fact is stark: a seed investor's cost basis of \$0.04 against a public price orders of magnitude higher is a paper multiple in the thousands. This is the cost-basis gap from Fund A's worked example, in the wild — the reason an insider can profit enormously even while later public buyers, entering years afterward at vastly higher prices, take on all the downside.

### 3. Arbitrum's March 2024 cliff unlock

On **16 March 2024**, the Arbitrum (ARB) network executed one of the most-watched unlocks of the cycle: about **1.11 billion ARB** — worth roughly **\$1.24 billion** at the ~\$1.12 price at the time — became unlocked, split between team/advisors (about 673.5 million tokens) and investors (about 438.25 million tokens). Critically, that batch equaled roughly **87% of the circulating supply** of about 1.275 billion, after which ARB continued to release tokens every four weeks. This is the low-float/high-FDV structure at industrial scale: an unlock nearly as large as the entire tradeable float, on a publicly-known date. ARB drifted lower into and around the event — though it is important to be precise: the broader market also pulled back in spring 2024, and the unlock was widely anticipated, so the price weakness is *consistent with* overhang pressure rather than cleanly attributable to it alone. That ambiguity is itself the lesson: when supply this large is scheduled in advance, its effect is spread across the anticipation window, which is exactly why "it barely moved on the day" is not evidence the overhang was harmless.

### 4. The low-float/high-FDV playbook (2024)

In May 2024, Binance Research published *Low Float & High FDV: How Did We Get Here?*, quantifying the structure this whole post describes. Its headline findings: tokens launched in 2024 had an average circulating-market-cap-to-FDV ratio of just **12.3%** — the lowest of the prior three years — with circulating supplies often **under 20% of total** at launch. It estimated roughly **\$155 billion** of tokens were scheduled to unlock between 2024 and 2030, and calculated that on the order of **\$80 billion** of new demand would be needed just to absorb the coming supply and hold prices flat. That is the machine described as a market-wide condition: an entire cohort of tokens launched on thin floats at high valuations, with the majority of their supply contracted to arrive later. Our ZEPH, at 12% float, was not an exaggeration — it was the median.

### 5. Why the funds can be this large

The capital behind Lever Two is not hypothetical either. a16z's crypto arm raised a **\$4.5 billion** fund in May 2022 — the largest single crypto venture fund on record — and a **\$2.2 billion** fifth fund in May 2026, per reporting in Forbes and Fortune. Funds of that size can seed research, sponsor conferences, back a broad portfolio, and lend weight to a thesis in a way that genuinely moves where retail attention — and therefore demand — flows. Scale is itself a lever: when the largest funds signal a narrative, the market treats the signal as information, and the reflexive loop does the rest.

## When this matters to you

If you never buy a newly-launched, VC-backed token, most of this is spectator knowledge. But if you do — and the majority of new listings are exactly this — the three levers are usually pointed at you, and the defense is concrete and checkable. Before you buy, pull the five things the fund already knows.

![The retail-defense checklist: before buying, pull the cap table, float versus FDV, any market-maker deal, the unlock calendar, and who was paid to talk — with a reassuring reading and a warning sign for each.](/imgs/blogs/how-vcs-move-price-listings-unlocks-and-narrative-8.webp)

1. **Read the cap table.** What percentage of supply do insiders (team + investors) hold, and how long is their vesting? Insiders under ~30% with long, back-loaded vesting is reassuring; insiders over 50% with a short cliff is a warning. On-chain, you can often see the actual wallets and watch them.
2. **Compute float versus FDV.** Divide circulating supply by total supply. A healthy float (say, above 30% of supply) means less overhang to come; a float under 15% with an FDV ten times the circulating market cap means you are buying the visible tip of a very large iceberg.
3. **Look for the market-maker deal.** If disclosed, are the market makers paid a plain monthly fee, or do they hold call options struck near the current price? Option-based deals mean a large, motivated seller appears exactly when the price rises. (This mechanic gets its own deep dive later in the series.)
4. **Read the unlock calendar.** When is the next cliff, and how big is it relative to daily volume? An unlock worth less than a day of volume is a non-event; one worth more than ten days of volume is a scheduled headwind you can see coming on a public tracker.
5. **Ask who is paid to talk.** Is the person telling you about this token disclosing an allocation? Organic, disclosed enthusiasm is one thing; an undisclosed KOL round is manufactured demand wearing the costume of a friend's tip.

#### Worked example: sizing an unlock in 30 seconds

The most useful of these is the unlock check, and you can do it on a public tracker in half a minute. Pull up the token's next unlock and read three numbers: the unlock's size in tokens, the current price, and the daily volume.

Say a tracker shows the next cliff releases **20,000,000** tokens, the price is **\$1.50**, and daily volume is **\$4,000,000**. Multiply the first two: 20M × \$1.50 = **\$30,000,000** of tokens about to become sellable. Divide by daily volume: \$30M ÷ \$4M = **7.5 days** of the token's entire trading volume, if all of it sold. Even assuming only a third actually sells near-term, that is \$10M — about **2.5 sessions** of one-sided flow. Now compare that unlock to the circulating float: if the float is 60M tokens, a 20M unlock is a **+33% jump in sellable supply**. Three numbers, two divisions, and you know whether the next unlock is a non-event or a scheduled headwind — before you have risked a dollar.

Do the same arithmetic on the *cost basis* if you can find it. If the unlocking cohort bought at \$0.05 and the token trades at \$1.50, they are sitting on a 30× gain and have every reason to sell into any liquidity you provide. An unlocking cohort that is barely above water behaves very differently from one that is up 30×. The cap table tells you which you are dealing with.

None of these requires special access. The cap table, the vesting schedule, the FDV, and the unlock calendar are public. The reason they work as a defense is the same reason the levers work as an offense: the information asymmetry in crypto is not that insiders know secret numbers — it is that they *read the public numbers* and most buyers do not. The synthesis of this whole series, [reading the tape and defending yourself as retail](/blog/trading/crypto-players/reading-the-tape-defending-yourself-as-retail), turns this checklist into a full framework; this post is the part of it about the funds specifically.

This is educational material about market structure, not financial advice, and certainly not an accusation that every VC-backed token is a trap or that every fund acts in bad faith. Plenty of funds back real projects and hold for years. The point is narrower and more useful: the *structure* lets a fund profit handsomely even when a token loses its public buyers money, and that structure is legible in advance if you know where to look. Seeing the machine does not tell you what to buy. It tells you who is on the other side of the trade — which is the first thing a professional wants to know and the last thing a newcomer thinks to ask.

## Sources & further reading

Primary sources behind the headline numbers in this post:

- Binance Research — *Low Float & High FDV: How Did We Get Here?* (May 2024): the 12.3% average market-cap-to-FDV ratio for 2024 launches, the sub-20% circulating-supply observation, and the ~\$155B (2024–2030) unlock estimate. [Binance Research report](https://www.binance.com/research/analysis/low-float-and-high-fdv-how-did-we-get-here).
- Messari / Roberto Talamas, "Coinbase Effect" analysis (April 2021): the ~+91% average five-day return for Coinbase listings, the −32% to +645% range, and the ~+29% outlier-excluded figure, as reported by [CoinDesk / Nasdaq](https://www.nasdaq.com/articles/coinbase-effect-means-average-91-token-price-gain-in-5-days-messari-says-2021-04-07).
- Arbitrum (ARB) unlock, 16 March 2024: ~1.11 billion ARB (~\$1.24B) unlocked, ~87% of the ~1.275B circulating supply, split team/advisors and investors — [CoinDesk](https://www.coindesk.com/markets/2023/08/16/arbitrum-will-unlock-12b-arb-in-march-2024-token-unlocks) and unlock trackers ([DefiLlama](https://defillama.com/unlocks/arbitrum)).
- Solana funding rounds: seed 5 April 2018 at \$0.04 (\$3.2M raised), later rounds \$0.20–\$0.25, March 2020 public sale ~\$0.22 — token-sale trackers ([ICO Drops](https://icodrops.com/solana/)).
- a16z crypto fund sizes: \$4.5B (May 2022, [Forbes](https://www.forbes.com/sites/alexkonrad/2022/05/25/a16z-crypto-record-4th-fund-doubles-down-on-web3-amid-market-crash/)) and \$2.2B (May 2026, [Fortune](https://fortune.com/2026/05/05/a16z-crypto-andreessen-horowitz-fifth-fund-2-2-billion/)).
- Unlock price-pressure research: Keyrock's analysis of 16,000+ unlock events (≈90% negative pressure) and the "priced-in for liquid tokens" nuance from event-study work summarized by unlock-data researchers ([Tokenomist](https://tokenomist.ai/)).

Related posts in this series:

- [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub: who these players are and how they make money.
- [The Lifecycle of a Token: From Seed Round to Unlock Cliff](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) — the full pipeline this post's levers sit on.
- [Follow the Money: Reading a Token's Cap Table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) — how to read who holds the unlocking supply.
- [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move) — the order-book mechanics behind thin-float re-rating and reflexivity.
- [Why a Token Is Not a Stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) — the structural gap that makes the early public exit possible.
- [Cui Bono: the Incentive Map of Crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) — whose interests collide across the stack.
