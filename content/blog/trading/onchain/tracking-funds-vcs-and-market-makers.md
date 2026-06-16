---
title: "Tracking Funds, VCs, and Market Makers: Reading the Institutions On-Chain"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The biggest on-chain moves come from funds, VCs holding vesting allocations, and market makers shuffling inventory — how to find their wallets, read their flows, and separate a real conviction trade from a mechanical, delta-neutral operation."
tags: ["onchain", "crypto", "venture-capital", "market-makers", "token-unlocks", "arkham", "nansen", "smart-money", "ethereum", "otc"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The wallets that move markets belong to **funds** (a16z, Polychain, Jump), **VCs** sitting on cheap, vesting token allocations, and **market makers** (Wintermute, GSR, Amber) who physically shuttle tokens between projects and exchanges. This post teaches you to find those wallets and read what their flows actually mean.
>
> - **Who they are**: VCs and funds buy tokens early and cheap and unlock them on a schedule; market makers borrow tokens from projects to quote both sides of the book. Their wallets are trackable because they are labeled, funded from known multisigs, and receive from vesting contracts.
> - **How to read it**: a VC moving freshly-unlocked tokens *toward an exchange* is imminent supply; a market maker *receiving* a big batch of tokens is usually a new listing or liquidity top-up, **not** a directional bet. The #1 misread is calling a market maker's inventory shuffle a "smart-money buy".
> - **What you DO with it**: front-run unlock sell-pressure (fade the token, tighten stops), ignore delta-neutral MM flow, and treat a fund's *first* on-chain accumulation into a fresh wallet as the rare real conviction signal.
> - The one rule to remember: **most institutional flow is operational, not directional.** A \$50M transfer with zero price impact is an OTC settlement or an inventory move — the size is not the signal; the *destination and the context* are.

On the morning of 20 January 2024, on-chain watchers noticed something simple and alarming on Ethereum: a wallet long-labeled as belonging to an early-stage venture investor in a mid-cap DeFi token claimed a tranche of newly-unlocked tokens out of a **vesting contract**, and within the same hour deposited a slice of them into a Binance hot wallet. The token had a scheduled "cliff" unlock that week — a date everyone could see months in advance — and here was hard, public evidence that at least one of the early backers was moving inventory toward an exit. The token bled ten-plus percent into and through the unlock. None of this required inside information. The vesting contract is public. The VC's wallet had a name on it. The Binance deposit address was labeled. Anyone reading the chain saw the supply coming before the order book did.

That episode is the whole job in miniature. The largest, most market-moving flows in crypto do not come from retail traders or even from the "smart money" wallets that get the most attention — they come from a small set of **institutions**: the funds that wrote the early checks, the VCs holding multi-year vesting allocations bought at a fraction of a cent, and the market makers who are contractually obligated to move tokens around on behalf of projects. These players are not anonymous. Their wallets are clustered, labeled, and funded from known multisigs. With the right tools you can watch a vesting claim, an exchange deposit, an inventory loan, and an OTC settlement happen in real time — and, crucially, tell which of those is a *signal* and which is just plumbing.

![Map of the institutional players — venture funds, vesting VCs, market makers and project treasuries — and their distinct on-chain footprints](/imgs/blogs/tracking-funds-vcs-and-market-makers-1.png)

This post builds the skill from zero. We define who funds, VCs, and market makers actually are and why their on-chain footprints differ. We go deep on the three flows that matter most — a VC dumping a vesting unlock, a market maker receiving inventory for a listing, and an OTC desk settling a block trade that never touches the order book. Then we tackle the single hardest discrimination in all of on-chain analysis: telling a fund's genuine conviction buy from a market maker's mechanical, delta-neutral rebalance, when both look like "\$3M of a token moving into a wallet." Every claim is grounded in money, because in the end a flow only matters if it changes what you do with your capital.

## Foundations: funds, VCs, market makers, vesting, and OTC from zero

Five ideas have to be solid before any of the tracking makes sense. None is exotic, but each is routinely confused, and the confusions are exactly what cost people money. We build each from the ground up, assuming no finance or crypto background.

### A fund and a VC are early, cheap, patient owners

A **venture capital fund** (VC) is a pool of money — raised from pension funds, endowments, family offices, the "limited partners" — that a small team of "general partners" invests on their behalf. In crypto, a VC like Andreessen Horowitz's a16z crypto, Polychain, Paradigm, or Pantera typically writes a check into a project *years before* the token is tradable by the public, and in exchange receives two things: equity in the company, and a large allocation of the future token at a deep discount — often a fraction of a cent when the eventual public price is dollars. A "fund" in the looser sense also includes liquid hedge funds (Jump Crypto's trading arm, Multicoin, Galaxy) that trade tokens that already exist. The thread that ties them together: they hold **large positions acquired cheaply**, and they are the natural supply when those positions vest.

Why does this matter on-chain? Because a VC's edge is *time and price*, not secrecy about where its coins sit. A fund that bought a token at \$0.004 and watches it trade at \$0.40 is sitting on a 100× paper gain; the only question the market cares about is *when does that owner sell, and how much*. The chain answers that question directly: it shows the tokens leaving the vesting contract, landing in the VC's wallet, and — the tell — moving toward an exchange. (For the business model behind these players, the crypto sibling post [Crypto VCs and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) covers the deal structures; here we focus on reading their wallets.)

It is worth being precise about *why* a VC's allocation is such a heavy overhang. The economics are asymmetric in a way retail rarely appreciates. A public-market buyer who paid \$0.40 has the same cost basis as the current price; a 10% drop hurts. A seed VC who paid \$0.004 is up roughly **100×** at \$0.40, so a 10% drop barely dents the return — the VC will happily sell into weakness, into strength, into anything, because *almost any* exit locks in a life-changing multiple. That is the structural reason unlocks press on price: the marginal seller has a cost basis so far below the market that price-insensitivity is rational. A holder who is up 100× is not "waiting for a better price"; they are managing how much they can sell without crashing their own exit. The chain shows you exactly that management in progress — the size and tempo of the deposits to exchanges.

There is also a distinction between two kinds of "fund" that changes how you read the wallet. A **venture fund** holds *vesting* tokens it bought pre-launch; its flow is dominated by the unlock calendar, and its wallets cluster around vesting contracts. A **liquid fund** (a crypto hedge fund) trades tokens that already exist on the open market; its flow looks like ordinary buying and selling — exchange withdrawals when it accumulates, deposits when it exits — and it has no special unlock overhang. When you find a fund wallet, your first question is *which kind* — because a vesting VC's exchange deposit means "cheap supply unlocking," while a liquid fund's exchange deposit means "an active trader changing its mind." Same action, very different information content.

### A market maker is a liquidity provider, not a directional bettor

A **market maker** (MM) is a firm that continuously quotes *both* a buy price (the bid) and a sell price (the ask) for a token, on centralized exchanges and increasingly on decentralized ones, and earns the **spread** — the small gap between the two — plus, often, fees and rebates from the venue or the project. Wintermute, GSR, Amber Group, Cumberland (DRW), and Jump's market-making desk are the names you will see labeled most often. The defining feature of a market maker, the one that *everything* in this post hinges on, is that a market maker is usually **delta-neutral**: it does not want to be long or short the token. It holds inventory only so it can quote, and it constantly hedges that inventory (with perpetual futures, options, or offsetting spot) so that the *price going up or down* is not how it makes money. The spread is how it makes money.

This single fact is the source of the most common, most expensive misread in on-chain analysis. When you see "Wintermute received \$20M of TOKEN," your instinct screams *a smart firm is buying, this is bullish*. Almost always, it is nothing of the sort: the project **lent** Wintermute that inventory so it could provide liquidity for a new exchange listing. Wintermute is not betting the token goes up; it is being paid to quote both sides and will hedge the exposure away. The transfer is **operational plumbing**, not a directional vote. Learning to feel that distinction in your gut is most of the value of this post.

How does the delta-neutral machine actually stay neutral? Say a market maker is quoting a token, posting a bid at \$1.99 and an ask at \$2.01 around a \$2.00 mid. Buyers keep lifting its ask, so it sells token and accumulates a *short* exposure (it owes the tokens it sold). To flatten, it buys an offsetting amount — or, more cheaply, goes *long* the perpetual future on the same token, which gains exactly when its short spot position loses. Now it is **delta-neutral**: if the price rips to \$3.00, the spot leg and the perp leg cancel, and the firm keeps only the spread it earned on the way. The reason this matters for *you*, the on-chain reader, is that the MM's spot wallet balance swings wildly — up \$5M, down \$3M, up \$8M — while its *economic* exposure barely moves. If you read the spot balance as a position, you will see a "firm aggressively buying" that is, in truth, a firm mechanically hedging. The balance is real; the directional meaning you attach to it is imaginary.

There is one more reason MMs receive large batches that has nothing to do with price: the **loan-and-option** structure of most market-making agreements. A project that wants liquidity for its new token does not pay the MM in cash; it *lends* the MM a tranche of tokens (say 1–2% of supply) and grants a **call option** to buy them at a set price later. The MM uses the borrowed tokens as working inventory and, if the token does well, exercises the option for a profit; if not, it returns the inventory. So a "Wintermute received 1% of supply" event is, mechanically, a *loan* being drawn down — the supply still belongs (economically) to the project, and most of it will come back. Counting that 1% as "an MM bought 1% of the token" double-counts supply that was never sold and overstates how much is in strong hands.

### A token unlock is a pre-scheduled supply event

When a VC or team gets its cheap token allocation, it almost never gets the tokens all at once — that would let insiders dump on day one. Instead the tokens are locked in a **vesting contract**: a smart contract that releases them on a public schedule. A typical schedule has a **cliff** (e.g. nothing for the first 12 months, then a big chunk releases at once) followed by **linear vesting** (a steady drip — say 1/36th of the allocation every month for three years). The token's circulating supply therefore grows on a calendar everyone can see in advance, because the vesting contract's terms are public bytecode on the chain.

An **unlock** is the moment a tranche becomes claimable. It matters because it is the cleanest, most predictable form of sell pressure in the entire asset class: new supply hits the market, and the holders receiving it bought far below the current price, so they have every incentive to take profit. (The dedicated mechanics of emission schedules are a sibling topic; this post uses the unlock as the *trigger* you watch the VC wallets around.) The on-chain tell is not the unlock itself — that is on the calendar — it is whether the unlocked tokens then *move toward an exchange*, which is the difference between an investor who holds and one who sells.

A subtlety that trips people up: there is a difference between an unlock being *claimable* and the new supply actually being *circulating and sold*. Many vesting contracts require the beneficiary to actively call a `claim` transaction; until they do, the tokens sit in the contract, technically unlocked but not yet in anyone's spendable wallet. So an unlock date can pass with *no* on-chain movement at all if the holders choose not to claim — a quiet sign they are not in a hurry to sell. The market often pre-emptively prices in the *worst case* (everyone claims and dumps) in the days before a big unlock; if the chain then shows the holders *not* claiming, or claiming and holding, the actual supply is smaller than feared and the token can relief-rally after the unlock passes. This is why reading the chain beats reading the calendar: the calendar tells you what *could* hit the market, the chain tells you what *did*.

It also matters *who* the unlock belongs to. A **team/founder** unlock and an **investor/VC** unlock are both cheap supply, but founders are often more reluctant to sell visibly (it signals a loss of faith and they answer to a community), while financial investors have a fiduciary duty to return capital to their own LPs and sell more mechanically. So when you read a cap table, tag each vesting wallet by *who* it belongs to — the investor tranches are usually the more reliable sellers, and therefore the ones whose exchange deposits you weight most heavily.

### An OTC deal is a block trade that bypasses the order book

When an institution wants to buy or sell a *very* large block — say \$50 million of a token — it cannot just market-buy on an exchange, because doing so would move the price violently against itself (sweeping the order book up 20% as it fills). Instead it uses an **over-the-counter (OTC) desk**: a counterparty (Cumberland, Genesis historically, an exchange's OTC arm) that negotiates a single agreed price privately and settles the whole block in one transfer. On-chain, an OTC settlement looks like a large transfer between two wallets — often via an OTC desk's known address — that produces **zero impact on the public order-book price**, because the trade never touched the order book. That signature, a big move with no price reaction, is one of the most useful and most *misunderstood* patterns on the chain.

OTC matters more in crypto than in most markets because crypto order books are *thin* relative to the size of institutional positions. A token might have a multi-billion-dollar market cap but only a few million dollars of resting bids within a few percent of the mid; the "market cap" is a price multiplied by total supply, not a measure of how much can actually be sold. So the gap between *paper* wealth and *realizable* wealth is enormous, and OTC desks exist precisely to bridge it — to let a large holder convert paper into cash without revealing the intent to the order book and crashing the very price they are trying to capture. Every large fund, VC, and treasury that needs to move size uses OTC routinely; it is not exotic, it is the default rail for institutional-scale flow. Which is exactly why so much of the most important institutional activity is *invisible on the chart* and *visible on the chain*.

### Why these wallets are trackable at all

The reason any of this is readable is the subject of the sibling post [Labeling and Attribution](/blog/trading/onchain/labeling-and-attribution), but the short version: institutional wallets get **named** because they leak ground truth. A fund's wallet is funded from a known **multisig** (a wallet requiring several signatures, often the fund's published treasury). A VC's vesting wallet is, by construction, the **recipient address in a public vesting contract** — the project's own docs and the contract's code name it. A market maker's wallets are confirmed by deposit-pattern clustering and by the projects themselves disclosing their MM partners. Platforms like **Arkham**, **Nansen**, and **Bubblemaps** aggregate these into entity labels you can filter and follow. The labels are probabilistic, not gospel — but for the big institutions they are usually solid, because the ground-truth seeds are strong.

It is worth dwelling on *why* institutions are paradoxically easier to track than ordinary users, because it is counterintuitive. You might expect the sophisticated, well-resourced players to be the hardest to follow. The opposite is true, for three reasons. First, **disclosure**: projects publicly announce their investors and their market-making partners (it is a credibility signal — "backed by a16z, liquidity by Wintermute"), so the *names* are known and only the *addresses* need attaching. Second, **structure**: vesting contracts are public bytecode with named beneficiaries, multisigs are public contracts with known signers, and OTC desks have well-mapped hot wallets — the institutional machine runs on standardized, on-chain primitives that leak their own identity. Third, **scale**: a fund moving \$50M cannot hide in the noise the way a retail user moving \$500 can; the size itself draws the eye of every investigator and labeling pipeline. Ordinary self-custodial users, transacting small amounts through standard wallets, are often *more* private in practice than the institutions, precisely because they are uninteresting and unremarkable. The whales are the most-watched addresses on the chain.

With those five foundations in place, we can read the flows.

## Identifying fund, VC, and market-maker wallets

The first practical skill is finding the wallets at all. You cannot track an entity you cannot locate. There are three reliable handholds, in descending order of certainty.

**The label.** The fastest path is to let Arkham or Nansen do the attribution. Search the entity name ("Polychain Capital", "Wintermute", "GSR") and the platform shows you the cluster it has attributed to that entity, with a portfolio, a transaction history, and a flow graph. This is the 90% solution for the named institutions. The catch — the entire lesson of the labeling sibling post — is that a label is a *probability*, not a fact: a "smart money" tag can be survivorship-biased, and a stale label can name a wallet the entity no longer controls. Read the *basis* of the label, not just the name.

**The funding source.** When a label is thin or you suspect a fund is using a *fresh* wallet to stay quiet, follow the money backward. A new wallet that received its first ETH (for gas) and its first stablecoins from a known fund multisig is, with high probability, that fund's operating wallet — even before any platform labels it. This is how you catch a fund *before* the dashboards do: the conviction buy, by definition, happens in a wallet that is not yet famous.

**The vesting-contract recipient.** For VCs specifically, the cleanest handhold is the vesting contract itself. The project's token-distribution docs list the investor and team allocations; the on-chain vesting contracts have **beneficiary** addresses hard-coded or registered. Read the contract, list its beneficiaries, and you have the exact wallets that will receive every future unlock — months before they do. This is the highest-quality VC signal on the chain because it is structural, not inferred.

A practical refinement on the funding-source handhold: institutions deliberately use **fresh wallets** to avoid telegraphing big trades, so the *most* informative flows often come from addresses with no label yet. The way you catch them is to watch the *known* wallets — the labeled fund multisig, the labeled treasury — and follow their *outflows* to new addresses. When a fund's treasury multisig funds a brand-new wallet with gas and stablecoins, and that new wallet then starts buying a token, you have found the operating wallet *before* any platform labels it. This "follow the funding one hop out" move is how you stay ahead of the dashboards, because the dashboards label wallets *after* they accumulate a history; the conviction trade happens in the gap before that history exists.

A second refinement: institutions cluster their wallets by *function*. A typical fund runs a cold treasury (rarely moves, holds the bulk), several hot operating wallets (active trading), and per-venue deposit wallets (one per exchange). A market maker runs a *lot* of hot wallets that churn constantly plus inventory-holding wallets per project. Recognizing the *function* of a wallet within an entity's cluster tells you how to read its flows: a transfer *out of cold storage* is a rare, deliberate, high-signal event; a transfer between two hot wallets is routine rebalancing you can ignore. The same dollar amount means something completely different depending on which functional wallet it left.

![Three handholds for finding an institution's wallet — the platform label the funding multisig and the vesting contract recipient](/imgs/blogs/tracking-funds-vcs-and-market-makers-2.png)

#### Worked example: pricing a VC's claimable unlock before it moves

You are tracking a DeFi token, "TKN", trading at **\$0.40**. Its docs say early investors hold **5%** of a 1-billion fixed supply — **50 million TKN** — vesting linearly over 36 months after a 12-month cliff. You read the vesting contract on Etherscan and confirm the beneficiary is `0xVC1ce…b30`, a wallet Arkham labels as a known seed-stage fund.

The monthly drip is `50,000,000 / 36 ≈ 1,389,000 TKN`. At today's \$0.40, **each monthly unlock is worth `1,389,000 × \$0.40 ≈ \$555,600`** of *potential* new supply from this one investor alone. The cliff month is larger: the first release is the 12 months of accrued linear vesting, `12 × 1,389,000 ≈ 16.7M TKN`, worth `16,700,000 × \$0.40 ≈ \$6.68M`. You now have a dollar-denominated supply calendar for this wallet, derived entirely from public data, before a single token has moved. *Knowing a single VC can make roughly \$6.68M claimable at the cliff — and where that wallet sends it — is the edge; the calendar is public, the destination is the read.*

## VC flows around unlocks: from vesting contract to exchange

This is the highest-value pattern in the post, because it is the most predictable. The sequence has three legs, and only the third one is the actual signal.

**Leg one: the claim.** On or after the unlock date, the VC's wallet calls the vesting contract's `claim` (or `release`) function. Tokens move *from the vesting contract to the VC's wallet*. By itself this means almost nothing — the tokens were always going to vest, and claiming them is mechanical. A VC that claims and then *sits* is no different from one that never claimed; the supply is the same. So do not react to the claim alone.

**Leg two: the routing.** Now watch where the claimed tokens go. If they move to a **cold wallet**, a staking contract, or a long-term DeFi position, the investor is holding — neutral to mildly bullish, because that supply is *not* hitting the market. If they move toward an **exchange deposit address**, the picture flips: an exchange deposit is, in almost every case, a precursor to a sale, because that is the one thing you do at an exchange that you cannot do in your own wallet. (The general mechanics of why an exchange *inflow* is potential supply are covered in the sibling post [Exchange Flows: Inflows and Outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows).)

**Leg three: the size and tempo.** The magnitude relative to the token's daily volume and the *speed* of the deposit tell you how aggressive the seller is. A VC that drips 5% of an unlock to an exchange over two weeks is managing market impact and is a slow, absorbable bleed. A VC that dumps the entire cliff into a hot wallet in one transaction the moment it unlocks is an urgent seller, and the token will likely gap down.

![Pipeline of an unlock — tokens leave the vesting contract reach the VC wallet then route to an exchange deposit as imminent sell pressure](/imgs/blogs/tracking-funds-vcs-and-market-makers-3.png)

#### Worked example: a VC claims \$8M and deposits \$5M to Binance

The cliff hits for token "TKN" at \$0.40. You watch `0xVC1ce…b30`. In one transaction it claims **20,000,000 TKN** from the vesting contract — at \$0.40 that is **\$8,000,000** of newly unlocked tokens landing in the VC's wallet. Leg one done; no conclusion yet.

Forty minutes later, the same wallet sends **12,500,000 TKN** — worth `12,500,000 × \$0.40 = \$5,000,000` — to a Binance deposit address (labeled by Arkham). The remaining **\$3M stays** in the wallet. The read is now sharp: roughly **\$5M of imminent sell pressure** is on its way to the order book, while \$3M is being held back (perhaps to sell into strength later). If TKN's average *daily* spot volume is, say, \$8M, then \$5M of one-sided supply is more than half a day's liquidity arriving at once — enough to push price down several percent as it absorbs. *A claim is noise; a claim followed by a \$5M exchange deposit within the hour is a quantified, datable sell signal you can position around.*

## Market-maker flows: inventory in, quotes out, delta-neutral

Now the flow that fools the most people. A market maker's on-chain life has a completely different shape from a VC's, and once you see the shape you stop misreading it.

A market maker does not *own* a directional view. It receives **inventory** — usually *borrowed* from the project under a market-making agreement (often structured as a loan with an option, but the on-chain footprint is a transfer of tokens from the project's treasury or a multisig to the MM's wallet). It then uses that inventory to **quote both sides** of the book on one or more exchanges: it places bids and asks, fills incoming orders, and pockets the spread. Critically, it **hedges**: if it accumulates a long position from buyers hitting its ask, it shorts perpetual futures to flatten the exposure, so that its profit comes from the spread and not from price direction. The whole machine is built to be indifferent to whether the token rises or falls.

What does this look like on-chain? Three recurring signatures:

- **A project-to-MM transfer** of a large, round batch of tokens, often timed just before a new exchange listing or a liquidity expansion. This is inventory delivery — bullish only in the weak sense that the project is investing in liquidity, *not* a bet that the price rises.
- **MM-to-exchange and exchange-to-MM churn**: continuous two-way flow between the MM's wallets and exchange deposit/withdrawal addresses as it rebalances inventory across venues. This is the heartbeat of market making and carries almost no directional information.
- **An MM-to-project return** of inventory, which often signals the *end* of a market-making engagement — a delisting, a contract expiry, or a project pulling liquidity. This can be quietly bearish for the token's depth.

Increasingly, market makers also provide liquidity on **decentralized** exchanges, which makes their footprint even more visible. On a centralized exchange you only see the MM's *deposits and withdrawals*; the quoting itself happens in the exchange's private matching engine, off-chain. On a decentralized AMM (an automated market maker like Uniswap), the MM's liquidity provision is *itself on-chain*: you can watch it add tokens to a liquidity pool, see the pool's depth change, and watch it remove liquidity when the engagement ends. A large MM adding liquidity to a fresh pool is the DEX-native version of inventory delivery for a listing; a large MM *removing* liquidity is the depth draining out. Both are operational, both are delta-neutral in intent (the MM hedges its impermanent-loss exposure), and both are misread as directional by people who see only the size. The mechanics of how an AMM pool works are covered in the crypto sibling [DeFi Protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao); here the point is simply that on-chain liquidity provision is *more* legible than CEX market making, not less.

![Graph of a market maker — inventory in from the project quotes on both sides of CEX and DEX with a hedge keeping it delta-neutral](/imgs/blogs/tracking-funds-vcs-and-market-makers-4.png)

#### Worked example: a market maker receives \$20M of inventory for a listing

A mid-cap token "MKR-X" trades at **\$2.00**. You see the project's treasury multisig transfer **10,000,000 MKR-X** — worth `10,000,000 × \$2.00 = \$20,000,000` — to a wallet Nansen labels "Wintermute". A naive reading: *Wintermute just bought \$20M, hugely bullish.* The correct reading: this is **inventory delivery**, almost certainly ahead of a new exchange listing, under a market-making loan. Wintermute did not pay \$20M for these tokens and is not long \$20M of MKR-X; it received them to *quote* MKR-X, and it will hedge the exposure with perpetual shorts so its net delta is roughly zero.

How do you confirm it is operational, not directional? Two checks. First, within hours you typically see MKR-X flowing from the Wintermute wallet *to* exchange deposit addresses — inventory being positioned to quote, not hoarded. Second, the token's open interest in perpetual futures jumps as the hedge goes on. The combination — a round \$20M batch from the project, fanned out to exchanges, with a matching perp hedge — is the unmistakable fingerprint of a listing, not a conviction buy. *A market maker receiving \$20M is the project buying liquidity, not the MM buying the token; misreading the two is the single most expensive beginner error in institutional on-chain analysis.*

## The #1 misread: conviction buy versus inventory shuffle

Here is the discrimination that separates a useful analyst from a dashboard-follower. You will constantly see transfers of similar *size* that mean opposite things. A \$3M token inflow to a fund's wallet might be a genuine conviction buy — a bet that the token goes up — or it might be a market maker's routine inventory rebalance that carries no view at all. Size alone cannot tell them apart. You need the **context signature**.

A **conviction buy** by a fund has a characteristic shape. The tokens are *acquired with stablecoins or ETH* — the wallet spends real money to buy spot, often *from* an exchange (an exchange **outflow** of the token into the fund's wallet, paid for with a matching stablecoin inflow to the exchange). It typically lands in a **fresh or cold wallet** rather than a hot operational one, signaling intent to hold. There is **no offsetting hedge** — the fund *wants* the directional exposure. And it is often a *first* position, with no prior history of that token in the cluster.

A **market-maker rebalance** has the opposite signature. The tokens arrive *from the project's treasury or from another exchange*, not bought with fresh stablecoins. They land in a **known hot/operational MM wallet** that already churns that token constantly. There is a **matching hedge** in the perp market. And it is part of a *continuous two-way pattern* — tokens flow out again almost as fast as they flow in. The wallet's *net* position barely changes; it is the *gross* throughput that is large.

![Matrix contrasting a fund conviction buy versus a market-maker inventory shuffle across funding hedge wallet type and net position](/imgs/blogs/tracking-funds-vcs-and-market-makers-5.png)

The disambiguation rule, compressed: **follow the payment and the hedge.** If the tokens were *paid for* with stablecoins/ETH and there is *no hedge*, someone took a directional view — that is signal. If the tokens *arrived from a project or venue* and there *is* a hedge (or the wallet immediately routes them back out to exchanges), it is operational plumbing — that is noise.

#### Worked example: a \$3M conviction buy versus a \$3M MM rebalance

Two wallets each receive **\$3,000,000** of the same token "ZTK" (at \$1.00, so **3,000,000 ZTK** each) on the same day. Identical size. Opposite meaning.

Wallet A is a fresh address funded from a fund's published treasury multisig. To get its 3M ZTK it sent **\$3,000,000 of USDC to Coinbase** and pulled **3,000,000 ZTK out** to its own wallet — a clean exchange outflow paid for with stablecoins. No perp short appears. The ZTK then sits. This is a **conviction buy**: real money spent, exposure kept, intent to hold. It is a (rare) genuine smart-money signal.

Wallet B is a long-labeled GSR market-making wallet. Its 3M ZTK *arrived from the ZTK project treasury*, not bought with stablecoins; within the hour, **2.8M ZTK flowed back out** to three different exchange deposit addresses, and ZTK perpetual open interest rose by roughly the position size. This is an **inventory shuffle**: net delta near zero, no view, pure plumbing. *Two \$3M inflows, one a directional bet and one a hedge-flat rebalance — the funding source and the hedge, not the size, are what tell them apart.*

## OTC desks: a \$50M move with no price impact

One pattern deserves its own section because it is so counterintuitive: a *huge* transfer that the price completely ignores. New analysts see "\$50M of TOKEN moved" and brace for a crash that never comes. The reason is that the trade was **OTC** — it never touched the order book.

Recall the mechanics. An institution that wants to sell \$50M of a token in the open market would have to walk down the order book, selling into successively lower bids, and might realize an average price several percent below the screen price while crashing it for everyone — the **market impact** of a large order. To avoid this, it routes the block through an **OTC desk**, which finds a counterparty (or warehouses the block itself) and agrees a single price privately. On-chain, the settlement is a transfer: \$50M of the token from seller to buyer (often hopping through the OTC desk's known wallet), and a matching stablecoin transfer the other way. The public order book never sees the supply, so the **price does not move** — even though ownership of \$50M just changed hands.

This has two consequences for the reader. First, **absence of price impact is not absence of a transaction.** A whale can fully exit a position OTC and the chart will show nothing; you only catch it on the chain, by watching the transfer and recognizing the OTC desk's wallet. Second, the *direction* of an OTC deal is informative even when the price isn't: a known long-term holder settling a large block *to* an OTC desk is distributing (a slow-motion exit), while a fund receiving a block *from* an OTC desk is accumulating without telegraphing it. The order book hides them; the chain does not.

![Before and after of an OTC block trade — 50M dollars settles off the order book so the public price is unchanged](/imgs/blogs/tracking-funds-vcs-and-market-makers-6.png)

#### Worked example: a \$50M OTC transfer the chart never sees

A holder wants out of **\$50,000,000** of a token "WHL" trading at **\$5.00** (10,000,000 WHL). The token's entire order book within 3% of the mid holds maybe **\$4M** of bids. Selling \$50M into that book would be catastrophic: after eating \$4M of bids the price would be down sharply, and the *next* \$46M would fill far lower — a realized average price perhaps **15–25% below** the screen, and a visible crash on every chart.

Instead, the holder calls an OTC desk. The desk lines up a buyer (or warehouses the block) at an agreed **\$4.95** — a modest **1% discount** to screen for the convenience and certainty. On-chain you see **10,000,000 WHL** move from the holder to the OTC desk's wallet, and roughly **\$49.5M of USDC** move to the holder. The public price ticks along at \$5.00, unbothered, because **not one of those 10M tokens hit the order book**. A reader watching only the chart concludes nothing happened; a reader watching the chain saw a \$50M ownership change and can infer distribution. *A \$50M move with zero price impact is the signature of an OTC settlement — proof that the most important flows are sometimes the ones the price action deliberately hides.*

## Reading a project's cap-table and treasury wallets

Zoom out from a single transfer to the whole ownership picture. For any token, the most valuable map is its **cap table on-chain**: who holds the supply, how much is locked, and where the big allocations sit. Arkham and Bubblemaps make this visible by clustering and labeling the top holders.

The wallets to find and watch are: the **project treasury** (the protocol's own war chest, usually a multisig — funds its operations and, sometimes, sells to cover runway); the **team/founder allocation** (vesting, watched exactly like a VC allocation); the **investor allocations** (the VC vesting contracts from earlier); the **market-maker inventory** wallets (the borrowed liquidity, which is *not* "supply someone bought" and should be mentally subtracted from holder-concentration alarm); and the **community/airdrop/ecosystem** buckets. Reading these together tells you what fraction of "circulating" supply is really controlled by a handful of aligned insiders versus genuinely distributed — the concentration question covered in the sibling post [Supply Distribution and Holder Concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration).

The treasury itself is a leading indicator. A project treasury that starts **routing tokens or stablecoins to an exchange** may be selling to fund operations — neutral-to-bearish supply that, unlike a VC unlock, is not on a fixed calendar and so can surprise the market. A treasury that *receives* stablecoins (from a fundraise or revenue) and holds is healthier. And a treasury that suddenly transfers a large batch to a market maker is, as we saw, usually preparing liquidity for a listing rather than selling.

Two important caveats keep this honest. First, **market-maker inventory is not ownership** — when Bubblemaps shows "Wintermute holds 4% of supply," that 4% is mostly *borrowed* and will return to the project; counting it as concentrated insider holding overstates the risk. Second, **a treasury multisig moving funds is often pure operations** — paying a vendor, rotating to a new cold wallet, topping up a bridge. Not every treasury outflow is a sale. As with everything in this series, the destination and the context decide the meaning, not the size.

There is a deeper reason the cap-table read is the most important habit you can build around a token. The single best predictor of a token's medium-term price is not the chart or the narrative — it is **supply overhang**: how much cheap, insider-held supply is going to unlock over the coming year relative to how much real demand exists to absorb it. A token can have a brilliant product and a passionate community and still bleed for a year because 60% of its supply belongs to VCs and a team who are unlocking on a relentless schedule and selling every tranche. The cap table, read on-chain, is how you quantify that overhang *before* it crushes you. You are effectively computing a forward supply schedule: for each insider wallet, how many tokens unlock each month, at what dollar value, and — by watching the historical pattern of whether that wallet sells its unlocks — how likely each tranche is to hit the market. That forward schedule, weighed against the token's daily absorptive volume, is the closest thing on-chain analysis has to a fundamental valuation input.

A worked instinct to internalize: a treasury or insider cohort that has *historically sold* every unlock into exchanges is a high-confidence future seller; one that has *historically held* (routed to staking, cold storage, or re-deployed into the protocol) has earned the benefit of the doubt. The chain remembers. You do not have to guess whether this VC will sell the next tranche — you can look at what it did with the last three.

![Graph of a token cap table — treasury team investors market-maker inventory and community wallets and what each flow implies](/imgs/blogs/tracking-funds-vcs-and-market-makers-7.png)

#### Worked example: separating real float from MM inventory

Token "CAP" has **100,000,000** tokens "circulating" at **\$3.00** — a **\$300M** circulating market cap on the dashboard. You read the cap table on Bubblemaps. The top wallets: a market maker holds **8,000,000 CAP** (\$24M), the team's vesting wallet holds **6,000,000** still-locked (\$18M), and two VC wallets hold **5,000,000** each (\$15M apiece, \$30M together) that unlock next quarter.

Subtract what is *not* genuinely free-floating, sell-ready retail supply: the **\$24M** of MM inventory is borrowed and delta-neutral (not someone's directional position), and the **\$18M** team allocation is locked. The two VC allocations, **\$30M**, are the real overhang — cheap tokens that *will* unlock and can be sold. So of the headline \$300M cap, roughly **\$30M** is imminent insider supply concentrated in two wallets, while \$24M of the "holdings" is just plumbing that flatters the concentration stats. *The dashboard's \$300M market cap hides the only number that matters for risk — the \$30M of cheap VC supply about to unlock — and tells you the \$24M of MM "holdings" is not a position at all.*

## How to read it: a walkthrough on Arkham and Etherscan

Tools turn this from theory into a repeatable routine. Here is a concrete pass for a token you are worried about, using **Arkham** (entity labels and flow graphs), **Nansen** (smart-money and MM labels), and **Etherscan** (the raw vesting contract). The token is the illustrative "TKN" from earlier, at \$0.40, with a cliff unlock this week.

**Step 1 — Map the cap table.** Open the token's page on Arkham. Sort holders by balance and read the labels: which are the treasury multisig, which are VC/investor wallets, which are market-maker inventory, which are exchange wallets. Note any wallet whose label is "Smart Money" or a named fund. This is your watchlist. Mentally tag each as *directional owner* (VC, fund, team) or *operational* (MM, exchange, treasury).

**Step 2 — Read the vesting contract.** On Etherscan, open the project's vesting contract (linked in its docs). Read its beneficiaries and the unlock schedule. Confirm the cliff date and the tranche size. Compute the dollar value at the current price, as in the first worked example. You now know *how much* cheap supply can move, *to whom*, and *when*.

**Step 3 — Set the watch.** On Arkham, follow the VC beneficiary wallets and the treasury multisig. Arkham can alert you when a watched wallet transacts. The trigger you care about is **vesting-contract claim → onward transfer to an exchange deposit address**. That two-hop sequence is the sell signal; the claim alone is not.

**Step 4 — Distinguish the MM noise.** When you see the project treasury transfer a large batch to a wallet, check Nansen's label. If it is a known market maker, expect inventory plumbing: look for the tokens fanning back out to exchanges and for a rise in perp open interest. *Do not* mark this as a buy or a sell — it is delta-neutral. Only an *investor or team* wallet routing to an exchange is directional supply.

**Step 5 — Watch for the OTC tell.** If a large holder's wallet sends a big block to a known OTC desk wallet (Arkham labels several), and the price *does not move*, you have caught a distribution (or accumulation) the chart will never show. Log the direction.

**Step 6 — Translate to a decision.** Sum the imminent insider supply (VC unlocks routed to exchanges) against the token's daily volume. If a single day's unlock-driven supply is a large fraction of daily liquidity, the token faces real downward pressure into and through the unlock — fade strength, tighten stops, or stand aside. If the unlocked tokens move to cold storage or staking, the supply threat is deferred and you can relax. This is the same decision logic the sibling case study [Tracing a Real Flow End to End](/blog/trading/onchain/case-study-tracing-a-real-flow-end-to-end) walks through for a full investigation.

![Decision matrix — entity type plus flow type maps to a read and an action for the trader or analyst](/imgs/blogs/tracking-funds-vcs-and-market-makers-8.png)

## Where the data does and doesn't exist

A note on honesty, because this series refuses to invent numbers. There is no clean public *fund-flow* time series — no curated dataset that says "VCs deposited \$X to exchanges around unlocks each month." The figures in this post that carry hard numbers (the unlock math, the OTC discount, the cap-table split) are *worked illustrations* with realistic but constructed values; the institutions and patterns are real, the specific dollar amounts are teaching examples, not a dataset.

Two *real* series do give legitimate context for institutional flow, and both come from this series' curated data:

- **Stablecoin supply** is the dry-powder gauge — the total pool of "ready cash" that funds, MMs, and OTC desks deploy. It grew from roughly **\$28B (2020)** to **\$230B (2025)**. When this pool expands, there is more institutional capital available to absorb unlocks and settle blocks; when it contracts, supply events bite harder. (The mechanics are in the sibling [Stablecoin Flows: The Dry-Powder Metric](/blog/trading/onchain/stablecoin-flows-the-dry-powder-metric).)
- **Bitcoin held on exchanges** is the long-run "supply on tap" — coins sitting where they can be sold. Its multi-year decline from **~2.6M to ~2.4M** coins is the macro backdrop against which any single unlock's supply must be weighed.

These two charts anchor the post in real, sourced data without pretending a fund-flow dataset exists that does not.

![Stablecoin total supply growing from about 28 billion dollars in 2020 to about 230 billion in 2025](/imgs/blogs/tracking-funds-vcs-and-market-makers-9.png)

The dry-powder reading is straightforward: a \$230B stablecoin float is a vastly deeper sponge for absorbing a \$5M unlock or a \$50M OTC block than the \$28B float of 2020 was. The *same* token unlock lands very differently into a flush market than a drained one — which is why institutional flow always has to be read against the liquidity backdrop, not in isolation.

#### Worked example: the same unlock in a flush market versus a drained one

Take the \$5M exchange deposit from the VC worked example earlier. In a flush market — a \$230B stablecoin float, deep order books, plenty of buyers — \$5M of one-sided supply is a ripple: market makers and dip-buyers absorb it over a session, and the token might dip **1–2%** before recovering. In a drained market — a post-crisis \$28B-equivalent float, thin books, no bid — that *same* \$5M hits a market with maybe \$1M of resting bids within a few percent, and the token can gap **down 10–15%** as the supply overwhelms the available demand. The unlock did not change; the *liquidity it landed in* did. *A \$5M deposit is a 1% wobble or a 12% crash depending entirely on the dry-powder backdrop — the institutional flow and the liquidity gauge must be read together, never apart.*

![Bitcoin held on exchanges declining from about 2.6 million to about 2.4 million coins](/imgs/blogs/tracking-funds-vcs-and-market-makers-10.png)

## Common misconceptions

**"A market maker receiving tokens is a bullish buy signal."** The most expensive myth in this post. A market maker receiving inventory is the *project* buying liquidity, usually for a listing — the MM is delta-neutral and will hedge the exposure away. The \$20M Wintermute "received" in the worked example was a loan to quote, not a directional bet. Read MM inflows as plumbing; only an *investor* or *team* wallet spending real money is directional.

**"A big transfer means a big price move is coming."** Not if it is OTC. A \$50M block settled through an OTC desk produces *zero* order-book impact, because the tokens never hit the book. Size is not the signal — destination and venue are. The transfers that move the price are the ones routed to *exchange order books*, not the ones settled privately.

**"A VC claiming an unlock is selling."** A claim is mechanical — the tokens were always going to vest. Claiming and *holding* (routing to cold storage or staking) is not selling. The sell signal is specifically the *second hop*: claim, **then** deposit to an exchange. Reacting to the claim alone front-runs a sale that may never come.

**"Institutional wallets are anonymous, so you can't track them."** Institutional wallets are among the *easiest* to track, because they leak strong ground truth: funding from published multisigs, beneficiary addresses hard-coded in public vesting contracts, and disclosure of MM partners. The labels are probabilistic, but for the major funds and MMs they are usually solid. (Why labels are still fallible: the sibling [Labeling and Attribution](/blog/trading/onchain/labeling-and-attribution).)

**"If the chart didn't move, nothing happened."** The opposite is often true at the institutional level. OTC settlements and inventory shuffles are *designed* to leave the price undisturbed. The most important institutional flows are frequently the ones that produce no chart reaction at all — which is precisely why you read the chain and not just the candles.

**"More holders / lower concentration is always healthier."** Usually, but the cap-table read complicates it. A token that looks beautifully distributed on a holder-count basis can still be dominated by a handful of insider vesting wallets whose tranches dwarf the retail float; conversely, a token with a "scary" 4% in a single wallet may be fine if that wallet is *market-maker inventory* that will be returned. Raw concentration numbers, read without knowing *which* big wallets are insiders, MMs, treasuries, or exchanges, mislead in both directions. The fix is to label the big holders by function before drawing any conclusion — exactly the discipline this post is about.

## The playbook: what to do with it

The if-then checklist, for a trader or analyst who has done the wallet-mapping above. Each line: the signal → the read → the action → the invalidation.

- **Signal: a VC vesting wallet claims an unlock, then routes to an exchange within hours.** Read: imminent, quantifiable sell pressure. Action: fade strength into the unlock, tighten stops, or stand aside; size the threat by unlock-\$ vs daily volume. Invalidation: the claimed tokens move to cold storage or staking instead → supply threat deferred, stand down.

- **Signal: a project treasury transfers a large round batch to a labeled market maker.** Read: inventory delivery for a listing or liquidity top-up — delta-neutral, *not* directional. Action: do nothing on the transfer itself; if anything, note that improved liquidity is mildly constructive. Invalidation: the wallet is *not* an MM but a fund spending stablecoins → re-classify as a possible conviction buy.

- **Signal: a \$3M-class inflow to a wallet.** Read: ambiguous until you check funding and hedge. Action: trace the payment — stablecoins/ETH spent + no hedge + fresh wallet = conviction buy (rare signal, weight it); arrived-from-treasury + hedged + churns back out = MM rebalance (noise, ignore). Invalidation: the "fund" wallet immediately routes the tokens back to exchanges → it was operational after all.

- **Signal: a large block settles to/from a known OTC-desk wallet with no price reaction.** Read: an off-book distribution (to the desk) or accumulation (from the desk) the chart hides. Action: log the *direction* as a slow-supply or slow-demand bias; weight it lightly but don't ignore it. Invalidation: the counterparty turns out to be an internal cold-wallet rotation, not a trade → no ownership change, disregard.

- **Signal: a project treasury starts routing tokens or stablecoins to an exchange off-schedule.** Read: possible operational selling to fund runway — unscheduled supply that can surprise. Action: treat as a soft bearish overhang; watch the cadence. Invalidation: the destination is a vendor, a bridge, or a new cold wallet, not an exchange → pure operations, no supply.

The meta-rule that ties the whole post together: **most institutional flow is operational, not directional.** Vesting claims, MM inventory loans, treasury rotations, OTC settlements — the *majority* of the big numbers you will see are plumbing. The rare directional signal — a fund spending real stablecoins into a fresh wallet and holding, or a VC routing an unlock straight to an exchange — is valuable precisely *because* it stands out against that operational noise. The skill is not seeing the big transfers; everyone sees those. The skill is knowing which ones mean a human took a view, and which ones are just the machine moving tokens from one slot to another.

## Further reading & cross-links

- [Crypto VCs and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the business models and deal structures behind the wallets this post tracks.
- [Labeling and Attribution](/blog/trading/onchain/labeling-and-attribution) — how a hex address becomes "Wintermute" or "Polychain", and why every label is a probability, not a fact.
- [Exchange Flows: Inflows and Outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — why a deposit to an exchange is potential supply and a withdrawal is potential demand.
- [Supply Distribution and Holder Concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — reading the cap table, and why MM inventory should be subtracted from concentration alarm.
- [Stablecoin Flows: The Dry-Powder Metric](/blog/trading/onchain/stablecoin-flows-the-dry-powder-metric) — the cash pool that funds, MMs, and OTC desks deploy.
- [Tracing a Real Flow End to End](/blog/trading/onchain/case-study-tracing-a-real-flow-end-to-end) — a full investigation that chains these reads together.
