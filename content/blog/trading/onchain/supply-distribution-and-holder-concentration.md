---
title: "Supply Distribution and Holder Concentration: Who Actually Owns This Token?"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-to-deep guide to reading a token's holder list — top-N concentration, supply bands, Gini and Nakamoto metrics, and the one skill that separates a rug from a fair launch: reclassifying the top holders before you judge."
tags: ["onchain", "crypto", "holder-concentration", "supply-distribution", "tokenomics", "whale-watching", "rug-pull", "bubblemaps", "etherscan", "memecoins"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A token's price is hostage to *who holds it*. If a few unlocked wallets control most of the float, one of them can crater the price; a broadly held token behaves completely differently. The whole skill is reading the holder list and reclassifying it before you judge.
>
> - **What the signal is**: the holder list is the token contract's balance ledger, ranked. Concentration metrics — top-10 %, top-100 %, Gini, Nakamoto coefficient — measure how lopsided that ledger is.
> - **How to read it**: open the contract's *Holders* tab on Etherscan/Solscan, then **reclassify the top 20**. A burn address, an LP pool, an exchange wallet, and a locked team vest are *benign*; a fresh person-controlled wallet holding 30% is a loaded gun. Bubblemaps shows you which "separate" wallets are secretly one entity.
> - **What you do with it**: a concentration *that's locked or custodial* is fine to hold through; a concentration that is *free-float and person-controlled* means avoid, hedge, or size tiny — and watch every unlock.
> - The rule to remember: **concentration is not automatically risk — un-reclassified concentration is.** Always ask *can this holder actually dump the free float right now?* before you call it a red flag.

On a quiet day in 2021, a token you have never heard of did what thousands of tokens do every week: it went vertical, then it went to zero. The chart told a clean story of a 90% collapse in under an hour. But the *chain* told a richer one. A single wallet — funded days earlier from a fresh address with no history — had quietly accumulated roughly a third of the circulating supply during the hype phase, then sold it all into the buyers chasing the green candle. The holders who got crushed could have seen it coming. The information was sitting in plain sight on the token's *Holders* tab the entire time: one address, one third of the float, no lock, no label. They didn't lose because the market was unfair. They lost because they read the price and never read the ledger.

That episode is not exotic. It is the single most common way retail money disappears in crypto, and it has a name: **holder concentration risk**. The mirror image is just as instructive — countless "scary looking" tokens where five wallets hold 70% of supply turn out to be perfectly safe to hold, because four of those five wallets are a burn address, an exchange, a liquidity pool, and a time-locked team contract. The concentration is real; the *risk* is not. The difference between those two situations — a deadly 30% whale and a harmless 70% spread across contracts — is the most valuable read in token due diligence, and almost nobody does it properly.

This post teaches that read from zero. We start with what a holder list actually *is* (the contract's balance ledger), build up the metrics that summarize it (top-N percentages, Gini, the Nakamoto coefficient), learn the wealth bands that group holders by size, and then spend most of our time on the skill that ties it all together: **reclassifying the top holders before you judge concentration.** Get that wrong and you mislabel a treasury as a rug or a rug as a treasury. Get it right and the holder list reads like a verdict.

![Supply concentration mental model showing total supply split into locked contract supply and free float with whales and a retail tail](/imgs/blogs/supply-distribution-and-holder-concentration-1.png)

The figure above is the mental model for the whole post. Total supply is *everything the contract ever minted*. A large slice of it is usually **locked or held by contracts** — the burn address, team vesting, the treasury, staking pools, and liquidity pools. What's left is the **free float**: the tokens that real holders can actually sell right now. And concentration is really a question about that free float: is it spread across thousands of small holders (a long tail), or is it bunched into a handful of whales who could each move the price alone? Keep this picture in your head; everything below is a way of measuring and verifying it.

## Foundations: what a holder list is, and the four supply numbers

Before any metric makes sense, four ideas have to be solid: what a holder list is, the difference between an *address* and an *owner*, what it means for supply to be held by a *contract* versus a *person*, and the three supply numbers (circulating, total, fully-diluted) that every concentration claim secretly depends on. None of this is crypto-magic — it is bookkeeping, made public.

### A holder list is the token contract's balance ledger, ranked

On a chain like Ethereum, a token is not a coin that physically moves. A token is a **smart contract** — a program living at one address — and inside that contract is a table that maps each holder's address to a number: their balance. When you "send 100 tokens," nobody moves a coin; the contract subtracts 100 from your row and adds 100 to the recipient's row. That table, called `balanceOf` in the ERC-20 standard, *is* the ownership record. (If you want the full mechanics of how a transfer and an approval actually work inside that contract, see the sibling post on [tokens, on-chain transfers, and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals).)

A **holder list** is simply that table, sorted from largest balance to smallest. An explorer like Etherscan reads the contract's balance ledger, ranks every non-zero address, and shows you: rank, address, balance, and percentage of supply. That's it. The "*Holders*" tab is not some special analytics product — it is the raw, public, always-up-to-date ledger of who owns what, presented as a leaderboard.

This is the first thing that makes crypto analysis possible at all: **the cap table is public.** In a private company, you cannot see who owns the equity without an NDA. On-chain, the equivalent of the cap table is one click away for anyone, updated every block. The holder list is the closest thing crypto has to a shareholder register, and unlike a shareholder register, it never lies about the balances — it can only fail to *explain* them. Explaining them is your job.

### An address is not an owner — the two-way mismatch

Here is the trap that catches every beginner. The holder list shows *addresses*. You want to know about *owners*. Those are not the same thing, and the mismatch runs in **both directions**.

![Diagram showing one project team controlling several addresses and one exchange address custodying balances for many users](/imgs/blogs/supply-distribution-and-holder-concentration-2.png)

In one direction, **one owner can control many addresses.** A project team will routinely hold supply across a treasury address, a separate vesting address, and a marketing address — three rows on the holder list, one real owner. A whale who wants to hide their size will split their stack across ten fresh wallets — ten rows, one owner. So the holder list can *understate* concentration: what looks like ten independent 4% holders might be one entity holding 40%. (This is exactly the problem [labeling and attribution](/blog/trading/onchain/labeling-and-attribution) and cluster tools exist to solve, and we'll come back to it.)

In the *other* direction, **one address can hold funds for many owners.** A centralized exchange's hot wallet might be the single largest "holder" of a token, sitting at the top of the list with 12% of supply. But that address does not *belong* to the exchange in any economic sense — it custodies the balances of hundreds of thousands of individual users who deposited the token. So the holder list can also *overstate* concentration: that scary 12% in one address is actually the most *diffuse* holding of all, spread across a crowd. The mechanics of why one custodial address fronts so many users are covered in [addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — the core lesson there is that **an address is a public string, not a person**, and a wallet is rarely one address.

So the holder list is your starting data, never your conclusion. Every serious read begins by translating *addresses* into *owners* and *holdings* into *control*.

### Supply held by contracts versus people

The single most important distinction on a holder list is whether a given balance is held by a **person** (an externally-owned account, an EOA, that a private key can sign for and sell from at any moment) or by a **contract** (an address whose behavior is governed by code). The reason this matters is brutally simple: **a person can dump; a contract usually can't, or won't, or sells in a way that isn't a dump.**

Walk through the common contract-held buckets, because they make up most of the "scary" top of a typical holder list:

- **The burn address** (`0x000…dead` or `0x000…000`). Tokens sent here are gone forever — no key exists for these addresses, so the balance can never move again. A burn address holding 28% of supply means 28% of the supply *does not exist* for trading purposes. This is the most benign concentration there is; it's the *opposite* of a risk.
- **Liquidity-pool (LP) contracts.** On a decentralized exchange like Uniswap, the tradable liquidity sits inside a pool contract. The pool might "hold" 18% of the token's supply, but that supply isn't a holder waiting to sell — it *is* the market. When someone trades against it, the price moves along the curve; it's not a discretionary dump. (For how these pools work, see [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).)
- **Staking and vesting contracts.** Tokens locked in a staking contract or a vesting schedule are held by code that *won't release them until a condition is met* (an unstake period, a cliff date, a vesting curve). They can't dump today. They become a risk only at and after the unlock — which is exactly why tracking unlock schedules is a discipline of its own.
- **Treasury / DAO contracts.** Often a multisig that requires several signers to agree before moving funds. Slower and more accountable than a single key, though not immovable.

Against all of those, the dangerous bucket is the plain **EOA** — a single private key, one signature away from selling everything, with no lock, no code, no accountability. When concentration sits in EOAs, it is *free-float concentration*, and free-float concentration is the thing that actually craters prices.

### Circulating vs total vs fully-diluted supply

Every concentration percentage is a fraction, and a fraction is meaningless until you know the denominator. There are three:

- **Total supply**: every token the contract currently recognizes as existing — minus anything provably burned. This is the contract's headline number.
- **Circulating supply**: the tokens actually in the hands of holders who *can* sell — total supply minus burned, minus locked (team vesting, treasury locks, staking locks), minus tokens not yet released. This is the float that matters for price.
- **Fully-diluted supply (and fully-diluted valuation, FDV)**: the supply if *every* token that will ever exist — including all future emissions, all unvested team and investor allocations — were already in circulation. FDV = fully-diluted supply × price.

Why this matters for concentration: a top wallet holding "10% of supply" can mean wildly different things depending on which supply. If it's 10% of *circulating* supply and that wallet is a free-float EOA, it's a serious overhang. If it's 10% of *total* supply but most of total supply is locked, the same wallet might be 30% of the *float* — far worse than it first looked. And FDV is where the nastiest surprises live: a token with a tiny circulating float and a giant unvested allocation can look broadly held today and become whale-dominated the moment a cliff unlocks. Always check *which* denominator a "% of supply" number uses, and prefer reading concentration against the float that can actually move.

There's a second, subtler reason the denominator matters: the **low-float, high-FDV** structure is itself a design choice, and a controversial one. A team can launch with, say, 5% of total supply circulating and 95% locked in team, investor, and "ecosystem" allocations. The circulating float is tiny, so a modest amount of buying sends the price (and therefore the *fully-diluted* valuation) to eye-watering levels — a token can sport a \$5,000,000,000 FDV on a circulating cap of \$250,000,000, meaning the market is pricing in twenty times the supply that actually exists today. Every future unlock then sells into that inflated valuation. Reading concentration on the circulating float alone makes such a token look fine; reading it against fully-diluted supply reveals that today's "holders" own a sliver and tomorrow's unlocks own the rest. The holder list is a snapshot; the *vesting schedule* is the movie. Always read both.

### Dust, airdrops, and why holder count is a weak signal

One more foundational point, because it underlies a misconception we'll return to. A token's *holder count* — the total number of non-zero addresses — is one of the most-quoted and least-meaningful metrics. Projects advertise "50,000 holders!" as if it proves decentralization. It proves almost nothing, for two reasons.

First, **dust**. Anyone can send a trivial amount of a token to any number of addresses for the cost of gas. A project (or an attacker, or a marketing bot) can airdrop one-millionth of a token to 50,000 random wallets and instantly manufacture a 50,000-holder count, while the *supply* remains concentrated in a handful of real wallets. The holder list then has a giant tail of near-zero "holders" that own, collectively, a rounding error. They pad the count and the inequality metrics without changing who controls the float. Some of this dust is benign marketing; some is **address poisoning** — sending dust from a lookalike address to get a victim to copy-paste the wrong destination later. Either way, dust holders are not distribution.

Second, **one owner, many addresses** (the mismatch again). A holder count of 50,000 can collapse to a few hundred real entities once you cluster shared-funder and inter-wallet-transfer links. So holder count is inflated from *both* ends: padded with dust at the bottom, and overstated by wallet-splitting at the top. The metric that survives both critiques is the one this whole post is building toward — the *reclassified distribution of the free float*, not the raw count of addresses.

#### Worked example: the denominator changes the verdict

Take a token with a **total supply of 1,000,000,000** and a price of **\$0.10**, so total-supply market cap is **\$100,000,000**. Suppose 60% is locked (team vesting + treasury + staking + the burn address), leaving a **circulating supply of 400,000,000** worth **\$40,000,000** at the same price.

A wallet shows up holding **40,000,000 tokens = 4% of total supply.** Sounds modest. But against the circulating supply it's **40,000,000 / 400,000,000 = 10% of the float**, and at \$0.10 that's **\$4,000,000** of tokens that can hit the market *now*. If that wallet is a free-float EOA, you are not looking at a 4% holder — you are looking at a single seller who controls a tenth of everything that can trade, holding \$4M of overhang. **The same token count is a yawn against total supply and an alarm against the float; always anchor concentration to the supply that can actually move.**

## Reading the top holders: the five things you'll see

Open any token's *Holders* tab and the top 20 rows are almost always some mix of five archetypes. Learning to recognize them on sight is 80% of the skill. The reclassification below is the heart of the post — internalize it.

![Before and after reclassification of the top five wallets into a burn address, an LP pool, an exchange, a team vest, and one real whale](/imgs/blogs/supply-distribution-and-holder-concentration-3.png)

The figure shows the move in miniature. On the left, the raw list: five wallets holding 28%, 18%, 12%, 8%, and 4% — a terrifying **70% in five hands**. On the right, the *same five wallets, labeled*: a burn address (28%, destroyed), an LP pool (18%, that's the market), an exchange hot wallet (12%, thousands of users), a team vesting contract (8%, time-locked), and exactly *one* real whale EOA (4%, can dump now). The headline "70% in five wallets" was technically true and almost completely meaningless. The honest number — the free-float dump risk — is the single 4% whale.

Here is how to recognize each archetype:

- **The burn address.** Always `0x000…000` or `0x000…dead` (and on some chains a contract that provably destroys tokens). Explorers usually label it "Null: 0x000…000" or "Burn." Tokens here are dead. Treat as *negative* risk — it shrinks the effective float.
- **The LP pool / pair contract.** Etherscan tags these (e.g. "Uniswap V2: TOKEN-WETH"). The balance here is the tradable liquidity. The relevant question isn't "will it dump" but "how *deep* is it" — a thin pool means even the legitimate whale can crater the price. (More on this when we get to manipulation.)
- **An exchange (CEX) wallet.** Etherscan, Arkham, and Nansen label major exchange hot and cold wallets ("Binance 14," "Coinbase: Hot Wallet"). A large CEX balance means many users custody the token there. It's mostly benign — it only sells if its users sell — but a *sudden inflow* to an exchange wallet can foreshadow selling, which is a flow signal, not a concentration signal. (See [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) for how these wallets are structured.)
- **A team / treasury / vesting wallet.** Sometimes labeled, often not. Tell-tales: it received its allocation in a single large mint at or near token launch; it's a multisig or a vesting contract (check the *Contract* tab); it appears in the project's published tokenomics. Locked until a cliff — a *future* risk, not a present one. Mark the unlock date.
- **A genuine whale (free-float EOA).** No code on the *Contract* tab, funded from an exchange or another EOA, accumulated over time or in one buy, *not* locked. This is the one that matters. A whale EOA holding a meaningful slice of the *float* is the holder that can actually dump on you.

The discipline: **for every one of the top 20, answer one question — can this holder dump the free float right now?** Burn: no. LP: no (it's the market). CEX: only if users do. Vest/staking: not until unlock. Fresh EOA: yes, instantly. The sum of the "yes" balances is your real free-float concentration. Everything else is noise dressed up as a red flag.

#### Worked example: a \$50M token with one 30% whale

A token has a **circulating market cap of \$50,000,000.** You open the holder list and the top wallet — a plain EOA, no code, no lock, funded from a single exchange withdrawal six months ago — holds **30% of the circulating supply.** Thirty percent of \$50M is **\$15,000,000** of tokens that one private key can sell whenever its owner wants.

Now ask what happens if they do. Suppose the token's on-chain liquidity (the depth in the LP plus exchange order books) can absorb maybe **\$2,000,000** of selling before the price falls 20%. The whale's \$15M is **more than seven times** the liquidity that can absorb it. There is no version of that wallet exiting that doesn't crater the price. Even if they're disciplined and sell in tranches, every buyer in the token is, in effect, providing the whale's exit. **A single unlocked 30% holder turns the token into a bet on that one person's patience; the \$15M overhang is the whole thesis, and no amount of good "fundamentals" changes it.**

#### Worked example: reclassifying "5 wallets = 70%" into the real free-float risk

Same token, \$50M cap, but now the top five wallets hold **70% combined** and the headline screams "rug." Reclassify before you panic. The **burn address** holds 28% — that's **\$14M** of supply destroyed, gone, *reducing* effective float. An **LP pool** holds 18% — **\$9M** of liquidity that *is* the market, not a seller. A **Binance hot wallet** holds 12% — **\$6M** custodied for thousands of users, only sells if they do. A **team vesting contract** holds 8% — **\$4M** locked behind a cliff 9 months out. And a single **whale EOA** holds 4% — **\$2,000,000** that can actually move now.

So the honest free-float dump risk is not 70%. It's the one EOA: **\$2M against a \$50M cap, against (say) \$2M of absorbable liquidity** — a roughly 1× overhang, manageable, watchable, not a death sentence. The other 66% was a burn address, the market itself, a custodian, and a locked vest. **The reclassification turned a "70% rug" into a single 4% whale you can actually monitor; the number that mattered was never on the headline.**

## Concentration metrics: top-N %, Gini, and the Nakamoto coefficient

Once you can read individual holders, you want a *summary* — one or two numbers that capture how lopsided the whole distribution is. Three are worth knowing.

### Top-N concentration (the workhorse)

The simplest and most-used metric is **top-N concentration**: the share of supply held by the largest N addresses. Top-10 and top-100 are standard. You'll see it everywhere: "top-10 hold 45%," "top-100 hold 80%."

Used naively it's misleading for exactly the reason we've belabored — it counts burn addresses, LPs, and exchanges as if they were whales. Used *after reclassification* it's powerful: compute top-N over only the free-float EOA holders and you get the metric that actually predicts dump risk. As a very rough rule of thumb on *reclassified free-float* concentration: a healthy, broadly distributed token might have a top-10 free-float share in the single digits to low teens; a top-10 free-float share north of ~40–50% is a token whose price is hostage to a handful of sellers. These are heuristics, not laws — but the gap between "8%" and "75%" is the difference between two completely different assets.

### Gini coefficient (inequality in one number)

The **Gini coefficient** is borrowed from economics, where it measures income inequality. Applied to a holder list, it measures how unequally the supply is spread across holders, on a scale from **0** (perfect equality — every holder owns exactly the same amount) to **1** (perfect inequality — one holder owns everything). It's computed from the *Lorenz curve*: rank holders from smallest to largest, plot cumulative share of holders against cumulative share of supply, and Gini is (roughly) how far that curve bows away from the 45-degree line of perfect equality.

The honest caveat: **Gini on a raw holder list is almost always extremely high (often 0.9+) and almost meaningless**, because most tokens have a huge number of dust addresses with near-zero balances and a few large ones — that's mechanically very "unequal" but tells you little about *dump risk*. Gini is most useful as a *relative* gauge: comparing two similar tokens, or tracking one token over time (is it getting more or less concentrated?). Don't treat an absolute Gini number as a verdict.

To make the Lorenz-curve idea concrete: rank every holder from smallest to largest and walk along the x-axis from 0% to 100% of *holders*, plotting on the y-axis the cumulative *share of supply* they collectively own. In a perfectly equal token the curve is the straight 45-degree line — the bottom 40% of holders own 40% of supply. In a real token the curve sags far below that line: the bottom 90% of holders might own 5% of supply, and the curve only shoots up at the very right edge where the whales sit. The Gini coefficient is twice the area between the 45-degree line and that sagging curve. The dust problem shows up directly here: ten thousand near-zero "holders" stretch the bottom of the curve flat along the x-axis, inflating the area and pushing Gini toward 1 without telling you anything about whether the *float* can be dumped. That's why Gini is a comparison tool, not a verdict — it's exquisitely sensitive to the dust tail that doesn't matter and only crudely sensitive to the whale head that does.

### The Nakamoto coefficient (the "how many to collude" number)

The **Nakamoto coefficient** asks a sharper question: **what is the minimum number of holders you'd need to combine to control more than 50% of the supply?** It's named after Bitcoin's pseudonymous creator and originally applied to validator/mining decentralization, but it maps cleanly onto holder concentration. A Nakamoto coefficient of **1** means a single holder controls a majority — maximal concentration. A coefficient of **3** means it takes the three largest to clear 50%. Higher is more distributed.

It's intuitive and actionable: a token where it takes *2 wallets* to control half the float is a token where two people could collude (or panic together) and end it; a token where it takes *200 wallets* is genuinely distributed. Like top-N, compute it over the *reclassified free float* to get the meaningful version — a Nakamoto coefficient of 1 where that "1" is the burn address is fine; a Nakamoto coefficient of 1 where that "1" is a free-float EOA is a one-wallet time bomb.

A useful way to combine the three metrics is to read each as answering a different question about the same ledger. *Top-N* answers "how much do the biggest few hold?" — a blunt headline. *Gini* answers "how unequal is the whole curve?" — good for tracking one token over time but noisy because of dust. *Nakamoto* answers "how many would have to act together to control it?" — the most decision-relevant of the three, because collusion or correlated panic among a handful of holders is exactly the failure mode you're guarding against. When all three are computed on the *reclassified free float*, they tend to agree: a token with a top-10 free-float share of 8%, a moderate Gini, and a Nakamoto coefficient in the dozens is genuinely distributed; a token with a top-10 free-float share of 75%, a Gini near 1 driven by the *head* (not the dust tail), and a Nakamoto coefficient of 2 is a hostage situation. When they *disagree*, the disagreement is the signal — usually it means hidden clustering is fooling one metric but not another, and it's time to open Bubblemaps.

#### Worked example: comparing a fair launch and a whale token at the same cap

Two tokens, **both at a \$20,000,000 circulating cap.** Token F ("fair launch"): top-10 free-float holders control **8%**, no single wallet over 2%, the rest spread across ~40,000 holders. Token W ("whale"): top-10 free-float holders control **75%**, with one wallet at 30%.

![Same market cap comparison showing a whale-dominated token versus a broadly held fair launch](/imgs/blogs/supply-distribution-and-holder-concentration-5.png)

Now price the risk. In Token F, the largest free-float seller controls under 2% of the float — about **\$400,000** of tokens at the \$20M cap. Even a full exit by the biggest holder is absorbable; no single seller can break the price, and the Nakamoto coefficient (how many to reach 50%) is in the dozens. In Token W, the single 30% holder controls **\$6,000,000** of float, and the top *two* wallets together likely clear 50% — a **Nakamoto coefficient of 2.** One holder's exit is roughly 15× the size of Token F's largest, against the same total market and (often) thinner liquidity because fewer hands means a shallower book. **Identical market caps, opposite assets: Token F is a market, Token W is a hostage negotiation with one or two wallets — and the \$6M overhang is why the chart of a whale token can free-fall on a single transaction.**

## Supply bands: shrimp to whale, and the HODL read

Top-N and Gini look at the *largest* holders. The complementary view looks at the *whole distribution* by bucketing every holder into **wealth bands** by balance size. Glassnode popularized the playful zoological naming for Bitcoin — shrimp, crab, fish, dolphin, shark, whale, humpback — but the idea generalizes: group holders by how much they hold, and look at how supply is spread across the bands.

![Holder cohorts by balance from whales over one percent each down to thousands of shrimp under a hundredth of a percent each](/imgs/blogs/supply-distribution-and-holder-concentration-4.png)

A practical four-band version for a generic token:

- **Whales** — wallets holding more than ~1% of supply *each*. Few in number, large in impact. A whale band that holds a large share of the float, *especially if several of those whales cluster together* (more on that next), is the concentration that moves price.
- **Sharks / dolphins** — 0.1% to 1% each. Often funds, market makers, and early buyers. Big enough to matter in aggregate.
- **Fish / crabs** — 0.01% to 0.1% each. The mid-size conviction base — holders with real money in but not enough to move the market alone.
- **Shrimp** — under 0.01% each. Thousands or tens of thousands of tiny holders. A *large* shrimp band — lots of supply held by lots of small wallets — is the signature of genuine broad distribution.

The **HODL read** layers time on top of size. Glassnode's "HODL waves" and similar tools color supply by *how long it has been held* (age bands) on top of size. The combined read is powerful: supply that is both *small-balance* and *old* (long-term shrimp and fish that haven't moved in a year) is the most stable, conviction-held float in the token — it rarely sells into rallies. Supply that is *large-balance* and *young* (whales who just accumulated) is the most dangerous: fast money that can leave as quickly as it came. When you watch a token's bands shift — supply migrating from old small hands into young whale hands — you are watching distribution turn into concentration in real time, often before the price tops.

For UTXO chains like Bitcoin the band math is slightly different (you're aggregating across a holder's UTXOs rather than reading one account balance — see [how blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account)), but the read is the same: where is the supply, in whose hands, and how long has it sat there.

### Distribution is a verb: read the bands over time

The single static holder list is the snapshot; the *trend* in the bands is the movie, and the movie is where the edge lives. A token's distribution is never frozen — supply migrates between cohorts every day — and the *direction* of that migration is one of the most reliable tells of where a token is in its life cycle.

A **healthy, maturing** token shows supply *spreading out* over time: the whale band shrinks as early concentration distributes into a growing shrimp and fish base; the number of holders rises while no single cohort dominates; long-held supply accumulates. The whales who started with everything are gradually selling to a widening crowd of smaller, longer-term holders. That's distribution in the good sense — concentration *decreasing*.

A **topping or distributing** token shows the opposite, and it often shows it *before* the price rolls over: supply migrating *out* of long-term small hands and *into* fresh whale wallets, or — more ominously — top holders quietly moving tokens *toward* exchange deposit addresses. An inflow from a known whale wallet to a CEX deposit address is a classic pre-sell signal: tokens don't go *to* an exchange to be held, they go there to be sold. When you see the largest free-float holders sending to exchanges while the retail bands are still buying the narrative, you are watching the smart float hand its bags to the slow float in real time. That's the migration that precedes a top.

The practical habit: don't just snapshot the holder list once. Pull it again a week later (or chart the cohorts on a tool that tracks them) and ask *which way is the supply moving?* Concentration falling into more hands is bullish-structural; concentration consolidating into fewer, fresher, exchange-bound hands is the setup for the dump. The number on any single day matters less than the direction it's traveling.

## Why you must reclassify before you judge: benign vs rug concentration

We've circled this idea repeatedly because it is *the* idea. Now let's make it a rule and a picture. **The same concentration number can be perfectly benign or a rug in progress, depending entirely on what the concentrated addresses are.** Reclassifying them is not optional polish — it is the difference between a correct and a backwards conclusion.

![Decision matrix mapping each top-holder type to whether it can dump now and a benign or risk verdict](/imgs/blogs/supply-distribution-and-holder-concentration-7.png)

The matrix above is the triage you run on every top holder. Read it column by column: *what it is*, *can it dump the free float now*, *verdict*. A burn address: tokens destroyed, can't dump, **benign**. An LP pool: AMM liquidity, sells move price but aren't a discretionary dump, **benign**. A CEX wallet: many users custodied, only sells if they do, **mostly benign**. A team vest: a vesting contract, can't dump until unlock, **watch the unlock date**. A staking contract: time-locked, only after unstake, **mostly benign**. A fresh EOA: a person-controlled wallet, can dump instantly, **real risk**. A hidden cluster of EOAs with one owner: can dump all at once, **rug risk**.

State the rule plainly:

> **A chain showing 60% of supply "in 5 wallets" is FINE if those five are a burn address, three exchanges, and an LP. The identical 60% in five fresh person-controlled EOAs is a rug waiting to happen.** Same number, opposite verdict. The number means nothing until you've labeled the addresses.

This is why a dashboard's red "high concentration" warning is, on its own, almost worthless — it's counting addresses, not control. And it's why a green "low concentration" badge can be a lie — if a single entity has split a 45% stake across fifty wallets, the dashboard sees fifty small holders and the entity sees one big one. Tooling gives you the raw distribution; *you* supply the labels. The next section is about the labeling that's hardest to do by eye: when "separate" wallets are secretly one.

## Hidden concentration: when "separate" wallets are one entity

The most dangerous concentration is the kind that doesn't show up as concentration at all. A sophisticated actor — a team that wants to hide how much it kept, a manipulator setting up a pump, a memecoin deployer who pre-bought their own launch — will *split* a large position across many fresh wallets so the holder list shows a comfortable spread of mid-size holders instead of one alarming whale. The raw top-10 looks fine. The reality is a single owner controlling a controlling stake.

![Bubblemaps-style cluster showing four separate holder wallets funded by one source aggregating into one entity with combined supply](/imgs/blogs/supply-distribution-and-holder-concentration-6.png)

The defense is **clustering** — grouping addresses that are provably linked into one entity before you judge concentration. The links are on-chain and public:

- **Common funding source.** If holders #3, #5, #7, and #9 were all seeded from the *same* funding wallet shortly before they bought the token, that's a strong signal they're one operator. The figure above shows exactly this: four "separate" wallets, one funder, 45% of supply combined — invisible on the raw list, obvious once you trace the funding.
- **Transfers between the wallets.** Money moving *between* the supposedly-independent holders ties them together.
- **Identical behavior.** Wallets that bought in the same block, with the same gas settings, the same amounts, or that move in lockstep, are almost certainly one bot or one operator.
- **Approval / contract interaction fingerprints.** Shared, unusual contract interactions can fingerprint a common operator.

You *can* trace this by hand on an explorer, but the practical tool is **Bubblemaps**, which visualizes a token's holders as bubbles (sized by holding) and draws lines between wallets that have transacted with each other or share funding — so a cluster of secretly-connected wallets renders as a tight, interconnected blob instead of scattered dots. One glance tells you whether the "distribution" is real or a costume. (The general discipline of tying addresses to entities is [labeling and attribution](/blog/trading/onchain/labeling-and-attribution); clustering is its core technique.)

#### Worked example: a memecoin where 8 sniper wallets bought 25% at launch

A memecoin launches at **\$0.0001** and rockets to **\$0.01** — a **\$10,000,000 market cap** at the peak (on, say, a 1,000,000,000 supply). The holder list looks broadly held: dozens of wallets, none individually huge. But run the funding trace and **8 wallets, all seeded minutes before launch from two related funding addresses, bought a combined 25% of supply** in the first few blocks — classic *sniper* behavior, buying the launch before the public could.

Price that. Their 25% at the \$0.01 peak is **\$2,500,000** of tokens, bought for a few thousand dollars at \$0.0001. At launch their cost basis on 25% of a 1B supply — **250,000,000 tokens at \$0.0001** — was about **\$25,000.** Their unrealized gain at the peak is on the order of **\$2.475M**, a roughly **100× exit** sitting on top of every retail buyer who chased the candle. Because the 8 wallets are one cluster, the "distributed" holder list was a costume; the real top holder is a single operator sitting on a quarter of the supply and a 100× profit. **When eight fresh wallets share a funder and buy 25% in the first blocks, the token isn't broadly held — it's one sniper wearing eight masks, and \$2.5M of overhang is aimed at whoever's buying.** This sniper/bundle pattern is the whole subject of [holder analysis for memecoins](/blog/trading/onchain/holder-analysis-for-memecoins), where reclassification and clustering are life-or-death.

## Concentration as manipulation risk

Concentration isn't only a *dump* risk — it's a *manipulation* risk, and the two compound. When a few wallets control most of the free float and the on-chain liquidity is thin, those wallets can do more than sell into you. They can:

- **Pump and dump.** Buy aggressively to spike the price (cheap to do when the float is thin), wait for retail FOMO and momentum traders to pile in, then unload the pre-accumulated stack into that demand. The concentration is what makes the dump devastating; the thin float is what makes the pump cheap.
- **Paint the tape / wash trade.** Trade between their own wallets to manufacture volume and the *appearance* of interest, drawing in real buyers. (Wash trading has its own tells and is a topic of its own.)
- **Squeeze and trap.** With most of the float in friendly hands, a coordinated group can run the price against shorts or trap momentum buyers, knowing the float is too small for anyone to fight them.

The defender's read is to combine *concentration* with *liquidity depth*. A 30% whale against a deep, liquid market is dangerous; the same 30% whale against a \$50k liquidity pool is a guaranteed catastrophe — they can move the price double digits with a single transaction. This is why "low float" tokens are so prone to violent moves: it doesn't take much capital to swing a price when there isn't much float to push against. Concentration tells you *who can move it*; liquidity depth tells you *how easily*. Read them together.

### Where concentration has bitten — the patterns to recognize

You don't have to take the mechanism on faith; the history of crypto is a museum of concentration accidents. The general shapes recur:

- **The single-whale dump.** Countless low-cap tokens have free-fallen 80–95% in minutes when one pre-positioned wallet exited into thin liquidity. The chart looks like a cliff; the chain shows one address selling, often into the buyers chasing the prior pump. The information was on the holder list the whole time. This is the most common retail-killing pattern and the one the reclassification check is built to catch.
- **The unlock cliff.** A token that traded calmly for months can drop hard the week a large team or investor allocation vests, because a beautifully "distributed" float suddenly absorbs a slug of supply from a previously-locked contract. The distribution didn't change by accident — it changed on a *known, scheduled date* that was printed in the tokenomics. The investors who got hurt read the holder list and never read the vesting schedule.
- **The supply-side death spiral.** The 2022 collapse of Terra's UST and LUNA is the extreme case of supply mechanics overwhelming everything: as the peg broke, the protocol minted LUNA to defend it, and the *supply itself* hyperinflated from hundreds of millions to *trillions* of tokens in days, vaporizing every holder's share. That's not holder concentration in the top-10 sense, but it's the same lesson at the protocol level — **who and what can change the supply is the whole game.** (The full mechanism is in [the Terra/LUNA 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse).)
- **The exchange-listing pop and dump.** Concentration plus a fresh, hyped float is the classic setup for a listing-day spike and fade: insiders and snipers who hold most of the float let it run on listing demand, then distribute into it. The volume looks real; much of it is the concentrated holders handing tokens to the crowd.

None of these required private information. Each was visible — as a holder list, a vesting schedule, a cluster, or a supply curve — to anyone who looked. That's the whole promise of on-chain analysis: the cap table and the supply schedule are public, so the people who read them trade against the people who don't.

#### Worked example: a thin-float token where the whale moves price at will

A token has a **circulating cap of \$8,000,000** but only **\$120,000** of liquidity in its main pool (a common mismatch for low-float tokens). The top whale, a free-float EOA, holds **20% of the float = \$1,600,000.** That single holder controls more than **13×** the entire absorbable liquidity. A market sell of even a fraction of their stack — say \$100,000, under a tenth of what they hold — is *comparable to the whole pool* and would move the price violently. They don't need to dump everything to manipulate; they can walk the price up with small buys to bait FOMO, then exit into it. **When one wallet's stake dwarfs the liquidity by an order of magnitude, "the market" is a fiction — the whale *is* the market, and the \$120k pool is just the lever they pull.** A concentration read that ignores liquidity depth misses half the danger.

## How to read it: a walkthrough across three tools

Enough theory. Here's the concrete, repeatable pass you run on any token. It uses three free (or freemium) tools, each answering a different question. (For the full landscape of explorers and analytics, see [the on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape).)

### Step 1 — Etherscan *Holders* tab: the raw list and the reclassification

Paste the token contract address into Etherscan (or the chain's explorer — Solscan for Solana, etc.) and open the **Holders** tab. You'll see the ranked balance ledger: rank, address, quantity, percentage, and often a label.

Now do the work that matters — **reclassify the top 20**, one row at a time:

1. **Is it the burn address?** `0x000…000` / `0x000…dead`, usually labeled "Null." If yes, mentally subtract it from the float — it's destroyed supply, not a holder.
2. **Is it a contract?** Click the address; if it has a **Contract** tab with code, it's not a person. Is it an **LP pool** (labeled as a DEX pair), a **staking/vesting** contract, a **bridge**, or the **token's own contract**? Each is benign in a different way. An EOA has *no* Contract tab.
3. **Is it a labeled exchange?** Etherscan tags major CEX wallets ("Binance," "Coinbase," "OKX"). A large CEX balance is custodial — many users, mostly benign — but note it as a flow-watch point.
4. **Is it a team/treasury wallet?** Tell-tales: funded by the token contract in a single large mint at launch; appears in the project's tokenomics; is a multisig. Mark its unlock schedule if it's vesting.
5. **What's left is the real risk** — plain EOAs, no code, no lock, that *can sell now*. Sum their balances. **That sum, against the circulating supply, is your true free-float concentration.** Compute top-10 and (if you can) a Nakamoto coefficient over *just these*.

This five-step pass is the single most valuable habit in token diligence. It converts a meaningless headline ("70% in 5 wallets") into an actionable number ("4% real free-float whale, watchable").

![Illustrative top-ten holder distribution bar chart colored by holder type, labeled as not a specific token](/imgs/blogs/supply-distribution-and-holder-concentration-8.png)

The bar chart above is an **illustrative** top-10 (not a specific token) colored by the verdict you'd reach after reclassifying: the green burn slice and blue LP slice are the market and destroyed supply; the lavender exchange/vest/staking slices are custodial or locked; only the *red* EOAs are the free-float dump risk. Notice that the largest bars are the *most* benign — which is exactly why reading the holder list by size alone gets you the wrong answer.

### Step 2 — Bubblemaps: which "separate" wallets are one entity

The explorer shows balances; it doesn't easily show *links*. Open the token in **Bubblemaps** (it supports major EVM chains and Solana). It renders holders as bubbles sized by holding, with lines connecting wallets that have transferred to each other or share a funding source. What you're looking for:

- **Tight, interconnected clusters** — several "separate" top holders bound together by transfer/funding lines. That's hidden concentration: treat the whole cluster as one entity and re-sum its combined share. If four mid-size holders are actually one wallet with 45%, your top-10 number was a lie.
- **A clean, unconnected spread** — top holders that *don't* link to each other. That's the look of genuine distribution (subject to the usual caveat that an absence of *on-chain* links isn't proof of independence).
- **The deployer's footprint** — for a fresh token, does the deployer (the address that created the contract) still hold a big linked cluster? On a memecoin that's the first thing to check.

Bubblemaps turns the funding-trace logic from the last section into one picture. It's the fastest way to catch the costume.

### Step 3 — Nansen / Arkham: who the holders *are*

Etherscan labels the obvious entities; **Nansen** and **Arkham** label far more — funds, smart-money wallets, market makers, named individuals, and the long tail of exchange and protocol wallets — because they maintain large entity-attribution databases. Use them to:

- **Get a clean holder breakdown** with entities already labeled — "X% smart money, Y% exchanges, Z% public figures, W% unlabeled" — so the reclassification is half-done for you.
- **Check the quality of the holders**, not just the quantity. A float held partly by reputable funds and long-term smart money reads very differently from a float held by anonymous fresh wallets — even at the same concentration.
- **Watch flows**, not just balances — are top holders accumulating or distributing? A concentrated but *accumulating* holder base is bullish-ish; a concentrated and *distributing* one is the setup for a top.

The three tools form a pipeline: **Etherscan** gives you the raw ranked ledger and lets you reclassify by type; **Bubblemaps** collapses hidden clusters into real entities; **Nansen/Arkham** put names and behavior on the entities. Run all three and you've gone from "a list of addresses" to "here is who actually owns this token, how concentrated the real free float is, and whether it's accumulating or heading for the exit." That's the read.

## Common misconceptions

**"High concentration always means a scam."** No — *un-reclassified* high concentration is a question, not a verdict. A token can have a top-10 of 80% that is almost entirely burn address, LP, exchange, and locked team supply, with a tiny real free float — perfectly safe to hold. The number is a prompt to investigate, not a conclusion. Conversely, a "low concentration" badge can hide a single entity split across many wallets. **The verdict lives in the labels, not the percentage.**

**"The biggest holder is the biggest risk."** Usually false. On most tokens the biggest holders are the *most* benign — the burn address, the LP, an exchange custodying thousands of users. The biggest *risk* is the largest *free-float, person-controlled, unlocked* holder, which is frequently rank 4 or 8, not rank 1. Sort by *type-adjusted dump risk*, not by raw balance.

**"More holders = safer."** Not necessarily. Holder *count* is one of the easiest metrics to fake — airdrop dust to 50,000 addresses and your holder count looks great while one entity still controls the float. And clustering can collapse a "10,000-holder" token into a handful of real owners. Count is a weak signal; *reclassified distribution of the actual float* is the real one.

**"A locked team allocation is safe forever."** No — it's safe *until the unlock*, and then it's a scheduled supply shock. A token that looks beautifully distributed today can become whale-dominated the day a cliff vests and a giant team/investor allocation hits the float. Locked concentration is *deferred* concentration; mark every unlock date and treat the cliff as a known future risk. This is also why FDV matters: it's the concentration you'll *eventually* have.

**"If the dashboard says the contract is fine, the holders are fine."** Two different checks. A contract can be clean (no malicious mint function, no honeypot trap — that's the subject of [rug-pull and honeypot detection](/blog/trading/onchain/rug-pull-and-honeypot-detection)) and the *holder distribution* can still be a loaded gun. A safe contract held by one whale is still hostage to that whale. Always run both the contract check *and* the holder check.

## The playbook: what to do with it

Here is the if-then checklist for reading supply distribution and holder concentration, from raw list to action.

- **Signal**: the *Holders* tab shows a high top-10 percentage.
  - **Read**: reclassify the top 20 — burn / LP / CEX / vest / staking / EOA / cluster. Compute top-N and (if possible) the Nakamoto coefficient over *only the free-float EOAs*.
  - **Action**: if the concentration is almost entirely burn/LP/locked/custodial, it's benign — hold/buy on its merits. If a meaningful slice is free-float EOA, size it as a bet on those wallets' patience; demand a margin of safety or avoid.
  - **False positive / invalidation**: a scary headline number that's 90% burn + LP + locked team — don't avoid a good token over a number you didn't reclassify.

- **Signal**: a single free-float EOA holds a large share of the *float* (say >10–20%).
  - **Read**: price the overhang in dollars (share × cap) and compare it to absorbable liquidity. If the overhang dwarfs the liquidity, the token is hostage to that wallet.
  - **Action**: avoid, hedge, or size tiny. Watch that wallet's flows — an inflow to an exchange wallet from it is your exit signal.
  - **Invalidation**: the "whale" turns out to be a labeled exchange, the project's own vesting contract, or a market-maker wallet under agreement — not a discretionary seller.

- **Signal**: a comfortable-looking spread of mid-size holders (no obvious whale).
  - **Read**: run Bubblemaps and a funding trace. Are the "independent" holders one cluster (shared funder, inter-wallet transfers, same-block buys)?
  - **Action**: if they cluster into one entity with a controlling stake, treat it as hidden concentration — same caution as a single whale. If they're genuinely unconnected, that's real distribution.
  - **Invalidation**: shared *exchange* funding is not the same as a shared *private* funder — thousands of people withdraw from Binance; that's not a cluster.

- **Signal**: a fresh memecoin with a "distributed" launch.
  - **Read**: check the deployer's footprint and trace the first buyers. Did a cluster of fresh, same-funder wallets snipe a big share in the first blocks?
  - **Action**: if snipers/bundlers control a meaningful share, the float is a trap — the "community" is a costume. Avoid, or treat it as a pure momentum gamble with a hard stop.
  - **Invalidation**: a fair launch with broad first-block participation and no dominant funder cluster — rare, but it happens.

- **Signal**: a beautifully distributed token with a big *locked* allocation.
  - **Read**: pull the vesting/unlock schedule. When does the cliff hit, and how large is it relative to the float?
  - **Action**: mark the unlock date. The token can be safe today and a supply shock on the cliff — plan around it, don't get surprised.
  - **Invalidation**: the "locked" supply is actually burned or permanently locked (e.g., LP locked forever) — no future overhang.

Tie this together with the broader process in the [on-chain due-diligence checklist](/blog/trading/onchain/onchain-due-diligence-checklist) — holder concentration is one of the load-bearing checks, but it sits alongside contract safety, liquidity, and flow analysis. The discipline that runs through all of it is the one this post hammered: **the chain gives you the distribution; you supply the labels. Concentration is not risk until you've proven the concentrated addresses can actually dump the free float.** Read the ledger, reclassify the top, cluster the hidden, price the overhang in dollars — and the holder list will tell you, before the price does, exactly who you're trading against.

## Further reading & cross-links

- [Addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — why an address is not an owner, and why a wallet is rarely one address (the bedrock for reclassification).
- [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution) — the discipline of tying addresses to real entities and clustering linked wallets.
- [Holder analysis for memecoins](/blog/trading/onchain/holder-analysis-for-memecoins) — sniper, bundle, and deployer concentration where reclassification is life-or-death.
- [Rug-pull and honeypot detection](/blog/trading/onchain/rug-pull-and-honeypot-detection) — the contract-side check that pairs with the holder-side check.
- [On-chain due-diligence checklist](/blog/trading/onchain/onchain-due-diligence-checklist) — the full process; holder concentration is one load-bearing step.
- [Tokens, on-chain transfers, and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals) — how the balance ledger actually updates inside the contract.
- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — how LP pools hold and quote supply.
- [Centralized crypto exchanges: Binance, Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — why one exchange address custodies thousands of users.
