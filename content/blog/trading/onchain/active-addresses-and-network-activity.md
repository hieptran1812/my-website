---
title: "Active Addresses and Network Activity: Reading a Blockchain's Pulse"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The most basic on-chain health check is usage — how many addresses are active, how many transactions, how much fee revenue. Learn to read those signals as a network's demand pulse, and to spot when they are faked by sybil farms and bots."
tags: ["onchain", "crypto", "active-addresses", "network-activity", "nvt-ratio", "transaction-fees", "glassnode", "dune", "sybil", "blockchain-metrics", "tokenomics", "adoption"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The most basic on-chain health check is *usage*: how many addresses are active, how many transactions clear, and how much fee revenue the chain earns. Together these are the "is anyone actually here?" pulse that separates a real network from a ghost chain or an inflated token.
>
> - **The three signals:** active addresses (who is on the chain), transaction count (how much is happening), and fees paid (real money spent for blockspace). Read them as a single demand pulse, not three isolated charts.
> - **How to read it:** trend (growth / plateau / decline) and *divergence*. The loudest warning in all of on-chain analysis is **price up while active addresses stay flat** — a rally with no new users underneath it.
> - **What you DO:** treat fees as the purest, hardest-to-fake demand signal; use the NVT ratio (network value ÷ value transacted) as the on-chain price-to-earnings ratio; and always sanity-check counts for sybil and bot inflation before you trust them.
> - **The one number to remember:** a fee of **\$2,000,000 per day** annualizes to **\$730,000,000** of demonstrated demand for that chain's blockspace — money nobody pays to use a dead network.

On 21 February 2025, the largest theft in the history of money moved across a public ledger in plain sight — roughly **\$1.46 billion** of ETH drained from a Bybit cold wallet. The whole industry could watch it live, because the chain is public. But step back from the drama and notice something quieter that the same ledger shows every single day, for every chain, with no hack required: it shows you whether anyone is *using* the thing at all. How many distinct addresses did something today. How many transactions cleared. How much people paid, in real money, to get those transactions included. That mundane stream is the network's pulse, and reading it is the first skill of on-chain analysis.

It matters because crypto is unusually good at manufacturing the *appearance* of activity. A token can 10x in price on a wave of speculation while the chain underneath it is a graveyard — a handful of wallets passing the same coins back and forth. A new chain can boast "two million addresses" the week before its airdrop, then collapse to a few thousand the week after, because most of those addresses were throwaway wallets spun up by a single farmer to game the snapshot. Price and marketing lie cheaply. Usage is harder to fake — and where it *is* faked, the fakery leaves fingerprints you can learn to see.

This post builds the usage lens from zero. We will define what an "active address" even means, why an address is not a user, and why a transaction *count* and a transaction *value* are completely different animals. Then we go deep: active addresses as an adoption proxy (and the Metcalfe intuition that a network's value tracks the square of its users), how to read trends and the all-important divergences, why fees are the purest demand signal on the chain, the NVT ratio as the on-chain P/E, and — crucially — exactly how these numbers get inflated and how a careful analyst strips the fakery back out. Throughout, we point you at the real tools: Glassnode and CryptoQuant for the big chains, a Dune query for a specific token, and a humble block explorer's chart tab.

![Three usage signals — active addresses, transactions, and fees — feeding one read of whether a chain is real](/imgs/blogs/active-addresses-and-network-activity-1.png)

## Foundations: what "usage" actually means on a blockchain

Before any chart, you need a precise vocabulary. On-chain usage gets discussed loosely — "users", "activity", "volume" — and the looseness is exactly where people get fooled. Let us nail each term down with the care it deserves, because every higher-level signal in this post is built from these primitives.

### An address, and what makes it "active"

A blockchain is a public ledger of accounts. On Ethereum and most chains you will analyze, an **address** is a 42-character string starting `0x…` that can hold a balance and send or receive value. (On Bitcoin the unit is slightly different — coins live in unspent outputs rather than a running balance — but the idea of "an identifier that participates in transactions" carries over.) If you have used a wallet like MetaMask, your wallet *is* one or more addresses.

An address is **active** in a given window if it did something on-chain in that window — and the standard, widely-used definition is: **the address sent OR received at least one transaction** during the period. Note the "OR". If someone sends you a token, *your* address counts as active that day even if you did nothing yourself. This generosity matters: it means a single airdrop blasting tokens to a million addresses makes a million addresses "active" that day, whether or not a single human did anything.

The window is usually a day or a month, which gives the two headline metrics:

- **DAA — Daily Active Addresses.** The count of distinct addresses that transacted in a 24-hour period. This is the high-frequency pulse: it spikes on busy days, sags on quiet ones, and is the first thing a trader glances at to ask "is this chain alive today?"
- **MAU — Monthly Active Addresses** (sometimes written MAU by analogy to a Web2 app's "monthly active users"). The count of distinct addresses active across a rolling 30-day window. It is steadier than DAA and better for "is adoption growing over the quarter?" questions.

Two refinements you will see and should understand:

- **New vs. returning addresses.** Of today's active addresses, how many appeared on-chain for the *first time* (new), and how many have transacted before (returning)? A healthy, growing network shows a steady stream of new addresses *and* a high return rate — people come and stay. A pump-and-dump shows a flood of new addresses during the hype, then almost no returns. The split between new and returning is one of the most honest adoption signals there is.
- **Active vs. non-zero addresses.** A separate, slower metric counts every address holding a non-zero balance. That number almost never goes down (coins rarely get fully swept to zero), so it is a poor *activity* gauge — it measures accumulated holders, not current usage. Do not confuse "10 million addresses hold this token" (a holder count) with "50,000 addresses used it today" (DAA). They answer different questions.

### Transaction count, and why it is not transaction value

The second primitive is the **transaction count**: how many transactions the network processed in a window (per day, usually). This is the raw "how much is happening" number. A chain doing 1.2 million transactions a day is busier, in raw activity, than one doing 30,000.

Here is the trap that catches beginners, and it deserves a heading of its own:

> **Transaction *count* and transaction *value* are different things, and confusing them is the single most common usage-metric mistake.**

A **transaction count** is just a tally of how many transactions happened. A **transaction value** (or "on-chain volume") is the *amount* moved, denominated either in the native token (e.g. "320,000 ETH moved today") or — far more useful — in dollars ("\$1.1 billion of value settled today"). These can move in opposite directions and tell opposite stories:

- A chain can have a sky-high transaction *count* made almost entirely of tiny, near-worthless transfers — bots ping-ponging dust, or a game minting thousands of \$0.001 items. High count, trivial value.
- Another chain can have a modest transaction *count* but enormous value per transaction — a handful of \$10 million stablecoin settlements between institutions. Low count, huge value.

Neither is "better" in the abstract; they measure different kinds of usage. But you must always ask *which one a chart is showing you*, because "activity is up 300%!" means nothing until you know whether that is count or value, and in what units. When the units are tokens rather than dollars, a 300% rise in "value" might just be the token's *price* rising — the same coins moving, now worth more. Always prefer **USD-denominated value** when you can get it, because it strips out the price illusion.

### Fees: real money paid for a scarce good

The third primitive — and the one professionals trust most — is **fees**. Every transaction on a busy chain pays a fee to be included. On Ethereum this is "gas"; on Bitcoin it is the miner fee; every chain has some version. The fee exists because blockspace is *scarce*: each block can only hold so much, so when more people want in than there is room, they bid up the fee to get prioritized.

This makes fees the **purest demand signal on the entire chain**, for one simple reason: paying a fee costs real money, so people only do it when they genuinely want the transaction to happen. You can mint a million empty addresses for almost nothing. You cannot make a million people each pay a real \$3 fee without a million dollars of real demand showing up. Fee revenue is, in effect, the network's *revenue line* — what users collectively pay to consume the product (blockspace). We will lean on this hard later.

### An address is not a user — and a contract is not one either

The most important foundation, the one that invalidates half of all naive "user growth" claims, is this: **an address is not a user.** The mapping between humans and addresses is many-to-many and messy.

![One person controlling four addresses while one router contract absorbs millions of callers](/imgs/blogs/active-addresses-and-network-activity-2.png)

Look at the figure. One real person — call her Alice — routinely controls *several* addresses: a main wallet, a cold-storage wallet, a throwaway for risky approvals, maybe a few addresses she spun up to qualify for an airdrop. To the chain, that is four or five "active addresses". To reality, it is one user. So a raw address count **overstates** the human population — sometimes wildly, when farming is involved.

The mirror image is just as important. A single smart **contract** — say, the main Uniswap router on Ethereum — is *one* address, but it is touched by millions of different users over its life. If you counted contracts as "users" you would massively *understate* the population; if you counted every interaction with a contract as a separate "active user" you would massively overstate it. The honest read sits in the middle, and getting there requires judgment, not just a number off a dashboard.

This is why serious analysts almost never say "this chain has X users" and stop there. They say "X active addresses, of which roughly Y appear to be distinct funded entities after clustering, of which Z paid fees" — each filter stripping more of the illusion away. Keep the address-is-not-a-user fact burned into your mind; it is the antidote to every inflated headline in this post.

A useful way to hold all of this together is to treat the active-address number as the *top* of a funnel, with each layer below it closer to "real humans who genuinely use the chain". The top layer is the raw active-address count — generous, easy to inflate, the number a marketing deck quotes. Below it sits the count after collapsing one person's many wallets into single entities (address clustering, the subject of the dedicated post on addresses and wallets). Below *that* sits the count of entities that paid meaningful fees, not dust. And at the bottom sits the count of fee-paying entities that *came back* across multiple periods — the closest thing to a true, sticky user base. Every step down the funnel throws away inflation and keeps signal. When you read a usage claim, always ask which layer of the funnel it is quoting: a headline almost always quotes the wide, flattering top, and the honest number lives several layers down.

## Active addresses as an adoption proxy

With the vocabulary in place, start with the headline use of all this: active addresses as a stand-in for **adoption**. A blockchain, like a phone network or a social app, is more valuable the more people use it — and active addresses are the cheapest available proxy for "how many people".

### The Metcalfe intuition: value tracks users-squared

There is an old idea from telecoms called **Metcalfe's law**: the value of a network grows roughly with the *square* of the number of users, because value comes from *connections*, and the number of possible connections between N users is proportional to N². Two phones can make one connection; five phones can make ten; a hundred phones can make nearly five thousand. Each new user adds value not just for themselves but for everyone they can now reach.

Crypto researchers have repeatedly found that a blockchain's market value tracks something like the square of its active-address count over long horizons — not perfectly, not as a tradeable timing tool, but as a sanity anchor. The reasoning is exactly Metcalfe's: a payment or settlement network is worth more when more counterparties are reachable on it. You do not need to take the N² literally. The usable lesson is directional and powerful: **value should grow with usage, so when value races ahead of usage, something has to give** — either usage catches up, or the value falls back. That tension is what divergence analysis, below, is built to detect.

It is worth being clear-eyed about the limits of the Metcalfe analogy, because over-applying it is its own trap. The N² relationship is loose, it breaks down badly when address counts are polluted by sybils (squaring a faked user base squares the lie), and it says nothing about *which* chains win — only that, for a given chain, value and usage should travel together over time. Treat it as a relationship to watch for *breaks* in, not a price target to compute. The moment someone hands you a precise "fair value" derived from active addresses squared, be skeptical: the input is too noisy and too gameable to carry that much precision. The honest use is qualitative — usage and value should rhyme, and when they stop rhyming you have found something worth investigating.

#### Worked example: two chains, same valuation, very different pulses

Take two chains that both happen to carry a **\$1,000,000,000** market capitalization. Chain Alpha shows **500,000 daily active addresses**; Chain Beta shows **5,000**. On a per-active-address basis, the market is valuing Alpha at **\$1B ÷ 500,000 = \$2,000 of network value per active address**, and Beta at **\$1B ÷ 5,000 = \$200,000 per active address** — a hundredfold difference.

That gap does not by itself tell you which is the better buy, but it tells you exactly what question to ask of each. For Beta, you must believe one of two things: either those 5,000 addresses are extraordinarily high-value (huge institutional settlements, in which case go look at transaction *value* and fees to confirm), or the valuation is detached from usage and resting on narrative. For Alpha, the per-address value is so low that the risk is the opposite — is the chain *under*-monetizing all that activity, or are those 500,000 addresses mostly dust-sized bot transactions that are not really "users" at all? *The intuition: the same \$1B price tag means something completely different depending on how many real, active participants sit underneath it — and the usage metrics are how you find out.*

### Reading the trend: growth, plateau, decline

A single day's DAA is noise. The signal lives in the **trend** over weeks and months. Three broad shapes, each with a meaning:

- **Sustained growth** — DAA and MAU rising over months, with a healthy mix of new and returning addresses. This is the real thing: adoption compounding. It is the strongest fundamental tailwind a token can have, and it tends to *lead* price rather than follow it, which is the whole reason traders watch it.
- **Plateau** — activity flat for a long stretch. Not necessarily bad; many mature networks plateau at a high level (a utility, not a growth story). The question is *at what level* it plateaus, and whether fees and value are holding up alongside the flat count.
- **Decline** — DAA falling over months. This is the canary. A chain bleeding active addresses is losing its users to somewhere else, and price almost always follows usage down eventually. A persistent decline in active addresses while the token price holds up is one of the cleaner short-side or avoid signals in the toolkit.

The art is in reading these against price. Which brings us to the single most valuable pattern in this entire post.

### Divergence: the warning when price and users part ways

![Illustrative sketch of price rising sharply while active addresses stay flat, opening a divergence gap](/imgs/blogs/active-addresses-and-network-activity-4.png)

The figure above is an **illustrative sketch, not real data** — it shows the *shape* of the pattern, not any specific token. The red line is price; the blue line is active addresses. Early on they move together: more users, higher price, the Metcalfe relationship holding. Then they split. Price keeps climbing — doubling, tripling, more — while active addresses go flat. The widening gap between the two lines is the **divergence**, and it is the loudest fundamental warning the chain can give you.

Why is it a warning? Because a price rising on flat usage is, almost by definition, a price rising on *speculation* rather than adoption. New buyers are bidding the token up, but no new *users* are showing up to use it. The Metcalfe anchor says value should track usage; when value detaches upward, it is borrowing from the future, and the bill comes due if the users never arrive. Divergences do not tell you *when* the gap closes — speculation can run for a long time — but they tell you the rally is fragile and not supported by what is happening on-chain.

The mirror case — a **bullish divergence** — is just as useful and rarer: price flat or falling while active addresses quietly climb. That is accumulation of *real usage* the market has not yet priced, and historically it has been some of the best risk-reward in crypto. Smart on-chain analysts spend more time hunting bullish divergences (usage leading, price lagging) than admiring bearish ones, because the former is where the edge is.

Two practical cautions keep divergence analysis honest. First, **divergence is a context signal, not a trigger.** It tells you the relationship between price and usage has stretched, not that it will snap *now*. Speculative price runs can outlast a bearish divergence for months, and a bullish divergence can sit unrewarded for just as long before the market notices the usage. Use it to set your bias and your risk, not to time an entry to the day. Second, **verify the usage line is real before you trust the divergence.** A "bullish divergence" built on a surge of sybil addresses ahead of an airdrop is not bullish at all — it is the farm we will dissect shortly, and it will collapse the moment the reward is paid. The whole value of a divergence rests on the active-address line being honest, which is why the faking-and-detection section below is not a side topic but the other half of this skill.

There is also a subtler read that separates good analysts from great ones: **watch which direction the new-vs-returning split moves during a divergence.** A price rally accompanied by a flood of *new* addresses that never return is hollow — tourists who showed up for the pump and left. The same rally accompanied by rising *returning* addresses is far healthier, because it means people who came before are coming back to use the chain again. Two tokens can show identical headline DAA growth during a rally and have opposite futures, and the new-versus-returning composition is what tells them apart.

#### Worked example: a 10x price, a flat 800 users

A token launches and trades quietly for months at a **\$50,000,000** market cap with about **800 daily active addresses**. Then a narrative catches and the price runs **10x** to a **\$500,000,000** cap over a few weeks. You open the on-chain data and find that across that entire run, daily active addresses barely moved — still hovering around **800**.

Do the arithmetic on what the market is now paying per active user: it went from **\$50M ÷ 800 = \$62,500** of valuation per active address to **\$500M ÷ 800 = \$625,000** per active address. The *price* of the network rose 10x; the *usage* underneath it rose 0%. Every dollar of new market cap was bought, not earned. That is a textbook bearish divergence, and it is a red flag flashing as brightly as on-chain data ever flashes: the valuation is resting entirely on speculation, with no adoption to catch it if sentiment turns. *The intuition: when price multiplies and the active-address count does not budge, you are not looking at a growing network — you are looking at a more expensive version of the same small one.*

## Transaction count versus value, and what each tells you

We separated count from value in the foundations; now put both to work. They are two different lenses on "how much is happening", and a complete read uses them together.

**Transaction count** is your gauge of *broad activity and accessibility*. A high and rising count means lots of distinct actions — swaps, transfers, mints, game moves. It tends to track retail engagement and the liveliness of an ecosystem. But it is the *easiest* number to inflate, because each transaction can be arbitrarily small. A bot doing 100,000 one-cent transactions a day adds 100,000 to the count and almost nothing to the economy.

**Transaction value in USD** is your gauge of *economic weight*. It answers "how much money actually moved across this chain?" It is much harder to fake convincingly, because moving real value requires real capital — though as we will see, *wash trading* can inflate value too, by moving the same dollars back and forth. Stablecoin settlement value is an especially clean cut of this: when \$30 billion of USDT and USDC change hands on a chain in a day, that is heavy, real economic usage, and it is one of the metrics that best distinguishes a serious settlement network from a casino.

The most informative read combines them into **value per transaction** (USD value ÷ count). A chain with high count but tiny value-per-transaction is doing lots of small things — retail, gaming, spam. A chain with modest count but huge value-per-transaction is doing few large things — settlement, institutional flows. Neither is wrong, but they are *different businesses*, and the value-per-transaction number tells you which one you are analyzing.

#### Worked example: same "volume", opposite networks

Two chains each report **\$500,000,000** of on-chain value moved on the same day. Chain Retail did it across **2,000,000 transactions**; Chain Settlement did it across **5,000 transactions**. Divide:

- Chain Retail: \$500M ÷ 2,000,000 = **\$250 of value per transaction** — small, frequent, consumer-grade activity.
- Chain Settlement: \$500M ÷ 5,000 = **\$100,000 of value per transaction** — large, infrequent, institutional-grade activity.

Same headline "\$500M volume", two completely different networks. If you were valuing them you would underwrite them differently: Retail's thesis is breadth and engagement (and its risk is bot inflation of that count); Settlement's thesis is being trusted rails for big money (and its risk is concentration in a few counterparties). *The intuition: "volume" is a single number hiding two stories, and dividing it by the transaction count tells you which story you are actually buying.*

## Fees: the purest demand signal on the chain

![Bidders paying different fees compete for limited space in one block; the highest bids win inclusion](/imgs/blogs/active-addresses-and-network-activity-6.png)

We saved the best signal for its own section. **Fees are a blockspace auction**, and the figure shows the mechanism. Each block has a hard limit on how much it can hold. When more transactions want in than fit, users compete by raising the fee they are willing to pay, and the block includes the highest bidders first. The losers wait for the next block, or raise their bid. The fee you pay is, quite literally, the market-clearing price of getting your transaction included *right now*.

This is why fees are the demand signal you trust above all others. Walk the logic:

- **Fees are paid in real money.** Unlike an address count or a transaction count, a fee cannot be conjured for free. Every dollar of fee revenue is a dollar someone chose to spend to use the chain.
- **Fees reveal urgency and willingness to pay.** A spike in fees means demand for blockspace is outstripping supply — people *need* in badly enough to bid up. That is the cleanest possible read of "this chain is in demand right now".
- **Fee revenue is the network's revenue line.** Annualize the fees and you have something like the network's sales: what users collectively pay to consume the product. This is the basis for thinking of a chain as a *business* with a P/E-like valuation, which we get to in the NVT section.

Fees **spike around events** — a hot NFT mint, a market crash triggering a wave of liquidations and panic transfers, a popular token launch. Those spikes are demand made visible: when everyone wants the chain at once, the fee auction heats up and the price of blockspace soars. Reading fee history against the calendar of events is a fast way to see what actually drove people to *pay* to use a chain, as opposed to what the marketing claimed drove adoption.

The deepest reason to love fees: **you cannot sybil a fee.** You can spin up fifty thousand addresses to fake adoption (we will see exactly how, next), but if those addresses pay no fees — or only the minimum dust fee — the fee revenue stays flat and exposes the fakery. A surge in active addresses *with* a flat fee line is a giant tell that the surge is manufactured. Real users pay to play.

#### Worked example: fees as the demand you cannot fake

A chain earns **\$2,000,000 per day** in total transaction fees. Annualize it: \$2,000,000 × 365 = **\$730,000,000 per year** of fee revenue — three-quarters of a billion dollars that users pay, in real money, simply to consume this chain's blockspace.

Now contrast a different chain boasting "**1,200,000 daily active addresses**" but earning only **\$8,000 per day** in fees — about **\$0.0067 of fee per active address**. Two-thirds of a cent. Real users transacting for real reasons pay far more than that; a fee that low across a million-plus addresses screams that the "activity" is near-free dust, bots, or subsidized spam rather than economic demand. The first chain's \$730M annual fee line is demand you cannot fabricate; the second chain's near-zero fees, despite a huge address count, is the fabrication exposing itself. *The intuition: address counts are cheap to inflate, but fees are paid in real money — so when the addresses balloon and the fees do not, the fees are telling you the truth.*

## NVT: the on-chain price-to-earnings ratio

If fees are the revenue line, you naturally want a *valuation* ratio — a way to ask "is this network expensive or cheap relative to the economic activity it processes?" That ratio is **NVT: Network-Value-to-Transactions.**

![High NVT versus low NVT compared as overvalued and cheap relative to on-chain activity](/imgs/blogs/active-addresses-and-network-activity-3.png)

NVT divides the network's market value by the value it transacts:

> **NVT = network value (market cap) ÷ daily value transacted on-chain.**

It is the on-chain analog of a stock's **price-to-earnings ratio**. In equities, P/E asks "how many dollars of price am I paying per dollar of earnings?" — a high P/E means the market is paying up, pricing in growth; a low P/E means it is cheap relative to current profits. NVT asks the same thing with *on-chain settlement* standing in for earnings: how many dollars of network value per dollar of daily economic activity? A **high NVT** says the market is paying a lot for each dollar of activity — richly valued, priced for growth, vulnerable if that growth does not materialize. A **low NVT** says the network is cheap relative to the real economic throughput it handles.

As the figure lays out, two chains can carry the *same* \$5 billion market cap and sit at wildly different NVTs depending on the activity beneath them. One moving \$50 million a day sits at NVT 100 — the market pays \$100 for every \$1 of daily activity, a growth-priced number that demands the growth show up. One moving \$500 million a day sits at NVT 10 — far cheaper per dollar of real settlement.

Use NVT the way you use P/E: as a *relative* and *historical* gauge, never an absolute oracle. Compare a chain's NVT to its own history (is it richer or cheaper than its usual range?) and to peer chains (is it expensive relative to similar networks?). An NVT far above its historical band is a classic "the price has outrun the usage" warning — the same divergence idea, expressed as a single ratio. And the same caveat applies as for P/E with manipulated earnings: **NVT is only as honest as the transaction-value figure in its denominator.** If that value is wash-traded, NVT will look deceptively cheap. Always pair an NVT read with a wash-trading sanity check.

#### Worked example: NVT as a cheap-vs-rich gauge

A chain has a **\$5,000,000,000** market cap and settles **\$50,000,000** of value a day. NVT = \$5,000,000,000 ÷ \$50,000,000 = **100**. The market is paying \$100 for every \$1 of daily on-chain activity — a richly-priced network, the on-chain equivalent of a 100x P/E stock that must grow into its valuation.

A second chain, same **\$5,000,000,000** cap, settles **\$500,000,000** a day. NVT = \$5,000,000,000 ÷ \$500,000,000 = **10**. Ten dollars of value per dollar of daily activity — ten times cheaper than the first, by this measure. If both chains are otherwise comparable, the first is priced for a future it has not yet delivered and the second is grounded in present-day throughput. But before you conclude the second is a bargain, you must verify that its \$500M of daily value is *real* settlement and not the same dollars wash-traded in a loop to flatter the denominator. *The intuition: NVT turns "expensive or cheap?" into a single number — but it only works if the activity in the denominator is genuine, so always check the value is not faked before you trust the ratio.*

## Comparing chains: high-fee versus high-throughput

A usage signal only means something *in context*, and the most important context is what kind of chain you are looking at. The same fee number, the same transaction count, the same active-address figure means radically different things on Ethereum than on a chain built for raw throughput like Solana. Comparing chains naively — "this one does ten times the transactions, so it's ten times healthier" — is one of the fastest ways to draw the wrong conclusion. Two archetypes anchor the spectrum.

A **high-fee settlement chain** like Ethereum is, by design, a place where blockspace is genuinely scarce and therefore expensive. Each block holds relatively little, so when demand rises the fee auction bites hard and fees can spike to dollars or even tens of dollars per transaction. The consequence: Ethereum's transaction *count* is modest by global standards, but its fee *revenue* and its value-per-transaction are enormous. People pay up because what they are settling — large DeFi positions, big stablecoin transfers, valuable NFTs — is worth the fee. On a chain like this, **fees and value are the headline signals**, and a low transaction count is not a weakness; it reflects high-value usage on premium-priced blockspace. Reading Ethereum's health through raw transaction count alone would badly understate it.

A **high-throughput chain** like Solana is built for the opposite tradeoff: enormous capacity, very low fees per transaction (fractions of a cent), and therefore a transaction *count* that can dwarf Ethereum's by an order of magnitude. That high count is real activity — memecoin trading, high-frequency bot strategies, consumer apps that would be unaffordable on expensive blockspace. But it also means the count is far more *dilutable* by spam and bots, precisely because each transaction is nearly free. On a chain like this, raw count is the *least* trustworthy signal and you lean harder on **value-weighted activity and fee-paying unique addresses** to find the real economic usage hiding inside the firehose.

The practical rule: **never compare a usage metric across chains without normalizing for the chain's cost structure.** A million transactions a day means something very different at \$0.0002 each than at \$3 each. Compare fees-to-value ratios, value-per-transaction, and fee-paying address counts — metrics that are robust to the underlying cost model — rather than raw counts. A chain is not "more used" because it prints a bigger transaction number; it is more used when more real economic value, paid for with real fees, flows across it.

#### Worked example: same fee revenue, opposite chains

Chain Premium (a high-fee settlement chain) earns **\$3,000,000** of daily fees from just **400,000 transactions** — that is **\$7.50 of fee per transaction**, the signature of expensive blockspace carrying valuable settlements. Chain Volume (a high-throughput chain) earns the same **\$3,000,000** of daily fees, but from **3,000,000,000 transactions** — **\$0.001 of fee per transaction**, the signature of near-free blockspace at massive scale.

Both chains earn identical \$3M-a-day, or **\$1,095,000,000 annualized**, of demonstrated demand — so by fee revenue they are equally "used" in dollar terms. But their *nature* is opposite: Premium's usage is a few hundred thousand high-stakes settlements; Volume's is billions of tiny consumer and bot actions. If you ranked them by transaction count, Volume looks 7,500 times bigger; if you ranked them by fee-per-transaction, Premium looks 7,500 times more valuable per action. Neither ranking is "true" — they are different businesses earning the same revenue two completely different ways. *The intuition: fee revenue lets you compare chains on equal footing, but the fee-per-transaction tells you whether you are looking at a vault for big money or a bazaar for small money — and you must know which before you value it.*

## How usage gets faked — and how to catch it

Everything above assumes the numbers are honest. They often are not. Because address counts and transaction counts are cheap to inflate, crypto has a thriving cottage industry in *manufacturing the appearance of usage*. As analysts and defenders, our job is to recognize the fakery and strip it out. This is defensive knowledge — how to *detect* inflation, not how to perform it.

![A single funder splitting money into thousands of sybil wallets to inflate the active-address count before an airdrop](/imgs/blogs/active-addresses-and-network-activity-5.png)

### Sybil farming: many addresses, one operator

The most common inflation is the **sybil attack** — named for a famous case of multiple-personality disorder, it means one operator masquerading as many independent participants. The figure shows the pattern. A single funder bankrolls thousands (sometimes hundreds of thousands) of fresh addresses, has each do the minimum activity needed to look like a "user", and so balloons the active-address count. The motive is usually an **airdrop**: many new chains and tokens reward early "users" with free tokens, allocating per-address, so a farmer who controls 50,000 addresses can capture 50,000 allocations.

This is why a chain's active-address count so often *spikes* right before a rumored or announced airdrop snapshot and *collapses* right after — the farmers showed up for the reward and left. A pre-airdrop address surge is one of the least trustworthy numbers in all of crypto, and you should mentally discount it hard.

The defender's job is to tell the 50,000 sybils from the few hundred real users hiding among them. The fingerprints, shown as the "defender filter" in the figure:

- **Common funding source.** Sybil addresses are usually funded from one wallet (or a short chain of them) — the farmer has to get gas into all those addresses, and the money trail clusters them. One funder bankrolling thousands of "independent users" is the loudest tell.
- **Identical, robotic behavior.** Real users are messy and varied; sybils run a script, so they do the same actions in the same order at the same intervals. Cookie-cutter behavior across thousands of addresses is a script, not a crowd.
- **No fees, or only dust.** Sybils minimize cost, so they pay the bare minimum and move dust-sized amounts. Filtering for addresses that paid *meaningful* fees and moved *meaningful* value collapses the count toward the real number.
- **Freshness and disposal.** Sybils are born just before the snapshot and go dormant right after. A cohort of addresses that all appeared the same week and all died the same week is a farm, not an organic user base.

#### Worked example: a 50,000-address farm before a \$0.10 airdrop

A new chain announces an airdrop and the dashboards light up: daily active addresses jump from **800** to **50,800**. A farmer is behind most of it. Say they spend about **\$5,000** in total gas to fund and operate **50,000** sybil addresses — roughly **\$0.10 of cost per address** — each doing the minimum to qualify. If the airdrop pays out even **\$0.10 of tokens per qualifying address**, the farmer's 50,000 addresses harvest **50,000 × \$0.10 = \$5,000** of tokens, exactly recouping the gas — and if the airdrop is worth \$1 per address, that same \$5,000 of effort harvests **\$50,000**. The economics *demand* the farming.

Now run the defender filter. Trace the funding: nearly all 50,000 addresses trace back to one funder. Check fees: the sybils paid only dust. Check behavior: identical action sequences, all born the same week. Strip them out and the real active-address count is back near **800** — the same 800 the chain had before the hype. The "50,800 users" headline was a 64x illusion. *The intuition: an airdrop turns address-creation into a paying job, so the count explodes with sybils — but a single funding source and zero real fees expose the farm and snap the true number back to reality.*

### Bot and wash inflation of transaction counts and value

The same logic extends to the other two signals. **Bots inflate transaction counts** by running endless tiny transactions — arbitrage bots, spam, automated dust transfers — that pad the count without adding economic substance. The tell is the same as for sybils on the value side: enormous transaction count paired with trivial value-per-transaction and near-zero fee contribution per transaction.

**Wash trading inflates transaction value.** Here the same dollars cycle back and forth between addresses the operator controls, so the "value transacted" looks huge while no real economic transfer occurs — the money ends up back where it started. This is the one that poisons NVT: a wash-traded chain shows a fat denominator (lots of "value") and so a deceptively *low*, cheap-looking NVT. The defender's checks: does the value flow in closed loops back to its origin? Is it concentrated among a few addresses trading with each other? Does fee revenue scale with the claimed value, or stay suspiciously flat? Real economic value drags real fees along with it; washed value does not.

### The sanity-check toolkit

Pulling the defense together, here is the standard set of cross-checks an analyst runs before trusting any usage number:

- **Fee-paying unique addresses**, not raw active addresses. The subset that paid a real fee is far harder to fake and far closer to the real user count.
- **Value-weighted activity**, not raw counts. Weight each address or transaction by the real USD value it moved; dust-sized sybil and bot activity falls away, and the heavy, real usage remains.
- **Funding-source clustering.** Trace where the addresses got their gas. A thousand "users" funded by one wallet are one user wearing a thousand masks.
- **New-vs-returning and cohort survival.** Do addresses come back after the event, or vanish? Organic adoption persists; farmed activity evaporates the moment the reward is paid.
- **Cross-check fees against counts.** The single fastest tell: if address or transaction counts surge but fee revenue does not, the surge is fake. Real demand pays.

## How to read it: a walkthrough with real tools

Theory is worthless without the tools. Here is exactly where to read each signal, and how to read a DAA chart against price like an analyst. (For the full map of the tooling landscape, the dedicated post in this series goes tool by tool; this is the usage-metrics slice of it.)

![A comparison table of Glassnode, Dune, and a block explorer, mapping each to the usage metric it best answers](/imgs/blogs/active-addresses-and-network-activity-7.png)

The figure above is your routing table; the prose below walks through using each.

### Glassnode / CryptoQuant — the big chains, ready-made

For Bitcoin and Ethereum, you almost never need to write a query — **Glassnode** and **CryptoQuant** have already computed the standard usage metrics into clean, charted series. Open Glassnode, search "active addresses", pick Bitcoin or Ethereum, and you get the DAA time series back years, often with new-vs-returning splits, alongside fee and MVRV charts. CryptoQuant is similar with a heavier exchange-flow focus.

How to *read* a DAA chart there, step by step:

1. **Set a long window** — at least a full cycle, two to four years. Usage trends only mean something over months; a two-week view is noise.
2. **Overlay price.** Both tools let you plot price on the same chart. This is where divergence jumps out: drag your eye along both lines and look for the gap opening — price climbing while DAA stays flat (bearish) or DAA climbing while price lags (bullish).
3. **Check the new-vs-returning split** if available. Rising DAA driven by *returning* addresses is stickier and healthier than a one-off flood of *new* addresses that never come back.
4. **Glance at the fee chart alongside.** If DAA and fees rise together, the activity is real and people are paying for it. If DAA rises but fees are flat, be suspicious.

The limit: these tools cover the major chains and gate their best metrics behind paid tiers. For a specific token or a newer chain, you go to Dune.

### Dune — any custom metric, if you can write the query

**Dune** lets you write SQL against decoded blockchain tables and chart the result. It is how you get the daily active addresses for a *specific token or contract* that no pre-built dashboard covers. The shape of a daily-active-addresses query is straightforward — count the distinct addresses that interacted with the token's contract per day:

```sql
select
  date_trunc('day', evt_block_time) as day,
  count(distinct address) as daily_active_addresses
from (
  select evt_block_time, "from" as address from erc20_transfers
  where contract_address = 0xYOUR_TOKEN_HERE
  union all
  select evt_block_time, "to" as address from erc20_transfers
  where contract_address = 0xYOUR_TOKEN_HERE
) t
group by 1
order by 1
```

The `union all` of `from` and `to` encodes the "sent OR received" definition of active from our foundations; the `count(distinct address)` de-duplicates within the day. Thousands of public Dune dashboards already do this for popular tokens, so you can often fork an existing query rather than write your own. Dune is also where you run the *defender* checks — group activity by funding source, weight by value moved, filter for fee-payers — because you control the query.

The limit: a wrong query gives a confident wrong number. SQL will happily count wash trades and sybils unless you explicitly filter them; the tool is only as honest as the analyst writing it.

### A block explorer — the free, instant sanity check

Every major chain has a **block explorer** (Etherscan for Ethereum, Solscan for Solana, and so on), and most have a **charts tab** with free daily-transaction and active-address history plus a fee chart. It is the fastest sanity check in existence: before you trust any third-party "this chain is booming" claim, open the explorer's charts tab and look at the raw daily-transaction and active-address lines yourself.

The limit, and it is the big one: a block explorer does **no de-duplication and no sybil filtering**. It counts every address and every transaction at face value — bots, sybils, and dust included. It tells you the *raw* numbers honestly, but it will not tell you how many of those addresses are real users. That judgment is on you, armed with the foundations from this post.

### Putting the walkthrough together

A complete usage read on a token chains these together: glance at the explorer's charts tab for the raw shape, pull the DAA-vs-price overlay from Glassnode (for a major chain) or a Dune query (for a specific token), check the new-vs-returning split and the fee line for honesty, and run the sybil/wash sanity checks on Dune before you trust any surge. Each tool covers the previous one's blind spot.

### Usage rides the broader liquidity cycle

One macro point grounds all of this in real, dated data. On-chain usage does not move in isolation — it rises and falls with the entire crypto liquidity cycle. The clearest curated proxy we have for that cycle is **DeFi total value locked (TVL)**, and its arc maps the same boom-bust that active addresses and fees traced over the same years.

![DeFi total value locked rising to a 2021 peak, crashing post-FTX, and recovering, as a proxy for the usage cycle](/imgs/blogs/active-addresses-and-network-activity-8.png)

TVL climbed from roughly **\$16 billion at the end of 2020** to a peak near **\$180 billion in November 2021**, collapsed to a trough around **\$39 billion** after the FTX implosion in late 2022, and recovered toward **\$135 billion by mid-2025** (DefiLlama, approximate). Active addresses, transaction value, and fee revenue across the major chains rode broadly the same wave — surging into the 2021 top, draining through the 2022 bust, rebuilding since. The lesson for usage analysis: always read a chain's active-address trend *against the cycle*. A chain whose DAA fell in 2022 was not necessarily failing — the whole market's usage fell. The chains to find are the ones whose usage *outgrew* the cycle on the way up or *held* better than the cycle on the way down. Relative usage, not absolute, is where the signal lives. (This curated TVL series is adjacent context; no curated active-address time series exists in this series' dataset, so the usage charts here are conceptual by necessity.)

## Common misconceptions

**"More addresses always means more adoption."** No — an address is not a user, and address counts are the single easiest metric to inflate. One person controls many addresses; one farmer controls fifty thousand. A rising address count is *evidence* of adoption only after you have stripped out sybils, bots, and a single user's many wallets. Raw counts overstate the human population, often by a lot.

**"High transaction count proves the chain is healthy."** Count is the cheapest signal to fake, because each transaction can be arbitrarily small. A chain doing a million transactions a day of one-cent bot dust is *less* economically alive than one doing fifty thousand transactions of real, fee-paying value. Always pair count with value-per-transaction and fee revenue before reading health into it.

**"On-chain volume is real economic activity."** Sometimes — but volume (transaction value) can be wash-traded, the same dollars cycling between an operator's own addresses to inflate the number. Washed value flows in closed loops, concentrates among a few addresses, and crucially does *not* drag proportional fee revenue along with it. Real value pays real fees; check that the fees scale with the volume.

**"A low NVT means a chain is cheap."** Only if the denominator is honest. NVT is network value divided by transaction value, so wash-trading the value inflates the denominator and makes a chain look cheap by fabrication. NVT is a genuinely useful on-chain P/E, but it inherits every distortion in its activity figure — never read it without first sanity-checking that the activity is real.

**"Fees are just a tax on users, the lower the better."** From a user's wallet, sure. But as an *analyst*, fees are the most valuable signal on the chain — the one thing nobody pays for a dead network. Healthy, rising fee revenue alongside rising usage is the strongest confirmation that the activity is real and economically motivated. A chain with lots of "activity" but no fees is showing you fake activity.

## The playbook: what to do with it

Translate everything above into an if-then checklist. Each line is a signal, the read, the action, and the way it can fool you.

- **Signal: active addresses and fees rising together over months, with healthy new-and-returning mix.** Read: genuine adoption, the strongest fundamental tailwind there is, and it tends to *lead* price. Action: a high-conviction long-bias input — accumulate or hold. Invalidation: confirm the rise is *not* funded by one wallet ahead of an airdrop; a pre-snapshot surge with flat fees is farming, not adoption.

- **Signal: price rising while active addresses stay flat (bearish divergence).** Read: the rally is speculation, not adoption — value detaching from usage. Action: tighten risk, take profits, or avoid chasing; the move is fragile. Invalidation: usage can still catch up — watch for active addresses and fees turning up to *confirm* the price, which would repair the divergence.

- **Signal: price flat or falling while active addresses quietly climb (bullish divergence).** Read: real usage building that the market has not yet priced — often the best risk-reward in crypto. Action: a candidate accumulation zone; size in while the usage trend holds. Invalidation: make sure the address growth is real (fee-paying, varied, not sybil) and not pre-airdrop farming dressed up as organic growth.

- **Signal: NVT far above the chain's own historical range.** Read: the network is richly valued relative to the activity it processes — the price has outrun the usage. Action: caution on the long side; a candidate for the avoid list. Invalidation: a genuine surge in real (fee-backed) transaction value would justify the higher valuation and bring NVT back to earth honestly.

- **Signal: address or transaction count surging but fee revenue flat or near-zero.** Read: the surge is manufactured — sybils, bots, or subsidized dust — because real demand pays real fees. Action: discount the headline entirely; do not let "X million users!" move your thesis. Invalidation: rising fees alongside the count would mean the activity is real after all — so always check the fee line before dismissing *or* trusting a surge.

- **Signal: an airdrop is rumored or announced and active addresses spike.** Read: assume most of the spike is farming until proven otherwise — pre-snapshot surges are the least trustworthy numbers in crypto. Action: wait for the post-airdrop cohort-survival data before crediting any "adoption"; see whether the farmers stayed or left. Invalidation: a high return rate after the snapshot — addresses that keep transacting and paying fees once the reward is gone — would mean some of the growth was real.

The thread through every line: **usage is the pulse, fees are the lie detector, and divergence is the warning.** Address counts tell you how loud the room is; fees tell you whether anyone is actually paying to be there; and the gap between price and usage tells you whether the market's enthusiasm is earned or borrowed. Read all three together, sanity-check them for inflation, and you can tell a living network from a beautifully-decorated ghost.

## Further reading & cross-links

- [Addresses, Wallets, and Contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — the deep dive on why an address is not a user, how one person controls many wallets, and how contracts differ from externally-owned accounts. The foundation under this whole post.
- [The On-Chain Tooling Landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — the full map of Etherscan, Glassnode, Dune, Nansen, and the rest, and where each tool's data comes from. Use it to route any usage question to the right tool.
- [Anatomy of a Transaction](/blog/trading/onchain/anatomy-of-a-transaction) — what is actually inside a transaction, including the gas and fee fields that make the blockspace-auction signal possible.
- [Ethereum and Programmable Money](/blog/trading/crypto/ethereum-and-programmable-money) — how smart contracts work, why one router address absorbs millions of interactions, and why gas fees exist in the first place.

Within this series, the natural next steps are the dedicated treatments of how usage gets faked and how fee revenue rolls up into a chain's fundamentals — the airdrop-farming-and-sybil-cohorts post (the full sybil-detection playbook) and the on-chain-fundamentals post (fees, revenue, and TVL as a valuation stack). Read alongside this one, they turn the usage pulse from a number you glance at into a signal you can actually trade and defend against.
