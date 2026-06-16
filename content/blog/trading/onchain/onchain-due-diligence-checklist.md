---
title: "On-Chain Due Diligence: A Repeatable Pre-Buy Checklist"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Before buying any token, run the same seven on-chain gates every time — contract safety, liquidity, distribution, unlocks, usage, smart money, and a hard GO/NO-GO — so you catch most of the ways you lose money."
tags: ["onchain", "crypto", "due-diligence", "token-analysis", "rug-pull", "liquidity", "holder-concentration", "risk-management", "etherscan", "dexscreener", "nansen"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — On-chain due diligence is a fixed seven-gate checklist you run on every token *before* you buy, because the contract code, the liquidity pool, the holder list, and the team's wallets are all public — and most ways you lose money show up there first.
>
> - The seven gates: **contract safety → liquidity → distribution → unlocks/emissions → usage/fundamentals → money flow → a GO/NO-GO synthesis**. Each maps to a free tool (Token Sniffer, DEXScreener, Etherscan, Bubblemaps, unlock calendars, DefiLlama, Nansen/Arkham).
> - Separate two questions: **"is it a scam?"** (safety) and **"is it a good investment?"** (quality). A token must pass both — a clean contract with no users still bleeds to zero.
> - Some checks are **hard fails** (honeypot, owner can mint, unlocked team-held LP, one wallet over ~30% of supply) that auto-reject at any price. Others are **soft flags** that don't reject — they shrink your position size.
> - The number to remember: on launchpad chains roughly **1.4% of tokens ever reach a meaningful market cap** — about 98.6% effectively go to \$0. A checklist that makes "no" cheap and automatic is the single highest-return habit in crypto.

On 2025-02-21, the largest theft in the history of money cleared in a single afternoon. Attackers tied by the FBI to North Korea's Lazarus Group drained roughly **\$1.46 billion** from a Bybit cold wallet by tricking signers into approving a malicious contract upgrade. Within hours the funds were fanned across hundreds of fresh addresses, swapped, and bridged — the laundering machine running before most of the world had read the headline. That was an exchange hack, not a token rug, but it makes the series' core point at maximum volume: **the chain is public, so everything that matters was visible in real time.** Analysts watched the outflow live. The defenders who lost were not blind; they had simply trusted a signing screen instead of verifying what the transaction actually did.

Now shrink that lesson down to your own buys. You will never sign a \$1.46B transfer. But every week you will be tempted by a token chart that is going up, a Telegram link, a "100× gem" thread. The same public ledger that let analysts trace the Bybit funds lets *you* read a token's plumbing in minutes: whether the contract can mint new supply into your face, whether the liquidity you'd sell into is locked or can vanish, whether one anonymous wallet owns 42% of the float, whether anyone real actually uses the thing. The information asymmetry that crushes retail in traditional markets — where the cap table, the insider sales, and the real revenue are private — is *inverted* on-chain. The data is right there. Most people lose money not because they couldn't check, but because they didn't have a repeatable way to check, so they skipped it when the chart was green.

This post is that repeatable way. It is the **hub of Track G** of the series: one workflow that synthesizes contract safety, liquidity, holder distribution, team and insider wallets, unlocks and emissions, real usage, and smart-money presence into a single pre-buy checklist with a hard GO/NO-GO at the end. The other Track G posts go deep on each gate; this one shows you how they fit together into a habit you can run in ten minutes, every time, without emotion.

![Seven-section on-chain due diligence checklist feeding a GO or NO-GO decision](/imgs/blogs/onchain-due-diligence-checklist-1.png)

## Foundations: what due diligence is and why on-chain DD is uniquely powerful

**Due diligence (DD)** is the investigation you do before committing money — the homework. The term comes from securities law: a broker who exercised "due diligence" before selling you a security had a legal defense if it later blew up. The everyday meaning is simpler. Before you buy a used car, you check the title, look for accident history, listen to the engine, and take it to a mechanic. You are not trying to *guarantee* the car is great; you are trying to *avoid the obvious ways you get cheated* — a salvage title, a rolled-back odometer, a flooded engine. DD is loss-avoidance first, opportunity-finding second.

Crypto DD is the same idea, but the "car" is a token and the "title and history" are public on a blockchain. Here is why that matters so much.

**In traditional markets, the data you most need is private.** Who owns a private company? You don't know. Are insiders dumping? You find out months later in a regulatory filing, if ever. Is the revenue real? You trust an audited statement you can't independently verify. The whole edifice of equity research exists because the facts are hidden and you must pay analysts to estimate them.

**On-chain, those exact facts are public, permanent, and queryable by anyone.** A token's entire holder list — every address and its balance — is on the ledger. The liquidity pool you would buy from and sell into is a public contract with a visible balance. The team's wallets, once identified, broadcast every sale the moment it happens. The token contract's *code* is often verified and readable. Vesting schedules are encoded in contracts. Protocol revenue is the sum of on-chain fee events. This is the deepest edge in the series: **the cap table, the liquidity, and the team's behavior are all public.** You are not guessing at hidden facts; you are reading them.

A few terms to fix before we go further, defined from zero:

- **Token vs. coin.** A *coin* is the native asset of a chain (BTC on Bitcoin, ETH on Ethereum). A *token* is an asset created *by a contract* on top of a chain — most things you'll do DD on are ERC-20 tokens on Ethereum/EVM chains or SPL tokens on Solana. Tokens are the ones that rug, because anyone can deploy one in minutes.
- **Smart contract.** A program living at an address on-chain that runs exactly as written. A token *is* a contract: the code defines who can mint, who can pause transfers, what the sell tax is. If you can read the code, you can read the rules of the game.
- **EOA (externally owned account).** A normal wallet controlled by a private key — a person or a bot. Contrast with a *contract account*, which runs code. The token's "owner" is usually an EOA that holds special powers over the contract.
- **Liquidity pool (LP) / AMM.** On a decentralized exchange (DEX) like Uniswap, you don't trade against another person; you trade against a pool that holds two assets (say the token and ETH). The pool's size is the liquidity. *LP tokens* represent ownership of that pool — and who holds them, and whether they're locked, decides whether the pool can be yanked out from under you.
- **Market cap (MC) vs. fully diluted valuation (FDV).** MC = circulating supply × price. FDV = *total* supply (including locked and not-yet-emitted tokens) × price. The gap between them is future selling pressure you haven't felt yet.
- **TVL (total value locked).** The dollar value of assets deposited in a protocol — a rough proxy for how much the protocol is actually used. We'll see it can be faked.

The most important conceptual split — and the reason a checklist needs more than one section — is this: **"is it a scam?" is a different question from "is it a good investment?"** Safety asks whether the plumbing can rob you (mint, freeze, honeypot, pull liquidity). Quality asks whether, assuming nobody robs you, the thing is worth owning (real demand, real revenue, sane valuation). DD must cover *both*, because they fail independently.

![Two-by-two matrix of safety versus quality with only the safe and high-quality cell as a buy](/imgs/blogs/onchain-due-diligence-checklist-2.png)

The matrix above is the mental model the whole checklist serves. A token sits in one of four quadrants. The honeypot with a beautiful narrative is *high quality, unsafe* — a deadly trap, because the story pulls you in and the plumbing keeps your money. The legitimate-but-pointless token is *safe, low quality* — no scam, but no demand either, so it bleeds slowly to near zero. The classic memecoin is *unsafe and low quality* — avoid on sight. Only the **safe AND high-quality** quadrant is a buy candidate, and even then you size to conviction. A checklist that only checks safety leaves you buying clean garbage; one that only checks quality leaves you buying a great story wrapped around a mint function.

Last foundational point: **what a checklist actually buys you is consistency.** The danger in DD is not that you don't know the checks — it's that you skip them when you're excited. The token is pumping, the group chat is euphoric, you feel the fear of missing out, and you tell yourself "I'll just check the basics." A *written, ordered* checklist removes the discretion. You run all seven gates, in order, every time, or you don't buy. The emotional skip — the single most expensive habit in crypto — becomes impossible, because the answer to "should I skip the holder check this once?" is "the checklist doesn't have a skip button."

## The checklist, gate by gate

We'll walk each of the seven gates in order: what it checks, how to read it with a named tool, the hard fail that auto-rejects, and the soft flags that size you down. Throughout, treat the chain as the source of truth and the dashboards as a *convenience* — green numbers on a website are claims to verify, not facts to trust.

The order is deliberate, and it's the order of *decisiveness and cost*. Gate 1 (contract safety) goes first because it's the cheapest check and the most fatal failure: a honeypot or a live mint takes you to \$0 instantly, so there's no point analyzing the fundamentals of a token you can never sell. Liquidity and distribution come next because they're the next-most-decisive hard fails and still fast to read. The quality gates — usage, fundamentals, and money flow — come later because they're slower to assess and only matter *if the token already cleared the ways it could rob you*. You run the gates in this sequence so that the moment any hard fail trips, you stop and save the analytical effort for a token that deserves it. A scammer's best weapon is your time and your hope; the ordering denies them both.

One discipline to repeat until it's reflexive: **a dashboard's number is a claim, the chain is the proof.** Every tool below — DEXScreener, Nansen, DefiLlama, even the explorer's own summary tiles — is an interpretation layer over raw on-chain data, and interpretation layers have bugs, stale caches, and gameable inputs. A "locked LP" badge can point at a locker contract that doesn't actually hold the tokens. A "1,000 holders" tile can count dust airdrops. A "Smart Money" label can be assigned to a wallet that already cashed out. When a gate's verdict is close, drop down to the raw chain — the contract source, the actual pool balance, the literal holder list, the real transfers — and read the fact yourself. The whole edge of on-chain DD is that you *can* do this; the discipline is actually doing it instead of trusting the tile.

### Gate 1 — Contract safety: can the code rob you?

This is the first gate because it's the one that takes you to **\$0 instantly** rather than slowly. A token contract is just code, and the code can contain functions that let whoever controls it steal from holders. The four things to check:

1. **Is the contract verified?** "Verified" means the deployer published the human-readable source code and Etherscan confirmed it compiles to the bytecode actually running on-chain. Unverified means you're trusting raw machine code you can't read. *Unverified contract on a token asking for your money = a flag bordering on a hard fail.* There's rarely a good reason for a legitimate token to hide its source.
2. **Is the mint function renounced (or absent)?** If the owner can call `mint()` and create new tokens at will, they can dilute you to nothing or dump infinite supply. "Renounced" means ownership was transferred to a dead address (`0x000…dead`), so no one can call privileged functions anymore. *Mint not renounced and an active owner = hard fail* unless there is a credible, contract-enforced reason (a real protocol with governance, not a memecoin).
3. **What owner powers exist?** Beyond mint, look for `pause()` (the owner can freeze all transfers — including yours), `blacklist()` (the owner can stop *your specific address* from selling), and a settable transfer tax (the owner can raise the sell tax to 100% after you buy). Any of these in the hands of an anonymous EOA is a loaded gun.
4. **Is it a honeypot?** A honeypot is a contract where *buys succeed but sells fail* — the trap closes after you're in. The sell function reverts, or the sell tax is secretly 100%, so your tokens are real but unsellable. This is the purest hard fail: a token you cannot sell is worth \$0 to you no matter what the chart says.

The tool: **GoPlus Security** and **Token Sniffer** run automated contract scans and flag honeypots, mint authority, high taxes, and ownership in seconds; **Etherscan** (or the chain's explorer) shows you the verified source and the contract's read/write functions directly. Always cross-check — automated scanners have false negatives, and reading the actual `_transfer` logic on Etherscan is the ground truth. For a full treatment of honeypots and rug mechanics, see [rug-pull-and-honeypot-detection](/blog/trading/onchain/rug-pull-and-honeypot-detection).

**How this gate deceives you.** The sophisticated version of a malicious contract is built to pass the scanners. A *delayed honeypot* lets the first wave of buyers sell freely — so the scanner's test transaction succeeds and the tool reports "not a honeypot" — then the owner flips a switch (raises the sell tax, enables a blacklist, toggles a transfer flag) once enough liquidity has piled in. A *proxy contract* shows a clean, renounced implementation while the real logic lives behind an upgradeable proxy whose admin can swap the code at any time; the "renounced owner" you saw was the proxy's puppet, not the upgrade key. And a *hidden mint* can be disguised as an innocuous-looking function or buried in a library call so it doesn't appear as a plain `mint()`. The defense is not to trust the one-line verdict: read the actual transfer logic, confirm the contract is *not* a proxy (or that the proxy admin is also renounced/timelocked), and treat a contract that's verified-but-complicated as worth a slower read than a simple one. Most retail buyers never open Etherscan at all — doing so is itself a meaningful edge.

#### Worked example: a token that passes safety but fails distribution

Say you scan a new token called HYPE. GoPlus comes back clean: verified contract, mint renounced, no blacklist, sell tax 3%, not a honeypot. Gate 1 passes — the plumbing won't directly rob you. But on Gate 3 you pull the holder list on Etherscan and find that one non-LP wallet, funded three days ago from a fresh address, holds **40% of the supply**. At a \$20M market cap, that wallet's stack is worth \$8M. The liquidity pool is only \$1.5M deep. If that single holder decides to exit, they are trying to push \$8M of selling through a \$1.5M pool — the price would collapse 80%+ before they're a fifth of the way out, and you're the exit liquidity. A clean contract does not protect you from a clean *dump*. The math says one wallet can erase your position, so this is a **NO-GO** despite a perfect safety scan. The lesson: safety and distribution are independent gates, and passing one tells you nothing about the other.

### Gate 2 — Liquidity: can you actually get out, and can the pool vanish?

Liquidity is the question "if I buy this, can I sell it again without destroying the price — and is the pool I'd sell into even going to *be there*?" Two sub-checks:

**Depth and price impact.** A DEX pool with \$2M of liquidity can absorb a \$5,000 sell with almost no price move. A pool with \$40,000 of liquidity cannot — a \$5,000 sell is 12.5% of the pool and will move the price violently. *Price impact* is how much your own trade moves the price; thin pools have brutal impact in both directions, which is why thin-liquidity tokens can pump 10× on tiny buys (and crash just as fast on tiny sells). Always check the pool depth and simulate your intended buy *and* exit size.

**Is the LP locked or burned?** This is the safety half of liquidity. When a team launches a token, they typically seed the pool with the token plus ETH/USDC, and receive LP tokens representing ownership of that pool. If the team *holds* those LP tokens, they can call `removeLiquidity()` and withdraw the entire pool — the original "rug pull." Legitimate launches either **lock** the LP tokens in a time-locked contract (Unicrypt, Team Finance) for months or years, or **burn** them (send to a dead address) so no one can ever pull the pool. *Unlocked, team-held LP = hard fail.* The tool: **DEXScreener** and **DexTools** show pool depth, the LP lock status, and price-impact estimates; for the deepest read of how pools and depth work, see [reading a DEX liquidity pool](/blog/trading/onchain/the-onchain-tooling-landscape) coverage in the tooling post and the smart-money flow posts.

**How this gate deceives you.** Liquidity is the most theatrical of the gates because depth can be staged. A team can pair the token with a *worthless* second token instead of ETH/USDC, so the "\$2M pool" is \$2M of a coin nobody will buy — your real exit liquidity is near zero. A "locked" LP can be locked for an absurdly short window (a 1-week lock that expires the day after launch) or locked in a contract the team secretly controls. And depth can be *rented*: a market maker provides liquidity that's pulled the moment the launch incentive ends, so the pool that looked deep on day one is hollow by day ten. The defense is to confirm *what* the token is paired against (it should be a real asset), *how long* the lock runs (months, not days) and *where* (a reputable locker whose contract you can read), and to re-check depth over time rather than trusting a single snapshot. Liquidity isn't a number you read once — it's a condition you confirm is durable.

#### Worked example: pricing the exit through a thin pool

You hold \$10,000 of a token whose DEX pool has \$50,000 of liquidity (\$25,000 token, \$25,000 ETH in a constant-product pool). You decide to sell the whole \$10,000. In a constant-product AMM, selling tokens worth 40% of the pool's token side does not give you 40% of the ETH — it gives you far less, because each token you sell gets a worse price than the last. A rough constant-product calculation says your \$10,000 nominal sell nets closer to \$7,000 after the price impact, and the act of selling craters the quoted price ~30% for the next person. Now compare a \$2M pool: that same \$10,000 sell is 0.5% of the pool and nets ~\$9,950 with negligible impact. The token didn't change — the *liquidity* changed what your position is actually worth on exit. This is why "market cap" on a thin pool is a fantasy number: you can never realize it. Always price your exit, not just your entry.

### Gate 3 — Distribution: who owns the supply, and is it really many people?

A token's value depends on its holders being many and independent. If ten wallets — or worse, ten wallets secretly controlled by one person — hold most of the supply, the "market" is a puppet show and you are the audience that gets dumped on. Three checks:

**Top-holder concentration.** Pull the holder list and look at the top 10 (excluding the LP pool, burn address, and known locked-team contracts, which aren't free-floating sell pressure). If the top non-contract wallet holds 30%+, or the top 10 together hold the vast majority of the float, that's dangerous concentration. The reference threshold most analysts use: any single non-LP wallet above ~30% of supply is a *hard fail*.

**Team and insider percentage.** How much did the team and early investors keep? A token where insiders hold 60% of supply is a token where your gains are at the mercy of their selling. This overlaps with Gate 4 (unlocks) — the team's tokens are usually vested, so you care both about *how much* and *when it unlocks*.

**Fresh-wallet clusters.** This is the on-chain superpower. A scam often disguises concentration by splitting one entity's holdings across many wallets so the holder list *looks* distributed. But those wallets were all funded from the same source, often within minutes of each other, often fresh addresses with no other history. **Bubblemaps** visualizes the holder graph as bubbles connected by funding links — a cluster of "different" holders that all trace back to one funder lights up instantly. Ten wallets each holding 4% looks fine until Bubblemaps shows they're one octopus. For the full method, see [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) and the [early-buyer and insider detection](/blog/trading/onchain/early-buyer-and-insider-detection) post.

**How this gate deceives you.** The distribution that *looks* healthiest is the one most carefully engineered. A patient deployer funds dozens of wallets days or weeks ahead through intermediary hops — a hop or two of laundering between the funder and the holding wallets so the direct funding link Bubblemaps draws is broken. They'll also use a few centralized-exchange withdrawals to "wash" the trail, since CEX hot wallets fund millions of addresses and break naive clustering. The counter is to look past a single funding hop: cluster on *behavior* too — wallets that bought in the same block, hold near-identical amounts, never interact with anything else, and all woke up to sell on the same candle are one entity even if their funding trail is obfuscated. And always exclude the legitimate big holders (the LP pool, the burn address, a verifiable locked-team vesting contract) before you judge concentration — counting the pool as a "whale" produces false alarms. Real distribution is messy and idiosyncratic; suspiciously *uniform* holdings across fresh wallets is itself the signal.

#### Worked example: a clean token that earns a GO

Now a token that passes. You scan token CALM: verified contract, mint renounced, no owner powers, not a honeypot (Gate 1 ✓). DEXScreener shows the LP is **locked for 12 months** and the pool is **\$2M deep** — a \$5,000 buy or sell barely moves the price (Gate 2 ✓). The holder list shows the top non-LP wallet at 4.8%, the team's allocation sitting in a public vesting contract that unlocks linearly, and **800+ holders** with no fresh-wallet co-funding cluster on Bubblemaps (Gate 3 ✓). Gate 6 shows three wallets that Nansen labels "Smart Money" have accumulated quietly over the past week. With \$2M of locked liquidity, a renounced contract, 800 real holders, and smart money present, this token clears the hard fails and carries no soft flags. That is a **GO** — at full planned size. Notice how different this *feels* from the HYPE example even though both passed Gate 1: the difference is entirely in the distribution and liquidity facts, which are exactly what the public chain hands you for free.

### Gate 4 — Unlocks and emissions: what selling is coming that you can't see yet?

Price reflects *current* float. But most tokens have a large supply that is locked and scheduled to unlock later — team tokens, investor tokens, ecosystem reserves. When those unlock, they become sellable, and the people holding them are often sitting on huge unrealized gains they'll want to harvest. An unlock is a scheduled wave of supply hitting the market.

**Vesting cliffs.** A *cliff* is a date when a big chunk unlocks all at once (e.g., "12-month cliff, then 25% unlocks instantly"). A cliff into a thin market is a price event you can predict to the day. **Token unlock calendars** (TokenUnlocks, CryptoRank) show every upcoming unlock and its size relative to circulating supply.

**FDV vs. MC.** If a token has a \$50M market cap but a \$500M fully diluted valuation, then 90% of the eventual supply is not yet circulating. You are buying into 10× future dilution. A huge FDV/MC gap means the chart you see is propped up by artificially low float, and every unlock dilutes you. *A large unlock cliff landing into thin liquidity is a soft flag that can become a hard fail* if the unlock is enormous relative to the float and the team is known to sell.

**Emission inflation.** Some tokens (especially DeFi "farm" tokens) mint new supply continuously as rewards. If a protocol emits 2% of supply per month to liquidity miners who immediately sell, the token has built-in, perpetual sell pressure. Check the emission schedule — high, sell-pressure emissions are a quality flag even when the contract is "safe." For the deep mechanics, see the tooling and fundamentals coverage in [the on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape).

**How this gate deceives you.** The trick here is *low float, high FDV* — a launch where only a tiny slice of supply circulates, so a small amount of buying produces an enormous quoted market cap and a vertical chart, while 80–95% of the eventual supply waits in locked allocations. The chart looks like price discovery; it's actually a thin float being marked up before the dilution arrives. Newer tokens have learned to make their unlock schedules look gentle (linear vesting, no scary cliffs) while still releasing a multiple of the current float over a year. The defense is to read the FDV/MC ratio as *future supply you'll have to absorb*, not as a vanity number, and to project the dollar value of the next 6–12 months of unlocks against the *current* liquidity depth. A schedule with no single cliff can still bury a token if the steady drip is large relative to the pool that has to absorb it. Read the whole release curve, not just the next date.

#### Worked example: an unlock cliff priced as supply pressure

A token trades at \$2.00 with a \$40M market cap (20M circulating). The unlock calendar shows a cliff in 18 days: **10M tokens** (an amount equal to 50% of the current float) unlock to early investors who bought at \$0.10. Those investors are sitting on a 20× gain (\$0.10 → \$2.00); at \$2.00 their newly unlocked stack is worth \$20M, against a DEX pool only \$3M deep. Even if just a third of them sell, that's ~\$6.7M of selling into \$3M of liquidity over the following weeks — the price cannot hold. You don't need to predict the exact bottom; you only need to recognize that a \$20M unlock at a 20× profit landing into a \$3M pool is overwhelming supply pressure. Either you don't buy before the unlock, or you treat the unlock date as a hard exit. This is a signal you literally could not see in a traditional market — the lock schedule is in a public contract.

### Gate 5 — Usage and fundamentals: does anyone actually use this?

This is the quality gate — the difference between a real protocol and a ticker with a logo. The question: stripped of hype, does this thing generate real economic activity?

**Real users.** Active addresses interacting with the protocol, not just holding the token. A DeFi protocol with 50 daily active users and a \$2B FDV is a valuation built on nothing. See [active addresses and network activity](/blog/trading/onchain/active-addresses-and-network-activity) for how to read this honestly (and how it's gamed).

**Fees and revenue.** The cleanest fundamental in crypto: how much do users *pay* to use the protocol, and how much of that accrues to the token or treasury? A DEX that earns \$1M/month in fees is doing real business; a token with \$0 in protocol fees is pure speculation. **DefiLlama** and **Token Terminal** publish protocol fees and revenue directly from on-chain events.

**TVL quality.** TVL is the headline DeFi metric, but it lies easily. The question is not "how high is TVL" but "*why* is it there." TVL that's parked to farm an unsustainable emission (mercenary capital) leaves the instant rewards stop — it's rented, not owned. TVL that's there because the protocol is genuinely useful (a lending market people actually borrow from) is sticky. Check whether TVL is concentrated in one incentivized pool, whether it survived the last time emissions were cut, and whether it's the same dollars round-tripping. On-chain fundamentals — fees, revenue, and sticky TVL — are covered in depth across the tooling and activity posts in this series.

#### Worked example: TVL quality versus a mercenary farm

Protocol A and Protocol B both show \$100M TVL. Protocol A earns \$400,000/month in fees from genuine borrowing demand and pays no token emissions; its TVL barely moved when the broader market sold off. Protocol B earns \$15,000/month in fees but emits \$1.2M/month worth of its own token as farming rewards — meaning depositors are paid \$1.2M to generate \$15,000 of real activity. Protocol B is *paying* \$80 in inflation for every \$1 of real revenue. The day emissions are cut, that \$100M of TVL evaporates because it was renting yield, not using the protocol. Same headline number, opposite reality. If you valued both at the same multiple of TVL you'd be massively overpaying for B. Always divide the headline by the *reason* it exists, and weight real fee revenue far above incentivized deposits.

### Gate 6 — Money: who is buying, and where is it going?

The final analytical gate flips from defense to offense. The first five gates ask "can I lose?"; this one asks "who else is in, and are they the kind of money I want to be next to?"

**Smart-money holders.** Tools like **Nansen** and **Arkham** label wallets with track records — funds, profitable traders, early buyers of past winners. If wallets labeled "Smart Money" are accumulating a token *before* it's trending, that's a real (if noisy) signal. But beware survivorship bias: "smart money" labels are assigned to wallets that already won, and a label is not a guarantee. See [what is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain) and [following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) for how to use these labels without becoming exit liquidity for them.

**Insider accumulation vs. distribution.** Are the team and early wallets *adding* or *selling*? Team wallets that quietly accumulate alongside you are aligned; team wallets that sell into every pump are not. Because the chain is public, you can watch the team's wallets directly — the moment they move tokens to an exchange, you see it.

**Exchange flows.** Tokens moving *to* a centralized exchange are usually heading toward a sale (you deposit to sell); tokens moving *off* an exchange into self-custody usually signal holding. A spike of token inflows to exchanges from team or whale wallets is a distribution warning. See [exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) for the full read. And critically — *don't blindly copy* the wallets you find; the [perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) post covers why a smart wallet's buy is not your buy (they have a different cost basis, size, and exit plan).

### Gate 7 — The GO/NO-GO synthesis: hard fails vs. soft flags

Now you collapse six gates of findings into one decision. The synthesis has two tiers, and conflating them is how people both miss good tokens and buy bad ones.

![Four hard fails acting as independent vetoes that auto-reject a token](/imgs/blogs/onchain-due-diligence-checklist-3.png)

**Hard fails are vetoes.** A hard fail auto-rejects the token *no matter how good everything else looks*. The four canonical hard fails:

1. **Honeypot** — buys work, sells revert or are taxed near 100%. You can't exit, so it's worth \$0 to you.
2. **Owner can mint or freeze** — the mint function isn't renounced and an active owner can dilute or pause transfers.
3. **Unlocked, team-held LP** — the team can pull the entire pool in any block.
4. **Extreme concentration** — one non-LP wallet holds more than ~30% of supply.

Any one of these is a stop sign. The figure above is the right mental model: these are not points that lower a score — they are independent circuit breakers, each wired directly to STOP. A token with smart money, real revenue, and a beautiful chart is still an instant NO-GO if it's a honeypot, because none of the good stuff matters when you can't sell.

**Soft flags don't reject — they size you down.** A token can be clean (no hard fail) but carry warning signs: thin-but-not-dead liquidity, a moderate unlock coming, usage that's real but small, no smart money present yet, a top wallet at 15% (concerning but under the 30% hard line). These don't make the token un-buyable; they make it *smaller*. The more soft flags, the smaller the position — so that if the risk you flagged actually materializes, it costs you less.

![GO and NO-GO decision tree routing tokens to reject size down or full size](/imgs/blogs/onchain-due-diligence-checklist-6.png)

The decision tree above is the synthesis as a flow: any hard fail → NO-GO; otherwise count soft flags and let the count set your size. Zero flags with smart money present → full size. One or two flags → a reduced position. Three or more flags → NO-GO by *score* — not because any single thing is fatal, but because too many things are uncertain at once and the expected value no longer justifies the risk.

**The limits — say this out loud.** DD reduces ruin risk; it does not guarantee gains. A token can pass all seven gates and still dump because the market turns, the narrative dies, or a holder you couldn't have flagged decides to exit. A clean contract with locked liquidity and 800 holders can still lose you 70% in a bad month. The checklist's job is to cut off the ways you go to *zero* — the honeypots, the rugs, the obvious dump setups — and to size your bets to the uncertainty you can measure. It tilts the odds; it doesn't remove them. Anyone selling "guaranteed" on-chain DD is themselves a flag.

## The tool map: which tool for which gate

You never have to guess where to look. Each gate has a primary tool that surfaces the on-chain fact in under a minute, almost all of them free.

![Matrix mapping each checklist section to a primary tool, what to read, and the red flag](/imgs/blogs/onchain-due-diligence-checklist-4.png)

The map above is worth internalizing because it turns a vague "do your research" into a concrete sequence of clicks:

- **Contract safety** → **Token Sniffer / GoPlus** for the automated scan, **Etherscan** (or BscScan/Solscan/Arbiscan for the relevant chain) for the verified source and the read/write functions. Red flag: unverified, or a honeypot result, or live mint/blacklist powers.
- **Liquidity** → **DEXScreener / DexTools** for pool depth, LP lock status, and price-impact estimates. Red flag: a thin pool, or LP that's unlocked and team-held.
- **Distribution** → **Etherscan holders tab** for the raw list, **Bubblemaps** for the funding-link graph that exposes fresh-wallet clusters. Red flag: one wallet over ~30%, or a cluster of "different" holders that all trace to one funder.
- **Unlocks** → **TokenUnlocks / CryptoRank** unlock calendars. Red flag: a big cliff landing into thin liquidity, or an FDV many multiples of MC.
- **Usage** → **DefiLlama** (TVL, fees, revenue) and **Token Terminal** / **Dune** dashboards. Red flag: TVL that's all one mercenary farm, or near-zero real fees.
- **Money** → **Nansen / Arkham** for smart-money labels and exchange-flow tracking; DEXScreener also surfaces top traders per token. Red flag: labeled scammer wallets accumulating, or team wallets sending to exchanges.

A practical sequencing note: run the **hard-fail tools first** (GoPlus, Etherscan source, DEXScreener LP status, Etherscan holders). If any of those trips a hard fail, you stop — you never spend time on the quality gates for a token that's already a NO-GO. The order isn't decorative; it's designed so the cheapest, most decisive checks happen before the slower analytical ones.

## How to read it: a full end-to-end walkthrough

Let's run the entire checklist on one token, start to finish, tool by tool. We'll use an illustrative token — call it `MEOW` at address `0xA11ce…` (a placeholder, not a real token) — and walk every gate as you'd actually do it. Watch how fast the chain answers.

**Gate 1 — Contract safety (≈90 seconds).** Paste the contract address into **GoPlus Security**. It returns: contract verified ✓, owner address renounced ✓, `is_honeypot: false` ✓, buy tax 2% / sell tax 3% (acceptable), no blacklist function ✓, no external mint ✓. To trust it, you open the contract on **Etherscan**, click *Contract → Read/Write*, and confirm there's no callable `mint` and the owner is `0x000…dead`. **Gate 1: PASS.** Had GoPlus shown `is_honeypot: true` or an active owner with mint, you'd have stopped here and never opened another tab.

**Gate 2 — Liquidity (≈60 seconds).** Search MEOW on **DEXScreener**. The main pool shows **\$1.8M liquidity**, paired with ETH. You click through to the LP lock status: the LP tokens are **locked on Team Finance for 9 months**. You eyeball price impact — a \$5,000 trade is ~0.3% of the pool, negligible. **Gate 2: PASS**, with a mild note that 9 months is decent but not the 12+ you'd prefer (a faint soft flag).

**Gate 3 — Distribution (≈3 minutes).** On **Etherscan's Holders tab**, the top "holder" is the LP pool (expected, ignore). The top *non-contract* wallet holds **6.2%** — under the 30% hard line. The top 10 non-contract wallets hold ~28% combined. You then drop the address into **Bubblemaps**: the top holders are *not* connected by funding links — no octopus cluster of fresh wallets traced to one funder. Holder count is ~640. **Gate 3: PASS**, with a soft flag that 640 holders is on the lower side and top-10 at 28% is a touch concentrated.

**Gate 4 — Unlocks (≈90 seconds).** On **TokenUnlocks**, MEOW shows team and investor allocations vesting linearly over 24 months with no large cliff in the next 90 days. FDV is **\$32M** against a **\$24M** market cap — a healthy ~1.3× ratio, meaning most supply is already circulating and you're not buying into massive future dilution. **Gate 4: PASS.**

**Gate 5 — Usage (≈3 minutes).** On **DefiLlama**, MEOW's protocol shows **\$9M TVL** that's been roughly flat for two months (not pumped by a new emission), **\$48,000/month in fees**, and emissions that are modest relative to fees. **Dune** dashboards show ~1,200 weekly active addresses actually transacting, not just holding. The protocol earns real money and the TVL looks sticky. **Gate 5: PASS.**

**Gate 6 — Money (≈3 minutes).** On **Nansen**, two wallets labeled "Smart Money" added MEOW over the past ten days; none of the holders carry "scammer" or "rug deployer" labels. **Arkham** shows no team-wallet inflows to exchanges in the last week (no distribution). **Gate 6: PASS**, leaning positive — smart money is quietly present.

**Gate 7 — Synthesis.** No hard fails. Soft flags: (a) LP lock is 9 months not 12+, (b) holder count and top-10 concentration are slightly elevated. That's **two soft flags** — clean enough to buy, but not a zero-flag full-size GO. Per the sizing rule, you take a reduced position: if your plan was \$5,000, you put in ~\$1,500. **GO, sized down.** The whole pass took under fifteen minutes and cost nothing, and it converted "the chart looks good" into a defensible, sized decision.

#### Worked example: sizing a position down for soft flags

Your standard position size for a small-cap token is **\$5,000**. MEOW above passed all hard fails but carried two soft flags (a shorter-than-ideal LP lock and slightly elevated holder concentration). Your sizing rule scales the position to the flag count: 0 flags = 100% (\$5,000), 1 flag ≈ 60% (\$3,000), 2 flags ≈ 30% (\$1,500), 3+ flags = NO-GO (\$0). Two flags → you deploy **\$1,500**, not \$5,000. Now suppose the smaller of your two flagged risks materializes: the LP lock expires in 9 months, the team doesn't re-lock, and the token drops 50% on the uncertainty. On a \$5,000 position that's a \$2,500 loss; on the flag-sized \$1,500 position it's a \$750 loss. You didn't avoid the flagged risk — you can't, that's why it's a flag — but you made it cost you a third as much. Sizing is how soft flags translate into dollars saved.

![Before and after comparison of a rigged token versus a healthy token on liquidity and holders](/imgs/blogs/onchain-due-diligence-checklist-5.png)

The before/after above is the fifteen-second version of the whole walkthrough: two tokens can post nearly identical prices and charts, but the liquidity and holder facts — unlocked vs. locked LP, \$40k vs. \$2M pool, a 42% top wallet vs. an under-5% top wallet, 60 co-funded holders vs. 800 independent ones — sort them into NO-GO and GO before you've looked at a single candlestick. The chart is the *last* thing you check, not the first.

## Why the base rate makes the checklist worth it

Here is the single number that should make running this checklist feel cheap: on Solana launchpads at their peak, roughly **8 million tokens** were launched cumulatively, and only about **1.4% ever reached a meaningful market cap.** The other ~98.6% effectively went to zero.

![Bar chart showing about 98.6 percent of launchpad memecoins go to zero versus 1.4 percent surviving](/imgs/blogs/onchain-due-diligence-checklist-8.png)

Sit with that. If you bought a *random* launchpad token, your prior probability of it going to ~\$0 is ~98.6%. The default outcome is total loss. This is why a checklist that makes "no" the cheap, automatic answer is the highest-return habit you can build: the base rate is so brutally negative that **avoiding the typical token is most of the game.** You don't need to find every winner; you need to not buy the 98.6%.

#### Worked example: the base rate makes skipping easy

Say you have **\$10,000** and you're tempted to "spray and pray" \$200 each across 50 random launchpad tokens, the way a euphoric group chat encourages. With a ~1.4% survival rate, you'd expect roughly **0–1 of your 50** to reach a meaningful cap; the other ~49 trend to \$0. If your one survivor 10×'d (\$200 → \$2,000) and the other 49 went to zero (−\$9,800), you'd be left with ~\$2,000 — a **\$8,000 loss** on \$10,000. The base rate is so negative that even an occasional 10× doesn't bail you out of buying randomly. The checklist flips this: by filtering to the small set of tokens that clear seven gates, you stop drawing from the 98.6% pile. You will still have losers — DD doesn't guarantee gains — but you've stopped funding the structural losing game. The cheapest \$8,000 you ever "make" is the \$8,000 you don't lose by saying no.

This is also why the *consistency* a checklist buys matters more than any single check. The 98.6% pile is seductive precisely when you're most likely to skip the checklist — when the chart is vertical and everyone's posting gains. The written gate is what stands between your excitement and the base rate.

## Running it as a habit: the meta-skill

The checklist only pays if you actually run it — every time, including the times you're certain you don't need to. So a few notes on turning it from a list into a reflex.

**Pre-commit to the gates before you feel the urge.** The reason a written checklist works is the same reason a pre-set stop-loss works: it moves the decision to a calm moment so a hot moment can't override it. Decide *now*, while no token is pumping, that your rule is "no buy without all seven gates." When the green candle arrives, you're not deciding whether to check — you already decided, and the only question is what the gates say. The most expensive trades in crypto are the ones where you knew the checklist and felt, just this once, too late to run it. There is never a token so urgent that fifteen minutes of DD would have cost you the opportunity but losing the whole position wouldn't.

**Speed comes from sequencing, not from skipping.** A practiced pass is fast because you front-load the hard fails. Scan the contract (GoPlus + a glance at Etherscan), check the LP lock and pool depth (DEXScreener), and pull the top holders (Etherscan + Bubblemaps) — three tools, maybe five minutes, and the large majority of NO-GO tokens die right there. You only spend time on unlocks, fundamentals, and smart-money flow for the small fraction that survive the hard fails. The goal isn't to do less work per token; it's to do *no* work on the tokens that were always going to fail.

**Write down the verdict and the reason.** Keep a one-line log: token, date, GO/NO-GO, and the deciding fact ("NO-GO: top wallet 41%, \$8M into a \$1.5M pool"). Two things happen. First, you stop re-litigating the same token every time it pumps — you already ruled on it. Second, after a few months you can audit yourself: did the NO-GO tokens you rejected actually die? Did your GO tokens cluster around the clean-contract, deep-liquidity, smart-money profile? The log turns the checklist from a static rule into a feedback loop that sharpens your thresholds with real outcomes.

**Calibrate, don't ossify.** The exact numbers here — 30% concentration, 12-month locks, the 60/30/0% sizing fractions — are reasonable defaults, not laws of physics. As you build your own outcome log, tighten or loosen them to fit your risk tolerance and the segment you trade (a blue-chip DeFi token and a day-old memecoin warrant different bars). What must *not* change is the structure: seven gates, hard fails as vetoes, soft flags as size, run every time. The thresholds are dials; the discipline is the machine.

## Common misconceptions

**"It's audited, so it's safe."** An audit checks the contract code for bugs at a point in time. It does *not* check that the team won't rug via legitimate functions (pulling unlocked liquidity, dumping their allocation), that the liquidity is locked, or that the token has any users. Audited tokens have rugged and dumped many times. An audit clears part of Gate 1 and nothing else — it's one input, not a verdict. Verify liquidity, distribution, and behavior yourself.

**"Lots of holders means it's distributed."** The holder *count* is trivially faked: a deployer can airdrop dust to 5,000 fresh wallets and show "5,000 holders" while controlling 80% of supply across a cluster. The count without the *concentration and funding graph* is theater. This is exactly what Bubblemaps exists to defeat — it shows whether the holders are independent or one entity wearing 5,000 masks. Always check the top-holder percentage and the funding links, not the headline holder number.

**"Smart money is buying, so I should buy."** Smart-money labels suffer survivorship bias (they're assigned to wallets that already won) and tell you nothing about *your* situation. A smart wallet that bought at \$0.002 and is up 25× can dump on you and still call it a win; you'd be buying their exit. Smart money is a *Gate 6 input* that tilts probability, not a buy signal that overrides hard fails. And copying their entry without their cost basis, size, or exit plan is a documented way to lose money — see [the perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain).

**"High TVL means the protocol is real."** TVL measures dollars parked, not value created. Mercenary capital chasing a 200% APY emission inflates TVL that vanishes the day rewards stop. The honest read is fees and revenue (what users *pay*) and TVL *stickiness* (did it survive the last emission cut), not the headline number. A \$1B TVL protocol earning \$15k/month in fees is a worse business than a \$100M protocol earning \$400k/month.

**"On-chain DD guarantees I won't lose money."** It guarantees nothing. It cuts off the ways you go to *zero* (honeypots, rugs, obvious dump setups) and sizes your bets to measurable uncertainty. A token that passes every gate can still fall 70% in a bad market, on a dead narrative, or because a holder you couldn't flag exited. DD shifts the distribution of outcomes in your favor and caps your downside — it does not promise upside. Treat anyone claiming otherwise as a flag.

## The playbook: what to do with it

Here is the checklist as an if-then sequence you can run on every token, in order, before any buy. The pattern for each gate is **the signal → the read → the action → the false-positive to watch.**

**Gate 1 — Contract safety.** *Signal:* token contract. *Read:* GoPlus/Token Sniffer scan + Etherscan verified source. *Action:* if honeypot, active mint, blacklist, or unverified → **NO-GO (hard fail)**. Else pass. *False positive:* automated scanners miss things — confirm the `_transfer` logic and owner on Etherscan; a "renounced" owner that's actually a proxy admin is still a risk.

**Gate 2 — Liquidity.** *Signal:* DEX pool. *Read:* DEXScreener depth, LP lock/burn status, price impact at your size. *Action:* if LP unlocked and team-held → **NO-GO (hard fail)**; if pool is thin relative to your exit → soft flag (size down). *False positive:* a "locked" LP on an obscure locker contract you can't verify is not really locked — check the locker.

**Gate 3 — Distribution.** *Signal:* holder list. *Read:* Etherscan holders + Bubblemaps funding graph. *Action:* if one non-LP wallet > ~30% → **NO-GO (hard fail)**; moderate concentration (top wallet 15–30%, few holders) → soft flag. *False positive:* exchange hot wallets and locked-team contracts look like big holders but aren't free-floating sell pressure — label them out before judging.

**Gate 4 — Unlocks/emissions.** *Signal:* vesting schedule. *Read:* unlock calendar + FDV/MC ratio + emission rate. *Action:* huge cliff into thin liquidity → NO-GO or hard exit before the date; FDV many multiples of MC, or high sell-pressure emissions → soft flag. *False positive:* an unlock to a foundation/treasury that demonstrably doesn't sell is less dangerous than one to early investors at a 20× gain — read *who* unlocks.

**Gate 5 — Usage/fundamentals.** *Signal:* protocol activity. *Read:* DefiLlama/Token Terminal fees + revenue + TVL trend, Dune for active users. *Action:* near-zero real fees or all-mercenary TVL → soft-to-hard flag depending on valuation; sticky TVL + real fees → positive. *False positive:* wash-traded volume and self-dealing TVL inflate the numbers — weight *fees paid by users* over volume and raw TVL.

**Gate 6 — Money.** *Signal:* who holds and where it flows. *Read:* Nansen/Arkham smart-money labels, team-wallet exchange flows. *Action:* smart money quietly accumulating + no team distribution → positive; labeled scammers in, or team sending to exchanges → flag to NO-GO. *False positive:* smart-money labels are survivorship-biased and copyable; treat as a tilt, never a buy trigger, and never copy without their cost basis and exit.

**Gate 7 — Synthesis (the GO/NO-GO).** *Read:* any hard fail? then count soft flags. *Action:* hard fail → NO-GO; 0 flags + smart money → full size; 1–2 flags → reduced size (use the grid below); 3+ flags → NO-GO by score. *False positive:* don't let a great narrative or a green chart talk you past a hard fail — that's the exact moment the checklist is protecting you.

![Grid showing position size shrinking as the number of soft flags rises](/imgs/blogs/onchain-due-diligence-checklist-7.png)

The sizing grid above closes the loop: hard fails set size to zero (the override that beats every other cell), and among clean tokens the soft-flag count sets the fraction of your planned size — 0 flags full, 1 flag ~60%, 2 flags ~30%, 3+ a NO-GO by score. The whole philosophy in one sentence: **flags don't make you skip a clean token, they make the bet smaller, so a bad surprise costs less.** That single rule converts every fuzzy worry you uncovered into a concrete dollar amount of capital at risk.

Run this every time. Write the seven gates on a sticky note next to your screen if you have to. The day you skip it because "this one's obviously fine" is the day you find out it wasn't — and the base rate says it usually isn't.

## Further reading & cross-links

This post is the Track G hub; each gate has (or will have) a dedicated deep-dive. Within the series:

- [Rug-pull and honeypot detection](/blog/trading/onchain/rug-pull-and-honeypot-detection) — the full Gate 1 method: reading contract powers, spotting honeypots, the rug mechanics.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — the full Gate 3 method, including Bubblemaps clustering.
- [Early-buyer and insider detection](/blog/trading/onchain/early-buyer-and-insider-detection) — tracing team and insider wallets for Gates 3 and 6.
- [What is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain) and [following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) — the Gate 6 signal, used without becoming exit liquidity.
- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — reading distribution pressure in Gate 6.
- [The perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) — why a smart wallet's buy is not your buy.
- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — the full toolbox behind every gate (Etherscan, DEXScreener, DefiLlama, Nansen, Bubblemaps, Dune).
- [What is on-chain analysis](/blog/trading/onchain/what-is-onchain-analysis) — the foundation: why the public ledger is an edge.

Background on the moving parts: [DeFi protocols (Uniswap, Aave, MakerDAO)](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) for how AMM pools and lending markets work, [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) for where exchange flows go, and [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) for what a token contract actually is.
