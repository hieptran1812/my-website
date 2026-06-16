---
title: "Analyzing DEX and AMM Activity: Volume, Fees, LP Returns, and Where the Flow Is"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How to read a decentralized exchange as a signal — real versus washed volume, the fees it generates, what a liquidity provider actually earns after impermanent loss, the volume-to-TVL capital-efficiency ratio, and how to spot where capital and traders are rotating next."
tags: ["onchain", "crypto", "dex", "amm", "volume", "fees", "impermanent-loss", "liquidity", "uniswap", "curve", "solana", "defillama"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Decentralized exchanges are where most on-chain price discovery and real flow actually happen, and their activity is fully public, so you can read volume, fees, and liquidity-provider returns directly instead of trusting a dashboard's green number.
>
> - A DEX is a set of automated-market-maker pools; its **volume** is dollars swapped, its **fees** are volume times a fee tier, and those fees are the only pay a **liquidity provider** gets for taking on **impermanent loss**.
> - The four numbers that matter: organic (not washed) volume, fees generated, LP return after impermanent loss, and the **volume-to-TVL ratio** — how hard each dollar of liquidity is working.
> - What you do with it: rising *organic* volume and fees is a real demand signal worth a long or a watch; falling vol/TVL is liquidity leaving; and tracking which DEX or chain is winning volume tells you where the next flow is rotating.
> - The number to remember: **a pool doing \$50M a day on \$10M of liquidity (a vol/TVL of 5) is working hard; one doing \$2M on \$50M (0.04) is lazy, stale capital on a timer.**

On the morning of November 10, 2021, the most-watched number in crypto was a market cap. By the spring of 2022, the most-watched number among the people who actually traded was a far less glamorous one: the daily volume flowing through a handful of decentralized-exchange pools, and the fees those pools were throwing off. The two numbers told completely different stories. The market cap was a multiplication problem that a marketing team could inflate; the volume and fees were a record of money actually changing hands, written permanently to a public ledger that nobody could quietly edit. When the cycle turned, the market-cap watchers were the last to know. The flow-watchers saw the volume drain out of the speculative pools and rotate into stablecoin pairs weeks before the price charts admitted what was happening.

That gap — between the headline number a project wants you to look at and the activity number that is actually true — is the entire reason to learn DEX analysis. A centralized exchange like Binance reports its own volume, and you have to take its word for it; a decentralized exchange (DEX) reports nothing, because it cannot. It is a smart contract, and every swap that touches it is a public, timestamped transaction. So the volume is not a claim. It is a count you can reproduce. The fees are not a press release. They are dollars that landed in a contract and got distributed to liquidity providers. If you learn to read these numbers, you stop reacting to price and start reading the flow that *moves* price.

This post is the protocol-level companion to [reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools), which taught you to read a single pool's depth and slippage. Here we zoom out to the *activity* across a pool, a DEX, or a whole chain: how much real volume is flowing, how many fees that volume generates, what a liquidity provider actually earns once you subtract impermanent loss, how hard the liquidity is working, and how to read where capital and traders are rotating. We will build every term from zero, ground every idea in dollar math, and finish with the workflow you run on DefiLlama, Dune, and DEX Screener to turn DEX activity into a trade, an LP decision, or a hard pass.

![Four linked numbers — volume feeding fees, fees against impermanent loss for the LP, and volume over TVL as capital efficiency](/imgs/blogs/analyzing-dex-and-amm-activity-1.png)

The figure above is the whole mental model in one frame, and the rest of the post earns it. Volume drives fees. Fees pay the liquidity provider against impermanent loss. Volume divided by the total value locked tells you how hard that liquidity is working. And reading where all of this is rotating tells you where the flow is going next. Four numbers, all public, all on-chain.

## Foundations: DEXes, AMMs, volume, fees, and what an LP earns

Before we can read DEX activity, we need the vocabulary. Every term is defined on first use, because the edge of on-chain analysis is precisely that you can verify each of these directly rather than taking a platform's word for it. If you have read the single-pool post, the first three definitions are a fast recap; the volume, fee, and impermanent-loss definitions are the new ground this post stands on.

**A DEX is a smart contract, not a company.** A *centralized exchange* (CEX) like Binance or Coinbase runs an *order book* — a matching engine that pairs buyers with sellers — and it reports its own statistics. A *decentralized exchange* (DEX) has no matching engine and no other people to wait for. It holds a pool of tokens inside a smart contract, and you trade directly against that pool. Because the pool is a public contract, every trade is a public transaction, and the exchange cannot misreport its own activity even if it wanted to.

**An AMM is the pricing formula.** Most DEXes use an *automated market maker* (AMM): a formula, not a human or an order book, sets the price. The classic design, popularized by Uniswap v2, is the *constant-product* rule `x \* y = k`, where `x` and `y` are the two token reserves in the pool and `k` is held constant by every trade. The price of one token quoted in the other is simply the ratio of the reserves, `y / x`. We derive this in full in the single-pool post; here you only need to know that a pool is two piles of tokens, and the price emerges mechanically from their ratio.

**A liquidity provider funds the pool and earns the fees.** The reserves do not appear from nowhere. People called *liquidity providers* (LPs) deposit both tokens into the pool and, in return, receive an *LP token* — a receipt for their share of the pool. Every swap pays a *fee* (classically 0.30%, with other tiers we will cover) that is added to the reserves and accrues to the LPs in proportion to their share. That fee is the LP's entire pay. It is the yield that has to compensate them for two costs: impermanent loss, and the opportunity cost of locking their capital up. We will spend a whole section on whether that fee actually beats those costs, because that is the question that decides whether providing liquidity makes or loses money.

**Volume is dollars swapped, not dollars held.** This is the central new term. A pool's *volume* over some window — usually 24 hours — is the total dollar value of all the swaps routed through it in that window. It is a *flow*, not a *stock*: it measures activity, how much money moved, not how much money is sitting there. A \$10M pool can do \$1M of volume in a quiet day or \$100M in a frantic one. Volume is the raw signal of demand and attention — *if* it is real. As we will see, volume is also the single easiest DEX number to fake, which is why "real versus washed" is step one of any honest read.

**Total value locked is the dollars sitting in the pool.** The *total value locked* (TVL) is the combined dollar value of both reserves — the depth you can trade against. Unlike volume, TVL is a *stock*: a snapshot of how much liquidity is parked in the contract right now. A pool with 1,000,000 TOK at \$0.50 and 500,000 USDC has \$500,000 + \$500,000 = \$1,000,000 of TVL. Volume and TVL are the numerator and denominator of the single most useful efficiency ratio in DEX analysis, which we will build below.

**Fees are volume times the fee tier.** This is the identity that makes fees readable. If a pool charges a 0.30% fee and does \$100M of volume in a day, it generates \$100M × 0.0030 = \$300,000 in fees that day, all of which goes to the LPs (on a pure-AMM pool; some protocols skim a slice for the treasury). Fee tiers vary: Uniswap v3 offers 0.01%, 0.05%, 0.30%, and 1.00%, chosen to match the pair's volatility — tight stablecoin pairs use the 0.01% or 0.05% tier, blue-chip pairs the 0.30%, and exotic long-tail tokens the 1.00%. The key insight, which we will return to: **fees are far harder to fake than volume**, because real fees require real counterparties paying real money, while wash volume can be self-funded round-trips that generate fees the wash trader is just paying to themselves.

**Impermanent loss is the LP's hidden cost.** Here is the term that trips up every new liquidity provider. When the price of the two tokens in a pool *diverges* — one rises a lot relative to the other — the constant-product rule automatically rebalances the pool: it sells the token that rose and buys the token that fell, to keep `x \* y = k`. The result is that an LP ends up holding *more* of the token that went down and *less* of the token that went up, compared to someone who simply held both tokens in a wallet. That shortfall, relative to just holding, is called *impermanent loss* (IL). It is "impermanent" only in the sense that it reverses if the price comes back to where it started; if the price stays diverged and the LP withdraws, the loss is entirely permanent. IL is the cost the fees have to beat. We will do the math in full below.

**The volume-to-TVL ratio is capital efficiency.** Divide a pool's daily volume by its TVL and you get the *turnover ratio*: how many times per day each dollar of liquidity is reused by traders. A pool doing \$50M of volume on \$10M of TVL has a vol/TVL of 5.0 — its liquidity works hard, turning over five times a day and earning fees five times. A pool doing \$2M of volume on \$50M of TVL has a vol/TVL of 0.04 — its liquidity is lazy, mostly idle, earning almost nothing. This single ratio is the fastest way to tell a thriving pool from a stranded one, and it is one division away from any explorer page.

**Aggregator-routed volume muddies the count.** One complication worth flagging up front. Most DEX trades today do not go directly to one pool; they go through an *aggregator* — a router like 1inch, Matcha, Jupiter (on Solana), or a wallet's built-in swap — that splits a single trade across several pools to get the best price. So the volume you see on a given pool is partly *its own* organic demand and partly slices of trades the aggregator routed there. This is mostly fine for reading total demand, but it means you should read volume at the *token* level (all pools combined) rather than obsessing over one pool, and it means the venue that "wins" volume is often the one aggregators route to, not the one users consciously chose.

With that vocabulary, we can read any DEX as a signal. The rest of the post is about reading it *well*: separating real volume from washed, turning fees into a usage read, doing the fee-versus-IL math that decides whether LPing pays, computing capital efficiency, and reading where the flow is rotating.

## Volume: real demand versus washed noise

Volume is the loudest number on any DEX dashboard and the most dangerous one to take at face value. It is loud because it is the headline — DEX Screener, DefiLlama, and every token page lead with 24-hour volume — and dangerous because it is the cheapest number to fake. So the first discipline of DEX analysis is not *reading* volume but *deciding whether the volume is real*.

**Why volume is the demand signal.** When volume is organic — many distinct addresses, swapping in both directions, at sizes that make sense for real traders — it is the cleanest demand signal on-chain. Price tells you where the last trade happened; volume tells you how much conviction is behind the move. A token whose price is rising on rising organic volume has real buyers stepping in. A token whose price is rising on *falling* volume is running out of fuel — the move is being carried by fewer and fewer participants, which is exactly the setup that reverses. This is the same logic technical analysts apply to equity volume, except on-chain you can verify it transaction by transaction instead of trusting an exchange's reported tape.

**Wash trading is the counterfeit.** *Wash trading* is the practice of trading with yourself — the same entity buying and selling through wallets it controls — to manufacture the *appearance* of volume and activity. On a DEX it is mechanically simple: route funds in a loop through a pool, and the volume counter goes up. The motive is almost always to make a token look more active and liquid than it is, so that real buyers feel safe stepping in. Detecting it is its own discipline, covered in depth in [detecting wash trading](/blog/trading/onchain/detecting-wash-trading) and [detecting fake volume vs organic demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand), but the core tells are worth stating here because they are the lens you read volume through:

- **Volume that dwarfs liquidity.** Healthy organic trading turns over some sensible fraction of the pool. A pool showing volume *many times* its TVL — \$10M of daily volume on a \$200k pool — is almost always washing. Real demand at that turnover would have moved the price violently; flat price plus huge volume means the same funds are cycling.
- **A handful of addresses doing most of the volume.** Organic volume comes from many distinct wallets. If three addresses account for 80% of the day's swaps, and they keep trading back and forth at similar sizes, that is a wash ring, not a market.
- **Round-trip timing.** Wash trades often show the same value buying and then selling within minutes, leaving the wash trader's token balance roughly unchanged but the volume counter inflated.
- **No price impact from "large" trades.** If a \$500k "trade" moved a \$300k pool's price by almost nothing, it was not a real trade against the pool — it was internal routing or a self-trade that never really consumed depth.

**The fee cross-check.** The single most powerful wash-trading filter is to compare volume against *fees actually earned by independent LPs*. Wash volume still pays fees — but if the wash trader is also the LP, they are paying fees to themselves, so the fees are not real revenue to anyone else. When you see huge volume but the pool's independent LPs are not earning a yield that matches (because the "fees" are circular), the volume is fiction. Real volume from real outside traders shows up as real fee revenue that real LPs keep. We will lean on this idea hard in the fees section.

#### Worked example: \$1M of organic volume growing into a real demand signal

A new token launches and, for its first week, trades \$1M of volume a day on a \$500k pool — a turnover of about 2× a day, which is high but plausible for a hot launch. You check the swaps: roughly 400 distinct addresses traded, the largest single address did under 8% of the volume, and buys outnumbered sells. Over the next three weeks the daily volume climbs to \$4M, the distinct-address count rises to over 1,500, and the price grinds up alongside it. Because the volume is *broad* (many wallets) and *two-sided* (real buyers and sellers), and because it is being matched by rising fee revenue to independent LPs, this is an organic demand signal, not a wash. A trader reading this sees accumulating real interest before it is obvious on the price chart. The intuition: \$1M of volume from 400 wallets is a market forming; \$1M of volume from 3 wallets is a magician's trick, and the address spread is what tells them apart.

**Volume trend beats volume level.** Once you have established that the volume is real, the *trend* in that volume carries more signal than the absolute number. A token doing a steady \$5M a day is a known quantity; a token whose organic volume is *accelerating* — \$1M, then \$2M, then \$4M across consecutive days, with the distinct-address count rising in step — is one where real demand is compounding, and that acceleration usually shows up on-chain before the price breaks out. The mirror image is the warning: a token whose price is grinding higher while its organic volume *fades* day over day is being carried by fewer and fewer participants, which is the classic exhaustion setup. Because every swap is timestamped on-chain, you can build this volume-trend read precisely — daily organic volume and daily distinct traders, side by side — rather than eyeballing a centralized exchange's reported bars. That precision is the on-chain edge: you are reading the actual demand curve forming, not a chart of it.

**DEX volume is where price discovery now happens.** It is worth stating plainly why DEX volume deserves this much attention. For a large and growing share of tokens — and for essentially all new and long-tail tokens — the DEX pool *is* the primary market. There is no deeper order book on a centralized exchange setting the "real" price; the AMM pool is the real price, and the DEX volume is the real flow. So when you read DEX activity, you are not reading a sideshow to some main venue — for these assets you are reading the main venue itself, with the bonus that its entire order flow is public. That is the structural reason DEX analysis has become central to on-chain reading: the chain is no longer just a settlement layer that price discovery happens *above*; for a huge swath of the market, the chain is *where* price discovery happens, swap by public swap.

The lesson of this section is a single habit: never read a volume number without first asking *whose* volume it is, and *which way it is trending*. A green volume bar is a question, not an answer.

## Fees: the LP's pay and the hardest usage number to fake

If volume is the loud, fakeable number, fees are the quiet, honest one. Fees are *volume that actually got paid*, and on a real pool they land as revenue that real liquidity providers keep. That is why, in the [on-chain fundamentals post](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl), fees sit at the center of valuing a protocol: they are the closest thing crypto has to revenue, and a DEX's fees are the closest thing it has to a top line.

**Fees are mechanically simple.** Every swap pays its pool's fee tier, and that fee is added to the reserves. An LP who owns 1% of the pool earns 1% of every fee. So a pool's *total* daily fee revenue is just its daily volume times its fee tier, and your slice is your pool share times that total. The next figure shows how much the tier alone changes the revenue from a fixed amount of real volume.

![Bar chart of fees per day generated by 100 million dollars of volume across four fee tiers from one basis point to one hundred basis points](/imgs/blogs/analyzing-dex-and-amm-activity-2.png)

The figure plots a fixed \$100M of daily volume across the four standard Uniswap-style fee tiers. At the 0.01% tier — used for tightly pegged stablecoin pairs where competition forces fees down — \$100M of volume generates only \$10,000 a day. At the 0.30% standard tier it generates \$300,000. At the 1.00% exotic tier it generates \$1,000,000. Same volume, thirty-fold difference in fees, purely from the tier. The numbers are illustrative — they are computed from the identity *fees = volume × tier*, not measured from a specific pool — but the point is exact: the fee tier is a deliberate choice that trades off attracting volume (lower fees) against earning more per trade (higher fees), and you cannot judge a pool's revenue from its volume alone without knowing its tier.

**Why fees beat volume as a usage proxy.** Volume can be washed; fees, on a pool with *independent* LPs, are much harder to fake, because faking them costs the wash trader real money paid to strangers. If I wash \$10M of volume through a pool I do not own the liquidity in, I pay \$30,000 in fees (at 0.30%) straight to the LPs — a \$30,000 bill to print a vanity number. Nobody sustains that. So when a DEX's *fee revenue* grows and stays grown, it is far stronger evidence of real usage than a volume spike, because real outside traders had to pay real money to generate it. This is why analysts who value DEX protocols anchor on fees, annualize them, and treat them as the revenue line against which to judge the token.

**The subsidy trap.** There is one big false positive, and it is the mirror image of wash trading. Many protocols *pay* users to trade or to provide liquidity — handing out their own governance token as a reward. This *liquidity mining* inflates both volume and apparent LP yield with subsidies that are not organic fee revenue. A pool can show a "60% APR" that is 5% real fees and 55% token emissions. When the emissions stop, the mercenary capital that came only for the rewards leaves, the TVL collapses, and the volume goes with it. So when you read fees, separate *organic* fees (paid by traders) from *subsidized* yield (paid by the protocol's own inflation). Organic fees are usage; subsidized yield is marketing on a timer. We return to this in the rotation section, because incentive-driven flow is the loudest and least durable kind.

#### Worked example: a 0.30% fee tier on \$100M of volume

Take a blue-chip pool — say ETH paired against a stablecoin — sitting at the 0.30% fee tier and doing \$100M of volume on a typical day. Its total fee revenue that day is \$100,000,000 × 0.0030 = \$300,000. If the pool's TVL is \$50M, those \$300,000 of daily fees annualize to roughly \$300,000 × 365 = \$109.5M against \$50M of liquidity — a fee yield on the order of 200% a year *before* impermanent loss, which is why deep, high-volume blue-chip pools attract liquidity even though they pay no token rewards. Now suppose you own \$100,000 of that \$50M pool, a 0.2% share. Your slice of the day's fees is \$300,000 × 0.002 = \$600, and at that pace your \$100,000 earns about \$219,000 a year in fees — again, before subtracting impermanent loss, which is the number that decides whether that headline yield survives contact with reality. The intuition: fees scale with volume and tier, but the LP only keeps what is left after IL, so a fat fee yield is the start of the LP analysis, not the end of it.

The takeaway: when you want to know whether a DEX is *really* being used, read its fees, not its volume — and make sure those fees are paid by traders, not printed by the treasury.

## LP returns and impermanent loss: the math that decides if LPing pays

We have reached the question that decides whether providing liquidity is a business or a slow bleed: does the fee revenue beat the impermanent loss? Everything else in this post — volume, fees, capital efficiency — feeds into this one comparison. An LP's true return over any period is, exactly:

`LP return = fees earned − impermanent loss − gas/opportunity costs`

If that number is positive, LPing was *positive expected value* (+EV) — you did better than just holding the two tokens. If it is negative, LPing was *negative expected value* (−EV) — you would have been richer doing nothing. Let us build the IL side carefully, because it is the half everyone underestimates.

**Why divergence costs the LP.** Recall the rebalancing: when one token rises relative to the other, the constant-product pool sells the riser and buys the faller to keep `x \* y = k`. So the LP is forced to be a *contrarian* — always selling strength and buying weakness within the pair. If the two tokens just chop sideways, that contrarian rebalancing is harmless (and the fees from all that chopping are pure profit). But if one token *trends* hard, the LP has systematically sold it on the way up, ending with a basket worth less than if they had simply held both tokens untouched. The size of that shortfall depends only on how far the price *ratio* diverged, and it is independent of fees — which is why fees and IL are two separate terms you net against each other.

The standard constant-product IL formula, for a price ratio that changes by a factor `r` (e.g. `r = 4` means one token quadrupled relative to the other), is:

`IL = 2 \* sqrt(r) / (1 + r) − 1`

You do not need to memorize it; you need to feel its shape. At `r = 1` (no divergence) IL is 0. At `r = 1.25` (a 25% relative move) IL is about −0.6%. At `r = 2` (a 2× relative move) IL is about −5.7%. At `r = 4` (a 4× move) IL is about −20%. At `r = 9` (a 9× move) IL is −37.5%. The loss accelerates with divergence, just like slippage accelerates with trade size — and it is always a loss relative to holding, never a gain. The next figure makes the fees-versus-IL race concrete with the two worked cases that matter.

![Before-and-after panels showing fees winning when price stays flat and impermanent loss winning when price diverges](/imgs/blogs/analyzing-dex-and-amm-activity-3.png)

On the left, the price chops sideways: the pool sees heavy two-way volume, the LP banks fees, IL stays near zero because the ratio barely moved, and the net result is solidly positive — LPing was +EV. On the right, the same position but the price diverges hard: the pool auto-sells the winner, the LP ends up well behind a simple hold, and the impermanent loss swamps the fees — LPing was −EV. Identical pool, identical fee tier; the *only* thing that changed is whether the price diverged. That is the whole LP gamble in one frame: **you are short volatility of the price ratio and long volume.** You win when the pair stays calm and trades a lot; you lose when it moves a lot and trades little.

#### Worked example: \$30k of fees but \$50k of IL on a divergence (LPing was −EV)

You provide \$2,000,000 of liquidity to a TOK/USDC pool — \$1,000,000 of TOK and \$1,000,000 of USDC, with TOK at \$1.00. Over the next three months the pool is active and your share earns \$30,000 in fees. Sounds great. But over those same three months TOK rips from \$1.00 to \$4.00 — a 4× relative move, `r = 4`. The IL formula gives `2 \* sqrt(4) / (1 + 4) − 1 = 2 \* 2 / 5 − 1 = 0.8 − 1 = −0.20`, a 20% impermanent loss. Against a hold position that would have been worth roughly \$2,500,000 after the move, a 20% IL is about \$50,000 of foregone value. Your net is \$30,000 of fees minus \$50,000 of IL = **−\$20,000**. You earned real fees and still lost \$20,000 versus just holding the two tokens in your wallet. The intuition: in a trending market, the fees have to be enormous to outrun the IL, and on a volatile pair they usually are not — which is why LPing a token you are bullish on is often the worst way to express that view.

#### Worked example: \$30k of fees and near-zero IL on a calm pair (LPing was +EV)

Same \$2,000,000 position, same \$30,000 of fees earned over the quarter — but this time the pair is a stablecoin pair, USDC against USDT, and the price ratio moves from 1.000 to maybe 1.002 and back. The relative move is a fraction of a percent, so `r` is essentially 1 and the IL formula gives essentially 0 — call it a few dollars. Your net is \$30,000 of fees minus roughly \$0 of IL = **+\$30,000**, a clean 1.5% quarterly return on the \$2,000,000, or about 6% annualized, with almost no divergence risk. The intuition: the *same* fee revenue is pure profit on a pair that does not diverge and a partial offset to a loss on a pair that does — so the pair you choose matters more than the fees you are quoted, and correlated or pegged pairs are where passive LPing reliably pays.

**Concentrated liquidity changes the math — in both directions.** Uniswap v3 and its imitators let an LP concentrate their capital in a *price range* instead of spreading it across all prices. Inside the range, your capital acts like a much larger v2 position, so you earn far more fees per dollar — capital efficiency can be 10× to 1000× higher. But there are two catches. First, the IL is *amplified* in the same proportion: concentrating your liquidity concentrates your divergence loss too, so a range LP suffers IL much faster when the price moves against the range. Second, when the price exits your range entirely, your position converts fully into the losing asset and **stops earning fees altogether** — you are now holding 100% of the token that fell, earning nothing, which is the worst of both worlds. So concentrated LPing is an active job: you earn spectacular fees while the price stays in range and you re-center promptly, and you bleed IL with no fee offset the moment it leaves. The vol/TVL ratio of a concentrated pool is therefore even more important — it tells you whether the *active* liquidity is being used enough to justify the amplified IL risk.

#### Worked example: a concentrated LP earning 5× the fees — until the price exits the range

You provide \$100,000 to a concentrated position in an ETH/USDC pool, tightly ranged around the current price, and because your capital is concentrated it behaves like \$500,000 of v2 liquidity *while the price stays in range*. The pool does \$20M of daily volume at the 0.05% tier, so total daily fees are \$20,000,000 × 0.0005 = \$10,000, and your effective share earns, say, \$250 a day — about 0.25% of your \$100,000 *daily*, an eye-watering rate. For three weeks the price stays in range and you bank roughly \$250 × 21 = \$5,250. Then ETH dumps 20% in a day and blows through the bottom of your range. Your position is now 100% ETH, down with the market, earning \$0 in fees until you re-range, and the concentration amplified your IL on the way out. If you do not re-center quickly, the \$5,250 you earned gets erased and then some. The intuition: concentrated liquidity is a leveraged bet that the price stays calm and in-range — it pays the highest fees in DeFi right up until it pays you nothing and hands you the falling asset.

So when does LPing actually pay? The decision collapses to one comparison run across two dimensions — how much volume the pool sees (which sets the fees) and how much the pair diverges (which sets the IL). The matrix below sorts the four cases.

![Matrix of the LP decision across high or low volume and stable or volatile pairs showing where fees beat impermanent loss](/imgs/blogs/analyzing-dex-and-amm-activity-4.png)

The four quadrants are the whole LP playbook. A high-volume, stable pair (a busy stablecoin pool) is the sweet spot: rich fees, negligible IL, clearly +EV — this is where passive liquidity reliably earns. A high-volume, *volatile* pair earns rich fees too, but a sharp divergence can hand you IL larger than a month of fees, so it is +EV only if the fees genuinely outrun the moves — size small and watch the pair. A low-volume, stable pair is near breakeven: little IL but little fee revenue, so the capital is better deployed elsewhere. And a low-volume, volatile pair is the trap: almost no fees to offset a pair that is pure IL waiting to happen — reliably −EV, the place you lose to a simple hold. The reason this matrix matters is that beginners read only the *advertised APR*, which lives entirely in the "fees" column and ignores the "IL" column — and the IL column is exactly where the volatile-pair quadrants go negative.

This is the deepest section of the post for a reason: **the headline LP yield is always before IL, and IL is the part that turns a 200% advertised APR into a loss.** Read the fees, then read the divergence, then net them. The matrix is just that netting, done in advance for the four cases you will actually meet. That is the LP analysis.

## Volume over TVL: how hard the liquidity is working

We now have the two flow numbers (volume, fees) and the LP math. The ratio that ties activity to capital is the *volume-to-TVL ratio*, and it is the fastest single read of a pool's health. It answers: how many times a day does each dollar of liquidity get reused by traders?

`vol/TVL = daily volume ÷ total value locked`

A high ratio means hard-working liquidity — every dollar of depth is earning fees many times a day. A low ratio means lazy liquidity — most of the capital is idle, earning almost nothing, and therefore likely to leave, because LPs do not park money where it earns no fees. The figure makes the two regimes visible side by side.

![Before-and-after panels comparing a hard-working pool at five times turnover with a lazy pool at four hundredths turnover](/imgs/blogs/analyzing-dex-and-amm-activity-5.png)

On the left, a hard-working pool: \$10M of TVL doing \$50M of volume a day, a vol/TVL of 5.0, where each dollar of liquidity is reused five times daily and the fees pile up. On the right, a lazy pool: \$50M of TVL — five times deeper — doing only \$2M of volume, a vol/TVL of 0.04, where the depth mostly sits idle. The lazy pool *looks* safer because it is "bigger," but its liquidity is stranded: the LPs are earning almost nothing, and that capital is on a timer until it gets redeployed somewhere it works. The hard-working pool, even though it is one-fifth the size, is the healthier one for both the LP (real yield) and the trader (depth that is actually being maintained because it pays). This is the inversion that catches beginners: *more TVL is not automatically better* — TVL that is not being used is just capital waiting to leave.

**What a "good" ratio is depends on the pair.** There is no universal threshold, but rules of thumb help. A blue-chip volatile pair (ETH/USDC) healthily turns over a meaningful fraction of its TVL daily — a vol/TVL between roughly 0.3 and 3 is normal and healthy. A stablecoin pair can sustain a much higher ratio (5, 10, even more) because the depth is reused constantly with tiny price moves. And a vol/TVL *far* above what the pair should support — say 20 or 50 — is a wash-trading flag, not an efficiency badge: that much real turnover would have moved the price. So you read the ratio in two directions: too low means stranded, soon-to-leave liquidity; impossibly high means likely-fake volume. The healthy zone is in between, and it differs by pair type.

#### Worked example: \$50M/day on \$10M TVL versus \$2M/day on \$50M TVL

Pool A holds \$10,000,000 of liquidity and does \$50,000,000 of volume a day, so its vol/TVL is 50/10 = 5.0. At a 0.05% fee tier it generates \$50,000,000 × 0.0005 = \$25,000 of fees a day on \$10M of capital — an annualized fee yield of roughly \$25,000 × 365 / \$10,000,000 ≈ 91% before IL. Pool B holds \$50,000,000 — five times more — but does only \$2,000,000 of volume, a vol/TVL of 2/50 = 0.04. At the same 0.05% tier it generates \$2,000,000 × 0.0005 = \$1,000 of fees a day on \$50M of capital — an annualized yield of \$1,000 × 365 / \$50,000,000 ≈ 0.7%. Pool A's liquidity works 130 times harder per dollar than Pool B's. An LP should be in Pool A; the capital in Pool B is earning less than a savings account and will eventually migrate. The intuition: vol/TVL is the productivity of the liquidity, and a small, busy pool beats a large, idle one for everyone who actually cares about yield or durable depth.

The vol/TVL ratio is the number I check first on any pool, before price, before market cap — because it tells me in one division whether the liquidity is alive or stranded.

## Where the flow is: DEX dominance and capital rotation

Zoom all the way out and DEX analysis becomes a map of *where capital and traders are*. Volume does not sit still. It rotates — between pools, between DEXes, between chains — chasing new pairs, incentives, and narratives. Reading that rotation is one of the most powerful uses of on-chain data, because the flow moves *before* the prices and the headlines catch up. The figure sketches the kind of migration you watch for.

![Stacked-area chart showing DEX volume share rotating from Ethereum L1 toward Solana and layer-two AMMs across four stylized periods](/imgs/blogs/analyzing-dex-and-amm-activity-6.png)

The figure is a stylized sketch — the periods are labels and the shares are round teaching numbers, not a measured market-share dataset, because the data file backing this series carries no DEX-volume-by-venue series. But the *shape* is exactly what real rotation looks like: Ethereum L1's Uniswap-style AMMs once dominated DEX volume; over successive periods, share migrated toward layer-two AMMs (Arbitrum, Base) as fees fell and toward Solana DEXes as the memecoin cycle pulled retail flow there. The lesson is that "which venue is winning volume" is itself a tradable signal, and it has three recurring drivers:

**Incentives pull mercenary flow.** When a new DEX or chain hands out token rewards for trading and providing liquidity, volume and TVL flood in — but it is *mercenary* capital, there for the yield, and it leaves the moment the rewards taper. The 2020 "DeFi summer" and countless chain-launch incentive programs since have followed this exact arc: a vertical spike in volume and TVL, a plateau while emissions are rich, and a collapse when they stop. So a venue topping the volume charts purely on incentives is a *fade* signal as much as a follow signal — you ride it while the emissions last and you are gone before they end.

**New pairs and new chains catch organic flow.** The more durable rotation is toward wherever the *new, interesting pairs* are listing. When a narrative heats up — a new L2, a memecoin season on Solana, a restaking token wave — the organic volume goes to the venue where those tokens actually trade. This rotation is real demand following real novelty, and it tends to persist as long as the narrative does. Reading it early (the flow showing up on a chain's DEXes before the chain's token rallies) is a genuine edge.

**Aggregators concentrate flow at the best-execution venue.** Because routers split trades to the deepest, cheapest pools, volume concentrates wherever liquidity is best — which is often a self-reinforcing loop (deep pools attract aggregator flow, which generates fees, which attracts more liquidity). So a DEX quietly winning aggregator-routed volume is winning the structural game even if it never tops a marketing leaderboard.

**How to actually track the rotation.** Reading rotation is not mystical; it is a small set of comparisons you run repeatedly. At the chain level, watch each chain's *share* of total DEX volume and fees on DefiLlama over weeks — a chain whose share is steadily rising is pulling flow, and a chain whose share is fading is losing it, regardless of where the absolute number sits. At the venue level, compare the fee-generating leaders, not just the volume leaders, because fees filter out the wash and subsidy noise. And at the pair level, watch which *categories* of pairs are growing: a rotation from speculative long-tail pairs into stablecoin pairs is a classic risk-off tell (capital parking in the safest pool while it waits), while the reverse — stablecoin TVL draining into volatile pairs — is risk appetite returning. None of these requires a forecast; they are all just reading the direction the public flow is already moving, which is the entire premise of the series: flow leads price, and the chain shows you the flow first.

#### Worked example: \$8M of stablecoin-pair volume rotating into a new chain's DEXes

Suppose over three weeks you watch a newer chain's DEX volume climb from near zero to \$8,000,000 a day, and you check the composition: roughly \$6,000,000 of it is in stablecoin and blue-chip pairs (not just memecoins), spread across more than 2,000 distinct addresses, and the chain's DEX *fees* are rising in step rather than lagging. That fee confirmation matters — at a blended 0.10% tier, \$8,000,000 of volume is about \$8,000 a day of real fees being paid to LPs, which a pure incentive program would not generate organically. Because the flow is broad, fee-confirmed, and not concentrated in a single incentive-farmed pool, this reads as durable rotation rather than a mercenary spike — capital genuinely relocating, which historically precedes the chain's token and ecosystem repricing higher. The intuition: \$8M of fee-confirmed, broad-based volume migrating to a venue is the flow voting with its feet, and the chain shows you that vote weeks before the price chart counts it.

**New-pool and new-pair flow as an early signal.** Drilling below venue-level rotation, the earliest signal of all is a *new pool* suddenly catching organic volume. On DEX Screener's "new pairs" and Dune's new-pool queries, a freshly created pool that goes from \$0 to meaningful two-sided volume from many distinct wallets — while passing the wash-trading checks — is often the first on-chain footprint of a token before it is widely known. This is exactly where smart-money and early-buyer tracking (see [early buyer and insider detection](/blog/trading/onchain/early-buyer-and-insider-detection)) overlaps with DEX analysis: the new pool catching real flow is the haystack, and the wallets buying it early are the needles. The risk, of course, is that most new pools are noise or traps, which is why the wash-trading and rug-check filters from the sibling posts run on every one before you act.

## How to read it: the analyst workflow on DefiLlama, Dune, and DEX Screener

Theory is cheap; the edge is the workflow. Here is the concrete pass you run to analyze a DEX, a pool, or a chain's activity, using the three free tools that cover almost everything.

**Step 1 — DefiLlama for the protocol-level overview.** Start at [DefiLlama](https://defillama.com)'s DEX section. It ranks DEXes by volume and, crucially, by *fees* — and it lets you see fees and volume *over time*, per chain. This is where you answer the big questions: Which DEXes are generating the most fees (real usage)? How are volume and fees trending — up, flat, or bleeding? Which chains are gaining DEX volume share? DefiLlama's fees-and-revenue view is the fastest way to separate protocols with real, growing usage from those coasting on past TVL. For any DEX you are evaluating as an investment, this is the top-line read: annualize the fees, compare to the token's valuation, and judge whether the price reflects the usage. That valuation logic is the [on-chain fundamentals](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl) playbook applied to DEXes specifically.

**Step 2 — DEX Screener for the pool and token level.** For a specific token, [DEX Screener](https://dexscreener.com) (or DEXTools) gives you, in one page: every pool the token trades in across every chain, each pool's TVL ("liquidity"), 24-hour volume, and the implied vol/TVL. Sort by liquidity to find the canonical pool. Then run the checks: Is the volume many times the TVL (wash flag)? Is the volume broad or concentrated in a few wallets? Is there a "new pairs" entry catching organic flow? DEX Screener is your fast triage — it answers "is this pool real and busy?" in under a minute.

**Step 3 — Dune for the custom, verifiable query.** When you need to *prove* something — that the volume comes from many distinct addresses, that the fees are real, that a wash ring is cycling funds, that a new pool's buyers are independent wallets — you go to [Dune Analytics](https://dune.com). Dune lets you query the raw decoded on-chain data with SQL and read community dashboards for every major DEX. A typical custom query counts distinct trader addresses per day, the share of volume from the top few addresses, and the fee revenue actually distributed — the exact cross-checks this post is built on. Here is the *shape* of such a query (read-only, illustrative):

```sql
-- distinct traders and volume concentration on a pool, last 7 days
-- (schema names vary by DEX; this is the analysis shape, not a copy-paste)
SELECT
  date_trunc('day', block_time)        AS day,
  count(distinct trader)               AS distinct_traders,
  sum(amount_usd)                      AS volume_usd,
  -- share of the day's volume done by the single busiest trader
  max(trader_volume) / sum(amount_usd) AS top1_share
FROM dex_trades_for_one_pool
WHERE block_time > now() - interval '7' day
GROUP BY 1
ORDER BY 1;
```

A healthy pool shows many distinct traders and a low top-trader share; a wash ring shows few traders and a high concentration. You do not need to be a SQL expert to start — most major DEXes have public Dune dashboards that already compute these, and you fork and tweak.

**Step 4 — net it into a decision.** With volume (verified real), fees (verified organic), vol/TVL (verified healthy), and the rotation context (where the flow is heading), you have everything to act. The next figure is that decision, laid out as the read-action-falsepositive table you will internalize.

![Matrix mapping each DEX metric to its read, its action, and its false positive](/imgs/blogs/analyzing-dex-and-amm-activity-7.png)

The matrix is the workflow compressed: each of the four metrics — volume, fees, vol/TVL, rotation — gets a *read* (what it means once verified), an *action* (what a trader, LP, or analyst does), and a *false positive* (the trap that makes the raw number lie). Volume's trap is wash trading; fees' trap is subsidies; vol/TVL's trap is a wash-driven spike; rotation's trap is mercenary, incentive-driven flow that reverses. Every action is paired with the thing that invalidates it, because the discipline of on-chain analysis is never trusting a green number without its disconfirming check.

## Common misconceptions

**"High volume means high demand."** Only if the volume is real. Volume is the single easiest DEX number to fake, because wash trading — the same entity cycling funds through a pool — manufactures volume at will. A \$10M-volume day on a \$200k pool, done by three addresses, is not \$10M of demand; it is a magic trick. Always check the address spread and the volume-to-TVL ratio before treating volume as a signal, and cross-check against fees earned by *independent* LPs. Volume is a question; fees from outside traders are closer to an answer.

**"More TVL is safer and better."** TVL that is not being used is stranded capital, not strength. A \$50M pool doing \$2M of volume (vol/TVL 0.04) earns its LPs almost nothing and will bleed liquidity as that capital migrates to where it works; a \$10M pool doing \$50M of volume (vol/TVL 5) is the healthier pool for everyone. Read TVL alongside volume, never alone — the productive ratio matters more than the raw size.

**"LPing is passive income."** LPing is a *short-volatility* position dressed up as yield. You collect fees, but you pay impermanent loss whenever the pair diverges, and on a volatile token that IL routinely exceeds the fees. The advertised APR is always before IL; the real return is fees minus IL minus opportunity cost. Providing liquidity for a token you are bullish on is often the worst way to express that view — you would do better just holding it, because the pool sells your winner on the way up.

**"Concentrated liquidity is strictly better than v2."** It is more capital-efficient *while the price stays in your range*, but it amplifies impermanent loss in the same proportion and stops earning fees entirely once the price exits the range — at which point you are holding 100% of the losing asset, earning nothing. Concentrated LPing is an active, attention-intensive job, not a set-and-forget yield. The capital efficiency is real and so is the cost of being wrong about the range.

**"The biggest DEX is where the flow is."** The biggest *yesterday* is not the biggest *tomorrow*. Volume rotates between venues and chains chasing new pairs, incentives, and narratives, and the rotation leads price. A DEX topping the leaderboard purely on token incentives is often a fade — mercenary capital that leaves when the emissions stop — while a venue quietly winning aggregator-routed volume is winning the durable game. Read the trend and the *driver* of the flow, not the snapshot ranking.

## The playbook: what to do with DEX activity

Here is the if-then checklist, by metric, with the signal, the read, the action, and the false positive that invalidates it. This is the section to keep open while you work.

**Volume.**
- *Signal:* rising 24-hour volume on a token or pool.
- *Read:* is it organic? Check many distinct addresses, two-sided flow, sensible sizes, and a vol/TVL that the pair could actually sustain.
- *Action:* organic rising volume is a demand signal — a reason to long, watch, or size into the token early, before the price chart confirms it.
- *Invalidation / false positive:* wash trading. Few addresses, volume many times TVL, round-trip timing, no price impact from large "trades." If the volume is not broad and two-sided, treat it as zero.

**Fees.**
- *Signal:* growing, sustained fee revenue on a DEX or pool.
- *Read:* are the fees *organic* (paid by traders) or *subsidized* (paid by the protocol's token emissions)? Separate the two on DefiLlama and Dune.
- *Action:* sustained organic fee growth is the strongest real-usage signal there is — a fundamentals long on the protocol token, since price often lags real fees. Annualize the fees and compare to the token's valuation.
- *Invalidation / false positive:* subsidies. If the yield is mostly token emissions, it vanishes when the rewards end, and the mercenary capital leaves with it.

**LP decision (fees vs IL).**
- *Signal:* an attractive advertised LP APR.
- *Read:* the APR is before IL. Estimate expected fees (volume × tier × your share) and expected IL (from the pair's likely divergence), and net them.
- *Action:* provide liquidity only when expected fees clearly beat expected IL — which favors high-volume pools of correlated or pegged pairs (stablecoin pairs, ETH/staked-ETH) and punishes thin pools of volatile tokens. For concentrated positions, size for active management and watch the range.
- *Invalidation / false positive:* a divergence you did not price in. If the pair trends hard, IL swamps the fees and you lose to holding. The advertised APR was real; your net return was negative.

**Capital efficiency (vol/TVL).**
- *Signal:* the volume-to-TVL ratio of a pool.
- *Read:* high and stable for the pair type means hard-working, durable liquidity; very low means stranded capital about to leave; impossibly high means wash volume.
- *Action:* as an LP, prefer high-and-stable vol/TVL pools for real yield; as a trader, treat a falling vol/TVL as a warning that depth is leaving and your future exit will be worse.
- *Invalidation / false positive:* a vol/TVL spike that is wash-driven, not efficiency. Verify the underlying volume is real before reading a high ratio as health.

**Rotation (where the flow is).**
- *Signal:* DEX or chain volume share migrating; a new pool catching organic flow.
- *Read:* is the rotation driven by *new pairs and real narrative* (durable) or by *incentives* (mercenary, reversible)? Check whether the volume persists when you mentally remove the rewards.
- *Action:* follow durable rotation early — new chains and new pools are where flow concentrates before prices and headlines catch up. For a new pool, run the wash and rug checks first, then look at who is buying early.
- *Invalidation / false positive:* incentive-driven flow that reverses when emissions stop. Mercenary TVL and volume are a fade as much as a follow — ride them only while the rewards last, and be gone before they end.

Run this loop on DefiLlama (protocol overview and fees), DEX Screener (pool and token triage), and Dune (the verifiable custom query), and DEX activity stops being a wall of green numbers and becomes a readable map of real demand, real revenue, and where the capital is going next. The single rule that ties it together: **a pool doing \$50M a day on \$10M of liquidity is working hard; one doing \$2M on \$50M is lazy capital on a timer — and which one you are looking at decides everything.**

## Further reading & cross-links

- [Reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools) — the single-pool companion: depth, slippage, LP composition, and the `x \* y = k` mechanic this post builds on.
- [On-chain fundamentals: fees, revenue, and TVL](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl) — how to value a protocol from its fees and TVL; the framework behind reading DEX fees as revenue.
- [Detecting wash trading](/blog/trading/onchain/detecting-wash-trading) — the full methodology for separating self-dealt volume from real demand.
- [Detecting fake volume vs organic demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand) — the broader volume-authenticity toolkit, applied beyond DEXes.
- [MEV, sandwiches, and frontrunning](/blog/trading/onchain/mev-sandwiches-and-frontrunning) — why thin, busy pools are where bots extract value, and what that means for your swaps and your LP returns.
- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — the protocol primer on how Uniswap and the AMM model came to be.
