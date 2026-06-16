---
title: "On-Chain Fundamentals: Fees, Revenue, and Valuing a Protocol by Its Usage"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Beyond memes, some tokens back real fee-earning businesses. Learn protocol fees vs revenue, the P/F and P/S ratios, TVL quality, and how to value a protocol by what it actually earns."
tags: ["onchain", "crypto", "defi", "fees", "revenue", "tvl", "valuation", "real-yield", "token-terminal", "defillama"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Some tokens are lottery tickets and some are claims on a real, fee-earning business; on-chain fundamentals are how you tell which is which before the narrative does.
>
> - A productive protocol earns **fees** (what users pay) and keeps **revenue** (the slice that accrues to the protocol after the supply side takes its cut) — that is the on-chain equivalent of a company's top line and operating income.
> - Read those numbers on **Token Terminal** and **DefiLlama**, then compute the **P/F** (price-to-fees) and **P/S** (price-to-revenue) ratios — crypto's version of the P/E multiple — to compare protocols on the same footing.
> - Then do the two checks the multiple hides: is the **TVL sticky or mercenary** (does it leave when emissions stop?), and does the **token actually capture any of the revenue** (buyback/burn, fee share, or nothing)?
> - Rule of thumb: a protocol earning **\$50M in fees at a \$500M market cap is a 10x P/F**; one at **40x P/F** needs to grow four times as fast just to catch up — and if the token captures \$0 of the fees, the multiple is meaningless.

In the first quarter of 2024, a quiet ranking on a website called Token Terminal told a story the price charts had buried. A handful of protocols — a perpetual-futures exchange, a couple of lending markets, a liquid-staking protocol — were each generating tens of millions of dollars a year in fees from people actually *using* them. Not from speculation, not from emissions, but from traders paying to trade and borrowers paying to borrow. One of them, the perps exchange GMX on Arbitrum, was at points distributing more in real fees to its stakers than its token's entire annualized inflation. Meanwhile, on Solana, the launchpad Pump.fun was minting hundreds of thousands of memecoins with no fees, no revenue, and no business behind them — just a casino with a token attached.

Both worlds traded on the same screens, in the same wallets, on the same chains. But only one of them had something underneath the price. The memecoin holder owned a claim on *nothing* — pure greater-fool dynamics. The fee-earning protocol's token holder owned a claim on a cash flow, however small and however contested. The whole discipline of on-chain fundamentals is the toolkit for separating those two cases: for asking, of any token, *does the thing it represents actually earn money, and does the token get any of it?*

This is the serious-investor counterpart to memecoin analysis. Where memecoin work is about survivorship, attention, and exit liquidity, fundamentals work is about cash flow, multiples, and durability. It will not tell you what pumps next week. It *will* tell you whether you own a business or a betting slip — and over a full cycle, that distinction is most of the difference between compounding and getting rugged.

![Diagram showing users paying fees, fees splitting into supply-side share and protocol revenue, and revenue flowing to either token buyback or nothing](/imgs/blogs/onchain-fundamentals-fees-revenue-and-tvl-1.png)

## Foundations: what it means for a protocol to "earn"

Before any ratio, we need the vocabulary. A reader coming from traditional markets knows the words *revenue*, *margin*, *P/E*. A reader coming from crypto knows *TVL*, *APY*, *emissions*. This section builds both into one shared language, from zero.

### A memecoin versus a productive protocol

Start with the cleanest contrast on-chain.

A **memecoin** is a token with no cash flow. There is no business behind it. Nobody pays a fee to "use" DOGE or a freshly launched Solana ticker; the token's only purpose is to be bought and sold. Its price is entirely a function of attention and the willingness of the next buyer to pay more. There is nothing to value because there is nothing being earned — you can only handicap the *odds* (how many holders, how concentrated, how much exit liquidity), which is exactly what memecoin analysis does. It is closer to handicapping a horse than valuing a company.

A **productive protocol** is the opposite: a piece of software that does a job for users and *charges them for it*. A decentralized exchange (DEX) like Uniswap charges a fee on every swap. A lending market like Aave earns a spread between what borrowers pay and what depositors receive. A perpetual-futures venue charges trading fees and funding. A liquid-staking protocol like Lido takes a cut of staking rewards. These are real businesses with real customers paying real money — the money just moves on-chain, in public, where anyone can read it.

That public readability is the whole opportunity. In traditional markets, you wait for a quarterly 10-K, audited and three months stale, to learn what a company earned. On-chain, the "income statement" updates every block. The fees a protocol collected today are visible today, to the dollar, to anyone who knows where to look. You are not at the mercy of management's narrative — you can read the ledger yourself. (For the underlying machinery of how these protocols work — AMMs, lending pools, the mechanics of a swap — see [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).)

### What "fees" actually are

**Protocol fees** are the total amount users pay to use the protocol over some period. If a million dollars of swaps flow through a pool charging 0.30%, the pool generated \$3,000 in fees on that volume. Sum it across every pool, every day, every chain the protocol runs on, and you get the protocol's **total fees** — the gross dollars its users paid. This is the top line. It is the closest on-chain analog to a company's *revenue* in the colloquial sense: the total money that came in the door.

Crucially, fees are denominated in real value — in the tokens being swapped, in ETH, in stablecoins — not in the protocol's own emissions. A fee is money a user *chose* to pay because the service was worth it. That makes total fees the single best proxy for genuine demand: nobody pays a swap fee for fun.

It's worth being concrete about *where* fees come from, because different protocol types earn in structurally different ways, and that shapes everything downstream:

- A **DEX** charges a percentage of swap volume (e.g. 0.05%–1.00% per trade). Fees scale with **volume**, so a DEX's fee line is volatile — it spikes in high-activity weeks and collapses in quiet ones. High volume on thin TVL is the perps signature; high TVL on thin volume is a parked-capital signature.
- A **lending market** earns the spread between borrow and supply rates, plus a reserve cut. Fees scale with **borrowed balances and rates** — a lending protocol earns more when utilization and rates are high, which tends to be in risk-on, leverage-hungry markets.
- A **perps / derivatives venue** earns trading fees plus funding and liquidation fees. Fees scale with **leveraged notional volume**, which can dwarf the actual capital at risk — a little TVL supports enormous fee generation, which is why perps show the highest fees-per-TVL ratios.
- A **liquid-staking protocol** takes a percentage of staking rewards. Fees scale with **assets staked and the underlying staking yield** — the steadiest, most annuity-like fee stream in the space.

Knowing the fee *engine* tells you what to watch: for a DEX, volume; for lending, utilization and rates; for perps, open interest; for staking, assets and yield. A fee number you can't tie to its driver is a number you don't understand yet.

### Fees versus revenue — the distinction that trips up everyone

Here is the most important and most-abused distinction in the entire field, so we will be precise.

**Fees** = the total amount users pay.
**Revenue** = the slice of those fees that accrues to the *protocol itself* (its treasury and/or its token holders) after the **supply side** is paid.

Who is the supply side? In a DEX, it is the **liquidity providers (LPs)** — the people who deposit the two tokens that make a pool tradable. They take the lion's share of swap fees as their reward for providing that liquidity. In a lending market, it is the **depositors**, who earn most of the interest borrowers pay. The protocol keeps only a "take rate" — a fraction of the fees, sometimes called the protocol fee switch or reserve factor.

So the chain of money is: *users pay fees → most of it goes to the supply side (LPs, lenders) → the protocol keeps a slice = revenue.*

![Two-column comparison showing fifty million dollars of fees on the left with forty million to the supply side, versus ten million of protocol revenue on the right](/imgs/blogs/onchain-fundamentals-fees-revenue-and-tvl-2.png)

This sounds like an accounting nuance. It is not. It is the difference between a protocol *looking* like a giant business and *being* a small one. A DEX can route \$50M of fees a year and keep only \$5M of it, because 90% flows straight to LPs. If you value the token off the \$50M, you have overstated its earning power tenfold. The whole game of comparing protocols is comparing them on the *right* number — and which number is "right" depends on what you are asking. Fees measure how much economic activity the protocol commands; revenue measures how much of that activity it monetizes for itself and its holders.

#### Worked example: fees versus revenue on a single DEX

Say a DEX does \$10B in annual swap volume at an average fee of 0.30%. Total fees = \$10,000,000,000 × 0.003 = \$30,000,000. Now suppose the protocol's "fee switch" sends 1/6 of swap fees to the treasury and 5/6 to LPs. Protocol revenue = \$30M × (1/6) = \$5,000,000; LPs keep \$25,000,000. So the same protocol is a "\$30M fee business" and a "\$5M revenue business" simultaneously — both numbers are true, and they answer different questions. A headline that screams "\$30M in fees!" while the token only ever sees \$5M is, technically, not lying — it is just quoting the number that flatters it. *Intuition: fees tell you how big the activity is; revenue tells you how much of it the protocol gets to keep.*

### TVL — the usage proxy, and its honesty problem

**TVL** stands for **Total Value Locked** — the dollar value of all the assets currently deposited in a protocol. It is the size of the LP positions in a DEX, the deposits in a lending market, the collateral in a stablecoin system. TVL is the headline metric of DeFi, plastered across every dashboard, and it is genuinely useful: a protocol with more capital deposited can generally support more activity, so TVL is a rough proxy for *scale and usage*.

But TVL has an honesty problem that we will spend a whole section on later: not all locked value is *committed*. Some of it is there only because the protocol is paying it to be there — bribing deposits with token emissions. That capital ("mercenary capital") leaves the instant the bribe stops. So TVL is a usage proxy the way a restaurant's headcount is a popularity proxy: real, until you learn half the diners are there for the free buffet and will vanish when it ends. Hold that thought.

### Real yield versus emissions-funded yield

The last foundational term. When a protocol pays you to deposit or stake, that yield comes from one of two sources, and they could not be more different.

**Real yield** is paid out of the protocol's actual fee revenue, usually in a "hard" asset — ETH, a stablecoin, or the fees themselves. It is sustainable because it is a *distribution of money the protocol genuinely earned*. If GMX pays its stakers in ETH and USDC drawn from trading fees, that is real yield.

**Emissions-funded yield** is paid by printing new units of the protocol's own token and handing them to depositors. The "yield" is just dilution wearing a costume. A 200% APY that is 100% new token emissions is not income — it is the protocol selling you its own inflation and calling it a reward. The token's price tends to fall as the new supply hits the market, so your dollar return is often far below the headline, and frequently negative. (This is the supply-side mirror of the unlock and emissions problem covered when we talk about [stablecoin flows as the dry-powder metric](/blog/trading/onchain/stablecoin-flows-the-dry-powder-metric) and supply schedules generally.)

The entire fundamentals discipline can be compressed into one instinct: *prefer the thing whose rewards come from fees over the thing whose rewards come from a printing press.*

### The income-statement analogy, made precise

It helps to line the on-chain vocabulary up against a familiar corporate income statement, because the mapping is almost one-to-one once you see it:

- **Gross fees** ≈ a company's **gross revenue / sales** — the total money customers paid.
- **Supply-side fees** ≈ **cost of goods sold** — the direct cost of delivering the service (here, paying LPs and lenders for the capital that makes the service possible).
- **Protocol revenue** ≈ **gross profit** — what's left after the direct cost.
- **Token incentives / emissions** ≈ **stock-based compensation and customer-acquisition spend rolled together** — a real economic cost paid in equity (the token) rather than cash, which dilutes existing holders.
- **Earnings (revenue − incentives)** ≈ **operating income net of stock comp** — the honest bottom line.

The reason crypto analysts obsess over the fees-versus-revenue split is the same reason equity analysts obsess over gross margin: a business with \$100M of sales and a 5% gross margin is a fundamentally smaller, more fragile thing than a business with \$100M of sales and a 60% gross margin, even though the top line is identical. A DEX that keeps 1/6 of fees has a ~17% "take rate"; a lending protocol's reserve factor might be 10%; a perps venue might keep 30–40%. The take rate is the protocol's gross margin, and it is structural — it tells you how much of the activity the protocol can ever monetize, no matter how large the volume grows.

#### Worked example: same fees, very different businesses

DEX P keeps a 10% take rate; perps venue Q keeps a 35% take rate. Both route \$60,000,000 of annual fees. P's revenue = \$60M × 0.10 = \$6,000,000; Q's revenue = \$60M × 0.35 = \$21,000,000. If both trade at a \$300,000,000 market cap, P's P/F is 5x but its P/S is \$300M ÷ \$6M = 50x, while Q's P/F is also 5x but its P/S is \$300M ÷ \$21M ≈ 14x. Identical on fees, Q is ~3.5x cheaper on the number that actually reaches the protocol. *Intuition: the take rate is the on-chain gross margin — two protocols with the same fees can be wildly different businesses depending on how much of the fee they keep.*

### Why "the chain doesn't lie, but it doesn't define your terms either"

One more foundational caution before we read real numbers. The blockchain records every fee paid, every token emitted, every dollar deposited, perfectly and publicly. That is the great advantage of on-chain fundamentals over equity research: there is no quarterly lag, no management adjustment, no "non-GAAP" sleight of hand at the raw-data layer. But the *labels* — which on-chain events count as "fees," which count as "revenue," how emissions are dollar-valued, whether a token sent to a staking contract is "revenue distributed" or "supply-side cost" — are human conventions, and they differ between DefiLlama, Token Terminal, and the protocol's own dashboard. Two honest analysts can read the same chain and report different "revenue" because they drew the line in different places. The discipline, then, is not to trust a single green number but to know *exactly which on-chain flows it sums* and to make sure you're comparing protocols on the same definition. The ledger removes the lying; it does not remove the thinking.

## Reading the income statement: Token Terminal and DefiLlama

Vocabulary in hand, let's read a real protocol's numbers the way an analyst reads a 10-K. There are two indispensable, free tools.

**DefiLlama** is the open, community-run dashboard of choice for TVL, fees, and revenue across nearly every chain and protocol. It is the on-chain Bloomberg terminal for DeFi aggregates, and crucially it documents its methodology in the open — you can click through to see exactly what it counts as a "fee" versus "revenue" for any given protocol.

**Token Terminal** is the more analyst-oriented platform: it standardizes fees, revenue, P/F, P/S, active users, and treasury across protocols so you can compare them on one screen, like a stock screener for on-chain businesses.

### What you are looking at, field by field

When you open a protocol's page, you will see some version of these rows. Here is what each one means and the trap in each:

- **Fees (total).** Gross dollars users paid. The biggest, most flattering number. *Trap:* it includes the supply side's cut, so it overstates what the token earns.
- **Revenue (protocol revenue).** The slice the protocol keeps. *Trap:* definitions vary — always check whether "revenue" means treasury-only or includes fees routed to token holders.
- **Supply-side fees / fees to LPs.** The complement of revenue: what flowed to liquidity providers or depositors. Fees = revenue + supply-side fees.
- **TVL.** Deposits. *Trap:* says nothing about whether they're sticky (more on this below).
- **Active users / transactions.** A demand signal that emissions can't fake as easily as TVL — though wash trading and airdrop farming can, which is why we cross-reference. (See [supply, distribution, and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) for how to read the holder side of the same protocol.)
- **Token incentives / emissions.** The tokens the protocol is printing to attract users and deposits. This is the single most important context line: revenue *minus* incentives tells you whether the protocol is profitable or buying its growth.

### The number that quietly matters most: earnings net of incentives

A protocol that earns \$10M in revenue but spends \$30M in token emissions to attract the deposits that generate that revenue is *losing \$20M a year* in economic terms — it is buying \$1 of revenue for \$3 of inflation. Token Terminal increasingly surfaces this as "earnings" (revenue minus token incentives). It is the on-chain analog of GAAP net income versus adjusted EBITDA: the difference between a business that funds itself and one that funds itself by diluting you.

#### Worked example: a protocol that "earns" but isn't profitable

A lending market reports \$12,000,000 in annual protocol revenue. Lovely. But it also emitted \$28,000,000 of its governance token over the same year as deposit and borrow incentives. Net economic earnings = \$12,000,000 − \$28,000,000 = −\$16,000,000. So despite a healthy-looking revenue line, the protocol is *spending \$2.33 in token dilution for every \$1 of revenue it books*. A naive P/S ratio on the \$12M revenue makes it look like a real business; netting out the \$28M of emissions shows it is renting its growth. *Intuition: revenue minus incentives is the line that tells you whether the business stands on its own or stands on its own printing press.*

## P/F and P/S: crypto's valuation multiples

Now the ratios — the heart of the matter, and the part traditional investors will recognize instantly.

In equities, the **P/E ratio** (price-to-earnings) is market cap divided by annual earnings. A P/E of 20 means you are paying \$20 for every \$1 of annual earnings. On-chain, most protocols don't have clean "earnings" yet (emissions muddy the picture), so the workhorses are:

- **P/F (price-to-fees):** valuation ÷ annualized fees.
- **P/S (price-to-revenue, "sales"):** valuation ÷ annualized revenue.

The lower the multiple, the cheaper the protocol per dollar of usage (P/F) or per dollar of protocol take (P/S). These are the on-chain P/E.

### One critical choice: market cap or FDV?

"Valuation" in the numerator can mean two things, and the gap between them is enormous for young tokens:

- **Market cap** = circulating supply × price (the tokens trading *today*).
- **FDV** (fully diluted valuation) = *total* eventual supply × price (every token that will ever exist, including locked team/investor allocations and future emissions).

For a protocol whose token is mostly still locked and vesting, FDV can be 5–10× the market cap. Valuing it off market cap flatters it; valuing it off FDV is the conservative, honest choice because those locked tokens *will* eventually dilute you. A serious analyst computes the multiple both ways and watches the gap, exactly the way an equity analyst tracks a company's option overhang. The vesting and unlock schedule is its own deep topic — the supply that is *about* to hit the market is as important as the supply already trading.

#### Worked example: the 10x protocol versus the 40x protocol

Protocol A earns \$50,000,000 a year in fees and trades at a \$500,000,000 market cap. P/F = \$500M ÷ \$50M = **10x**. Protocol B earns the same \$50,000,000 in fees but trades at a \$2,000,000,000 market cap. P/F = \$2,000M ÷ \$50M = **40x**. On identical fees, you are paying four times as much for B's dollar of usage. For B to be the better buy at that price, its fees must *grow* roughly four times faster than A's — and keep doing so — just to close the gap. If both grow at the same rate, A compounds your capital and B re-rates *down* toward A. *Intuition: the multiple is the price of a dollar of usage; pay 40x only when you have a concrete reason the usage will quadruple faster than the cheap one's.*

![Horizontal bar chart of illustrative price-to-fees multiples for four protocols ranging from ten times to one hundred eighty times](/imgs/blogs/onchain-fundamentals-fees-revenue-and-tvl-3.png)

### How to compare across protocols (and how not to)

The multiple is only meaningful *within a comparable set*. A few rules:

1. **Compare like with like.** A DEX's P/F is not directly comparable to a lending market's, because their take rates and growth profiles differ structurally. Rank DEXs against DEXs, perps against perps, lending against lending. Cross-category comparisons are eyeballing, not analysis.

2. **Annualize honestly.** A protocol that did \$10M in fees last *month* is not a \$120M-fee business if last month was a once-a-cycle volume spike. Use a trailing window (often a 30-day annualized *and* a trailing-12-month figure) and note when they diverge — divergence is information about whether the business is spiking or steady.

3. **Adjust for growth.** A 30x P/F on fees doubling every quarter can be cheaper than a 10x P/F on fees shrinking 20% a quarter. This is the on-chain PEG ratio (P/F divided by growth). The cheap-looking dying protocol is the classic value trap.

4. **Adjust for value capture and TVL quality.** A 10x P/F means nothing if the token captures \$0 of those fees, or if the fees come from mercenary TVL that's about to leave. These two adjustments are so important they each get their own section next.

### A note on P/S and the "fee switch"

Many protocols generate large fees but deliberately keep their *revenue* near zero — they route nearly everything to the supply side to stay competitive, holding a "fee switch" in reserve that governance could one day flip to divert more to the treasury. Uniswap spent years like this: enormous fees, essentially zero protocol revenue, with a long-running governance debate about turning the switch on. For such protocols, P/F captures the *potential* (the size of the pie the protocol could one day tax) while P/S captures the *realized* (what it currently takes). The gap between them is an option on governance — valuable, but not yet cash. Price it as optionality, not as income.

### Three real episodes that taught the market these lessons

The reason this framework exists is that the market learned each piece of it the hard way, on a specific date, with real money. Three episodes are worth knowing.

**The 2020–2021 "DeFi summer" emissions farms.** When yield farming exploded in mid-2020, dozens of protocols bootstrapped enormous TVL by emitting governance tokens at triple- and quadruple-digit APYs. TVL on the protocols offering the richest emissions ballooned into the billions within weeks. Then, protocol by protocol, the emissions tapered — and the TVL evaporated almost as fast as it had arrived, because nearly all of it was mercenary. The canonical pattern, repeated across "food coin" farms (the SushiSwap "vampire attack" on Uniswap being the most famous), was: emit aggressively, watch TVL spike, watch the token price crater under the new supply, watch the farmers rotate out, and end up with a fraction of the deposits and a token down 90%+. This episode is *why* the sticky-versus-mercenary distinction is the first thing a serious analyst checks on any TVL number.

**GMX and the rise of "real yield" (2022).** In the bear market that followed the 2022 collapses, the market's appetite for emissions-funded farms collapsed with it — burned investors wanted yield that came from somewhere real. The perps DEX GMX on Arbitrum became the poster child for "real yield": it paid its stakers a share of actual trading fees in ETH (and its liquidity providers in the basket of assets they backed), denominated in hard assets, not in freshly printed GMX. At points, the protocol's annualized fee distributions to stakers were large relative to its market cap, and "real yield" became a rallying cry for a cohort of protocols that earned their distributions rather than printed them. The episode crystallized the distinction between a yield that is a *distribution of earnings* and a yield that is a *distribution of inflation* — and made "what is the yield paid in, and where does it come from?" a standard question.

**The Uniswap fee-switch debate (2022–2024).** Uniswap is the largest DEX by volume and has generated billions of dollars in cumulative fees — essentially all of which went to liquidity providers, with the protocol (and the UNI token) capturing zero. The "fee switch" — a governance lever that could divert a fraction of fees to the protocol/token — sat unflipped for years amid debate over legal risk, LP competitiveness, and how to distribute the proceeds. For that whole period, UNI was the textbook case of a token whose protocol earned enormous fees while the token itself captured *nothing*: a pure governance option on a someday-decision. Whatever the eventual resolution, the multi-year saga is the cleanest real-world illustration of why "does the token capture value?" is a separate question from "does the protocol earn fees?" — the answer to the first can be *no* even when the answer to the second is an emphatic *yes*.

#### Worked example: pricing a fee-switch option

Suppose a DEX generates \$200,000,000 a year in fees, of which the token currently captures \$0 (all to LPs). The token's market cap is \$3,000,000,000. On *current* economics the P/S is undefined (revenue is zero) — the token is pure governance. Now suppose the market assigns a 25% probability that governance flips a switch sending 1/5 of fees to the token: expected future token revenue = \$200M × (1/5) × 0.25 = \$10,000,000 of probability-weighted annual revenue, which against the \$3B cap is an implied forward P/S of 300x. That is a rich price to pay for an *option* — it only makes sense if you think the switch is far more likely than 25% or the fees will grow enormously. *Intuition: a fee switch is an option, so price it like one — probability times payoff — never as if the cash were already flowing.*

## TVL quality: sticky versus mercenary

Now the metric everyone quotes and almost nobody adjusts. We said TVL is a usage proxy with an honesty problem. Here is the problem in full.

When a new protocol launches, the fastest way to bootstrap deposits is to *pay* for them: emit your governance token to anyone who deposits. Offer a 100% APY in your token and capital floods in — but most of that capital is **mercenary**. It is run by farmers whose only loyalty is to the highest yield; they will deposit, harvest the emitted tokens, sell them, and rotate to the next farm the moment your emissions taper. This capital inflates your TVL while it's there and evaporates when the music stops. It was never *using* your protocol; it was *renting* your incentives.

The opposite is **sticky** (or organic) TVL: deposits that are there because users genuinely want to use the protocol — they need the liquidity, they trust the lending market, they'd stay even if you paid them nothing extra. Sticky TVL is the real franchise. Mercenary TVL is a number on a chart.

![Two-column comparison showing one billion dollars of headline TVL with eight hundred million mercenary, collapsing to two hundred million real TVL after emissions stop](/imgs/blogs/onchain-fundamentals-fees-revenue-and-tvl-4.png)

### How to tell them apart on-chain

You can't read "stickiness" off a single number, but several on-chain tells expose mercenary TVL:

- **Fees per dollar of TVL.** Sticky capital is *used* — it generates fees. If a protocol has \$1B in TVL but trivial fees, that capital is parked, not working; it's there for the emissions, not the service. High fees relative to TVL is the signature of real usage.
- **TVL versus emissions overlay.** Plot TVL against the token-incentive schedule. If TVL tracks emissions up and down in lockstep, it's mercenary by definition. If TVL holds when emissions are cut, it's sticky. The historical "emissions cliff" — what happened to TVL the last time rewards were reduced — is the single most revealing test.
- **Depositor concentration and behavior.** A handful of large wallets that deposit, harvest, and withdraw in tight cycles around emission epochs are farmers. Many small, long-duration deposits look like real users. (This is exactly the clustering and behavioral work covered in [following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) and [the perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain).)

#### Worked example: the \$1B TVL that's really \$200M

A yield protocol proudly displays \$1,000,000,000 in TVL. But it's running an aggressive emissions program, and on-chain you find that ~80% of deposits arrived within days of the rewards launching and are concentrated in wallets that harvest-and-dump the emitted token weekly. Estimate the mercenary share at 80%: real, sticky TVL ≈ \$1,000,000,000 × (1 − 0.80) = **\$200,000,000**. The protocol is paying, say, \$40,000,000 a year in emissions to hold the \$800,000,000 of rented capital. The day those \$40M of emissions stop, expect TVL to fall toward \$200M — and any valuation that priced the protocol off the \$1B headline just lost 80% of its denominator. *Intuition: TVL you have to pay for isn't a moat, it's a subscription — cancel the payment and it cancels itself.*

#### Worked example: fees-per-TVL as the lie detector

Protocol X has \$2,000,000,000 in TVL and \$8,000,000 in annual fees. Fees/TVL = \$8M ÷ \$2,000M = 0.4% — almost none of that capital is doing anything. Protocol Y has \$400,000,000 in TVL and \$40,000,000 in annual fees. Fees/TVL = \$40M ÷ \$400M = 10% — that capital is intensely used. Y's "smaller" TVL is worth far more than X's, because Y's deposits are *working* and X's are *parked for the airdrop*. If you ranked these two by TVL alone you'd pick X; ranked by fees-per-TVL you'd correctly pick Y. *Intuition: TVL counts the capital; fees-per-TVL tells you whether the capital is actually being used — always divide.*

### The second-order effects of incentivized TVL

Mercenary TVL is not just a number that overstates a protocol's size — it actively distorts the protocol's economics in ways that bite later. First, it creates a **reflexive trap**: high emissions attract TVL, which attracts users, which generates fees, which seems to justify the token price, which justifies more emissions — a flywheel that runs beautifully in reverse the moment emissions slow, because every link in the chain was partly propped by the emission. Protocols that grew this way often discover that their "organic" fee base was smaller than it looked, because some of the fee-generating activity was itself farmers churning to qualify for rewards (a cousin of wash trading).

Second, incentivized TVL **misprices the cost of capital**. A protocol paying \$40M a year to hold \$800M of deposits is paying a 5% cost of capital for capital it doesn't really need — and that \$40M is value transferred from token holders (via dilution) to transient farmers. The token holder is, in effect, subsidizing the headline TVL number that the token holder is then asked to pay a premium for. It is circular and value-destructive, and reading it requires putting the emissions figure *next to* the TVL figure, never looking at either alone.

Third, the **emissions cliff is predictable**. Most incentive programs have a published schedule — a halving of rewards on a known date, a fixed emission budget that will run out, a governance vote to cut incentives. An analyst who reads that schedule knows roughly when the mercenary TVL will face its exit decision, and can watch the prior cliffs as a natural experiment: what fraction of TVL stayed last time rewards were cut is the best available estimate of the sticky share. This is the single most useful, most overlooked piece of on-chain diligence on any incentivized protocol.

#### Worked example: what the emissions cliff reveals

A protocol cut its daily emissions by half on a known date. In the four weeks before the cut, TVL averaged \$600,000,000. In the four weeks after, it settled at \$360,000,000. The TVL that left = \$600M − \$360M = \$240,000,000, or 40% of the deposits — that 40% was mercenary capital tied to the *marginal* emissions just removed. Extrapolating, if the protocol cut emissions to zero, you'd expect a further large outflow; the sticky core is well below the current \$360M. The cliff is a free, real-world stress test of TVL quality, and the market hands it to you on the chain. *Intuition: don't guess the mercenary share — find the last emissions cut and measure how much capital walked.*

## Does the token capture value? The "does it do anything" question

Here is the question that quietly kills more crypto theses than any other. You've found a protocol with real fees, real revenue, sticky TVL, a reasonable P/F. You're ready to buy the token. One more question: *when the protocol earns money, does the token get any of it?*

Astonishingly often, the answer is no. A token can govern a protocol that prints money while capturing exactly \$0 of that money for its holders. The fees pile up in a treasury the token can't touch; the token is a "governance" token that lets you vote but never pays you. You own a vote in a profitable company that pays no dividend, can't buy back its own stock, and can't be acquired. That vote has *some* value — control is worth something, and governance could one day turn on a value-accrual mechanism — but it is not a cash flow, and it should never be valued like one.

![Three-by-three matrix mapping buyback-burn, fee-share, and treasury accrual designs against whether the token earns and its fundamental claim](/imgs/blogs/onchain-fundamentals-fees-revenue-and-tvl-5.png)

### The three value-accrual designs

Protocols capture (or fail to capture) value for their tokens in three broad ways:

1. **Buyback-and-burn.** The protocol uses revenue to buy its own token on the open market and destroy it, permanently reducing supply. Every dollar of revenue retires a dollar's worth of token, so revenue is a continuous bid under the price and a continuous reduction of the float. This is the cleanest "the token earns" design — economically equivalent to a buyback in equities. MakerDAO's surplus buffer and various "burn" mechanisms work this way.

2. **Direct fee share to stakers.** Holders who stake the token receive a cut of fees, paid in a hard asset (ETH, stablecoins) or in the fees themselves. This is real yield: the token is a claim on a revenue stream, and you can compute its yield directly. GMX paying stakers in ETH/USDC, or a lending protocol sharing reserve income with stakers, are examples.

3. **Nothing (treasury accrual / pure governance).** Fees accrue to a protocol treasury or to the supply side, and the token holder receives no direct economic benefit — only a governance vote. The token's value rests entirely on the *expectation* that governance will someday flip a fee switch or that control itself commands a premium. Sometimes that expectation pays off; often it's a story that never resolves.

There are subtler variants worth recognizing. A **veToken** model (vote-escrowed, popularized by Curve) lets holders lock their tokens for boosted rewards and a share of fees and "bribes" — a hybrid of fee-share and governance that can capture real value but also concentrates power and invites bribe markets, so you must trace where the value actually lands. A **treasury that buys back but doesn't burn** accumulates the token rather than retiring it, which supports price while the buying continues but leaves an overhang if the treasury ever sells. And some tokens capture value *indirectly* — a stablecoin issuer's token might accrue value from the issuer's float income rather than from on-chain fees. The point of the taxonomy is not to memorize labels but to force the question to a concrete answer: *trace one dollar of revenue and write down exactly where it ends up.* If you can't draw that arrow to the token holder, the token does not capture value, whatever the marketing says.

### Why this single check reorders your whole ranking

Two protocols with identical \$50M fees and identical 10x P/F are *not* identically valuable if one burns 100% of revenue into its token and the other sends 0% to holders. The first token is a claim on \$50M of value capture; the second is a claim on a vote. The P/F is the same; the *thing the P/F is a multiple of* is completely different. This is why value-accrual is not a footnote — it can flip your entire ranking. A "more expensive" token at 20x P/F that burns 100% of fees may be far cheaper, in claim-on-cash-flow terms, than a "cheap" 8x token that captures nothing.

#### Worked example: \$50M in fees, \$0 to the token versus 100% burned

Token N's protocol earns \$50,000,000 in revenue a year, all of which accrues to a treasury the token cannot access; holders get a governance vote and nothing else. Token N's market cap is \$500M — a 10x P/F. Token B's protocol earns the same \$50,000,000, and uses 100% of it to buy back and burn Token B. Token B's market cap is also \$500M — also a 10x P/F. On the screen they look identical. But Token B holders are receiving the economic equivalent of a \$50,000,000 / \$500,000,000 = **10% buyback yield**, retiring 10% of the float a year at the current price, while Token N holders receive **\$0** of cash flow and own only an option on a future governance decision. At the same 10x P/F, Token B is a business and Token N is a bet. *Intuition: the multiple tells you the price; the value-accrual mechanism tells you whether there's anything behind the price at all.*

## Real yield versus emissions-funded yield, in pictures

We defined the two kinds of yield in the foundations. Now let's see why the distinction is so financially violent, because the headline APYs lie in opposite directions.

A protocol advertising "200% APY" on its staking page is almost always paying that yield in freshly emitted tokens. As those tokens are emitted and sold, the token price falls; your 200% in *token terms* can easily be −40% in *dollar terms* once dilution is accounted for. The yield is real in units and illusory in value. By contrast, a protocol paying a humble-looking 8% in stablecoins from actual fee revenue gives you 8% in dollars, full stop — and it can keep paying it as long as the fees keep coming, without diluting anyone.

![Line chart comparing the value of a one thousand dollar stake under real yield growing steadily versus emissions-funded yield bleeding below the starting amount](/imgs/blogs/onchain-fundamentals-fees-revenue-and-tvl-6.png)

#### Worked example: 120% emissions APY versus 12% real yield over a year

Put \$1,000 into a real-yield protocol paying 12% in stablecoins from fees: after a year you have roughly \$1,000 × 1.12 = **\$1,120**, in actual dollars, with the principal token's value unaffected because nothing was diluted. Now put \$1,000 into an emissions farm advertising 120% APY, paid in a token whose price falls ~7.5% a month as the emissions hit the market. The rewards pile up in token units, but each unit is worth less every month; netting the ~120% of new tokens against ~60% of cumulative price decay, the stake is worth roughly **\$830** after a year — you *lost* about \$170 despite the triple-digit headline. The "120%" was a number printed on inflation. *Intuition: a yield you can spend in dollars beats a yield you can only spend in a token that the yield itself is debasing.*

The deeper point: real yield is a *distribution* (the protocol earned money and gave you some), while emissions yield is a *dilution* (the protocol printed money and gave you some of the printing). The first transfers value *to* you; the second transfers value *among* holders and mostly *away* from the patient ones. Always ask what currency the yield is paid in and where that currency comes from.

## How to value a protocol: a walkthrough

Let's put it all together into a concrete, repeatable workflow — the on-chain analyst's version of building a quick valuation. We'll run it on a hypothetical perps DEX so every number is explicit, but the steps are exactly what you'd do on DefiLlama and Token Terminal for a real one.

### Step 1 — Pull the raw numbers

Open the protocol on DefiLlama and Token Terminal and record:

- Trailing-30-day fees, annualized, *and* trailing-12-month fees (note any divergence).
- Protocol revenue (and confirm whether it includes fees routed to holders).
- TVL, and the token-incentives/emissions figure over the same period.
- Circulating market cap *and* FDV.

For our perps DEX, suppose: annualized fees = \$80,000,000; protocol revenue = \$24,000,000 (30% take rate, 70% to LPs); TVL = \$300,000,000; emissions = \$10,000,000/yr; market cap = \$480,000,000; FDV = \$720,000,000.

### Step 2 — Compute the multiples both ways

P/F on market cap = \$480M ÷ \$80M = **6.0x**. P/F on FDV = \$720M ÷ \$80M = **9.0x**. P/S on revenue (market cap) = \$480M ÷ \$24M = **20x**; on FDV = \$720M ÷ \$24M = **30x**. So the protocol is 6–9x fees and 20–30x revenue depending on dilution. The FDV figures are the honest floor — use those for the decision.

#### Worked example: the full multiple stack for the perps DEX

Fees \$80,000,000, revenue \$24,000,000, FDV \$720,000,000. P/F (FDV) = \$720M ÷ \$80M = 9.0x. P/S (FDV) = \$720M ÷ \$24M = 30x. Earnings net of emissions = \$24,000,000 − \$10,000,000 = \$14,000,000, so the *economic* P/E on FDV = \$720M ÷ \$14M ≈ **51x**. Notice how the multiple climbs as you move from the flattering number (9x on gross fees) to the honest one (51x on emissions-adjusted earnings): the same protocol is "cheap" on fees and "richly priced" on real earnings. *Intuition: always walk the multiple from fees → revenue → earnings-net-of-emissions; the honest valuation is the one furthest down that ladder.*

### Step 3 — Check value accrual

Does the token capture the \$24M revenue? Suppose this DEX shares 60% of revenue with stakers in ETH/USDC and sends 40% to a treasury. Stakers' real-yield pool = \$24M × 0.60 = \$14,400,000. Against a \$480M market cap that's a real-yield of \$14.4M ÷ \$480M = **3.0%**, paid in hard assets — modest but genuine. The token *does* something. (If the answer had been "0% to holders," we'd stop and reprice the whole thing as a governance option, per the section above.)

### Step 4 — Check TVL quality

Fees-per-TVL = \$80M ÷ \$300M = 27% — extremely high, which is normal for perps (a little capital supports enormous notional trading volume) and a strong sign the TVL is *used*, not parked. Overlay emissions: \$10M/yr of incentives against \$24M of revenue means the protocol earns more than it emits — it is not renting its deposits. Sticky.

### Step 5 — Adjust for growth and render a verdict

Suppose fees grew 40% year-over-year. A 9x P/F (FDV) on a business growing 40% with genuine 3% real yield, sticky TVL, and positive earnings-net-of-emissions is a *cheap, real* business — the kind of setup that compounds. Contrast it with a peer at 40x P/F, flat fees, 0% value accrual, and emissions exceeding revenue: same "DEX" label, opposite investment. The workflow turned two superficially similar tokens into a clear buy and a clear avoid.

![Four-row decision matrix mapping price-to-fees, TVL quality, value accrual, and growth against good signs and red flags](/imgs/blogs/onchain-fundamentals-fees-revenue-and-tvl-7.png)

## Common misconceptions

**"High TVL means a healthy protocol."** TVL is deposits, not earnings, and much of it can be rented with emissions. A \$1B-TVL protocol with \$5M in fees is mostly parked, incentive-chasing capital; a \$200M-TVL protocol with \$40M in fees is a far better business. Always divide fees by TVL before being impressed by a TVL number.

**"Fees and revenue are the same thing."** Fees are the gross top line (what users pay); revenue is the slice the protocol keeps after the supply side. A DEX can have \$50M of fees and \$5M of revenue. Valuing the token off the wrong one over- or under-states its earning power by an order of magnitude. Confirm which number a dashboard is showing before you build a multiple on it.

**"A low P/F is always cheap."** A low multiple on shrinking fees, with a token that captures nothing, sitting on mercenary TVL, is a value trap, not a value. The multiple is only the first of four checks; cheapness without growth, value accrual, and stickiness is a falling knife.

**"High APY means high returns."** Headline APY is usually paid in emitted tokens whose price the emissions themselves are crushing. A 200% token-APY can be a negative dollar return; an 8% stablecoin yield from real fees is a positive one. Always ask what the yield is paid in and where it comes from.

**"On-chain fundamentals are exact because the data is public."** The *raw* data is public and precise, but the *definitions* are not standardized — what counts as "revenue" differs across DefiLlama, Token Terminal, and the protocol's own dashboard, and emissions accounting is genuinely hard. The public ledger removes management spin; it does not remove the need to read methodology footnotes carefully. (For how to build a repeatable rubric out of these inputs, the natural next step is a structured token scorecard that scores each protocol on the same axes every time.)

## The playbook: what to do with it

The signal-to-action checklist for valuing a protocol by its usage. Run it top to bottom; a single red flag doesn't kill a thesis, but two or three should.

1. **Signal: the token is pitched as backed by a "real business."** → **Read:** open DefiLlama and Token Terminal; pull fees, revenue, TVL, emissions, market cap, and FDV. → **Action:** if there are no fees yet (early protocol), this is a *narrative/venture* bet, not a fundamentals one — size it as a speculation and say so. → **False positive:** "revenue" that's actually emissions recycled through the protocol; check the methodology.

2. **Signal: the protocol earns real fees.** → **Read:** compute P/F and P/S on *both* market cap and FDV; annualize honestly (30-day and trailing-12-month). → **Action:** rank it against true peers (DEX vs DEX, perps vs perps); a lower multiple on faster, steadier growth is the target. → **False positive:** a fee spike from one anomalous month annualized into a fake run-rate — always check the trend, not the peak.

3. **Signal: the multiple looks cheap.** → **Read:** check value accrual — does the token burn, fee-share, or capture nothing? → **Action:** only treat the multiple as a claim-on-cash-flow if the token actually receives value; if it captures \$0, reprice it as a governance option and demand a far lower entry. → **False positive:** a "fee switch" that *could* be turned on — that's optionality, not income; don't price it as cash.

4. **Signal: TVL is large.** → **Read:** compute fees-per-TVL and overlay TVL against the emissions schedule; look at depositor behavior around emission epochs. → **Action:** discount mercenary TVL toward zero (it leaves when emissions stop); value the protocol on its *sticky* base and on fees, not on headline deposits. → **False positive:** a one-time TVL surge from a points/airdrop campaign that reverses the week after the snapshot.

5. **Signal: a high advertised yield.** → **Read:** identify what the yield is paid in and where the money comes from — fees (real) or emissions (dilution). → **Action:** treat real, hard-asset yield as income you can underwrite; treat emissions yield as a depreciating coupon and model the token-price decay. → **False positive:** a "real yield" that's really the protocol routing emissions through a staking contract — trace the source asset.

6. **Signal: everything checks out — cheap, growing, sticky, value-accruing.** → **Read:** confirm earnings net of emissions is positive and the FDV (not just market cap) multiple is reasonable. → **Action:** this is the rare on-chain *business* worth holding through a cycle; size it accordingly and re-check the fee trend each month. → **Invalidation:** fees roll over for two consecutive quarters, the team turns *off* a value-accrual mechanism, or emissions outgrow revenue — any one flips the thesis.

### The limits, stated honestly

A few hard limits deserve to be named before you lean on any of this, because over-trusting fundamentals fails in its own distinctive ways.

*Early protocols have no fees, and that's not a verdict.* A protocol launched three months ago with negligible fees is not "expensive at infinite P/F" — it is simply pre-revenue, and this entire toolkit is silent on it. Valuing such a token requires venture-style judgment about future usage, team, and market, not a multiple. The mistake is to either dismiss every pre-revenue token as worthless or to back-fit a fundamentals story onto what is really a narrative bet. Call it what it is and size it accordingly.

*Definitions and data quality vary, and emissions accounting is genuinely hard.* Dollar-valuing emissions requires a token price that the emissions themselves are moving, so the "earnings net of incentives" line is an estimate with a fat error bar, not a precise figure. Cross-check at least two sources and treat the *direction and magnitude* as reliable while distrusting the last significant digit.

*Multiples are only comparable within a category and a regime.* Perps trade at different multiples than lending in the same month, and everything trades at different multiples in a bull market than a bear. A 30x P/F that looked rich in the 2022 trough might be the cheapest perps token in a 2025 frenzy. Anchor your comparison to current peers, not to a number you remember from a different cycle.

*The market can ignore fundamentals for a very long time.* This is the limit that hurts most. A token can trade at 100x P/F for an entire bull run because the story is winning, and a cheap, real business can stay cheap for many quarters because no one is paying attention. Fundamentals give you a floor and a reason to hold through a drawdown; they do not give you timing. If you need the multiple to compress on a schedule, you will be disappointed and probably stopped out before the thesis plays.

The honest limit, stated plainly: fundamentals are necessary but not sufficient, and they work on a *delay*. Early protocols have no fees, so this entire toolkit says nothing about them — there, you are making a venture bet on future usage, and you should know that's what you're doing. And even for mature protocols, narratives can dominate fundamentals for a long time: a token can trade at 100x P/F for an entire bull market because the *story* is winning, and a cheap, real business can stay cheap for quarters because nobody's paying attention. Fundamentals tell you what you *own* and give you a floor to lean on when the narrative breaks; they do not tell you what pumps next week. Used that way — as the thing that keeps you holding a real business through a drawdown and keeps you from holding a vote in a treasury you can't touch — they are the most durable edge in a market built mostly on stories.

## Further reading & cross-links

- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — the mechanics of the AMMs and lending markets whose fees we just learned to read.
- [Stablecoin flows: the dry-powder metric](/blog/trading/onchain/stablecoin-flows-the-dry-powder-metric) — the capital that funds the deposits and pays the fees, read as a macro gauge.
- [Supply, distribution, and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — the holder-side companion to the cash-flow side covered here.
- [Following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) — how to read the depositor behavior that distinguishes sticky from mercenary TVL.
- [The perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) — why "smart money" in a protocol can be farmers, not believers.
- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — where DefiLlama and Token Terminal sit among the rest of the on-chain toolkit.
- [Crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) — the top-down view on why protocol fees rise and fall with the broader liquidity tide.
