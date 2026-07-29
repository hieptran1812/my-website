---
title: "Whales, Smart Money, and On-Chain Wallet Watching: What the Ledger Can and Cannot Tell You"
date: "2026-07-29"
publishDate: "2026-07-29"
description: "A build-from-zero guide to crypto whales — why size is measured against order-book depth rather than total supply, how Nansen, Arkham and Etherscan actually build their wallet labels, every way that data lies, and the honest arithmetic on what edge is left once you are twenty minutes late."
tags: ["crypto", "whales", "smart-money", "on-chain-analysis", "wallet-tracking", "market-impact", "liquidity", "order-book", "amm", "labeling", "copy-trading", "crypto-players", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 54
---

> [!important]
> **TL;DR** — A "whale" is not someone who owns a lot of coins. A whale is someone whose position is large *relative to the liquidity standing in front of it*. That is a statement about the order book, not about the balance sheet.
>
> - **Depth, not supply, is the denominator.** A holder of 5,000 BTC owns about 0.025% of the circulating supply — a rounding error — while simultaneously holding nearly five times the entire bid stack sitting within 2% of the price at a major venue. Both facts are true. Only the second one predicts anything.
> - **Automated market makers are far thinner than they look.** A constant-product pool advertising \$3 million of total value locked holds only about **\$15,200** of liquidity within 2% of its price. Selling an amount equal to the pool's own token reserve costs you almost exactly half your money.
> - **"Smart money" is a backward-looking profit screen, not a forecast.** Rank ten thousand coin-flipping wallets by their last twenty trades and roughly **207** of them will show a 75% win rate purely by luck. The label describes what already happened.
> - **The chain records custody, not trades.** Coins arriving at an exchange deposit address means the owner bought optionality. Selling is one of at least six reasons that happens, and the rest of the story is invisible off-chain.
> - **The edge decays in minutes.** On illustrative but realistic assumptions, a followed wallet's +8.8% per-trade expectancy is worth roughly **zero** to someone twenty minutes late, and about **−4.15%** after a 4% round-trip cost.

Every few weeks a chart does the rounds on crypto social media. A big red candle, an arrow, and a caption: *whale moved 3,000 BTC to Binance, dump incoming.*

The chart is usually real. The transfer usually happened. And the inference is usually wrong — not because the person posting it is lying, but because they are quietly making four assumptions that the blockchain does not support. They assume the address belongs to one person. They assume the person controls it. They assume a deposit is a sale. And, most importantly, they assume that the size is large *in a way that matters*, when the only thing that determines whether size matters is a number that is not on the chain at all.

This post builds the whole thing from zero: what a whale actually is in market-impact terms, how the tools that label wallets really work, every documented way those labels break, what happens once the person being watched knows they are being watched, and — with the arithmetic shown in full — how much edge is left at the end.

![Two panels: the transfers, swaps, bridges and deposits that are visible on-chain, and the order book, internal ledger transfers, OTC blocks and identities that are not](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-1.webp)

The diagram above is the mental model for everything that follows. The left side is what a blockchain is: a public log of *custody changing hands* — this address now controls coins that used to sit at that address. The right side is what actually sets prices: an order book, matched inside an exchange's own database, where nothing is ever written to a chain. There is exactly one bridge between them, the exchange deposit address, and the moment coins cross it your visibility ends. Almost every mistake in wallet-watching is a claim about the right-hand side made using only left-hand-side data.

## Foundations: the building blocks

Nothing in this section assumes you have ever looked at a block explorer. If you already have, skim to the worked example.

### An address is a lock, not a person

A blockchain **address** is a short string like `0x` followed by forty hexadecimal characters (on Ethereum) or a string beginning `bc1` (on Bitcoin). It is derived from a **public key**, which is in turn derived from a **private key** — a secret number. Whoever holds the private key can authorize spending from that address. That is the entire security model.

Three consequences follow immediately, and every one of them matters later:

1. **An address is not an identity.** Nothing in the protocol records who generated the key. Attribution — deciding that an address "belongs to" someone — is always an *inference layered on top of the chain*, never a fact read out of it.
2. **One person can have unlimited addresses.** Generating a new one costs nothing and takes microseconds. There is no registry, no limit, and no way to tell from the chain alone whether two addresses share an owner.
3. **One address can serve many people.** When you deposit coins to an exchange, you generally stop owning coins. You own a *database row* at that exchange saying you are owed some. The coins themselves get pooled into exchange-controlled wallets — **omnibus** wallets, in the industry term, meaning a single account holding many customers' assets commingled. An exchange wallet holding 200,000 BTC is not a whale. It is a hundred thousand people in a trenchcoat.

A **wallet**, informally, means either a piece of software that manages your keys or — the sense used throughout this post — a cluster of addresses believed to share one owner. Note the word "believed."

### The order book, in one paragraph

Most crypto trading happens on a **central limit order book (CLOB)**. Buyers post **bids** (offers to buy at a price) and sellers post **asks** (offers to sell). The highest bid and lowest ask define the **spread**, and the midpoint between them is the **mid price** — the number quoted as "the price." A **limit order** rests in the book waiting; a **market order** executes immediately against whatever is resting there, taking the best price first, then the next best, and so on. That last mechanic — *walking the book* — is where whales live. If you want the fuller treatment, [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) covers the microstructure in detail.

Two measurements matter and neither appears on a token's price page:

- **Depth** — the total value of resting orders within some distance of the mid. The industry convention is **1% depth** or **2% depth**: how many dollars of bids sit within 1% or 2% below the mid price. This is the number that decides whether your sell order is a ripple or a wave.
- **Average daily volume (ADV)** — how much actually trades in 24 hours. Note that volume and depth are different things: volume is *flow* over time, depth is *stock* at a moment. A market can have enormous volume and terrible depth if the same liquidity is being recycled thousands of times per day, which is exactly what a fast market maker does. Some reported volume is not real at all; [wash trading, spoofing, and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) covers how that number gets faked.

### Balance is the wrong yardstick; depth is the right one

Here is the definition this entire post rests on:

> A whale is not a holder whose balance is large relative to supply. A whale is a holder whose *intended trade* is large relative to the liquidity standing in front of it.

Supply is the denominator that circulates in headlines because it is easy to compute and requires no market data. It is also nearly useless. The relevant comparison is between the size someone wants to move and the depth available to absorb it.

#### Worked example 1: the same holder, measured two ways

Suppose someone holds **5,000 BTC**, and suppose for the arithmetic that BTC trades at **\$100,000** — a round illustrative price, so the multiplication stays legible. The position is worth **\$500 million**.

**Measured against supply.** With roughly **20.06 million BTC** in circulation as of 29 July 2026 — about 96% of the 21 million that will ever exist ([Blockchain.com](https://blockchain.info/q/totalbc)) — this holder owns

$$
\frac{5{,}000}{20{,}060{,}000} = 0.000249 \approx 0.025\%
$$

of all bitcoin that exists. Two and a half hundredths of one percent. On that basis the position is trivial — there are index funds that own a larger share of the average listed company.

**Measured against depth.** Now use the illustrative order book we will walk in a moment, where **\$104 million** of bids sit within 2% of the mid at one major venue. The same position is

$$
\frac{\$500{,}000{,}000}{\$104{,}000{,}000} = 4.8\times
$$

the entire bid stack within 2%. Even if you assume the aggregate book across all major venues is three times deeper than one venue's — call it \$312 million — the position is still **1.6 times** the whole two-percent book of the entire market.

**The intuition:** the same holder is 0.025% of the asset and 480% of the market for it. Supply tells you nothing about impact; depth tells you everything.

![A three-by-four grid comparing 2 percent depth, 24-hour volume, and the slippage of a 1.5 million dollar and a 200 million dollar sell across BTC spot, a mid-cap altcoin pair, and a thin DEX pool](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-2.webp)

The grid makes the same argument three times. A \$1.5 million sell is a **0.05%** event on a major BTC pair, a **1.0%** event on a mid-cap altcoin, and a **50.1%** catastrophe in a three-million-dollar liquidity pool. Identical dollars. The difference is entirely in the denominator, and the denominator is a market-structure fact that no amount of on-chain data will hand you.

This is why "whale alert" accounts that report transfers in dollar terms are, on their own, close to noise. Ten million dollars moving is either nothing or everything depending on where it is pointed.

## 1. Market impact from first principles: walking the book

Let's make the mechanism concrete. When you send a market sell order, the exchange fills you against resting bids in price order — best first. Your **execution price** is the size-weighted average of every level you consumed, and it is always worse than the mid you saw before you traded. That gap is **slippage**.

<figure class="blog-anim">
<svg viewBox="0 0 680 372" role="img" aria-label="A 2,000 BTC market sell consuming seven bid levels one at a time, from 99,950 dollars down to 95,400 dollars, for an average fill of 97,654 dollars" style="width:100%;height:auto;max-width:760px">
<style>
.w1-hdr{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.w1-vwap{font:700 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:end}
.w1-px{font:600 13px ui-monospace,SFMono-Regular,monospace;fill:var(--text-primary,#1f2937);text-anchor:end}
.w1-sz{font:400 12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280)}
.w1-note{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.w1-bar{fill:var(--border,#d1d5db);rx:4}
.w1-rule{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:3 4}
.w1-mark{stroke:var(--accent,#6366f1);stroke-width:2.5}
@keyframes w1-eat{0%{fill:var(--border,#d1d5db)}7%,93%{fill:var(--accent,#6366f1)}100%{fill:var(--border,#d1d5db)}}
@keyframes w1-drop{0%{transform:translateY(0)}100%{transform:translateY(228px)}}
.w1-bar{animation:w1-eat 9s ease-in-out infinite}
.w1-b2{animation-delay:.55s}.w1-b3{animation-delay:1.1s}.w1-b4{animation-delay:1.65s}
.w1-b5{animation-delay:2.2s}.w1-b6{animation-delay:2.75s}.w1-b7{animation-delay:3.3s}
.w1-mark{animation:w1-drop 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.w1-bar{animation:none;fill:var(--accent,#6366f1)}.w1-mark{animation:none;transform:translateY(228px)}}
</style>
<text class="w1-hdr" x="105" y="24">mid $100,000 — illustrative bid ladder, deepest fills last</text>
<text class="w1-vwap" x="672" y="24">VWAP $97,654</text>
<line class="w1-rule" x1="105" y1="34" x2="672" y2="34"/>
<text class="w1-px" x="98" y="70">$99,950</text>
<rect class="w1-bar" x="105" y="52" width="80" height="26"/>
<text class="w1-sz" x="193" y="70">80 BTC</text>
<text class="w1-px" x="98" y="108">$99,800</text>
<rect class="w1-bar w1-b2" x="105" y="90" width="140" height="26"/>
<text class="w1-sz" x="253" y="108">140 BTC</text>
<text class="w1-px" x="98" y="146">$99,500</text>
<rect class="w1-bar w1-b3" x="105" y="128" width="200" height="26"/>
<text class="w1-sz" x="313" y="146">200 BTC</text>
<text class="w1-px" x="98" y="184">$99,000</text>
<rect class="w1-bar w1-b4" x="105" y="166" width="280" height="26"/>
<text class="w1-sz" x="393" y="184">280 BTC</text>
<text class="w1-px" x="98" y="222">$98,200</text>
<rect class="w1-bar w1-b5" x="105" y="204" width="350" height="26"/>
<text class="w1-sz" x="463" y="222">350 BTC</text>
<text class="w1-px" x="98" y="260">$97,000</text>
<rect class="w1-bar w1-b6" x="105" y="242" width="450" height="26"/>
<text class="w1-sz" x="563" y="260">450 BTC</text>
<text class="w1-px" x="98" y="298">$95,400</text>
<rect class="w1-bar w1-b7" x="105" y="280" width="500" height="26"/>
<text class="w1-sz" x="613" y="298">500 BTC</text>
<line class="w1-mark" x1="100" y1="65" x2="620" y2="65"/>
<line class="w1-rule" x1="105" y1="320" x2="672" y2="320"/>
<text class="w1-note" x="105" y="344">2,000 BTC sold at market — $200M nominal, 2.35% slippage, $4.69M shortfall</text>
<text class="w1-note" x="105" y="364">Last print $95,400 — 4.6% below the mid one second earlier</text>
</svg>
<figcaption>A single 2,000 BTC market sell eats seven bid levels in turn; the average fill lands 2.35% below the $100,000 mid, and the last print is 4.6% below it.</figcaption>
</figure>

#### Worked example 2: a 2,000 BTC market sell

Take an illustrative bid ladder on one venue with the mid at \$100,000. Round numbers, chosen to be legible rather than realistic to the dollar, but the *shape* — thin at the top, fatter as you go down, gapping out at the bottom — is how real crypto books look.

| Bid price | Size (BTC) | Cumulative BTC | Cumulative proceeds |
| --- | --- | --- | --- |
| \$99,950 | 80 | 80 | \$7,996,000 |
| \$99,800 | 140 | 220 | \$21,968,000 |
| \$99,500 | 200 | 420 | \$41,868,000 |
| \$99,000 | 280 | 700 | \$69,588,000 |
| \$98,200 | 350 | 1,050 | \$103,958,000 |
| \$97,000 | 450 | 1,500 | \$147,608,000 |
| \$95,400 | 500 | 2,000 | \$195,308,000 |

Now sell 2,000 BTC at market. You consume every level:

- 80 × \$99,950 = \$7,996,000
- 140 × \$99,800 = \$13,972,000
- 200 × \$99,500 = \$19,900,000
- 280 × \$99,000 = \$27,720,000
- 350 × \$98,200 = \$34,370,000
- 450 × \$97,000 = \$43,650,000
- 500 × \$95,400 = \$47,700,000

Total proceeds: **\$195,308,000**. Divide by 2,000 BTC:

$$
\text{VWAP} = \frac{\$195{,}308{,}000}{2{,}000} = \$97{,}654
$$

**VWAP** is *volume-weighted average price* — the average price you actually got. Against the \$100,000 mid you started from:

$$
\text{slippage} = \frac{100{,}000 - 97{,}654}{100{,}000} = 2.346\% \approx 2.35\%
$$

In cash, the difference between what the screen said your coins were worth and what you received is

$$
2{,}000 \times (\$100{,}000 - \$97{,}654) = \$4{,}692{,}000
$$

**\$4.69 million, gone into the spread**, on a nominal \$200 million sale. And the last print — the number everyone sees on the chart, the number that triggers liquidations and stop losses — is \$95,400, **4.6% below** where the market was one second earlier.

**The intuition:** the "price" of your position is the price of the *next* coin you can sell, not the average of all of them. Size destroys its own exit.

### Two refinements that matter in practice

**The book refills.** Real books are not static. Market makers replenish bids within milliseconds, so a patient seller working the same 2,000 BTC over an hour pays far less than \$4.69 million. This is exactly why professionals use **TWAP** (time-weighted average price) and **VWAP** execution algorithms that slice a parent order into hundreds of children, and why they use over-the-counter desks that never touch the book at all. [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price) walks through that machinery.

**Impact grows roughly with the square root of size.** Across equities, futures and crypto, the empirical regularity that keeps reappearing is that the price impact of an order scales approximately as

$$
\Delta P \propto \sigma \sqrt{\frac{Q}{V}}
$$

where $Q$ is your order size, $V$ is the market's daily volume and $\sigma$ its volatility. The practical reading: **doubling your size does not double your cost — it multiplies it by about 1.41.** But equally, halving your size only saves you 29%. There is no order size small enough to be free, and the penalty for being clumsy is milder than intuition suggests while the penalty for being enormous is worse than linear over short horizons. Treat the square-root law as a scaling intuition, not a pricing model; the constant in front of it is venue-specific and changes with the regime. The canonical empirical statement of it is Tóth et al., *Anomalous price impact and the critical nature of liquidity in financial markets* (**Physical Review X**, 2011, [arXiv:1105.1694](https://arxiv.org/abs/1105.1694)), and it is worth knowing that the exponent is contested — other studies fit powers meaningfully different from one half.

## 2. The other kind of book: constant-product pools

Half of on-chain activity does not happen on an order book at all. It happens in an **automated market maker (AMM)** — a smart contract holding two assets and quoting a price from a formula rather than from resting orders. The dominant formula, used by [Uniswap v2](https://developers.uniswap.org/docs/get-started/concepts/how-uniswap-works) and its many descendants, is the **constant product** rule:

$$
x \cdot y = k
$$

where $x$ is the amount of one token in the pool, $y$ is the amount of the other, and $k$ is a constant that trades cannot change. The **spot price** is simply the ratio of the reserves, $p = y / x$. When you sell tokens into the pool you increase $x$; to keep the product $k$ fixed, $y$ must fall — so the pool hands you fewer of the second asset per unit as your trade gets bigger. That curvature *is* the price impact.

Uniswap v2 charges a **0.30%** fee, taken off your input before the swap math runs. That fee is not a parameter in a document somewhere — it is hard-coded as the constants 997 and 1000 in the router's own [`getAmountOut`](https://raw.githubusercontent.com/Uniswap/v2-periphery/master/contracts/libraries/UniswapV2Library.sol), which is the function that prices every v2 swap:

```solidity
// UniswapV2Library.sol
function getAmountOut(uint amountIn, uint reserveIn, uint reserveOut)
    internal pure returns (uint amountOut)
{
    uint amountInWithFee = amountIn.mul(997);
    uint numerator      = amountInWithFee.mul(reserveOut);
    uint denominator    = reserveIn.mul(1000).add(amountInWithFee);
    amountOut = numerator / denominator;
}
```

Written out, selling $\Delta x$ tokens returns

$$
\Delta y = \frac{997 \cdot \Delta x \cdot y}{1000 \cdot x + 997 \cdot \Delta x}
$$

![A constant-product curve with the start and end reserve points marked, the chord between them labelled as the execution price, and the wedge between chord and curve shaded as the price impact](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-3.webp)

The figure is the whole idea. Your trade moves the pool along the curve from point A to point B. The **spot price** is the slope of the curve at a point — it is only ever true for an infinitely small trade. What you actually receive is the **chord** between A and B: the average price along the path. The wedge between the chord and the curve is your cost, and it widens quadratically as the trade grows relative to the pool.

#### Worked example 3: selling into a \$3 million pool

Take an illustrative pool holding **1,000,000 TOKEN** and **1,500,000 USDC**. The spot price is

$$
p = \frac{1{,}500{,}000}{1{,}000{,}000} = \$1.50 \text{ per TOKEN}
$$

and the pool's advertised **total value locked (TVL)** — both sides added together — is \$1.5 million + \$1.5 million = **\$3,000,000**. That is the number that appears on the analytics dashboard.

A holder sells **1,000,000 TOKEN** — nominally \$1,500,000 at spot. After the 0.30% fee, the effective input is 997,000, so:

$$
\Delta y = \frac{997{,}000 \times 1{,}500{,}000}{1{,}000{,}000 + 997{,}000} = \frac{1{,}495{,}500{,}000{,}000}{1{,}997{,}000} = \$748{,}873
$$

Their execution price is \$748,873 ÷ 1,000,000 = \$0.7489 per TOKEN. Against the \$1.50 they saw on the screen:

$$
\text{price impact} = 1 - \frac{0.7489}{1.50} = 50.1\%
$$

They expected \$1,500,000 and received \$748,873. **\$751,127 evaporated** — not stolen, not front-run, just paid to the curve.

And look at what happened to the market. The pool now holds 2,000,000 TOKEN and \$751,127, so the new spot price is

$$
p' = \frac{751{,}127}{2{,}000{,}000} = \$0.376
$$

The quoted price fell **75%** on a single transaction. The token's "market cap" — spot price times total supply — fell by three quarters because one person sold one and a half million dollars.

**The intuition:** in a constant-product pool, selling an amount equal to the pool's own reserve of that token costs you almost exactly *half* your money and quarters the quoted price. There is no size discount and no one to negotiate with.

### The rule of thumb worth memorizing

Ignore the fee for a moment. The ratio of your execution price to spot in a constant-product pool is exactly

$$
\frac{p_{\text{exec}}}{p_{\text{spot}}} = \frac{1}{1 + \Delta x / x}
$$

so your **price impact is approximately your trade size divided by the pool's reserve of the token you are selling**, plus the fee. Check it: selling 10,000 TOKEN (1% of the reserve, \$15,000 at spot) into the same pool gives $\Delta y = \$14{,}807$, an execution price of \$1.4807 and an impact of **1.28%** — which is the 0.99% the formula predicts plus the 0.30% fee. Selling 100 times more produces 39 times the impact rate. The cost per dollar traded is not constant; it accelerates.

### Why "TVL" flatters a pool by two orders of magnitude

Here is the number that should change how you read a dashboard. How much can you sell into that \$3 million pool before the quoted price falls just 2%?

Since $p = k / x^2$, a quoted price ratio of 0.98 requires $x'/x = 1/\sqrt{0.98} = 1.01015$. So you can add just **1.015%** of the reserve before the quoted price falls 2% — that is 10,150 TOKEN, worth

$$
10{,}150 \times \$1.50 \approx \$15{,}200
$$

**A pool advertising \$3,000,000 of liquidity has about \$15,200 of it within 2% of the price.** That is a ratio of roughly 200 to 1 between the headline and the usable depth. Set that against the \$104 million of 2% depth in our BTC book and the pool is about **6,800 times shallower** where it actually counts — while its dashboard headline of \$3,000,000 is only about 35 times smaller than that \$104 million. The advertised number understates the real gap by two orders of magnitude.

This single fact explains most of what people find mysterious about small-cap token charts: why they gap, why a \$50,000 sell prints a 6% candle, why "the whale only sold \$200,000 and it dropped 40%."

### The v3 wrinkle

Uniswap v3 and its successors let liquidity providers concentrate their capital in a chosen price range instead of spreading it along the whole curve. Inside an active range this makes the pool dramatically deeper than the v2 formula predicts. Outside it, the pool can be **empty** — a trade that pushes price out of every provider's range falls off a cliff with no liquidity at all beneath it.

The practical consequence for wallet-watching: **you cannot infer impact from TVL on a v3-style pool in either direction.** A \$3 million v3 pool tightly concentrated around spot may absorb a \$500,000 trade at 0.3% impact; the same pool with liquidity parked 30% away is functionally a \$0 pool. You have to read the actual tick-level liquidity distribution. Any "impact estimate" that starts from a TVL number and ends at a percentage is, for v3, a guess dressed as arithmetic.

## 3. How wallet labels are actually made

Everything above is about size. The rest of the post is about identity — deciding whose coins those are — and this is where the real error bars live.

There is no identity layer on a public blockchain. Every label you have ever seen on Etherscan, Nansen, Arkham, Dune or a Bubblemaps graph is produced by one of five techniques, and they are not equally reliable. [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution) goes deeper on the mechanics; here is the structural picture.

### Deposit-address clustering: the workhorse

This is the single most productive heuristic in the industry, and it exists because of how exchanges are built.

When you open an account at an exchange and ask for a deposit address, the exchange generates a fresh address just for you. Then, periodically — hourly, daily, at a balance threshold — it **sweeps** the balances from thousands of these deposit addresses into a small number of consolidated hot wallets, because managing a million tiny UTXOs or a million gas-funded accounts is operationally miserable.

That sweep is the tell. An analyst who sees ten thousand addresses all forwarding their entire balance to the same destination, on a regular schedule, with no other activity, can conclude with high confidence that all ten thousand are deposit addresses of one entity — even without knowing which entity. One deposit made by a known party (deposit \$10 to your own Binance account and watch where it goes) then names the whole cluster.

![Six user deposit addresses sweeping into one hot wallet to form an exchange cluster, plus two false-positive branches: a payment processor that also sweeps, and a whale sweeping their own sub-wallets](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-4.webp)

The two red outcomes in the figure are the part nobody puts in the marketing material. The heuristic does not detect *exchanges*. It detects *sweeping behaviour*. A payment processor sweeps. A custodian sweeps. A bridge's relayer sweeps. A gambling site sweeps. A sufficiently organized individual sweeps their own sub-wallets. All of them can be pulled into a cluster that gets labelled with the name of whichever entity someone identified first, and the mislabel then propagates because downstream tools consume upstream labels.

### The UTXO heuristics

On Bitcoin and other **UTXO** chains — where a transaction consumes discrete "unspent transaction outputs" rather than debiting an account balance — there are two classic clustering rules, formalized by [Meiklejohn et al. in 2013](https://dl.acm.org/doi/10.1145/2504730.2504747):

- **Common-input-ownership.** If a transaction spends several UTXOs as inputs simultaneously, whoever built it must have held the private key for every one of them. Therefore those addresses share an owner. This is powerful, cheap, and *wrong* in exactly one important case: **CoinJoin** and similar collaborative transactions, which deliberately combine inputs from many unrelated people to break the assumption.
- **Change-address detection.** Bitcoin transactions rarely spend a UTXO exactly; the remainder comes back to a fresh "change" address the sender controls. A cluster of heuristics (round-number outputs, script-type matching, address reuse patterns) guesses which output is change and adds it to the owner's cluster. Every one of those heuristics is beatable, and modern wallet software beats several of them by default.

### Funding-source inference

The weakest of the widely used techniques, and the most common source of confident nonsense. The logic: *this fresh wallet was funded from a wallet we already labelled "Jump Crypto", therefore this wallet is Jump Crypto.*

Sometimes that is right. Often it is an employee, a counterparty who got paid, a market-making client, a borrower, an OTC buyer, or simply someone who received a transfer. The chain shows a payment; a payment is not an ownership claim. When you see a label described as "linked to" or "associated with" an entity rather than "belonging to" it, this is usually the technique underneath.

### ENS, manual research, and crowd submissions

The remaining sources are human. **ENS** (Ethereum Name Service) names — `vitalik.eth` and the like — are self-assigned and genuinely informative, because setting one requires controlling the address. Manual labels come from OSINT: court filings, bankruptcy exhibits, hacked databases, official disclosures, a screenshot in a Discord, a project publishing its treasury address. And crowd-submitted tags are contributed by users. Etherscan is admirably direct about how its public name tags work: you suggest one through a contact form, and Etherscan curators then "evaluate the owner's interest in displaying their address publicly or whether the address is of public interest." That is *curation*, not verification — and Etherscan says so itself, warning in the same documentation that "it may be later reported that a user was falsely claiming to own a specific address." 

![A five-tier tree of label evidence from self-disclosed or provable at the top down to crowd-submitted tags at the bottom](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-5.webp)

Put them in order and the practical rule falls out:

| Tier | Evidence | Example | How it breaks |
| --- | --- | --- | --- |
| 1 | Self-disclosed or cryptographically provable | Proof-of-reserves address, signed message, verified contract source, ENS record | Barely ever — but the entity chose what to disclose, and disclosed addresses are rarely the interesting ones |
| 2 | Structural heuristic | Deposit-sweep clustering, common-input-ownership | CoinJoin, shared custody, any non-exchange that also sweeps |
| 3 | Funding-source inference | "Funded by a labelled fund, therefore that fund" | Employees, counterparties, borrowers, customers |
| 4 | OSINT and manual research | Court filing, bankruptcy exhibit, forum post, screenshot | Stale after a key rotation; often true once and never rechecked |
| 5 | Crowd-submitted tag | Etherscan public name tag, bounty submission | Anyone can submit; rarely re-verified; incentives favour volume |

**Most labels you will ever act on live in tiers 3 to 5.** That is not a scandal — it is the honest state of a field trying to attach names to a system designed not to have them. It just means "Arkham says this is X" is a *hypothesis with a provenance*, and your job before trading on it is to find out which tier it came from.

The correct way to write about any of this, incidentally, is the way the analysts themselves write about it: *a wallet labelled X by Arkham*, not *X's wallet*. The distinction is not pedantry. It is the difference between a claim you can defend and one you cannot.

## 4. What "smart money" actually measures

"Smart Money" is a product feature, not a fact about the world. Nansen popularized the term, and the mechanics across every vendor that offers something similar are broadly the same: take the universe of addresses, compute realized profit and loss over a trailing window, apply thresholds for profitability and activity, and attach a label to the top slice. Some variants add category labels for funds, high-performing liquidity providers, or wallets that were early to tokens that later ran.

Read that description again and notice what it is. It is a **screen on past returns**. It contains no forward-looking information by construction. It tells you which wallets *were* right, over a specific window, on a specific set of trades, net of nothing.

Three problems follow, and they compound.

### Problem one: selection on noise

If you rank enough wallets by trailing performance, the top of the list is populated by skill *and* by luck, and you cannot tell them apart from the ranking alone. This is not a subtle statistical point; the magnitudes are enormous.

#### Worked example 4: how many lucky wallets does a screen manufacture?

Imagine a universe of **10,000 wallets** that have no skill whatsoever. Each makes 20 trades. Each trade is a coin flip: 50% chance of a win, 50% of a loss. Nobody in this universe knows anything.

Now screen for wallets with a **75% win rate or better** — 15 wins out of 20. That sounds like a serious track record. How many of our pure coin-flippers clear it?

The probability of at least 15 heads in 20 flips is

$$
P(X \ge 15) = \frac{\binom{20}{15} + \binom{20}{16} + \binom{20}{17} + \binom{20}{18} + \binom{20}{19} + \binom{20}{20}}{2^{20}}
$$

$$
= \frac{15{,}504 + 4{,}845 + 1{,}140 + 190 + 20 + 1}{1{,}048{,}576} = \frac{21{,}700}{1{,}048{,}576} = 2.07\%
$$

Multiply by 10,000 wallets:

$$
10{,}000 \times 0.0207 \approx \mathbf{207 \text{ wallets}}
$$

**207 wallets with a 75% hit rate, in a universe containing zero skill.** Raise the bar to 80% (16 of 20) and you still get 59 of them. A perfect 20-for-20 is rarer — about 1 in a million — so at a 10,000-wallet scale a flawless record would genuinely mean something.

But the real screening universes are not 10,000 addresses. Nansen [describes](https://docs.nansen.ai/llms-full.txt) its Smart Money product as "a curated list of the top 5,000 highest-performing wallets ranked by realised profit, winrate, and strong performance across market cycles," drawn from a label corpus the company says spans hundreds of millions of blockchain addresses. Suppose the pool with enough trading history to be screened at all is only one million. Then **20,695** of them clear the 75% bar on luck alone — four times the size of the entire published list — and roughly **one** posts a perfect 20-for-20 record having never known anything at all.

Notice also what is *not* published. There is no disclosed profit threshold, no win-rate cutoff, no stated lookback window, no minimum trade count and no refresh cadence in the public documentation. That is not a criticism — a vendor is entitled to its methods — but it does mean you cannot compute how much of the list is selection noise. You are being asked to take the ranking on trust, and worked example 4 is the reason that matters.

**The intuition:** a leaderboard built by ranking a large population on a short history is a machine for manufacturing false skill. The screen is not lying to you; you are asking it the wrong question.

The defence is the same as in traditional fund analysis: demand a longer track record, demand more trades, and demand that the returns survive an out-of-sample period *after* the wallet was labelled. That last test is the only one that matters, and it is the one almost nobody runs.

### Problem two: the label describes a strategy that may be over

Even a genuinely skilled wallet earned its returns under specific conditions. A wallet that made 400% farming a particular airdrop cycle is labelled "smart" for the following ninety days, during which the airdrop cycle it exploited no longer exists. A wallet that made its money as a liquidity provider is labelled from realized PnL that may be dominated by fee income you cannot replicate without their capital. A wallet that was early to three memecoins may be the deployer's friend, which is a relationship, not a skill.

### Problem three: crowding

The moment a wallet is labelled, its trades stop being private information. If ten thousand people receive an alert when it buys, the price the wallet gets and the price its followers get are different prices — and the gap is exactly the followers' loss. We will put numbers on this shortly.

None of this makes smart-money labels useless. They are an excellent *filter for attention*: a way of deciding which of the ten million daily transactions deserve a human look. They are a terrible *signal for execution*. [What is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain) develops the taxonomy further.

## 5. Every way the data lies to you

Here is the failure catalogue, in rough order of how often it produces a wrong public claim.

### The omnibus problem

The largest "wallets" on every chain are exchanges and custodians. Their balances move constantly for reasons that have nothing to do with anyone's market view: customer deposits and withdrawals, cold-to-hot rebalancing, wallet migrations after a security upgrade, chain migrations, proof-of-reserves snapshots. A headline reading "the third-largest bitcoin wallet just moved 50,000 BTC" is, the overwhelming majority of the time, an exchange doing housekeeping.

The tell is usually visible if you look: the funds arrive at another address in the same cluster, the net position of the entity is unchanged, and nothing hits a trading venue. The reason the tell often gets missed is that a transfer of 50,000 BTC is a much better post than a transfer of 0 BTC.

### Internal transfers read as sells

Related but distinct. Large entities hold assets across many venues and many custodians, and they move inventory between them constantly for margin, settlement, and operational reasons. A market maker moving 3,000 ETH from a custodian to an exchange is almost always **funding inventory to quote with**, not liquidating. Their business model requires holding balances at every venue they make markets on; [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) explains why their on-chain footprint looks like constant enormous activity in both directions with no directional view at all.

![A wide fan showing the six common reasons a thousand bitcoin arrives at an exchange deposit address, with only the first one being an actual sale](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-6.webp)

### "Moved to exchange" is not "sold"

This deserves its own treatment because it is the single most common inference in retail wallet-watching, and it is a probabilistic claim being made as a certainty. When coins arrive at an exchange deposit address, the owner has bought **optionality**. Selling is one thing that optionality buys. So is:

- posting the coins as **collateral for a loan**, so the owner keeps their exposure and gets cash;
- posting **margin for a hedge** — depositing spot to short the perpetual against it, which is directionally *neutral*, not bearish;
- a **custody migration** between two venues the same entity controls;
- **market-maker inventory** funding, as above;
- the **settlement leg of an OTC trade** that was already agreed at a fixed price and will never touch the order book.

Only the first is a sale. And in the last case the price impact already happened — or rather, deliberately did not happen, because the entire point of the OTC desk is that a block trades at a negotiated price without walking anyone's book.

#### Worked example 5: what a deposit is actually worth as a signal

Assume you keep a disciplined log for a quarter — and note that the counts below are **illustrative**, chosen to show you the shape of the calculation you must run on your own data, not results of a study. You record every deposit of 1,000 BTC or more from a labelled non-exchange wallet to an exchange deposit address. You get **40 events**. You follow each one for 30 days and resolve it:

| Outcome | Count | Share |
| --- | --- | --- |
| Consistent with a sale | 11 | 27.5% |
| Withdrawn again within 30 days | 14 | 35.0% |
| Identified as a custody migration | 9 | 22.5% |
| Unresolved | 6 | 15.0% |

So your base rate is

$$
P(\text{sale} \mid \text{large deposit}) = \frac{11}{40} = 27.5\%
$$

Now price the trade. Suppose the average 24-hour move when it *was* a sale is **−2.8%**, and when it was not, the average move is **+0.4%** (there is a mild positive drift because coins arriving at an exchange are often about to become collateral for a leveraged long). The expected move after any deposit is

$$
0.275 \times (-2.8\%) + 0.725 \times (+0.4\%) = -0.77\% + 0.29\% = -0.48\%
$$

Shorting captures that, so a rule of "short on every large deposit" earns **+0.48%** per signal before costs. With a 0.35% round-trip cost on a perpetual — fees plus spread — you net **+0.13%** per signal. Across 40 signals a year that is about **5.2%** gross. Real, but thin.

Now stress it. Suppose one label in four is wrong, so your true sale rate is 20% rather than 27.5%:

$$
0.20 \times (-2.8\%) + 0.80 \times (+0.4\%) = -0.24\%
$$

Net of the same 0.35% cost, that is **−0.11% per signal** — a losing strategy. The edge and the error bar on your own labelling are the same size.

**The intuition:** the honest output of a deposit alert is a *probability*, and when you write that probability down the trade stops looking obvious.

### The trail simply ends

Everything above assumes you can see the relevant activity. Very often you cannot.

<figure class="blog-anim">
<svg viewBox="0 0 720 250" role="img" aria-label="Coins travelling from a whale wallet through a bridge to a exchange deposit address and hot wallet, then fading out as they cross into the off-chain order book" style="width:100%;height:auto;max-width:760px">
<style>
.w2-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5;rx:8}
.w2-dark{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:6 5;rx:8}
.w2-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.w2-dim{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.w2-zone{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.w2-edge{stroke:var(--border,#d1d5db);stroke-width:2}
.w2-wall{stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:5 5}
.w2-dot{fill:var(--accent,#6366f1)}
@keyframes w2-run{0%{transform:translateX(0);opacity:0}5%{opacity:1}62%{opacity:1}78%{opacity:.35}100%{transform:translateX(582px);opacity:.06}}
.w2-dot{animation:w2-run 10s linear infinite}
.w2-d2{animation-delay:3.33s}.w2-d3{animation-delay:6.66s}
@media (prefers-reduced-motion:reduce){.w2-dot{animation:none;transform:translateX(250px);opacity:1}}
</style>
<text class="w2-zone" x="290" y="24">visible on-chain</text>
<text class="w2-zone" x="645" y="24">off-chain — nothing is written to any ledger</text>
<line class="w2-wall" x1="563" y1="34" x2="563" y2="196"/>
<text class="w2-zone" x="563" y="214">visibility ends</text>
<rect class="w2-box" x="8" y="80" width="110" height="60"/>
<text class="w2-lbl" x="63" y="115">Whale wallet</text>
<line class="w2-edge" x1="118" y1="110" x2="143" y2="110"/>
<rect class="w2-box" x="143" y="80" width="85" height="60"/>
<text class="w2-lbl" x="185" y="115">Bridge</text>
<line class="w2-edge" x1="228" y1="110" x2="253" y2="110"/>
<rect class="w2-box" x="253" y="80" width="135" height="60"/>
<text class="w2-lbl" x="320" y="106">CEX deposit</text>
<text class="w2-lbl" x="320" y="124">address</text>
<line class="w2-edge" x1="388" y1="110" x2="413" y2="110"/>
<rect class="w2-box" x="413" y="80" width="125" height="60"/>
<text class="w2-lbl" x="475" y="106">Exchange</text>
<text class="w2-lbl" x="475" y="124">hot wallet</text>
<line class="w2-edge" x1="538" y1="110" x2="580" y2="110"/>
<rect class="w2-dark" x="580" y="80" width="130" height="60"/>
<text class="w2-dim" x="645" y="106">Order book</text>
<text class="w2-dim" x="645" y="124">matching engine</text>
<circle class="w2-dot" cx="63" cy="110" r="8"/>
<circle class="w2-dot w2-d2" cx="63" cy="110" r="8"/>
<circle class="w2-dot w2-d3" cx="63" cy="110" r="8"/>
<text class="w2-zone" x="290" y="240">every hop is a public transaction</text>
<text class="w2-zone" x="645" y="240">the trade that sets the price</text>
</svg>
<figcaption>On-chain you can follow the coins to the exchange's door and no further — everything that actually determines the price happens on the other side of the dashed line.</figcaption>
</figure>

Centralized exchange trading is invisible. Two counterparties matching inside Binance's matching engine produce no on-chain record whatsoever. Neither do internal ledger transfers between two customers of the same exchange, or an OTC block negotiated over Telegram and settled by a database update at a custodian. A holder can sell their entire position without a single byte hitting a blockchain, provided the coins were already on the exchange. Conversely, an entity can be enormously active on-chain while having no market view at all.

The estimate that matters is roughly this: the coins are visible, the *trades* are not, and the trades are what set the price.

### False clusters and stale labels

Clusters merge in error and, once merged, rarely un-merge. A label attached in 2021 to a fund that has since wound down, been acquired, rotated keys, or had its addresses seized will happily sit on a dashboard in 2026 with no indication of its age. Bankruptcy estates are a particularly rich source: an address labelled with a defunct firm's name may now be controlled by a trustee liquidating on a court-mandated schedule, which is a completely different behavioural model from the one the label implies.

And there is the reverse failure: **wrapped and derivative exposure**. An entity that appears to have sold all its ETH may have simply staked it, wrapped it, deposited it into a lending market, or moved it to a chain your tool does not index. Multi-chain visibility is uneven, and "the balance went to zero" and "they sold" are different claims.

## 6. The reflexive game: watching the watchers

Now the part that makes wallet-watching genuinely different from reading a balance sheet.

A company's 10-K does not change because you read it. A wallet does. **The moment an address is publicly labelled, its owner learns that everything they do is broadcast in advance to a crowd of people who will trade against it.** They are not passive. They adapt.

![A two-by-two matrix of whether a wallet is publicly labelled against whether its owner splits or obfuscates, with the four resulting outcomes](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-8.webp)

The adaptations are well known and cheap to execute:

**Splitting across wallets.** Instead of one address holding \$200 million, thirty addresses hold \$6-7 million each, funded through paths that do not obviously connect. This defeats balance-ranking entirely and degrades funding-source inference, at the cost of gas and operational complexity.

**Fresh wallets per position.** A new address, funded through a bridge or a mixer-adjacent route or simply through an exchange withdrawal, used for one trade and then abandoned. By the time the wallet has enough history to be labelled, the trade is over. This is the single most effective countermeasure and it costs almost nothing.

**Decoy transfers.** Moving funds in patterns that look like the prelude to a trade that never comes. A deposit to an exchange followed by a withdrawal three days later costs a network fee and produces a false alert for every follower — and, more usefully, degrades their base rates, because now a larger share of the deposits in their log resolve as "withdrawn again."

**Deliberate telegraphing.** The inverse: an entity that *wants* observers to see a move can make it maximally visible. There are legitimate reasons — a foundation announcing a treasury operation in advance, a fund signalling a long-term commitment, a project demonstrating that a token unlock went to a custodian rather than to an exchange. There are also reasons that are not legitimate, and the boundary between "communicating a position" and "manufacturing a reaction" is one that securities regulators spend a lot of time on in other markets. I am describing a structural possibility here, not alleging that any specific person or firm has done it.

**Executing where visibility is lowest.** OTC desks, internal exchange transfers, and off-chain settlement all move size without producing a public trail. An entity that is being watched on-chain has a strong incentive to move its activity to venues where it is not.

### The extreme case: when positions are public by design

There is a category of venue where this reflexivity is not theoretical but structural — decentralized perpetual exchanges that settle on-chain, where **open positions, entry prices, and liquidation prices are all publicly readable in real time**. On a venue like that, a large leveraged trader is not merely watched; their liquidation level is a published number, and the size required to reach it is arithmetic anyone can do.

This inverts the usual assumption that information is an advantage. A trader whose stop is public is a trader whose stop is a target, and the incentive for well-capitalized counterparties to push toward that level is mechanical rather than conspiratorial — the liquidation itself is a forced market order, which is free money for whoever is on the other side of it. Large public positions on such venues have repeatedly become focal points for exactly this dynamic, with the position holder, the counterparties, and thousands of spectators all watching the same number.

The lesson generalizes: **transparency is not symmetric.** It helps small observers see large actors, and it helps large actors hunt exposed ones. If wallet-watching is the first thing, the second thing arrives with it.

## 7. What edge is actually left?

Suppose everything went right. You have a genuinely skilled wallet, correctly labelled, whose strategy is still live. How much is that worth?

![Two lines showing per-trade expectancy decaying with the delay after a transaction confirms, one before costs and one after a 4 percent round-trip cost, against a shaded breakeven line](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-7.webp)

#### Worked example 6: the arithmetic of being twenty minutes late

Take a wallet with a genuinely good illustrative record: **100 trades, 46 winners**, average winner **+38%**, average loser **−16%**. Its expectancy per trade is

$$
0.46 \times 38\% + 0.54 \times (-16\%) = 17.48\% - 8.64\% = +8.84\%
$$

Nearly nine percent per trade. That is an outstanding record, and it is why you wanted to follow this wallet.

Now be a follower. The wallet's buy is not visible until the transaction is included in a block — on Ethereum, [a slot every 12 seconds](https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/) — and it is visible to searchers in the mempool before that. You are a human with an alert, so realistically you see it, check it, and execute perhaps **twenty minutes later**. In that window the token has already moved; assume it is **+9%** by the time you buy.

Every outcome now gets rescaled by your worse entry:

- A winner: you buy at 1.09× the wallet's entry and exit at 1.38× it, so your return is 1.38 ÷ 1.09 − 1 = **+26.6%**
- A loser: 0.84 ÷ 1.09 − 1 = **−22.9%**

Your expectancy becomes

$$
0.46 \times 26.6\% + 0.54 \times (-22.9\%) = 12.24\% - 12.39\% = -0.15\%
$$

**The entire +8.8% edge is gone.** (The chart above rounds this to −0.1%.) Not reduced — gone. And we have not paid any costs yet. Add a realistic **4% round-trip** for a thin token (0.30% AMM fee each way, roughly 1.5% slippage each way on a small pool, plus gas) and your expectancy is

$$
-0.15\% - 4\% = -4.15\% \text{ per trade}
$$

Compound that. Risking 10% of your capital per trade, your portfolio return per trade is about −0.41%, so over 100 trades:

$$
(1 - 0.0041)^{100} = 0.663 \implies \$10{,}000 \rightarrow \$6{,}631
$$

A 34% drawdown from following a wallet that was genuinely excellent. Size up to full bankroll each time and $(1 - 0.0415)^{100} \approx 0.015$ — ten thousand dollars becomes about **\$150**.

**The intuition:** you are not copying the wallet's trade. You are buying from the people who copied it faster than you, and the price you pay them is the wallet's entire edge.

### Where the breakeven actually sits

Read the figure again with costs in mind. Before costs the strategy breaks even at about **20 minutes** of delay. After a 4% round trip it breaks even at about **one minute**. One minute is not a human timescale. It is a bot timescale — and the bots competing for that window are the same infrastructure that runs sandwich attacks and priority-fee auctions, which is to say [MEV](/blog/trading/crypto/crypto-mining-staking-and-mev) searchers with co-located infrastructure and negative-latency access to the mempool. You are not going to beat them to it, and neither is the alert service you subscribed to.

Three honest conclusions:

1. **Copy-trading at retail latency has negative expectancy** on almost any wallet worth copying, because the wallets worth copying are the ones most heavily watched. [The perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) goes through the failure modes in more depth.
2. **The surviving edge is at longer horizons**, where 20 minutes is noise: accumulation patterns over weeks, treasury behaviour over quarters, a fund's rotation across sectors over a cycle. Nobody front-runs a thesis that takes three months to play out.
3. **The other surviving edge is defensive** — using the data to avoid things rather than to buy them. That asymmetry is the subject of the next-to-last section, and it is where I think the real value is.

## How it shows up in price

Strip the narrative away and there are only four channels through which a whale's activity actually reaches the price. It is worth being explicit about them, because a claim that does not run through one of these is not a mechanism, it is a vibe.

**Channel 1: direct consumption of liquidity.** The whale sends a market order; it walks the book; the print moves. This is worked example 2. The move is mechanical, immediate, proportional to size over depth, and it partially reverts as market makers refill — the permanent component is typically a fraction of the temporary one.

**Channel 2: the reflexive cascade.** The print from channel 1 triggers things that are not discretionary: **stop-loss orders**, **liquidations** of leveraged positions, and automated de-risking. Each of those is itself a market order, which walks the book further, which triggers more. This is why a \$200 million sell can produce a move far larger than \$200 million of impact, and it is the mechanism behind every crypto flash crash — the initiating trade is the match, the leverage stack is the fuel.

**Channel 3: information and imitation.** Observers infer that the whale knows something and trade in the same direction. Note that this channel operates *whether or not the whale knows anything*, and note that it is strongest precisely where labelling is best — a well-labelled wallet has more followers, so its trades have more imitation impact than an anonymous one of identical size. Attribution creates price impact that would not otherwise exist.

**Channel 4: inventory and hedging.** When a whale sells to an OTC desk or a market maker rather than to the book, the desk now holds unwanted inventory and hedges it — usually by shorting perpetual futures. The spot price never saw the trade, but the perpetual **funding rate** turns negative and the basis moves, and those are readable. This is the channel that on-chain-only analysis misses completely, and it is the reason a large deposit sometimes precedes no spot move at all while the derivatives market moves plenty.

The practical reading: channels 1 and 2 are visible in the tape, channel 3 is visible in social data, and channel 4 is visible in funding and open interest. None of them is visible in the transfer itself.

## Common misconceptions

**"The top 10 wallets hold 30% of the supply, so the token is dangerously concentrated."** Very often those wallets are exchanges, bridges, staking contracts, the token's own vesting contract, and a liquidity pool. Concentration metrics that do not exclude labelled infrastructure addresses are close to meaningless, and the ones on most free dashboards do not exclude them. The real concentration question is what share of the **freely tradeable float** — supply not locked, staked, or held in infrastructure — sits with entities that could sell tomorrow.

**"Market cap is what would change hands if everyone sold."** Market cap is the last print multiplied by supply, and the last print is what one marginal buyer paid for one marginal coin. In the pool from worked example 3, selling \$1.5 million took 75% off the quoted price and therefore 75% off the "market cap" — a paper destruction of many multiples of the cash involved. Market cap is a ratio, not a pool of money.

**"This wallet was dormant for ten years and just moved, so it must be about to sell."** A dormant wallet moving is evidence that someone still has the keys. That is all it is evidence of. Long-dormant coins move for custody upgrades, inheritance, estate settlement, security migrations after a wallet-software vulnerability, and moves to institutional custody — all of which look identical on-chain to the first hop of a sale, and none of which are one.

**"Nansen labelled it Smart Money, so it is a smart trade."** It is a wallet that made money on past trades in a trailing window. Worked example 4 shows how many wallets earn that description with no skill at all. The label is a filter for where to look, and treating it as a recommendation inverts what it measures.

**"Whale alerts front-run the market."** By the time an alert fires, the transaction is confirmed and public, and every automated system saw it in the mempool before it confirmed. An alert is a notification that something has already happened and has already been traded on. The information is not late by seconds; it is late by an entire class of counterparty.

**"On-chain data is objective, so it cannot be wrong."** The *transactions* are objective. Everything you actually use — the entity names, the smart-money tags, the "exchange inflow" aggregates, the clusters — is inference built on top, produced by heuristics with real error rates, and the errors are not random. They systematically over-attribute to whoever was labelled first.

## How it shows up in real markets

### 1. Eleven thousand bitcoin, one exchange, and two readings

In late October 2025, Arkham Intelligence flagged movement out of a cluster of wallets it attributes to **Owen Gunden**, an early bitcoin holder ([Yahoo Finance, 21 November 2025](https://finance.yahoo.com/news/bitcoin-billionaire-dumps-entire-1-200442319.html)). Over the following weeks roughly **11,000 BTC** — reported at around **\$1.3 billion** — moved from that cluster to Kraken, with a final tranche of about **2,499 BTC** landing on **20 November 2025**.

Start with the attribution, because it is the part the coverage skipped. The name did not come from Gunden. He does not appear to maintain an active public presence, and there is no signed message or disclosure tying him to those addresses. The identification is **Arkham's own on-chain inference** — tier 2 or tier 3 on the ladder in section 3, depending on which heuristics carried the weight. It may well be right. It is still an inference, and the honest sentence is *a cluster Arkham attributes to Gunden*, never *Gunden's wallet*.

Now the part that matters. Arkham's reading was that the transfers indicated he had sold his entire position. The outlet reporting it added its own qualifier in the very next breath: he could equally be using Kraken to custody the coins, or to reach its staking product and earn yield on them.

Both readings are consistent with **every byte of on-chain evidence**, because the on-chain evidence stops at Kraken's deposit address. This is worked example 5 in the wild with a billion dollars riding on it: the chain proved that 11,000 BTC changed custody and proved nothing at all about whether they changed owner. A headline reading "dumps entire stash" is not reporting a fact. It is reporting the 27.5% branch of figure 6 as though the other five branches were not there.

### 2. Eighty thousand bitcoin wake up, and nothing happens

On **4 July 2025**, eight wallets dormant since 2011 moved a combined **80,000 BTC** — about **\$8.6 billion** at the time. Arkham Intelligence flagged the movement and assessed that the same entity appeared to control all eight ([CoinDesk, 5 July 2025](https://www.coindesk.com/markets/2025/07/05/eight-bitcoin-wallets-move-80000-btc-in-largest-ever-satoshi-era-transfers)).

Three things about it are worth more than the headline number.

**Nobody knew whose coins they were.** No individual or company publicly claimed them, and — contrary to a great deal of what circulated — Arkham did not attribute them to Satoshi Nakamoto. The wallets were 2011-vintage, which suggests an early miner or early buyer and is evidence of nothing more specific. Note how the drift happens: the phrase "Satoshi-era", meaning *from the period when Satoshi was active*, sits one hyphen away from "Satoshi's". A timestamp becomes an identity in the space of a headline.

**It reads as a custody upgrade, not a sale.** The coins moved into fresh wallets using a modern, lower-fee address format, and as of the following day had not moved again. [The Block's framing](https://www.theblock.co/post/361269/bitcoin-whale-movement-arkham) was "possible address upgrade, no signs of sale". That is exactly what a fourteen-year-old holder migrating to better key management looks like.

**Which is why "dormant wallet awakens" is close to information-free.** The event proves someone still has the keys. It does not distinguish a migration from an inheritance, an estate settlement, a custody transfer or the first hop of a liquidation — all of which produce the same on-chain shape. Anyone who shorted the alert was trading a hypothesis whose most likely resolution was "nothing happens."

### 3. When your stop-loss is a published number

Some venues make the reflexive problem structural rather than incidental. Decentralized perpetual exchanges that settle on-chain publish open positions, entry prices and liquidation levels in real time — not as a leak, but as a design property.

In May 2025 a trader operating publicly as **James Wynn** ran very large leveraged positions on one such venue, Hyperliquid, at a size and leverage that [secondary reporting](https://www.linkedin.com/pulse/hyperliquid-trader-james-wynns-168b-saga-ends-stunning-yee-chun-lim-szclc) puts at roughly \$1 billion notional at around 40 times leverage. Across late May those positions were liquidated in a series of cascades, with losses reported in the region of \$100 million. Treat the exact dollars as approximate — they come from secondary crypto media rather than primary venue data — but the structural facts are not in dispute: enormous public leveraged size, then a run of liquidations.

The mechanism needs no allegation to work. On a venue where a position's liquidation price is public, that price is a **published, precise level at which a large forced market order will be created**. Anybody can compute how far the market must travel to reach it, and a forced liquidation is a counterparty that has to trade at whatever price it finds. Commentators attributed the cascades in part to exactly that visibility. Whether anyone deliberately pushed toward the level I have no way to know and am not claiming; the point is that the incentive exists mechanically, and it exists *because the information was public*.

Generalize it and you get the uncomfortable half of transparency. The same publicity that lets you watch a whale lets the market see precisely where a leveraged whale breaks.

### 4. The most telegraphed supply event in crypto

Not every large flow is a surprise. In 2024 the Mt. Gox trustee began distributing roughly **142,000 BTC** to creditors of the exchange that collapsed in 2014, with repayments starting in **July 2024** ([trustee announcement, June 2024](https://kryptomoney.com/mt-gox-btc-and-bch-repayments-starting-july-2024/)). The schedule was announced in advance and the addresses were watchable throughout.

This is the control experiment for the entire discipline. Everyone knew the size. Everyone knew roughly when. The coins were traceable as they moved. And the market still had to price two things the chain could not answer: how many creditors would sell rather than hold after a decade of waiting, and over what horizon they would do it.

On-chain data answered *where the coins are* with perfect precision, and *what the holders intend* not at all. That is the boundary condition of the whole field, and it does not move no matter how good the tooling gets.

### 5. The label that tells you it might be wrong

The last case is not a market event but a sentence, and it may be the most useful thing on this page.

Etherscan — the block explorer whose name tags get screenshotted as ground truth across the entire ecosystem — [publishes how those tags are made](https://info.etherscan.com/public-name-tags-labels/). You suggest one through a contact form. Curators then evaluate "the owner's interest in displaying their address publicly or whether the address is of public interest." And the documentation carries this warning: "it may be later reported that a user was falsely claiming to own a specific address."

Read it twice. The most widely cited labelling source on Ethereum states, in its own documentation, that someone can claim an address they do not own and be tagged accordingly until another user reports it. It is to Etherscan's credit that they say so in public. But it means every screenshot of an Etherscan tag offered as proof of ownership is a **tier-5 claim** from figure 5's ladder — routinely presented, and routinely accepted, as though it were tier 1.

## Retail defense: how to use wallet data without becoming the exit liquidity

If the honest conclusion is that copy-trading loses money, what is this data actually for? Quite a lot, but almost all of it is defensive. The asymmetry is worth stating plainly:

> On-chain data is far better at telling you what to avoid than at telling you what to buy. Avoidance signals are structural, slow-moving, and not competed away. Entry signals are fast, crowded, and gone before you see them.

![A six-stage pipeline from picking an entity through verifying the label, classifying the movement, sizing against depth and estimating the probability of a sale, to the decision](/imgs/blogs/whales-smart-money-and-on-chain-wallet-watching-9.webp)

Here is the workflow the figure describes, as six gates. Most candidate signals should die at gate 4 or 5, and a process where they do not is a process that is not filtering anything.

**Gate 1 — pick the entity, not the address.** Decide whose behaviour you care about and why *before* you go looking. "I want to know whether this token's treasury is funding operations by selling" is a question. "What are the big wallets doing" is not. Starting from an alert rather than from a question is how you end up rationalizing noise.

**Gate 2 — verify the label to a tier.** Use the table in section 3. Ask specifically: is this self-disclosed or provable (tier 1-2), or is it an inference (tier 3-5)? Check the label's *age*. Check whether two tools agree — and check whether they agree because they verified independently or because one consumes the other's data, which is common and makes agreement worthless. Where the label is contested, hold it as *reported* or *labelled by*, and size accordingly.

**Gate 3 — classify the movement.** Wallet-to-wallet inside the same cluster is housekeeping. Wallet to a DEX router is an actual trade you can price. Wallet to a bridge is a chain migration you must follow before you can say anything. Wallet to an exchange deposit address is the ambiguous case that needs gate 5. These are genuinely different events and conflating them is the most common error in the whole discipline.

**Gate 4 — size it against real depth, never against market cap.** Pull the actual 2% depth on the venues that matter and the actual pool reserves on-chain, and compute the size as a multiple of them. Use the constant-product rule of thumb for v2-style pools and read the tick-level distribution for v3-style ones. A transfer worth 0.4% of the 2% book is not news at any dollar value; one worth 4× the book is news at any dollar value.

**Gate 5 — estimate the probability of a sale, and write it down.** Run worked example 5 on your own log. Keep the log. The number you get is the only thing standing between you and treating a 27% event as a certainty, and it is also the number that tells you when your labelling has degraded — if your resolution rate falls, your labels are going stale.

**Gate 6 — decide honestly what it is.** There are three answers, and two of them are not trades: *actionable* (rare), *context* (common and valuable — it changes your sizing or your patience, not your direction), or *noise* (the majority).

### The four things this data is genuinely good for

**Checking supply overhangs before you buy.** Vesting contracts, foundation treasuries, and early-investor wallets are typically tier 1-2 labels because the project published them. Knowing that 18% of supply unlocks over the next quarter, and watching whether previous tranches went to a custodian or to an exchange, is slow, verifiable, and directly relevant to whether you want to own the thing. This is the highest-value use of on-chain data for a non-professional, by a wide margin.

**Detecting that a token's float is fake.** If nine of the top ten holders are the deployer's cluster, the tradeable float is far smaller than the supply figure implies, and both the upside and the downside are far more violent than the market-cap number suggests. [Follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) is the systematic version of this check.

**Measuring the depth you would actually be exiting into.** Before you buy, compute what *your* position would cost to sell. If your intended \$20,000 position is 1.3× the pool's 2% depth, you have not bought an asset, you have bought a position you cannot leave. This calculation takes two minutes and is skipped essentially always.

**Sanity-checking a narrative against the ledger.** When a project claims a partnership, a buyback, a burn, or a treasury diversification, the chain often shows whether it happened. This is a genuine and underrated edge, because it is verification rather than prediction, and verification is not competed away by latency.

### And the discipline that makes it safe

Never trade on a single transfer. Never trade on an alert you did not seek out — an alert is a broadcast, and a broadcast is by definition not private information. Never assume a label is current. Never infer intent from a deposit. Size every position on the depth you could exit into, not the size you can afford to enter with. And ask, before every trade sourced from public data: **if ten thousand other people saw this, who am I buying from?** If the answer is "the people who saw it first," you are not acting on the signal — you are the liquidity that lets them act on it.

This is educational material about market mechanics, not investment advice.

## When this matters to you

Most people encounter this world through a screenshot: a wallet, a number, an arrow, a conclusion. The purpose of this post is to make you slower than that screenshot.

The three questions worth carrying away are small and cheap to ask. *Compared to what depth?* — because size means nothing until it is divided by liquidity, and the liquidity number is never in the post. *How was this label made?* — because the difference between a proof-of-reserves address and a crowd-submitted tag is the difference between a fact and a rumour with good typography. *Who else can see this?* — because everything on a public chain is public to everyone simultaneously, which means the trade you are considering has already been made by someone faster, and your entry is their exit.

Later in this series the forensic track takes these tools apart in much more depth — `tracing-a-market-makers-onchain-footprint` reconstructs a named desk's inventory cycle from its transfers, and `following-token-flows-from-insiders-to-exit-liquidity` follows allocations from a cap table through to the wallets that end up holding them. Both build directly on the labelling tiers and the depth arithmetic here.

The chain is the most transparent financial ledger ever built. It is also, by design, a record of *custody* — and custody is not intent, size is not impact, and a label is not a name. Hold those three distinctions and most of what passes for on-chain analysis resolves into what it actually is: a hypothesis, with a provenance, that you now know how to test.

## Sources & further reading

**The case studies**

- ["Eight Bitcoin Wallets Move 80,000 BTC in Largest Ever Satoshi-Era Transfers"](https://www.coindesk.com/markets/2025/07/05/eight-bitcoin-wallets-move-80000-btc-in-largest-ever-satoshi-era-transfers) — CoinDesk, 5 July 2025. The 4 July 2025 movement and Arkham's same-entity assessment. (The headline's "Satoshi-era" is the outlet's phrasing for the vintage of the wallets, not an attribution to Satoshi Nakamoto — see case 2.)
- ["\$8.7 billion in OG bitcoin moved in possible address upgrade, no signs of sale"](https://www.theblock.co/post/361269/bitcoin-whale-movement-arkham) — The Block, July 2025. The custody-migration reading of the same event.
- ["Bitcoin Billionaire Dumps Entire \$1.3 Billion BTC Stash After 14 Years"](https://finance.yahoo.com/news/bitcoin-billionaire-dumps-entire-1-200442319.html) — Yahoo Finance, 21 November 2025. The Gunden transfers, Arkham's attribution, and the outlet's own custody-or-staking qualifier.
- [Recap of the Hyperliquid positions attributed to James Wynn](https://www.linkedin.com/pulse/hyperliquid-trader-james-wynns-168b-saga-ends-stunning-yee-chun-lim-szclc) — secondary recap, 2025. Source of the approximate position and loss figures in case 3; secondary sourcing, so the exact dollars should be treated as indicative.
- ["Mt. Gox BTC and BCH Repayments Starting July 2024"](https://kryptomoney.com/mt-gox-btc-and-bch-repayments-starting-july-2024/) — 2024. The trustee's distribution schedule.

**How the labels are actually made**

- ["Public Name Tags, Labels & Public Notes"](https://info.etherscan.com/public-name-tags-labels/) — Etherscan Information Center, accessed 29 July 2026. Submission, curation, and the false-claim caveat quoted in case 5.
- [Nansen documentation](https://docs.nansen.ai/llms-full.txt) — Nansen, accessed 29 July 2026. The Smart Money category list and the "top 5,000 highest-performing wallets" definition.
- [Arkham Intelligence](https://info.arkm.com/) — accessed 29 July 2026. The company's own description of its attribution product.
- Meiklejohn, Pomarole, Jordan, Levchenko, McCoy, Voelker and Savage, ["A Fistful of Bitcoins: Characterizing Payments Among Men with No Names"](https://dl.acm.org/doi/10.1145/2504730.2504747) — ACM Internet Measurement Conference, 2013. The origin of the common-input-ownership and change-address heuristics.

**The market mechanics**

- ["How Uniswap works"](https://developers.uniswap.org/docs/get-started/concepts/how-uniswap-works) — Uniswap developer documentation. The constant-product invariant, and concentrated liquidity in v3/v4.
- [`UniswapV2Library.sol`](https://raw.githubusercontent.com/Uniswap/v2-periphery/master/contracts/libraries/UniswapV2Library.sol) — Uniswap v2-periphery. `getAmountOut` and the 997/1000 constants that encode the 0.30% fee.
- Tóth, Lempérière, Deremble, de Lataillade, Kockelkoren and Bouchaud, ["Anomalous price impact and the critical nature of liquidity in financial markets"](https://arxiv.org/abs/1105.1694) — Physical Review X, 2011. The square-root impact law.
- [Proof-of-stake](https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/) — ethereum.org. Ethereum's 12-second slot time.
- [Bitcoin circulating supply](https://blockchain.info/q/totalbc) — Blockchain.com API, accessed 29 July 2026. The ~20.06 million figure used in worked example 1.

**Elsewhere on this blog**

- [How crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) — the order-book microstructure this post assumes.
- [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price) — where whales actually execute.
- [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution) — the labelling machinery in more depth.
- [What is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain) — the taxonomy behind the label.
- [The perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) — the failure modes of acting on it.
