---
title: "The Low Float, High FDV Game: How 5% of a Token Prices the Other 95%"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "Why the modern token launch prices a project at an enormous fully diluted valuation on a tiny circulating float, who is paid in that headline number, and why the late buyer ends up funding everyone else's mark."
tags: ["crypto", "fdv", "tokenomics", "token-unlocks", "float", "market-structure", "crypto-players", "valuation", "market-making", "retail-defense", "vesting"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR**  -  A "low float, high FDV" token is one where only a few percent of the supply can actually be traded on launch day, while the price those few percent change hands at is multiplied by *all* the supply to produce a headline valuation. The small number is real. The big number is an extrapolation.
>
> - **Float** is the share of total supply that can trade today. **Fully diluted valuation (FDV)** is today's price multiplied by *total* supply, including the tokens that are locked, unminted, or sitting in a foundation wallet.
> - At a 5% float, roughly **\$200 of headline FDV is created per \$1 of net buying** (illustrative constant-depth book). That leverage is the entire mechanism  -  it is arithmetic, not manipulation.
> - Almost every insider is paid in the big number: the VC's mark, the treasury's balance sheet, the market maker's option strikes, the exchange's listing volume. The day-one buyer is the only participant who pays it in cash.
> - The aftermath is arithmetic too. If supply grows 20x while the circulating market cap merely stays flat, the price must fall **95%**  -  and that is the *good* scenario.
> - The number to remember: tokens that launched in 2024 carried an average market-cap-to-FDV ratio of roughly **12.3%**, meaning about **88% of the supply had not hit the market yet** (Binance Research, data as of April 14, 2024).

There is a specific moment that has ruined more crypto portfolios than any hack.

A new token lists on a major exchange. The chart is beautiful  -  it opens, it runs, and every feed you read is celebrating. You pull up the price page. Market cap: \$50 million. That feels *small*. Small caps go up. You have watched enough cycles to know that a \$50 million token becoming a \$500 million token is an ordinary thing that happens most months.

So you buy. And then, for two years, the chart does nothing but bleed  -  not in a crash, not on bad news, just a patient, grinding, seemingly causeless decline. The product ships. The team posts updates. Users grow. And the price goes down anyway.

What you were not looking at was the number sitting immediately below the one you read. Next to that \$50 million market cap was a second figure: **fully diluted valuation, \$1 billion**. You had not bought a \$50 million project. You had bought a \$1 billion project in which only \$50 million of it was allowed to trade, and the other \$950 million was arriving on a public, pre-announced schedule, into a market that had already spent its enthusiasm.

![Two panels comparing a fifty million dollar circulating market cap against a one billion dollar fully diluted valuation for the same token on the same day](/imgs/blogs/the-low-float-high-fdv-game-1.webp)

The figure above is the mental model for this entire post: **one price, two supply numbers, two valuations that differ by 20x.** Neither number is a lie. The circulating market cap honestly describes the value of what trades. The fully diluted valuation honestly describes what the market is implying about the whole project. The problem is not that either is false  -  it is that the structure of the launch decides *which one you experience*. Insiders are marked at the big number. You buy at a price implied by the big number. And over the following years, the supply that justifies the big number arrives and asks the market to pay for it.

This post builds the whole thing from zero. What supply, float, market cap and FDV actually are. Why a tiny float mechanically converts small amounts of buying into enormous headline valuations. Why the structure is individually rational for every single insider  -  which is why it persists despite being criticised continuously since 2024. And then the part that matters most for you: what the aftermath looks like in arithmetic, and the six things you can check for free before you buy.

One note on honesty before we start. Every number in this post is one of two kinds, and they are always labelled. **Illustrative** numbers belong to a made-up token I will call NIM, and they exist so the arithmetic is clean enough to follow in your head. **Real** numbers carry a source and an as-of date. I will never blur the two, and I will criticise the *structure* rather than accuse any named firm or founder of acting in bad faith  -  the whole point of this post is that you do not need bad faith to produce this outcome. Incentives are enough.

## Foundations: supply, float, market cap and FDV

If you have never held a token, start here. Nothing below assumes prior knowledge, and every term is defined the first time it appears.

### What "supply" means for a token

A company issues **shares**. A crypto project issues **tokens**. In both cases there is some number of units in existence, and each unit represents a claim  -  on a company's profits in the case of shares, on something much fuzzier in the case of tokens (governance votes, fee discounts, staking rewards, or in many cases nothing enforceable at all; that distinction deserves its own treatment, and it gets one in [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock)).

Three supply numbers matter, and confusing them is the single most common beginner mistake in crypto. The definitions below follow [CoinGecko's supply methodology](https://support.coingecko.com/hc/en-us/articles/32294647667865-CoinGecko-Supply-Methodology):

- **Total supply**  -  every token that exists right now, whether it can move or not. Includes tokens sitting in locked contracts, foundation treasuries, and unclaimed airdrops.
- **Max supply**  -  the largest number of tokens that will ever exist, if the protocol has a cap. Bitcoin's is 21,000,000. Many tokens have no cap and inflate indefinitely.
- **Circulating supply**  -  the tokens that are actually free to be traded *today*: unlocked, in someone's wallet, not contractually restricted.

The gap between circulating supply and total supply is where this entire post lives.

### Float: the number almost nobody looks at

**Float** is circulating supply divided by total supply, expressed as a percentage. It answers one question: *what fraction of this project can actually change hands right now?*

A traditional stock market analogy helps. When a company does an initial public offering (IPO), it typically sells a meaningful slice of itself to the public  -  historically often in the range of 10% to 30% of shares outstanding, with insiders locked up for a defined period afterwards. That locked-up portion is not tradable, and public filings tell you exactly how much of it there is and when it releases. Everyone knows the float is a fraction of the company, and the stock's *market capitalisation* is quoted on the whole company regardless.

Crypto inverted this. In a modern token launch, the fraction released to the public at listing is frequently in the low single digits of total supply, and the price is quoted with a headline valuation calculated on the *entire* supply. The lockup schedule is public, but it is published in a docs site or a Medium post rather than a regulatory filing, and almost nobody reads it before buying.

![A ten by ten grid of one hundred squares showing token supply allocation, with only five squares highlighted as the tradable float](/imgs/blogs/the-low-float-high-fdv-game-2.webp)

Look at that grid. Each square is 1% of the supply of our illustrative token, NIM  -  10,000,000 tokens out of a total of 1,000,000,000. Five squares are the airdrop and are liquid on day one. The other ninety-five belong to investors, the team, the foundation treasury and an ecosystem incentives budget, and none of them can move yet.

Every trade that happens on day one happens between people holding those five squares. Every price printed by those trades is then applied to all one hundred.

### Market capitalisation: what it is and what it is not

**Market capitalisation** ("market cap") is the simplest formula in finance:

$$\text{market cap} = \text{price per unit} \times \text{number of units}$$

That is it. If a token trades at \$1.00 and 50,000,000 of them are circulating, the *circulating* market cap is \$50,000,000.

What market cap is **not** is "the amount of money in the asset". This misunderstanding causes real losses, so let me be blunt about it: nobody put \$50 million into a token with a \$50 million market cap. Market cap is a multiplication, not a measurement of inflow. The last trade might have been for \$300. That \$300 trade re-priced all 50,000,000 circulating tokens, because market cap takes whatever the most recent print was and applies it to every unit.

This is true of stocks too, and it is not a crypto-specific scandal. Apple's market capitalisation is not the amount of cash ever invested in Apple either. But in a deep, mature market, the last print is backed by an enormous amount of resting willingness to trade at nearby prices. In a thin one, it is not  -  and that difference is the whole ballgame, which we get to in the next section.

### Fully diluted valuation: the same multiplication, different multiplicand

**Fully diluted valuation (FDV)** takes the same price and multiplies it by total supply instead of circulating supply. This is the standard theoretical definition used by [CoinGecko](https://www.coingecko.com/en/glossary/fully-diluted-valuation):

$$\text{FDV} = \text{price per unit} \times \text{total supply}$$

For NIM at \$1.00 with 1,000,000,000 tokens total, FDV is \$1,000,000,000.

FDV is not a crypto invention and it is not inherently dishonest. In equity markets, analysts routinely compute a "fully diluted" share count that includes options, warrants and convertible notes that have not been exercised yet, because those claims are real and will eventually dilute existing holders. Ignoring them flatters the valuation. FDV in crypto plays exactly the same role: it asks *what is this whole thing worth if every token that will exist, exists*.

The honest reading of FDV is: **this is the valuation the market is implicitly assigning to the project, if you assume the locked supply is worth the same per-token as the liquid supply.**

That assumption is doing a spectacular amount of work, and the rest of this post is about what happens when it turns out to be wrong.

#### Worked example 1: the same token, two valuations

Let us do the arithmetic explicitly. All numbers illustrative.

NIM lists on a large exchange. The facts:

- Total supply: **1,000,000,000 NIM**
- Circulating at listing: **50,000,000 NIM** (the airdrop tranche)
- Float: 50,000,000 ÷ 1,000,000,000 = **5.0%**
- Opening price: **\$1.00**

Two valuations follow immediately:

- Circulating market cap = 50,000,000 × \$1.00 = **\$50,000,000**
- Fully diluted valuation = 1,000,000,000 × \$1.00 = **\$1,000,000,000**

The ratio of those two  -  market cap divided by FDV  -  is 50 ÷ 1,000 = **5%**. That ratio is just the float again, and this is worth internalising because it makes the number checkable in two seconds on any price page:

$$\frac{\text{market cap}}{\text{FDV}} = \text{float}$$

Whenever you see a market-cap-to-FDV ratio, you are looking at the float. A token showing a 5% ratio is telling you that 95% of it has not arrived yet.

**The intuition this teaches:** the "small cap" you think you are buying and the "big valuation" the project is being celebrated at are the same price wearing two different hats, and the hat you get to wear is decided by the launch structure rather than by you.

### The units, one more time

Because these get mixed up constantly, here they are side by side.

| Term | Formula | What it honestly describes | Common misuse |
| --- | --- | --- | --- |
| Circulating supply |  -  | Tokens that can trade today | Quoted as if it were the whole project |
| Total supply |  -  | Every token that exists, locked or not | Ignored entirely by new buyers |
| Float | circulating ÷ total | The share of the project that is liquid | Rarely computed at all |
| Market cap | price × circulating | The value of what trades | Read as "money invested" |
| FDV | price × total | The implied value of the whole project | Read as a target, or dismissed as fake |
| Market cap ÷ FDV |  -  | The float, restated | Treated as a quality score |

## Why 5% of the supply can price 100% of it

Here is the part that feels like a trick the first time you see it, and then never feels like one again.

### Price comes from the marginal trade, not from the average holder

Nobody polls token holders to determine a price. The price of anything traded on an exchange is simply **the price of the most recent transaction**  -  the marginal trade, the one at the edge. That trade is a meeting between the most eager buyer and the most willing seller *at that instant*.

An exchange holds an **order book**: a list of resting offers to buy (the *bids*) and resting offers to sell (the *asks*), each with a price and a size. When you place a *market buy*  -  an instruction to buy immediately at whatever price is available  -  the exchange matches you against the cheapest ask, then the next cheapest, and so on until your order is filled. If your order is large relative to the resting asks, you consume several price levels on the way up, and the price you leave behind is higher than the one you arrived at. That gap between the price you expected and the average price you actually paid is called **slippage**.

The total dollar value of resting orders near the current price is called **depth**. Depth is the shock absorber of a market. Deep books absorb size without moving; thin books do not.

Now the crucial link: **depth is roughly proportional to float.** The tokens available to be sold at a price near the current one are, by definition, tokens that are unlocked and in someone's wallet. Locked tokens cannot be posted as asks. A 5% float does not just mean 5% of the supply is tradable  -  it means the order book is built out of that 5%, and is therefore roughly 5% as deep as it would be with everything unlocked. (For a fuller treatment of how books, takers and makers interact, see [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move).)

### The same order, two different books

![Two order book ladders side by side showing a two million dollar market buy moving price forty percent on a thin book and three percent on a deep book](/imgs/blogs/the-low-float-high-fdv-game-3.webp)

Take a single, identical \$2,000,000 market buy and send it into two versions of the same token.

In the thin-float version, suppose there is **\$50,000 of resting asks at every 1% price step** above the current \$1.00  -  an illustrative but unremarkable book for a newly listed asset. Your \$2,000,000 consumes 2,000,000 ÷ 50,000 = **40 steps**. The last print lands at **\$1.40**, up 40%.

In the deep-float version  -  same project, same news, but 60% of supply unlocked instead of 5%  -  the book is roughly twelve times as deep, so there is **\$600,000 at every 1% step**. The same \$2,000,000 consumes 2,000,000 ÷ 600,000 = **3.3 steps**. The last print lands at **\$1.03**, up 3.3%.

Identical demand. Twelve times the float. Twelve times less price movement.

Watch that happen one price level at a time. The animation below sends the same \$2,000,000 into the thin book in five blocks of \$400,000, and tracks what each block does to the headline valuation.

<figure class="blog-anim">
<svg viewBox="0 0 720 330" role="img" aria-label="A two million dollar market buy consuming five blocks of resting ask orders on a thin book, lifting the last price from one dollar to one dollar forty and the fully diluted valuation from one billion to one point four billion dollars" style="width:100%;height:auto;max-width:760px">
<style>
.ob-hdr{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ob-px{font:600 12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-primary,#1f2937);text-anchor:end}
.ob-fdv{font:600 12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280)}
.ob-note{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ob-rule{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:3 4}
.ob-bar{fill:var(--border,#d1d5db);rx:4}
.ob-mark{stroke:var(--accent,#6366f1);stroke-width:2.5}
@keyframes ob-eat{0%{fill:var(--border,#d1d5db)}9%,90%{fill:var(--accent,#6366f1)}100%{fill:var(--border,#d1d5db)}}
@keyframes ob-rise{0%{transform:translateY(0)}12%{transform:translateY(-38px)}24%{transform:translateY(-76px)}36%{transform:translateY(-114px)}48%,90%{transform:translateY(-152px)}100%{transform:translateY(0)}}
.ob-bar{animation:ob-eat 9s ease-in-out infinite}
.ob-b2{animation-delay:1.08s}.ob-b3{animation-delay:2.16s}.ob-b4{animation-delay:3.24s}.ob-b5{animation-delay:4.32s}
.ob-mark{animation:ob-rise 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ob-bar{animation:none;fill:var(--accent,#6366f1)}.ob-mark{animation:none;transform:translateY(-152px)}}
</style>
<text class="ob-hdr" x="130" y="24">Resting asks  -  5% float  -  $50,000 at every 1% step, so $400,000 per block</text>
<line class="ob-rule" x1="130" y1="34" x2="700" y2="34"/>
<text class="ob-px" x="122" y="66">$1.32  -  $1.40</text>
<rect class="ob-bar ob-b5" x="130" y="48" width="300" height="28"/>
<text class="ob-fdv" x="442" y="66">$400,000 &#8594; FDV $1.40B</text>
<text class="ob-px" x="122" y="104">$1.24  -  $1.32</text>
<rect class="ob-bar ob-b4" x="130" y="86" width="300" height="28"/>
<text class="ob-fdv" x="442" y="104">$400,000 &#8594; FDV $1.32B</text>
<text class="ob-px" x="122" y="142">$1.16  -  $1.24</text>
<rect class="ob-bar ob-b3" x="130" y="124" width="300" height="28"/>
<text class="ob-fdv" x="442" y="142">$400,000 &#8594; FDV $1.24B</text>
<text class="ob-px" x="122" y="180">$1.08  -  $1.16</text>
<rect class="ob-bar ob-b2" x="130" y="162" width="300" height="28"/>
<text class="ob-fdv" x="442" y="180">$400,000 &#8594; FDV $1.16B</text>
<text class="ob-px" x="122" y="218">$1.00  -  $1.08</text>
<rect class="ob-bar" x="130" y="200" width="300" height="28"/>
<text class="ob-fdv" x="442" y="218">$400,000 &#8594; FDV $1.08B</text>
<line class="ob-mark" x1="126" y1="230" x2="434" y2="230"/>
<line class="ob-rule" x1="130" y1="252" x2="700" y2="252"/>
<text class="ob-note" x="130" y="276">$2,000,000 of market buying clears the ladder. Last price $1.00 &#8594; $1.40, up 40%.</text>
<text class="ob-note" x="130" y="298">Fully diluted valuation $1.00B &#8594; $1.40B  -  $400M of headline valuation for $2M of cash.</text>
<text class="ob-note" x="130" y="320">Illustrative constant-depth book. $200 of FDV created per $1 spent.</text>
</svg>
<figcaption>Illustrative. Two million dollars of market buying walks the thin book from \$1.00 to \$1.40. Because the price applies to all 1,000,000,000 tokens, the headline valuation moves \$400 million.</figcaption>
</figure>

### The multiplier: dollars of FDV per dollar of buying

Now connect that to the headline number, because this is where the structure earns its name.

In the thin book, the price went from \$1.00 to \$1.40, a 40% increase. FDV is price times total supply, so FDV went from \$1,000,000,000 to **\$1,400,000,000**. Four hundred million dollars of headline valuation appeared, and it was produced by two million dollars of cash.

$$\frac{\text{FDV created}}{\text{cash spent}} = \frac{400{,}000{,}000}{2{,}000{,}000} = 200 \quad \text{(dollars per dollar)}$$

**Two hundred dollars of fully diluted valuation per dollar of net buying.** Meanwhile the circulating market cap went from \$50,000,000 to \$70,000,000  -  a much more modest \$10 of circulating cap per dollar spent.

This multiplier is not a coincidence and it is not fraud. It is a mechanical consequence of two things multiplying together: a thin book means each dollar moves the price a lot, and a big total supply means each unit of price change is applied to a lot of tokens.

#### Worked example 2: the FDV leverage of a float

Hold the illustrative model fixed  -  depth scales with float, total supply is 1,000,000,000, price starts at \$1.00  -  and vary only the float.

At a float of 5%, depth is \$50,000 per 1% step. One dollar of buying moves the price by 1 ÷ 50,000 of one percent, and one percent of a \$1,000,000,000 FDV is \$10,000,000. So:

- FDV per \$1 = \$10,000,000 ÷ 50,000 = **\$200**

Run the same computation at every float, remembering depth scales up with the float:

| Float | Depth per 1% step | FDV created per \$1 of net buying |
| --- | --- | --- |
| 5% | \$50,000 | **\$200** |
| 10% | \$100,000 | \$100 |
| 25% | \$250,000 | \$40 |
| 50% | \$500,000 | \$20 |
| 100% | \$1,000,000 | \$10 |

![Bar chart of dollars of fully diluted valuation created per dollar of net buying at floats of five, ten, twenty-five, fifty and one hundred percent](/imgs/blogs/the-low-float-high-fdv-game-4.webp)

The relationship is exactly inverse. Write it as a rule:

$$\text{FDV per dollar of net buying} = \frac{k}{f}$$

where $f$ is the float as a fraction and $k$ is a constant set by how deep the book is per unit of float (here, \$10). **Halve the float, double the valuation each dollar of buying produces.**

**The intuition this teaches:** low float is not a side effect of a launch. It is a valuation amplifier with a dial on it, and the dial position is chosen before the token ever trades.

### Why this is not, by itself, manipulation

It is important to be precise here, because the low-float structure gets described as a scam and that description is both unfair and unhelpful  -  unfair because the arithmetic works the same whether anyone intends it or not, and unhelpful because it stops you from seeing the mechanism clearly enough to protect yourself.

Nothing in the preceding section required anyone to lie, wash trade, or coordinate. There was one honest buyer, one honest book, and one honest price feed. The FDV computed by every data site was correctly calculated. The multiplier appeared anyway.

That said  -  a structure this leveraged is also unusually *easy* to manipulate deliberately, precisely because a small amount of capital moves a large headline number. Manufactured volume, coordinated buying and painted closes all get more bang per dollar on a thin float. Those are separate practices with their own mechanics and their own detection signatures, covered in [wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) and [the anatomy of a token pump](/blog/trading/crypto-players/the-anatomy-of-a-token-pump). For this post, the point is that you do not need to allege any of that to explain the outcome. The structure alone is sufficient.

## Who is paid in the big number

If low float and high FDV were bad for everyone, it would not have become the default. It became the default because the headline number is a form of compensation, and almost everyone at the table receives some of it.

![A branching map showing the headline fully diluted valuation flowing to the seed investor, treasury, market maker, exchange and media, and being paid for by the day-one buyer](/imgs/blogs/the-low-float-high-fdv-game-5.webp)

Work around that map one seat at a time. The general shape of who benefits from what in this industry is mapped more broadly in [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto); here we are looking specifically at what a *high FDV on a small float* does for each participant.

### The venture investor: the mark is the product

A crypto venture fund raises money from limited partners (LPs)  -  pensions, endowments, family offices, funds of funds  -  and reports to them periodically on what the portfolio is worth. Private, illiquid positions are hard to value; a private company is typically carried at the price of its last funding round until something changes.

A token is different, and better, from the fund's point of view: **once it lists, it has an observable market price.** The position stops being an estimate and becomes a mark-to-market number, quoted by exchanges and reproduced by every data provider. It goes in the quarterly report as a real, externally verifiable figure.

That is a genuinely valuable thing for a fund. Marked-up positions are how a fund demonstrates performance, and demonstrated performance is how it raises the next fund. Note carefully: this does not require the fund to sell a single token, or to want the price to be artificially high. It only requires the fund to *prefer a higher observable price to a lower one*, which is not a controversial preference. The full mechanics of how a crypto fund actually operates  -  sourcing, structuring, marks, and exits  -  are laid out in [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model).

The problem is that the mark is computed at the marginal price on a 5% float, and applied to a position that may itself be larger than the entire float.

#### Worked example 3: the mark and the money

Illustrative throughout. A seed fund bought **50,000,000 NIM at \$0.04** in the private round, writing a cheque for **\$2,000,000**.

**At listing.** NIM opens at \$1.00. The fund's position is 50,000,000 × \$1.00 = **\$50,000,000**. On a \$2,000,000 cost, that is a **25x** mark. This number is real in the sense that it is computed correctly from a real price. It is also completely unrealisable on the day it is printed, for two independent reasons:

1. **The tokens are locked.** They vest with a 12-month cliff  -  meaning nothing releases at all for a year  -  followed by linear release out to month 48.
2. **The position is bigger than the market.** The fund holds 50,000,000 NIM. The *entire circulating float* is 50,000,000 NIM. If the tokens were unlocked and the fund tried to sell them all at once, it would be selling an amount equal to 100% of everything that trades, into a book with \$50,000 per price step. There is no price at which that clears anywhere near \$1.00.

**What actually gets realised.** Suppose the fund sells steadily as it unlocks, and suppose the market is neither generous nor cruel  -  the circulating market cap simply stays flat at \$50,000,000 as supply arrives (we derive that price path in the next section). Then:

- At the month-12 cliff, 25% of the allocation  -  12,500,000 NIM  -  releases at a price of **\$0.174**. Sold: **\$2,175,000**.
- The remaining 37,500,000 NIM releases evenly across months 13 to 48, during which the price grinds from \$0.17 down to \$0.05. Selling evenly across that path realises an average of roughly **\$0.086 per token**: about **\$3,225,000**.
- Total cash actually received: about **\$5,400,000**.

![Two panels comparing a fifty million dollar paper mark against five point four million dollars of realised cash, and that realised cash against a two million dollar cost basis](/imgs/blogs/the-low-float-high-fdv-game-6.webp)

So the fund turned \$2,000,000 into roughly \$5,400,000  -  a **2.7x** return over about five years from the original investment. That is a decent outcome. It is also **one-ninth of the \$50,000,000 mark** that was reported when the token listed.

**The intuition this teaches:** a mark is a price multiplied by a quantity. Money is the price you can actually sell at, multiplied by how much you can actually move. On a thin float those two numbers are not close, and the gap between them is invisible in every report until the selling starts.

There is an uncomfortable corollary. Notice that the fund's *best available strategy* under this arithmetic is to sell earlier and faster than the vesting schedule's midpoint, because the price is highest at the start. Nobody has to be cynical for that to be true; it falls out of the numbers. This is why unlock schedules generate the price behaviour they do, which is the subject of the companion piece, [unlock cliffs and the supply overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade).

### The project treasury: a balance sheet made of its own token

Most token projects are governed by a foundation or DAO holding a large treasury allocation  -  in our illustrative NIM, 220,000,000 tokens, or 22% of supply. At the listing price of \$1.00, that treasury is "worth" \$220,000,000.

This matters operationally, not just cosmetically. A treasury marked at a large number can fund grants, pay contributors partly in tokens, offer liquidity mining incentives, and negotiate partnerships from a position of apparent strength. A treasury marked at \$11,000,000 cannot do those things at the same scale.

The same problem applies as to the VC: the treasury's tokens are a claim on a market that cannot absorb them. Foundations that actually try to convert treasury tokens into operating cash discover exactly how thin the bid is, which is why so many of them either sell through over-the-counter desks at a discount or borrow against the tokens rather than sell them. Both routes are covered in [token foundations and treasuries: the on-chain central banks](/blog/trading/crypto-players/token-foundations-and-treasuries-the-on-chain-central-banks) and [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price).

### The market maker: paid in optionality on the headline

A newly listed token needs someone to quote continuous two-sided prices, or the book is empty and the exchange has an embarrassing asset on its hands. That is the market maker's job, and what one actually does day to day is described in [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does).

The dominant compensation structure in crypto is the **token loan plus call options** arrangement: the project lends the market maker a quantity of tokens to quote with, and grants call options  -  the right, not the obligation, to buy tokens at a fixed price later  -  struck at levels above the listing price. The market maker returns the loan at the end of the term or exercises the options.

Two things follow. First, the market maker's upside is a function of where the price goes, so its payoff is not neutral to the headline number even though its day job is quoting fairly. Second  -  and this is the structural point  -  the strikes are typically set relative to the listing price, which was itself set on a thin float. The full mechanics, including how the same structure can align or misalign incentives depending on how it is written, are in [the loan-plus-options deal: how market makers get paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid).

### The exchange: volume is the business

An exchange earns fees on volume. A new listing with a compelling story and a violent price reaction generates enormous volume relative to its market cap, because a thin float means the same tokens change hands repeatedly. That is revenue.

Exchanges are not passive venues in this system  -  they choose what lists, when, and with what fanfare, and a listing is itself a price event. That is developed at length in [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues).

### The narrative layer: a big number is a story

"New project valued at \$1 billion" is a headline. "New project with \$50 million of tradable tokens" is not. Media, research posts, conference stages and paid promotional accounts all run on the larger figure because it is the more interesting one. How that layer is organised and paid is the subject of [influencers, KOLs and the narrative-for-hire machine](/blog/trading/crypto-players/influencers-kols-and-the-narrative-for-hire-machine).

### The day-one buyer: the only participant paying cash for the mark

Everyone above receives the headline number as a mark, an option strike, a fee stream, or a story. Exactly one participant converts it into cash out of their own bank account.

If you buy NIM at \$1.00, you are paying a price set by a \$1,000,000,000 valuation, and receiving tokens that are part of a 50,000,000-token float. You have paid the full valuation for a claim on a project whose supply will grow twentyfold over the next four years. You are not being cheated  -  the schedule was public. You are simply on the other side of every incentive in the diagram.

This is why the phrase "exit liquidity" gets used, and why I would rather you understood the mechanism than the slogan. Nobody has to conspire to make you exit liquidity. You become exit liquidity by buying the only part of the supply that is for sale, at a price computed on the part that is not.

## The aftermath: what happens when the float catches up

Now the part that explains the two-year bleed.

### The schedule was always public

![A timeline of token unlocks from month zero to month forty-eight showing circulating supply rising from five percent to one hundred percent with a cliff at month twelve](/imgs/blogs/the-low-float-high-fdv-game-7.webp)

Our illustrative NIM has a completely ordinary vesting structure:

- **Airdrop, 50,000,000 tokens (5%)**  -  liquid at listing.
- **Foundation + ecosystem, 500,000,000 tokens (50%)**  -  released linearly over 48 months, about **10,400,000 tokens per month** from day one.
- **Team + investors, 450,000,000 tokens (45%)**  -  a **12-month cliff** releasing 112,500,000 tokens in a single day at month 12, then the remaining 337,500,000 linearly to month 48, about **9,400,000 tokens per month**.

Running that forward gives the circulating supply at each milestone:

| Month | Circulating supply | Float | What happened |
| --- | --- | --- | --- |
| 0 | 50,000,000 | 5.0% | Listing |
| 6 | 112,500,000 | 11.3% | Foundation and ecosystem drip |
| 12 | 287,500,000 | 28.8% | **The cliff: +112,500,000 in one day** |
| 24 | 525,000,000 | 52.5% | Both engines running |
| 36 | 762,500,000 | 76.3% |  -  |
| 48 | 1,000,000,000 | 100% | Fully diluted |

Every single one of those numbers was knowable on listing day. None of them are surprises. They are in the project's documentation, and third-party trackers publish them as calendars.

### The arithmetic of catching up

Here is the sentence that explains the two-year bleed, and I want it in isolation because it is the whole post compressed:

> If the total dollars willing to hold a token stay the same while the number of tokens goes up 20x, the price must go down 20x.

That is not a market opinion. It is division. Market cap equals price times supply; hold market cap constant, multiply supply by 20, and price divides by 20.

#### Worked example 4: the price path at a flat market cap

Illustrative. Suppose the market decides NIM is worth exactly \$50,000,000 of circulating value  -  the same as on listing day  -  and never changes its mind. Not a crash. Not a loss of faith. The *identical* dollar valuation, held perfectly steady for four years.

Price at each milestone is simply \$50,000,000 divided by the circulating supply:

| Month | Circulating supply | Price at a flat \$50M cap | Change from listing |
| --- | --- | --- | --- |
| 0 | 50,000,000 | \$1.000 |  -  |
| 6 | 112,500,000 | \$0.444 | −56% |
| 12 | 287,500,000 | \$0.174 | −83% |
| 24 | 525,000,000 | \$0.095 | −90% |
| 36 | 762,500,000 | \$0.066 | −93% |
| 48 | 1,000,000,000 | \$0.050 | **−95%** |

![Two stacked panels showing circulating supply rising from fifty million to one billion tokens while the price required to hold a flat fifty million dollar market cap falls from one dollar to five cents](/imgs/blogs/the-low-float-high-fdv-game-8.webp)

A **95% drawdown** with the market cap completely unchanged. The project could ship every roadmap item, grow users tenfold, and generate real revenue, and if that success merely maintained rather than grew the circulating valuation, a day-one buyer would still be down 95%.

The animation below shows the same thing as a single moving picture. The dashed frame is the fully diluted valuation  -  fixed by total supply, it never moves. The shaded bar is the part that actually trades, growing into it.

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="A fixed fully diluted valuation frame with the circulating float growing from five percent to one hundred percent over forty-eight months, while the price required to hold a flat fifty million dollar market cap falls from one dollar to five cents" style="width:100%;height:auto;max-width:760px">
<style>
.fv-ttl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.fv-row{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:end}
.fv-tick{font:600 12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.fv-val{font:700 13px ui-monospace,SFMono-Regular,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
.fv-red{font:700 13px ui-monospace,SFMono-Regular,monospace;fill:var(--danger,#dc2626);text-anchor:middle}
.fv-note{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.fv-frame{fill:none;stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:7 5;rx:6}
.fv-fill{fill:var(--accent,#6366f1);opacity:.85;rx:3;transform-box:fill-box;transform-origin:0 50%}
.fv-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.fv-mark{stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:4 4;transform-box:fill-box}
@keyframes fv-grow{0%{transform:scaleX(.05)}12.5%{transform:scaleX(.113)}24.5%{transform:scaleX(.175)}25.5%{transform:scaleX(.288)}50%{transform:scaleX(.525)}75%{transform:scaleX(.763)}92%{transform:scaleX(1)}97%{transform:scaleX(1);opacity:.85}99%{opacity:.12}100%{transform:scaleX(.05);opacity:.85}}
@keyframes fv-sweep{0%{transform:translateX(0)}12.5%{transform:translateX(67px)}25%{transform:translateX(135px)}50%{transform:translateX(270px)}75%{transform:translateX(405px)}92%,97%{transform:translateX(540px)}99%{opacity:0}100%{transform:translateX(0);opacity:1}}
.fv-fill{animation:fv-grow 14s ease-in-out infinite}
.fv-mark{animation:fv-sweep 14s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.fv-fill{animation:none;transform:scaleX(1)}.fv-mark{animation:none;transform:translateX(540px)}}
</style>
<text class="fv-ttl" x="150" y="24">The frame never moves. The float grows into it.</text>
<rect class="fv-frame" x="148" y="40" width="544" height="42"/>
<rect class="fv-fill" x="150" y="42" width="540" height="38"/>
<text class="fv-row" x="140" y="66">supply</text>
<line class="fv-axis" x1="150" y1="100" x2="690" y2="100"/>
<line class="fv-mark" x1="150" y1="36" x2="150" y2="192"/>
<text class="fv-tick" x="150" y="120">M0</text>
<text class="fv-tick" x="218" y="120">M6</text>
<text class="fv-tick" x="285" y="120">M12</text>
<text class="fv-tick" x="420" y="120">M24</text>
<text class="fv-tick" x="555" y="120">M36</text>
<text class="fv-tick" x="690" y="120">M48</text>
<text class="fv-row" x="140" y="148">float</text>
<text class="fv-val" x="150" y="148">5%</text>
<text class="fv-val" x="218" y="148">11%</text>
<text class="fv-val" x="285" y="148">29%</text>
<text class="fv-val" x="420" y="148">53%</text>
<text class="fv-val" x="555" y="148">76%</text>
<text class="fv-val" x="690" y="148">100%</text>
<text class="fv-row" x="140" y="180">price</text>
<text class="fv-red" x="150" y="180">$1.00</text>
<text class="fv-red" x="218" y="180">$0.44</text>
<text class="fv-red" x="285" y="180">$0.17</text>
<text class="fv-red" x="420" y="180">$0.095</text>
<text class="fv-red" x="555" y="180">$0.066</text>
<text class="fv-red" x="690" y="180">$0.05</text>
<line class="fv-axis" x1="150" y1="204" x2="690" y2="204"/>
<text class="fv-note" x="150" y="228">Illustrative. Circulating market cap is held flat at $50M for all 48 months.</text>
<text class="fv-note" x="150" y="250">Supply goes up 20x, so price goes down 20x  -  a 95% decline with no change in market cap.</text>
<text class="fv-note" x="150" y="272">The step at month 12 is the team-and-investor cliff: 112,500,000 NIM released in one day.</text>
</svg>
<figcaption>Illustrative. The fully diluted valuation frame is fixed by total supply; only the shaded part actually trades. As the float grows from 5% to 100%, holding the circulating market cap flat at \$50M requires the price to fall from \$1.00 to \$0.05.</figcaption>
</figure>

**The intuition this teaches:** in a low-float launch, "the price went down" and "the project failed" are almost unrelated statements. The price has a job to do that has nothing to do with the product  -  it has to make room for supply.

For the price to be *flat* over those four years, the circulating market cap would have to grow from \$50,000,000 to \$1,000,000,000. In other words, the project must grow into its listing-day FDV *just to keep the day-one buyer at breakeven*. That is the trap in one sentence. The FDV is not a target that success will eventually justify  -  it is the hurdle that has to be cleared before the first buyer stops losing money.

#### Worked example 5: what the same path did for the seed round

Now put the earlier cap table next to that price path. Illustrative.

At the very bottom of the schedule  -  month 48, price \$0.05, a 95% drawdown from listing  -  here is where each participant stands per token:

| Participant | Entry price | Price at month 48 | Return on entry |
| --- | --- | --- | --- |
| Seed investor | \$0.04 | \$0.05 | **+25%** |
| Series A investor | \$0.15 | \$0.05 | −67% |
| Day-one buyer | \$1.00 | \$0.05 | **−95%** |

The seed round is still *profitable* at a price that has destroyed the public buyer, because its entry was 25 times lower. Note that this is the return measured at the worst point on the path; the seed fund does not have to sell there, and in Worked Example 3 we computed that selling steadily along the whole path realised 2.7x rather than 1.25x.

And notice the Series A. At \$0.15 entry  -  still a large discount to the \$1.00 listing  -  the later private round is *also* underwater at month 48. This is worth sitting with, because it complicates the simple "VCs versus retail" story. The structure does not reward all insiders equally. It rewards *early* entry and *early* exit, and it punishes everyone whose entry price was set close to the listing price, whether they are a fund or a person.

**The intuition this teaches:** the question is never "is this an insider or retail". The question is "what price did this holder pay, and when can they sell". A seed cheque at 4 cents and a public buy at a dollar are not the same asset with different labels  -  they are different assets.

### Where the selling pressure actually comes from

One more piece of mechanism, because "unlocks cause selling" is too crude to be useful.

An unlocked token does not automatically get sold. What an unlock does is convert a locked position into a *decision*. The holder now faces a choice they did not have before, and the arithmetic above tells you which way most of them lean:

- **A cost basis far below the market.** A seed holder at \$0.04 looking at \$0.17 is up more than 4x and knows the supply schedule as well as you do.
- **A position larger than the float.** Selling gradually is not caution, it is necessity  -  there is no other way to exit.
- **A fiduciary clock.** Funds have fund lifetimes, LPs who want distributions, and internal risk limits on single-name concentration.
- **A visible future.** Every holder can see the same unlock calendar. Anyone who expects others to sell into month 24 has a reason to sell into month 23.

That last point is the reflexive one, and it is why the price weakness often arrives *before* the unlock rather than on the day. Traders who position ahead of a known supply event are running a supply-overhang trade, and the professional version of it  -  including how to hedge it with perpetual futures rather than by selling spot  -  is the whole subject of [unlock cliffs and the supply overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade).

<!--RESEARCH_SLOT-->

## How it shows up in price

Strip away the numbers and the structure produces a recognisable set of price behaviours. None of these are guarantees  -  they are patterns with mechanical causes, and knowing the cause is what lets you tell a real one from a coincidence.

### The listing pop that does not hold

A listing on a large venue concentrates enormous attention on a book that is thin by construction. The result is a violent upward move on modest dollar volume, exactly as computed in Worked Example 2. The move is real; the depth underneath it is not. When the attention rotates to the next listing  -  and it always rotates, usually within days  -  the same thin book that amplified the move up amplifies the move down, because there is no more resting demand below than there was resting supply above.

The tell is the ratio of **volume to depth**. A token trading many multiples of its order-book depth per day is not being accumulated; it is being churned by short-horizon flow.

### The bid-side vacuum

In a normal asset, a falling price attracts value buyers. In a low-float token with a heavy unlock calendar, a falling price attracts *arithmetic*: every potential buyer can compute that supply is still growing, so the rational patient buyer waits. This produces the characteristic "no bid" feel  -  the book has plenty of asks and almost nothing beneath the spread, and small sells produce disproportionate downticks.

### The pre-unlock drift

Because the schedule is public, the market front-runs it. Price weakness clustering in the two to four weeks *before* a large cliff, followed by a relief bounce on the day itself, is a common signature  -  the supply everyone feared has arrived and the uncertainty is resolved. Perpetual futures funding rates often go negative into the same window, as traders pay to be short.

### The market-cap illusion in reverse

Late in the schedule, something perverse happens: the token's *market cap* can be rising while its *price* falls, because supply is growing faster than the price is dropping. A holder reading price feels a disaster; an analyst reading market cap sees growth. Both are correct. This is why comparing two tokens on price performance alone is close to meaningless if their floats differ, and why any serious comparison should be done on circulating market cap or on FDV  -  consistently, one or the other, never mixed.

### The derating of the whole cohort

When a structure disappoints repeatedly, buyers eventually stop paying for it. That shows up not as a crash in one token but as a *derating*: new launches with similar structures open at lower multiples, because the marginal buyer has learned to discount the headline. This is the healthy market response, and it is slow, because each cycle brings in participants who have not seen the previous one.

<!--MARKETS_SLOT-->

## Common misconceptions

### "A low market cap means there is room to grow"

This is the single most expensive misreading in the category. A \$50,000,000 market cap sitting under a \$1,000,000,000 FDV is not a small project  -  it is a large project with a small door. The number that tells you how much room there is to grow is the FDV, because that is what the market is already implying the whole thing is worth. Compare FDV to FDV when you compare projects, or compare market cap to market cap. Never compare one token's market cap to another's FDV.

### "FDV is a fake number, only circulating supply matters"

The opposite error, and popular among people who have been burned by the first one. FDV is not fake; it is a claim on the future that is contractually scheduled to become real. Every locked token has an owner, a cost basis and a release date. Ignoring them is exactly the mistake that produces the two-year bleed. The correct treatment is neither to worship FDV nor dismiss it, but to ask: *what has to be true for the market to absorb this supply, and is there any evidence of that happening?*

### "The unlock is priced in"

Sometimes it is. Often it is partially priced in and the market has mis-sized it. "Priced in" is a claim about the aggregate positioning of every participant, and you have no direct way to observe that. What you *can* observe is whether the price weakened into the event, whether perpetual funding went negative, and whether the tokens actually moved to exchanges afterwards. Treat "priced in" as a hypothesis you can test, not a reason to skip the check.

### "Low float is a scam invented to fleece retail"

This is emotionally satisfying and analytically useless. The structure has boring, legitimate explanations that account for most of it: teams want long vesting to align contributors; exchanges want a listing that does not immediately collapse; regulators and lawyers prefer restricted distributions to broad public sales; and a project with four years of unvested supply has four years of runway in token form. It is entirely possible to arrive at a 5% float through a sequence of individually reasonable decisions. That is precisely what makes it durable  -  and precisely why calling it a scam does not help you avoid it.

### "If the team is good and the product works, the price will follow"

Worked Example 4 is the refutation. A project can execute perfectly and still deliver a 95% price decline to a listing-day buyer, because the price has to do supply-absorption work that has nothing to do with product quality. Product quality determines whether the *circulating market cap* grows. Whether that growth outruns supply growth is a separate question with its own arithmetic  -  and it is the question that determines your return.

### "You can just wait for the unlocks to finish and buy then"

A reasonable instinct, and better than buying at listing, but not free. Two wrinkles. First, "finished" is rarer than it sounds  -  many tokens have ongoing emissions (staking rewards, liquidity incentives) that continue after the vesting schedule ends, so supply never stops growing. Check whether the protocol is inflationary as well as whether the vesting is complete. Second, by the time supply has fully arrived, the attention has usually gone, and a token can trade at a depressed valuation for a very long time regardless of fundamentals. Patience solves the supply problem, not the demand problem.

## Defending yourself: six things you can check before you buy

None of this requires special access, paid tools, or on-chain skills. All six are public, free, and take about ten minutes. But before the checklist, one more piece of arithmetic  -  the one you will use most often.

#### Worked example 6: comparing two tokens without fooling yourself

Illustrative. Two tokens are on your screen.

| | Token A | Token B |
| --- | --- | --- |
| Price | \$2.00 | \$0.50 |
| Circulating supply | 100,000,000 | 40,000,000 |
| Total supply | 200,000,000 | 2,000,000,000 |
| **Circulating market cap** | **\$200,000,000** | **\$20,000,000** |
| **FDV** | **\$400,000,000** | **\$1,000,000,000** |
| **Float** | **50%** | **2%** |

The instinctive read is that B is the cheap one: a \$20 million market cap next to A's \$200 million looks like a tenth of the size, and therefore a tenth of the valuation to grow from.

The correct read is the reverse. On the number that describes the whole project, **B is 2.5 times more expensive than A**  -  \$1,000,000,000 of FDV against \$400,000,000. B is not a small project. It is the larger valuation of the two, wearing a small market cap because 98% of it is locked.

Two follow-on calculations make it concrete:

- **For B to be as cheaply valued as A**  -  same FDV of \$400,000,000  -  B's price would need to fall to \$400,000,000 ÷ 2,000,000,000 = **\$0.20**, a 60% decline from here.
- **For a buyer of B at \$0.50 to break even once fully diluted**, the circulating market cap must reach \$0.50 × 2,000,000,000 = \$1,000,000,000. That is **50 times** B's current circulating market cap of \$20,000,000. Token A's buyer needs the equivalent multiple of 2x.

Neither token is "good" or "bad" here  -  A might be a worse project. The point is only that the comparison people actually make (B's market cap against A's market cap) is not the comparison that determines their returns.

**The intuition this teaches:** compare like with like. FDV against FDV, or market cap against market cap, and always alongside the float that connects them. Mixing the two is how a \$1 billion valuation gets bought as a \$20 million one.

![A three column checklist matrix of what to check, where to find it, and what a red flag looks like, for float, unlocks, cap table, market maker, volume and treasury](/imgs/blogs/the-low-float-high-fdv-game-9.webp)

**1. Compute the float before you look at the chart.** On any major data site, find circulating supply and total supply and divide. Or read the market-cap-to-FDV ratio directly  -  it is the same number. A ratio under roughly 10% tells you that nine-tenths of the supply is still coming.

**2. Pull the unlock calendar.** Third-party unlock trackers and the project's own tokenomics documentation will give you the schedule. What you are looking for is not "are there unlocks"  -  there always are  -  but *when the first cliff is, and how large it is as a percentage of the current float*. An unlock worth 30% of the existing float is a different event from one worth 3%.

**3. Compute the insider multiple at the listing price.** Fundraising announcements usually disclose the amount raised and sometimes the valuation; the tokenomics page gives the allocation. Divide the listing price by the seed price. If you can compute a 20x or higher multiple for a round that closed eighteen months ago, you know exactly how much room there is between the listing price and the point at which early holders are still profitable. That number is your downside reference, not zero.

**4. Find out who is making the market and how they are paid.** Listing announcements and, increasingly, project disclosure posts name the market maker. What matters is whether the arrangement is disclosed at all. A project that publishes its market-making terms  -  loan size, option strikes, term  -  is telling you something real about how it intends to behave. Silence is not proof of anything, but disclosure is genuine evidence.

**5. Compare volume to depth, not volume to market cap.** Open the order book on the venue where most of the volume trades and look at how many dollars sit within 2% of the mid price on each side. Then compare that to the reported daily volume. A token doing \$40,000,000 a day against \$200,000 of two-sided depth is telling you the volume is not accumulation. Real methods for reading this, including on-chain, are in [reading the tape: defending yourself as retail](/blog/trading/crypto-players/reading-the-tape-defending-yourself-as-retail) and [whales, smart money and on-chain wallet watching](/blog/trading/crypto-players/whales-smart-money-and-on-chain-wallet-watching).

**6. Watch the treasury and insider wallets on-chain.** Token allocations usually live in identifiable contracts and wallets. What you are watching for is a specific, observable event: transfers from a vesting contract or treasury into an exchange deposit address. That is not proof of selling, but it is the necessary precondition for it, and it is visible to anyone.

And one habit that is worth more than all six: **read the tokenomics page before the price chart.** The chart tells you what a small number of people have paid recently. The tokenomics page tells you what everyone else is contractually entitled to do next. For the general skill of reading a token's ownership structure, [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) goes deeper on the same material, and [the lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) walks the full timeline from private round to full dilution.

## When this matters to you

If you never buy a newly listed token, this post is still useful, because the same structure shows up wherever a small tradable slice sets a price for a large illiquid one  -  closed-end funds trading away from net asset value, employee stock in a private company marked at the last round, illiquid credit marked off a handful of trades. The general lesson transfers: **a mark is a price times a quantity, and the price is only as trustworthy as the market that produced it.**

If you do buy newly listed tokens, the practical takeaway is narrower and more concrete. You are not being asked to avoid the category. You are being asked to *price it correctly*  -  to look at the FDV rather than the market cap when judging whether something is expensive, to read the supply schedule as a series of dated, sized events rather than a vague future, and to notice that a price which merely stands still while supply doubles has, in the only sense that matters to your account, already fallen 50%.

## Sources & further reading

- [Low Float & High FDV: How Did We Get Here?](https://www.binance.com/en/research/analysis/low-float-and-high-fdv-how-did-we-get-here)  -  Binance Research, May 2024. The report's market-cap-to-FDV comparison for 2024 launches is based on data as of April 14, 2024; it reports a 12.3% ratio and discusses low circulating supply, unlocks, and their implications.
- [Low Float & High FDV: How Did We Get Here?  -  report PDF](https://public.bnbstatic.com/static/files/research/low-float-and-high-fdv-how-did-we-get-here.pdf)  -  Binance Research, May 2024. The underlying report and methodology notes, including the cited CoinMarketCap and Token Unlocks data snapshots.
- [CoinGecko Supply Methodology](https://support.coingecko.com/hc/en-us/articles/32294647667865-CoinGecko-Supply-Methodology)  -  definitions and treatment of max, total, outstanding, and circulating supply; updated November 17, 2025.
- [Definition of Fully Diluted Valuation](https://www.coingecko.com/en/glossary/fully-diluted-valuation)  -  CoinGecko's definition of FDV as a theoretical market capitalization based on current price and full supply; updated October 26, 2023.

This is educational material about market structure, not investment advice, and nothing here is a recommendation to buy or sell anything.

<!--SOURCES_SLOT-->
