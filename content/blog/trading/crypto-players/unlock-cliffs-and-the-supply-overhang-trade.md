---
title: "Unlock cliffs and the supply overhang trade: reading the calendar that everyone can see"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "Token unlocks are the rarest thing in crypto: selling pressure published months in advance. This is how a cliff works from zero, how to turn an unlock date into a dollar number, why the identity of the unlocker matters more than the size, and what three real 2024 cliffs actually did to price."
tags:
  [
    "crypto",
    "tokenomics",
    "token-unlocks",
    "vesting",
    "supply-overhang",
    "market-structure",
    "crypto-vc",
    "risk-management",
    "arbitrum",
    "celestia",
  ]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 55
---

> [!important]
> **TL;DR**  -  A token unlock is the one form of selling pressure that is published in advance, which makes it the cleanest laboratory in crypto for watching a market price a known future event badly.
>
> - Most tokens launch with a small **float** (the tradeable slice) sitting under a large **locked** balance. The unlock calendar is a dated schedule for when that locked balance can move.
> - The number that matters is not "how many tokens" but **how many days of real volume** the unlock represents, and **how much the float grows**  -  Arbitrum's 16 March 2024 cliff released **1,111,750,000 ARB**, which the Arbitrum DAO's own forum described as set to "nearly double the number of tokens in circulation."
> - **Who** unlocks changes everything. A seed fund at a 50x cost basis, a team on a four-year schedule, and a foundation treasury are three completely different sellers holding the same token.
> - Because the date is public, much of the move happens **before** it. All three of the 2024 cliffs studied here fell into the date  -  and then one of them (Celestia) rose 67% in the following month. An overhang is a pressure, not a verdict.
> - The honest defensive move for a retail holder is not to trade the unlock. It is to **know the date exists before you buy**, and to price the dilution into what you are willing to pay.

There is a particular kind of chart that shows up over and over in crypto, and once you have seen it you cannot unsee it. A token drifts lower for five or six weeks. Nothing is obviously wrong. The protocol is shipping, the Discord is busy, the developers are posting. And then on some Tuesday the token gaps down, bounces, and the timeline fills with people asking what just happened.

What happened was on a calendar. It had been on that calendar for a year. Anyone could have looked it up in the project's own documentation, on a public web page, months before it mattered.

This is the strangest feature of token markets, and it is worth sitting with for a moment. In equities, the big supply events  -  a secondary offering, a lockup expiry after an IPO, an insider selling programme  -  are partly disclosed and partly not, and the disclosure often arrives with the event. In crypto, the entire future supply schedule is frequently published at launch, down to the month, sometimes down to the block. The market is handed a dated list of exactly when hundreds of millions of dollars of new sellable tokens will appear.

And it still gets mispriced, over and over, in both directions.

This is not a niche corner of the market. Unlock-tracker data (Tokenomist, BlockEden) put more than **$6 billion** of tokens unlocking across **144 projects** in March 2026 alone  -  roughly three times the typical monthly figure  -  and around **$1.99 billion** scheduled between 1 July and 1 August 2026. Whatever else is true about token markets, a steady multi-billion-dollar stream of newly sellable supply arriving on a published timetable is a permanent feature of them.

This post is about that gap. We are going to build the machinery from zero  -  what "locked" actually means, who is holding the locked tokens, what a cliff is and why it exists at all  -  and then get to the part practitioners actually argue about: how much of a known unlock is already in the price, how you would size the overhang in dollars rather than vibes, how a large holder gets out without simply dumping, and what the record of some real, completed, verifiable unlocks looks like.

![The supply overhang mental model: a large locked stack of token supply sitting above a small circulating float, with the unlock schedule as the gate between them](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-1.webp)

The diagram above is the mental model, and almost everything else in this post is an elaboration of it. A token's price is set by a small pool of tradeable supply  -  the float. Sitting above that pool is a much larger reserve of tokens that exist, are counted in the fully diluted valuation, and are owned by identifiable people, but cannot currently be sold. The unlock schedule is the gate between the two. The "supply overhang" is simply the reserve, weighted by how soon it can come through the gate and how motivated its owners are to walk through it.

A quick framing note before we start. This is an explanation of a market mechanism, not advice, and emphatically not a trading strategy. Most of the practical value here is defensive: understanding why a token you own might have a structural seller queued up behind it, and being able to check before rather than after.

## The foundations: what "locked supply" actually means

If you have never looked at a token's supply page, this section is the whole vocabulary. If you have, skim to the part about where the lock physically lives, because that is the bit that most people get wrong.

### Supply, float, and the two ways to measure a token's size

Start with the simplest possible object: a token with a fixed number of units.

**Total supply** is how many units exist. If a project mints one billion tokens at launch, total supply is 1,000,000,000. Some tokens can mint more later (inflationary, like a staking reward); some cannot. **Max supply** is the ceiling, if there is one.

**Circulating supply** is how many of those units are actually free to trade right now. This is a much smaller number at launch, and it is the source of endless argument, which we will come back to.

**Float** is the informal word for circulating supply. It is borrowed from equities, where "free float" means the shares actually available to the public rather than locked up with founders and strategic holders. It means the same thing here: the tokens that can meet a buyer.

Now the two valuation numbers, which beginners routinely mix up and which the entire post depends on:

- **Market capitalisation** = price x circulating supply. This is what the tradeable slice is worth.
- **Fully diluted valuation (FDV)** = price x total supply. This is what every token that will ever exist would be worth if they all traded at today's price.

Here is the thing to internalise. When a token launches with 15% of supply circulating, the market cap is 15% of the FDV. The price is being set by trading in that 15%. The other 85% is valued at the same price on paper  -  but nobody has tested whether the market would actually pay that price for it. The unlock schedule is the process by which that test gets run, one tranche at a time.

An analogy that gets the intuition across. Imagine a town with a thousand identical houses, but only fifty are ever for sale in a given year. Prices are set by those fifty transactions. Now suppose the other nine hundred and fifty are owned by a developer who has publicly announced they will release two hundred houses onto the market next March. Nothing about the town has changed. The houses were always there. But the price discovered by fifty sales a year is not obviously the price that survives two hundred sales in a month, and everyone in the town knows the date.

### Vesting, cliffs, and the shapes a release schedule can take

**Vesting** is the process by which someone's allocation becomes theirs to sell over time. The word comes from employee equity, and it means the same thing: you have been promised something, and you earn the right to use it gradually.

**TGE**  -  the token generation event  -  is the moment the token is created and usually first trades. Every schedule is measured from it.

A **cliff** is a period during which nothing vests at all, ending in a single moment when a chunk vests at once. A "one-year cliff" means: for twelve months you can sell nothing, and on the anniversary a block of tokens becomes sellable in one step. The word is well chosen. The supply curve is flat and then it falls off an edge.

**Linear vesting** is the opposite: a steady trickle, typically monthly or per-block, with no discontinuity.

Almost every real crypto schedule is a combination  -  a cliff followed by linear vesting. The industry-standard shape, inherited more or less directly from Silicon Valley employee equity, is a **one-year cliff followed by three years of monthly vesting**. That single shape explains an enormous fraction of the price action in the token's second year of life.

![Three vesting shapes plotted as cumulative supply unlocked against months since TGE: a pure cliff, pure linear vesting, and the standard cliff-then-linear schedule](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-2.webp)

The chart makes the point that the totals are identical and the experiences are not. All three schedules release 100% of the allocation over 48 months. The pure cliff concentrates the entire event into one day. The linear schedule spreads it so thin that on any given day the new supply is a rounding error against normal volume. The standard cliff-then-linear does both: one large discontinuity, then a permanent monthly drip that never stops for three years.

That permanent drip is underrated. People fixate on cliff day because it is dramatic. But a schedule that releases 25% at month 12 also releases roughly 2% of the allocation *every single month* for the next 36 months. Cliff day is a headline. The drip is the weather.

### Where the lock physically lives  -  and why it matters

This is the part that people skip, and it is the difference between a schedule you can verify and a schedule you have to take on faith.

There are two fundamentally different ways to lock a token.

**A smart contract lock.** The tokens sit in a vesting contract on-chain. The contract has code that will not release them before a timestamp. You can read that code. You can read the balance. You can compute, with certainty, how many tokens can move on a given date and to which addresses. This is verifiable in the strongest sense available in this industry.

**A contractual lock.** The tokens sit in an ordinary wallet controlled by the investor or the team, and a legal agreement says they will not sell before a date. There is no code enforcing anything. The lock is a promise backed by a contract that would have to be litigated.

Both are called "locked" in a tokenomics blog post. They are not the same thing at all. A contractual lock can be broken quietly  -  the tokens can be moved, lent, used as loan collateral, or sold over the counter to someone who inherits the same promise  -  and the on-chain record may show nothing that looks like a violation. This is not a hypothetical concern; disputes about whether tokens described to a community as locked were in fact being moved have been a recurring theme on project governance forums since at least 2023.

The practical consequence for you: when you read that a token is "85% locked," the first question is *locked how*. If the answer is a vesting contract address you can look up, that is a fact. If the answer is a pie chart in a blog post, that is a representation.

### The four kinds of holder sitting behind the gate

An unlock is not an abstract quantity of supply. It is a specific set of people receiving the ability to sell, and they are not interchangeable. Broadly there are four groups.

**Investors.** Funds that bought tokens in private rounds before launch  -  seed, Series A, strategic. Their defining feature is **cost basis**: what they paid. A seed fund that paid three cents for a token now trading at two dollars is sitting on a 66x gain and is profitable at essentially any price you can imagine. A fund that bought the last private round at $1.80 is barely above water. These two funds behave completely differently on the same unlock date, and they are usually on the *same* schedule.

**Team and core contributors.** Founders and employees, cost basis effectively zero, typically on the longest schedule of anyone  -  four years is standard. They have the strongest incentive to sell (this is often their entire net worth) and the strongest reason not to (visible selling by a founder is read as a signal about the project, and on a public ledger it is quite visible).

**Ecosystem and community allocations.** Grants to developers, liquidity mining rewards, incentives for users, tokens lent to market makers. These usually unlock continuously from day one rather than on a cliff, and  -  critically  -  they are *distributed to be used*, which usually means sold. A grant recipient who needs to pay engineers converts tokens to stablecoins. This is a small, constant, structural bid-taker.

**The foundation or DAO treasury.** Often the single largest allocation. Nominally unlocked in the sense that the entity controls it, but not circulating in the sense of being on an exchange. Treasuries generally do not market-sell, because doing so is politically explosive and self-defeating; they transact over the counter, fund grants, or simply sit. Their supply is real but rarely arrives as market pressure. It arrives as *potential* market pressure, which is a different thing that still weighs on sentiment.

Whether the tokens held by that last group should count as "circulating" is exactly the argument that makes circulating-supply figures unreliable. We will return to it when we look at how to read a calendar.

## Why cliffs exist at all

It is worth understanding why anyone designs a schedule with a discontinuity in it, because "the founders were careless" is not the answer and the real answer tells you what the schedule is trying to achieve.

A project launching a token is trying to satisfy three goals that fight each other.

**It needs a price.** A token with no float has no price, no listing, and no ability to be used as a unit of account for anything. Some supply must be liquid on day one.

**It needs that price to be high.** Not out of vanity: the token is a treasury asset, a recruiting currency, a grant instrument, and a marketing surface. A launch that immediately trades to zero damages all four. The reliable way to get a strong opening price is to make the float small. Demand meeting a thin supply produces a high number.

**It needs its insiders aligned for years.** A team that can sell everything on day one has no reason to stay. Investors who can exit at launch are not investors, they are flippers with a longer settlement period.

The cliff resolves all three at once. Launch with a small float, so the price is discoverable and strong. Lock the insiders for a year, so nobody can leave immediately. Then release gradually.

The cost of that resolution is that you have manufactured a dated event. You have taken a large, diffuse quantity of future selling and concentrated it into a single day that you then published. The reason this is worth doing anyway is that the alternative  -  no lock  -  is worse for everyone including the buyers. But it does mean that the tidy launch chart and the ugly month-13 chart are the same design decision, seen from two ends.

![A token unlock calendar laid out as a timeline: TGE, a quiet period, the twelve-month cliff, thirty-six months of monthly vesting, and full dilution](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-3.webp)

There is a second-order effect here that deserves naming. Because the small-float launch produces a high price, it also produces a high FDV  -  and the FDV is the number against which every locked holder marks their position. A seed fund that bought at a $30 million valuation and sees the token open at a $6 billion FDV has, on paper, made 200x. That paper number is what gets reported to their own investors. It is a real number in the sense that it is arithmetic, and an unreal number in the sense that no one has tried to sell 85% of the supply at it. The cliff is the day the arithmetic starts meeting the order book.

How extreme did this get? **Binance Research**, in a widely-cited May 2024 analysis of that year's token launches, put the average ratio of market capitalisation to fully diluted valuation at listing at **12.3%**  -  meaning the typical 2024 launch had under an eighth of its eventual supply actually trading  -  and projected roughly **$155 billion** of tokens scheduled to unlock between 2024 and 2030. Those two numbers together are the entire subject of this post: a very small float sets a price, and a very large queue of supply then has to be absorbed at or below it.

If you want the longer version of how that launch structure is deliberately engineered, the companion piece to this one is [the low float, high FDV game](/blog/trading/crypto-players/the-low-float-high-fdv-game), and the full path from private round to public exit is in [the lifecycle of a token, seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock).

## Turning a date into a number

Here is where we stop describing and start calculating. An unlock date on a calendar is useless. An unlock date converted into "this is X days of the token's real trading volume, arriving in the hands of holders with a cost basis of Y" is a piece of analysis you can act on.

There are three calculations, and you do them in this order.

#### Worked example 1: what a real cliff was worth in dollars

Take the Arbitrum cliff, because its schedule is published by the Arbitrum Foundation and its size was stated in the DAO's own governance forum, which makes it about as verifiable as these things get.

The [Arbitrum Foundation's distribution documentation](https://docs.arbitrum.foundation/airdrop-eligibility-distribution) states an initial supply of 10 billion ARB, with 26.94% (2.694 billion) to team and contributors plus advisors, and 17.53% (1.753 billion) to investors. It also states the schedule directly: "All investor and team tokens are subject to 4 year lockups, with the first unlocks happening one year after the token generation event (3/16/2023) and then monthly unlocks for the remaining three years."

Now do the arithmetic.

```
Team + contributors + advisors    2,694,000,000 ARB
Investors                       + 1,753,000,000 ARB
                                ---------------------
Total locked to insiders          4,447,000,000 ARB

Four-year schedule, first unlock at year 1
  => one year of four vests at the cliff = 25%

Cliff release = 4,447,000,000 x 0.25 =  1,111,750,000 ARB
```

That derived figure  -  1,111,750,000 ARB  -  is exactly the number an [Arbitrum DAO forum proposal](https://forum.arbitrum.foundation/t/proposal-for-change-of-gradual-unlocking-of-arbitrum-tokens-to-sustain-ecosystem-value/14907) used when it argued in June 2023 for slowing the release down: "Rather than unlocking 1,111,750,000 tokens all at once, it is suggested that max of 3% ... of the total unlocked tokens be introduced into circulation each year." So the derivation and the DAO's own reading agree.

Now price it. ARB traded at **$1.88 on 16 March 2024** ([DefiLlama historical price API](https://coins.llama.fi/)):

```
1,111,750,000 ARB x $1.88 = $2,090,090,000
```

**Roughly $2.09 billion of tokens became sellable in one day.** That is the headline number, and on its own it is still not very informative, because $2.09 billion means one thing for Bitcoin and something else entirely for a mid-cap token. The next two calculations are what make it mean something.

*Intuition: the first job is always to convert a token count into a dollar figure at a dated price, because token counts hide the scale.*

#### Worked example 2: days of volume  -  scaling the unlock to liquidity

The single most useful ratio in this whole subject is **days of volume**: how many days of the token's normal trading it would take to absorb the unlock if every token were sold.

Let us do this one on clean illustrative numbers so the arithmetic stays visible, then discuss why the real version is harder.

*Illustrative example.* A token trades at $2.00. An unlock releases 50,000,000 tokens. Reported average daily volume is $40,000,000.

```
Unlock value   = 50,000,000 x $2.00 = $100,000,000
Days of volume = $100,000,000 / $40,000,000 = 2.5 days
```

Two and a half days of volume. That sounds survivable, and if the reported volume were real it probably would be.

Here is the problem: **reported volume is not the volume available to absorb a seller.** A large share of reported exchange volume in crypto is market makers trading with each other, arbitrage bots moving the same inventory between venues, and in some venues outright fabricated print. None of that is a buyer who wants to own the token. (The mechanics of manufactured volume, and how to spot it, are the subject of [wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume).)

So redo it with a haircut. Suppose only 40% of reported volume represents genuine directional interest:

```
Absorbable volume = $40,000,000 x 0.40 = $16,000,000/day
Days of volume    = $100,000,000 / $16,000,000 = 6.25 days
```

The same unlock is now more than six days of real demand, which is a very different picture.

The rigorous version of this calculation abandons volume entirely and uses **order book depth**  -  specifically, how much you could sell before moving the price by some fixed amount, usually 2%. Suppose the book holds $1,200,000 on the bid side within 2% of mid:

```
Unlock / 2% depth = $100,000,000 / $1,200,000 = 83x
```

To sell the whole unlock you would need to consume the entire visible near-touch bid side more than eighty times over, waiting for it to refill in between. That number tells you immediately that the unlock cannot simply be dumped, which in turn tells you the holder will have to do something cleverer  -  which is the subject of a later section.

*Intuition: an unlock is only as dangerous as the liquidity it lands on, and reported volume flatters that liquidity considerably.*

#### Worked example 3: dilution of the float

The third calculation is the simplest and the most quoted: what percentage does the float grow by?

For Arbitrum, the tokens that were genuinely in public hands at TGE were the airdrop to users (11.62% = 1,162,000,000 ARB) and the allocation to DAOs building on Arbitrum (1.13% = 113,000,000 ARB):

```
Approximate pre-cliff float   1,162,000,000 + 113,000,000 = 1,275,000,000 ARB
Cliff release                                                1,111,750,000 ARB

Float growth = 1,111,750,000 / 1,275,000,000 = 0.872  =>  +87%
Post-cliff float = 2,386,750,000 ARB  =>  the float 1.87x
```

An 87% increase. That matches, independently, the language in the DAO forum proposal quoted above, which described the schedule as set to "nearly double the number of tokens in circulation."

Do the same for the other two cliffs we will study in detail later:

- **Pyth.** The [Pyth documentation](https://docs.pyth.network/pyth-token/pyth-distribution) states a total of 10,000,000,000 PYTH and that "the initial circulating supply (November 2023) was 1,500,000,000 (15%) PYTH," with locked tokens unlocking "6, 18, 30 and 42 months after the initial token launch." Four equal tranches of the remaining 8.5 billion gives **2,125,000,000 PYTH** per tranche. Against a 1.5 billion float, that is **+142%**  -  the sellable supply more than doubles in a day.
- **Celestia.** The [Celestia documentation](https://docs.celestia.org/learn/TIA/staking-governance-supply/) gives a genesis supply of 1,000,000,000 TIA, with Early Backers Series A&B at 196,700,000, Early Backers Seed at 159,000,000, and Initial Core Contributors at 176,400,000, each with "33.33% unlocked at year 1." That is about **177 million TIA** at the year-one mark. The supply unlocked at genesis was the 200,000,000 public allocation plus 25% of the 267,900,000 R&D and ecosystem allocation  -  that is 200,000,000 + 66,975,000 = **266,975,000 TIA**. So the year-one cliff increased unlocked supply by about **+66%**. (Note that two similar-looking figures appear here: 267,900,000 is the R&D and ecosystem *allocation*, while 266,975,000 is the total *unlocked at genesis*. They are unrelated.)

*Intuition: float growth is the cleanest single measure of how much the ownership base has to change hands for price to stay flat.*

#### Worked example 4: the monthly drip, quantified

Cliff day gets the attention. The arithmetic says the drip behind it deserves just as much.

Return to Arbitrum. The cliff released a quarter of the insider allocation. The remaining three quarters vest monthly over the following 36 months:

```
Remaining after cliff = 4,447,000,000 - 1,111,750,000 = 3,335,250,000 ARB
Monthly release       = 3,335,250,000 / 36 = 92,645,833 ARB per month
At $1.88              = 92,645,833 x $1.88 = $174,174,000 per month
Per calendar day      ≈ $5.8 million of newly sellable supply, every day
```

Now the observation that reframes the entire event. Multiply the monthly release by twelve:

```
12 x 92,645,833 = 1,111,750,000 ARB
```

That is exactly the size of the cliff. It has to be  -  the schedule releases 25% at year one and the remaining 75% evenly across the next three years, so every subsequent twelve-month window delivers another 25%.

**The market absorbs a second full cliff every year for three years. It simply arrives in 36 slices instead of one.**

Measured against the post-cliff float of 2,386,750,000 ARB:

```
Monthly dilution = 92,645,833 / 2,386,750,000   = 3.9% of float per month
Annual dilution  = 1,111,750,000 / 2,386,750,000 = 47% of float per year
```

And that ignores any distribution out of the 3,528,000,000-token DAO treasury, so 47% is a floor rather than an estimate.

This is why the "the unlock is behind us" relief you see in commentary after a cliff is usually misplaced. The concentrated event is behind you. The cumulative supply is almost entirely ahead of you.

*Intuition: the cliff is not the largest supply event in a token's life  -  it is only the most concentrated one, and the drip behind it is the same size every single year.*

## Who is unlocking is most of the trade

Two unlocks of identical size, identical days-of-volume, identical float dilution, can produce completely different outcomes, and the reason is almost always the identity of the holder.

![A matrix of unlock holder types  -  seed investor, late-round investor, team, ecosystem, foundation treasury  -  showing cost basis, lock length, typical behaviour and pressure on price](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-4.webp)

The variable doing the most work in that table is **cost basis**, and it is worth being concrete about how extreme the spread is.

Private rounds are priced at valuations that bear no relationship to the launch price. A seed round might value the whole network at $20 million. The token might open at a $4 billion fully diluted valuation. That is a 200x paper gain before a single line of the product has been used at scale. What that means practically is that a seed investor's decision at the unlock is not "should I take a profit"  -  it is "is there any price at which I would rather hold this than have the money." When your cost basis is effectively zero, every price is a good price, and the only thing stopping you is whether you think the token goes higher.

Contrast a fund that came in at the last private round at a valuation close to where the token opened. They may be flat or down. Their decision is genuinely difficult. They are not a reliable seller.

This is why the composition of the cap table matters more than the headline unlock size, and why [following the money on a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) is the prerequisite work for reading an unlock at all. An unlock dominated by a 2019-vintage seed round is a different event from an unlock dominated by a 2022 strategic round done at a valuation nobody has seen since.

Two more behavioural notes that the table compresses.

**Teams sell more slowly than investors, but they do sell.** A founder holding their entire net worth in one illiquid asset is in an objectively bad risk position, and selling some of it is financially rational rather than a betrayal. What constrains them is visibility: a founder wallet moving tokens to an exchange is watchable by anyone, and the community will watch. The result is that team selling tends to be routed through structures that are less visible  -  over-the-counter sales, or borrowing against the tokens rather than selling them.

**Funds have their own clocks.** A crypto fund with a fixed life has to return capital to *its* investors at some point, and a token that unlocks in year seven of a ten-year fund is going to be sold on a schedule that has more to do with the fund's structure than with the token's prospects. Similarly, a fund facing redemptions sells what it can sell, which is whatever just unlocked. None of this is a view on the project. It is plumbing. The operating constraints that produce this behaviour are laid out in [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model).

**Some unlocks make a token safer rather than riskier.** This is the nuance that a headline unlock number always destroys. Ecosystem allocations frequently go to market makers as loan inventory, because a desk cannot quote a two-sided market in a token it does not hold. When that tranche unlocks, the tokens do not arrive as selling pressure  -  they arrive as *quotes*, deepening the order book and reducing the impact cost of every subsequent trade, including yours. Read carelessly, that unlock is new supply and therefore bad. Read correctly, it is the mechanism by which the token stops being fragile. The two are indistinguishable on a tracker and obvious once you know the recipient, which is why identifying the holder is the step that carries the most weight. What the desk does with that inventory is the subject of [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does).

#### Worked example 5: what a locked position is actually worth

Here is a calculation that almost nobody outside the funds themselves does, and it explains a great deal about behaviour around unlocks. What is a locked token position actually worth, as opposed to what it marks at?

*Illustrative example, round numbers.* A seed fund bought 20,000,000 tokens at $0.05 in a private round.

```
Cost = 20,000,000 x $0.05 = $1,000,000
```

The token now trades at $2.00. The fund's reported mark:

```
Paper value = 20,000,000 x $2.00 = $40,000,000   (40x)
```

Now the realistic version. The position is on a one-year cliff plus three years of monthly vesting, and we are at the cliff. Only 25%  -  5,000,000 tokens  -  is sellable.

*The sellable quarter.* The order book can absorb roughly 500,000 tokens a day without excessive impact, so this is ten trading days of steady selling. Assume an average realised discount of 8% to the screen price across that period:

```
Realisable now = 5,000,000 x $2.00 x 0.92 = $9,200,000
```

*The still-locked three quarters.* 15,000,000 tokens vest monthly over the next three years. They carry price risk, liquidity risk and schedule risk. Private markets routinely apply an illiquidity discount to positions like this; take an illustrative 40%:

```
Discounted value = 15,000,000 x $2.00 x 0.60 = $18,000,000
```

Add them up:

```
Realistic value = $9,200,000 + $18,000,000 = $27,200,000
Paper mark      = $40,000,000
Ratio           = 68% of the mark
```

The position marks at 40x and is worth something closer to 27x. That 32% gap is the entire reason unlock behaviour looks the way it does. The fund is not being greedy when it sells into the cliff. It is closing a gap between an accounting number and a realisable number, and the gap only closes by selling.

*Intuition: a locked token is worth meaningfully less than price times quantity, and the holder knows it even when the mark does not say so.*

## How the overhang gets priced  -  and why it does not get priced fully

Now the interesting question. The date is public. The size is computable. Every participant with a spreadsheet can do the three calculations above. Why is there anything left to happen on the day?

The efficient-markets answer is that there should not be. A known future increase in supply should be discounted into the price the moment it is known, and the unlock itself should be a non-event  -  the way a dividend that everyone knew about does not surprise anyone when it is paid.

Something close to that does happen, and it produces the characteristic shape below.

![A stylised anticipation curve showing price indexed to the unlock day, drifting down over the forty-five days before and bouncing modestly afterwards](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-5.webp)

The pattern that shape describes is the "sell the rumour, buy the news" structure, and the mechanism is straightforward once you see who is doing what.

If you know that a large seller becomes able to sell on 16 March, and you also know that everyone else knows, then selling on 16 March is the worst possible plan  -  you will be selling alongside the very supply you were trying to get ahead of. So you sell in February. But the person on the other side of that reasoning knows you will sell in February, so they sell in January. The anticipation walks backwards until it is diluted across enough time that it stops being worth front-running further.

By the time the actual date arrives, a great deal of the selling has already been done by people who never held a locked token at all. And then, on the day, something counterintuitive happens: the event passes, the feared thing is now in the past, the shorts who positioned for it take profit by buying back, and the price frequently bounces.

That is the theory. Here is why it does not work cleanly in practice, and the reasons are worth going through because each one is a real market friction rather than a hand-wave.

**Pre-positioning is expensive and uncertainly timed.** To short a token for six weeks into an unlock you need capital, you need a venue, and you need to pay the carrying cost the whole time. If the token rallies 40% on unrelated news in week two, you are stopped out before the thing you correctly predicted happens. Being right about the mechanism and wrong about the path is the normal outcome.

**Borrow is often unavailable, so the trade only exists on perps.** In equities, shorting a stock into a lockup expiry is a well-worn trade with deep borrow markets. In crypto, the ability to borrow a mid-cap token to sell is limited. Nearly all of the shorting happens through perpetual futures, which means the pressure shows up in funding rates and the futures basis rather than in the spot book  -  and it means the spot market is not fully expressing the view.

**Nobody actually knows how much will be sold.** This is the deepest reason. Unlocked is not sold. The schedule tells you the maximum. It tells you nothing about intent. A cliff where every recipient sells immediately and a cliff where every recipient holds are the same event on the calendar and completely different events in the order book, and you cannot distinguish them in advance.

**The marginal buyer disappears exactly when you need them.** The float is thin and the buyer base is reflexive: people buy tokens that are going up. A token drifting down for six weeks into a known unlock loses its momentum buyers precisely during the period when it most needs them. The supply is scheduled; the demand is not. This is a specific instance of a general market pathology explored in [reflexivity: markets that watch themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves).

**Everyone is looking at the same date.** When a single public number coordinates the attention of every participant, the result is not smooth discounting but clustering  -  a crowd all doing the same thing in the same window, which is exactly the setup studied in [information cascades and herding](/blog/trading/game-theory/information-cascades-and-herding-when-rational-traders-follow-the-crowd).

### What would it take to actually know?

You will read confident claims about unlocks  -  that the average unlock produces some specific percentage decline, that cliff unlocks are measurably worse than linear ones. Treat the effect sizes in those claims with considerably more suspicion than the mechanism, because measuring this properly is genuinely hard, and it is worth understanding why before you lean on anyone's number.

The standard tool would be an **event study**: line up many unlock events, measure the return in a window around each one, and compare it against what the token "should" have done. Four problems attack that design.

**Choosing the benchmark decides the answer.** A token's raw return around an unlock is mostly market beta  -  it moved because crypto moved. To isolate an unlock effect you have to subtract an expected return, which means choosing a benchmark: Bitcoin, Ether, a basket of comparable tokens, a sector index. Those choices are not equivalent, and for a high-beta token in a fast-moving month they can flip the sign of the measured effect. Any single number quoted without its benchmark is not interpretable, and most of the numbers you will see are quoted without their benchmark.

**The events are not independent observations.** Tokens launched in the same vintage unlock in the same months. A sample of 2024 unlocks is substantially a sample of one market regime, so the "average unlock" partly just measures that regime. What is presented as two hundred observations may contain only a handful of genuinely independent ones.

**The schedule is not exogenous.** Projects choose their own vesting terms, and they choose them for reasons correlated with everything else about the project. Teams confident of a long runway may lock for longer; teams under pressure sometimes renegotiate. Comparing tokens with different schedules compares projects that differ in many other respects simultaneously, and the schedule may be a symptom rather than a cause.

**Everything happens at once.** In the quarter surrounding a large unlock, a token may also receive a major exchange listing, launch an incentive programme, ship an upgrade, or run a contested governance vote. Attributing the return to the unlock requires those to have been immaterial, which is rarely true and almost never checked.

The honest position is that the *mechanism* is well understood and directly observable  -  the supply is real, the anticipation trade is real, the funding-rate footprint of crowded shorts is real and visible on any venue  -  while the *effect size* is poorly identified. That is an uncomfortable place to sit. It is also where the evidence actually is, and a reader who understands why is much harder to sell a spurious statistic to.

The animation below is the mechanical picture underneath all of this: the reservoir, the gate, and the pool that has to absorb what comes through.

<figure class="blog-anim">
<svg viewBox="0 0 720 470" role="img" aria-label="A locked reservoir of token supply sits above a small circulating float; at the twelve-month cliff a gate opens, blocks of supply fall into the float, the float pool grows and the price marker steps down" style="width:100%;height:auto;max-width:760px">
<title>An unlock cliff releasing locked supply onto a thin float</title>
<style>
.uc-res{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.uc-lock{fill:#2f9e44;opacity:.75}
.uc-pool{fill:#e8590c;opacity:.8;transform-box:fill-box;transform-origin:bottom;transform:scaleY(.34)}
.uc-poolbox{fill:none;stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:5 5}
.uc-gate{stroke:#e8590c;stroke-width:3;stroke-linecap:round}
.uc-blk{fill:#e8590c;opacity:0}
.uc-h{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.uc-s{font:500 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.uc-t{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.uc-hot{font:700 13px ui-sans-serif,system-ui;fill:#e8590c;opacity:.25}
.uc-px{stroke:#e8590c;stroke-width:2.5;stroke-linecap:round}
.uc-pxt{font:700 13px ui-sans-serif,system-ui;fill:#e8590c}
@keyframes uc-open{0%,34%{opacity:1}42%,100%{opacity:.15}}
@keyframes uc-f1{0%,34%{opacity:0;transform:translateY(0)}40%{opacity:.9}46%,100%{opacity:.9;transform:translateY(196px)}}
@keyframes uc-f2{0%,38%{opacity:0;transform:translateY(0)}44%{opacity:.9}50%,100%{opacity:.9;transform:translateY(196px)}}
@keyframes uc-f3{0%,42%{opacity:0;transform:translateY(0)}48%{opacity:.9}54%,100%{opacity:.9;transform:translateY(196px)}}
@keyframes uc-grow{0%,36%{transform:scaleY(.34)}58%,100%{transform:scaleY(1)}}
@keyframes uc-drop{0%,36%{transform:translateY(0)}62%,100%{transform:translateY(52px)}}
@keyframes uc-flash{0%,32%{opacity:.25}40%,100%{opacity:1}}
.uc-gate{animation:uc-open 12s ease-in-out infinite alternate}
.uc-b1{animation:uc-f1 12s ease-in-out infinite alternate}
.uc-b2{animation:uc-f2 12s ease-in-out infinite alternate}
.uc-b3{animation:uc-f3 12s ease-in-out infinite alternate}
.uc-pool{animation:uc-grow 12s ease-in-out infinite alternate}
.uc-pricer{animation:uc-drop 12s ease-in-out infinite alternate}
.uc-hot{animation:uc-flash 12s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){
.uc-gate{animation:none;opacity:.15}
.uc-b1,.uc-b2,.uc-b3{animation:none;opacity:.9;transform:translateY(196px)}
.uc-pool{animation:none;transform:scaleY(1)}
.uc-pricer{animation:none;transform:translateY(52px)}
.uc-hot{animation:none;opacity:1}}
</style>
<text class="uc-h" x="48" y="30">Locked supply  -  850M tokens</text>
<rect class="uc-res" x="48" y="42" width="336" height="132" rx="8"/>
<rect class="uc-lock" x="62" y="56" width="308" height="24" rx="4"/>
<rect class="uc-lock" x="62" y="86" width="308" height="24" rx="4"/>
<rect class="uc-lock" x="62" y="116" width="308" height="24" rx="4"/>
<rect class="uc-lock" x="62" y="146" width="308" height="16" rx="4"/>
<line class="uc-gate" x1="48" y1="192" x2="384" y2="192"/>
<text class="uc-s" x="396" y="197">unlock gate</text>
<rect class="uc-blk uc-b1" x="86" y="150" width="66" height="20" rx="4"/>
<rect class="uc-blk uc-b2" x="182" y="150" width="66" height="20" rx="4"/>
<rect class="uc-blk uc-b3" x="278" y="150" width="66" height="20" rx="4"/>
<rect class="uc-poolbox" x="48" y="256" width="336" height="132" rx="8"/>
<rect class="uc-pool" x="48" y="256" width="336" height="132" rx="8"/>
<text class="uc-h" x="48" y="410">Circulating float</text>
<text class="uc-s" x="48" y="430">the only place price is set</text>
<g class="uc-pricer">
<line class="uc-px" x1="470" y1="120" x2="600" y2="120"/>
<text class="uc-pxt" x="470" y="110">price</text>
</g>
<text class="uc-s" x="470" y="240">the float absorbs</text>
<text class="uc-s" x="470" y="258">what comes through</text>
<text class="uc-s" x="470" y="276">the gate</text>
<text class="uc-t" x="48" y="458">month 11</text>
<text class="uc-hot" x="196" y="458">month 12  -  cliff</text>
<text class="uc-t" x="330" y="458">month 13</text>
</svg>
<figcaption>The cliff does not create tokens  -  they existed all along and were counted in the fully diluted valuation. It moves them across the only line that matters: from a balance nobody can sell into a pool where every sale sets the price. Illustrative quantities.</figcaption>
</figure>

## Getting out without dumping: how large holders actually hedge

If you hold a large locked position, "sell it on unlock day" is the worst available plan and you know it. Everyone has your calendar. The order book cannot take the size. So what do sophisticated holders actually do?

![A matrix of the exits available to a locked token holder  -  spot sale, perpetual short, put option, OTC block sale, and lending to a market maker  -  with the cost, spot price impact and main risk of each](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-6.webp)

The critical insight in that table is that **hedging separates the timing of your price risk from the timing of your liquidity**. You cannot sell a locked token. But in most cases you *can* take on an offsetting short position today, which fixes your economics at today's price, and then unwind it slowly as the tokens actually vest. The tokens stay locked; the risk is transferred.

That is the theory. The costs are where it gets real.

#### Worked example 6: hedging an unlock with a perpetual future

A **perpetual future**  -  "perp"  -  is a derivative that tracks a token's price with no expiry date. To keep it pinned to spot, it uses a **funding rate**: a periodic payment (typically every eight hours) between longs and shorts. When the perp trades above spot, longs pay shorts. When it trades below spot, shorts pay longs. The common baseline quoted on major venues is 0.01% per eight-hour period, which is 0.03% per day, but it moves a long way from that under pressure. The full mechanics of how the perp became crypto's dominant instrument are in [Arthur Hayes, BitMEX and the perp that ate crypto](/blog/trading/crypto-players/arthur-hayes-bitmex-and-the-perp-that-ate-crypto).

*Illustrative example.* You hold 5,000,000 tokens at $2.00  -  a $10,000,000 position  -  vesting in 60 days. You short an equivalent notional of perps today.

Now the crucial detail about the sign. Ordinarily a short *receives* funding, because perps usually trade at a premium in a bull market. But into an anticipated unlock, everyone wants to be short the same token at the same time. The perp is pushed below spot, funding turns negative, and **shorts pay longs**. Being on the crowded side of an obvious trade costs money. Suppose funding averages −0.02% per eight hours, which is −0.06% per day paid by you:

```
Daily cost = $10,000,000 x 0.0006 = $6,000
60 days    = $6,000 x 60 = $360,000
As a % of notional = 3.6%
```

Compare that with dumping 5,000,000 tokens into a thin book at an 8% average discount:

```
Slippage cost = $10,000,000 x 0.08 = $800,000  (8.0%)
```

The hedge costs 3.6% against 8.0% for the dump, and it does not push the spot price down, which matters if you have another 15,000,000 tokens still vesting behind this tranche.

But now the risk that ends careers. A hedge requires **margin**, and margin is a live obligation against an asset you cannot sell. Say the venue requires 20% initial margin:

```
Collateral posted = $10,000,000 x 0.20 = $2,000,000 in stablecoins
```

If the token rallies 50% before your tokens vest, your short loses:

```
Short loss = $10,000,000 x 0.50 = $5,000,000
```

You owe $5,000,000 in variation margin, in cash, right now  -  and the offsetting $5,000,000 gain is sitting in tokens you are legally unable to sell for another two months. Your hedge is economically perfect and operationally fatal. This mismatch  -  a hedge that is right on paper and liquidates you in practice  -  is the single most common way an unlock hedge fails, and it is the same inventory-and-margin problem that professional desks manage full time, described in [inventory risk, hedging and delta-neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality).

*Intuition: a hedge converts price risk into funding cost and margin risk, and the margin risk is the one that bites.*

#### Worked example 7: paying for a put instead

A **put option** gives you the right, but not the obligation, to sell at a fixed price (the **strike**) before a date. Unlike a short, it cannot lose more than the premium you paid, which solves the margin-call problem entirely. If you want the payoff mechanics from scratch, start with [calls, puts and the payoff diagram](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options); the portfolio use is in [hedging with protective puts and collars](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk).

So why does nobody do this? Price.

*Illustrative example.* You want a 90-day at-the-money put on 5,000,000 tokens with spot at $2.00. Altcoin implied volatility is routinely in the region of 90% annualised. A standard rule-of-thumb approximation for an at-the-money option is:

```
Premium ≈ 0.4 x spot x volatility x sqrt(time in years)
        ≈ 0.4 x $2.00 x 0.90 x sqrt(0.25)
        ≈ 0.4 x $2.00 x 0.90 x 0.5
        ≈ $0.36 per token
```

That is **18% of spot** for three months of protection.

```
Total premium = 5,000,000 x $0.36 = $1,800,000
```

$1,800,000 to protect a $10,000,000 position for one quarter  -  against $360,000 for the perp hedge and $800,000 for simply dumping. Insurance on an asset that moves 90% a year is priced like insurance on an asset that moves 90% a year. The very volatility that makes you want the hedge is what makes it unaffordable.

There is a second problem on top of the price: for most tokens there is no listed option market at all. Options liquidity in crypto is concentrated in a handful of the largest assets. For a mid-cap token, that quote is not a screen price  -  it is a bilateral quote from a desk that will charge for the privilege of warehousing a risk nobody else wants.

*Intuition: options convert an open-ended risk into a known cost, and on high-volatility tokens that known cost is large enough that most holders decline it.*

### The two exits that never touch the order book

The remaining routes are the ones professionals actually use, and neither shows up as an obvious sale.

**The OTC block.** You sell the whole tranche to an over-the-counter desk at a negotiated discount to spot  -  say 10 - 15% for a large, illiquid block. The desk warehouses the inventory and works it out gradually. From the chart's perspective, nothing happened on unlock day. The pressure did not disappear; it was transferred to a professional who is much better at distributing it slowly. This is exactly the service described in [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price).

**The loan-plus-option to a market maker.** The holder lends tokens to a market-making firm and writes the firm call options as part of the compensation. The holder gets liquidity or fees without a sale; the market maker gets inventory to quote with and upside exposure. This structure is so standard it is effectively the default relationship between a token project and its market maker, and it is dissected in [the loan-plus-options deal: how market makers get paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid).

Both of these matter enormously for interpretation. **An unlock that produces no visible on-chain selling has not necessarily been held.** It may have been sold in a single transaction to a desk that will distribute it over the next quarter. The absence of evidence in the block explorer is genuinely not evidence of absence.

## How to actually read an unlock calendar

Here is the method, assembled from everything above.

![The six-step method for reading an unlock: find the source of truth, size it in dollars, scale it to liquidity, identify the holder, check what is already priced, and decide](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-7.webp)

**Step 1: find the source of truth, and know which one you are using.**

There is a hierarchy, and most people stop at the bottom of it:

1. *The on-chain vesting contract.* Highest confidence. The tokens are provably immobile until a timestamp. You can read the balance and the release logic.
2. *The project's own tokenomics documentation.* High confidence about intent, no enforcement guarantee. This is what we used for all three case studies below, because in each case the project publishes the allocations and schedule directly.
3. *An unlock tracker.* Convenient, aggregated, and derived from the two above  -  which means it inherits their errors and adds its own. Trackers are a starting point for finding out that a date exists, not an authority on what happens on it.

In practice, getting to tier one works like this. Open the token in a block explorer and sort the holder list by balance. Exchange wallets are usually labelled, and what remains are the interesting balances. Check whether each address is a contract or an ordinary wallet  -  a *contract* holding 15% of supply, often with a name like `TokenDistributor` or `Vesting`, is what you are looking for. Read its release logic for the cliff timestamp and the vesting rate, then look at its outbound transfer history to watch the schedule actually executing, month by month. A schedule you have seen execute three times is a schedule you can forecast.

Three things break this. Allocations are frequently held in **multisig wallets** rather than vesting contracts, which enforce a signing threshold and nothing else  -  a multisig can move tokens whenever its signers agree, regardless of what any document says. Tokens held with a **qualified custodian** may not appear on-chain in any interpretable form at all. And a token deployed across **several chains** has its supply split across several sets of contracts, so a single explorer view understates the total. Whenever one of these applies, you have quietly dropped to tier two  -  the project's word  -  whether the documentation acknowledges it or not.

### The calendar is not a contract

The most common mistake made by people who have learned to check unlock schedules is to treat the schedule as fixed. It is not. It is a plan, published by a party with a strong interest in the token's price, and it gets revised.

The Arbitrum DAO forum proposal quoted earlier is one form this takes: a community member formally proposed slowing the release from 1,111,750,000 tokens at once to a maximum of 3% per year. That proposal did not change the outcome  -  the unlock happened on schedule  -  but the attempt is the point.

Sometimes the attempt succeeds. Three dated instances, each announced by the project itself:

- **Starknet revised the size, not the date.** StarkWare announced on **22 February 2024** that the first early-contributor and investor unlock  -  always scheduled for **15 April 2024**  -  would release roughly **64 million STRK** rather than the approximately **1.34 billion** the original schedule implied. Anyone who had sized that overhang from the published plan was working with a number more than twenty times too large.
- **AltLayer paused its schedule outright.** On **23 July 2024**, following its first unlock, AltLayer announced it was **pausing its entire vesting schedule for six months**.
- **Worldcoin stretched the term.** On or around **19 - 24 July 2024**, roughly **80%** of team and investor vesting was extended from a three-year schedule to a **five-year** one.

Note the pattern in the timing. All three revisions came under price pressure, and all three moved in the direction of *less* near-term supply. That asymmetry is worth internalising, because it cuts against the naive short: the project has both the motive and, frequently, the ability to defuse the event you are positioned for. A schedule is most likely to be revised precisely when it would have hurt most.

The practical rule: **a tracker showing a date twelve months out is showing you a plan, not a fact.** Re-check the schedule close to the date, and treat the project's current documentation as superseding anything  -  including this post  -  that quotes an older version of it.

### What to actually use, and what not to trust about it

A brief, deliberately non-prescriptive note, because tooling in this space churns badly. Several widely-recommended free analytics products have shut down or moved behind paywalls during 2025 and 2026, and free tiers that once showed years of unlock history have narrowed sharply. Any specific tool list in a blog post  -  including one written today  -  starts decaying immediately.

So rather than name a stack, demand three capabilities and find whatever currently provides them:

1. **A schedule you can trace to a contract or a project document**, not an unattributed chart. If a tracker cannot tell you where its schedule came from, you are looking at someone's reconstruction. Good trackers are explicit about which of their schedules are read from on-chain vesting logic and which are inferred from a published graph; that distinction is exactly the tier-one-versus-tier-two line from the list above, and it is the first thing to look for.
2. **Order-book depth, not just volume.** For the depth calculation in worked example 2 you need bid-side depth within a fixed percentage of mid. Free, no-login per-exchange pair tables that expose roughly ±2% depth per trading pair exist on the major data aggregators and are the fastest way for a non-professional to run that check.
3. **Holder-level on-chain visibility**  -  the ability to look at the largest addresses and see whether a vesting contract's balance is actually leaving it.

If a tool gives you a date and nothing else, it has told you the least useful part.

**Step 2: size it in dollars.** Tokens times price, at a dated price you can cite. (Worked example 1.)

**Step 3: scale it to liquidity.** Divide by realistic daily volume, and separately by 2% book depth. Divide by float to get dilution. (Worked examples 2 and 3.)

**Step 4: identify the holder.** Which bucket unlocks, at what cost basis, on what fund clock. This is the step that most changes the answer and the one that requires actual work.

**Step 5: check what is already priced.** Look at perpetual funding  -  persistently negative funding into an unlock means the short trade is crowded and already paying to exist. Look at the futures basis. Look at whether the token has already drifted down for six weeks. If the anticipation trade is full, the asymmetry may point the other way.

**Step 6: decide**  -  which for most people should mean deciding what to hold rather than what to trade.

### The traps that make the numbers lie

**Unlocked is not transferred is not sold.** Three distinct events. The schedule tells you only the first. Treat "tokens unlocked" as an upper bound on selling pressure, never as a forecast of it.

**Circulating supply is a contested number, not a measured one.** There is no consensus on whether tokens sitting in a foundation treasury, in an unclaimed airdrop contract, or in a market maker's loan inventory are "circulating." Different data providers make different choices, which is why the same token can show materially different market caps on different sites, and why float-dilution percentages should be treated as approximate. In the Celestia case below I derive the pre-unlock float from the project's own published genesis allocations, and I would expect a tracker to quote a different denominator  -  that is a definitional disagreement, not an error by either party.

**Round-number token counts are usually derived, not disclosed.** Projects publish percentages far more often than they publish dated token counts. Most unlock figures in circulation  -  including two of the three below  -  are arithmetic done on published percentages. That is legitimate, but the derivation should be shown rather than presented as a disclosed fact, which is why the calculations above are written out in full.

**Attribution is the hardest part.** A token that falls after an unlock may have fallen because of the unlock, or because the entire market fell that week. Separating the two requires comparing against a benchmark, and even then it is an inference. Be very careful about the leap from "the price fell after the unlock" to "the unlocked holders sold," and much more careful still about naming who. Unless a specific wallet is traceable on-chain from a known vesting contract to an exchange deposit address, the attribution is a guess. The techniques for doing this properly  -  and their limits  -  are in [whales, smart money and on-chain wallet watching](/blog/trading/crypto-players/whales-smart-money-and-on-chain-wallet-watching).

## How it shows up in price: three real cliffs

Now the evidence. Three large cliff unlocks from 2024, chosen because each project publishes its own schedule so the token counts can be derived transparently, and because all three are complete  -  we can see what happened afterwards.

All prices are from the [DefiLlama historical price API](https://coins.llama.fi/), quoted at the dates shown.

![A comparison of three real 2024 cliff unlocks  -  Arbitrum, Pyth and Celestia  -  showing tokens released, float dilution, and price 30 days before and after](/imgs/blogs/unlock-cliffs-and-the-supply-overhang-trade-8.webp)

### Arbitrum, 16 March 2024

The setup is the one we worked through: 1,111,750,000 ARB, roughly 87% float dilution, about $2.09 billion at the unlock-day price. It had been publicly known since the token launched a year earlier, and had been formally debated in the DAO nine months in advance.

The price path:

```
2024-02-15   $2.11
2024-03-16   $1.88     unlock day        -10.9% over the prior month
2024-03-31   $1.65
2024-04-11   $1.47                       -21.8% from unlock day
```

The anticipation shape is visible: down about 11% in the month before, and the larger decline coming *after* the date rather than on it.

The honest caveat, which matters more than the pattern: **April 2024 was a broad drawdown across the entire altcoin market.** ARB fell, and so did nearly everything else. Nothing in this price series establishes that the unlock caused the decline, and I am not going to claim it did. What the series does show is that the unlock did not produce a relief rally either  -  the "buy the news" half of the pattern simply did not appear.

### Pyth Network, 20 May 2024

The most extreme dilution of the three. [Pyth's documentation](https://docs.pyth.network/pyth-token/pyth-distribution) states an initial circulating supply of 1,500,000,000 PYTH out of 10,000,000,000 total, with locked tokens unlocking at 6, 18, 30 and 42 months. The first tranche  -  2,125,000,000 PYTH  -  landed roughly six months after the November 2023 launch, on 20 May 2024. At $0.4445 that is about **$945 million**, and it increased the sellable supply by about **142%**.

```
2024-05-10   $0.5040
2024-05-20   $0.4445    unlock day       -11.8% over the prior ten days
2024-05-27   $0.4422                      -0.5% in the week after
2024-06-20   $0.3253                     -26.8% from unlock day
```

Look closely at the middle of that series, because it is the most instructive data point in this post. The token fell nearly 12% into the date. Then in the week *following* the unlock  -  the week when a supply more than doubling the float became sellable  -  the price was essentially unchanged, down half a percent.

That is what "priced in" looks like. The anticipation did the work. The day itself was a non-event. And then the token declined another 27% over the following month, during a period when the broader altcoin market was also weak.

The lesson is not "unlocks do not matter." It is that **the date is the worst possible moment to react**, because it is the one moment everybody is watching.

### Celestia, 30 October 2024

[Celestia's documentation](https://docs.celestia.org/learn/TIA/staking-governance-supply/) is unusually explicit, including the detail that "yearly unlock intervals will occur on October 30th of each year." At the one-year mark, 33.33% of each of three allocations vested  -  Early Backers Seed (159,000,000), Early Backers Series A&B (196,700,000) and Initial Core Contributors (176,400,000)  -  for a derived total of about **177 million TIA**, roughly **$876 million** at the unlock-day price, against the 266,975,000 TIA unlocked at genesis.

```
2024-09-30   $5.93
2024-10-30   $4.94     unlock day        -16.7% over the prior month
2024-11-06   $4.76
2024-11-30   $8.27                       +67.4% from unlock day
```

The pre-unlock drift is the largest of the three at nearly 17%. And then the token rose 67% in the month after the largest single supply release in its history.

The explanation is not subtle and I am not going to dress it up as an unlock effect: **November 2024 was a violent market-wide crypto rally**, and TIA rallied with it. That is precisely why the case is worth including. An unlock is one input into a price, and it is regularly overwhelmed by a completely unrelated one. Anyone who shorted this unlock on the mechanics  -  correctly identifying the largest dilution event in the token's life  -  lost badly, because the mechanics were right and the market regime was against them.

### What the three cases actually support

Being careful about what three data points can and cannot demonstrate:

- **All three fell into the date.** Between 11% and 17% in the month before. This is consistent with the anticipation mechanism and with what practitioners describe. Three cases is not evidence of a general law.
- **None of the three crashed on the day.** The Pyth case is the sharpest: down 0.5% in the week following a supply increase of 142%.
- **What happened afterwards was dominated by the market, not the unlock.** Two fell in a falling market; one rose sharply in a rising one.
- **Attribution is not available.** I have not claimed that any specific holder sold into any of these unlocks, because I cannot document it. The price moved; who moved it is inference.

If there is a general claim these cases support, it is a modest one: the overhang is real, it is largely anticipated, and it is not the dominant term in the price of a token over any horizon longer than a few weeks.

## Common misconceptions

**"The unlock will crash the price on the day."** This is the most common and the most reliably wrong. Because the date is public, the repositioning happens before it. In all three cases above, the month *before* the unlock was worse than the day of, and in the Pyth case the week after a 142% supply increase was flat. If you are structuring a decision around unlock day itself, you are arriving after the market.

**"Unlocked tokens are sold tokens."** They are not. Unlocking creates the *ability* to sell. The tokens may be held, lent to a market maker, posted as collateral, sold over the counter in a single block, or simply left alone. The gap between "can be sold" and "was sold" is where every wrong forecast in this subject lives.

**"No selling appeared on-chain, so the holders diamond-handed."** A large OTC block sale looks like one transfer. A loan to a market maker looks like one transfer. Both move the economic risk without leaving anything that resembles distribution in a block explorer. Absence of visible selling is weak evidence.

**"FDV is a made-up number, so ignore it."** FDV is a bad measure of what a network is worth today and a good measure of what you are implicitly agreeing to pay. If a token has a $10 billion FDV and a $1 billion market cap, then buying at today's price means accepting that nine times the current float has to be absorbed by someone, eventually, at prices you would be happy with. Ignoring FDV is how people end up surprised by arithmetic that was published at launch.

**"A big unlock is bearish and a small unlock is fine."** Size matters far less than the ratio to liquidity and the identity of the holder. A $50 million unlock into a token with $2 million of daily real volume is a much heavier overhang than a $500 million unlock into a token with $400 million of daily volume  -  and a $500 million unlock to a foundation treasury that has never market-sold is different again.

**"The team said they are not selling, so there is no overhang."** Statements of intent are not constraints, and circumstances change. More importantly, the overhang exists in the mind of every *other* holder regardless of what the team intends. A market where everyone believes a large seller might appear behaves like a market with a large seller, because the buyers price in the possibility.

**"Vesting schedules are fixed."** They are frequently renegotiated, extended, or restructured, and the revisions cluster in periods of price weakness. Starknet cut the size of its first unlock by more than twenty-fold weeks before it happened; AltLayer paused its schedule entirely for six months; Worldcoin stretched most of its insider vesting from three years to five. All three were announced by the projects themselves in 2024. Read the current documentation, not the launch blog post  -  and be aware that the event you are positioned for may simply be cancelled.

## What this means if you are just holding a token

The genuinely useful takeaways here are defensive, and there are five of them.

**Check the schedule before you buy, not after you are down.** This is the whole post compressed into one sentence. Before buying any token launched in the last three years, spend ten minutes finding out: what percentage of supply is currently circulating, when the next significant unlock is, and who receives it. If the project does not publish this clearly, that is itself a finding.

**Treat a very low float as a risk disclosure.** A token with 10 - 15% of supply circulating is not cheap because its market cap is small. It is a token where 85 - 90% of the ownership has not yet been tested against a real bid, and where the entire future path includes absorbing that supply. The strong launch price and the difficult second year are the same design.

**Do not build a plan around unlock day.** The evidence such as it is  -  including all three cases above  -  points to the anticipation window mattering more than the event. If you have decided you do not want to hold through a large dilution, the time to act is not the morning of.

**Do not short an unlock because the mechanics are obvious.** They are obvious to everyone, which means the trade is crowded, the funding is against you, and you are exposed to a market regime you do not control. Celestia's holders were correct about the largest supply release in the token's history and would still have been carried out on a stretcher for shorting it. Being right about supply is not the same as being right about price.

**Distrust attribution, including your own.** When a token falls near an unlock, the instinct is to name a seller. Resist it. Unless you can trace a vesting contract to an exchange deposit on-chain, you are constructing a story. The mechanism is knowable; the intent behind any particular candle usually is not.

The broader point, and the reason this topic sits inside a series about who actually moves crypto prices, is that unlock schedules are the clearest available map of *structural* rather than *discretionary* selling. Most of the sellers you worry about in a market are making decisions. The seller behind an unlock cliff was determined years ago by a term sheet, a fund's life cycle, and a vesting contract  -  and then published. If you want to understand why a token behaves the way it does in its second year, the cap table and the calendar explain more than the roadmap does.

For where these schedules come from in the first place, see [how VCs move price: listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) and the series hub, [crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers). For the launch structure that makes cliffs so consequential, [the low float, high FDV game](/blog/trading/crypto-players/the-low-float-high-fdv-game). And for the practical habits that keep a retail holder out of the worst of this, [reading the tape: defending yourself as retail](/blog/trading/crypto-players/reading-the-tape-defending-yourself-as-retail).

## Sources & further reading

**Primary tokenomics documentation**

- Arbitrum Foundation, [Airdrop eligibility and distribution](https://docs.arbitrum.foundation/airdrop-eligibility-distribution)  -  ARB initial supply of 10 billion; allocations of 26.94% to team and contributors plus advisors, 17.53% to investors, 35.28% to the DAO treasury, 11.62% to the airdrop, 7.5% to the Foundation, 1.13% to DAOs; and the four-year lockup with first unlocks one year after the 16 March 2023 TGE followed by monthly unlocks for three years.
- Celestia documentation, [Staking, governance and supply](https://docs.celestia.org/learn/TIA/staking-governance-supply/)  -  genesis supply of 1,000,000,000 TIA; allocations of 20.00% public, 26.79% R&D and ecosystem, 19.67% Early Backers Series A&B, 15.90% Early Backers Seed, 17.64% Initial Core Contributors; unlock terms including 33.33% at year one for backers and core contributors, and the note that yearly unlock intervals occur on 30 October.
- Pyth Network documentation, [PYTH distribution](https://docs.pyth.network/pyth-token/pyth-distribution)  -  total supply of 10,000,000,000 PYTH; initial circulating supply of 1,500,000,000 (15%) in November 2023; category allocations and initial unlocked amounts; and the statement that locked tokens unlock 6, 18, 30 and 42 months after launch.

**Governance record**

- Arbitrum DAO forum, [Proposal for change of gradual unlocking of Arbitrum tokens to sustain ecosystem value](https://forum.arbitrum.foundation/t/proposal-for-change-of-gradual-unlocking-of-arbitrum-tokens-to-sustain-ecosystem-value/14907), posted 14 June 2023  -  states that the 16 March 2024 unlock was "estimated to nearly double the number of tokens in circulation" and gives the release size as 1,111,750,000 tokens.

**Price data**

- DefiLlama historical price API (`coins.llama.fi`)  -  all dated prices quoted in this post for ARB, PYTH and TIA.

**Dated attributions not linked above**

The following are attributed to the named publisher or to the project's own announcement, with the date shown, but  -  unlike the three tokenomics documents above  -  I have not reproduced a URL for them. Treat them as dated attributions to check rather than as documents quoted from directly, and note that each project's *current* documentation supersedes any schedule stated here.

- **StarkWare**, announcement of 22 February 2024  -  the first early-contributor and investor unlock, scheduled for 15 April 2024, revised to approximately 64 million STRK against roughly 1.34 billion under the original schedule.
- **AltLayer**, announcement of 23 July 2024  -  a six-month pause of its vesting schedule following its first unlock.
- **Worldcoin**, on or around 19 - 24 July 2024  -  extension of approximately 80% of team and investor vesting from three years to five.
- **Binance Research**, May 2024  -  an average market-capitalisation-to-FDV ratio at listing of 12.3% across 2024 token launches, and roughly $155 billion of tokens projected to unlock between 2024 and 2030.
- **Tokenomist / BlockEden** unlock-tracker data  -  more than $6 billion unlocking across 144 projects in March 2026, roughly three times the monthly average; approximately $1.99 billion scheduled between 1 July and 1 August 2026.

**Derivations declared**

- The ARB cliff size of 1,111,750,000 tokens is derived as 25% of the combined 4,447,000,000 team-plus-investor allocation, consistent with a four-year schedule whose first unlock falls at year one; the figure is corroborated by the DAO forum proposal above.
- The TIA year-one release of approximately 177 million tokens is derived as 33.33% of the combined 532,100,000 held by Early Backers Seed, Early Backers Series A&B and Initial Core Contributors.
- The PYTH tranche of 2,125,000,000 tokens is derived as one quarter of the 8,500,000,000 locked at launch, per the stated four unlock dates.
- Pre-unlock float figures are derived from each project's published genesis allocations and will not necessarily match third-party trackers, which apply different definitions of circulating supply.

All worked examples using round numbers ($2.00 tokens, 5,000,000-token positions, 90% implied volatility, 20% margin) are **illustrative arithmetic**, not observations of any specific market. Nothing here is investment advice.
