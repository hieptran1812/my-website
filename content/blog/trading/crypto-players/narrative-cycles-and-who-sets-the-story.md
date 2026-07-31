---
title: "Narrative cycles and who sets the story"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "Crypto narratives are not marketing  -  they are the allocation mechanism. How a thesis becomes a sector, why rotations run on a clock set by fund vintages and unlock schedules, and how to tell an organically adopted technology from a manufactured one."
tags: ["crypto", "narratives", "venture-capital", "market-structure", "sector-rotation", "tokenomics", "kols", "unlocks", "liquidity", "retail-defense", "due-diligence", "crypto-players"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR**  -  A crypto narrative is not a marketing wrapper around a technology. It is the mechanism by which capital is allocated to a sector, and  -  more importantly  -  the mechanism by which capital *leaves* one.
>
> - The binding constraint on a large crypto position is not price, it is **liquidity**. In the illustrative arithmetic below, a \$100 million position in a quiet sector takes **125 trading days** to sell at a safe participation rate. The same position at a narrative peak takes **17**. A narrative is what converts the first number into the second.
> - Narratives move through a repeatable seven-stage supply chain: thesis published → portfolio funded → research report → listing and market-maker deal → paid amplification → sector index products → rotation out. **Retail arrives at stage five.**
> - Rotations run on a roughly predictable clock because two calendars overlap: the **fund clock** (a ten-year fund that must deploy in three years and show marks by year five) and the **token clock** (a twelve-month cliff followed by a two-to-three-year linear vest).
> - The arithmetic is what makes a narrative structurally necessary. A \$10 million cheque for 5% of a token's supply implies a \$200 million entry valuation; for that stake to return \$500 million to the fund, the token must reach roughly a **\$10 billion** fully diluted valuation. Very few tokens do that without a sector to sit at the top of.
> - The tells that separate organic adoption from a manufactured sector are all observable *before* the drawdown  -  usage that survives an incentive cutoff, developer retention after grants end, fees paid by real users rather than by a treasury, widening holder distribution  -  and **none of them is the price**.
> - One number to remember: in the illustrative cycle traced below, the same token is an **8.4×** for the seed buyer and an **85% loss** for the buyer who bought on listing day. Both are looking at the same chart.

Here is a question worth sitting with for a moment. Between 2020 and 2026, crypto ran through at least eight distinct "sectors"  -  decentralized finance, non-fungible tokens, layer-one blockchains, blockchain games, modular infrastructure and restaking, memecoins, real-world assets, and AI agents. Each one arrived with a wave of research reports explaining why it was the obvious next thing. Each one attracted billions of dollars. Most of them then fell 80 - 95% from their peak while the technology itself kept working exactly as before.

If the technology did not change, what did?

The answer is that the **story** changed, and the story is not decoration. In a market where most assets have no cash flows to discount, the story *is* the valuation model. That makes storytelling a financial function rather than a communications one  -  and it means the question "who sets the story?" is a question about power, not about media.

![The narrative supply chain: seven stages from a published thesis to rotation out, with retail entering at stage five](/imgs/blogs/narrative-cycles-and-who-sets-the-story-1.webp)

The diagram above is the mental model for this entire post. A narrative is manufactured the way anything else is manufactured: through a supply chain with distinct stages, distinct participants at each stage, and a definite order. The order matters more than any individual stage, because the order determines who is early and who is late  -  and in crypto, early and late are not degrees of the same trade. They are opposite sides of it.

Before going further, one thing needs saying plainly, and it constrains everything that follows.

> [!note]
> **On intent.** This post describes an *incentive structure* and an *observable sequence*. It does not claim that any named fund, exchange, market maker or individual deliberately manufactured a narrative in order to sell into retail. Those are two different claims and only the first one is supportable from public information. When a structure rewards a behaviour, you will see the behaviour  -  but the behaviour also arises when everyone involved is sincere. A venture investor who genuinely believes in modular blockchains and a venture investor who needs an exit for a modular blockchain position do exactly the same things in public. That ambiguity is not a rhetorical convenience; it is the honest description of the situation, and it is precisely why the defensive advice at the end is built on *observables* rather than on guesses about anyone's motives.

## The foundations: everything you need before the mechanism makes sense

If you have never invested in anything, this section is the whole prerequisite list. Read it once and the rest of the post will follow. If you already know the vocabulary, skim to the next section.

### What a "narrative" means in this context

In everyday use, a narrative is a story. In markets, a **narrative** is a shared causal explanation that tells investors *why a group of assets should be worth more in the future than they are now*. "Artificial intelligence will need decentralized compute markets" is a narrative. "Banks will settle bonds on public blockchains" is a narrative. It bundles a technology claim, an adoption claim, and a value-capture claim into one sentence short enough to repeat.

Narratives matter more in crypto than almost anywhere else for a structural reason: **most crypto assets produce no cash flows**. A share of Toyota can be valued, however roughly, by forecasting the cars it will sell and the profits those cars will generate. A governance token for a protocol that earns no fees and distributes nothing has no such anchor. In the absence of cash flows, the price is whatever the marginal buyer believes about the future  -  and beliefs are transmitted as stories. This is not a criticism of crypto; it is the mechanical consequence of an asset class where the majority of tokens are, in the language of finance, *pure duration on a narrative*.

### The players, in one paragraph each

A **venture capital fund (VC)** is a pool of other people's money. The people who put money in are **limited partners (LPs)**  -  pension funds, endowments, family offices, wealthy individuals. The people who invest it are **general partners (GPs)**. The GPs typically charge a *management fee* (commonly around 2% of committed capital per year) and keep a share of the profits, called **carried interest** or *carry* (commonly around 20%). A crypto VC buys tokens or equity in early-stage projects, usually at a large discount to whatever price the public will later pay.

A **market maker (MM)** is a firm that continuously quotes both a price at which it will buy and a price at which it will sell an asset. The gap between the two is the **bid-ask spread**  -  the difference between the price you can sell at and the price you can buy at  -  and it is the market maker's basic income. In crypto, market makers are frequently paid not in cash but in a *loan of tokens plus a call option* on those tokens; the mechanics of that arrangement, and why it matters, are laid out in [the loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid). What matters here is that a market maker's inventory and incentives are frequently tied to the token whose market it makes.

A **key opinion leader (KOL)** is the crypto industry's term for an influencer with a distribution channel  -  a large following on X, a YouTube channel, a Telegram group, a newsletter. Some are paid in cash, some in tokens, some in early allocations at a discount, and some are not paid at all. The commercial structures behind this are covered in detail in [influencers, KOLs and the narrative-for-hire machine](/blog/trading/crypto-players/influencers-kols-and-the-narrative-for-hire-machine).

A **token foundation** or **treasury** is the entity that holds the project's own unissued tokens and cash reserves, and that funds grants, liquidity incentives and marketing. Functionally it behaves like a small central bank with a marketing department attached  -  see [token foundations and treasuries](/blog/trading/crypto-players/token-foundations-and-the-on-chain-central-banks) for the mechanics of how those balance sheets actually work.

### The five numbers that govern everything below

You cannot follow the argument in this post without these five. Each is defined here and used repeatedly afterwards.

**1. Total supply**  -  every token that will ever exist, including the ones nobody can sell yet. If a project's documentation says "1,000,000,000 TOKEN", that is total supply.

**2. Float (circulating supply)**  -  the tokens that are actually free to trade right now. Tokens locked in a vesting contract, sitting in a foundation treasury, or subject to a contractual lockup are *not* in the float. A token can have 1 billion total supply and a float of 120 million  -  12%. This gap is the single most consequential number in crypto market structure, and it has its own post: [the low-float, high-FDV game](/blog/trading/crypto-players/the-low-float-high-fdv-game).

**3. Market capitalization**  -  price multiplied by *float*. If 120 million tokens trade at \$1.20, the market cap is \$144 million.

**4. Fully diluted valuation (FDV)**  -  price multiplied by *total supply*. Same token: \$1.20 × 1 billion = \$1.2 billion. Note that FDV is more than eight times market cap here. That is not unusual; it is the norm for tokens in their first year.

**5. Average daily volume (ADV)**  -  the dollar value of the token that changes hands on an average day. This is the number that determines whether a large holder can leave. It is worth internalising that ADV is not a property of the technology at all. It is a property of *attention*.

### Why the fifth number is the one that matters

Here is the idea the rest of this post is built on, stated as simply as possible.

If you own a small position, price is your problem. If you own a large position, **liquidity** is your problem, and price is almost a detail.

Suppose you own \$100 million of a token. The token is up 400%. On paper you are rich. Now try to sell. If you dump \$100 million into a market that normally trades \$8 million a day, you will not receive anything close to the quoted price  -  you will walk the order book down through every resting bid until there are none left, and the last tokens will sell at a fraction of the first. Anyone watching on-chain will see it happening, front-run the rest, and make it worse. The mechanics of that walk-down are covered in [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move).

So you cannot sell it at once. You must sell it slowly, taking a small share of each day's volume so that your selling is hidden in the noise. Traders call this a **participation rate**  -  the fraction of a day's volume your order represents. Keeping to roughly 10% of ADV is a common rule of thumb for staying inconspicuous; the professional version of this problem, and the desks that solve it, are covered in [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price).

Now the arithmetic writes itself:

$$\text{days to exit} = \frac{\text{position size}}{\text{participation rate} \times \text{ADV}}$$

And this is where narrative enters, not as marketing but as a financial input.

## Worked example 1: what a narrative is actually worth

#### Worked example: the 125-day problem

*All numbers in this example are illustrative  -  chosen to be round and easy to follow, not drawn from any specific token.*

You run a fund. You hold a position worth **\$100,000,000** in a token belonging to a sector nobody is currently talking about.

- The sector's token trades an average of **\$8,000,000 per day**.
- You are willing to be **10%** of daily volume  -  any more and your selling becomes visible in the price.
- So you can sell **\$8,000,000 × 10% = \$800,000 per day**.
- Time to exit: **\$100,000,000 ÷ \$800,000 = 125 trading days**.

Crypto trades every day, so 125 days is about **four months** of selling every single day without a break, without a holiday, and without anyone noticing. Four months during which the price can fall for reasons that have nothing to do with you, during which a competitor can launch, during which your own selling gradually pushes the price down anyway.

Now suppose the sector becomes the thing everyone is talking about. Volume does not rise a little  -  attention-driven volume in crypto routinely rises by a factor of five or ten. Say ADV goes to **\$60,000,000 per day**:

- You can now sell **\$60,000,000 × 10% = \$6,000,000 per day**.
- Time to exit: **\$100,000,000 ÷ \$6,000,000 ≈ 17 trading days**.

**Under three weeks instead of four months**  -  and at a higher price, because the same attention that brought the volume also brought the bid.

![Before and after: the same $100 million position takes 125 days to exit in a quiet sector and 17 days at a narrative peak](/imgs/blogs/narrative-cycles-and-who-sets-the-story-3.webp)

The intuition this teaches: **a narrative does not create value, it creates liquidity  -  and for a large holder, liquidity is the thing that converts paper wealth into money.**

That single sentence explains more about crypto behaviour than any technical analysis. It explains why research reports appear before unlocks rather than after. It explains why sectors get names and index products. It explains why a fund with a large position has a reason to care enormously about public enthusiasm for a category, entirely independent of whether the technology works.

It also explains something that confuses newcomers: why insiders often keep promoting a sector *after* it has already run. They are not trying to make the price go higher. They are trying to keep the volume high enough that they can keep leaving.

## The arithmetic that makes narratives structurally necessary

The previous example showed why a fund *benefits* from a narrative. This section shows why a fund essentially *cannot succeed without one*. This is the part most retail investors never see, and it reframes the whole industry once you do.

### How a venture fund has to perform

A venture fund is not trying to make a modest return. Its structure requires an extreme one.

A conventional venture fund has a **term** of about ten years, sometimes with one or two one-year extensions, and an **investment period**  -  the window in which it makes new investments  -  of roughly three to five years. After that, the remaining years are for supporting existing positions and, crucially, for *exiting* them. The fund must eventually return cash (or liquid tokens) to its LPs, because a fund that cannot distribute cannot raise its next fund.

Venture returns follow a power law: most positions return nothing, a few return the fund. So a fund does not need every investment to work. It needs *one or two* to be enormous. And "enormous" has a specific numeric meaning.

#### Worked example: why the fund needs a \$10 billion token

*Illustrative arithmetic, using round numbers.*

Consider a fund that raised **\$500,000,000**. Its LPs did not accept crypto's risk for a 12% return; assume the fund is targeting roughly **3× gross**, so about **\$1,500,000,000** returned.

Suppose it makes 30 investments. Under the power law, assume two of them must produce roughly 60% of the total  -  call it **\$900,000,000** between them, so about **\$450,000,000** each.

Now look at what that requires from a single position:

- A typical early cheque: **\$10,000,000** for **5% of total token supply**.
- That implies an entry fully diluted valuation of **\$10,000,000 ÷ 0.05 = \$200,000,000**.
- For that 5% stake to be *worth* \$450,000,000, the token's FDV must reach **\$450,000,000 ÷ 0.05 = \$9,000,000,000**  -  call it **\$9 - 10 billion**.

So the fund's entire model rests on at least one of its thirty positions reaching a roughly \$10 billion fully diluted valuation. And because the fund sells gradually over a vesting schedule rather than at one instant, the token's *peak* FDV needs to be meaningfully higher than \$10 billion for the *average* realised price to clear that bar.

Now ask the operative question: **how many tokens are worth \$10 billion at any given moment?** Historically, a small number  -  a couple of dozen at the very top of the market, most of them large layer-one blockchains and stablecoins. A token does not reach that valuation by being good software. It reaches it by being the *leading asset in a category that investors believe is important*.

The intuition this teaches: **the fund does not merely want a sector to exist  -  its return model requires one, because a token cannot reach a fund-returning valuation as a standalone product. It has to be the flagship of a story.**

This is the honest, unconspiratorial core of the whole subject. Nobody has to plot anything. A structure that requires a \$10 billion outcome from a \$200 million entry will, entirely predictably, generate enormous effort directed at making categories seem important. The effort is sincere at the individual level and systematic at the aggregate level. Both things are true at once. The broader map of who benefits from what in this industry is laid out in [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto).

## The narrative supply chain, stage by stage

Now we can walk the diagram from the top of this post properly. Seven stages, in order, with the participant and the observable artifact at each one.

### Stage 1  -  The thesis is published

Someone with capital publishes a view. In practice this takes recognisable forms: a long essay on a fund's website, an annual "state of the industry" report, a list of "big ideas" for the coming year, a conference keynote, a podcast appearance.

It is important to be fair about what this is. Publishing a thesis is a completely legitimate and often valuable activity. Funds publish because they want to attract founders working on the problem, because they want to recruit, because they want LPs to understand what they are buying, and frequently because they simply believe it. Some of the best technical writing in the industry comes out of venture funds, and dismissing it as marketing would be both unfair and analytically lazy.

What is worth noticing is *timing and position*. A published thesis is a public statement of what the publisher has already positioned in, or is about to. The thesis usually **follows** the position rather than preceding it  -  funds do not typically publish a view and then go looking for companies; they invest, then explain. That is normal. But it means the reader of a thesis post is, by construction, reading it after the author has already bought.

This is not hidden. It is often disclosed. The point is simply that "a respected fund published a bullish thesis on X" is information about the fund's book at least as much as it is information about X.

### Stage 2  -  The portfolio is funded

The thesis becomes cheques. Five to fifteen companies in the category get funded, often by an overlapping set of investors who co-invest with each other routinely. The details of the deal  -  the valuation, the token allocation, the vesting schedule  -  are usually not public, or are public only as a headline number in a press release.

Two structural features of this stage matter later:

**Cap-table concentration.** A small number of funds end up owning a large share of an entire sector, across multiple competing projects. That is a rational way to bet on a category rather than a company. It also means the sector's investor base is far more correlated than it appears. Reading a token's cap table is a skill in itself; see [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).

**Vesting alignment.** Projects funded in the same year tend to launch tokens in the same year and unlock on similar schedules, because they copy each other's tokenomics. A cohort of investments creates a *cohort of unlocks* eighteen to thirty months later. Hold that thought; it is half of the rotation clock.

### Stage 3  -  The research report and the benchmark

Once several projects exist, the category needs a name and a way to be measured. This is the stage where "modular blockchains" or "restaking" or "AI agents" stops being a phrase and becomes a *bucket*: a research report defines it, a data provider adds it as a category, an index or dashboard tracks its total value.

This step is enormously consequential and almost invisible. Creating a category does three things at once:

1. It makes the sector **measurable**, so it can be compared to other sectors and shown to be growing.
2. It makes the sector **investable** as a theme, so an allocator can decide to have "some exposure to X" without picking a winner.
3. It makes each individual token **relatively valuable**  -  if the category is worth \$40 billion and this token is the leader, then a leader's multiple can be argued for.

Notice that the third one is a valuation argument built entirely on the existence of the category. Remove the category and the argument evaporates, even though nothing about the software changed.

### Stage 4  -  The listing and the market-maker deal

The token lists on exchanges. A market maker is engaged, usually under the loan-plus-option structure mentioned earlier. Order books appear, spreads tighten, and the asset becomes something a retail buyer can actually purchase with two taps.

This is the stage at which the asset becomes *available* to the public, and it is worth being precise about the sequencing: in a healthy asset, demand precedes the listing (people want it, so a venue lists it). In a manufactured one, the listing precedes demand (the venue lists it, and then demand is generated). Exchanges are not neutral infrastructure in this  -  they select what gets listed, they earn fees on the volume, and they sometimes hold the token themselves. That conflict is unpacked in [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues).

### Stage 5  -  Amplification

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Four stages left to right  -  thesis and funding, research and listing, paid amplification, retail bid  -  with dots travelling between them, one dot in the first gap, two in the second, five in the third and nine in the fourth, showing how a single private thesis is amplified into a crowd by the time it reaches retail" style="width:100%;height:auto;max-width:760px">
<style>
.nc2-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.6}
.nc2-t{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#374151);text-anchor:middle}
.nc2-s{font:400 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.nc2-note{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.nc2-hdr{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#374151)}
.nc2-rail{stroke:var(--border,#d1d5db);stroke-width:1.4;stroke-dasharray:4 5}
.nc2-d{fill:var(--accent,#4c6ef5);r:5;animation:nc2go 3.2s linear infinite backwards}
@keyframes nc2go{0%{transform:translateX(0);opacity:0}12%{opacity:1}88%{opacity:1}100%{transform:translateX(70px);opacity:0}}
@media (prefers-reduced-motion: reduce){.nc2-d{animation:none;opacity:.75}}
</style>
<text class="nc2-hdr" x="20" y="24">The same idea, amplified at every hand-off</text>
<text class="nc2-note" x="20" y="44">One private view becomes a crowd. Retail meets it at the widest, latest, most expensive point.</text>
<rect class="nc2-box" x="15" y="74" width="120" height="76" rx="8"/>
<text class="nc2-t" x="75" y="104">Thesis + funding</text>
<text class="nc2-s" x="75" y="124">private, month 0</text>
<rect class="nc2-box" x="205" y="74" width="120" height="76" rx="8"/>
<text class="nc2-t" x="265" y="104">Research + listing</text>
<text class="nc2-s" x="265" y="124">semi-public</text>
<rect class="nc2-box" x="395" y="74" width="120" height="76" rx="8"/>
<text class="nc2-t" x="455" y="104">Paid amplification</text>
<text class="nc2-s" x="455" y="124">public, everywhere</text>
<rect class="nc2-box" x="585" y="74" width="120" height="76" rx="8"/>
<text class="nc2-t" x="645" y="104">Retail bid</text>
<text class="nc2-s" x="645" y="124">the exit liquidity</text>
<line class="nc2-rail" x1="135" y1="112" x2="205" y2="112"/>
<line class="nc2-rail" x1="325" y1="112" x2="395" y2="112"/>
<line class="nc2-rail" x1="515" y1="112" x2="585" y2="112"/>
<circle class="nc2-d" cx="137" cy="112"/>
<circle class="nc2-d" cx="327" cy="102" style="animation-delay:-.2s"/>
<circle class="nc2-d" cx="327" cy="122" style="animation-delay:-1.7s"/>
<circle class="nc2-d" cx="517" cy="92" style="animation-delay:-.1s"/>
<circle class="nc2-d" cx="517" cy="105" style="animation-delay:-.8s"/>
<circle class="nc2-d" cx="517" cy="118" style="animation-delay:-1.5s"/>
<circle class="nc2-d" cx="517" cy="131" style="animation-delay:-2.2s"/>
<circle class="nc2-d" cx="517" cy="144" style="animation-delay:-2.9s"/>
<line class="nc2-rail" x1="40" y1="196" x2="700" y2="196"/>
<text class="nc2-s" x="75" y="216">month 0</text>
<text class="nc2-s" x="265" y="216">months 6&#8211;18</text>
<text class="nc2-s" x="455" y="216">weeks around listing</text>
<text class="nc2-s" x="645" y="216">listing day onward</text>
<text class="nc2-note" x="20" y="252">Dots stand for the number of people who can act on the idea at that stage &#8212; one desk, then a few dozen funds,</text>
<text class="nc2-note" x="20" y="272">then every timeline at once. The months shown are a typical shape, not a measured average.</text>
</svg>
<figcaption>Why being early is structural rather than clever. The same view is held by one desk at month zero and by a million timelines around listing day; each hand-off widens the audience and shortens the remaining upside. Stage timing is illustrative.</figcaption>
</figure>

This is the stage where the general public encounters the narrative  -  through influencer posts, podcast appearances, sponsored newsletters, conference panels, and the ambient sense that "everyone is talking about X". Some of that coverage is paid. Some is organic. Most audiences cannot reliably tell which, and that is the entire problem.

The commercial and legal structure of this stage is important enough to get its own section below.

### Stage 6  -  Sector products

Once a category has a name, a benchmark and liquid tokens, it becomes packageable: sector indices, "baskets", themed portfolios, structured products, and in the most developed cases exchange-traded products. This stage matters because it converts an active decision ("should I buy this token?") into a passive one ("should I have some exposure to this theme?"), and passive flows are far stickier and far less price-sensitive than active ones.

It also marks the point at which the narrative has been fully institutionalised  -  which, from the perspective of whoever entered at stage one, is the point of maximum exit capacity.

### Stage 7  -  Rotation out

The final stage is the least discussed and the most important. Capital does not usually leave crypto entirely; it *moves*. Positions in the maturing sector are reduced, and the proceeds go into the next thesis, which is already at stage one somewhere else.

From the outside this looks like a sector "dying". From the inside it is simply an allocation decision. And crucially, the next narrative is often already published  -  because a fund raising its next vehicle needs a *forward*-looking story, not a description of what worked last cycle.

## The two clocks: why rotations are roughly predictable

If narratives were purely driven by technological progress, they would arrive whenever the technology matured  -  irregularly, unpredictably. They do not. They arrive with a rhythm, and the rhythm comes from two calendars that most retail investors never think about.

![Two clocks: the fund clock of a ten-year vehicle and the token clock of a cliff plus vest, and the window where they overlap](/imgs/blogs/narrative-cycles-and-who-sets-the-story-4.webp)

### Clock one: the fund vintage

A fund's life is a calendar, and the calendar creates pressure at known points.

- **Year 0**: the fund closes. Capital is committed but not yet deployed.
- **Years 1 - 3**: the investment period. The fund must put the money to work. There is real professional pressure here  -  an undeployed fund is a fund that charged fees for nothing.
- **Years 3 - 5**: marks start to matter. The GPs want to raise the *next* fund, and to do that they must show LPs that this one is working. Since most positions are illiquid, the marks that matter most are the ones with a public price  -  which means tokens.
- **Years 5 - 10**: distributions. The fund must actually return capital. Paper marks stop being enough.

The key insight is that the pressure to have a liquid, visibly-appreciating, sellable position peaks somewhere in years three to six of a fund's life. And since funds in a given vintage year all close at roughly the same time, that pressure is *synchronised across the industry*.

### Clock two: the token vesting schedule

Almost every token sold to early investors follows a variant of the same schedule, copied project to project:

- **Token generation event (TGE)**  -  the token launches and lists. Early investors' tokens are locked.
- **Cliff**: typically **12 months** during which the early investor can sell nothing at all.
- **Linear vest**: typically a further **24 to 36 months** during which tokens unlock in equal monthly (sometimes daily) tranches.

So an investor who buys in a seed round roughly 18 months before TGE cannot sell a single token until 30 months after their investment, and does not hold their full position freely until roughly 54 months  -  four and a half years  -  after the cheque. The mechanics and the price consequences are covered in [the lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) and in the companion post on [unlock cliffs and the supply overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade).

### Where the clocks overlap

Now put the two calendars on top of each other. A fund deploys into a category in years one and two. Those projects launch tokens twelve to twenty-four months later. The cliffs expire twelve months after that  -  which lands in years three to five of the fund's life, exactly when the fund most needs liquid, appreciating, sellable positions.

The result is a structural convergence: **for a whole cohort of investments, the moment when supply becomes sellable coincides with the moment when the seller most needs a bid.**

This is why narratives rotate on something resembling a schedule. Not because anyone coordinates it, but because a cohort of funds raised in the same year invests in the same year, launches tokens in the same year, and reaches its unlock cliffs in the same year. The clock is set by capital formation, not by technology.

There is one more accelerant worth noting: **narratives that fail to attract retail get abandoned faster than narratives that fail technically.** A sector whose technology is disappointing but whose tokens trade actively will keep receiving support. A sector whose technology is excellent but whose tokens are illiquid will quietly stop being mentioned. The selection pressure is on liquidity, not on merit  -  which is a genuinely uncomfortable observation about the information environment retail investors are reading.

## The rotation itself: where the money actually comes from

<figure class="blog-anim">
<svg viewBox="0 0 720 400" role="img" aria-label="Six sector bars rising and falling one after another from left to right  -  DeFi, NFTs, GameFi, modular and restaking, memecoins, AI agent coins  -  while a dashed line marks the single pool of attention and capital they all draw from, showing that the sectors take turns rather than growing together" style="width:100%;height:auto;max-width:760px">
<style>
.nc1-bar{fill:var(--accent,#4c6ef5);transform-box:fill-box;transform-origin:50% 100%;animation:nc1rot 18s linear infinite backwards}
.nc1-b1{animation-delay:0s}
.nc1-b2{animation-delay:3s}
.nc1-b3{animation-delay:6s}
.nc1-b4{animation-delay:9s}
.nc1-b5{animation-delay:12s}
.nc1-b6{animation-delay:15s}
.nc1-axis{stroke:var(--border,#d1d5db);stroke-width:2}
.nc1-pool{stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:7 6;fill:none}
.nc1-lab{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#374151);text-anchor:middle}
.nc1-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.nc1-note{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.nc1-hdr{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#374151)}
@keyframes nc1rot{
0%{transform:scaleY(.10);opacity:.30}
6%{transform:scaleY(1);opacity:1}
13%{transform:scaleY(.82);opacity:.92}
26%{transform:scaleY(.24);opacity:.45}
100%{transform:scaleY(.10);opacity:.30}
}
@media (prefers-reduced-motion: reduce){.nc1-bar{animation:none;transform:scaleY(.45);opacity:.7}}
</style>
<text class="nc1-hdr" x="20" y="26">Capital does not arrive. It rotates.</text>
<text class="nc1-note" x="20" y="46">Each sector takes its turn at the front of the queue &#8212; the pool it draws from is the same pool.</text>
<line class="nc1-pool" x1="40" y1="96" x2="700" y2="96"/>
<text class="nc1-note" x="40" y="88">One pool of attention and capital</text>
<line class="nc1-axis" x1="40" y1="320" x2="700" y2="320"/>
<rect class="nc1-bar nc1-b1" x="52" y="106" width="76" height="214" rx="4"/>
<rect class="nc1-bar nc1-b2" x="160" y="106" width="76" height="214" rx="4"/>
<rect class="nc1-bar nc1-b3" x="268" y="106" width="76" height="214" rx="4"/>
<rect class="nc1-bar nc1-b4" x="376" y="106" width="76" height="214" rx="4"/>
<rect class="nc1-bar nc1-b5" x="484" y="106" width="76" height="214" rx="4"/>
<rect class="nc1-bar nc1-b6" x="592" y="106" width="76" height="214" rx="4"/>
<text class="nc1-lab" x="90" y="342">DeFi</text>
<text class="nc1-sub" x="90" y="360">2020</text>
<text class="nc1-lab" x="198" y="342">NFTs</text>
<text class="nc1-sub" x="198" y="360">2021</text>
<text class="nc1-lab" x="306" y="342">GameFi</text>
<text class="nc1-sub" x="306" y="360">2021&#8211;22</text>
<text class="nc1-lab" x="414" y="342">Modular</text>
<text class="nc1-sub" x="414" y="360">2023&#8211;24</text>
<text class="nc1-lab" x="522" y="342">Memecoins</text>
<text class="nc1-sub" x="522" y="360">2024&#8211;25</text>
<text class="nc1-lab" x="630" y="342">AI agents</text>
<text class="nc1-sub" x="630" y="360">2024&#8211;25</text>
<text class="nc1-note" x="20" y="388">Schematic of sequencing, not a chart of market caps. Bar height stands for share of attention and flow, not dollars.</text>
</svg>
<figcaption>Narrative rotation, schematically. One sector at a time swells to the top of the pool while the others sag, then the peak moves right. Nothing here is a market-cap series; the point is only the <em>sequencing</em>  -  sectors take turns, and the turn-taking is what a rotation is.</figcaption>
</figure>

The animation above is the shape of the thing. What it deliberately does not show is *magnitude*, because the magnitude is where the most common misunderstanding lives.

When a sector's market capitalization rises by \$30 billion, most people assume \$30 billion arrived. It did not. Two mechanisms produce most of that number, and neither is new money.

**Mechanism one: reallocation.** The dollars are not new to crypto; they moved from one sector to another. A trader selling their layer-one position to buy an AI-agent token has added nothing to the asset class. This is why sectors so often peak in sequence rather than together  -  they are drinking from the same glass.

**Mechanism two: the float multiplier.** This one is arithmetic and it surprises people every single cycle.

Market cap is price × float. Fully diluted valuation is price × total supply. But price is set at the margin  -  by the last trade, in the *float*. So a purchase that only touches a small slice of the float can revalue the entire supply.

![Where the money goes in a rotation: reallocation from one sector plus a small net inflow, multiplied by the fact that only the float trades](/imgs/blogs/narrative-cycles-and-who-sets-the-story-8.webp)

#### Worked example: how \$3 billion "becomes" \$30 billion

*Illustrative throughout.*

Take a sector with a combined market capitalization of **\$15,000,000,000**, which is **3%** of a \$500 billion universe of non-Bitcoin, non-Ethereum tokens. A rotation lifts it to **9%**, or **\$45,000,000,000**  -  an increase of **\$30,000,000,000**.

Now ask how much actual buying that required. Suppose the sector's tokens have, on average, a float of **10%** of total supply. The tradable value is therefore roughly **\$1,500,000,000**  -  one tenth of the market cap.

If a wave of buying pushes prices up 200%, and that buying only ever interacts with the tradable float, then the dollars required are a fraction of the headline market-cap change. In the simplest version: **roughly \$3,000,000,000 of net buying** revalues **\$30,000,000,000** of market capitalization, because every dollar spent in the float re-prices about ten dollars of locked supply.

Two honest caveats, because this is exactly the kind of arithmetic that gets abused:

1. **The 10× multiplier is not a law of nature.** It is what you get if price impact is proportional to the fraction of *float* traded and if the locked supply is marked at the same price. Real price impact is non-linear, varies by venue, and depends heavily on how much of the float is genuinely willing to sell. Published estimates of crypto "money multipliers" vary enormously and are not stable across time. Treat the number as a demonstration of *direction and rough scale*, not a coefficient you can trade on.
2. **The multiplier works in both directions, and it is worse on the way down.** The same locked supply that got revalued upward on thin buying gets revalued downward on thin selling  -  and unlike the upward move, the downward move often coincides with *more* float arriving as vesting continues. That asymmetry is the mechanical reason narrative drawdowns are so much sharper than narrative rallies were.

The intuition this teaches: **a headline "the sector added \$30 billion" is not a measurement of inflows. It is a measurement of price times supply, most of which never traded.**

## The KOL leg: the part with a legal boundary

![The KOL leg: a sponsor pays a promoter, and the path forks on whether the payment is disclosed](/imgs/blogs/narrative-cycles-and-who-sets-the-story-6.webp)

Stage five is where a narrative meets the public, and it is the only stage of the supply chain with a bright legal line running through it.

That line is worth stating precisely, because it is very widely misunderstood in both directions. **Being paid to promote an asset is not illegal.** Advertising is legal. Sponsored content is legal. What United States securities law prohibits  -  under **Section 17(b) of the Securities Act of 1933**, the provision usually called the anti-touting rule  -  is describing a security for compensation *without disclosing that compensation and its amount*. The offence is the concealment, not the payment.

For the reader, the practical consequence is not legal but epistemic. When you cannot tell whether a post was paid for, you cannot weight it. A recommendation from someone who bought the token at the same price you can is a different piece of evidence from a recommendation by someone who received tokens at a ninetieth of that price with a shorter lockup than yours. Both may be sincere. They are not equally informative.

#### Worked example: what an amplification budget buys

*Illustrative campaign figures  -  see the sourced enforcement record in the case-study section for the amounts that are actually documented.*

Suppose a sponsor allocates a promotion budget:

- **12 mid-tier promoters × \$15,000** = **\$180,000**
- **3 large promoters × \$60,000** = **\$180,000**
- **Total: \$360,000**

Now compare that to what it can plausibly be worth to a holder trying to leave. Assume the sponsor needs to sell **\$4,000,000** of the token.

*Without the campaign*: the token trades \$800,000 a day, so a \$4,000,000 sale at a 10% participation rate takes 50 days, and the persistent selling drags the price down  -  call it 15% of average realised value lost to impact and drift.

*With the campaign*: volume rises, the sale clears in days rather than months, and it clears at an average price meaningfully above the pre-campaign level  -  call the combined benefit **35%** of the notional.

- Benefit: **\$4,000,000 × 35% = \$1,400,000**
- Cost: **\$360,000**
- Ratio: roughly **3.9× on the promotion spend**

The intuition this teaches: **promotion spending in crypto is not an expense line, it is an execution cost  -  the price of manufacturing the liquidity needed to exit.** Once you see it that way, the size of the industry's marketing budgets stops being puzzling.

Two things must be said immediately after that example. First, every number in it is invented for illustration. Second  -  and this is the important one  -  **the same arithmetic holds for a completely sincere project that genuinely needs distribution.** A team with a real product also benefits from awareness, also needs liquidity, and also pays promoters. The arithmetic does not distinguish motives. Only disclosure does, and only partially.

## One cycle, two ledgers

![One cycle, two ledgers: the fund's 8.4x and the listing-day buyer's 85% loss on the same token](/imgs/blogs/narrative-cycles-and-who-sets-the-story-7.webp)

Everything above comes together in the single most clarifying exercise in this post: trace one token through one full narrative cycle, and keep two separate ledgers  -  the fund's and the retail buyer's.

#### Worked example: the same chart, two outcomes

*Entirely illustrative. Round numbers chosen for arithmetic clarity.*

**The fund's ledger.**

1. Seed round, 18 months before launch. The fund buys **50,000,000 tokens at \$0.05** = **\$2,500,000**. Total supply is 1,000,000,000, so the fund owns **5%**, implying an entry FDV of **\$50,000,000**.
2. **Token generation event.** The token lists at **\$1.20**. Float is **12%**  -  120,000,000 tokens  -  so the market cap is **\$144,000,000** while FDV is **\$1,200,000,000**.
3. On paper the fund's stake is worth **50,000,000 × \$1.20 = \$60,000,000**. That is **24×** the cost. It is also completely unsellable: the 12-month cliff has not expired.
4. **Months 12 - 36 after TGE.** The cliff expires and tokens vest linearly over 24 months  -  about **2,083,333 tokens per month**. Over that window the price declines as supply arrives, and the fund's *realised* average price comes in at **\$0.42**.
5. Total realised: **50,000,000 × \$0.42 = \$21,000,000**.
6. Multiple on cost: **\$21,000,000 ÷ \$2,500,000 = 8.4×**.
7. Elapsed time from cheque to fully vested: 18 months to TGE + 12-month cliff + 24-month vest = **54 months, or 4.5 years**. Annualised, 8.4× over 4.5 years is an internal rate of return of about **60% per year**.

**The retail ledger.**

1. A buyer reads about the sector, sees the listing, and buys at **\$1.20** on the first day.
2. Three years later the token trades at **\$0.18**.
3. Return: **(\$0.18 − \$1.20) ÷ \$1.20 = −85%**.

Both ledgers describe the same asset, the same chart, and the same three years. The fund earned an excellent venture return. The retail buyer lost 85%. Nobody needs to have committed fraud for both statements to be true.

And here is the part worth dwelling on: **the fund's return did not require the token to succeed.** It required the token to be *liquid at a high price during the vesting window*. Those are different things, and only one of them has anything to do with the technology.

The intuition this teaches: **entry price and lockup length, not conviction, determine who wins a narrative cycle  -  and both are set before the public can participate.**

For completeness, note that this example is deliberately unkind to the fund in one respect and kind in another. Unkind: many seed positions go to zero, and the fund needs winners like this one to pay for them. Kind: the realised average of \$0.42 assumes disciplined selling into strength that not every fund achieves. The example is a shape, not a claim about anyone's actual returns.

## How it shows up in price

If narratives were only a media phenomenon they would leave no fingerprints in market data. They leave several. None of them is conclusive on its own; together they are a recognisable signature.

**1. Volume leads price, and both lead fundamentals.** In a manufactured rotation, the order is: coverage rises, volume rises, price rises, and usage metrics either follow much later or never. In organic adoption the order is closer to: usage rises, then price, then coverage. Checking which came first is the single most useful thing an outsider can do, and it requires nothing but a chart of on-chain activity next to a chart of price.

**2. Correlation inside the sector goes to nearly one.** During a narrative run, tokens in the same category start moving together almost perfectly, regardless of how different their actual products are. This is the tell that the market is trading the *category* rather than the assets. When correlation within a sector approaches one, you are no longer being paid for picking the better project.

**3. The rally is concentrated in low-float names.** Because the float multiplier is stronger when float is smaller, the biggest percentage moves in a narrative cluster in the tokens with the least tradable supply  -  which are also, by construction, the tokens with the most insider supply still to come.

**4. Volume decays faster than price at the top.** The end of a narrative usually shows up in volume before it shows up in price. Price can be held near its highs by a thin bid long after the flow that justified it has gone; ADV cannot be faked as easily (though it can be faked  -  see [wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume)). A sector where price is flat but volume has halved is a sector where the exit is getting narrower.

**5. Unlock dates become the calendar.** Once a narrative matures, price action starts organising itself around vesting events rather than product news. The trade around those dates is its own discipline; see [unlock cliffs and the supply overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade).

**6. The next narrative is already being published.** The most reliable signal that a sector is late is that the people who made money in it have started writing about something else. This is not cynicism  -  it is exactly what you would expect from allocators doing their job. But it is observable, and it is free to observe.

## Eight narratives, 2017 to 2026

![Eight crypto narratives from the 2017 ICO boom through DeFi, NFTs, L1 wars, GameFi, modular and restaking, memecoins and AI agent coins](/imgs/blogs/narrative-cycles-and-who-sets-the-story-2.webp)

<!-- SOURCED-HISTORY -->

## Organic or manufactured: the tell sheet

![Tell sheet comparing organic adoption and manufactured sectors across usage, developers, fees, holders, listing sequence, float and promotion density](/imgs/blogs/narrative-cycles-and-who-sets-the-story-5.webp)

The whole point of understanding the machinery is to be able to answer one question about a sector you are looking at *now*: is this thing being adopted, or is it being distributed?

No single test settles it. But there are seven observable signals, and they are all available to anyone with a browser and an hour. Critically, **none of them is the price**.

### 1. Does usage survive an incentive cutoff?

The strongest test in crypto is also the simplest. Most new protocols pay users to use them  -  in token emissions, points, airdrop eligibility, or gas rebates. That is not inherently bad; it is a customer-acquisition subsidy, and every consumer business runs them.

The question is what happens when the subsidy stops or steps down. Organic usage survives it; rented usage does not. Because emission schedules are usually public, you can often see the step-down coming and watch what happens to transaction counts and fee revenue in the weeks afterward. A protocol whose activity falls 90% within a month of an emissions cut was renting its users.

### 2. Do developers stay after the grants end?

Sectors are built by people, and people leave when the money does. Public repository activity  -  commits, contributors, whether the main repositories are still maintained six months after a grant programme concluded  -  is a slow but honest indicator. It is slow enough that it is useless for trading and valuable for allocation.

### 3. Who is actually paying the fees?

Distinguish **fees paid by users** from **fees paid by the treasury**. A protocol that generates \$1 million of "revenue" while distributing \$4 million of its own token to the users who generated it has negative unit economics dressed as traction. This is not a subtle accounting trick  -  it is usually visible in the emissions schedule next to the revenue chart, and there are public dashboards for both.

### 4. Is holder distribution widening or concentrating?

Genuine adoption spreads a token out: more holders, smaller average balances, a declining share held by the top wallets. Manufactured distribution tends to look different  -  a stable or rising concentration among a small number of addresses, often with recognisable clustering. On-chain tooling makes this observable; the methods are covered in [whales, smart money and on-chain wallet watching](/blog/trading/crypto-players/whales-smart-money-and-on-chain-wallet-watching).

### 5. Did demand precede the listing, or follow it?

This one is almost a syllogism. If a token was hard to obtain and people wanted it before an exchange listed it, demand preceded supply. If a token was listed on several venues simultaneously with a market-maker agreement in place and *then* people wanted it, supply preceded demand. Only one of those orders is consistent with organic adoption.

### 6. What is the float, and what is the FDV?

A token with 8% float and a \$3 billion FDV is telling you something specific: 92% of the supply is owned by people who cannot sell yet but will be able to. That is not an accusation, it is an arithmetic fact about future supply. The full treatment is in [the low-float, high-FDV game](/blog/trading/crypto-players/the-low-float-high-fdv-game).

### 7. Is the promotion synchronised?

Organic enthusiasm is messy: it arrives at different times, in different formats, with different levels of sophistication, and includes criticism. Coordinated promotion tends to cluster  -  many accounts posting similar framings within a short window, often with similar visual assets and a shared vocabulary that appeared from nowhere. Clustering is not proof of payment. It is a reason to look for disclosure before acting.

## Common misconceptions

**"A narrative is just marketing  -  the fundamentals will win eventually."**

This assumes the narrative and the fundamentals are separable, and for a large class of crypto assets they are not. If a token has no cash flows, there is no fundamental value to converge to; the narrative is not a distortion of the valuation, it *is* the valuation. That does not mean nothing has value  -  protocols with real fee revenue can be analysed like businesses  -  but it does mean "the fundamentals will assert themselves" is a claim you have to earn by identifying which fundamentals, measured how.

**"If a fund I respect is publishing about a sector, that is a bullish signal."**

It is a signal about the fund's *position*, which is a different thing. A published thesis usually follows the investment rather than preceding it. Respecting the fund's judgement is entirely reasonable; the error is treating the publication date as the moment the opportunity became available. By the time you read the thesis, the entry price you can access is not the entry price the author accessed.

**"The sector's market cap went up \$30 billion, so \$30 billion flowed in."**

As the worked example showed, no. Market cap is price times supply, and most of that supply never traded. Headline market-cap moves systematically overstate the capital actually committed  -  in both directions. If you want to know about flows, look at flows: stablecoin supply changes, exchange net inflows, and fund-flow data, all of which are measured directly.

**"Insiders promote a token to make the price go up."**

Usually the goal is subtler and more important: to make the *volume* go up. Price without volume is a paper mark that cannot be converted into money at size. Once you understand that liquidity rather than price is the binding constraint, a great deal of otherwise inexplicable behaviour  -  promotion continuing well after a token has already tripled, for instance  -  becomes straightforward.

**"Every narrative is manufactured, so none of it is real."**

This is the mirror-image error and it is just as costly. Several crypto narratives described genuine technical developments that persisted long after the trade stopped working: automated market making, stablecoins as a payment rail, and proof systems all outlived their hype cycles and are in production use. The existence of a promotion machine does not mean nothing is being built. It means the promotion machine is not evidence about what is being built, in either direction. You still have to look at the thing itself.

**"I can just avoid the narrative and buy quality."**

You can try, and it is a defensible strategy, but be aware of what you are giving up and what you are still exposed to. Quality assets in an out-of-favour sector are subject to exactly the liquidity problem described at the top of this post  -  they can be excellent and untradeable at the same time. And when the sector rotates back, correlation within the sector rises toward one, which means your careful selection stops mattering for a while in both directions.

## What this means if you are retail

The defensive posture here is not "avoid narratives". Narratives are how crypto allocates capital; refusing to engage with them means refusing to engage with the asset class. The defensive posture is to know **which stage you are entering at**, and to size accordingly.

**Locate yourself on the supply chain before you buy.** If you learned about a sector from a mainstream podcast, a sponsored newsletter, or a wave of similar posts appearing at once, you are at stage five or later. That is not a reason not to buy; it is a reason to be honest that the people who bought at stage one are, from this point onward, potential sellers into your bid.

**Check the float and the vesting schedule first, before anything else.** These are usually published, and they take ten minutes to read. If more than 80% of the supply is locked and the cliff expires within the next year, you now know that the largest single scheduled event in this asset's future is a supply increase. You may still want to own it. You should not be surprised by what happens next.

**Separate "the technology is good" from "this token will go up".** These are close to independent questions. Excellent technology with a punishing unlock schedule and a 6% float is a bad trade and a good product. Mediocre technology with a wide float and real fee revenue can be the opposite. Conflating them is the single most expensive mistake in the sector.

**Watch volume, not price, for the turn.** Price can be supported by a thin bid long after the flow has gone. Sustained volume decay while price holds is the pattern that most often precedes an air pocket, because it means the exit is narrowing while the marks stay flattering.

**Ask what a promoter's entry price and lockup are, not whether they are honest.** Honesty is unobservable; entry price and lockup are frequently disclosed and always relevant. Someone can be entirely sincere and still be your counterparty. If the disclosure is missing, that absence is itself the most useful thing you learned.

**Treat correlation as a risk measure.** When everything in a sector moves together, you own the sector, not the asset  -  so size it as one position, not as five. Diversifying across six tokens in the same narrative is not diversification; it is leverage on the narrative with extra research work.

**Assume the next thesis is already published.** If you want to know where the rotation goes next, the best available public signal is what the people who benefited from the last one are writing about now. This is free, legal, and hiding in plain sight on fund websites. It tells you where stage one is happening  -  which is the only stage where the arithmetic favours the person arriving.

None of this is investment advice, and none of it predicts which sector is next. It is a description of a machine and a list of the parts of that machine you are allowed to inspect. The machine is not secret. It is simply not usually described from the outside.

## Sources & further reading

<!-- SOURCES-LIST -->

### Related reading on this site

- [The hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto)  -  the map this series is built on
- [Crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers)  -  the series hub
- [How VCs move price: listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative)
- [The crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model)
- [The anatomy of a token pump](/blog/trading/crypto-players/the-anatomy-of-a-token-pump)
- [Reading the tape: defending yourself as retail](/blog/trading/crypto-players/reading-the-tape-defending-yourself-as-retail)
