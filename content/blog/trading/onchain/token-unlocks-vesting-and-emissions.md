---
title: "Token Unlocks, Vesting, and Emissions: Reading the Sell Pressure Calendar"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How to read a token's vesting schedule, cliffs, the FDV-vs-market-cap gap, and emission rate so you can see the most predictable sell pressure in crypto coming."
tags: ["onchain", "crypto", "token-unlocks", "vesting", "emissions", "tokenomics", "fdv", "defillama", "tokenunlocks", "ethereum", "supply", "trading"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The single most predictable sell pressure in crypto is the unlock calendar: team and VC tokens that were bought cheap or free, locked at launch, and released onto the market on a published, dated schedule, plus the ongoing emissions a protocol mints forever.
>
> - **What it is:** vesting locks tokens then releases them — as a *cliff* (a big block on one dated day) or *linearly* (a daily drip). Emissions are brand-new tokens minted as staking, LP, and farming rewards. Both add supply the price has to absorb.
> - **How to read it:** pull the unlock calendar on DefiLlama or TokenUnlocks, compare market cap to fully-diluted value (FDV) to size the overhang, and read the emission/inflation rate. The vesting schedule lives in the token contract — it is public and dated.
> - **What you do with it:** size the next cliff against daily volume; fade or de-risk into a large insider unlock; never value a token at market cap when a 10× FDV overhang looms.
> - **The number to remember:** a cliff worth more than a *few days* of the token's real trading volume is a supply shock the float cannot absorb — that is when a dated unlock becomes a dated dump.

In April 2024 a wave of "low-float, high-FDV" tokens listed at eye-watering valuations — several launched with under 15% of supply circulating but fully-diluted valuations in the billions. The exchanges that listed them, and the funds that backed them, knew exactly what was coming: a vesting schedule that would, month after month, pour the other 85% of supply onto the market. By the second half of 2024 the pattern was so consistent that "the unlock trade" — fading a token into a large scheduled insider release — had become one of the most discussed setups in crypto. It worked often enough that projects started front-running it themselves, announcing buybacks and "unlock smoothing" to fight the very supply they had structured.

Here is the thing that makes unlocks special. Almost everything else you trade on-chain is a *signal you have to infer* — you watch exchange flows and *guess* whether a whale is about to sell, you read a smart-money wallet and *guess* whether it is conviction or a hedge. An unlock is different. It is **written into the token contract on day one**, published on free dashboards, and dated to the block. The supply is going to arrive. The only questions are how much, when, who holds it, and whether the market has already priced it in. That makes the unlock calendar the closest thing crypto has to a scheduled earnings date — except instead of a number that might beat or miss, it is a known quantity of new sellers walking through the door.

This post teaches you to read that calendar. We start from zero — what supply even *means* in crypto — then build up through vesting, cliffs, emissions, the FDV-vs-market-cap gap, and finally how to position around an unlock you can see coming.

![Diagram showing team VC and ecosystem allocations flowing through vesting and emissions into a dated unlock calendar that becomes sell pressure on price](/imgs/blogs/token-unlocks-vesting-and-emissions-1.png)

## Foundations: supply, vesting, cliffs, and emissions from zero

Before any of the trader stuff makes sense, you need four building blocks. None of them require a finance background — just patience to define each term exactly once.

### The three supply numbers (and why they disagree)

Every token has *three* different "how many coins are there" numbers, and confusing them is the single most common beginner mistake.

- **Circulating supply** — the tokens that exist and can be freely traded *right now*. These are the coins actually on exchanges, in private wallets, and available to be bought or sold today. When a site quotes a token's **market cap**, it multiplies price × *circulating* supply.
- **Total supply** — every token that has been minted (created) so far, *including* tokens that are locked, vesting, or held in a treasury and can't trade yet. Total ≥ circulating, always.
- **Max supply** — the absolute ceiling the protocol will ever mint. Bitcoin's max supply is 21 million. Some tokens have no hard cap (they emit forever), in which case max supply is "infinite" and you fall back to a forward estimate.

From the max (or total) supply you get the number that matters most for this whole discussion: **fully-diluted value (FDV)** = price × *maximum* supply. FDV answers the question "what would this token be worth if every coin that will ever exist were trading at today's price?" It is the market cap of the *whole pie*, not just the slice circulating today.

The gap between market cap and FDV is the **supply overhang** — all the tokens that exist or will exist but aren't trading yet. That overhang is exactly what vesting schedules and emissions are slowly converting into circulating supply.

#### Worked example: the three numbers for one token

Say a token trades at \$1.00. The contract's max supply is 1,000,000,000 (one billion) tokens, but only 100,000,000 (one hundred million) are circulating today.

- Market cap = \$1.00 × 100,000,000 = **\$100M** — that is the headline number on a price site.
- FDV = \$1.00 × 1,000,000,000 = **\$1,000M = \$1B** — the value of every coin if all were trading.
- Overhang = FDV − market cap = \$1B − \$100M = **\$900M** of supply still to arrive, at today's price.

The market is pricing \$100M of token, but \$900M more is queued up to enter on a schedule. The intuition: you are not buying a \$100M asset, you are buying a 10%-deep slice of a \$1B supply structure whose other 90% is coming for your bid.

### What vesting is

**Vesting** means tokens are locked when they are allocated, then released to their owner over time on a fixed schedule. The classic case: a project allocates 18% of supply to its team and 20% to its VC backers, but those tokens don't unlock the day the token launches — they vest over, say, three or four years. The point is alignment: a team that gets all its tokens on day one can dump and walk away; a team whose tokens drip out over four years has a reason to keep building.

The schedule itself lives in a **vesting contract** — a smart contract that holds the locked tokens and only lets a recipient withdraw the portion that has vested as of the current block. Because it is a contract on a public chain, anyone can read its release schedule. That is the source of truth behind every unlock dashboard.

### Cliffs versus linear vesting

A vesting schedule has two shapes, and the difference is everything for sell pressure.

- A **cliff** is a date before which *nothing* unlocks, and then a big block unlocks at once. A "12-month cliff, then quarterly" schedule releases zero for a year, then dumps a quarter's worth on a single day. A cliff is a *dated supply shock*.
- **Linear vesting** releases a small amount every block (effectively every day) over the vesting period. The same total tokens come out, but spread into a drip the market can absorb continuously.

Most real schedules combine the two: a cliff (often 12 months for team/VC), then linear vesting for two or three more years after the cliff. The cliffs are the dangerous dates; the linear portion is background drift.

There is a subtle third shape worth naming: **back-loaded vesting**, where the schedule releases very little early and accelerates later. A project can advertise "4-year vesting" but structure it so that 70% of the supply unlocks in years three and four. The headline duration looks responsible; the supply curve is a wall in the back half. When you read a schedule, never trust the *length* alone — read the *shape* month by month, because two "4-year vests" can have wildly different per-quarter pressure.

### Who gets the tokens: the allocation table

The other half of reading tokenomics is the **allocation** — what slice of supply each group received at genesis. A typical breakdown for a venture-backed token might look like: team and advisors 15–20%, private investors (VCs) 15–25%, ecosystem and community 20–35%, treasury/foundation 10–20%, public sale and airdrop 5–15%, and liquidity 2–10%. The exact split is in the project's tokenomics docs, but the slices you care about for sell pressure are **team + private investors** — that combined 30–45% is the cheap-cost-basis, motivated-seller supply, and it is almost always the slice with the cliffs.

The reason allocation matters so much: it tells you, before a single token unlocks, *how much of the eventual circulating supply is insider supply*. A token where 45% of max supply sits with team and VCs is structurally heavier with future sellers than one where insiders hold 20% and the rest is broad community distribution. Pair the allocation table with the vesting schedule and you can forecast the *entire* future supply curve from day one.

### What emissions are

**Emissions** are brand-new tokens the protocol *mints* and hands out as rewards — to people who stake the token, who provide liquidity (LP) to a pool, or who farm a yield program. Unlike vesting (which releases tokens that already exist in the total supply), emissions *increase* the total supply over time. A token "emitting 40% per year" is inflating its supply by 40% annually, and almost all of those new tokens are minted to people who will sell them to harvest the yield. Emissions are a constant, structural sell pressure that you net against demand.

Put the three drivers together and you get the shape of circulating supply over a token's life: a tiny float at launch, dated cliff jumps, a linear ramp, and an emissions wedge that never stops climbing.

![Layered stack showing total and max supply mapping to FDV with the locked overhang above circulating supply which maps to market cap and the thin liquid float](/imgs/blogs/token-unlocks-vesting-and-emissions-2.png)

## Why an unlock is predictable sell pressure

The reason unlocks move price so reliably comes down to *who* holds the unlocking tokens and *what their cost basis is*.

Think about the team. They created the token; their cost basis is effectively zero. When their cliff unlocks, every token they sell is pure profit — there is no price at which they are "underwater." That is a structurally motivated seller.

Now the VCs. A venture fund that backed the project in a private round typically bought at a discount of 5×, 10×, sometimes 50× below the listing price. Their job is to return capital to their limited partners. The moment their tokens unlock, they are sitting on a large multiple, and many of them have an explicit mandate to take profit on a schedule. They will not sell *everything* on day one — that would crater the price they sell into — but they are net sellers, and the market knows it.

Compare that to a regular holder who bought on the open market at \$1.00. They might sell, they might hold, they might buy more — their behavior is a coin flip. The unlocking insider is not a coin flip. They bought cheap or free, they have a mandate or a motive to realize gains, and their tokens are arriving on a known date. **That asymmetry — cheap-or-free supply, with a structural reason to sell, on a public schedule — is what makes an unlock the most predictable sell pressure in the asset class.**

#### Worked example: a 5% cliff into a thin market

Take the \$100M-market-cap / \$1B-FDV token from before. A 5%-of-max-supply cliff unlocks for the team.

- 5% of 1,000,000,000 tokens = 50,000,000 tokens unlocking on one day.
- At \$1.00, that is **\$50M of tokens** suddenly able to trade — a block worth half the *entire* circulating market cap.
- Now size it against liquidity. Say the token does \$2M of real daily trading volume. The cliff is **\$50M / \$2M = 25 days of volume** arriving in a single unlock.
- Even if the team sells only a fifth of it in the first week, that is \$10M of selling into a book that normally turns over \$2M a day — roughly **5× the daily volume** of one-directional supply.

The intuition: a cliff isn't dangerous because the *number* is big; it's dangerous because the number is big *relative to how much the market trades*. \$50M into a \$2M/day book is a flood; the same \$50M into a \$500M/day book is a ripple.

This is the core skill: **always size an unlock against daily volume and circulating float, never in isolation.** A 2% unlock can be a non-event for a deep, liquid token and a catastrophe for a thin one.

## The 2024 low-float, high-FDV era: a dated case study

The clearest real-world demonstration of everything in this post was the cluster of token launches in the first half of 2024. A pattern repeated across many of the year's most-hyped listings: a token would launch with a *very* small percentage of supply circulating — often single-digit to low-teens percent — paired with a fully-diluted valuation in the billions of dollars. The float was thin enough that a modest amount of buying pushed the price (and therefore the FDV) sky-high; the locked supply was enormous; and the vesting schedules guaranteed that the locked supply would unlock, tranche after tranche, for the next several years.

The mechanics were brutal and entirely predictable. Day-one buyers were valuing the token on its market cap and the excitement of the listing. But the *real* valuation — the one that had to hold as supply unlocked — was the FDV, and at a multi-billion-dollar FDV with single-digit float, the math was unforgiving: for the price to stay flat, demand had to absorb a multiple of the entire current float over the following year. When the first cliffs hit and demand didn't keep pace, many of these tokens bled 60–90% from their listing highs over the following months, even when the underlying products were perfectly functional. It wasn't fraud; it was supply.

What made this a genuine *signal* rather than hindsight: every one of those unlock schedules was published in advance. A trader who pulled the calendar, computed the market-cap/FDV ratio, and saw single-digit float against a multi-billion FDV had all the information needed to either avoid the token or fade the unlocks. The episode is the reason "look at the FDV, not the market cap" became a reflex for serious crypto investors, and why the low-float launch model drew so much criticism. It also triggered a counter-move: by late 2024 and into 2025, several projects deliberately launched with *higher* initial float and flatter vesting precisely to avoid the overhang stigma — a direct, observable response to the sell-pressure dynamics this post describes.

#### Worked example: a low-float launch by the numbers

A token lists. It opens at \$3.00 with 50,000,000 tokens circulating out of a 1,000,000,000 max supply.

- Market cap = \$3.00 × 50,000,000 = **\$150M** — looks like a reasonable mid-cap.
- FDV = \$3.00 × 1,000,000,000 = **\$3,000M = \$3B** — a market-cap/FDV ratio of just **0.05 (5% circulating)**.
- Over the next 12 months, suppose 25% of max supply unlocks: 0.25 × 1,000,000,000 = 250,000,000 tokens, worth **\$750M at the listing price**.
- That is **5× the entire current circulating market cap** of new supply arriving in a year. For \$3.00 to hold, demand must absorb \$750M of fresh selling on top of the existing float.

The intuition: a 5% float against a \$3B FDV isn't "cheap with room to run" — it is a token whose price is set by a thin slice while a flood five times larger than the whole float waits in the contract.

## How to read the unlock calendar: a walkthrough

Here is the hands-on part — how to actually pull a token's unlock schedule and turn it into a number you can trade. We'll use the two free tools every on-chain analyst lives in: **DefiLlama** (and its unlocks section) and **TokenUnlocks**. Both reconstruct the schedule by reading the vesting contracts on-chain, so they agree with the ledger.

**Step 1 — Pull the calendar.** On a token-unlocks dashboard you get a forward calendar: each future date, the percentage of supply unlocking, the dollar value at the current price, and — crucially — the *recipient category* (team, private investors, ecosystem, public, etc.). Sort by dollar value to find the dates that matter.

**Step 2 — Find the next big cliff.** Scan for the next single date where a large chunk unlocks. A schedule that drips 0.3% a day for years is mostly background noise; one date that releases 4% at once is the event. Note its date and dollar size.

**Step 3 — Compute the float increase.** Divide the unlock size by *current circulating supply*, not by max supply. An unlock of 50M tokens when 100M circulate is a **50% increase in float in one day** — that is the number that scares the market, far more than "5% of max supply." A small percentage of the *total* can be a huge percentage of the *circulating* amount.

**Step 4 — Size it against volume.** Divide the dollar value of the unlock by the token's average daily trading volume. Anything over a handful of days of volume is a genuine supply shock. This single ratio — unlock value ÷ daily volume — is the most useful number on the page.

**Step 5 — Read who is unlocking.** A cliff going to *team and private investors* is the high-sell-probability case. A cliff going to *ecosystem/treasury* (a DAO that releases tokens for grants and incentives) is slower and far less likely to dump immediately. Same size, very different sell pressure.

**Step 6 — Trace the recipients on-chain (the deep read).** This is where you go past the dashboard. The vesting contract has a known address; the wallets it releases to are visible. After an unlock, you can watch those recipient wallets: did the tokens move straight to an exchange deposit address (a strong sell signal), or sit still, or get staked back (a hold signal)? This is the same onward-flow tracing covered in the [funds, VCs, and market makers post](/blog/trading/onchain/tracking-funds-vcs-and-market-makers) — an unlock is just a dated trigger for the wallet-tracking you'd do anyway.

**A note on reading the schedule across chains.** The same logic applies everywhere, but the tooling differs. On Ethereum and EVM L2s the vesting contract is usually a standard, auditable contract you read on Etherscan or a chain-specific explorer; the schedule is explicit in the contract's storage. On Solana, vesting is often handled by programs like Streamflow or Bonfida's token-vesting, and you read it on Solscan. Bridged and multi-chain tokens add a wrinkle: an unlock might release on one chain and the supply then bridges to where the liquidity is, so "where does it sell" is a separate question from "where does it unlock." When the chains diverge, follow the liquidity — the supply sells wherever the deepest book is, which is not always the chain the unlock happened on.

**Watch out for the dashboard's blind spots.** Unlock calendars are reconstructions, and they can be wrong. They sometimes miss tokens released outside the standard vesting contract (a team that sends itself tokens via a separate multisig), mislabel a recipient category, or use a stale price for the dollar figure. They also can't see *intent* — a calendar shows a cliff, not whether the recipient has already pre-sold it OTC or intends to stake it. Treat the dashboard as the starting map, then verify the size against the contract and the *behavior* against the recipient wallets. The number on the dashboard is the supply that *can* sell; the chain tells you how much *did*.

#### Worked example: modeling the next cliff

You're looking at a token. The dashboard says:

- Price \$0.40, circulating supply 250,000,000, so market cap = \$0.40 × 250,000,000 = **\$100M**.
- Max supply 1,250,000,000, so FDV = \$0.40 × 1,250,000,000 = **\$500M** (a 5× overhang).
- Next cliff: in 21 days, 50,000,000 tokens to *private investors*, value = \$0.40 × 50,000,000 = **\$20M**.
- Average daily volume: \$4M.

Now the reads. Float increase = 50M / 250M = **+20% circulating supply in one day**. Unlock vs volume = \$20M / \$4M = **5 days of volume** arriving at once, going to VCs (likely sellers), into a token already carrying a 5× FDV overhang. The intuition: a +20% one-day float jump worth five days of volume, handed to motivated sellers — this is a textbook fade-into-the-unlock setup, and the date is three weeks out in plain sight.

## The FDV-vs-market-cap gap: the overhang you can't ignore

Spend any time around new tokens and you'll hear the phrase "low float, high FDV" — and it is usually a warning, not a compliment. Here's why the gap between market cap and FDV is the most important valuation number for an early token.

When a token launches with, say, 10% of supply circulating and 90% locked, the market cap looks small and "cheap." But the price is being set by supply and demand on that thin 10% float. The other 90% is not gone — it is scheduled to arrive. To *hold* the current price as that supply unlocks, demand has to keep absorbing wave after wave of new tokens. If demand merely stays flat while supply grows, price falls. That is the overhang at work.

![Before and after panels contrasting the small market cap headline view of a token with the fully diluted view showing ninety percent of supply still locked and coming](/imgs/blogs/token-unlocks-vesting-and-emissions-3.png)

The **FDV trap** is valuing a token by its market cap while ignoring the FDV. A token at \$100M market cap might *look* like it has 10× upside to reach a \$1B peer — but if that peer is at \$1B *FDV* and our token is *already* at \$1B FDV with 90% of supply still locked, there is no gap to close; they are valued the same on a like-for-like basis. The market cap "discount" is an illusion created by the locked supply.

This is why serious analysts compare tokens on **FDV, or on the market-cap-to-FDV ratio**, not on market cap alone. A market-cap/FDV ratio of 0.1 (10% circulating) tells you the supply story is almost entirely ahead of you. A ratio of 0.9 (90% circulating) tells you most of the dilution has already happened and there is little overhang left.

There is a useful contrast that sharpens the point: the **fair launch** token. Some tokens (Bitcoin is the archetype, and many memecoins copy the structure) have *no* pre-mine, no team allocation, and no VC round — supply is distributed by mining or a fully public sale. A fair-launch token has little to no insider overhang; its market-cap/FDV ratio is high early because there is no locked tranche waiting to dump. That is a genuinely different risk profile from a venture-backed token with 40% insider supply on multi-year cliffs. Neither is "better" — a fair launch can still have heavy emissions, and a backed token can have a strong product — but the supply structure is the first thing to classify, because it tells you whether your main risk is *future unlocks* or *ongoing emissions* or neither.

A word of caution on FDV itself: for tokens with no hard max supply (they emit indefinitely), FDV is technically infinite, so analysts use a *forward FDV* based on projected supply a year or two out. And FDV is a valuation *yardstick*, not a price target — a token genuinely will not trade at its FDV until all supply circulates, and by then the price may be very different. Use FDV to compare tokens on a like-for-like basis and to size the overhang, not as a literal prediction of where price goes.

#### Worked example: two tokens, same price, different supply structure

You're comparing two tokens both priced at \$2.00.

- **Token X (venture-backed):** 100,000,000 circulating, 1,000,000,000 max. Market cap = \$2.00 × 100,000,000 = **\$200M**; FDV = \$2.00 × 1,000,000,000 = **\$2B**; ratio 0.10. Insiders hold ~40% of max on cliffs, so roughly **\$800M of insider supply** sits locked.
- **Token Y (fair launch):** 800,000,000 circulating, 1,000,000,000 max. Market cap = \$2.00 × 800,000,000 = **\$1,600M = \$1.6B**; FDV = \$2B; ratio 0.80. No insider allocation; the remaining 20% is mining emissions over a decade.

The intuition: identical \$2.00 prices hide completely different futures — Token X carries an \$800M insider overhang you must clear, Token Y carries a slow emissions drip and almost no unlock risk, so the supply structure, not the price, is your real exposure.

#### Worked example: the FDV trap in dollars

Two tokens both trade with a \$200M market cap. You're choosing one to hold for a year.

- **Token A:** 80% of supply circulating, FDV = \$200M / 0.80 = **\$250M**. Over the next year roughly 20% more supply unlocks — about \$50M of new supply at today's price.
- **Token B:** 20% of supply circulating, FDV = \$200M / 0.20 = **\$1,000M = \$1B**. Over the next year a far larger share unlocks — easily \$200M+ of new supply at today's price hitting the market.

Same \$200M headline. But to hold price, Token A's market must absorb ~\$50M of new supply while Token B's must absorb \$200M+. The intuition: identical market caps can hide a 4× difference in the supply you're fighting — always check FDV before you call a token "cheap."

## Cliffs as dated dump events

Let's go deeper on the cliff, because it is the event you most often actually trade. A cliff is the moment a vesting contract flips a tranche from locked to claimable. From the chain's point of view it is a single, dated step-change in claimable supply.

The contrast with linear vesting is the whole point. The same 5% of supply, released as a 12-month cliff versus over 36 months of linear vesting, produces wildly different per-day pressure.

![Before and after panels contrasting a cliff unlock dumping fifty million dollars on one day against linear vesting releasing forty six thousand dollars per day](/imgs/blogs/token-unlocks-vesting-and-emissions-4.png)

The danger of a cliff scales with three things: its **dollar size**, the **thinness of the order book** it lands in, and the **motivation of the recipient**. A small cliff to an ecosystem treasury is a non-event. A large cliff to insiders, into a thin book, on a known date, is the most fade-able pattern in crypto.

#### Worked example: cliff versus drip, same tokens

A project must release 50,000,000 tokens (worth \$50M at \$1.00) to its team. Two structures:

- **As a 12-month cliff:** zero for a year, then 50M tokens — \$50M — claimable on one day. Against a \$2M/day book, that is **25 days of volume** in a single block. The market front-runs it; price fades for weeks before the date.
- **As 36-month linear vesting:** \$50M / 1,095 days ≈ **\$46,000 of tokens per day**. Against \$2M of daily volume, that is **about 2.3% of volume** — a drip the market never even notices on most days.

The intuition: the *same amount* of insider supply is a terrifying dated dump as a cliff and a complete non-event as a linear drip. When you read a schedule, the cliff dates are the only ones worth circling.

## Who is unlocking: not all supply sells the same

Two unlocks of identical dollar size can have opposite effects on price, because the *recipient* determines the probability and speed of selling. Sort recipients by how likely and how fast they dump.

**Team and founders (highest, fastest sell probability).** Cost basis is effectively zero — every sale is profit, with no underwater price. Teams are often subject to the most scrutiny, so the savvier ones sell quietly over time or via OTC, but the supply is going to market. A large team cliff is the single most fade-able recipient category.

**Private investors / VCs (high sell probability, paced).** Bought at a deep discount, with an explicit mandate to return capital to limited partners. They rarely dump everything on day one — that would crater their own exit — but they are net sellers on a schedule, often over the weeks and months following each unlock. Watch their wallets: VC selling tends to be a steady distribution to exchanges rather than a single block.

**Ecosystem and community (lower, slower).** Tokens earmarked for grants, incentives, and partnerships. These unlock into a treasury or DAO multisig and are *spent into the ecosystem* over time, not sold for cash — though when they're handed to grantees or used for liquidity incentives, some does become sell pressure downstream. The key tell: an ecosystem unlock that moves to a multisig and sits is benign; one that fans out to a hundred wallets that immediately bridge to an exchange is not.

**Treasury / foundation (lowest immediate, but watch).** Held to fund operations and runway. Foundations *do* sell to cover expenses, but typically in managed tranches, often pre-announced or via OTC desks to minimize market impact. A treasury unlock is rarely the dated dump a team cliff is.

**Emissions recipients (constant, near-100% sell on the yield portion).** Stakers, LPs, and farmers receiving freshly minted tokens. The portion of recipients there purely for yield will harvest and sell almost continuously. This is structural, not dated.

The practical workflow: when you see a large unlock on the calendar, the *first* question is not "how big" but "to whom." A \$50M ecosystem unlock to a quiet multisig and a \$50M team cliff to fresh wallets are the same headline number and completely different trades.

#### Worked example: same size, opposite recipients

Two tokens each have a \$40M unlock next week, into a \$5M/day book (8 days of volume each).

- **Token A — team cliff to founder wallets.** Zero cost basis, motivated sellers, fresh wallets. Expect distribution to exchanges within days. Against \$5M/day volume, even 25% sold in week one is **\$10M = 2 days of volume** of one-way supply. Fade or de-risk.
- **Token B — ecosystem unlock to a DAO multisig.** Tokens land in the treasury and sit, earmarked for grants over the next year. Immediate market supply ≈ **\$0**. The \$40M headline is real but the *float-hitting* portion this week is near zero. No trade.

The intuition: the dollar size sets the *ceiling* on sell pressure; the recipient sets how much of that ceiling actually reaches the order book this week.

## Second-order effects: buybacks, OTC, and unlock smoothing

Once a market learns to fade unlocks, projects start fighting back, and that creates a second layer you have to read.

**Buybacks and burns.** A protocol with real fee revenue can use it to buy its own token off the market, directly offsetting unlock and emission supply. If a token unlocks \$5M of new supply a month but the protocol buys back \$6M a month from fees, the *net* supply is shrinking despite the unlocks. Always net the announced buyback against the scheduled unlocks — a buyback smaller than the unlock is marketing; one larger than it is a real supply sink.

**OTC (over-the-counter) sales.** Large holders increasingly sell unlocked tokens *off-exchange* through OTC desks, directly to a buyer, so the supply never hits the public order book as a visible dump. This is why some large unlocks pass with no price impact: the supply moved, but privately. The on-chain tell is a large transfer from a recipient wallet to a known OTC-desk address rather than an exchange deposit — the supply is still changing hands, just not into the visible book.

**Unlock smoothing.** Some projects renegotiate or restructure cliffs into linear vesting, or stagger an unlock across many small dates, specifically to avoid the dated-dump dynamic. A cliff that gets "smoothed" into a drip is genuinely less dangerous — but verify it on-chain; an announced smoothing that isn't reflected in the vesting contract is just words.

**Funding rates around unlocks.** Because so many traders fade unlocks by shorting perpetual futures, the **funding rate** (the periodic payment between longs and shorts on a perp) often goes deeply negative into a large, well-known unlock — shorts are crowded. A crowded short into a fully-priced unlock is exactly the setup for a violent short squeeze on the relief rally. When everyone is positioned the same way for an event everyone can see, the surprise is usually the other direction.

#### Worked example: netting a buyback against an unlock

A protocol unlocks \$5M of supply per month and earns \$8M/month in fees, of which it commits 50% to buybacks.

- Monthly unlock supply = **\$5M**.
- Monthly buyback = 0.50 × \$8M = **\$4M** of its own token bought off the market.
- Net new supply = \$5M − \$4M = **\$1M/month** — the unlocks are 80% absorbed by the buyback.
- If fees grew to \$12M/month at the same 50% commitment, buyback = \$6M > \$5M unlock, so net supply turns **negative \$1M/month** — the token becomes deflationary despite ongoing unlocks.

The intuition: an unlock is only the supply side of the ledger; a real, fee-funded buyback is the demand side, and the net of the two — not the unlock alone — is what price actually feels.

## Emissions: the sell pressure that never stops

Vesting eventually ends — once the schedule completes, no more locked supply enters. Emissions are different: many protocols mint new tokens *forever* to pay for staking security, liquidity, or farming incentives. That makes emissions a *structural*, ongoing sell pressure rather than a dated event.

The mechanics: a protocol mints, say, X tokens per day and distributes them to people who lock up the token (stakers), provide liquidity (LPs), or farm a yield program. Most of those recipients are there for the yield — they harvest the emitted tokens and sell them to realize the return. So a large fraction of emissions becomes near-continuous market selling.

The number to compute is the **emission rate** (also called inflation rate): new tokens minted per year ÷ current supply. A token emitting 40% per year is growing its supply by 40% annually. For the price to stay flat, demand must grow ~40% per year *just to absorb the new supply* — before any price appreciation. If demand grows slower than emissions, the token bleeds down even with healthy usage. This is why high-emission "farm" tokens so often chart as a steady decline punctuated by brief pumps: emissions are the gravity.

![Illustrative stacked area chart of circulating supply rising over forty eight months through a launch float two cliffs a linear vesting ramp and a never ending emissions wedge](/imgs/blogs/token-unlocks-vesting-and-emissions-5.png)

The honest way to read emissions is to **net them against demand**. Emissions are supply; the demand side is real buyers — fee revenue that gets used to buy back the token, new users who need it, treasuries accumulating it. If a protocol emits \$80M of new tokens a year but its actual usage drives only \$20M of new demand, the net is \$60M of structural selling and the chart goes one way. If a protocol emits \$80M but real demand is \$120M, emissions are absorbed and then some. (For how to measure that real demand — fees, revenue, and value locked — the protocol-fundamentals lens is the companion read, even though emissions themselves are pure supply.)

The trap that catches the most people is the **token-count illusion of high APY**. A farm advertising "300% APY" is, in almost every case, paying that yield in its own freshly emitted tokens. Your *token count* grows 300% — but if the protocol is emitting fast enough to fund a 300% yield, the per-token price is being diluted by that same emission. Stake \$10,000, and a year later you might hold three times as many tokens each worth a third as much: your dollar value is roughly flat, minus the price decay from everyone else selling their emissions too. The nominal yield is real; the *real* (dilution-adjusted) yield is often near zero or negative. The only emissions yield that builds wealth is one funded by real demand outrunning the new supply — and that is rare among the highest advertised numbers.

This is also why **emission *schedules* matter as much as the rate.** Many protocols front-load emissions to bootstrap liquidity, then taper them on a halving-style schedule. A token emitting 60% in year one but 10% by year three has a very different forward supply curve than one emitting a flat 30% forever. Read the emission *curve*, not just today's rate — the sell pressure two years out is what determines whether holding through the bootstrap phase pays off.

A second-order point: emissions interact with **vesting**. A token can have *both* a heavy unlock schedule (locked insider supply arriving) *and* heavy emissions (new supply minted) at the same time. In that case you sum the two: total annual new supply = scheduled unlocks + emissions, and you net that combined figure against demand. The worst supply structures stack a large insider overhang on top of high emissions, so the float grows from two directions at once.

#### Worked example: emissions diluting a \$200M token

A protocol has a \$200M market cap and emits 40% of supply per year to LPs and stakers.

- New supply minted this year = 40% × \$200M = **\$80M of new tokens** at today's price, handed mostly to yield farmers.
- If those farmers sell ~75% of what they earn (typical for a pure yield play), that is **0.75 × \$80M = \$60M of structural annual selling** — about \$165,000 of sell pressure every single day, before any other news.
- For price to hold flat, the protocol needs roughly **\$60M of net new annual demand** just to soak up the sold emissions.
- If real demand grows only \$20M, the net deficit is **\$60M − \$20M = \$40M of unabsorbed annual supply**, and the token grinds lower even while "usage is up."

The intuition: a high emission rate is a hidden tax on every holder — you are diluted by the new supply unless demand outruns it, and most of the time it doesn't.

## Positioning around an unlock you can see coming

Now the trader's payoff. Because the date is public, a known unlock develops a fairly repeatable price *shape*, and you can position around each phase. None of this is a guarantee — markets price things in, and a fully-anticipated unlock can be a non-event — but the shape recurs often enough to be a framework.

![Timeline of positioning around a cliff from thirty days before through pre-unlock fade de-risk window unlock day sell through and a relief rally ten days after](/imgs/blogs/token-unlocks-vesting-and-emissions-6.png)

**The pre-unlock fade.** As a large insider unlock approaches and gets discussed, traders front-run it — they sell or short ahead of the supply they know is coming, so price often drifts *down into* the unlock date. The fade is strongest when the unlock is large relative to volume, goes to insiders, and the token has rallied into the date (more profit to protect, more reason to sell).

**The de-risk.** If you *hold* the token through a known large cliff, the simplest play is to reduce risk before the date — trim the position, or hedge it with a short perpetual future so the unlock's downside is offset. You are not predicting a crash; you are declining to stand in front of a dated flood of supply.

**The post-unlock relief rally.** Counter-intuitively, price sometimes *rises* after the unlock clears. Once the feared supply is out and absorbed, the overhang for that tranche is gone and the front-runners who shorted into it cover. "Sell the rumor, buy the news" applies: if the unlock was fully priced in, the *removal of the uncertainty* can be bullish. This is why blindly shorting every unlock is a losing game — the market often discounts it in advance, and you're left short into a relief bounce.

**The invalidation.** The setup is wrong when (a) the unlock is small relative to volume (no shock), (b) it goes to a slow recipient like an ecosystem treasury, (c) the recipients visibly *stake or hold* rather than send to exchanges (watch the on-chain flows after the date), or (d) the move is already fully priced — the token sold off weeks ago and the unlock day is a yawn.

**Timing the fade is the hard part.** Knowing *that* a token will fade into an unlock is the easy half; knowing *when* the fade begins is where most traders get hurt. If you short three weeks early, you may sit through a rally as the rest of the market hasn't focused on the date yet, and get stopped out before the move you predicted ever happens. The fade typically sharpens in the final one to two weeks, when the unlock starts appearing in "this week's unlocks" coverage and the front-runners pile in. A practical refinement: scale into the fade rather than committing all at once, and watch the **perpetual funding rate** as a crowding gauge — when funding goes sharply negative (shorts paying longs heavily), the trade is crowded and the risk has shifted toward a squeeze. The best fades are the unlocks that are *large and under-discussed*; by the time an unlock is the headline of every newsletter, the edge has largely been competed away.

**The relief rally is a real, tradeable second leg.** For a fully-anticipated unlock that has been faded hard, the *clearing* of the event is often the bottom. Once the supply is out and absorbed, the structural reason to be short for that tranche is gone, the crowded shorts cover, and the token can bounce sharply — especially if the on-chain flows show recipients staking rather than dumping. The disciplined version of this is not "buy every post-unlock dip" but "buy when the feared supply demonstrably did *not* flood exchanges." The chain tells you which it is within a day or two of the unlock.

#### Worked example: a crowded short into a priced-in unlock

A \$25M VC unlock (5 days of volume) is the most-discussed event of the week. The token has already fallen 30% over the prior three weeks as everyone faded it, and the perp funding rate is deeply negative — shorts are crowded.

- **The naive fade:** short \$5,000 on the day. But the supply is already priced in; recipients move only \$3M to exchanges and stake the rest. Price *rises* 12% on the relief, and your short loses **0.12 × \$5,000 = \$600**, plus you were paying negative funding to hold it.
- **The informed read:** the 30% pre-fade plus crowded negative funding says the event is priced. You skip the short, and instead buy \$5,000 on confirmation that recipients staked rather than dumped. A 12% relief bounce gains **0.12 × \$5,000 = \$600**.

The intuition: the unlock that *everyone* is already short is usually the one to fade *in the other direction* — the supply is priced in, and the surprise is the squeeze.

#### Worked example: de-risking a \$10,000 position before a cliff

You hold \$10,000 of a token. A \$30M VC cliff — about 8 days of the token's daily volume — unlocks in 14 days, and the token has rallied 60% into the date.

- **Do nothing:** you carry full \$10,000 exposure into a known supply flood. If the unlock prints the typical 15% fade, that is a **\$1,500 unrealized loss** you saw coming.
- **Trim half:** sell \$5,000 now, hold \$5,000 through the date. A 15% drop costs the remaining half **0.15 × \$5,000 = \$750**, and you have \$5,000 of dry powder to rebuy lower or into the relief bounce.
- **Hedge instead:** keep the \$10,000 spot, open a \$10,000 short perpetual. A 15% drop loses \$1,500 on spot and gains ~\$1,500 on the short — **net roughly flat through the event**, minus funding costs, and you stay positioned for the longer thesis.

The intuition: you don't need to predict the exact move — you just decline to hold full directional risk into the one supply event on the calendar you could see from 14 days away.

Which response fits depends on *who* is unlocking and *how big* it is relative to the market. That two-variable read — recipient × size — is the whole playbook in one grid.

![Decision matrix of unlock recipient type against size relative to volume mapping team VC ecosystem and emissions to watch fade or avoid actions](/imgs/blogs/token-unlocks-vesting-and-emissions-7.png)

## Reading it on-chain: the vesting contract and onward flows

Everything above can be done from a dashboard, but the deepest read goes to the chain itself. The vesting schedule is enforced by a contract, and that contract is public.

On a block explorer (Etherscan for Ethereum, Solscan for Solana, and the equivalents per chain) you can open the vesting contract and read its release function and schedule — the cliff dates and per-period amounts are in the contract's state. This is the ground truth that the unlock dashboards are reconstructing; checking it yourself protects you from a dashboard that is stale or mislabeled a recipient category.

The higher-value move is the **onward-flow read after an unlock**. The vesting contract releases to a set of recipient wallets. Once an unlock fires, you watch those wallets:

- Tokens move straight to a labeled **exchange deposit address** → the recipient is positioning to sell. Bearish confirmation of the unlock thesis.
- Tokens get **staked back** into the protocol or moved to a long-term cold wallet → the recipient is holding. The feared supply isn't hitting the market; the fade thesis is weakening.
- Tokens **sit idle** in the recipient wallet → undecided; keep watching.

This is exactly the wallet-tracking workflow from the [funds and VCs post](/blog/trading/onchain/tracking-funds-vcs-and-market-makers) and the [holder-concentration post](/blog/trading/onchain/supply-distribution-and-holder-concentration) — an unlock just tells you *when* to look and *which* wallets to watch. It also connects to [airdrop and sybil cohorts](/blog/trading/onchain/airdrop-farming-and-sybil-cohorts): an airdrop is essentially a one-shot unlock of farmer-held supply, and the same "dated supply flooding exchanges" logic applies.

A few practical signals make the onward-flow read more reliable. First, **timing relative to the unlock**: tokens that hit an exchange deposit address within hours of unlocking are the most aggressive sellers; supply that sits for weeks before moving is a softer, slower distribution. Second, **the destination's labels**: explorers and analytics tools tag many addresses (exchange hot wallets, known OTC desks, staking contracts), so a transfer to a labeled Binance or Coinbase deposit address is a far stronger sell signal than a transfer to an unlabeled wallet. Third, **the fraction that moves**: a recipient who unlocks \$30M and moves \$3M to an exchange is taking 10% off the table, which is very different from one who moves the whole \$30M. Quantify the share that actually heads toward liquidity, not just the binary "did anything move."

The defensive framing matters here too: this is *recipient-behavior reading*, not surveillance of a private individual. The vesting contract and its recipient categories are published by the project; the flows are public by the nature of the chain. You are reading the supply behavior of declared insider allocations to protect your own position — exactly the legitimate analyst-and-investor use this whole series is built around. You are not deanonymizing a private person; you are watching a publicly-declared team or investor allocation behave on a public ledger.

One more on-chain refinement worth the effort: cross-reference the unlock against **exchange inflow spikes**. Even without identifying the specific recipient wallets, a sudden jump in the token's net flow *to* exchanges around an unlock date is corroborating evidence that the unlocked supply is heading to market. Pair the calendar (what *should* arrive) with the exchange-flow data (what *is* arriving) and you have both the forecast and the confirmation in one read.

One honest caveat on the data: the curated dataset behind this series tracks aggregate metrics — exchange reserves, TVL, stablecoin supply, hack totals — but **no per-token vesting or emission series**. The supply curve in this post (figure 5) and the dollar figures in the worked examples are therefore *illustrative*, chosen to show the mechanics and the math, not a specific real token's schedule. For a live token, pull its actual numbers from DefiLlama, TokenUnlocks, or the vesting contract — the *method* here is exact even though the example numbers are representative.

## Common misconceptions

**"Low market cap means it's cheap."** Not when the float is tiny. A \$100M-market-cap token with 90% of supply locked is sitting on a \$1B FDV — it is not "10× cheaper" than a \$1B token, it is *the same valuation* with most of the supply still to arrive. Always check the market-cap/FDV ratio before calling anything cheap; a ratio near 0.1 means the supply story is almost entirely ahead of you.

**"An unlock always dumps the price."** Often the market has already priced it in. A widely-known unlock that everyone has faded for weeks can produce a *relief rally* the moment it clears, as the uncertainty is removed and shorts cover. Blindly shorting every unlock loses money; the edge is in unlocks that are large, insider-bound, and *under-discussed* relative to their size.

**"Vesting protects me — the team is locked up."** Vesting aligns incentives, but it doesn't remove the supply, it *schedules* it. A four-year vest just means four years of overhang. And a cliff at the end of a "lock-up" is a dated dump, not a comfort. Read the *shape* of the schedule, not just the headline "team tokens are locked."

**"Emissions are fine, the APY is high."** A 200% staking APY funded by 200% annual token emissions is mostly the protocol paying you in its own diluted supply. Your token-count goes up while the per-token value bleeds from the very emissions paying you. Net the emission rate against real demand before you trust a yield.

**"FDV doesn't matter, only circulating does."** It matters the instant any supply unlocks — which it always does, on the schedule in the contract. Pricing a token as if the locked 90% doesn't exist is exactly the trap that left "low-float, high-FDV" buyers underwater through 2024.

## The playbook: what to do with the unlock calendar

The if-then checklist for trading the most predictable sell pressure in crypto:

- **Signal — a large cliff is approaching.** Pull the calendar (DefiLlama / TokenUnlocks), find the next single date with a big release. → **Read:** compute unlock value ÷ daily volume and unlock size ÷ circulating float; check the recipient category. → **Action:** if it is large vs volume, goes to team/VC, and is under-discussed, set up the pre-unlock fade or de-risk your spot. → **Invalidation:** small vs volume, ecosystem recipient, or already fully priced (sold off weeks ago).

- **Signal — you hold a token into a known cliff.** → **Read:** how many days of volume is the unlock, and has the token rallied into it? → **Action:** trim or hedge with a short perp to neutralize the dated downside; keep dry powder for the relief bounce. → **Invalidation:** the cliff is tiny relative to the float, or recipients are visibly staking/holding on-chain.

- **Signal — a token screens "cheap" on market cap.** → **Read:** compute FDV and the market-cap/FDV ratio. → **Action:** if the ratio is below ~0.2 (under 20% circulating), treat it as expensive — value it on FDV and discount for the overhang. → **Invalidation:** ratio near 0.8–0.9; most dilution has already happened, overhang is small.

- **Signal — a high yield / APY.** → **Read:** compute the emission rate (new supply per year ÷ current supply) and net it against real demand (fee-driven buybacks, genuine new users). → **Action:** if emissions far exceed demand, treat the token as a structural short or avoid; the yield is dilution. → **Invalidation:** real demand outruns emissions — usage is absorbing the new supply.

- **Signal — an unlock just fired.** → **Read:** watch the recipient wallets on the explorer. → **Action:** tokens → exchange deposit = bearish confirmation, stay faded; tokens → staking/cold storage = hold signal, cover the fade. → **Invalidation:** the onward flow contradicts your thesis — let the chain, not the narrative, decide.

The through-line: **an unlock is the rare on-chain event you don't have to infer.** The supply is written into the contract, dated to the block, and published for free. Your edge isn't in *predicting* it — it's in sizing it honestly against volume and float, reading who holds it, and refusing to value a token as if its locked supply will never arrive. It always arrives. The calendar tells you when.

## Further reading & cross-links

- [Tracking funds, VCs, and market makers](/blog/trading/onchain/tracking-funds-vcs-and-market-makers) — how to read the onward flows of the wallets that receive an unlock, and tell a real seller from an inventory shuffle.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — the static picture of who owns the float that an unlock then changes.
- [Airdrop farming and sybil cohorts](/blog/trading/onchain/airdrop-farming-and-sybil-cohorts) — an airdrop is a one-shot unlock of farmer-held supply; same dated-sell-pressure logic.
- [Crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — who the locked VC allocations belong to and how their mandates drive unlock selling.
- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — where emissions to LPs and stakers actually come from.
- [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) — the smart-contract foundation that makes a vesting schedule public and dated.
