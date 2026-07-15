---
title: "Follow the Money: Reading a Token's Cap Table and On-Chain Trail"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A from-scratch guide to reading a token's allocation table, vesting schedule, and on-chain wallet flows so you can see who owns the supply and when they are likely to sell it."
tags: ["crypto", "tokenomics", "cap-table", "on-chain-analysis", "vesting", "token-unlocks", "wallet-tracking", "smart-money", "crypto-players", "defi"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A token's price is set by a thin sliver of its supply that trades freely today, while most of the coins sit locked with insiders who paid a fraction of what you will. Learning to read the cap table and the chain tells you, in advance, who is on the other side of your trade.
>
> - A **cap table** for a token is its allocation: how the total supply is split between team, investors, community, treasury, and market makers — and on what **vesting** schedule each slice unlocks.
> - The number that matters most is **insider overhang**: the share of supply held by team and investors that will unlock on a *known* schedule and can be sold into a market that only floats a fraction of the coins.
> - You can watch the money move for free. **Block explorers** (Etherscan) show every transfer and the top-holder list; **Arkham** and **Nansen** put names on the big wallets; **DefiLlama** and **Token Unlocks** show when locked supply is scheduled to hit.
> - The single most useful on-chain tell is a labeled insider or treasury wallet sending tokens to an **exchange deposit address** — the on-chain footprint that usually comes *before* a sale.
> - In December 2024, a market maker reportedly sold about **66 million MOVE** tokens the day after listing, banking around **\$38 million** — a real, sourced case of listing-day insider selling that on-chain trackers and the exchange both flagged.
> - The defense is a repeatable checklist you can run before you buy: read the allocation, compute the insider percentage and float, check the unlock calendar, label the big wallets, and size the overhang against daily volume.

Here is a puzzle that trips up almost everyone who is new to crypto. A token trades at \$0.50. Its "market cap" — the number the price site shows you — is \$115 million, which sounds like a small, early project with room to run. But the same site, one line down, lists a "fully diluted valuation" of \$500 million. Where did the other \$385 million go? And why does the price keep sagging even though, as far as you can tell, nobody is selling?

The answer is that the price you see is set by a *tiny* fraction of the coins that actually exist. The rest — usually the majority — are locked in the hands of the team, the venture funds that seeded the project, the foundation's treasury, and the market maker that quotes the order book. Those coins are not gone. They are a *future claim* on the market, released on a schedule that is often published in a document you can read, held in wallets you can watch. If you learn to read that document and follow those wallets, the sagging price stops being a mystery. You are simply watching insiders sell their cheap supply into your expensive bids.

This post teaches that skill from zero. We will define every term as we go — cap table, allocation bucket, vesting, wallet, block explorer, on-chain, exchange deposit address, labeling, smart money — and ground each idea in a worked example with round numbers you can do in your head. Then we will point the same tools at a real, documented case. This is educational, not financial advice: the goal is to help you see the machinery, not to tell you to buy or sell anything.

The diagram above the next section is the mental model. Everything else is a tour of its six steps.

![The follow-the-money pipeline: from the tokenomics document, to finding big wallets on a block explorer, to labeling them, to the unlock calendar, to watching flows to exchanges, to sizing the pressure against volume.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-1.webp)

Reading a token's money is not a mystical art. It is a pipeline. Start with the **tokenomics document** to learn who owns what and when it unlocks. Move to a **block explorer** to find the actual wallets that hold those coins. Put names on them with a **labeling tool**. Cross-reference an **unlock calendar** to know when locked supply becomes sellable. Then **watch the flows** — especially tokens heading to an exchange — and finally **size** whatever you see against the token's daily trading volume. Each step is something a beginner can do this afternoon, for free.

## Foundations: the building blocks

Before we can follow the money, we need to agree on what the words mean. If you already trade crypto, skim this; if you do not, read it carefully, because every later section leans on these definitions.

### What a "cap table" is, and why a token has one

In the world of startups, a **capitalization table** — cap table for short — is the spreadsheet that lists who owns the company: how many shares the founders hold, how many the venture investors bought, how many are reserved for employees. It answers one question: *if we sold the whole thing tomorrow, who gets what slice of the money?*

A token has the same thing, and it is usually more honest about it than a startup, because the numbers are published up front. When a crypto project launches a token, it decides in advance how the **total supply** — the maximum number of coins that will ever exist — is divided among different groups. That division is the token's cap table. In crypto it is usually called **tokenomics** (token + economics), and the core of it is the **allocation**: the percentage of supply assigned to each group.

Those groups are the **allocation buckets**. The common ones are:

- **Team / founders / core contributors** — the people who built it.
- **Investors** — the venture funds and angels who put in money during private rounds, often at a steep discount to the eventual public price. (For the fund's-eye view of this, see [crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers).)
- **Community / airdrop** — coins given to early users, sometimes free, to bootstrap a user base.
- **Treasury / foundation** — a reserve controlled by the project's foundation or DAO to fund development, grants, and operations.
- **Ecosystem / rewards** — coins emitted over time to reward staking, liquidity provision, or usage.
- **Liquidity / market maker** — coins lent to a **market maker**, the firm that quotes both a buy and a sell price so the token can trade at all.

Two of these buckets — team and investors — are what we will call **insiders**. They received their coins early and cheaply, and their interests are not automatically aligned with someone buying on the open market months later.

### Vesting: coins you own but cannot sell yet

If insiders could sell all their coins on day one, every launch would collapse instantly under their selling. So projects impose **vesting**: a schedule that releases locked coins gradually rather than all at once. Vesting has two common pieces:

- A **cliff** — a period at the start during which *nothing* unlocks. A "one-year cliff" means an insider's coins are completely frozen for twelve months.
- **Linear vesting** — after the cliff, coins unlock in equal installments (say, monthly) over a further period.

The key insight, which we will return to constantly, is that vesting does not *remove* insider selling — it *schedules* it. A one-year cliff followed by linear unlocks is a promise that a steady stream of new supply will hit the market on dates you can look up today.

### Wallets, addresses, and the block explorer

A **wallet** (or **address**) is an account on a blockchain — a string like `0x7a25...b4c1` that can hold and send tokens. Every wallet's entire history is public. That is the single fact that makes on-chain forensics possible: unlike a bank account, a crypto wallet's transactions are visible to anyone.

The tool that lets you read them is a **block explorer** — a website that indexes a blockchain and lets you look up any wallet, any transaction, or any token. On Ethereum, the standard one is **Etherscan**. Type in an address and you see every coin it has ever received or sent, when, and to whom. Type in a token and you see its **top holders** — the largest wallets and what percentage of supply each controls.

### On-chain vs off-chain

**On-chain** means recorded on the blockchain itself — a transfer of tokens between wallets, visible to all. **Off-chain** means everything that happens *inside* a company's private database. The crucial example: when you deposit tokens to an exchange like Binance and then sell them, only the *deposit* (tokens moving to the exchange's wallet) is on-chain. The actual sale happens off-chain, inside the exchange's private order book, where you cannot see it directly. This is why the deposit is the tell we watch: it is the last public step before the private sale.

### The exchange deposit address

When you sign up for a centralized exchange, it gives you a unique **deposit address** — a wallet the exchange controls, earmarked to receive *your* coins. Send tokens there, and the exchange credits your trading balance. Because these addresses are controlled by the exchange and used only for incoming deposits, analysts can identify them. A large transfer from an insider wallet to a known exchange deposit address is therefore a strong signal: those coins are being positioned to sell.

### Labeling and "smart money"

The chain shows you *addresses*, not *names*. **Labeling** (or attribution) is the work of attaching a real-world identity to an address — "this wallet is Binance's hot wallet," "this one belongs to the Jump Crypto trading desk," "this cluster is the project's treasury." Some labels are published by the entities themselves; most are inferred by analytics firms from behavior and are, at best, high-probability guesses. (There is a whole craft here; see [labeling and attribution](/blog/trading/onchain/labeling-and-attribution).)

**Smart money** is a specific kind of label. Analytics firms tag a set of wallets that have a track record of profitable, well-timed trades and call them "smart money," on the theory that watching what they do is informative. It is a useful lens, but remember two things: it is a vendor's heuristic, not an objective category, and once a wallet is publicly labeled, its owner knows it is being watched. (For the ground-up version, see [what is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain).)

### Float vs FDV

Finally, the two numbers that resolve our opening puzzle:

- **Float** (or circulating supply) — the coins that are actually free to trade right now. The float sets the price, because it is all that is available to buy and sell.
- **Fully diluted valuation (FDV)** — the price multiplied by the *total* supply, as if every coin, locked or not, existed and traded today.

A token with a small float and a large total supply has a market cap that looks tiny next to its FDV. The gap between them is the locked supply — the future claim on the market. Understanding that gap is most of the game. (The structural reasons a token can do this while a stock cannot are the subject of a sibling post, [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock).)

#### Worked example: reading an allocation pie into a table

Let us make this concrete with a hypothetical token we will call ACME. Suppose ACME has a total supply of **1,000,000,000** coins (one billion) and publishes this allocation:

- Team: 18%
- Investors: 22%
- Community / airdrop: 15%
- Treasury / foundation: 25%
- Ecosystem / rewards: 12%
- Liquidity / market maker: 8%

The first thing to do with any pie chart is turn it into a table with real coin counts, because percentages hide the magnitudes. Multiply each share by the one-billion supply:

- Team: 18% of 1B = **180 million** coins
- Investors: 22% = **220 million**
- Community: 15% = **150 million**
- Treasury: 25% = **250 million**
- Ecosystem: 12% = **120 million**
- Liquidity / MM: 8% = **80 million**

The shares add to 100% and the coins add to one billion, so the table is internally consistent. Now add the two columns that actually matter — *when does each bucket unlock*, and *is it tradable today* — and color-code by who benefits.

![Reading ACME's allocation as a table: insiders (team plus investors) hold 40 percent, locked behind a one-year cliff, while only the 23 percent community-plus-liquidity float trades at launch.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-2.webp)

The table exposes what the pie chart hid. Two buckets — team and investors — are **insiders**: 180M + 220M = **400 million coins, or 40% of supply**, and both are locked behind a one-year cliff. Two buckets are liquid at launch: the 150M community airdrop and the 80M market-maker allocation, together **230 million coins, or 23% of supply**. The treasury and ecosystem buckets sit in between — controlled by the project, released at its discretion or over years.

> The intuition to carry forward: at launch, ACME trades on 23% of its supply, but 40% of that supply belongs to insiders who are simply waiting for the clock to run out.

That single sentence — *insiders hold 40%, the float is 23%* — is the reason this whole skill exists.

To put the foundations together: a token's cap table is an allocation split into buckets; vesting schedules turn that split into a calendar of future selling; wallets and block explorers let you find and watch the actual holders; float and FDV measure how much of the supply is a claim on the future rather than a price today. Every later section is a way of turning one of those static facts into a live signal. Hold onto the ACME numbers — 1 billion supply, 40% insiders, 23% float, and a \$0.50 launch price — because we will reuse them all the way through.

## 1. The cap table: who owns what, and what "insider overhang" means

We have the table. Now let us name the risk it reveals. **Insider overhang** is the amount of supply held by insiders that is not yet circulating but will be — the coins hanging over the market like a ledge of snow over a slope. It is "overhang" because everyone can see it up there, and everyone knows it will eventually come down.

Why does overhang matter so much more in crypto than in the stock market? Because of the asymmetry we just measured. In a traditional IPO, insiders are typically locked up for 90 to 180 days, the company has audited financials, and the float grows relatively slowly. A crypto token can list with only 20-30% of its supply circulating, no enforced disclosure, and insiders who bought at a tenth or a hundredth of the listing price. When their coins unlock, the people selling have an enormous cost-basis advantage over the people buying.

#### Worked example: computing insider percentage and reading concentration

Take ACME again. We already found the headline: insiders (team + investors) hold 40% of supply. But "40%" is only the start. A careful reader asks three follow-up questions, each answerable from the cap table:

1. **What is the insider-to-float ratio?** Insiders hold 400M; the launch float is 230M. So the locked insider supply is *1.7 times larger* than everything currently trading (400 ÷ 230 ≈ 1.74). When that supply unlocks, it does not arrive into a deep, liquid market — it arrives into a market that today only floats 230M coins.

2. **How concentrated are the insiders?** 22% investors versus 18% team tells you the venture funds, as a group, own slightly more than the builders. If a block explorer's top-holder list shows those 220M investor coins split across, say, eight venture wallets, that is eight decision-makers who each control ~2.5% of all supply and who all unlock on the same schedule.

3. **What actually floats?** Do not assume the whole 23% is genuinely liquid. Market-maker coins (the 80M) are *lent* for quoting, not necessarily sold; some of the community airdrop may sit in wallets that never move. The tradable float can be even thinner than the table suggests — which, as we will see, makes any selling hit harder.

The lesson: a single percentage is a headline, not an analysis. The cap table is a set of relationships — insider vs float, team vs investors, locked vs liquid — and each one is a question you can answer. (For the on-chain version of question 2, reading real holder distributions, see [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration).)

## 2. Vesting and the overhang you can see coming

The cap table is a snapshot. The **vesting schedule** is the movie: it tells you *when* the locked supply becomes sellable. This is the part beginners most often ignore and professionals watch most closely, because it turns a static risk into a calendar of dated events.

![ACME's vesting timeline: only 23 percent floats at the token generation event, a twelve-month cliff releases nothing, then roughly ten million insider coins unlock every month until the supply is fully vested around month 52.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-3.webp)

Read the timeline left to right. At **month 0** — the **token generation event (TGE)**, the moment the token first exists and lists — only the 230M float is circulating. For the next twelve months, the **cliff** holds: not a single insider coin unlocks. This is the honeymoon. Then, at **month 12**, the cliff ends and the drip begins.

#### Worked example: the overhang at the next unlock

ACME's 400M insider coins vest with a twelve-month cliff, then unlock linearly over the following forty months. How much hits the market each month once the cliff ends?

$$\text{monthly unlock} = \frac{400{,}000{,}000 \text{ coins}}{40 \text{ months}} = 10{,}000{,}000 \text{ coins per month}$$

Ten million coins a month, every month, for over three years. Now put that in context using the float. At month 12, the circulating supply is roughly the 230M launch float plus whatever ecosystem and treasury coins have trickled out — call it 250M for a round number. The first unlock adds 10M coins on top of that:

$$\frac{10{,}000{,}000}{250{,}000{,}000} = 4\% \text{ of circulating supply, in a single month}$$

Four percent of the entire tradable float, arriving as new sellable coins, every single month. And it compounds: by month 24, roughly 120M insider coins have unlocked (12 months × 10M), so the circulating supply has grown by nearly half from insider unlocks alone. This is why so many tokens grind downward for years after listing even with no news — the supply is programmatically increasing, and a large share of it belongs to people sitting on huge gains.

The practical move: for any token you are considering, find the *next* unlock date and its size as a percentage of circulating supply. An unlock worth 1% of float is background noise. An unlock worth 20% of float is an event. (This mechanic gets its own deep dive in [token unlocks, vesting, and emissions](/blog/trading/onchain/token-unlocks-vesting-and-emissions).)

**When this breaks — cliff unlocks vs linear drips.** ACME's linear 10M-per-month schedule is the gentle case. Some tokens use a *cliff* unlock that dumps a huge slug at once — imagine a project where 15% of supply belonging to a single investor unlocks on one date. That is not a drip; it is a step-change in supply, and the market treats it as a scheduled event to trade around. The two shapes call for different reactions: a linear vest is a slow grind you weigh against volume month after month, while a single large cliff is a dated risk you can plan a position around. Always look not just at *how much* unlocks but at *what shape* — a smooth curve or a wall.

## 3. Float vs FDV: the price is a sliver of the supply

Now we can finally resolve the opening puzzle with numbers. The gap between market cap and FDV is not an accounting quirk — it is a map of the overhang.

![Float versus FDV for ACME: only the 23 percent green float sets the 115 million dollar market cap, while the locked treasury and insider supply make up a 500 million dollar fully diluted valuation — the gap is future supply, mostly insiders.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-4.webp)

#### Worked example: market cap, FDV, and the gap

Suppose ACME lists at **\$0.50** per coin. Compute both valuations:

- **Market cap** = price × circulating float = \$0.50 × 230,000,000 = **\$115 million**. This is the number that "feels" like the size of the project.
- **FDV** = price × total supply = \$0.50 × 1,000,000,000 = **\$500 million**. This is what the project is "worth" if every coin traded at today's price.

The gap is \$500M − \$115M = **\$385 million** of value sitting in locked coins. Break that gap down using the cap table: at \$0.50, the 400M insider coins are worth \$200 million, and the 370M treasury-plus-ecosystem coins are worth \$185 million. So of the \$385 million overhang, **\$200 million belongs to team and investors** — money they can realize by selling into the market on the vesting schedule.

Here is why this is the most important number to internalize. To keep the price at \$0.50 as those coins unlock, the market has to absorb up to \$200 million of insider selling *without* the price falling. But the market that is supposed to absorb it only floats \$115 million today. The buyers on the other side would have to nearly triple the dollars they are committing just to hold the line. When people say a token has "bad tokenomics," this arithmetic — a small float carrying a large, insider-heavy FDV — is usually what they mean. (The launch-design version of this game, pricing a huge FDV on a tiny float, is covered in the series post on the low-float / high-FDV playbook.)

> If you cannot see who is selling, assume it is the person who received their coins for a fraction of what you are paying — and that the schedule tells them exactly when to start.

## 4. On-chain forensics from zero: explorers, labels, and smart money

Everything so far came from a document — the tokenomics page. Now we cross over to the chain itself, where we can verify the document and, more importantly, watch what the holders actually *do*. This is where a handful of free and freemium tools turn "someone owns 40%" into "*this wallet*, right here, just moved."

Each tool does one thing well. Using the wrong one for a question wastes hours; using the right one answers it in minutes.

| Tool | What it actually shows | Cost | Best for |
| --- | --- | --- | --- |
| **Etherscan** (block explorer) | Every transaction, transfer, contract call, and event log; a token's **top-holder list** with each wallet's % of supply; public name tags and labels | Free | Ground truth: reading a wallet's raw history and a token's holder distribution |
| **Arkham Intelligence** | Deanonymization — groups fragmented addresses into named **entities** (funds, treasuries, exchanges), draws an entity flow graph, and sends alerts on movements | Free tier | Attribution: "which real entity is this wallet, and where is its money going?" |
| **Nansen** | 300M+ labeled addresses across 25+ chains; **Smart Money** labels; "Token God Mode" showing holdings, inflows/outflows, and exchange flows | Paid | Behavior: "are labeled smart-money or insider wallets accumulating or distributing?" |
| **DefiLlama** | Protocol TVL plus a free **token-unlocks calendar** with dates, amounts, and % of supply for 500+ protocols | Free | The schedule: when locked supply is due to hit, and how big |
| **Token Unlocks** (Tokenomist) | Vesting schedules, allocation breakdowns, next-unlock cliffs, and emission curves for major tokens | Free / paid | The cap table over time: who unlocks, when, and how much |

The workflow chains them together. Start on **Etherscan**: open the token's page, click "Holders," and read the top of the list. You will see a handful of wallets controlling large percentages — plus, usually, some already labeled as exchange wallets or the token's own contracts. Copy the big *unlabeled* ones. Paste them into **Arkham**, which will often resolve them to a named entity — a venture fund, the project's treasury, a market maker — and let you visualize where their coins flow. Cross-check the vesting story on **Token Unlocks** and the next unlock date on **DefiLlama**. If you have a **Nansen** subscription, its Token God Mode summarizes whether the labeled smart-money and insider cohort is, in aggregate, sending coins *to* exchanges or accumulating.

A caution that separates careful analysts from confident ones: **a label is a hypothesis, not a fact.** Arkham and Nansen are usually right about exchange wallets and major funds, but attribution of a specific "team wallet" can be wrong, and clusters can mix. Treat a label as strong evidence to be confirmed by behavior, not as proof. (For the practical craft of building this into a routine, the on-chain series has a full [on-chain due diligence checklist](/blog/trading/onchain/onchain-due-diligence-checklist) and a guide to [following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets).)

#### Worked example: reading concentration off the top-holder list

The tokenomics page tells you the *intended* allocation; the block explorer tells you the *actual* distribution, which is how you verify the document is not lying. Open ACME on Etherscan, click "Holders," and read the top of the list. Suppose it looks like this once you set aside the token's own contracts:

- The vesting contract (locked team + investor coins): **40%**
- The treasury multisig: **25%**
- The market maker's quoting wallet: **8%**
- Eight venture wallets, once vested coins start flowing out: **~2.75% each**
- Everyone else (the airdropped community): the long tail

Two lessons fall out immediately. First, add up the top few controllable holders — treasury 25% plus the eight venture wallets at ~22% plus the team's 18% — and you get **65% of supply in fewer than a dozen decision-makers.** That is the concentration number, and it is far more alarming than any single percentage, because it means a handful of coordinated (or panicking) sellers can move the market.

Second, and this is a beginner trap worth naming: **a large holder is not always a person.** The single biggest "holder" here is a *vesting contract* — a piece of code holding the locked 40% until the schedule releases it. It cannot sell; it can only unlock coins to the wallets behind it. When you read a top-holder list, separate the contract holders (vesting escrows, staking pools, the liquidity pool itself) from the *wallet* holders (people and funds who can actually click sell). Etherscan usually labels the obvious contracts; for the rest, checking whether an address is a contract or a plain wallet is one click. Mistaking a vesting escrow for a whale about to dump — or missing that a "whale" is really eight funds that all unlock on the same day — is the most common way beginners misread concentration.

**When this breaks:** on chains or tokens where holdings are spread across many wallets controlled by one entity (a technique called wallet-splitting), the top-holder list *understates* concentration. That is where labeling tools earn their keep — Arkham grouping fragmented addresses into one entity is precisely the fix for a top-holder list that has been deliberately fragmented to look decentralized.

## 5. The exchange-deposit tell: watching the flows

We have found the wallets and put names on them. Now we watch them. Of all the on-chain signals, the single most actionable is a labeled insider or treasury wallet sending tokens **to an exchange deposit address**. Remember from the foundations: the sale itself happens off-chain, inside the exchange, where you cannot see it. But the *deposit* is on-chain. It is the last public step before a likely sale.

![The on-chain footprint of a sell: a vesting contract releases coins to a labeled insider wallet, which routes eight million tokens to a Binance-tagged deposit address; the exchange credits the balance and the coins are sold into the order book, and the price rolls over as bids are eaten.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-5.webp)

Follow the chain of custody in the figure. A **vesting contract** releases newly unlocked coins to an **insider wallet**. That wallet sends a batch to a **CEX deposit address** that a tool has tagged as belonging to Binance. The exchange's **hot wallet** credits the depositor's balance. The coins are then sold into the **order book**, and as those market-sell orders eat through the resting buy orders (the "bids"), the price rolls over. Only the first two arrows are visible on-chain — but they are enough to see the sell coming before it prints on the chart.

#### Worked example: interpreting a large wallet-to-exchange deposit

You are watching ACME. Your labeling tool flagged wallet `0x9f...2a` as an investor wallet, and Etherscan shows it holds **30 million ACME**. This morning, it sent **8 million ACME** to an address tagged as a Binance deposit wallet. What do you actually know?

First, size it in dollars: 8,000,000 coins × \$0.50 = **\$4 million** of ACME heading to an exchange. Second, size it against the holder: 8M of a 30M position is about **27% of this investor's stack** moving in one transaction. Third — and this is the discipline — state what you *do not* know. A deposit is not a confirmed sale. The investor might be:

- selling (the most common reason, and the base case), or
- moving coins to the exchange to *provide* them as collateral or to an OTC desk, or
- transferring custody for an internal or compliance reason.

So the honest read is: **\$4 million of a known investor's ACME is now positioned to sell, and this is a meaningful fraction of their holdings.** That is a strong, dated, sourced observation — not a proof of intent. The next section shows how to weigh it. (Exchange inflows as a class of signal, and how to read them without over-reading, are the subject of [exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows).)

## 6. Sizing the sell pressure against daily volume

A \$4 million deposit sounds like a lot. Is it? The only way to answer is to compare the potential selling against the market's capacity to absorb it — the token's **daily trading volume**. The same dollar figure is a catastrophe for one token and a non-event for another.

![The same five-million-dollar unlock is a wall for a thin token and a ripple for a deep one: it is a quarter of a twenty-million-dollar daily volume but only 2.5 percent of a two-hundred-million-dollar daily volume.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-6.webp)

#### Worked example: overhang vs volume for two tokens

Recall ACME unlocks 10M coins a month. At \$0.50 that is **\$5 million** of newly sellable supply per month. Now compare two versions of the same market, as the figure above lays out:

- **Token A — thin float.** Daily trading volume is \$20 million. If the month's \$5 million of unlocked supply were sold in a single day, it would be \$5M ÷ \$20M = **25% of a day's volume** — a wall of supply the order book cannot swallow without the price dropping. Even spread across a month of trading, it is persistent, one-directional selling pressure into a shallow book.
- **Token B — deep liquidity.** Daily volume is \$200 million. The same \$5 million is \$5M ÷ \$200M = **2.5% of a day's volume** — a ripple the book absorbs almost invisibly.

Same unlock, same dollars, opposite outcomes. This is the final piece of judgment: **overhang matters relative to liquidity, not in absolute terms.** A \$50 million unlock into a market that trades \$5 billion a day is nothing; a \$500,000 unlock into a token that trades \$200,000 a day is a crisis. Always divide the potential selling by the daily volume before you decide whether an unlock is scary.

This also explains a subtle trap. Insiders and market makers know this arithmetic too. If they can keep *reported* volume high — sometimes through genuine activity, sometimes through wash trading, which the on-chain series covers in [detecting fake volume vs organic demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand) — the overhang looks small by comparison, and their selling is easier to hide. When you size an unlock against volume, sanity-check that the volume is real.

## 7. How it shows up in price

Put the mechanics together and a recognizable pattern emerges around unlocks and dumps — one you can often see *before* the price move rather than after.

The **pre-unlock drift**: markets are forward-looking, so a large, well-known unlock is frequently "priced in" by traders selling *ahead* of the date. A token can drift down for weeks into a cliff, then sometimes bounce on the unlock itself as the anticipated selling turns out to be smaller than feared — the classic "sell the rumor" dynamic applied to supply. The unlock calendar is public, so the reaction to it is a game everyone plays.

To see the pre-unlock drift concretely, stay with ACME. Say the market knows a large cliff is coming at month 18 — a batch worth, at \$0.50, around \$40 million against a token that trades \$20 million a day. Traders who did the overhang-vs-volume arithmetic see a two-day wall of supply approaching and start trimming a week ahead, so the price bleeds from \$0.50 toward \$0.42 into the date. Then, on the unlock, two things can happen. If the unlocked holders sell into the already-lowered price, it keeps falling — the "wall" was real. Or, if much of that supply turns out to be held rather than sold, the feared selling never materializes, shorts cover, and the token bounces — the classic "sell the rumor, buy the news" on supply. Either way, the *calendar* drove the move, and anyone reading it was not surprised.

The **pre-dump wallet move**: this is the on-chain edge. Before a discretionary sale — one not tied to a scheduled unlock — insiders have to move coins to an exchange, and that move is visible. On-chain watchers who see a dormant treasury or insider wallet suddenly send a large batch to an exchange deposit address often see it *hours to days before* the price rolls over. The deposit is the leading indicator; the price drop is the lagging one. This is precisely the sequence the wallet-flow figure above traces, and it is why the exchange-deposit tell is worth watching.

The honest caveat, repeated because it matters: seeing a deposit is not seeing intent. You are observing that coins moved to a venue where they *can* be sold, and inferring — from context, size, and history — that they *will* be. Sometimes you will be wrong. The discipline is to frame every on-chain read as "observed behavior, elevated probability," never "proven plan."

## 8. When the sell tell goes quiet: OTC desks and fresh wallets

If the exchange-deposit tell is so useful, why doesn't everyone see every insider sale coming? Because the sophisticated sellers know you are watching, and there are two well-worn ways to sell without leaving the obvious footprint.

The first is the **OTC desk**. "OTC" means over-the-counter — a trade negotiated privately between two parties rather than executed on a public order book. A large holder who wants to sell \$50 million of a token can call an OTC desk, agree a price, and hand the coins over in a single transfer to the *buyer's* wallet or the desk's wallet. On-chain, this looks like a plain wallet-to-wallet transfer, not a deposit to a recognizable exchange address — so it does not trip the alert. The selling is just as real; it simply happens off the visible order book, and its price impact shows up later, indirectly, as the buyer eventually distributes or hedges. This is why OTC flow is the hardest supply to see, and why "no exchange deposits" is not the same as "no selling." (The mechanics of moving size without moving price get their own treatment in the series post on OTC desks.)

The second is the **fresh wallet**. An insider whose main wallet is labeled can simply move coins to a brand-new, unlabeled address first, then deposit from *that* to the exchange. To a casual watcher, the deposit appears to come from an anonymous wallet with no history, not from "the team." Defeating this is exactly what attribution tools are built for: they follow the coins backward, notice the fresh wallet was funded by a known insider address moments earlier, and re-attach the label. The tell is the *funding trail* — a new wallet that receives a large batch from a labeled insider and immediately forwards it to an exchange is not anonymous; it is an insider wearing a thin disguise.

**What this costs you as a watcher:** the presence of OTC and wallet-hopping means on-chain analysis has a floor of uncertainty. You will catch the lazy and the fast, and miss some of the careful. Treat a *clean* chain — no visible deposits — as "no confirmed selling seen," not "no selling." Combine the on-chain read with the boring fundamentals (the unlock calendar, the float, the concentration) so that when the chain goes quiet, the schedule still tells you supply is coming. (For how to fuse the two, see [combining on-chain with off-chain signals](/blog/trading/onchain/combining-onchain-with-offchain-signals).)

## Common misconceptions

**"A low token price means it's cheap."** Price per coin is meaningless without supply context. A \$0.50 token with a \$500 million FDV on a 23% float can be far more expensive — and carry far more overhang — than a \$50 token with a fully circulating supply. Always look at market cap and FDV, never the sticker price alone.

**"The team is locked up, so insiders can't sell."** Locks expire on a schedule, and that schedule is the point. Worse, "insider" is broader than "team": the treasury, ecosystem, and market-maker buckets are often partly liquid at launch. A locked team does not mean an absence of insider selling — it means the selling is dated, not cancelled.

**"On-chain analysis proves what someone is going to do."** It does not. A transfer to an exchange deposit address is a strong tell, not a confession. Coins can go to an exchange for collateral, OTC settlement, or custody reasons. Treat on-chain reads as elevated-probability observations you can date and source, and say so.

**"Smart-money labels are objective truth."** They are a vendor's heuristic — usually a profitability threshold over some window — and the labeled set rotates as wallets fall in and out of the criteria. Worse, once a wallet is publicly labeled, its owner can bait the watchers, buying in a labeled wallet while selling elsewhere. Follow smart money as one input, not a signal to copy blindly. ([The perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) is a full treatment.)

**"Vesting protects me as a buyer."** Vesting protects the *launch* from a day-one collapse. For a buyer months later, vesting is a schedule of guaranteed future selling. A long, gentle vest can be worse for a holder than a short, brutal one, because it means years of steady supply pressure rather than one clearing event.

**"If it's on a major exchange, it's been vetted."** A listing is a business decision, not a safety certificate. As the next section shows, a token can list on the largest exchanges in the world and still see a market maker sell tens of millions of coins into the order book on day two.

**"No exchange deposits means no one is selling."** The exchange-deposit tell catches the obvious sellers, not the careful ones. A holder can sell privately through an OTC desk — a wallet-to-wallet transfer that never touches a recognizable exchange address — or route coins through a fresh, unlabeled wallet first. A quiet chain means "no confirmed selling *seen*," which is not the same as "no selling." That is exactly why the schedule and the concentration matter even when the flows look calm.

## How it shows up in real markets

The mechanics above are not hypothetical. Here are documented episodes where reading the cap table and the chain would have told you who was on the other side.

### 1. Movement (MOVE): listing-day selling, flagged on-chain

Movement's MOVE token listed on Binance on **December 9, 2024**, trading at roughly a **\$1.3 billion market cap** at launch and simultaneously listing on Coinbase, Upbit, and Bithumb ([CoinDesk, December 9, 2024](https://www.coindesk.com/business/2024/12/09/movement-networks-move-will-be-listed-on-binance-upbit-and-bithumb-amid-airdrop)). One day later, on **December 10**, a market maker sold approximately **66 million MOVE** into the order book "with little buy orders," banking a reported profit of about **\$38 million** ([The Block](https://www.theblock.co/post/347931/binance-move-market-maker-movement-38-million-usdt-buyback-program); [CoinDesk, March 25, 2025](https://www.coindesk.com/markets/2025/03/25/binance-offboards-market-maker-that-it-said-made-usd38m-profit-on-move-listing)).

![Timeline of the MOVE listing-day selling: MOVE lists on Binance at roughly a 1.3 billion dollar market cap on December 9, 2024; the market maker sells about 66 million MOVE into thin bids the next day for a reported 38 million dollar profit; Binance offboards the market maker in March 2025 and a 38 million dollar buyback is pledged.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-7.webp)

The structure is exactly the "liquidity / market maker" bucket from our cap table, gone wrong. The market maker had been supplied MOVE tokens to provide two-sided liquidity — to quote both a buy and a sell price — but instead sold heavily into the launch. This is worth dwelling on, because it is the bucket beginners most often overlook. The market-maker allocation is not sitting quietly like a treasury reserve; it is *lent* to a trading firm with its own incentives, and in crypto those arrangements often let the firm profit directly from the token's launch. When a project hands a market maker a large slug of coins, it is trusting that firm to quote fairly rather than to sell — a trust that, in the MOVE case, was reportedly broken. (The economics of these loan-and-option deals, and why market makers can profit from launching the very token they quote, are the subject of the series post on how market makers get paid.) Reporting described a firm, Rentech, positioned on both ends of the arrangement in a way that let it collect the ~66 million MOVE, with Web3Port identified as the market maker behind the distribution; the Movement Network Foundation stated it had no knowledge this was happening and that the entity had acted against the project's wishes and in breach of its agreement. Binance investigated, **offboarded the market maker in March 2025**, and the foundation pledged to use recovered proceeds for a **\$38 million buyback** of MOVE. Coinbase later moved to suspend MOVE trading (transitioning to limit-only) effective mid-May 2025 as the token no longer met its listing criteria.

The lesson for a reader: the coins that hit the market came from a *known bucket* (the market-maker allocation), the selling left an on-chain footprint that trackers and the exchange both surfaced, and the biggest venues in crypto did not prevent it. Framed carefully — this is reported and observed on-chain behavior, with the intent contested by the parties — it is a near-perfect case study of listing-day insider supply. (For the exchanges'-eye view of listing power, see [centralized crypto exchanges: Binance and Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase).)

### 2. Arbitrum (ARB): the foundation that sold before the vote

Arbitrum's ARB token has a fixed total supply of **10 billion**, allocated (per the Arbitrum Foundation's published distribution) as roughly **42.78% to the DAO treasury, 26.94% to Offchain Labs team and advisors, 17.53% to investors, 11.62% to a user airdrop, and 1.13% to ecosystem DAOs** ([Arbitrum Foundation docs](https://docs.arbitrum.foundation/airdrop-eligibility-distribution)). Read as a cap table, insiders (team + investors) hold **44.47%** of supply, subject to a four-year lockup with first unlocks one year after the March 2023 token generation event, then monthly.

In early April 2023, days after the airdrop, controversy erupted over the treasury. Before the DAO had ratified the foundation's roughly \$1 billion budget (about 750 million ARB), the foundation had already moved **50 million ARB** — reportedly selling 10 million on exchanges for stablecoins to fund operations and lending 40 million to the market maker Wintermute, with the rest in a multisig ([CoinDesk, April 2, 2023](https://www.coindesk.com/business/2023/04/02/contentious-arbitrum-vote-over-1b-in-tokens-ratification-not-request-says-foundation)). ARB fell roughly **9-11%** on the news ([BeInCrypto](https://beincrypto.com/arb-drops-11-foundation-dumps-50m-tokens/)). Here the mover was the *treasury* bucket — the least-discussed large holder — and the on-chain transfers to exchanges and to a market maker were public. A reader watching the treasury wallet would have seen the coins move before the price reacted.

### 3. The unlock-calendar era: scheduled supply as a tradeable event

The reason unlocks are now a first-class market signal is that they became *watchable at scale*. **DefiLlama** publishes a free token-unlocks calendar covering 500+ protocols, with dates, amounts, and each unlock's size as a percentage of supply; **Token Unlocks** (now Tokenomist) does the same with vesting and allocation breakdowns. As of 2026, large scheduled monthly unlocks for major tokens are routinely tracked and traded — the "cliff" is a known date on a public calendar, and the market's anticipation of it (the pre-unlock drift discussed earlier) is itself part of the price. The takeaway is not any single number but a structural fact: because the schedule is public, *you have no excuse for being surprised by an unlock*. The calendar is one click away.

### 4. Deanonymization: when treasury and insider wallets became public objects

The final shift is attribution. Firms like **Nansen** (300M+ labeled addresses across 25+ chains) and **Arkham** (which groups addresses into named entities and pays a bounty marketplace to expand its labels) have made it routine to point at a specific wallet and say "this is a fund," "this is a treasury," "this is an exchange deposit address." That turned the abstract "insiders hold 40%" into concrete, watchable objects — and created a reflexive game. Insiders know their labeled wallets are watched, so some route selling through fresh, unlabeled wallets or OTC desks to stay off the radar, while watchers hunt for the new wallets. The result is an ongoing cat-and-mouse between the people moving supply and the people trying to see it move — a dynamic the on-chain series explores in [what is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain). For you, the practical upshot is simply that the tools exist and are largely free: the big holders are not as invisible as they were even a few years ago.

## The follow-the-money checklist before you buy

Here is the whole post compressed into a routine you can run in fifteen minutes, before you commit a dollar. It is the retail defense — a way to see who is on the other side of your trade.

![The follow-the-money checklist as a decision flow: five reads on the cap table and the chain — insider percentage and float, next unlock, big wallets to exchanges, and overhang versus volume — tell you whether you are the buyer or the exit liquidity.](/imgs/blogs/follow-the-money-reading-a-tokens-cap-table-8.webp)

1. **Read the allocation.** Find the tokenomics page. Turn the pie into a table with coin counts. Compute the **insider percentage** (team + investors) and the **launch float** (what actually trades). If insiders hold a large multiple of the float, note it.
2. **Check the unlock calendar.** On DefiLlama or Token Unlocks, find the **next unlock date and its size as a percentage of circulating supply**. A cliff in the next few weeks is a dated event, not a vague risk.
3. **Compare float and FDV.** If the FDV dwarfs the market cap, the gap is future supply. Estimate how much of that gap is insider-owned — that is the money that can be sold into your bids.
4. **Label the big wallets.** On Etherscan, read the top-holder list. Paste the big unlabeled wallets into Arkham to attribute them. Note where the insider and treasury coins sit.
5. **Set deposit alerts and size the overhang.** Watch for insider or treasury wallets sending coins to exchange deposit addresses — the sell tell. When you see one, size it against **daily volume** to judge whether it is a wall or a ripple.

Run those five checks and one of two pictures emerges. Either the cap table is concentrated, an unlock is near, wallets are flowing to exchanges, and the overhang dwarfs the volume — in which case you may be the *exit liquidity* for insiders, and the honest move is to step back or size the position very small. Or the checks come back clean — a real float, a distant or small unlock, no ominous flows, an overhang the volume can absorb — in which case at least you are buying with your eyes open. The checklist does not tell you what to do. It tells you *who else is in the room.*

## When this matters to you

If you ever buy a token — especially a newly launched one — this is the difference between guessing and seeing. Most of the pain retail buyers take in crypto is not from scams in the criminal sense; it is from buying the thin, insider-heavy float of a token whose real supply is scheduled to land on their heads. That pattern is legible in advance, for free, from a document and a block explorer. You do not need a Nansen subscription or a trading desk. You need the tokenomics page, Etherscan, a unlock calendar, and the willingness to do the arithmetic in this post.

It also reframes the whole "who moves crypto prices" question that this series is about. The funds, market makers, foundations, and exchanges we profile elsewhere are not shadowy — their footprints are on-chain and their supply is on a schedule. Reading the cap table is how you connect the players to the price you actually see. For the next layers, the sibling posts cover [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock), [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock), and [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move).

Educational, not advice: nothing here is a recommendation to buy or sell any token. The point is sight, not a signal. When you can see the supply, you can decide for yourself.

## Sources & further reading

Primary sources behind the headline figures:

- Movement / MOVE market-maker case: [The Block — "Binance identifies alleged MOVE-dumping market maker; Movement commits to \$38 million buyback"](https://www.theblock.co/post/347931/binance-move-market-maker-movement-38-million-usdt-buyback-program); [CoinDesk — "Binance Offboards Market Maker that it said made \$38M profit on MOVE listing" (March 25, 2025)](https://www.coindesk.com/markets/2025/03/25/binance-offboards-market-maker-that-it-said-made-usd38m-profit-on-move-listing); [CoinDesk — MOVE listing (December 9, 2024)](https://www.coindesk.com/business/2024/12/09/movement-networks-move-will-be-listed-on-binance-upbit-and-bithumb-amid-airdrop).
- Arbitrum / ARB allocation and April 2023 treasury sale: [Arbitrum Foundation — airdrop eligibility & distribution](https://docs.arbitrum.foundation/airdrop-eligibility-distribution); [CoinDesk — "Arbitrum Foundation Sold ARB Tokens Ahead of 'Ratification' Vote" (April 2, 2023)](https://www.coindesk.com/business/2023/04/02/contentious-arbitrum-vote-over-1b-in-tokens-ratification-not-request-says-foundation); [BeInCrypto — "ARB Drops 11% as Foundation Sells 50M Tokens"](https://beincrypto.com/arb-drops-11-foundation-dumps-50m-tokens/).
- Tools: [DefiLlama token-unlocks calendar](https://defillama.com/unlocks); [Nansen — on-chain tracking and Smart Money](https://www.nansen.ai/post/how-to-monitor-wallet-activity-track-smart-money-in-crypto); [Etherscan — public name tags, labels & notes](https://info.etherscan.com/public-name-tags-labels/).

Related posts on this blog:

- [Crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the players behind the insider and liquidity buckets.
- [Token unlocks, vesting, and emissions](/blog/trading/onchain/token-unlocks-vesting-and-emissions) and [exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — the on-chain mechanics in depth.
- [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution), [following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets), and [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — building the forensics habit.
- Series siblings: [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock), [the lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock), and [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move).
