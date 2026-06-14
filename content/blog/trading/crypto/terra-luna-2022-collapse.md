---
title: "The Terra-Luna Collapse: How a 40 Billion Dollar Algorithmic Stablecoin Died in Days"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How TerraUSD, a dollar kept stable by an algorithm instead of reserves, fell into a reflexive death spiral that erased tens of billions in a single week."
tags: ["terra", "luna", "ust", "stablecoins", "algorithmic-stablecoin", "defi", "crypto", "death-spiral", "anchor-protocol", "do-kwon", "case-study"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — TerraUSD (UST) promised a decentralized dollar held stable not by cash reserves but by an algorithm and a sister token, LUNA; when confidence cracked in May 2022, the design's own feedback loop turned it into a tens-of-billions death spiral within days.
>
> - A "stablecoin" is a crypto token meant to always be worth one dollar. UST tried to hold its peg with code, not collateral.
> - The mechanism: you could always burn 1 UST for exactly one dollar's worth of newly minted LUNA, and vice versa. Arbitrage was supposed to nudge UST back to \$1.
> - Demand for UST was manufactured by a lending app, Anchor, paying a near-fixed ~20% yield — a subsidy dressed up as interest, not organic return.
> - When large holders pulled UST out and sold, the peg broke; defending it minted ever more LUNA, which collapsed LUNA's price, which scared more holders, which broke the peg further. That is a reflexive death spiral.
> - LUNA's supply exploded from roughly 350 million tokens to about 6.5 trillion in days; its price fell from roughly \$80 to fractions of a cent. Around \$40-60 billion in combined value evaporated.
> - The wreckage spread: leveraged funds like Three Arrows Capital and lenders like Celsius and Voyager were holding the rubble, and their failures defined the 2022 crypto winter.

The diagram above is the mental model for everything that follows: a peg that had held for more than a year unwound from a one-cent wobble to near-total collapse in roughly five days, because the very tool meant to defend it — minting LUNA — became the accelerant that destroyed it. Hold that single image in your head. The rest of this post is the slow-motion replay of how each day made the next day worse, and why a stablecoin with no hard asset behind it was always one bad week away from this.

![Timeline of the May 2022 Terra and Luna collapse over five days](/imgs/blogs/terra-luna-2022-collapse-1.png)

This is a case study, not a victory lap. Terra was, for a while, one of the most celebrated projects in crypto, with serious investors, real engineers, and a community that genuinely believed they were building a better dollar. The collapse is interesting precisely because the design was clever, not stupid. It worked for a long time. Understanding *why* it worked for a long time, and then why it could not survive a loss of confidence, teaches you something durable about money itself — about pegs, runs, reflexivity, and the difference between "stable because it is backed" and "stable because everyone believes it is stable." Let us build all of that from zero.

## First principles: what a stablecoin actually is

Before we can explain how Terra died, we need to define every term the story turns on. If you have never touched crypto, do not skip this section — it is the whole foundation, and the failure is impossible to understand without it. I will define each piece inline the first time it appears.

A **blockchain** is a shared public ledger — a database that nobody owns and everybody can read — that records who holds what and who sent what to whom. Instead of a bank's private server tracking your balance, thousands of computers around the world keep identical copies of the ledger and agree, by a voting process called *consensus*, on which transactions are valid. Terra ran on its own blockchain (the Terra chain). Ethereum and Bitcoin run on theirs.

A **token** (or **coin**) is simply an entry on that ledger that represents some unit of value or ownership. "Alice holds 100 LUNA" is a row in the ledger. Tokens can be created (**minted**) and destroyed (**burned**) according to rules written into the chain's software. This minting and burning is central to the Terra story, so keep it in mind: on a blockchain, you can genuinely create or destroy units of a token by following the protocol's rules, the way a central bank can print or retire currency — except here the rules are code.

A **wallet** is the software that holds your private keys — the secret codes that let you authorize moving your tokens. "Self-custody" means *you* hold those keys; nobody can freeze or seize your tokens. **On-chain** means something happens by a recorded transaction on the blockchain itself (as opposed to **off-chain**, on a company's internal books). A **smart contract** is a program that lives on the blockchain and runs automatically when called — it can hold tokens, enforce rules, and pay people out without a human in the loop. Anchor, the lending app at the heart of this story, was a set of smart contracts.

One more piece of plumbing matters for the ending. The Terra chain used a consensus method called **proof of stake**, in which the right to validate transactions and produce new blocks is given to those who lock up — "stake" — large amounts of the chain's native token, here LUNA. Validators stake LUNA, earn rewards, and *secure* the chain; in return, controlling a majority of the staked LUNA means controlling the chain itself. Hold that fact: when LUNA's price later collapsed to fractions of a cent, the cost of buying enough LUNA to seize control of the chain collapsed with it, which is why the validators eventually had to hit the emergency brake. **Staking** is also how Anchor's collateral worked — borrowers posted *bonded* (staked) LUNA, called bLUNA, as collateral. So LUNA was simultaneously the security of the chain, the collateral inside Anchor, and the shock absorber for UST. Three load-bearing roles, one token. When that one token fell, all three failed at once.

Now the key term: a **stablecoin** is a crypto token engineered to hold a constant value, almost always one US dollar. Ordinary crypto tokens swing wildly — Bitcoin can move 10% in a day. That makes them terrible for paying rent or pricing a loan. A stablecoin is supposed to be the boring, dependable dollar *inside* the crypto system: 1 token = \$1, today, tomorrow, and next year. The price it is supposed to hold is called its **peg**. When a stablecoin trades at exactly \$1.00 it is "on peg"; when it slips to \$0.98 or \$0.90 it has "de-pegged," and that is an alarm bell, because the entire point of the thing is that it never does.

### Two ways to keep a coin at a dollar

There are fundamentally two families of stablecoin, and the distinction is the crux of this entire case study.

**Collateralized stablecoins** are backed by real assets you could redeem the coin for. The dominant examples are Tether (USDT) and USD Coin (USDC). For (roughly) every token in circulation, the issuer claims to hold one dollar of reserves — cash, short-term US Treasury bills, and similar — sitting in bank accounts and custody. The promise is concrete: hand the issuer one token, get one real dollar back. If the price wobbles below \$1 in the market, large players buy the cheap token, redeem it for a full dollar from the issuer, and pocket the difference; that buying pressure shoves the price back to \$1. The peg is held by an *asset floor*. (The real-world plumbing and risks of these reserves — the "shadow dollar" of crypto — are their own deep topic, covered in [stablecoins and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar).)

**Algorithmic stablecoins** hold no such redeemable asset floor. Instead, they try to keep the peg with code and market incentives. There is no vault of dollars you can claim. UST was the most famous algorithmic stablecoin, and the mechanism it used — mint-and-burn against a sister token — is what we will dissect next. The one-sentence preview: an algorithmic stablecoin is stable as long as *the market believes it is stable*, and it has no hard backstop for the day the market stops believing.

![Comparison matrix of algorithmic versus collateralized stablecoins](/imgs/blogs/terra-luna-2022-collapse-6.png)

The matrix above is the comparison to keep returning to. The difference is not cosmetic. A collateralized coin in a crisis can be redeemed for a hard asset, so its worst case is a *reserve shortfall* — bad, but bounded. An algorithmic coin in a crisis has nothing to redeem into except more of a token whose price is collapsing *because* of the crisis. Its worst case is unbounded: zero.

### The UST and LUNA pair: mint, burn, and the arbitrage that was supposed to hold the peg

Terra had two linked tokens. **UST** (TerraUSD) was the stablecoin, the one meant to stay at \$1. **LUNA** was its sister token — a normal volatile crypto asset whose price floated freely and which was meant to *absorb* UST's volatility. Think of LUNA as the shock absorber and UST as the smooth ride on top.

The link between them was a rule baked into the Terra protocol, available to anyone, at any time:

```
You may always burn 1 UST and receive exactly $1.00 worth of newly minted LUNA.
You may always burn $1.00 worth of LUNA and receive exactly 1 newly minted UST.
```

Read that twice, because the whole design lives in it. The protocol values UST at \$1.00 *for the purpose of this swap*, no matter what UST is actually trading for on the open market. The amount of LUNA you get is computed from LUNA's current market price. If LUNA is \$80, then \$1 of LUNA is 0.0125 LUNA; burn 1 UST and the protocol mints you 0.0125 LUNA.

Why would this hold the peg? Through **arbitrage** — the practice of buying something cheap in one place and selling it dear in another to capture a risk-free spread, an activity that, as a side effect, pushes the two prices together. Here is the conceptual loop, and I will walk the exact numbers in a worked example below.

![Pipeline of the UST and LUNA mint-and-burn swap and the arbitrage profit](/imgs/blogs/terra-luna-2022-collapse-3.png)

The pipeline above is the arbitrage in five steps. Suppose UST slips to \$0.98 on the open market. An arbitrageur buys UST cheaply at \$0.98, then uses the protocol to burn that 1 UST for \$1.00 of freshly minted LUNA, sells the LUNA for \$1.00, and walks away with a 2-cent profit. Crucially, every UST burned this way is *destroyed* — it leaves circulation. As arbitrageurs repeat this, UST supply shrinks, scarcity pushes UST's price back up toward \$1.00, and the peg is restored. The symmetric move handles UST trading *above* \$1: mint cheap UST by burning LUNA, sell the UST for more than a dollar, and the new supply pushes the price back down.

This is genuinely elegant, and it is why Terra worked for over a year. As long as LUNA had a large, liquid market — far larger than the UST that might need to be redeemed against it — the shock absorber could soak up shocks. The fatal assumption, which we will return to, is hiding in that "as long as."

### Seigniorage, Anchor's 20%, and the words for what goes wrong

Two more concepts, then we have our full vocabulary.

**Seigniorage** is the profit a money-issuer earns from creating money. A government mints a coin that costs five cents of metal and spends it as one dollar; the 95-cent gap is seigniorage. Terra's design was sometimes called a "seigniorage-style" stablecoin because expanding the UST supply (minting UST by burning LUNA when demand was high) captured value that flowed to the LUNA side. When the system was growing, this looked like a money machine. The machine ran in reverse just as efficiently.

The single most important non-mechanism fact about Terra is **Anchor Protocol**. Anchor was a savings-and-lending app built on Terra. You deposited UST and Anchor paid you a yield that hovered near **20% APY** — twenty percent a year, on a "stable" dollar, at a time when a US savings account paid well under 1%. That yield is what made millions of people *want* to hold UST, which is what created demand for UST, which is what kept the whole machine spinning. We will dissect, with arithmetic, exactly where that 20% came from. Spoiler: most of it was not organic interest. It was a subsidy — a reserve being spent down to manufacture demand. Hold the word **subsidy** here.

Finally, the two terms that name the failure:

A **bank run** is what happens when too many holders of a claim try to redeem at once and the issuer cannot honor them all simultaneously. Classic banks are vulnerable because they lend out deposits, so they cannot return everyone's cash on the same day; the *fear* that others will withdraw first makes withdrawing first rational, which makes the run self-fulfilling. (The mechanics of how banks create and lend out money — and why this fragility is structural — are laid out in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).)

A **reflexive death spiral** is a bank run with a turbocharger. "Reflexive" means the outcome feeds back into the cause: the act of fleeing makes the thing worth less, which gives everyone more reason to flee. In a normal bank run the bank's *assets* are fixed; in Terra, the defense mechanism itself manufactured the poison. Selling UST broke the peg, defending the peg minted LUNA, minting LUNA crushed LUNA's price, and a crushed LUNA price meant the shock absorber could no longer absorb anything — so the peg broke harder. That is the loop the whole post is about, and it is worth seeing as a picture before we watch it happen.

![Forward graph of the reflexive death spiral feeding into hyperinflation](/imgs/blogs/terra-luna-2022-collapse-2.png)

The graph above traces the loop the way it actually ran: a de-peg triggers arbitrage, arbitrage balloons LUNA's supply, the ballooning supply both craters LUNA's price *and* spooks holders into fleeing, and both of those terminate in the same place — a hyperinflationary spiral. Notice there is no self-correcting arrow anywhere. Every path points downhill.

## The setup: Do Kwon, Terraform Labs, and a machine built to manufacture demand

Terra was created by **Terraform Labs**, a company co-founded in 2018 by **Do Kwon**, a young, combative, intensely confident South Korean entrepreneur who became the public face of the project. Kwon was famous for dismissing critics — he once publicly bet \$1 million that LUNA would not fall, and routinely mocked anyone who questioned the design's stability. That bravado matters to the story only because it shaped how the community and the market read warning signs: doubt was treated as ignorance or jealousy, right up until the doubt was vindicated catastrophically.

The product had three layers stacked on top of each other, and each layer existed to prop up the one below it.

**Layer one was UST itself**, the algorithmic dollar with the mint-burn mechanism we just described. On its own, UST was just a tool. A tool needs users.

**Layer two was LUNA**, the volatility-absorbing sister token. LUNA was where speculation and value accrued. As UST adoption grew, the protocol burned LUNA to mint UST, shrinking LUNA supply and (when demand was rising) pushing LUNA's price up. LUNA holders were, in effect, betting that UST demand would keep growing forever. At its peak in April 2022, LUNA traded around \$80-120 and the two tokens together were worth roughly \$40 billion or more — by some measures LUNA alone was a top-ten crypto asset.

**Layer three was Anchor**, and Anchor is where the demand actually came from. A stablecoin is only useful if people hold it, and people hold a "boring dollar" only if there is a reason to. Anchor manufactured the reason by paying that ~20% yield. By early 2022, of the roughly \$18 billion of UST in existence, around \$14 billion — the large majority — was parked inside Anchor earning that yield. Sit with that number. The overwhelming reason UST existed at all was a single app paying an extraordinary, near-fixed return. Remove the 20%, and most of the demand for UST had no reason to stay.

### Where the 20% actually came from

Anchor was nominally a money market: depositors lend UST, borrowers post collateral (mostly *staked* LUNA and similar assets) and borrow UST, and borrowers pay interest that funds the depositor yield. In an honest, self-sustaining money market, the deposit rate must be *below* what borrowers pay, because borrower interest is the only source of depositor income. Anchor's problem was that borrowers never paid enough to fund a 20% deposit rate. Borrowing demand was far smaller than deposit demand — of course it was, because being paid 20% to deposit is wildly more attractive than paying to borrow.

So the gap was filled artificially. Anchor ran a **yield reserve** — a pot of money that topped up the shortfall whenever borrower interest fell short of the 20% promise. That reserve was repeatedly refilled by Terraform Labs and affiliated entities with hundreds of millions of dollars (a \$450 million top-up in early 2022 was the most famous). On top of that, borrowers were effectively *paid to borrow* via LUNA token rewards — distributions of LUNA that diluted everyone but made borrowing look free or even profitable.

![Stack showing the layers funding Anchor's 20% yield](/imgs/blogs/terra-luna-2022-collapse-5.png)

The stack above is the honest accounting of that 20%. The top layer is the promise. The next layer is the only organic source — real borrower interest, which covered perhaps half. Everything below the dashed line is subsidy: the yield reserve being spent down, and LUNA token emissions diluting holders. A return funded by a reserve that is being drained and by printing your own token is not interest. It is a marketing budget. It can run for a long time — long enough to attract \$14 billion — but it is a clock, not a business, and everyone holding UST for the yield was, knowingly or not, standing on a layer that had to be refilled from somewhere.

#### Worked example: the 20% Anchor yield on a 10,000 dollar deposit

Suppose you deposited \$10,000 of UST into Anchor in early 2022, believing it was a safe dollar account paying 20%.

- At 20% APY, your promised annual income is \$10,000 x 0.20 = \$2,000 a year, or about \$167 a month. By comparison, \$10,000 in a typical US bank savings account that year earned well under \$50 for the *entire year*. The gap is enormous, and an enormous gap should always prompt the question: who is paying it?
- Now trace the funding. Borrowers in Anchor were paying interest at a rate that, after accounting for how little was actually borrowed relative to deposits, generated only enough to cover roughly half of the promised yield — call it \$1,000 of your \$2,000.
- The other \$1,000 had to come from the yield reserve and token subsidies. So of your \$2,000 "interest," roughly \$1,000 was real and roughly \$1,000 was a reserve being drained to keep you happy.
- At the system's scale, with around \$14 billion deposited at 20%, the protocol owed depositors on the order of \$2.8 billion a year in yield. Borrower interest covered a fraction; the rest bled the reserve at hundreds of millions of dollars per quarter. The reserve, even after the \$450 million top-up, was visibly on a path to run dry within months.

The intuition: a yield far above the risk-free rate, funded by a reserve that is shrinking rather than by income that is growing, is not a high interest rate — it is a countdown, and the deposits chasing it are the fuel that makes the eventual unwind violent.

## The blow-up, day by day

In April 2022 everything looked fine. UST was at \$1.00, LUNA near its highs, and the Terra community was riding high — Terraform Labs had even started building a war chest of Bitcoin (more on that in a moment) to make the peg look bulletproof. Then, over a single week in May, the whole structure came apart. The figure at the very top of this post is the day-by-day chronology; here is the narration.

It is worth noting that this was not the *first* time UST had wobbled. In May 2021, almost exactly a year earlier, UST had briefly de-pegged to around \$0.90 during a sharp market sell-off. Back then the peg was restored, and the episode was held up as proof the design worked — the algorithm had taken a punch and recovered. The SEC would later allege that the 2021 recovery was not the algorithm healing itself at all, but the quiet result of a large trading firm stepping in to buy UST under an arrangement that was not disclosed to investors. If true, that matters enormously: it means the market's confidence in the design's resilience was partly built on a rescue that the public mistook for a self-healing mechanism. The 2022 holders who believed "it de-pegged before and bounced right back" were, on this account, relying on a precedent that had been engineered, not earned.

### The conditions before the spark

Two things made the system flammable that spring. First, the macro backdrop had turned: the US Federal Reserve was raising interest rates aggressively to fight inflation, risk assets everywhere were selling off, and crypto in particular was deep in a downturn. A 20% "risk-free" yield looks a lot less magical when the whole market is falling and you start to wonder whether the dollar paying it is really a dollar. Second, and more technically, Terra had recently moved a large amount of UST into a new liquidity pool on a platform called Curve as part of a planned expansion (the "4pool"). The transition temporarily *thinned* the liquidity available to defend the peg — meaning a large sell order could move the price more than usual. The kindling was dry.

### May 7: the first large withdrawals

On May 7, several very large holders pulled hundreds of millions of dollars of UST out of Anchor and began swapping it. Whether this was a deliberate, coordinated attack designed to break the peg and profit from the fallout, or simply large players de-risking into a falling market, is still debated and may never be fully resolved. The mechanism does not care about intent. What matters is that a large volume of UST hit a market that, post-4pool, had thinner liquidity to absorb it. UST started to slip below \$1.00.

### May 9: the peg breaks in earnest

On May 9, UST fell to roughly \$0.985 and kept sliding. Under normal conditions, this is exactly when the arbitrage is supposed to kick in and pull it back: buy UST cheap, burn it for \$1 of LUNA, sell the LUNA, repeat. And the arbitrage *did* kick in — that was the problem. With UST de-pegging hard and the market already in fear, the arbitrageurs burning UST were minting enormous quantities of LUNA and immediately *selling* it, because nobody wanted to hold a token in the middle of a crisis. That selling crushed LUNA's price. As LUNA's price fell, each \$1 of redeemed UST required minting *more* LUNA units — and minting more LUNA pushed its price down further. The shock absorber was being shredded by the very shock it was supposed to absorb.

### May 10-11: the Bitcoin defense fails, UST hits 30 cents

To make the peg look credible, Terraform Labs had set up the **Luna Foundation Guard (LFG)**, a separate entity that had accumulated a reserve of around \$3 billion in Bitcoin and other assets, explicitly as a backstop to buy UST and defend the peg in a crisis. This is exactly the crisis it was built for, and on May 10 the LFG began deploying it — selling its Bitcoin and using the proceeds to buy UST and prop up the price.

It did not work, for two compounding reasons. First, \$3 billion is small next to \$18 billion of UST trying to exit and a collapsing \$40 billion LUNA market — the backstop was a fraction of the panic. Second, the LFG dumping billions in Bitcoin into an already-falling market *pushed Bitcoin's price down too*, which deepened the broader crypto sell-off and made everyone more eager to flee everything, including UST. The defense fed the fire. By May 11, UST had fallen to roughly \$0.30 — thirty cents on a dollar that was supposed to never move.

### May 12-13: hyperinflation and the halt

By May 12, the spiral was in full hyperinflation. LUNA's supply, which had been around 350 million tokens, was exploding into the *trillions* as the protocol minted astronomical amounts to honor UST redemptions at a LUNA price approaching zero. LUNA, which had been ~\$80 in early April, fell below \$0.01 — a fraction of a cent. The two tokens had inverted from their healthy state: the LUNA meant to backstop UST was now worth far less than the UST it was supposed to support, so each redemption minted absurd quantities of a near-worthless token.

On May 12-13 the Terra validators **halted the blockchain** — literally stopped processing transactions — twice, to prevent governance attacks now that LUNA was so cheap that an attacker could buy enough of it to control the chain for pocket change. UST briefly traded in the \$0.10-0.30 range and never meaningfully recovered. Within roughly five days, a stablecoin worth \$18 billion and a sister token worth tens of billions had both gone, for practical purposes, to zero. Combined, on the order of \$40-60 billion of nominal value evaporated, depending on whether you measure from the April peaks or the early-May levels.

## The mechanism dissected: why the same loop that held the peg also destroyed it

The eerie thing about Terra is that nothing new happened in the collapse. The exact same mint-burn mechanism that held the peg for a year is the mechanism that destroyed it in a week. It did not break; it ran *as designed*, in the direction nobody wanted. Let us take the loop apart piece by piece with numbers.

### Step one: the arbitrage that was supposed to help

In calm conditions, the arbitrage is a stabilizer because LUNA is a deep, liquid market that dwarfs the UST being redeemed against it. Burning a few million UST to mint a few million dollars of LUNA barely moves LUNA's price, the UST that gets burned leaves circulation, scarcity nudges UST back to \$1, and balance is restored. This is the "before" state.

![Before-and-after of the peg holding versus the spiral when LUNA shrinks](/imgs/blogs/terra-luna-2022-collapse-4.png)

The before-and-after above is the hinge of the entire story. On the left, the peg holds because LUNA's market cap (roughly \$30 billion) is far larger than the UST it must absorb (roughly \$18 billion), so arbitrage mints only small amounts of LUNA and the price barely flinches. On the right, once panic drives LUNA's price down far enough that LUNA's market cap falls *below* the UST trying to redeem, the same arbitrage must mint enormous, then astronomical, quantities of LUNA — and each batch craters the price further. The mechanism did not change. The *ratio* changed, and the ratio was the only thing keeping the loop pointed uphill.

#### Worked example: the mint-burn arbitrage at a de-peg to 0.98 dollars

Take the healthy case first, with LUNA at \$80.

- UST is trading at \$0.98 on the open market. The protocol still values 1 UST at \$1.00 for redemption.
- You buy 1 UST for \$0.98.
- You burn that 1 UST through the protocol and receive \$1.00 worth of newly minted LUNA. At \$80 per LUNA, that is 1.00 / 80 = 0.0125 LUNA.
- You sell the 0.0125 LUNA for \$1.00.
- Your profit is \$1.00 - \$0.98 = \$0.02, risk-free, per UST.
- Two effects ripple out: the 1 UST you burned is gone from circulation (UST supply shrinks, nudging the price up toward \$1.00), and you minted only 0.0125 of LUNA — a rounding error against hundreds of millions of LUNA in existence, so LUNA's price barely moves.

The intuition: when LUNA is large and liquid, the arbitrage destroys UST and creates negligible LUNA, so it gently pulls UST back to its peg exactly as designed.

### Step two: the same arbitrage as poison

Now run the identical mechanism in the panic, where LUNA's price is collapsing as it gets minted and dumped.

#### Worked example: the death-spiral arithmetic round by round

Watch what minting 1 dollar of LUNA does to its supply and price once selling overwhelms the market. The exact figures here are illustrative — chosen to show the *shape* of the collapse, which is what matters — but they track the real sequence of LUNA's fall from roughly \$80 to fractions of a cent.

- **Round 1.** LUNA is \$80. A wave of UST redemptions burns UST and mints, say, \$300 million of LUNA: that is 300,000,000 / 80 = 3.75 million new LUNA. Arbitrageurs dump it. The flood of sell orders pushes LUNA down to, say, \$30.
- **Round 2.** LUNA is now \$30. The de-peg has not healed — if anything the falling LUNA has spooked more holders — so another \$300 million of UST gets redeemed. At \$30, that mints 300,000,000 / 30 = 10 million new LUNA, nearly three times as much as round 1. More LUNA dumped pushes the price to, say, \$5.
- **Round 3.** LUNA is \$5. The same \$300 million of redemptions now mints 300,000,000 / 5 = 60 million LUNA. The price craters to \$0.50.
- **Round 4.** LUNA is \$0.50. The same \$300 million mints 300,000,000 / 0.50 = 600 million LUNA — already more than LUNA's *entire original supply* of ~350 million, in a single round. Price falls to \$0.01.
- **Round 5.** LUNA is \$0.01. \$300 million of redemptions mints 300,000,000 / 0.01 = 30 billion LUNA. And it keeps going: at \$0.0001, the same redemption mints 3 trillion LUNA.

Notice the structure. Each round redeems a similar *dollar* amount of UST, but because LUNA's price is collapsing, each round mints exponentially more *units* of LUNA, and each fresh flood of units pushes the price down further, which makes the next round mint even more. The denominator is racing to zero, so the number of LUNA minted races to infinity.

The intuition: the death spiral is just division by a shrinking number — as LUNA's price falls toward zero, the LUNA minted per dollar of UST redeemed explodes toward infinity, which guarantees the price keeps falling. There is no equilibrium except zero.

#### Worked example: LUNA's supply exploding from 350 million to 6.5 trillion

Step back from the per-round arithmetic to the aggregate, which is almost hard to believe.

- Before the collapse, LUNA's circulating supply was roughly **350 million tokens**, trading near \$80, for a market cap on the order of \$28-30 billion.
- As the spiral ran and the protocol minted LUNA to honor every UST redemption at an ever-lower price, the supply ballooned. Within days it passed into the *trillions*.
- By the time the chain was halted, LUNA's supply had reached roughly **6.5 trillion tokens** — an increase of about **18,000x** from the starting 350 million.
- Multiply that out: 6.5 trillion tokens at a price of, say, \$0.0001 is only about \$650 million of total value — for a token that days earlier had been worth nearly \$30 billion. The supply went up ~18,000x while the value fell roughly 98%, because price collapsed far faster than supply could inflate into any meaningful market cap.

The intuition: hyperinflation is not a metaphor here — the protocol literally printed eighteen thousand times more LUNA in days, and printing money to defend a currency is the textbook way to destroy that money's value.

### Step three: the run becomes self-fulfilling

The arithmetic above explains the supply side. The human side is a classic bank run, sharpened by the on-chain transparency. Every holder could watch, in real time, UST sliding and LUNA's supply exploding. The rational move, once you believed the peg might not hold, was to get out *first* — redeem your UST before the LUNA you would receive became worthless, or just sell your UST on the market for whatever you could get. But everyone reasoning that way at once *is* the run. Each person fleeing pushed the price down, which confirmed everyone else's fear, which made more people flee. Reflexivity again: the belief that it would collapse was the thing that made it collapse.

#### Worked example: a 40 billion dollar market cap evaporating

Tie the threads together with the loss.

- At the April 2022 peak, LUNA's market cap was roughly \$28-30 billion and UST in circulation was roughly \$18 billion. Combined, the Terra ecosystem represented something like \$45-48 billion of nominal value; headline figures of "\$40 billion" or "\$60 billion" wiped out come from measuring this combined value at different dates.
- By mid-May, UST traded around \$0.10 and LUNA at fractions of a cent. The \$18 billion of UST was worth perhaps \$1-2 billion; the LUNA was worth a few hundred million. Combined value had fallen by well over 95%.
- The \$3 billion Bitcoin reserve held by the LFG was almost entirely spent or sold in the failed defense, contributing to a further leg down in Bitcoin's own price as it was dumped into a falling market.
- The loss was not paper for most holders. Retail savers who had parked real money in Anchor chasing 20% — including many who genuinely believed UST was a safe dollar account — lost most or all of it. The human cost was severe and widely reported, including suicides.

The intuition: "tens of billions evaporated" is not hyperbole — a near-\$50 billion combined market collapsed to low single-digit billions in under a week, and most of that value belonged to ordinary holders who had been told the dollar they held could not break.

### The two design sins, stated plainly

Strip away the detail and Terra failed for two linked reasons.

**No hard collateral floor.** A collateralized stablecoin has a redemption value that does not depend on confidence: USDC is (claimed to be) redeemable for \$1 of real reserves whether the market is calm or panicking. UST's "redemption" was into LUNA, an asset whose value *fell precisely when UST was under stress*, because the same stress was minting and dumping it. There was no floor — the backstop and the thing being backstopped were correlated to one, and that correlation went to its worst possible value exactly when it mattered.

**Demand manufactured by an unsustainable subsidy.** The 20% Anchor yield created demand that had no reason to exist once the subsidy wavered. Most UST holders were yield tourists, not people who needed a decentralized dollar. When the peg wobbled, there was no sticky, fundamental demand to lean against — just \$14 billion of hot money that had every reason to run at the first sign of trouble. A pile of capital attracted purely by an above-market yield is the most flighty capital there is.

Put together: an asset with no floor, held overwhelmingly by capital that would flee the moment the yield or the peg looked shaky, defended by a mechanism that turned every defense into more poison. The collapse was not a freak accident. It was the design meeting a bad enough week.

To feel the contrast viscerally, run the same panic through a *collateralized* coin. Suppose USDC slips to \$0.98 in a frightened market. A large holder buys USDC at \$0.98 and presents it to the issuer for redemption; the issuer hands back \$1.00 of real reserves — cash and Treasury bills sitting in a bank. The holder pockets two cents, and crucially the dollar they received did *not* come from minting a sister token whose price falls as more is minted. The redemption asset is fixed in value: a Treasury bill is worth what it is worth regardless of how scared the crypto market is. So the harder USDC de-pegs, the *more* profitable redemption becomes and the stronger the pull back to \$1 — the loop points uphill no matter how bad the panic gets, exactly the opposite of UST. (USDC's own March 2023 scare came from a *different* worry — that some of the reserves were trapped in a failing bank, i.e. that the floor itself might be shorter than claimed — not from a reflexive minting spiral. When the reserves turned out to be recoverable, the peg snapped back. A floor that is questioned is recoverable; a floor that does not exist is not.) The single structural difference — does redemption pull from a fixed asset or from a reflexive token — is the entire difference between a bounded scare and an unbounded collapse.

## The aftermath: contagion, charges, and a regulatory pivot

Terra did not die quietly in a corner. It was a load-bearing piece of the 2022 crypto market, and its collapse set off a chain of failures across the most leveraged corners of the industry.

![Contagion graph from Terra into Three Arrows, Celsius, and Voyager](/imgs/blogs/terra-luna-2022-collapse-7.png)

The contagion graph above traces the cascade. The most consequential casualty was **Three Arrows Capital (3AC)**, a large, aggressive crypto hedge fund that had built heavily leveraged positions and held a substantial stake in LUNA (reportedly a position once worth over \$200 million, plus exposure via a locked LUNA investment). When LUNA went to zero, 3AC was insolvent — but because it had *borrowed* heavily from crypto lenders to fund its bets, its collapse did not stay contained. 3AC defaulted on loans across the industry.

Those lenders were the next dominoes. **Voyager Digital** had lent 3AC a sum reported around \$650 million and could not absorb the default; it froze withdrawals and filed for bankruptcy. **Celsius Network**, a large crypto lender that had itself been paying eye-catching yields on customer deposits and had been entangled with Terra and other risky strategies, froze withdrawals on June 12 and filed for bankruptcy in July, trapping billions of dollars of customer funds. Other lenders and funds — Babel Finance and others — halted or wobbled in the same window. The full mechanics of how this lender-and-fund contagion propagated are their own case study, in [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion).

The reason a single fund's failure could topple multiple lenders is leverage layered on leverage, the same structure that turns a contained loss into a systemic one in traditional finance. Follow the chain of claims: retail depositors lent their savings to Celsius and Voyager chasing high yields; those lenders lent the pooled deposits to 3AC; 3AC put the borrowed money into leveraged bets including LUNA. When LUNA hit zero, the loss did not stay with 3AC — it ran *up* the chain. 3AC could not repay the lenders; the lenders could not repay the depositors. Each link had borrowed short (deposits redeemable on demand) to fund long, illiquid, risky positions, so each link was itself a bank run waiting for a trigger. Terra was the trigger. The depositors at the top of that chain — ordinary people who thought they had a high-yield savings account — were the ones left holding nothing, often discovering only when the withdrawal button stopped working that their "deposit" had been lent to a hedge fund that no longer existed.

The result was the **2022 crypto winter** — a prolonged, deep bear market in which prices fell hard and a string of major firms failed. Terra was the first big domino. It was not the last: later in 2022, the failure of the exchange FTX delivered a second, even larger shock, detailed in [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried). The two were not directly the same failure, but they shared a backdrop of leverage, opacity, and yields that were too good to be sustainable, and together they defined the worst stretch the industry had seen.

### Legal fallout and the regulatory pivot

For Do Kwon and Terraform Labs, the legal consequences were severe and protracted. The US Securities and Exchange Commission charged Terraform Labs and Kwon with orchestrating a multi-billion-dollar securities fraud, alleging among other things that they had misled investors about UST's stability and about how the peg was actually maintained during an earlier 2021 de-peg (which, the SEC alleged, had been quietly rescued by a third party rather than by the algorithm working as advertised). In 2024 a US jury found Terraform Labs and Kwon liable for fraud, and Terraform Labs agreed to a settlement on the order of billions of dollars. Kwon, who had left South Korea, became the subject of an international manhunt; he was eventually detained in Montenegro on charges including travel-document fraud and faced extradition fights between the US and South Korea. South Korea separately pursued criminal charges.

The wider effect was a hard pivot in how regulators around the world thought about stablecoins. Terra was a vivid demonstration that "stablecoin" was not a synonym for "safe," and that an algorithmic design with no reserves could vaporize tens of billions of dollars held by ordinary people. Regulators in the US, the EU (whose MiCA framework imposed strict reserve and redemption requirements on stablecoin issuers), and elsewhere moved toward rules that, in spirit, draw exactly the line this post draws: stablecoins should be *backed* by high-quality, redeemable reserves, and purely algorithmic designs with no asset floor are treated with deep suspicion or effectively disallowed at scale. The collapse did not end algorithmic stablecoins as an idea, but it ended the era in which one could attract \$18 billion with a 20% yield and little scrutiny.

The crux of the regulatory response is worth stating precisely, because it is the lesson the rules encode. A stablecoin that the public uses *as money* is, functionally, a bank: it takes in value and issues a redeemable, par-valued claim against it. Banks are heavily regulated for one reason above all — they are run-prone, and a run on something the public treats as money has consequences far beyond the issuer's own investors. The post-Terra rules therefore reach for the same tools that stabilize banks: high-quality liquid reserves held one-to-one against the claims, clear and enforceable redemption rights, disclosure of what the reserves actually are, and limits or bans on issuing a "money-like" claim with no real asset behind it. Whether or not you think those rules are correctly calibrated, their logic is a direct read of May 2022: the failure was a run on a money-like claim with no floor, so the fix is to require a floor and to make redemption real. Terra, in that sense, did for stablecoin regulation roughly what the bank failures of earlier eras did for deposit insurance and reserve requirements — it turned an abstract fragility into a politically unavoidable one.

## Common misconceptions

**"UST was a scam from day one."** This is the most tempting and the least useful framing. UST was not a Ponzi in the sense that Madoff was — there was a real, public, working mechanism, real engineers, and a design that genuinely held for over a year. The dangerous part was subtler: a sound-looking mechanism with a hidden fragility (no floor) plus a demand engine (Anchor's 20%) that was a subsidy dressed as a yield. Calling it "a scam" lets you off the hook from understanding *why* a clever, non-fraudulent-looking design could still be doomed. The SEC's fraud findings concern specific misrepresentations — notably about how the 2021 de-peg was actually fixed — not the bare fact that the algorithm existed. Both things are true: the mechanism was real, *and* investors were misled about how reliably it worked.

**"The algorithm broke."** It did not break. That is the unsettling lesson. The mint-burn mechanism executed flawlessly the entire way down. It was *designed* to mint \$1 of LUNA for every burned UST, and that is exactly what it did — including when doing so meant minting trillions of tokens. The failure was not a bug; it was the mechanism working as specified in a regime its designers assumed would never arrive. Most catastrophic financial failures are like this: the system does precisely what it was built to do, in conditions nobody planned for.

**"A bigger reserve would have saved it."** The \$3 billion Bitcoin reserve is often cited as "too small," with the implication that, say, \$10 billion would have held the line. This misunderstands the problem. The reserve was fighting a reflexive loop, and *spending* the reserve (dumping Bitcoin) actively worsened the broader sell-off that was driving people out of UST. More importantly, any reserve denominated in volatile crypto assets *falls in value precisely during a crypto-wide panic* — the exact moment you need it. A reserve helps only if it is large relative to the panic *and* stable relative to the thing it is defending. Terra's was neither. A bigger pile of correlated, volatile collateral does not fix a design whose backstop is correlated with the disaster.

**"20% APY is just a high-yield savings account."** No — and this is the single most important practical lesson. A savings account's yield comes from a bank lending your deposit out at a higher rate; it is income net of a spread, bounded by what borrowers will actually pay. Anchor's 20% was not bounded by borrower income — it was topped up from a reserve and from printing LUNA. A yield that exceeds what the underlying activity can generate, and is funded by drawing down a reserve or issuing more of a token, is a subsidy with a deadline. The size of the gap between an offered yield and the genuine risk-free rate is a rough measure of how much you should worry about where the money is really coming from.

**"Stablecoins in general are unsafe because of Terra."** Terra was an *algorithmic* stablecoin with no collateral floor. The dominant stablecoins today (USDC, USDT) are collateralized — they hold reserves you can, in principle, redeem the coin for. They have their own real risks (the quality and transparency of those reserves, custody, the issuer's solvency, regulatory action — see [the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar)), and one of them, USDC, briefly de-pegged in March 2023 when some of its reserves were stuck in the failing Silicon Valley Bank. But the *kind* of failure is different: a collateralized coin's worst case is a bounded reserve shortfall, not an unbounded death spiral. Lumping all stablecoins together obscures the one distinction — is there a redeemable hard floor or not — that mattered most in May 2022.

**"This could never happen again now that everyone knows."** Knowing the mechanism does not immunize the market against it, because the *driver* is human — the lure of an above-market yield and the belief that "this time the design is different." Reflexive, confidence-based assets keep being reinvented under new names. The specific UST design is now widely understood and regulated against, but the pattern (manufactured demand via unsustainable yield + an asset with no hard floor + reflexive defense) is a template, not a one-off, and variations of it continue to appear.

## How it echoes in other markets

Terra is not a crypto curiosity. The dynamics that killed it are old and recurring, and seeing them in other settings is the best way to internalize the pattern.

**Earlier algorithmic stablecoin failures.** UST was not the first algorithmic coin to die this way — it was the largest. Basis Cash, Empty Set Dollar, and others had already collapsed on smaller scales with the same flaw: a peg defended by a sister "share" or "bond" token that lost value exactly when the peg was stressed. Terra's defenders argued it was different because of scale, Anchor demand, and the Bitcoin reserve. It died of the identical disease. The recurrence is the lesson: the death spiral is intrinsic to *uncollateralized, sister-token* designs, not specific to one implementation.

**Classic bank runs.** Strip the crypto vocabulary and Terra is a bank run, the oldest financial failure there is. A bank holds illiquid assets against demandable deposits; if enough depositors demand at once, the bank cannot pay and fails, and the *fear* of others running makes running rational. The same self-fulfilling logic drove the runs on Northern Rock in 2007, on numerous banks in 1929, and on Silicon Valley Bank and Credit Suisse in 2023 ([the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) cover those). UST's twist was that its "defense" minted poison, but the trigger — too many claims demanded at once, accelerated by fear — is identical. The structural reason banks are run-prone in the first place comes straight from [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier): they fund long-dated, illiquid assets with short-dated, demandable liabilities.

**Ponzi-like yields.** Anchor's subsidized 20% is a cousin of every "guaranteed high return" that turned out to be paid from incoming money rather than genuine profit — most famously Madoff, whose fictitious steady returns were funded by new deposits until redemptions outran them. Anchor was not a deliberate fraud in the Madoff sense, but the *cash-flow shape* was similar: payouts exceeding what the underlying activity generated, funded by a finite pool that had to keep being refilled. Whenever a yield far exceeds the genuine risk-free rate and the explanation for *who pays it* is vague, the Madoff cash-flow shape is worth checking for.

**The contrast with collateralized money.** The clearest echo is the *positive* one. Tether and Circle's USDC survived 2022 (and USDC survived its own 2023 de-peg scare) precisely because, whatever their flaws, they held redeemable reserves — a floor. The same is true, at a different scale, of national currencies and the banking system, which rest on central banks that can act as a true lender of last resort. Terra had no lender of last resort and no floor, only an algorithm and a falling token. The contrast is the whole point: *backing* is not a bureaucratic nicety; it is the thing that turns an unbounded spiral into a bounded loss.

**Leverage and contagion in linked institutions.** Terra's spread into 3AC, Voyager, and Celsius rhymes with every leveraged-counterparty cascade in finance: LTCM in 1998, the interbank freeze of 2008, Archegos in 2021. The pattern is always the same — a leveraged player blows up, its lenders eat the loss, and because those lenders are themselves leveraged and interconnected, the loss propagates outward. The lesson Terra reinforces is that the *first* failure is rarely the whole story; what determines the damage is how much borrowed money and how many interlinked balance sheets sat on top of the thing that failed.

## When this matters to you, and further reading

You will probably never design a stablecoin. But the Terra collapse hands you a small, sturdy toolkit for evaluating *any* financial promise, in crypto or out.

**Ask what the floor is.** For anything that claims to hold a stable value — a stablecoin, a "capital-protected" product, a pegged currency — ask exactly what it can be redeemed for, by whom, and whether that redemption asset holds its value when the thing is under stress. "Stable because it is backed by redeemable reserves" and "stable because the market believes it is stable" are different universes of risk. Terra was the second kind and never admitted it clearly.

**Ask who pays the yield.** When a return is far above the risk-free rate, the question is not "how do I get in" but "who is funding this, and for how long." Trace it to a concrete, sustainable source — borrower interest, business cash flow, genuine risk premium — or treat the gap as the size of the risk you are taking. A yield funded by a draining reserve or by printing more of a token is a clock.

**Respect reflexivity.** The most dangerous financial structures are the ones where the act of losing confidence *causes* the loss — where selling makes the thing worth less, which makes more people sell. These structures look stable for a long time and then fail almost instantly, because there is no equilibrium between "fine" and "zero." If a system's defense mechanism gets *weaker* the more it is used (as Terra's did), that is the signature of a reflexive trap.

**Distrust certainty.** Do Kwon's public certainty was not a side note; it was a signal. A design that cannot tolerate questioning, whose proponents answer "what if confidence cracks" with mockery rather than mechanism, is a design that has not honestly modeled its own failure.

If you want to keep pulling these threads, the most useful next reads are the sibling pieces this post links to: [stablecoins and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) for how the collateralized coins that survived actually work and what risks they still carry; [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) for the cascade Terra set off; [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) for the second great shock of the 2022 winter; and, for the foundational money mechanics underneath all of it, [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier), which explains why demandable claims against illiquid backing are structurally run-prone whether they live in a bank or on a blockchain.

The durable takeaway is simple enough to carry around: a dollar that is stable only because everyone agrees it is stable is not a stable dollar — it is a confidence trade wearing a dollar's clothes, and confidence, unlike collateral, can vanish in an afternoon.
