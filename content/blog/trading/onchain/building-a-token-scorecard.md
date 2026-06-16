---
title: "Building a Token Scorecard: Turning On-Chain Signals Into One Number"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Pull the whole token-selection track into one weighted scorecard: hard gates, 0-5 factors, weights by token type, and a composite that maps straight to a position size."
tags: ["onchain", "crypto", "token-scorecard", "due-diligence", "position-sizing", "risk-management", "ethereum", "solana", "defi", "memecoins"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A token scorecard is a weighted rubric that turns a dozen scattered on-chain signals into one comparable number, and that number maps directly to a decision: avoid, small, standard, or conviction size.
>
> - The scorecard has three layers: **hard gates** (any one fails → instant reject, score irrelevant), **weighted factors** (safety, liquidity, distribution, unlocks, fundamentals, smart money — each scored 0-5 on on-chain evidence), and a **composite** (weights times scores, summed).
> - You read it by running the gates first, then grading each factor against explicit anchors, then weighting by token type — a memecoin weights liquidity and distribution, a DeFi protocol weights fundamentals.
> - What you DO with it: map the composite to a decision band and a dollar size. A clean 3.85/5 token gets a full standard position; a honeypot gets \$0 regardless of every other score.
> - The one rule to remember: **a scorecard reduces ruin and enforces discipline; it does not predict price.** Garbage in, garbage out — verify every input on-chain before you trust the number.

On 21 February 2025, a routine multisig transfer at Bybit turned into the largest crypto theft on record: roughly \$1.46B walked out of a cold wallet through a signing exploit, later attributed to the Lazarus Group. No scorecard would have caught that — it was an exchange custody failure, not a token you screen before buying. But the episode is the perfect frame for this post, because it makes the point that *the most expensive losses come from things you could have checked.* The signing flow was opaque; the human approving it could not see what they were really authorizing. A scorecard is the opposite philosophy applied to token selection: make the checks explicit, write them down, and force every buy decision through the same gates so nothing slips past on a feeling.

This is the capstone of the token-selection track. Across the series you have learned to read a contract for [rug-pull and honeypot risk](/blog/trading/onchain/rug-pull-and-honeypot-detection), to map [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration), to model [token unlocks, vesting and emissions](/blog/trading/onchain/token-unlocks-vesting-and-emissions), and to read [on-chain fundamentals — fees, revenue and TVL](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl). Each of those is a lens. The problem with lenses is that a human juggling six of them, under time pressure, with money on the line, makes inconsistent decisions — strict on Monday, lazy on Friday, generous toward a token a friend shilled. The scorecard fixes that. It is the machine that takes all those lenses and produces *one repeatable number you can compare across tokens and across days*.

![Token scorecard pipeline from raw signals through hard gates and weighted factors to a composite and a position size](/imgs/blogs/building-a-token-scorecard-1.png)

The figure above is the whole post in one frame. Raw on-chain signals enter on the left. They first hit the **hard gates** — a small set of vetoes that auto-reject the token if any one trips. Survivors flow into the **weighted factors**, where each of six dimensions is scored 0 to 5 against concrete on-chain criteria. Those scores get multiplied by **weights** that reflect what actually drives that token type, summed into a single **composite** out of 5.0, and that composite maps to a **decision and a dollar size**. By the end of this post you will build this scorecard from scratch, calibrate it to your own strategy, and run a worked token end-to-end with the actual arithmetic.

## Foundations: what a scorecard is and why it works

Start from zero. A **scorecard** is a weighted rubric — a structured way to convert many separate judgments into one comparable score. You meet scorecards everywhere outside crypto. A bank's credit-scoring model takes your income, debt, payment history and account age, weights each, and produces a single number (a FICO score) that maps to a loan decision and an interest rate. A restaurant inspector grades a kitchen on hygiene, temperature control, pest signs and documentation, weights them, and produces a letter grade. In both cases the value is not the precision of the final number — it is that *the same inputs always produce the same output*, so the decision is consistent and defensible rather than a mood.

A token scorecard does the same thing for an on-chain asset. The inputs are signals you can read on the blockchain: is the contract safe, is there real liquidity, is the supply concentrated, are there unlocks coming, is anyone actually using the protocol, is smart money accumulating or dumping. The output is a number that tells you whether to buy and how big. That is it. There is no magic — the entire skill is in choosing the right inputs, grading them honestly, and weighting them sensibly.

The deeper reason a scorecard beats ad-hoc judgment comes from how humans actually decide. Left to ourselves, we weight whatever signal is most *vivid* in the moment — the explosive green candle, the influencer we trust, the fear of missing out — and we under-weight the boring, decisive signals like "is the LP locked." A scorecard inverts that. By writing down the factors and weights *in advance*, when no money is on the line and no chart is screaming, you force your calm self to constrain your excited self. The score is a precommitment device as much as a measurement. That is why the discipline of *always running it*, even on the token you are sure about, matters more than the exact factor weights: the value is in never letting a buy skip the gauntlet.

There is one more structural property worth naming early. A scorecard is **monotonic and decomposable**. Monotonic means a better factor never lowers your score — improve any input and the composite can only rise or stay flat, which makes the tool predictable and trustworthy. Decomposable means that when the composite changes, you can point to exactly which factor moved it. If a token you held drops from a 3.8 to a 2.9, you can read off that the smart-money factor collapsed from 4 to 1 because a tagged insider wallet dumped into exchanges — the number does not just change, it *explains*. A gut feeling has neither property: it cannot tell you why it changed, so it cannot teach you anything. The scorecard, by construction, keeps a paper trail of its own reasoning.

### Why weighting matters: not all signals are equal

Here is the crucial idea that separates a real scorecard from a checklist. A naive checklist treats every item as equal — six green checkmarks, you buy. But the signals are not equal. A token can have deep liquidity, a wide holder base, real revenue, and smart-money inflows, and *still be a honeypot that you can buy but never sell.* If you average all six signals, the honeypot scores high — five great factors and one bad one averages to "pretty good," and you buy a contract that eats your money the moment you try to exit.

That is unacceptable. The fix is to recognize that some signals are **hard gates** and others are **weighted factors**. A hard gate is a disqualifier: a property so dangerous that its failure makes every other score irrelevant. A weighted factor is a graded dimension that contributes to the composite in proportion to how much it matters. Confusing the two is the single most common scorecard mistake, and it is the one that loses the most money.

### A disqualifier versus a scored factor

Make the distinction sharp, because the rest of the post rests on it.

A **disqualifier** (hard gate) answers a yes/no question whose "yes" is fatal. *Is this a honeypot?* If yes, reject — there is no score that redeems an asset you cannot sell. *Is the liquidity-pool token unlocked and held by the team?* If yes, reject — they can pull the pool any block and your position goes to zero with no warning. *Can the owner mint unlimited new supply or freeze your wallet?* If yes, reject. *Does one non-liquidity wallet hold more than 30% of supply?* If yes, reject — a single seller can erase the price. Hard gates are vetoes. They run first, and a single failure ends the process. The score is never computed.

A **scored factor** answers a question of degree. *How deep is the liquidity?* Not a yes/no — a token with \$2M of locked liquidity is better than one with \$80,000, which is better than one with \$8,000, and you want that gradient in your number. *How concentrated is the supply, short of the 30% veto?* Top-10 holders at 35% is worse than at 22%, which is worse than at 12%. Scored factors get a 0-to-5 grade and feed the weighted sum. They never single-handedly reject a token; they pull the composite up or down.

The mental shortcut: **gates protect you from ruin; factors optimize among survivors.** First you survive (gates), then you choose (factors).

### Calibration: your weights reflect your strategy

The last foundational idea is that there is no universal correct weighting. The factors are fixed — every serious token gets graded on the same six dimensions — but the *weights* are yours. They encode your strategy and your risk tolerance.

A short-term flipper who holds a memecoin for six hours does not care about revenue or token unlocks two years out; they care intensely about liquidity (can I get out?) and smart-money flow (is momentum behind me?). A fundamental holder who buys a DeFi protocol to hold for a year cares enormously about fees, revenue and dilution from unlocks, and weights short-term flow near zero. A risk-first defender — someone managing other people's money or simply allergic to drawdowns — weights safety and distribution heaviest and treats any borderline signal as a reason to size down. Same rubric, same chain data, three different composites, three different decisions. That is correct. The scorecard is a tool that makes *your* judgment consistent, not a verdict handed down from on high.

With those four ideas — scorecard, weighting, gate-versus-factor, calibration — you can build the thing. The next four sections are the four design layers: the hard gates, the weighted factors, the weights, and the composite-to-size mapping.

## Layer A: the hard gates, run first

The hard gates are the cheapest, highest-leverage part of the entire scorecard. They cost you sixty seconds of on-chain checks and they prevent the catastrophic, unrecoverable losses — the ones where you do not lose 40%, you lose 100%. Run them before you score anything, because if a gate trips, scoring is wasted effort.

![Hard gates as four independent vetoes that send a token straight to reject before any scoring](/imgs/blogs/building-a-token-scorecard-2.png)

The figure shows the structure: four gates, each an independent veto, all feeding a single REJECT outcome. The logic is OR, not AND — *any one* failure rejects. Only a token that clears *all* of them proceeds to weighted scoring. Let me define each gate and its on-chain check.

**Gate 1 — Honeypot.** A honeypot is a contract engineered so that buys succeed but sells revert or are taxed at near 100%. You put money in and you cannot take it out. The on-chain check: simulate a sell (tools like Honeypot.is, Token Sniffer, or a forked-node simulation run the buy-then-sell round trip), and read the transfer/tax functions in the verified source. If the sell path reverts, or the sell tax is set to 99%, or the contract has an owner-controlled blacklist that can flip your wallet's `canSell` to false, it is a honeypot. There is no factor score that makes a honeypot buyable. Reject.

**Gate 2 — Liquidity pull risk.** The liquidity pool (LP) is what lets you trade. If the LP tokens are *unlocked* and held in a wallet the team controls, the team can withdraw the entire pool in one transaction — the classic rug pull — leaving the token untradeable and your position worth zero. The on-chain check: find the LP token, check whether it is locked (in a time-locked locker contract like Unicrypt or Team Finance) or burned (sent to a dead address), and for how long. Unlocked, team-held LP with no lock is a hard gate failure. Locked-but-expiring-soon is a soft flag that hurts the liquidity factor score, not an automatic reject.

**Gate 3 — Mint and freeze powers.** Read the contract for owner powers. Can the owner mint new tokens (inflating your stake to nothing)? Can the owner pause transfers or freeze specific wallets? Is the contract an upgradeable proxy where the logic can be swapped for malicious code after you buy? On Ethereum you read this in the verified source and the proxy admin; on Solana you check whether mint authority and freeze authority are renounced (set to null). If mint authority is live and held by a fresh wallet, that is a gate failure for a memecoin. For a serious protocol, an upgradeable contract is normal — but then it must be governed by a timelock and a multisig, which downgrades it to a factor consideration rather than a hard reject. Context matters; the gate is "uncontrolled mint/freeze power in a single hand."

**Gate 4 — Extreme concentration.** If one wallet that is not the liquidity pool, not a known locked vesting contract, and not a CEX holds more than ~30% of circulating supply, a single seller can crater the price before you react. The on-chain check: pull the holder list (Etherscan/Solscan token holders tab, or a Bubblemaps cluster view), exclude the LP and known locked contracts, and look at the largest free-floating holder. Above your threshold (30% is a common line; tighten to 15-20% for thin memecoins), it is a gate failure. This is where you also catch the case where five "different" wallets are one entity funded from a single source — clustering turns five 8% holders into one 40% whale.

#### Worked example: a token failing the honeypot gate

Say a new token shows a beautiful chart: \$4M in liquidity, 8,000 holders, a top-10 concentration of only 18%, and three wallets tagged "smart money" buying. On the weighted factors alone it might score 4.5/5 — it looks like a conviction buy. You run a sell simulation and the transaction reverts with `TRANSFER_FAILED`. Reading the source, the `_transfer` function checks a `_isSellAllowed` mapping that only the owner can set, and it is false for every address except the deployer. That is a honeypot. The factor score of 4.5 is irrelevant. Your position size is \$0, not the \$5,000 you were about to deploy. The gate just saved you \$5,000 — and the lesson is that the gates run *first* precisely so a gorgeous factor profile never seduces you past a fatal flaw.

The hard gates are binary by design. They are deliberately conservative — better to wrongly reject a few legitimate tokens than to ever buy a single honeypot. You will pass on some winners because of an overcautious gate. That is the correct trade: the downside of a missed winner is an opportunity cost; the downside of a tripped gate you ignored is a 100% loss. For the full mechanics of these checks, lean on the dedicated posts — the [rug-pull and honeypot detection](/blog/trading/onchain/rug-pull-and-honeypot-detection) walkthrough and the [supply-distribution and holder-concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) deep-dive — this post assumes you can perform them and focuses on wiring them into one number.

## Layer B: the weighted factors, scored 0 to 5

A token that clears all four gates is not "safe" — it is merely *not obviously fatal*. Now you score it. Six factors, each graded 0 (worst) to 5 (best) against explicit on-chain anchors. The anchors are what make the score reproducible: if you and I grade the same token's distribution, we should land within a point of each other, because the rubric tells us what a 0, a 3, and a 5 look like.

![Weighted-factor rubric matrix with each factor anchored at scores zero, three and five](/imgs/blogs/building-a-token-scorecard-3.png)

The matrix above is the rubric. Read it as: for each factor, here is what the worst case (0), the passable case (3), and the best case (5) look like in on-chain terms. The 1, 2 and 4 grades interpolate. Let me walk through each factor, what it measures, and the evidence you read.

### Factor 1 — Safety (beyond the gates)

The honeypot/mint/freeze gates caught the fatal contract flaws. The safety *factor* grades what remains: is the contract verified and readable, has the deployer renounced ownership, is it a battle-tested standard or a bespoke fork, has it been audited, are there admin functions that are uncomfortable but not fatal (a 5% fee the owner can change, a pausable transfer governed by a timelock)? A 0 is an unverified contract with live owner powers; a 3 is verified with mint renounced; a 5 is an audited, immutable contract with no admin keys at all. You read this in the verified source and the proxy/admin slots.

### Factor 2 — Liquidity depth

Liquidity is *can you actually get out, and at what cost*. The gate caught the rug-able pool. The factor grades depth relative to your intended position size. The number that matters is **price impact**: if you sell your position, how far does it move the price? A token where exiting \$5,000 moves the price 20% has thin liquidity (score 0-1); one where \$5,000 moves it 1% has deep liquidity (score 5). You read this from the pool reserves and the AMM curve, or simply by simulating a sell of your size on the DEX and reading the slippage. Locked-and-deep scores high; locked-but-shallow scores middling; expiring-lock is a haircut.

#### Worked example: a memecoin with high liquidity but a 1/5 distribution

A Solana memecoin clears all four gates — sells work, LP is burned, mint authority is renounced, no single wallet exceeds 30%. Its liquidity is genuinely deep: a \$2,000 sell moves the price under 2%, so liquidity scores 4/5. But when you cluster the holders, you find that the top 20 wallets — all funded within ten minutes from the same source two days before launch — collectively hold 55% of the float across "different" addresses. No single address trips the 30% gate, but the *cluster* is one entity. Distribution scores 1/5. This is the textbook "good liquidity, terrible distribution" profile: it trades cleanly right up until the insider cluster decides to exit, and then the deep liquidity you admired just means they can dump \$1.2M into your bids efficiently. The composite lands in the small/lotto band, so the correct action is a \$500 lotto-size position you are fully prepared to lose, not the \$5,000 the chart's liquidity tempted you toward. The deep liquidity is real; it is just liquidity that works for the insiders' exit as much as your entry.

### Factor 3 — Distribution

Distribution grades how spread the ownership is, below the gate threshold. Top-10 holders at 12% with thousands of holders is a 5; at 30-40% with a handful of clusters is a 3; concentrated in fresh, single-funder clusters is a 0-1. The richest version of this factor uses clustering ([address clustering and heuristics](/blog/trading/onchain/supply-distribution-and-holder-concentration) covers the technique): you collapse wallets that share a funding source or behavioral fingerprint into entities before measuring concentration, because raw holder counts are trivially gamed by splitting one whale across fifty wallets.

### Factor 4 — Unlock overhang

This factor grades future supply pressure. Tokens with venture-capital and team allocations on vesting schedules have *known future sellers*. The metric is the relationship between fully-diluted valuation (FDV) and circulating market cap, plus the timing of cliffs. A token where 80% of supply is already circulating and emissions are low scores 5; one where the circulating float is 15% of supply and a large cliff unlocks next month scores 0-1, because a wave of cost-basis-near-zero VC tokens is about to hit the market. You read the vesting contracts and the emission schedule. The [token unlocks, vesting and emissions](/blog/trading/onchain/token-unlocks-vesting-and-emissions) post is the deep mechanic; here it is one 0-5 input.

### Factor 5 — Fundamentals (usage)

Does anyone actually use this thing? For a DeFi protocol, this is the most important factor: real fees paid by real users, revenue accruing to the protocol or token, and TVL that is *sticky* rather than mercenary (yield-farming capital that flees the moment incentives stop). A protocol earning \$2M/month in fees from organic usage scores 5; one with high TVL but near-zero fees (incentive-farmed) scores 2-3; a token with no usage at all scores 0. For a pure memecoin, fundamentals are mostly inapplicable — which is exactly why the *weights* differ by token type, the subject of the next section. You read fees and revenue from protocol dashboards on [Dune or Token Terminal, grounded in the on-chain fundamentals](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl) post.

### Factor 6 — Smart money and momentum

Finally, who is on the other side of your trade? Are wallets with a strong track record accumulating, or are early insiders distributing into the rally? Is exchange flow net-inflow (supply arriving to sell) or net-outflow (coins leaving to cold storage)? Strong, credible accumulation with no distribution scores 5; a few tagged smart wallets present scores 3; insider-funded with no credible holders and net inflow to exchanges scores 0-1. This factor is the noisiest and the most prone to survivorship bias in "smart money" labels — weight it accordingly. The base technique lives in the [following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) and exchange-flow material.

A caution that earns this factor its low weight in most templates: "smart money" labels are reconstructed *after* the fact, so they suffer brutal survivorship bias. A wallet is tagged smart because its past trades happened to win; that says little about whether its next trade wins, and nothing about whether it is positioned the same way you are (a smart wallet might be hedged, market-making, or front-running an unlock you have not spotted). Treat the factor as a tie-breaker and a sentiment read, never as the thesis. A token that scores well on safety, liquidity, distribution and fundamentals but only middling on smart money is still a strong token; a token carried entirely by a "smart money is buying" badge with weak everything else is a trap dressed up as a signal. The factor adds the most value as a *negative* read — credible early holders quietly distributing into a pump is a louder warning than their accumulation is a green light.

The output of Layer B is six numbers, each 0-5. On their own they are still just a list — the next layer collapses them into one.

## Layer C: the weights, set by token type

Six factor scores do not become one number until you decide how much each matters. That is the weighting step, and it is where the scorecard stops being generic and starts reflecting *what actually drives a given token*. The factor list is fixed; the weights move.

![Memecoin versus DeFi protocol weighting of the same six factors](/imgs/blogs/building-a-token-scorecard-4.png)

The before/after figure makes the contrast concrete. The *same six factors* get radically different weights depending on token type, because different things kill different tokens.

A **memecoin** has no revenue to model, no real fundamentals, and a holding period measured in hours or days. What kills a memecoin is thin liquidity (you cannot exit) and an insider cluster dumping (distribution collapse). So a memecoin weighting might be: liquidity 25%, distribution 30%, smart money/momentum 25%, safety 15%, unlocks 0%, fundamentals 5%. The weights live where the risk lives. Putting 30% weight on fundamentals for a dog-themed token would be nonsense — there are no fundamentals to weigh.

A **DeFi protocol** held for months is the opposite. What drives its value is fees, revenue and sticky TVL (fundamentals); what threatens it is VC unlock dilution and a contract holding hundreds of millions that could be exploited or rug-governed. So a protocol weighting might be: fundamentals 30%, unlock overhang 20%, safety 20%, distribution 15%, liquidity 10%, smart money 5%. Here liquidity is a minor factor — a serious protocol token trades on deep CEX and DEX markets — and fundamentals dominate.

The discipline is that weights must sum to 100% (or 1.0), so increasing one factor's weight necessarily decreases others. You cannot weight everything heavily; you are forced to declare what matters most for *this* token type. Write your weight templates down in advance — one for memecoins, one for protocol/governance tokens, one for infrastructure/L1 tokens — and apply the matching template rather than re-deciding weights in the heat of a hot launch, which is exactly when you would cheat them toward "buy."

#### Worked example: comparing two \$50M tokens scoring 4.2 versus 2.1

Two tokens both sit at a \$50M market cap and both pass the hard gates. Token A is a DeFi protocol; under the protocol weights it scores fundamentals 5, unlocks 4, safety 4, distribution 4, liquidity 4, smart money 3 → composite 0.30×5 + 0.20×4 + 0.20×4 + 0.15×4 + 0.10×4 + 0.05×3 = 1.50 + 0.80 + 0.80 + 0.60 + 0.40 + 0.15 = 4.25, call it 4.2. Token B is a hyped governance token with a big cliff next month and no revenue: fundamentals 1, unlocks 1, safety 3, distribution 3, liquidity 4, smart money 3 → 0.30×1 + 0.20×1 + 0.20×3 + 0.15×3 + 0.10×4 + 0.05×3 = 0.30 + 0.20 + 0.60 + 0.45 + 0.40 + 0.15 = 2.10. Same market cap, but the scorecard says deploy a full \$5,000 standard position into A (composite 4.2, conviction-adjacent) and \$0 into B (composite 2.1, avoid). The market-cap number told you nothing; the composite told you everything. That is the entire value of running both tokens through the identical, weighted rubric — the comparison is finally apples to apples.

## Layer D: the composite and the decision bands

Now the arithmetic. The composite is a weighted average:

```
composite = sum over factors of (factor_score * factor_weight)
```

Because weights sum to 1.0 and scores range 0-5, the composite also ranges 0-5. A composite of 0 means everything is terrible; 5 means everything is perfect (you will essentially never see a 5). Most real, gate-passing tokens land between 2.0 and 4.5.

Why a weighted *average* rather than some other combiner? It is worth understanding the choice, because the combiner encodes a philosophy. A weighted average says "factors trade off against each other" — a strong fundamentals score can compensate for a mediocre liquidity score, which is reasonable once you are past the gates, because among non-fatal tokens you genuinely are willing to accept some weakness in exchange for strength elsewhere. Two alternatives are worth dismissing. A *minimum* combiner (the composite is your worst factor) is too harsh for the factor layer — it would reject a great protocol for one merely-okay factor, and that is the gates' job, not the factors'. A *product* combiner (multiply normalized factors) collapses to near-zero if any single factor is weak, which again duplicates the gate behavior. The clean design is: vetoes handle the all-or-nothing failures (a minimum-like logic, but binary), and the weighted average handles the smooth trade-offs among survivors. Keeping those two logics separate — vetoes for ruin, averaging for optimization — is the structural insight that makes the whole scorecard coherent.

The composite alone is just a number. It becomes a decision when you map it to **position-size bands**.

![Composite score mapped to four position-size bands from avoid to conviction](/imgs/blogs/building-a-token-scorecard-6.png)

The bands convert the composite into a fraction of your standard position unit. Define a "unit" once — say one unit is \$5,000, the amount you would normally risk on a clean, average opportunity — and then:

- **Below 2.5 → Avoid.** Position \$0. Too many soft flags; the expected value is not worth the risk and the attention. Walk away.
- **2.5 to 3.4 → Small / lotto.** 10-25% of a unit (\$500 to \$1,250 on a \$5,000 unit). A speculative position you are fully prepared to lose, sized so that a total loss is irrelevant to your portfolio.
- **3.5 to 4.2 → Standard.** 100% of a unit (\$5,000). A clean token with normal risk gets your normal position.
- **Above 4.2 → Conviction.** 150-200% of a unit (\$7,500 to \$10,000). Rare. Everything lines up — strong fundamentals, deep liquidity, clean distribution, smart money in, no overhang. Size up, but carefully, and never so far that one position can hurt the book.

The bands are yours to set, like the weights. A defensive investor might cap conviction at 100% (never oversize) and start "avoid" at 3.0. The point is to *pre-commit the mapping* so the score mechanically produces a size, removing the in-the-moment temptation to oversize a token you have fallen in love with.

#### Worked example: a clean token scoring 3.85 → a standard \$5,000 position

Take a token that clears all four hard gates and scores: safety 4, liquidity 4, distribution 4, unlocks 3, fundamentals 4, smart money 4, under a balanced weighting of 0.20 / 0.20 / 0.20 / 0.15 / 0.15 / 0.10. The contributions are 0.80, 0.80, 0.80, 0.45, 0.60, 0.40, summing to a composite of 3.85 out of 5.0. That lands in the 3.5-4.2 standard band, so the position is one full unit — \$5,000. Not \$2,000 because you are nervous, not \$12,000 because the chart looks explosive: \$5,000, because that is what a 3.85 maps to under the rules you set when you were calm. The scorecard's job here is not to find you a 10x; it is to make sure that when you do deploy \$5,000, it is into something that earned it on the chain, and that you deployed the *right amount* rather than a number pulled from your gut.

![Worked scorecard grid showing six factor scores times weights summing to a 3.85 composite](/imgs/blogs/building-a-token-scorecard-5.png)

The grid above is that exact worked scorecard, laid out cell by cell so you can see how six judgments become 3.85. This is the artifact you should produce for every serious token — a filled grid you can save, revisit when the position moves, and compare against the next candidate. Notice the bottom row is just the sum; there is no hidden step, no black box. Everything that produced the decision is visible and checkable, which is precisely what you want when the position goes against you and you need to ask "did the thesis break, or did the price just wobble?"

## How the scorecard really behaves: edge cases and how it deceives

A scorecard is a model, and every model has failure modes. The honest version of this post has to cover the ways the scorecard misleads, because a tool you trust blindly is more dangerous than no tool at all — it launders a bad decision in the costume of rigor. Three failure modes matter most.

**Garbage in, garbage out.** The composite is only as good as the factor scores, and the factor scores are only as good as the on-chain verification behind them. The most common way to ruin a scorecard is to grade factors from a dashboard's green number rather than from the chain. A dashboard might report "liquidity: \$2M" while most of that pool is single-sided team liquidity that vanishes on the first sell; a "smart money: 12 wallets in" badge might be twelve wallets a marketing team funded to fake the label. If you score liquidity 4 and smart money 4 off those badges without verifying, the composite says standard and you deploy \$5,000 into a token whose true scores were 1 and 0. The model did not fail; your inputs did. The fix is a rule: *no factor is scored above 2 without first-hand on-chain evidence you pulled yourself* — the simulated sell, the clustered holder list, the fee transactions. The scorecard cannot verify for you; it can only combine what you verified.

**The borderline-band trap.** A composite of 3.49 maps to small and 3.51 maps to standard, a 3x difference in dollar size from a 0.02 difference in a number built on coarse 0-5 judgments. Hard band edges create false precision exactly where the data is least precise. Two defenses: first, round honestly to one decimal and accept that anything within ~0.2 of a band edge is genuinely ambiguous — when in doubt at a boundary, size to the *lower* band, because the asymmetry of crypto losses rewards caution. Second, if you find yourself agonizing over whether a token is a 3.4 or a 3.6, that agony is itself information: the token is mediocre, and a mediocre token deserves a small size regardless of which side of the line it lands. The band trap only hurts you if you let a rounding artifact talk you into oversizing.

**Stale scores.** A scorecard is a snapshot, and tokens change. An LP lock expires; a vesting cliff arrives; the smart money that was accumulating starts distributing; a protocol's fees collapse when an incentive program ends. A 4.0 you bought three months ago may be a 2.5 today, and if you never re-score, you are holding on a thesis the chain has already invalidated. The discipline is to re-run the gates and re-score the factors on a schedule (or on an alert) for every open position, not just at entry. The gates especially: a token that passed the LP-lock gate at entry can *fail* it the day the lock expires, and that is a now-or-never exit signal, not a "let it ride" moment.

#### Worked example: a held position whose unlock factor decays

You bought \$5,000 of a protocol token at a composite of 3.8 (standard band) when 60% of supply was circulating and the next VC cliff was six months out. Five months later, the alert fires: the cliff unlocks ~12% of supply next week, held by early investors whose cost basis is a fraction of the current price — textbook sell pressure. You re-score: the unlock-overhang factor drops from 3 to 1, and because it carries a 0.20 weight in your protocol template, the composite falls by 0.20 × (3−1) = 0.40, from 3.8 to 3.4 — out of the standard band and into small. The rule mechanically tells you to cut the \$5,000 position down toward a \$1,250 small-band size ahead of the unlock, locking in gains before the supply hits. Nothing about the price changed to trigger this; a *factor* changed, the decomposable score caught it, and the size adjusted. That is the scorecard working as a living instrument rather than a one-time gate at the door.

## How to read it: building and running the scorecard end-to-end

Theory is cheap. Let me run a complete token through the scorecard the way you actually would, with the on-chain evidence at each step. The token is invented (call it **Wallet/contract `0xT0ken…`**, an illustrative DeFi governance token at a \$50M market cap) so that no real ticker is implied, but the workflow is exactly what you would do on Etherscan, a DEX, a holder explorer, and a fundamentals dashboard.

**Step 0 — Pick the weight template.** It is a DeFi protocol token held for months, so I load the protocol weights: fundamentals 0.30, unlocks 0.20, safety 0.20, distribution 0.15, liquidity 0.10, smart money 0.05. Sum = 1.00.

**Step 1 — Run the four hard gates.**
- *Honeypot?* Simulate a sell on the DEX; it succeeds with a 0.3% fee. Pass.
- *LP pull risk?* The DEX LP is partly protocol-owned liquidity locked for two years and partly third-party; no single team wallet can pull it. Pass.
- *Mint/freeze?* The contract is an upgradeable proxy — but the admin is a 5-of-9 multisig behind a 48-hour timelock, and there is no arbitrary mint. Not a fatal gate; it becomes a safety-factor consideration. Pass (with a note).
- *Concentration?* Largest free-floating non-LP, non-vesting wallet holds 7% of circulating supply. Below 30%. Pass.

All four gates clear. Proceed to scoring.

**Step 2 — Score the six factors against on-chain evidence.**
- *Safety = 4.* Verified source, audited twice, upgradeable but timelock + multisig governed, no mint. The upgradeability is the only reason it is not a 5.
- *Liquidity = 4.* Selling a \$5,000 position moves the price under 1.5% across CEX+DEX depth.
- *Distribution = 4.* Top-10 free-floating holders at 22% after clustering; thousands of holders; no single-funder cluster.
- *Unlocks = 3.* 60% of supply circulating; a linear VC vest releases ~3% of supply per quarter — real overhang, but gradual, not a cliff.
- *Fundamentals = 4.* The protocol earns roughly \$1.8M/month in fees with growing active addresses and TVL that survived the last incentive cut, which is the signature of sticky rather than mercenary capital.
- *Smart money = 3.* A handful of credibly-tagged wallets accumulating; no large insider distribution to exchanges; nothing dramatic.

**Step 3 — Weight and sum.**

```
fundamentals:  4 * 0.30 = 1.20
unlocks:       3 * 0.20 = 0.60
safety:        4 * 0.20 = 0.80
distribution:  4 * 0.15 = 0.60
liquidity:     4 * 0.10 = 0.40
smart money:   3 * 0.05 = 0.15
                          ----
composite             =   3.75
```

**Step 4 — Map to a size.** A composite of 3.75 lands in the 3.5-4.2 standard band. With a \$5,000 unit, the position is \$5,000. Done.

#### Worked example: how the same token under flipper weights changes the size

Run the identical factor scores through a short-term flipper's weights — liquidity 0.30, smart money 0.25, distribution 0.20, safety 0.15, unlocks 0.05, fundamentals 0.05 — and the composite shifts: 4×0.30 + 3×0.25 + 4×0.20 + 4×0.15 + 3×0.05 + 4×0.05 = 1.20 + 0.75 + 0.80 + 0.60 + 0.15 + 0.20 = 3.70. Almost the same number here, because this particular token happens to be strong across the board. But on a token with great fundamentals and mediocre momentum, the flipper's composite would drop into the small band while the holder's stayed standard — same chain data, different decision, because the flipper's edge does not pay for fundamentals they will never hold long enough to realize. If a flipper buying this token had \$10,000 of intended capital and the composite said standard (one \$5,000 unit), they deploy \$5,000 and keep \$5,000 dry — the scorecard sized the bet, the conviction band did not get reached, so no oversizing. The lesson: the scorecard is not just a filter; it is the thing that stops you from putting your whole \$10,000 into one merely-good token because it "feels" like a winner.

Save the filled grid the moment you act on it — a row per factor with the score, the weight, the contribution, the on-chain evidence you cited, the composite, the band, and the dollar size. This artifact is doing three jobs at once. At entry it is your decision record. While you hold, it is the baseline you re-score against, so you can tell a thesis-break from a price-wobble. And after you close the position — win or lose — it is a labeled data point you can study to calibrate your own weights, the legible track record the playbook section returns to. A scorecard you compute and discard captures none of that compounding value; the grid is the deliverable, not the number.

That is the entire loop: pick weights → gates → score → weight → sum → size. Fifteen minutes per token once you are practiced, and the output is a saved, comparable artifact rather than a fading memory of "yeah I looked at it, seemed fine." Wire it into your daily routine so that candidates arrive at the scorecard already pre-filtered — the scorecard sits downstream of your screening and alerting, taking the handful of tokens that survived your scans and putting each one through the identical gauntlet before any capital moves.

## Why the scorecard exists at all: the base rate

Step back and ask why any of this is worth the effort. The answer is the base rate.

![Base rate showing roughly 98 percent of launched tokens go to zero versus the small share that survive](/imgs/blogs/building-a-token-scorecard-8.png)

On permissionless launchpads, of the millions of tokens ever created, only on the order of **1-2% ever reach a meaningful market cap** — the rest are rugs, honeypots, or simply die from neglect. The base rate for a random low-cap token is "goes to approximately zero." That is the brutal context every buy decision lives inside.

A scorecard does not predict which of the survivors will moon — nothing does. What it does is *shift the distribution your capital meets*. If random selection has a ~98% loss rate, then a disciplined scorecard that rejects every honeypot, every rug-able pool, and every insider-cluster dump removes a huge chunk of that 98% from your funnel. You are not finding winners; you are systematically not buying the losers that you could have detected. Over hundreds of decisions, that edge compounds: the trader who deploys \$5,000 into ten gate-passing, factor-scored tokens has a wildly better expected outcome than one who deploys the same \$50,000 across ten chart-chasing buys, even though neither can predict the next 10x. The scorecard's payoff is in the losses it prevents, and against a 98% base loss rate, prevention is the whole game.

Run the arithmetic of prevention to feel it. Suppose without screening, your average low-cap buy has a 5% chance of a 5x and a 95% chance of going to roughly zero — expected value per \$5,000 buy is 0.05 × \$25,000 + 0.95 × \$0 = \$1,250, a brutal 75% expected loss, which is why undisciplined low-cap trading is a slow bleed even when the occasional winner feels great. Now suppose the scorecard does nothing magical — it does not improve the winners at all — but it filters out the honeypots and rugs, which were maybe half of your zeros. Among the tokens you now buy, the 5x odds rise to ~10% simply because the worst outcomes were removed from the pool: expected value becomes 0.10 × \$25,000 + 0.90 × \$0 = \$2,500 per \$5,000 buy. You have not gained foresight; you have removed the part of the loss distribution you could see coming, and the expected value doubled. That is the entire mechanism — not prophecy, just refusing to step on the rakes you can identify on the chain.

A subtler benefit: the scorecard makes your *track record legible*. Because every buy is a saved grid with a composite and a band, you can later sort your wins and losses by entry composite and ask "do my 4.0-plus buys actually outperform my 3.0 buys?" If they do not, your factors or weights are miscalibrated and you have the data to fix them. A trader with only gut decisions has no such feedback loop — they cannot learn faster than the market punishes them. The scorecard turns each decision into a labeled training example for your future self.

## Common misconceptions

**"A high composite predicts the token will go up."** No. The composite measures *quality and risk of the asset as read on-chain*, not future price. A 4.2 token can still fall 60% in a market-wide drawdown, and a 2.5 token can 5x on pure hype. The scorecard improves your *odds and your sizing*; it is not a crystal ball. Treating the composite as a price forecast is the fastest way to misuse it — it is a risk dial, not an oracle.

**"More factors and more decimal places make it more accurate."** No — this is false precision, and it is dangerous. A composite of 3.847291 is not more meaningful than "3.8, standard band." The inputs are coarse human judgments graded 0-5 against fuzzy on-chain evidence; carrying six decimals implies a precision the data does not have. Round to one decimal, map to a band, and resist the urge to add a fourteenth factor that splits hairs. The value is in the *consistency* of a few well-chosen factors, not the granularity.

**"If I just average all the signals I do not need hard gates."** This is the most expensive misconception, and the whole post exists to kill it. Averaging lets five great factors drown out one fatal flaw — a honeypot averages to "good." Hard gates are non-negotiable vetoes precisely because the failure modes they catch are unrecoverable. Without gates, your scorecard will eventually score a honeypot a 4.5 and tell you to deploy \$5,000 into a contract you can never sell. Gates first, always.

**"The score is objective, so my judgment does not matter."** The opposite is true. Every factor score is *your* read of the on-chain evidence, and every weight is *your* strategy. The scorecard makes your judgment consistent and explicit; it does not replace it. Garbage in, garbage out: if you grade liquidity a 4 without actually simulating a sell, the composite is worthless. The discipline is in honestly grading inputs you verified on-chain, not in trusting a number you produced lazily.

**"One scorecard works for every token."** No — a memecoin and a DeFi protocol need different weights, as Layer C showed. Applying protocol weights (30% fundamentals) to a memecoin with no fundamentals produces a meaningless number. Keep a small set of weight templates by token type and use the right one. The factors are universal; the weights are not.

**"A passing scorecard means I can stop watching the position."** The opposite — a pass is the *start* of monitoring, not the end. Tokens decay: locks expire, cliffs unlock, smart money rotates out, fees collapse. The score that earned your \$5,000 at entry is a snapshot, and a snapshot of a moving target goes stale. The scorecard's job continues after the buy: re-run the gates and re-score the factors on a schedule or on alert, and let the band mechanically tell you to trim or exit when a factor decays. A scorecard run once and forgotten is a seatbelt you unbuckle after starting the car.

**"The composite tells me my profit target or stop-loss."** No — the composite is about the *asset's quality and risk*, not about price levels. It tells you whether to be in and how big, not where to take profit or cut. Your exit rules (a trailing stop, a fundamental-break rule, a re-score trigger) are a separate layer of your plan. Conflating "this is a 4.0 token" with "this token will reach price X" is how a sound risk tool gets misused as a price oracle, which it is not and was never built to be.

## The playbook: what to do with it

Here is the if-then checklist that turns this post into a repeatable practice. For every token you are seriously considering:

- **Signal: a candidate token clears your screen.** → **Read:** before anything else, run the four hard gates — honeypot, LP pull risk, mint/freeze power, extreme concentration. → **Action:** any gate fails → reject, position \$0, stop. No gate fails → proceed. → **False positive:** an upgradeable protocol contract is not a gate failure if it is timelock+multisig governed — downgrade it to a safety-factor consideration, do not auto-reject a legitimate protocol.

- **Signal: token passed all gates.** → **Read:** load the weight template matching the token type (memecoin vs protocol vs infra), then score each of the six factors 0-5 against the rubric anchors, using actual on-chain evidence you verified (simulate the sell, cluster the holders, read the fee dashboard). → **Action:** multiply each score by its weight, sum to a composite. → **Invalidation:** if you scored any factor without verifying it on-chain, the composite is garbage — go back and verify.

- **Signal: you have a composite score.** → **Read:** map it to a band — below 2.5 avoid, 2.5-3.4 small/lotto, 3.5-4.2 standard, above 4.2 conviction. → **Action:** deploy the corresponding fraction of your unit in dollars (\$0 / 10-25% / 100% / 150-200%). Save the filled scorecard grid. → **False positive:** do not override the band because the chart looks exciting; the band is the rule you set when calm.

- **Signal: the position moves (up or down) after you buy.** → **Read:** pull up the saved scorecard and ask whether any *factor* has changed — did a cliff unlock, did the smart money exit, did distribution concentrate? → **Action:** if a factor materially worsened (or a gate now trips — e.g. the LP lock expired), re-score and re-size, including cutting to \$0. If only the price moved but the on-chain factors are intact, hold the thesis. → **Invalidation:** a price drop with no factor change is noise, not a signal to panic-sell; a factor change with no price move yet is exactly when to act.

- **Signal: you are comparing several candidates.** → **Read:** run all of them through the *identical* rubric and weights for their type. → **Action:** size by composite, not by hype or market cap — the higher composite gets the larger position, mechanically. → **False positive:** two tokens at the same market cap can have wildly different composites; trust the composite, not the cap.

The deepest value is not any single decision — it is the *discipline*. A scorecard you actually run on every token, honestly, makes you consistent across the days when you are sharp and the days when you are tired or greedy. It reduces your ruin rate by gating out the catastrophes, and it removes the position-sizing decision from your emotions by mapping a number to a dollar amount. It will not make you a genius stock-picker of tokens; nothing will. It will make you the disciplined survivor who is still in the game after the 98% have gone to zero — and in a market with that base rate, survival *is* the edge.

![Calibration matrix showing how a flipper, a holder and a defender weight the same factors differently](/imgs/blogs/building-a-token-scorecard-7.png)

The calibration matrix above is the last word: there is no universal scorecard, only your scorecard. A short-term flipper weights liquidity and flow and rejects on any liquidity red flag; a fundamental holder weights revenue and dilution and rejects on fake fundamentals; a risk-first defender weights safety and distribution and rejects on any single hard-gate trip. Pick the column that matches who you actually are, write your weights and bands down, and run the same machine on every token. The number it produces is only as good as the inputs you verify on-chain and the honesty of the weights you set — but run faithfully, it is the most reliable edge available to a retail on-chain trader: not a way to win, but a way to stop losing the ways you can see coming.

## Further reading & cross-links

This post is the capstone of the token-selection track; each factor has a dedicated deep-dive that supplies the mechanics this scorecard assumes:

- [On-chain due-diligence checklist](/blog/trading/onchain/onchain-due-diligence-checklist) — the seven-gate qualitative pass that the scorecard quantifies into a number.
- [Rug-pull and honeypot detection](/blog/trading/onchain/rug-pull-and-honeypot-detection) — the mechanics behind hard gates 1 and 2.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — the distribution factor and the clustering that powers gate 4.
- [Token unlocks, vesting and emissions](/blog/trading/onchain/token-unlocks-vesting-and-emissions) — the unlock-overhang factor and how to read vesting on-chain.
- [On-chain fundamentals: fees, revenue and TVL](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl) — the fundamentals factor and the sticky-versus-mercenary TVL distinction.
- [Following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) — the smart-money factor and why its labels demand a low weight.

For the broader context of how these tokens fit into markets, see [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) and [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset).
