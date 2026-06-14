---
title: "Three Arrows Capital and the Crypto Lender Contagion: When Trust-Based Leverage Detonated"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How one crypto hedge fund borrowed billions on its reputation, blew up when the market fell, and dragged a chain of crypto lenders and their retail depositors down with it."
tags: ["crypto", "three-arrows-capital", "celsius", "voyager", "blockfi", "genesis", "leverage", "contagion", "defi", "cefi", "crypto-lending", "stablecoin"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Three Arrows Capital (3AC) was a crypto hedge fund so trusted that lenders handed it billions with almost no collateral; when Terra and the broader market crashed in 2022, 3AC's hidden leverage detonated and dragged a chain of crypto lenders down with it.
>
> - 3AC controlled an estimated \$10 billion at its peak and borrowed from nearly every major crypto lender — Genesis, BlockFi, Voyager, Celsius — often on little more than its name.
> - Those lenders paid retail depositors eye-watering yields (Celsius advertised up to ~18% APY) and funded those yields by re-lending the deposits to leveraged firms like 3AC.
> - When Terra's UST stablecoin de-pegged in May 2022 and crypto prices fell, 3AC's concentrated bets on LUNA, staked ETH, and GBTC collapsed; it could not meet margin calls and defaulted on roughly \$3.5 billion.
> - The default cascaded: Celsius froze withdrawals in June and went bankrupt in July, Voyager went bankrupt, BlockFi was crippled and later swallowed by the FTX collapse, and Genesis collapsed months later — trapping ordinary depositors.
> - The deep cause was trust-based, undercollateralized lending and rehypothecation: the same dollar of collateral backed many loans, so the system's real leverage was far higher than any single balance sheet showed.
> - The hard contrast: DeFi loans auto-liquidate by code the moment collateral falls short, while these CeFi loans relied on a borrower honoring a phone call — and 3AC did not.

The diagram above is the mental model for this whole story: a single fund's default did not stay contained, it cascaded outward through the firms that had lent to it and finally reached the ordinary people whose savings sat inside those firms. Three Arrows Capital was not a fraud in the way a Ponzi scheme is. It was something arguably more instructive — a respected, sophisticated fund that took on enormous, hidden leverage with the full cooperation of an industry that had decided its reputation was as good as collateral. When the bet went wrong, there was no collateral to seize, no code to liquidate the position, and no buffer to absorb the loss. There was only a chain of counterparties who had all assumed someone else was checking.

![Contagion web from Terra to 3AC to lenders to retail](/imgs/blogs/three-arrows-capital-and-crypto-lender-contagion-2.png)

This post rebuilds the collapse from first principles. We will define every term — a hedge fund, leverage, collateral, the difference between DeFi and CeFi lending, rehypothecation, a margin call, contagion — before we trace the timeline, dissect the exact financial mechanism that failed, and walk through the aftermath that defined the 2022 crypto winter. Throughout, we will ground the abstractions in worked dollar examples, because the numbers are where the recklessness becomes legible. The goal is not to gawk at a blow-up but to understand the structural flaw it exposed: an entire lending industry that ran on trust and discovered, all at once, that trust is not the same thing as collateral.

## First principles: the building blocks of the blow-up

Before any of the names — Su Zhu, Kyle Davies, Celsius, Genesis — make sense, we need a shared vocabulary. Crypto borrowed most of its machinery from traditional finance and then stripped away most of the guardrails. If you understand a handful of plain ideas, the entire collapse reads like a logical, almost inevitable consequence rather than a freak event.

### What a hedge fund is

A **hedge fund** is a private investment firm that pools money from wealthy individuals and institutions and tries to grow it using strategies far more aggressive than a normal mutual fund would touch. Where an ordinary fund might just buy and hold a basket of stocks, a hedge fund can borrow heavily, bet that prices will fall (short selling), use derivatives, and concentrate large amounts of money into a small number of convictions. In exchange for chasing higher returns, hedge funds face less regulation and accept more risk. We cover the economics of these firms in detail in [how hedge funds work and what 2-and-20 means](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20); the one-line version is that they are paid to take risk, and the smartest ones get treated as if their risk-taking is infallible.

Three Arrows Capital was a crypto-native hedge fund. Instead of stocks and bonds, it traded Bitcoin, Ethereum, and a long tail of newer tokens, plus crypto-flavored financial products. Its founders had built a reputation as some of the most thoughtful, well-connected traders in the industry. That reputation, as we will see, became the most dangerous asset on its balance sheet.

### What leverage is

**Leverage** means using borrowed money to increase the size of a bet. If you have \$1 and you borrow \$9, you can now control a \$10 position. If that position rises 10%, you make \$1 — a 100% return on your own dollar. But if it falls 10%, you lose \$1 — your entire stake — and you still owe the \$9 you borrowed. Leverage multiplies gains and losses symmetrically, and it introduces a brutal asymmetry that buy-and-hold investors never face: you can be forced out of a position before it has a chance to recover.

The phrase to hold onto is **hidden leverage**. A fund's leverage is hidden when no single lender can see the whole picture — when the fund has borrowed from ten different places, and each lender thinks it is one of only a few. Each lender sizes its loan as if it is the fund's main creditor. Stack ten such loans and the fund's true leverage is a multiple of what any one lender believed. That is precisely what 3AC did.

There is one more property of leverage worth internalizing before we go further: the **forced seller** problem. An ordinary investor who buys an asset with their own money can simply wait out a downturn — they are never compelled to sell at the bottom. A leveraged investor cannot. The borrowed money comes with terms: maintain a minimum buffer, or repay on demand. When the asset falls, the leveraged investor is *forced* to sell to meet those terms, regardless of whether they think the price will recover. And because leverage is everywhere in a hot market, many investors are forced to sell at the same moment, into the same falling market, driving the price down further and forcing yet more selling. Leverage does not just amplify one investor's losses; it synchronizes everyone's, and turns an orderly decline into a cascade. Keep this in mind — it is the engine that ran inside the 3AC collapse.

### Collateral, and the great divide between DeFi and CeFi lending

**Collateral** is something of value a borrower pledges so the lender can be made whole if the borrower fails to repay. A home mortgage is collateralized by the house; if you stop paying, the bank takes the house. In crypto there are two utterly different worlds of lending, and the difference between them is the spine of this entire story.

**DeFi** — decentralized finance — means lending that runs on a **blockchain** (a shared, tamper-resistant public ledger) through **smart contracts** (self-executing programs that hold and move money according to fixed rules, with no human in the loop). DeFi lending is almost always **overcollateralized**: to borrow \$100 of a stablecoin, you might have to lock up \$150 of Ethereum. If your collateral's value falls toward the loan amount, the smart contract **auto-liquidates** it — it sells your collateral automatically, the instant a price feed (an **oracle**) says you have crossed a threshold. No trust is required. The code does not care who you are.

**CeFi** — centralized finance — means lending run by a company, the way a bank runs lending. A CeFi crypto lender like Celsius or Genesis is a business with employees, a balance sheet, and discretion. Crucially, CeFi lenders could and did lend **undercollateralized** or even uncollateralized to firms they trusted. They lent on relationships, on a borrower's reputation, on a handshake and a credit assessment — exactly the way a bank extends an unsecured line of credit to a blue-chip company. When a CeFi loan goes bad, there is no code to liquidate it. There is a phone call, a lawyer, and a hope that the borrower pays.

![DeFi auto-liquidation versus CeFi trust-based default](/imgs/blogs/three-arrows-capital-and-crypto-lender-contagion-7.png)

Look at the figure above, because it is the single most important contrast in this post. When collateral drops, a DeFi loan flows down the left path — the smart contract liquidates the position by code and the lender is made whole automatically. A CeFi loan flows down the right path — the lender issues a margin call, a request that the borrower top up or repay, and then waits. If the borrower honors it, fine. If the borrower ignores it, the lender simply eats the loss. The entire 3AC contagion is the right-hand path firing at scale.

Why does DeFi auto-liquidate while CeFi relies on trust? It comes down to *who holds the asset and who decides*. In a DeFi loan, the collateral is locked inside a smart contract — the borrower does not hold it, the code does. The contract continuously reads the collateral's price from an oracle, and the moment the ratio of collateral to debt crosses a pre-set threshold, anyone in the world can trigger the liquidation (and is rewarded for doing so), or the contract does it itself. There is no negotiation, no relationship, no possibility of the borrower saying "give me a few days." The borrower's identity, reputation, and feelings are irrelevant; the code only sees a number, and it acts on that number in seconds. The lender's protection is mechanical and unconditional.

In a CeFi loan, the borrower holds the assets and the lender holds an IOU. To enforce the loan, the lender must *ask* the borrower to top up or repay, and then rely on the borrower's willingness and ability to comply. The lender's protection is a promise, backed by a legal claim that takes months or years to enforce in court. When the borrower is solvent and cooperative, this works fine and is actually more flexible than rigid code. When the borrower is insolvent and uncooperative — exactly when the lender most needs protection — the promise is worth nothing, and the lender discovers it was never really protected at all. This is the difference between a system that enforces its rules with physics and a system that enforces its rules with trust, and 2022 was the year the difference became a tens-of-billions-of-dollars lesson.

### What a centralized crypto lender is, and where the yield came from

A **centralized crypto lender** in 2021–2022 looked, to an ordinary user, like a high-interest savings account for crypto. You deposited your Bitcoin or stablecoins, and the company paid you interest — often advertised as an **APY** (annual percentage yield), the total return you would earn over a year. Celsius famously marketed rates as high as roughly 18% on certain assets, at a time when a normal bank savings account paid well under 1%.

Where does an 18% yield come from? This is the question that should have stopped everyone, and we will answer it concretely with arithmetic later. The short answer: the lender took your deposit and **re-lent it** — to leveraged trading firms, to other crypto businesses, and into yield strategies of its own — at high rates, kept a slice, and passed the rest back to you. The yield was real only as long as the borrowers on the other side kept paying. The deposits were not sitting safely in a vault; they were out working, exposed to exactly the kind of firm that 3AC was.

### Rehypothecation, the multiplier nobody saw

**Rehypothecation** is a fancy word for a simple, dangerous practice: pledging the same collateral more than once, or re-using collateral you are holding for one purpose to back a second obligation. Here is the everyday version. You pawn a watch for a loan; the pawnbroker then uses your watch as security for a loan it takes from someone else; and that someone else does it again. Now one watch is "backing" three loans. If the watch's value drops, three lenders are all relying on the same shrinking object — and at most one of them can actually be made whole.

In crypto's 2021 boom, collateral was rehypothecated up and down the chain. A retail deposit became a loan to a lender, which became collateral for the lender's own borrowing, which financed a fund's position, which was itself pledged elsewhere. The real leverage of the whole system was vastly higher than the sum of the visible balance sheets, because the same dollar appeared in many places at once.

### What a margin call is

A **margin call** is the moment a lender demands that a leveraged borrower either add more collateral or repay part of the loan, because the value backing the loan has fallen. In a healthy market it is a routine top-up. In a crash, margin calls arrive all at once, for amounts the borrower cannot raise without selling assets — into a market where everyone else is also selling. A wave of simultaneous margin calls is one of the classic mechanisms by which a falling market turns into a collapsing one.

### What contagion is

**Contagion** is the spread of financial failure from one firm to the next through their connections. Firm A fails and cannot repay Firm B; B takes a loss it cannot absorb and fails, hurting C; and so on. Contagion is most violent when many firms are connected to the same central node and are all exposed to the same bet — because then a single shock does not hit one firm, it hits all of them through their shared counterparty. 3AC was that central node.

There is also a faster, more psychological channel of contagion: the **run**. A run happens when depositors or lenders, fearing a firm will fail, all rush to withdraw at once. Because no lending business keeps every dollar of deposits sitting idle — the whole point is to lend them out and earn a spread — a firm facing simultaneous withdrawals cannot return everyone's money at the same time, even if it is fundamentally sound. The fear of insolvency *causes* the insolvency. In traditional banking, deposit insurance and a central bank standing ready to lend exist precisely to stop runs by reassuring depositors they will be paid no matter what. Crypto lenders had neither. So when the news of 3AC's trouble spread, depositors at the connected lenders did exactly what depositors have done for centuries — they ran for the door — and the lenders, holding illiquid loans they could not call back fast enough, froze withdrawals to stop the bleeding. A freeze is a run that the institution has lost.

With these eight ideas — hedge fund, leverage, collateral, the DeFi/CeFi divide, the CeFi lender, rehypothecation, the margin call, and contagion — the rest of the story assembles itself. Now let us meet the fund.

## The setup: a fund too trusted to question

Three Arrows Capital was founded in 2012 by Su Zhu and Kyle Davies, two former classmates who had cut their teeth as traders at traditional finance firms before going independent. For years 3AC ran a relatively quiet arbitrage strategy in emerging-market currencies and then in crypto. By the 2021 bull market it had transformed into one of the most influential crypto funds in the world, with an estimated **\$10 billion** in assets under management at its peak and a founder, Su Zhu, whose market commentary moved sentiment across the industry.

The reputation mattered more than any specific trade. Lenders did not just believe 3AC was smart; they believed 3AC was *safe*, in the way that mattered for credit. When a firm is perceived as a pillar of the industry, lending to it feels riskless, and the price of credit — the collateral demanded, the interest charged — drops toward zero. 3AC could borrow large sums against tiny collateral, or sometimes against a signed promise and a balance-sheet attestation, because no lender wanted to be the one that demanded full collateral from the industry's golden fund and lost the relationship to a competitor who didn't.

This is a recurring pattern in finance worth naming explicitly: **credit is most freely extended exactly when it should be most scrutinized**. In a boom, defaults are rare, everyone is making money, and a lender who insists on tough collateral terms looks paranoid and loses business to rivals offering easier terms. The discipline that would prevent a blow-up is competed away precisely because there has not been a blow-up recently. Lenders' risk teams, who might have demanded more collateral or a fuller picture of 3AC's other borrowings, were overruled by the commercial reality that the relationship was lucrative and the borrower was prestigious. The terms got looser the longer the good times lasted, which is the same as saying the system got more fragile the safer it appeared. By the time the music stopped, the loans had been made on terms that only made sense in a world where prices never fell — and prices always, eventually, fall.

It also mattered that crypto credit was largely **opaque and uncoordinated**. There was no shared credit registry where a lender could look up 3AC's total borrowings across the industry, the way a bank can pull a corporate borrower's full debt profile. Each lender saw only its own loan and 3AC's voluntary, unaudited representations about the rest. So a fund could tell ten lenders a story that was individually plausible to each, while the sum across all ten was a leverage profile that none of them would have accepted had they seen it whole. Opacity was not an accident of the technology; it was, functionally, the mechanism that let the hidden leverage hide.

This dynamic — competitive lenders racing to extend the easiest possible terms to a trophy client — is the soil in which hidden leverage grows. 3AC borrowed from, by various accounts, nearly every major crypto lender: **Genesis** (the largest single creditor, owed roughly \$2.3 billion), **Voyager Digital** (owed roughly \$650 million), **BlockFi**, **Celsius**, and a long list of trading desks and exchanges. Each loan, viewed alone, might have looked prudent. Stacked together, they amounted to a fund running enormous leverage that no single creditor could see in full.

### The concentrated bets

A leveraged fund survives only if its positions are diversified or genuinely hedged. 3AC was neither. By early 2022 its book had become a set of large, correlated, illiquid bets that all depended on the crypto bull market continuing:

- **GBTC** — the Grayscale Bitcoin Trust, a TradFi-listed vehicle that held Bitcoin and let investors get Bitcoin exposure through a brokerage account. For years GBTC shares traded at a *premium* to the Bitcoin they represented, and a trade existed to capture that premium. By 2022 the premium had flipped to a deep *discount* — GBTC shares traded well below the value of their underlying Bitcoin — and 3AC was one of the largest holders, trapped in a position that bled value as the discount widened.
- **Staked ETH (stETH)** — a token representing Ethereum that had been "staked" (locked up to help secure the network) and that was *supposed* to be redeemable one-for-one for regular ETH, but only after a network upgrade that had not yet happened. Until then stETH could only be sold on the open market, where it was meant to trade close to ETH's price. 3AC held a large, leveraged stETH position.
- **LUNA** — the token of the Terra ecosystem, paired with the UST algorithmic stablecoin. 3AC was a high-profile, leveraged believer in Terra, with a position reportedly worth hundreds of millions of dollars at its height.

Three bets, all crypto-directional, all illiquid in size, all financed with borrowed money. There was no real hedge. If crypto fell, every leg fell together — and the borrowed money would have to be repaid regardless.

It is worth slowing down on two of these, because they are subtle and they each contained a trap that looked, on the way in, like a clever arbitrage rather than a directional gamble.

The **GBTC trade** had once been one of the safest-looking strategies in crypto. The Grayscale Bitcoin Trust let big investors create new shares by handing Grayscale actual Bitcoin, then sell those shares to the public after a lock-up period. For years the public paid a *premium* — GBTC shares traded above the value of the Bitcoin inside the trust — because it was one of the few ways a regulated brokerage account could hold Bitcoin exposure. A fund could deposit Bitcoin, receive shares, wait out the lock-up, and sell into the premium for a low-risk gain. The trap: this only worked while a premium existed. As more competition for Bitcoin exposure appeared and sentiment soured, the premium collapsed and flipped to a *discount* — shares traded *below* the Bitcoin they represented, sometimes 30% or more below — and there was no easy way to redeem the shares for the underlying Bitcoin to close the gap. A fund holding a large GBTC position was now sitting on shares worth far less than the Bitcoin they nominally tracked, with no clean exit. What had been an arbitrage had silently become a large, illiquid, losing directional bet.

The **stETH trade** had a similar shape. When you stake ETH on certain platforms you receive stETH, a token meant to represent your staked ETH plus accrued rewards, redeemable one-for-one for ETH — *eventually*. In 2022 that redemption was not yet possible; the network upgrade that would allow it had not shipped. So the only way to turn stETH back into ETH was to sell it to someone else on the open market. As long as everyone believed stETH would eventually redeem at parity, it traded very close to ETH's price, and you could earn the staking yield essentially for free. The trap: the parity depended on confidence, not on any mechanism that could force it. When panic hit and many holders rushed for the exit at once, stETH's market price fell several percent below ETH's. For an unleveraged holder, a few percent is an annoyance. For a holder running the position with borrowed money — as 3AC and several lenders were — a few percent of de-peg, multiplied by the leverage, was enough to vaporize the equity behind the trade.

Both trades shared a fatal feature: they relied on a relationship (premium, parity) holding, with no enforcement mechanism behind it. They were short *liquidity* and short *confidence* — exactly the two things that vanish first in a crash.

### The yield machine behind the lenders

Why were lenders so eager to push money into 3AC? Because they had promised their own depositors yields they could only meet by lending aggressively. Celsius, Voyager, BlockFi, and Genesis all ran some version of the same machine: take retail deposits, promise a high yield, and earn that yield by re-lending the deposits to firms that would pay up — firms running leveraged trades, like 3AC.

![How CeFi yield actually worked, from deposit to leveraged bet](/imgs/blogs/three-arrows-capital-and-crypto-lender-contagion-4.png)

The pipeline above is the way this worked, end to end. A retail depositor hands the CeFi lender \$10,000. The lender re-lends it to funds like 3AC. The fund uses it to put on leveraged bets — LUNA, stETH, GBTC. As long as those bets generate returns, the chain pays interest backward: the fund pays the lender, the lender pays the depositor the advertised APY and keeps a margin. The depositor sees a clean 18% and never sees the leveraged bet at the far end of the pipe. The yield was not magic; it was the compensation for a risk the depositor could not see and had not knowingly accepted.

This is the crucial structural fact to carry into the timeline: **the lenders and the fund were not independent**. The lenders' ability to pay retail depositors depended on the fund's ability to pay the lenders, which depended on the bets continuing to work. They were links in a single chain, and a chain fails at its weakest link.

## The blow-up, step by step

The collapse compressed into roughly eight weeks. It is worth walking through the chronology before dissecting the mechanism, because the *speed* is part of the lesson — contagion does not give you time to react.

![Timeline of the collapse from Terra to bankruptcy, May to July 2022](/imgs/blogs/three-arrows-capital-and-crypto-lender-contagion-1.png)

The timeline above marks the milestones. Each is a step in the same cascade.

### May 9–13: Terra detonates

In early May 2022, **UST**, the algorithmic stablecoin at the heart of the Terra ecosystem, lost its peg to the U.S. dollar — it began trading below \$1 and then collapsed. A stablecoin is supposed to be worth exactly \$1 at all times; UST's design relied on a feedback loop with its sister token LUNA to defend that peg, and when confidence cracked, the loop ran in reverse and destroyed both. Within days, LUNA fell from the mid-\$80s to fractions of a cent, and tens of billions of dollars of value evaporated. (The mechanism of that specific failure is its own story; see [the Terra/LUNA 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse).)

For 3AC this was the first detonation. Its leveraged LUNA position, reportedly worth hundreds of millions of dollars at its peak, went to essentially zero. A diversified fund would have absorbed it. A fund with three correlated, leveraged crypto bets had just watched one of them vanish while the other two were about to fall in the broad sell-off Terra triggered.

### Mid-to-late May: the other bets crack

Terra's collapse did not happen in isolation; it dragged the entire crypto market down with it as confidence drained. As crypto prices fell:

- The **GBTC discount** widened further. With Bitcoin falling and sentiment broken, GBTC shares traded at a steeper and steeper discount to their underlying Bitcoin, deepening the loss on 3AC's large position.
- **stETH de-pegged** from ETH. In a panic, holders rushed to sell stETH for regular ETH, but because stETH could not yet be redeemed directly, the only exit was the open market. The selling pressure pushed stETH's price meaningfully below ETH's — a de-peg of a few percent, which sounds small until you remember it was held with leverage. A leveraged position needs only a modest adverse move to wipe out the equity behind it.

Now all three legs were losing at once, financed with borrowed money. The losses were no longer hypothetical; they were eating through whatever thin equity buffer 3AC had behind its loans.

### June: the margin calls 3AC could not meet

As 3AC's collateral fell in value, its lenders did what lenders do: they issued **margin calls**, demanding that 3AC top up collateral or repay. But 3AC's assets were illiquid and falling, and the demands came from many lenders at once. The fund could not raise the cash. It could not sell its GBTC or stETH at anything near the marks on its books without crushing those prices further. The leverage that had multiplied its gains on the way up now multiplied its losses on the way down, and the forced-seller dynamic kicked in: to meet one margin call it would have to dump assets, which would lower prices, which would trigger more margin calls.

Reports emerged that 3AC was failing to meet its obligations and was scrambling. By mid-June the rumor had hardened into fact: the industry's most trusted fund was insolvent.

#### Worked example: a margin call 3AC could not meet

Walk through a single margin call to feel the trap. Suppose 3AC has put on a \$1 billion leveraged position, financed with \$900 million borrowed and \$100 million of its own equity as the buffer — a 10x leverage ratio. The lender's rule is simple: the equity buffer must stay at or above 10% of the position, or 3AC must top it up.

The position now falls 8%. The \$1 billion of assets is worth \$920 million. But 3AC still owes the \$900 million it borrowed — that number does not move. So 3AC's equity has shrunk from \$100 million to just \$20 million (920 minus 900). The buffer is now only about 2.2% of the position, far below the 10% floor. The lender issues a margin call: restore the buffer to 10% of the \$920 million position — roughly \$92 million — which means 3AC must wire in about \$72 million of fresh cash *today*.

Here is why 3AC could not pay. To raise \$72 million, it would have to sell some of its other holdings — GBTC, stETH — but those were illiquid and falling, and selling them in size would push their prices down further, deepening the losses on the positions it kept. And the same margin call was arriving from several lenders at once, each demanding cash 3AC did not have. The only way to satisfy them all was to liquidate everything into a collapsing market, which would crystallize losses larger than its remaining equity. So it paid no one. The position the lender held had \$920 million of assets against \$900 million owed — a thin \$20 million cushion that the next day's price drop erased entirely, flipping the loan underwater.

The intuition: leverage means a small adverse move eats your entire buffer, and once the buffer is gone, a margin call is a demand for money you can only raise by selling into the very crash that triggered the call.

### June 27 – July: the default and the cascade

On June 27, 2022, a court in the British Virgin Islands ordered 3AC into liquidation, and the firm's lender Voyager issued a notice of default after 3AC failed to repay a loan. 3AC ultimately owed creditors on the order of **\$3.5 billion**, against assets that no longer existed in anything like that amount. The fund had defaulted.

Now the contagion fired. The lenders that had funded 3AC took losses they could not absorb, and because *they* owed *their* depositors, the failure rolled downhill:

- **Celsius** had already **frozen withdrawals on June 12**, locking in roughly \$12 billion of customer assets it could not return, and **filed for bankruptcy on July 13**. Its model — high retail yield funded by risky lending and leveraged DeFi strategies — could not survive the market drop and the loss of confidence.
- **Voyager Digital**, directly exposed to 3AC for roughly \$650 million, **halted withdrawals on July 1** and **filed for bankruptcy on July 5**.
- **BlockFi** was severely wounded by the broader credit losses and a margin call it issued to 3AC; it took an emergency credit line and was, months later, drawn into the orbit of FTX before that exchange's own collapse finished it off (see [the FTX collapse and Sam Bankman-Fried](/blog/trading/crypto/ftx-collapse-sam-bankman-fried)).
- **Genesis**, 3AC's largest creditor with roughly \$2.3 billion of exposure, absorbed an enormous loss; though it limped on for a time with support from its parent, it ultimately filed for bankruptcy in early 2023.

The people at the very end of the chain — ordinary depositors who had simply wanted a better yield than a bank offered — found their accounts frozen and their savings tied up in bankruptcy proceedings for years.

## The mechanism dissected

The timeline tells you *what* happened. The mechanism tells you *why* it was structurally guaranteed to be this bad once the first domino fell. Four interlocking flaws turned a fund's bad bet into an industry-wide failure.

### Flaw 1: undercollateralized lending built on reputation

The foundational error was lending billions to a single fund without enough collateral to cover a default. In DeFi, this is impossible by construction — the code demands overcollateralization and liquidates automatically. In CeFi, it was a choice, and the lenders chose to substitute trust for collateral because 3AC's reputation made full collateral feel unnecessary and competition made it commercially awkward.

![3AC balance sheet before and after the crash](/imgs/blogs/three-arrows-capital-and-crypto-lender-contagion-3.png)

The before-and-after above shows why undercollateralized lending is so fragile. On the left, in the boom, 3AC's bets are marked at roughly \$10 billion, it has borrowed from many lenders, and on paper it looks solvent. On the right, after the crash, the bets are near zero — but the roughly \$3.5 billion of borrowed liabilities has not moved at all. Assets are volatile; debts are fixed. When the volatile side collapses and there is no collateral buffer, the gap between the two sides is the loss the lenders eat. The asymmetry is the whole point: the borrower's upside was unbounded, but the lenders' collateral floor was nonexistent.

#### Worked example: borrowing billions on almost nothing

Suppose a lender extends 3AC a \$500 million loan and, because it trusts the fund, asks for only \$50 million of collateral — a 10% collateral ratio, meaning the loan is 90% uncollateralized. Compare that to a DeFi loan, where to borrow \$500 million you would have to lock up perhaps \$750 million of crypto (a 150% collateralization).

Now the market falls 30% and 3AC's pledged \$50 million of collateral drops to \$35 million. The lender issues a margin call for the shortfall, but 3AC cannot pay. The lender liquidates the \$35 million of collateral it holds — and is still owed \$465 million it will likely never see. The collateral covered 7% of the loan. In the DeFi case, the \$750 million of collateral, even after falling 30% to \$525 million, still exceeds the \$500 million owed, so the smart contract liquidates and the lender is made whole.

The intuition: collateral is the only thing that protects a lender when trust fails, and 3AC's lenders had almost none.

### Flaw 2: rehypothecation hid the true leverage

The second flaw multiplied the first. Because collateral was re-used up and down the chain, the system's real leverage was a multiple of what any one balance sheet revealed.

![The leverage and rehypothecation layers stacked on retail deposits](/imgs/blogs/three-arrows-capital-and-crypto-lender-contagion-6.png)

The stack above shows the layers. At the bottom sit retail deposits — the original dollar. The CeFi lender re-lends that deposit. 3AC borrows it with little collateral. The same collateral gets pledged again to another counterparty. And at the top sit the concentrated bets. Every layer adds leverage, and because the collateral at one layer is reused at the next, the dollar at the bottom is "supporting" a tower many times its size.

#### Worked example: one \$100 of collateral, pledged twice

Imagine 3AC holds \$100 of crypto as collateral. It pledges that \$100 to Lender A to secure a \$90 loan. Through rehypothecation and the murky way crypto credit was tracked, the *same* \$100 of value effectively also stands behind a \$90 obligation to Lender B — because the asset was reused, double-counted, or pledged across overlapping arrangements.

On paper, the system believes there is \$200 of collateral supporting \$180 of loans — comfortably covered. In reality there is \$100 supporting \$180. When the \$100 falls to \$60 in a crash, both lenders try to claim it. At most one can be made whole; the other is left with a loss against a collateral pool that was never really there. Multiply this across an entire industry and the visible leverage ratios become fiction.

The intuition: rehypothecation makes the system look better-collateralized than it is, so the true risk only becomes visible at the exact worst moment — when everyone reaches for the same collateral at once.

### Flaw 3: everyone crowded into the same trades

The third flaw was correlation. 3AC was not the only firm long LUNA, long stETH, and exposed to the GBTC discount — those were *consensus* trades, held across the industry. When Terra fell, it did not hurt one firm; it hurt every firm holding the same positions, simultaneously. And because those firms were also each other's counterparties, the simultaneous losses meant the counterparties failing you were the very same firms whose failure you were exposed to.

Correlation turns independent risks into a single risk. A lender that thought it had diversified by spreading loans across many borrowers discovered that all its borrowers held the same bets and would all default together. Diversification across counterparties is worthless if the counterparties are all exposed to the same shock.

### Flaw 4: unsustainable CeFi yields demanded reckless lending

The fourth flaw closes the loop. The lenders did not lend recklessly out of carelessness alone; they were *forced* into it by the yields they had promised. To pay an 18% APY, you must earn more than 18% on what you do with the deposits. There are not many safe ways to earn 18% in any market. So the lenders reached for risk — re-lending to leveraged funds, running their own leveraged DeFi strategies, taking the GBTC and stETH trades themselves. The promised yield was a contract to take risk, and the deposit base grew faster than the safe opportunities to deploy it.

#### Worked example: where did Celsius's 18% on \$10,000 come from?

You deposit \$10,000 with a CeFi lender advertising an 18% APY. That promise commits the lender to pay you \$1,800 over the year. To afford that and still profit, the lender must earn, say, 22% on your \$10,000 — \$2,200 — keeping \$400 for itself.

Where on earth do you earn 22% safely? You don't. So the lender re-lends your \$10,000 to a fund like 3AC at a high rate, or locks it into a leveraged DeFi yield strategy, or buys a high-yielding crypto trade. Each of those pays well *only while the market rises and borrowers stay solvent*. The 18% was never the yield on a safe asset; it was your share of the proceeds from an undisclosed, leveraged bet. When the bet failed, the yield did not just stop — the principal itself was gone, locked behind a withdrawal freeze.

The intuition: a yield far above the risk-free rate is not a free lunch, it is a hidden risk you are being paid to bear, and in 2022 depositors discovered exactly what risk they had been bearing.

### Putting it together: the contagion chain

The four flaws compound into a single failure path. Reputation replaced collateral, rehypothecation hid the leverage, correlation synchronized the losses, and unsustainable yields guaranteed the lending was reckless. When Terra fell, 3AC's correlated bets collapsed together, the missing collateral meant lenders could not be made whole, the hidden leverage meant the losses were larger than anyone had modeled, and the yield promises meant the lenders had no buffer to absorb them.

#### Worked example: who owed whom, summing to billions

Trace the chain in dollars. 3AC owed Genesis roughly \$2.3 billion, Voyager roughly \$650 million, and substantial sums to BlockFi, Celsius, and other desks — totaling on the order of \$3.5 billion across all creditors. Set those losses against the lenders' own depositor obligations:

- Genesis, having lost \$2.3 billion of exposure to 3AC, could not honor its own obligations to counterparties and depositors, and the loss rolled toward its parent and ultimately into bankruptcy.
- Voyager, out \$650 million, had been funding withdrawals for retail customers; with that asset gone, it froze roughly \$1.3 billion of customer funds and filed for bankruptcy.
- Celsius, carrying roughly \$12 billion of customer assets against a balance sheet hollowed out by the same market and the same kind of lending, froze withdrawals and filed for bankruptcy showing a deficit of well over \$1 billion.

Add the depositor money trapped across these firms and the figure runs into the **tens of billions of dollars** of customer assets frozen — far more than 3AC's own \$3.5 billion default, because each lender's failure trapped a much larger pool of retail money behind it. The default was the spark; the frozen deposits were the fire.

![Matrix comparing the four failed lenders](/imgs/blogs/three-arrows-capital-and-crypto-lender-contagion-5.png)

The matrix above lines up the four lenders side by side. The columns are the same for each row: their exposure to 3AC, the retail yield they had promised, and their outcome. Read down any column and the pattern is unmistakable — every one of them funded 3AC in some form, every one paid retail an unsustainable yield, and every one ended in bankruptcy or a forced sale. They were not four independent failures; they were one failure expressed four times, because they all ran the same business model and all sat downstream of the same fund.

## The aftermath

The immediate aftermath was a wave of bankruptcies and a freeze on billions of dollars of ordinary people's money. Celsius depositors, Voyager depositors, and later Genesis creditors entered multi-year bankruptcy processes, recovering cents on the dollar in slow, contested proceedings. Many depositors had treated these accounts like savings accounts; they learned, painfully, that an unregulated crypto lender offering 18% is nothing like a bank deposit insured by the government.

The 2022 **crypto winter** deepened. The Terra collapse had started the sell-off; the 3AC and lender contagion accelerated it, draining confidence and liquidity from the entire market. Crypto firms that had appeared rock-solid were revealed to be interconnected in ways outsiders — and many insiders — had never appreciated. The contagion also set the stage for the year's second great collapse: a weakened, interconnected industry was far more vulnerable when FTX imploded that November.

The founders of 3AC, Su Zhu and Kyle Davies, became the public faces of the blow-up. They were initially evasive about the firm's whereabouts and obligations, drawing the ire of creditors and a court; Su Zhu was later detained in connection with the liquidation proceedings. The liquidators spent years trying to trace and recover assets from a fund whose books had been, charitably, opaque.

Regulators took notice. The collapse intensified scrutiny of crypto lenders specifically — the firms that had marketed bank-like yields without bank-like protections. Cases and enforcement actions followed against several of the lenders for offering unregistered securities in the form of their yield products, and the episode became Exhibit A in arguments that centralized crypto intermediaries needed the kind of capital, custody, and disclosure rules that govern traditional financial institutions. The deeper regulatory question — how to supervise undercollateralized, rehypothecated lending in an industry built to route around supervision — remains contested.

The bankruptcy proceedings themselves became a slow, public education in what these "deposits" actually were. In the Celsius and Voyager cases, courts had to grapple with a question most depositors never asked when they signed up: did the customer still *own* the crypto they deposited, or had they *lent* it to the company, making them an unsecured creditor standing in line behind everyone else? The fine print, in many cases, said the latter — that title to the deposited assets had passed to the company, which was free to use them. That single legal distinction determined whether a depositor was getting their coins back or getting cents on the dollar at the end of a multi-year process. Worse, because crypto prices had fallen so far, even the recoveries that were eventually distributed were measured against a much-diminished pool, and depositors who had wanted the safety of cash found themselves holding claims whose value swung with a market they had been trying to step away from.

There was also a stark unfairness in the *timing* of who got out. Because the freezes happened suddenly, the customers who happened to withdraw in the days and weeks before a freeze got their money out in full, while those who waited got trapped — a lottery based on nothing but luck and the rumor mill. This, too, is a classic feature of a run: the early movers are made whole at the expense of the late ones, which is precisely why the rational response to any whiff of trouble is to run, which is precisely what makes runs self-fulfilling.

One thing the collapse did *not* break is instructive. DeFi lending protocols, which auto-liquidate by code, largely kept functioning through the crisis. As 3AC and the CeFi lenders imploded, the on-chain lending protocols processed liquidations mechanically and continued to honor withdrawals. The contrast was stark and not lost on the industry: the trust-based, human-discretion CeFi lenders froze and failed, while the trustless, code-enforced DeFi protocols, for all their other risks, did exactly what they promised.

## Common misconceptions

**"3AC was a Ponzi scheme."** No. A Ponzi scheme pays old investors with new investors' money and has no real strategy. 3AC was a real hedge fund making real (if reckless) leveraged bets. It failed because the bets went wrong and it was over-levered, not because it was a fraud paying fake returns. The distinction matters: the lesson of 3AC is about leverage and trust-based credit, not about a con. (The yield *products* sold to retail by some lenders are a separate question, and several lenders did face securities charges — but 3AC itself was not a Ponzi.)

**"The lenders were just unlucky."** They were not unlucky; they were structurally exposed. They had chosen to lend undercollateralized, to crowd into the same trades, and to promise yields that forced them into risky lending. The Terra shock was the trigger, but the gunpowder had been laid for months. A different trigger would have produced a similar collapse.

**"If only 3AC had had more collateral, this wouldn't have happened."** More collateral at 3AC would have softened the blow, but the deeper problem was systemic: the rehypothecation, the correlation, and the yield promises meant that even well-collateralized loans were sitting on a foundation that was weaker than it looked. The collateral that did exist was often the same collateral pledged elsewhere.

**"DeFi is riskier than CeFi because it's unregulated and automated."** This episode showed the opposite can be true. DeFi's automation — auto-liquidation by code — is exactly what protected DeFi lenders during the crash, while CeFi's human discretion is exactly what failed. DeFi has real and serious risks (smart-contract bugs, oracle manipulation, governance attacks), but "the borrower won't honor a margin call" is not one of them, because there is no margin call to honor — the code simply acts.

**"Retail depositors knew they were taking a risk for that yield."** Most did not understand the risk. The yield was marketed as a safe, bank-like return. The chain from their deposit to a leveraged LUNA bet was invisible to them. "You should have known better" misplaces the responsibility for a risk that was deliberately obscured.

**"The losses were contained to crypto, so it didn't matter."** The losses were largely contained to crypto, which is why this did not become a 2008-style economy-wide crisis — but it absolutely mattered to the hundreds of thousands of people whose savings were frozen, and it mattered as a clean, real-world demonstration of contagion mechanics that apply to any leveraged, interconnected financial system, crypto or not.

## How it echoes in other markets

The 3AC contagion was novel in its details and ancient in its structure. The same mechanisms have detonated repeatedly across financial history.

**LTCM, 1998.** Long-Term Capital Management was a hedge fund run by Nobel laureates that took on enormous leverage with the full cooperation of banks that trusted its sophistication — exactly the reputation-as-collateral dynamic. When markets moved against its correlated bets, it could not meet its obligations, and its forced unwind threatened the firms that had lent to it, prompting a coordinated bailout. The parallel to 3AC is almost exact, line for line: a fund whose prestige let it borrow on thin collateral; lenders who each saw only their slice and underestimated the total leverage; correlated positions that all moved against the fund at once; and a forced unwind that threatened to take down the creditors. The one difference is the ending. LTCM's creditors were a small club of major banks that the Federal Reserve could herd into a room to organize an orderly bailout, because letting LTCM dump its positions would have hurt all of them. 3AC's creditors were scattered, uncoordinated crypto lenders with no central bank to convene them and no appetite to rescue a rival, so the unwind ran to completion and the losses fell where they landed. See [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed).

**The 2008 shadow-bank runs.** In 2008, the failure point was not traditional bank deposits but the "shadow banking" system — the repo market and money-market funds — where short-term, undercollateralized or thinly-collateralized lending between financial firms suddenly froze when trust evaporated. Lehman Brothers' collapse propagated through counterparties the way 3AC's did, and rehypothecation of collateral was a central feature there too: the same securities had been pledged and re-pledged through chains of firms, so when prices fell, far more claims chased the collateral than the collateral could satisfy. The crypto lenders of 2022 were, structurally, a shadow-banking system rebuilt from scratch — the same maturity mismatch (borrowing short from depositors, lending long to funds), the same rehypothecation, and the same run dynamics — but without the deposit insurance, capital requirements, and central-bank backstop that traditional banking had bolted on after its own historical disasters. Crypto had, in effect, re-derived the conditions for a banking panic and then removed the safeguards that exist specifically to prevent one.

**FTX, November 2022.** The very next collapse in crypto repeated the theme: an exchange that mingled customer funds with a trading firm's leveraged bets, propped up by trust in a charismatic founder, that imploded when the bets failed and the trust broke. The 3AC contagion had already weakened the industry and trapped capital, making the FTX shock land harder. See [the FTX collapse and Sam Bankman-Fried](/blog/trading/crypto/ftx-collapse-sam-bankman-fried).

**Bank runs and the 2023 regional-bank failures.** A withdrawal freeze at Celsius is a bank run in slow motion: more claims on the institution than it can immediately honor. The classic remedy in banking — deposit insurance and a central-bank lender of last resort — is exactly what crypto lenders lacked, which is why a run that a regulated bank might survive was fatal to them.

**Any leveraged carry trade.** The structure — borrow cheaply, lend or invest at a higher rate, pocket the spread, and pray the spread holds — is the eternal carry trade. It works until the market moves, at which point the leverage that amplified the spread amplifies the loss. From currency carry trades to leveraged real-estate to crypto yield products, the failure mode rhymes.

## When this matters to you and further reading

You may never lend billions to a hedge fund, but the lessons here apply directly to anyone who touches crypto or evaluates any financial product:

- **A yield far above the risk-free rate is a risk in disguise.** When something pays much more than a bank or a government bond, ask where the money comes from. If you cannot trace the source of the yield to a real, sustainable activity, you are being paid to bear a risk you have not been shown. An 18% APY is a flashing sign, not a deal.
- **Custody is not the same as a promise.** When you deposit crypto with a centralized lender, you are not storing it; you are lending it to a company that will do something with it. "Not your keys, not your coins" is the crypto-native way of saying that an IOU from an unregulated firm is worth exactly what that firm is worth in a crisis.
- **Leverage and collateral are the questions that matter.** For any financial firm or product, the load-bearing questions are: how much is it borrowing, against what collateral, and what happens to its counterparties if it fails? Reputation is not collateral, and a chain of trust is only as strong as its most overextended link.
- **Automation can be a feature.** DeFi's auto-liquidation looked cold and risky in the boom and looked like the only honest mechanism in the bust. Rules enforced by code do not get talked out of a margin call.

For the surrounding context, the natural companions to this post are [the Terra/LUNA 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse), which is the shock that lit the fuse; [the FTX collapse and Sam Bankman-Fried](/blog/trading/crypto/ftx-collapse-sam-bankman-fried), the contagion's sequel; [how hedge funds work and what 2-and-20 means](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20), for the economics of the firm at the center; and [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed), the traditional-finance template that 3AC repeated almost line for line.

The figures in this post — peak assets, default amounts, exposures, yields, and frozen-deposit totals — are drawn from bankruptcy filings, court documents, and contemporaneous reporting, and are approximate, rounded, and as-of the 2022–2023 proceedings. None of this is investment advice; it is an autopsy. The point of an autopsy is not to mourn the patient but to learn how the body actually works — and the body of a trust-based, leveraged, interconnected lending system works exactly the way 3AC and its lenders demonstrated, every single time the trust runs out.
