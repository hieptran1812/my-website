---
title: "Margin and the Default Waterfall: How a CCP Survives a Blowup"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How a clearinghouse uses margin, a guarantee fund, and an ordered default waterfall to absorb a member's failure without taking the whole market down with it."
tags: ["capital-markets", "clearing", "ccp", "margin", "default-waterfall", "risk-management", "post-trade", "systemic-risk"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A central counterparty (CCP) does not avoid the risk that a member fails; it pre-funds that failure with margin and a mutualised guarantee fund, then absorbs the loss in a fixed order called the default waterfall.
>
> - **Initial margin** is collateral posted up front, sized to the worst loss the CCP expects while it closes out your book over a few days; **variation margin** is daily cash that settles your gains and losses so exposure never piles up.
> - The **default waterfall** absorbs a failure in order: (1) the defaulter's own margin, (2) the defaulter's guarantee-fund slice, (3) the CCP's own skin-in-the-game, (4) the survivors' mutualised fund, (5) emergency assessments and recovery tools.
> - Members accept **mutualisation** — paying into a shared pool that can be used to cover someone else's failure — because that is the price of getting an anonymous, guaranteed counterparty on every trade.
> - The dark side: margin is **procyclical**. When volatility spikes, margin calls jump at the exact moment cash is scarce — the 2020 dash-for-cash, the 2022 UK gilt/LDI crisis, and the 2022 LME nickel squeeze all turned a margin call into a liquidity crisis.

On the morning of 8 March 2022, the price of nickel on the London Metal Exchange did something it had never done before: it doubled in a matter of hours, blowing past \$100,000 a tonne. A single producer with an enormous short position was facing a margin call measured in billions of dollars — money it did not have on hand. The clearinghouse standing in the middle of that trade, LME Clear, was staring at the possibility that the member carrying those positions could not pay. If it could not pay, the clearinghouse itself was on the hook to everyone on the other side. The LME's response — cancelling hours of trades and halting the market — was so drastic that it triggered lawsuits and a regulatory inquiry. But the underlying question was simple and ancient: when one party to a trade cannot make good, who eats the loss?

That question is the entire subject of this post. A clearinghouse, or central counterparty (CCP), exists to answer it. When you trade through a CCP, the CCP legally inserts itself between buyer and seller so that each of you faces the CCP rather than each other. This is the magic trick that lets you trade with a complete stranger and never worry about whether they will be solvent next week. But the trick only works if the CCP itself never fails. And the CCP has no crystal ball — members *will* occasionally blow up. So the CCP builds a fortress of pre-funded defences, drawn down in a strict order, designed so that one member's collapse is absorbed quietly instead of cascading into the failure of the clearinghouse and everyone behind it.

That fortress has two main walls — **margin** and the **default waterfall** — plus a set of last-resort tools for the truly catastrophic day. This post is about how those defences are built, how a real default gets managed, and the uncomfortable truth that the very machinery designed to make markets safe can, in a crisis, become the thing that drains liquidity out of them.

![CCP default waterfall layers absorbing a member loss in order](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-1.png)

## Foundations: what a CCP is and why it needs a fortress

Before we get to margin, let's build the picture from zero, because everything that follows hangs on one idea.

A **security** is a tradable claim — a share of a company, a bond, a futures contract. When you buy one, you and the seller agree on a price now, but the actual exchange of cash for the asset happens slightly later (a day or two for stocks, and for derivatives the obligation can stretch out for months or years). That gap is the problem. Between "we agreed" and "it's done", the other side can vanish, go bankrupt, or simply refuse to pay if the market moved against them. This is **counterparty risk**: the risk that the person on the other side of your trade doesn't make good.

A **central counterparty** removes that risk through a legal step called **novation**. The original contract between you and the seller is torn up and replaced by two new contracts: one between you and the CCP, and one between the seller and the CCP. The CCP becomes "buyer to every seller and seller to every buyer." Now you don't care who you traded with — you only care that the CCP is solvent. (The mechanics of novation, and why this is what makes anonymous exchange trading possible, are the subject of the sibling post [the clearinghouse: how a CCP removes counterparty risk](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk); here we take novation as given and focus on what it forces the CCP to do.)

Here is the catch that creates this entire post. The moment a CCP novates every trade, it has concentrated all the counterparty risk in the market onto itself. If a big member fails owing money, the CCP still has to pay everyone on the other side. So the CCP must hold enough resources to cover the failure of its members — and it must hold them *before* anyone fails, because by the time a member blows up it is far too late to ask them for more. Those pre-funded resources are margin and the guarantee fund, and the rules for spending them are the default waterfall.

The **members** of a CCP — called clearing members — are typically large banks and brokers. Ordinary investors and smaller firms clear *through* a member, the way you bank through a branch rather than holding an account at the central bank. So when we say "a member defaults," we mean one of these large institutions, carrying not just its own positions but those of its clients, fails to meet an obligation to the CCP.

The whole edifice rests on a deal: in exchange for the CCP guaranteeing every trade, each member agrees to post collateral and to **mutualise** — to contribute to a shared pool that can be tapped to cover *another* member's default. That trade-off, money and shared liability in exchange for a guaranteed counterparty, is the thread we'll pull on through the rest of this post.

## Margin: the first and largest wall

Margin is collateral — cash or high-quality securities — that members post to the CCP to back their positions. There are two distinct kinds doing two completely different jobs, and conflating them is the single most common confusion in this whole area.

![Variation margin settles daily losses while initial margin pre-funds the close-out](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-3.png)

### Variation margin: settling the past every day

**Variation margin (VM)** is the answer to "don't let losses build up." Every day — and during stressed markets, intraday — the CCP marks every position to the current market price and works out who has gained and who has lost since the last mark. Losers pay cash *to* the CCP; winners receive cash *from* it. This is not collateral sitting in a vault; it is a real transfer of money that settles the day's profit and loss.

The point of VM is that it resets exposure to roughly zero every single day. If you bought a futures contract and the market fell, you pay that loss in cash tonight. You don't get to let a small loss quietly grow into a catastrophic one over weeks while the CCP's exposure to you balloons. The CCP never carries a large *unrealised* loss against a member, because VM keeps draining it to cash as it accrues.

#### Worked example: variation margin settling a daily loss

Suppose you are a clearing member holding a long position in equity-index futures with a notional value of \$50,000,000. Overnight, the index falls 0.4%. Your mark-to-market loss is:

\$50,000,000 × 0.4% = \$200,000.

The next morning, the CCP issues a variation-margin call: you must wire \$200,000 in cash to the CCP, which passes it to whoever was short the same contracts and therefore gained \$200,000. Your loss is now *settled* — it has left your account as cash. The CCP's exposure to you on yesterday's move is back to zero. If the index falls another 0.4% today, that's a fresh \$200,000 call tomorrow, settled again. **Variation margin is the mechanism that stops a loss from ever becoming a stale, unfunded IOU.**

### Initial margin: pre-funding the future

**Initial margin (IM)** answers a different question: "if this member fails *right now*, how much could we lose while we close out their positions?" When a member defaults, the CCP can't unwind their book instantly. It takes time — typically assumed to be one to several days — to hedge and auction the portfolio, and during that window the market keeps moving. Initial margin is the buffer that covers that potential adverse move over the **close-out period**.

Because IM is meant to cover a *future* loss the CCP hasn't seen yet, it is sized by a statistical model. The two workhorses are **VaR-style models** (Value-at-Risk: "what is the loss we'd exceed only X% of the time over the close-out window?") and **SPAN** (a scenario-grid system long used by futures clearinghouses that prices a portfolio under a fixed set of stressed up/down moves and takes the worst). The output is the same in spirit: a collateral number large enough that, with high confidence (commonly 99% or 99.5%), it covers the loss the CCP would suffer closing the book out.

#### Worked example: sizing initial margin to a 2-day 99% move

Take a \$10,000,000 position in an asset whose daily returns have a standard deviation (volatility) of 2%. The CCP assumes a 2-day close-out window and wants 99% confidence.

- The 99% point of a normal distribution is about 2.33 standard deviations.
- Scaling daily volatility to two days: 2% × √2 ≈ 2.83%.
- The 99% two-day move: 2.33 × 2.83% ≈ 6.59%.
- Initial margin: \$10,000,000 × 6.59% ≈ \$659,000.

So the CCP demands roughly \$659,000 of collateral up front against this \$10,000,000 position. The reading: **initial margin is a pre-paid estimate of the worst loss the CCP expects to suffer in the days it takes to clean up after your default.** Note how sensitive it is to that 2% volatility — double the volatility and the margin roughly doubles. Hold that thought; it is the seed of the procyclicality problem later.

The crucial difference: VM moves *with realised losses* (it's settlement of what already happened), while IM is a *static-ish buffer* against losses that haven't happened yet. VM protects the CCP against the past; IM protects it against the future close-out gap. A member posts IM once and tops it up as positions or volatility change; it pays or receives VM every day.

## The default waterfall: who eats the loss, and in what order

Now the centerpiece. Suppose margin isn't enough — a member fails and the loss of closing out their book exceeds the collateral they posted. Where does the rest of the money come from? The CCP rulebook specifies an exact, pre-agreed sequence of resources, drawn down strictly in order. This is the **default waterfall**. Each layer must be completely exhausted before the next one is touched.

![Default waterfall layers and their order-of-magnitude loss-absorbing capacity](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-2.png)

The chart above shows the layers at a stylised large CCP, in order-of-magnitude dollar terms. The shape matters more than the exact numbers: the defaulter's own resources are by far the biggest wall, and the layers that hit *other people* sit deep in the stack. Let's walk through each.

**Layer 1 — the defaulter's initial margin.** The first money spent is the defaulter's own collateral. This is the entire philosophy of the waterfall in one line: *the person who blew up pays first, with their own money.* In the overwhelming majority of defaults, this layer alone covers the loss and nobody else is affected.

**Layer 2 — the defaulter's guarantee-fund contribution.** Every member also pre-funds a slice of the **guarantee fund** (also called the default fund or clearing fund). When a member defaults, *their own* slice of that fund is consumed before anyone else's. Still, only the defaulter's money has been touched.

**Layer 3 — the CCP's skin-in-the-game.** Next, the CCP puts in a tranche of *its own* capital. This is deliberately placed *before* the survivors' money. The amount is usually modest relative to the fund, but its purpose is incentive alignment: the CCP — which sets the margin models and runs the risk management — must feel pain before it reaches into its members' pockets. A CCP with zero skin-in-the-game would be tempted to under-margin to win business, knowing members would foot the bill.

**Layer 4 — the survivors' mutualised guarantee fund.** Only now, with the defaulter's resources *and* the CCP's own capital gone, does the CCP reach into the contributions of the *surviving* members. This is **mutualisation** made concrete: your money can be spent to cover someone else's failure. This is the layer members fear, and the reason CCP risk management is scrutinised so heavily.

**Layer 5 — assessments and recovery tools.** If even the whole guarantee fund is exhausted — a truly extreme event — the CCP turns to its last-resort powers, set out in the rulebook:

- **Assessment powers (cash calls):** the CCP can demand additional contributions from surviving members, usually capped at some multiple of their existing fund contribution.
- **Variation-margin gains haircutting (VMGH):** the CCP withholds part of the VM *gains* it would otherwise pay to the members who were on the winning side. In effect, winners are partially paid in IOUs so the CCP can plug the hole.
- **Partial tear-ups:** the CCP forcibly cancels some of the contracts that mirror the defaulter's unhedgeable positions, capping the loss by simply making the offending trades disappear.

These recovery tools are the difference between a CCP that survives a once-in-a-century event and one that itself collapses — which would be a systemic catastrophe, since by construction the CCP sits behind the whole market.

#### Worked example: walking a \$500M member default down the waterfall

Let a large clearing member default, and after the CCP hedges and auctions their book, the total close-out loss is **\$500,000,000**. Assume this member had posted \$300M of initial margin and \$30M to the guarantee fund; the CCP's skin-in-the-game tranche is \$50M; and the survivors' fund holds \$400M. Walk it down:

- **Layer 1 — defaulter's IM:** \$300M absorbed. Remaining loss: \$500M − \$300M = \$200M.
- **Layer 2 — defaulter's fund slice:** \$30M absorbed. Remaining: \$200M − \$30M = \$170M.
- **Layer 3 — CCP skin-in-the-game:** \$50M absorbed. Remaining: \$170M − \$50M = \$120M.
- **Layer 4 — survivors' fund:** \$120M absorbed, leaving \$400M − \$120M = \$280M of the survivors' fund intact.

The loss is fully covered at layer 4, and we never reached assessments or VMGH. The reading: **the defaulter's own resources took \$330M of the \$500M hit — two-thirds — and the survivors collectively lost \$120M, a fraction of the fund they had pre-committed.** No surviving member faced an uncapped, surprise bill; everyone knew their maximum exposure when they signed the rulebook. That predictability is what makes mutualisation acceptable.

## Default management: how the CCP actually closes out a blown-up member

The waterfall tells you *who pays*. But there's a parallel operational story: *how* the CCP gets the defaulter's positions off its books in the first place. This is the **default management process**, and it is a high-stress, time-boxed operation rehearsed in regular "fire drills."

![Default management process from missed margin call to auction](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-5.png)

The sequence runs roughly like this. First, a member **misses a margin call** — they fail to post the VM or IM the CCP demanded by the deadline. After a short cure period, the CCP **declares a default**, a formal legal step that hands it control of the member's positions and collateral.

Now the CCP holds a portfolio it didn't choose and doesn't want. Its job is to get back to a matched book — for every position the defaulter held, the CCP still faces the surviving member on the other side, so it has a one-sided risk it must neutralise. It does this in stages. Client positions are **ported** (transferred) to a healthy member wherever possible, so end-clients aren't punished for their broker's failure. The CCP then **hedges** the residual house book — putting on offsetting trades to stop the bleeding while it arranges disposal. Finally, it **auctions** the portfolio to the surviving members.

The auction is the clever, slightly brutal heart of the process. The CCP packages the defaulter's positions into lots and asks surviving members to bid on them. Members are *strongly* incentivised to bid aggressively (i.e., at prices favourable to the CCP), because — here's the twist — the guarantee fund that backstops any shortfall is *their own money*. A member who lowballs the auction, leaving a big residual loss, is helping to drain the very fund they contributed to. Many CCPs make this explicit with **juniorisation**: a member that bids poorly has its guarantee-fund contribution moved earlier in the loss-allocation order. The auction thus aligns everyone toward a fast, fair close-out.

#### Worked example: an auction that limits the survivors' bill

Continue the \$500M-loss member from before, and suppose the portfolio is auctioned. If the surviving members bid well and the realised close-out loss comes in at \$420M instead of \$500M, re-walk the waterfall:

- Defaulter's IM \$300M + fund slice \$30M = \$330M absorbed; remaining \$90M.
- CCP skin-in-the-game \$50M; remaining \$40M.
- Survivors' fund: only \$40M consumed, versus \$120M in the worse case.

A tighter auction saved the survivors \$80M of mutualised loss. **The auction is where surviving members' bidding behaviour directly sets how much of their own pooled money gets burned** — which is exactly why the rulebook gives them a sharp incentive to bid as if the loss were theirs, because it is.

## Why netting makes all of this affordable

A reasonable question: if the CCP guarantees trillions of dollars of trades, how can it possibly hold enough margin? The answer is **netting** — the CCP doesn't collateralise the gross face value of trades, only the *net* exposure after offsetting positions cancel.

![Netting shrinks gross trade obligations to a small net settlement](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-4.png)

Because the CCP is everyone's counterparty, it can net a member's buys against their sells across the entire market. If you bought \$1bn of a contract and sold \$0.98bn of the same contract through the day, your net exposure is \$20m, not \$1.98bn — and that net is what margin and the waterfall have to cover. As the chart shows, the daily *gross* obligations passing through a large clearing utility can be on the order of \$2 trillion while the *net* amount that must actually settle is a tiny fraction of it. (DTCC reports netting rates around 98% at its US equity clearing arm.) Netting is what makes the whole risk pool small enough to pre-fund. Without it, no amount of margin would be enough. The settlement side of this story — how the net obligations actually move and settle — is covered in [what happens after the trade: the post-trade lifecycle](/blog/trading/capital-markets/what-happens-after-the-trade-the-post-trade-lifecycle).

## Mutualisation: the deal members accept for novation

Step back and look at the bargain. A clearing member gives up two things. First, money: initial margin (large), variation margin (daily), and a guarantee-fund contribution (mutualised). Second, and more subtly, **the freedom to choose its counterparties** — once you clear through a CCP, you can't refuse to face a risky bank, because you face the CCP, and so does everyone else. You're now jointly liable, through the fund, for the failures of members you never chose to trade with.

![Mutualisation lets the CCP guarantee anonymous trades by pooling member resources](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-7.png)

Why would any sane institution accept that? Because of what it gets back: an **anonymous, guaranteed counterparty on every trade.** Without a CCP, every trade requires you to assess and monitor the creditworthiness of whoever's on the other side, to set bilateral credit limits, to demand bilateral collateral, and to live with the risk that your judgment was wrong. With a CCP, all of that collapses into one relationship — and the trade is guaranteed even if your real counterparty implodes. That is what makes deep, liquid, anonymous markets possible: you can hit any bid or lift any offer on the screen without knowing or caring who posted it.

This is the spine of the whole capital-markets machine showing up in the plumbing. A capital market works because **secondary-market liquidity makes primary issuance possible** — nobody funds a 30-year project unless they can sell their claim tomorrow morning. But "sell it tomorrow morning to a stranger" only works if settlement is safe. Mutualisation through a CCP is the price members pay to manufacture that safety. The guarantee fund is, quite literally, the collateral behind the promise that the market will still be there to trade in tomorrow. Closely related financing plumbing — how members fund the collateral they post — lives in [securities lending and repo: the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing).

## How big the fortress must be: "Cover 2" and stress testing

A natural question is: *how much* is enough? Margin plus the default fund could in principle be tiny or enormous; somebody has to set the dial. The global answer comes from the **Principles for Financial Market Infrastructures (PFMI)**, the standard written by international regulators after 2008, and it is refreshingly concrete. A systemically important CCP must hold pre-funded resources sufficient to withstand the default of the **two largest clearing members and their affiliates** under extreme-but-plausible market conditions. That standard is called **"Cover 2."** A smaller, less systemic CCP may only have to clear **"Cover 1"** — survive its single largest member's default.

Cover 2 is why the guarantee fund is sized the way it is. The CCP does not pick a round number; it runs **stress tests** — daily — that revalue every member's portfolio under hundreds of severe hypothetical scenarios (a 1987-style crash, a 2008-style credit shock, a 2020-style volatility spike, sharp rate moves, historical and hypothetical). For each scenario it asks: if our two biggest members defaulted *right now*, in *these* conditions, would our pre-funded resources (defaulters' margin + their fund contributions + our skin-in-the-game + the mutualised fund) cover the loss? If any plausible scenario breaks through, the CCP must raise more resources — bigger margin, a bigger fund. Regulators also require **reverse stress tests**: instead of asking "do we survive scenario X," the CCP works backwards to find the scenarios that *would* exhaust it, to understand its own breaking point.

This is the discipline that turns the waterfall from a hopeful diagram into a sized, funded fortress. It is also why CCP resources balloon in volatile periods — the same stress tests that pass in a calm market demand far more collateral when scenarios get scarier, which feeds straight into the procyclicality problem we turn to next.

#### Worked example: Cover 2 sizing

Suppose a CCP's daily stress tests show that, under its worst extreme-but-plausible scenario, its single largest member would owe \$3.2 billion beyond its own margin, and its second-largest would owe \$2.4 billion. Cover 2 requires pre-funded resources to absorb *both* defaults at once: `\$3.2bn + \$2.4bn = \$5.6 billion` available *after* the defaulters' own margin is used up. If the mutualised default fund plus the CCP's skin-in-the-game currently totals \$4.5 billion, the CCP is \$1.1 billion short of Cover 2 and must top up the fund — a call that lands on the surviving members. The intuition: "enough" is not a vibe, it is a measured number that the two-largest-member rule and daily stress tests pin down, and it moves with the market.

## Procyclicality: when the safety machine drains the system

Here is the dark side, and it is the most important idea in this post for understanding modern systemic risk. Margin is **procyclical** — it demands the *most* cash at the *worst* possible moment.

![Volatility spikes in crises drive procyclical margin calls](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-6.png)

Recall the initial-margin worked example: margin scales with volatility. In calm markets, volatility is low, so margin is low and cheap to fund. Then a crisis hits, volatility explodes (the chart shows the VIX punching from a long-run average near 20 to panic highs of 65–83 in 2020 and 2024), and the margin models — fed that fresh volatility — demand far more collateral overnight. The CCP is doing exactly what its rulebook says, and the result is a giant, synchronised cash call across the entire market on the very day cash is hardest to find. Everyone is selling assets to raise the cash to meet margin, which pushes prices down and volatility up, which triggers *more* margin. The safety mechanism becomes an amplifier.

#### Worked example: a margin call doubling overnight when volatility spikes

Return to the \$10,000,000 position. In calm markets, daily volatility was 2%, and we computed initial margin of about \$659,000. Now a crisis hits and daily volatility jumps to 4%. Re-running the same model:

- 2-day volatility: 4% × √2 ≈ 5.66%.
- 99% two-day move: 2.33 × 5.66% ≈ 13.18%.
- New initial margin: \$10,000,000 × 13.18% ≈ \$1,318,000.

The required collateral has **doubled, from \$659,000 to \$1,318,000**, overnight, with no change in the position — purely because volatility doubled. The member must find an extra \$659,000 in cash by the morning deadline, at the exact moment funding markets are seizing up. **Margin is mechanically procyclical: it asks for the most money precisely when money is scarcest** — and across a whole market, that synchronised demand is itself a source of systemic stress.

This isn't theoretical. Three episodes made "margin is the new systemic variable" the conventional wisdom among central banks:

- **The 2020 dash-for-cash.** As COVID hit in March 2020, volatility spiked and initial-margin calls across CCPs jumped by hundreds of billions of dollars in days. Funds and dealers dumped even safe assets — including US Treasuries — to raise cash for margin, breaking the normal "flight to safety" and forcing the Fed to intervene massively.
- **The 2022 UK gilt / LDI crisis.** UK pension funds running liability-driven-investment (LDI) strategies held leveraged gilt positions. When gilt yields spiked after the September 2022 "mini-budget," they faced enormous collateral and margin calls, sold gilts to meet them, drove yields higher still, and triggered more calls — a doom loop that only the Bank of England's emergency gilt-buying broke.
- **The 2022 LME nickel squeeze.** The episode that opened this post: a violent price spike turned a short position's margin call into a multi-billion-dollar demand the member couldn't meet, and the exchange's extraordinary decision to cancel trades exposed how a CCP's survival instinct can collide head-on with market integrity.

In all three, the trigger was the same shape: a price/volatility shock, a mechanical margin call, a scramble for cash, fire-sales, and feedback. The lesson regulators drew is that CCP margin models can't only be tuned to protect the CCP — they must also be *anti-procyclical*, using floors, buffers, and stress-period volatility so margin doesn't collapse in calm times only to whipsaw upward in a crisis. The classic study in *uncleared* leverage blowing up the same way is [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed); the CCP framework exists precisely to keep a modern LTCM from taking its counterparties down with it.

## Common misconceptions

**"A CCP removes risk from the market."** No — it *relocates and mutualises* it. Counterparty risk doesn't vanish; it's concentrated on the CCP, then pre-funded by margin and spread across members through the guarantee fund. The risk is still there; it's been turned from a chaotic web of bilateral exposures into one managed, collateralised pool. That's a huge improvement, but it also means the CCP itself becomes a critical single point of failure — which is why CCP recovery and resolution is now a front-line regulatory concern.

**"Initial margin and variation margin are basically the same collateral."** They do opposite jobs. VM is daily cash settlement of losses that *already happened* — it leaves your account for good and goes to the winners. IM is a buffer against losses that *might happen* during close-out — it sits as collateral and comes back to you when you close the position. You pay VM continuously and post IM once (topped up as risk changes).

**"The guarantee fund is the CCP's money."** Mostly it isn't. The bulk of the fund is *members'* contributions — mutualised capital. The CCP's own money is the comparatively thin skin-in-the-game tranche placed at layer 3, ahead of the members' fund, specifically so the CCP's incentives stay aligned. Members are the ones who ultimately backstop the system.

**"If a member defaults, surviving members are exposed to an unlimited bill."** Their exposure is capped and known in advance. The defaulter's own resources and the CCP's capital come first; the survivors' pre-funded fund is finite; and assessment powers are capped at a defined multiple of each member's contribution. The point of the waterfall is to make every member's worst case *calculable* before they ever join.

**"Big margin calls mean the system is working."** Sometimes the opposite. A margin call that's correctly sized protects the CCP — but a *synchronised, procyclical* spike in calls across the whole market can be the thing that turns a price shock into a liquidity crisis. Margin is both the shield and, in extremis, the amplifier.

**"Initial margin is money the CCP takes from me."** No — initial margin is *your* collateral, posted to the CCP and held against your positions; it is returned when you close them, and in well-run regimes client margin is segregated so it is not pooled with the defaulter's. What you actually *give up* is the use of that collateral while it sits there (its financing cost) plus your non-refundable default-fund contribution. The distinction matters in a default: the waterfall consumes the *defaulter's* margin first, never an innocent member's posted margin. Margin is a performance bond, not a fee — it is yours, parked as proof you can perform, and the only thing it truly costs you in normal times is the return you forgo on the cash and high-quality bonds you have tied up. That opportunity cost is real, which is exactly why members fight over what collateral is *eligible* (cash, Treasuries, sometimes equities at a haircut) and why the financing of margin is its own corner of the plumbing.

## How it shows up in real markets

The most liquid book is also the cheapest to close out — a fact embedded in margin and auction design. Less-liquid positions take longer to unwind and move more against you while you do it, so they attract larger margin and tougher auction haircuts. The bid-ask spread by liquidity tier is a clean proxy for that close-out cost.

![Bid-ask spread by liquidity tier in basis points](/imgs/blogs/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup-8.png)

A mega-cap stock might trade at a one-basis-point spread (you can sell a huge block almost instantly at the screen price), while a micro-cap might cost 80 basis points round-trip — and that's in calm markets. A CCP holding a defaulter's micro-cap or illiquid-derivative book faces exactly this: it can't dump it instantly without crushing the price, so the close-out window stretches, the potential loss grows, and the model demands more initial margin up front. This is why CCPs charge **liquidity add-ons** and **concentration add-ons** on top of base margin: a member holding an outsized fraction of the open interest in an illiquid contract is charged extra, because *their* book is the one that would be most expensive to auction.

The nickel case is the canonical real-world failure of this logic under stress. The defaulting producer's short position was so large relative to the market that there was no orderly way to close it out — buying back that much nickel that fast would have driven the price even higher, generating still-larger losses. The CCP's models hadn't fully captured how concentrated and illiquid the position would become in a squeeze. The "solution" — cancelling trades — protected the clearinghouse's solvency at the direct expense of the members who were *winning* on the other side and watched their gains evaporate. It was, in effect, an ad-hoc tear-up: a recovery tool deployed in real time, and a vivid demonstration that layer 5 of the waterfall is not just a footnote in a rulebook.

The 2020 and 2022 episodes round out the picture. In March 2020, the issue wasn't a member default at all — no major clearing member failed. The damage came from the *funding strain* of meeting correctly-calculated margin calls all at once. That's the modern systemic insight: even a CCP that performs flawlessly, with no default and no waterfall breach, can *contribute* to a crisis simply by demanding more cash from everyone simultaneously. The plumbing that makes individual trades safe can make the system as a whole more fragile to a liquidity shock — which is why post-2020 reform has focused less on "is the CCP solvent?" and more on "how much cash does the CCP suck out of the system in a stress, and how fast?"

## The takeaway: the price of safe, liquid, anonymous trading

Strip it all down and the CCP is a machine for converting *unpredictable, bilateral, catastrophic* counterparty risk into *predictable, pooled, survivable* losses. It does this with two walls and a backstop: margin (the defaulter's own pre-funded resources), the mutualised guarantee fund (the survivors' shared capital, reached only after the defaulter and the CCP have paid), and recovery tools (the last-resort powers for the unthinkable day). The waterfall's whole genius is *order and predictability* — every member knows, before they ever trade, exactly who pays in what sequence and what their own maximum exposure is.

That predictability is not free, and it is not magic. It is *bought* — with the collateral members post, the mutual liability they accept, and the procyclical liquidity demands they bear in a crisis. This is the deep point that ties back to the spine of the whole series: a capital market turns savings into long-term investment only because the secondary market is liquid enough that nobody fears being trapped in their claim. That liquidity rests on the ability to trade with anonymous strangers and walk away settled. The margin and the waterfall are what manufacture that ability. They are, quite precisely, **the price of safe, liquid, anonymous trading** — paid up front, in collateral and mutual liability, so the market can stay open the morning after a member blows up.

The next time you read that a CCP "made markets safer," hold both halves of the truth in your head. It did — by relocating risk into a managed, pre-funded pool that absorbs a default quietly. And it created a new one — a critical node that, in a violent enough shock, calls for cash from everyone at once. Understanding the waterfall is understanding both how the modern market survives its blowups and why the next crisis might run straight through the very plumbing built to prevent it.

## Further reading & cross-links

- [The clearinghouse: how a CCP removes counterparty risk](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk) — novation and the basics of how a CCP inserts itself between buyer and seller (the foundation this post builds on).
- [What happens after the trade: the post-trade lifecycle](/blog/trading/capital-markets/what-happens-after-the-trade-the-post-trade-lifecycle) — how cleared net obligations actually clear and settle.
- [Securities lending and repo: the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing) — how members fund the collateral they must post as margin.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — where exchanges and CCPs sit in the market structure.
- [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — leverage and margin spirals in an *uncleared* world; the cautionary tale the CCP framework exists to prevent.
