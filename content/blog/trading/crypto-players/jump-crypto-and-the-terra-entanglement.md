---
title: "Jump Crypto and the Terra Entanglement: When a Market Maker Becomes the Backstop"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero look at Jump Crypto — a prop-trading giant that quotes markets, builds blockchain infrastructure, and can write a nine-figure cheque from its own balance sheet. How it replaced $320M after the Wormhole hack, what the SEC's settled order says about its role in the May 2021 UST repeg, and the three tests that separate ordinary liquidity from directed price support. With worked dollar examples and careful, sourced framing."
tags: ["crypto", "market-makers", "jump-crypto", "terra", "ust", "wormhole", "prop-trading", "crypto-players", "stablecoins", "case-study"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 27
---

> [!important]
> **TL;DR** — Jump Crypto is the digital-asset arm of **Jump Trading Group**, a Chicago proprietary trading firm founded in 1999 that trades only its *own* capital. That last fact is the whole story: because no outside investor can ever redeem Jump's money, Jump can do something ordinary market makers cannot — write a nine-figure cheque, from its own balance sheet, in hours, to backstop a market it is involved in.
>
> - **Jump wears three hats over the same assets.** It is a high-speed **market maker**, it builds critical **infrastructure** (the Wormhole bridge, the Firedancer validator client for Solana), and it is a **principal investor** with permanent capital. Each is legitimate; together they mean the firm that quotes a token can also be the firm that wrote the rails it settles on and holds the reserve that defends it.
> - **The Wormhole backstop is the clean case.** When the Wormhole bridge was exploited on 2 February 2022 for about 120,000 ETH (worth over $320M at the time), Jump replaced the funds within roughly a day. Only a permanent-capital prop firm can move that fast, because there are no LPs to consult and nothing to redeem.
> - **The Terra case is the contested one.** Per the SEC's **settled order of 20 December 2024**, a Jump-affiliated entity, **Tai Mo Shan**, played a role in the May 2021 recovery of the UST stablecoin's dollar peg — a repeg the public read as the algorithm self-healing. Tai Mo Shan settled for **$123.1 million total, without admitting or denying** the findings.
> - **The line that matters is liquidity versus support.** Buying a falling asset is ordinary market making — until it is *undisclosed*, *compensated in the asset being defended*, and *load-bearing* all at once. Those three tests are how you tell one from the other.
> - **Count the hats before you size the trade.** When one firm quotes a token, wrote its bridge, and holds a defence reserve, the token's price, rails, and float share a single point of failure. That is a concentration risk you can actually measure.

Every other named market maker in this series makes money by quoting prices. Jump is the one that also builds the road the trade drives on and keeps a fire truck in the garage. It is the most *vertically integrated* player in crypto — a firm that can, over the same token, set the market price, ship the code that settles the transfer, and deploy its own permanent capital to defend a level. None of those activities is wrong on its own. The question this post is built around is what happens when one firm does all three at once, and how you, on the other side of the trade, can reason about it without either excusing it or inventing a conspiracy.

I am going to be scrupulous about the difference between what is *documented* and what is *characterized*, because this is one of the most legally sensitive stories in the series. The Wormhole replacement is a matter of public record and, frankly, looks admirable. The Terra involvement is the subject of a **settled SEC order** that Jump's affiliate resolved *without admitting or denying* the findings — so I will describe what the order says, attribute it to the order, and not assert intent beyond it. The point is not to accuse Jump of anything; it is to use a well-documented case to teach the mechanics of how a market maker's balance sheet can become a price-support mechanism, and how to spot the pattern.

## Foundations: three hats over the same token

Jump Trading Group is not a crypto-native company. It is a serious, secretive Chicago **proprietary trading firm** founded in **1999**, one of the giants of traditional high-frequency and futures trading, and it trades exclusively its own money — no outside limited partners, no fund investors, no redemption gates. **Jump Crypto** is the digital-asset division that carried that DNA into crypto around 2021. To understand everything that follows, you have to hold three roles in your head at once.

![Jump wears three hats over the same token: a high-frequency market maker setting the fill price, an infrastructure builder shipping the Wormhole bridge and Firedancer client the trade settles on, and a permanent-capital principal able to write a nine-figure discretionary backstop.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-1.webp)

- **Hat 1 — market maker.** Jump quotes two-sided markets at high-frequency speed, capturing tiny spreads across millions of fills. This sets the **fill price you actually get**.
- **Hat 2 — infrastructure builder.** Jump acquired and developed the **Wormhole** cross-chain bridge and builds **Firedancer**, a high-performance validator client for Solana. This is the **protocol code and uptime** — the rails your trade settles on.
- **Hat 3 — principal capital.** Jump invests and can deploy its **own permanent capital** with no outside LPs and nothing to redeem. This is the **discretionary backstop** — the ability to, for example, replace 120,000 ETH in February 2022.

It helps to appreciate where Jump came from, because its crypto behavior is a direct import of its traditional-markets DNA. Jump Trading is one of the most secretive and successful high-frequency firms in the world, built on microsecond-latency trading of futures and equities, famous for microwave-tower networks between exchanges and for saying almost nothing in public. That culture — extreme technical capability, permanent private capital, and a deep aversion to disclosure — is not a crypto quirk; it is how the firm has always operated. When such a firm enters crypto, it does not become a chatty, announcement-driven venture market maker like the newcomers profiled elsewhere in this series. It does what it has always done: build the fastest infrastructure, trade its own money, and keep quiet. The secrecy that is normal and legal for a private prop firm is precisely what makes the *undisclosed* dimension of the Terra case so natural to how Jump works — not as an accusation, but as an observation about how a firm with no reporting obligations behaves by default.

The reason these three hats are more consequential together than apart is that they touch *different layers of the same trade*. Hat 1 sets your price. Hat 2 controls whether your transfer settles at all. Hat 3 can step into the market to move the price on purpose. A firm wearing all three over the same token is not just a large trader; it is a participant at the price layer, the settlement layer, and the capital layer simultaneously. Whether that is reassuring (a deep-pocketed backstop) or concerning (a single point of failure) depends entirely on the specifics — which is exactly why the specifics are worth learning. And notice how the three hats *reinforce* one another: the market-making flow tells the firm what is happening in real time, the infrastructure tells it what is settling, and the permanent capital lets it act on both faster than anyone constrained by outside investors. Information, control, and capital compounding together is what makes a firm like Jump formidable — and what makes counting its hats a matter of your own risk, not idle curiosity.

## How an algorithmic stablecoin was supposed to work

To understand the Terra entanglement, you first need to understand what UST was and why it was fragile, because the whole episode is about defending a mechanism that could not defend itself.

**UST** (TerraUSD) was an **algorithmic stablecoin**: a token that aimed to hold a value of **$1.00** with **no cash or bond reserve** behind it. Instead of reserves, its peg rested entirely on an arbitrage loop with its sister token, **LUNA**, and on traders finding that loop profitable.

![How an algorithmic stablecoin was supposed to hold a dollar: with no cash reserve, UST's peg relied on a mint-and-burn arbitrage against LUNA that only worked as long as LUNA itself held value.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-2.webp)

### Worked example 1 — the mint-and-burn arbitrage

The protocol let anyone swap $1 of LUNA for 1 UST and vice versa, always treating UST as worth exactly $1 in the swap. That created two correcting trades:

- **When UST trades above $1** — say **$1.02** — you **burn $1 of LUNA, mint 1 UST, and sell it for $1.02**, pocketing $0.02. Everyone doing this *increases UST supply*, pushing its price back down toward $1.
- **When UST trades below $1** — say **$0.98** — you **buy UST for $0.98, burn it, and mint $1 of LUNA**, pocketing $0.02. Everyone doing this *decreases UST supply*, pushing its price back up toward $1.

On paper, self-healing. But look at the load-bearing assumption: the below-peg trade requires **minting $1 of LUNA** for every UST redeemed. That only works if **LUNA holds its value** while it absorbs the selling. If confidence cracks and UST falls hard, the protocol mints ever more LUNA to honor the swaps, LUNA's price collapses under the new supply, and the "arbitrage" that was supposed to restore the peg instead becomes a **death spiral** — each redemption printing more of a token that is falling, which destroys the very value the peg depends on. That is precisely what happened, catastrophically, in May 2022. But a year earlier, in May 2021, UST had a *smaller* wobble — and that smaller wobble is where Jump enters the story.

There is one more ingredient that made the stakes enormous: **demand for UST was manufactured by a yield.** The Terra ecosystem's Anchor Protocol offered holders of UST an advertised return of around **20% a year** — a rate with no sustainable source, effectively subsidized to pull capital in. That yield is why UST ballooned from a curiosity to tens of billions of dollars in supply: people were not holding it because they believed in algorithmic stablecoins; they were holding it to farm 20%. And that matters for the peg-defence story because **the more UST there was, the more catastrophic a failed peg would be, and the more valuable it was to anyone involved to keep the "it self-heals" narrative intact.** A stablecoin whose growth depends on a story about its own resilience is exquisitely sensitive to that story — which is exactly the kind of narrative a well-timed, undisclosed defence can protect. The reflexivity is the whole point: confidence drove demand, demand drove supply, and supply raised the cost of ever letting confidence break.

## Why only a prop firm can write the backstop cheque

Before the Terra case, understand the structural fact that makes Jump able to play backstop at all, because it is the same fact that makes the Wormhole rescue possible and the Terra involvement plausible: **permanent capital.**

![A comparison showing that a prop firm's own permanent capital — with no outside LPs and nothing to redeem — is what lets it commit hundreds of millions in hours, where a hedge fund answering to investors cannot.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-3.webp)

Ask four questions of any firm that might step in to defend a market:

- **Whose money is it?** For a prop firm like Jump, it is its *own* capital — 100% permanent. For a hedge fund, it is outside LPs' money.
- **Can it be pulled?** Jump's cannot: there are no outside LPs, nothing to redeem, no gate that can be triggered by a panic. A hedge fund's capital can be redeemed out from under it exactly when it is most needed.
- **How fast can it commit $320M?** For Jump, *hours* — one internal decision. For a fund, days or weeks of investor consultation and risk committees.
- **What must it disclose?** For a private partnership like Jump, minimal — no public filings about most of what it does.

Stack those up and the conclusion is stark: **capital that nobody can redeem is what turns a market maker into a lender of last resort.** A hedge fund cannot credibly promise to defend a token, because its own investors could yank the money mid-defence. A permanent-capital prop firm can. This is the superpower — and, as we will see, the entanglement — that Jump's structure creates. It is why, when a nine-figure hole opened in the Wormhole bridge, Jump could fill it faster than any committee-bound institution on Earth.

## Five ways a crypto prop desk gets paid

Jump's structure also shapes *how* it earns, and this is where the entanglement starts to become visible, because most of its revenue channels pay it in the very assets it also quotes.

![Five revenue channels for a crypto prop desk — spread capture and cross-venue arbitrage paid in cash, versus token market-making deals, principal investing, and infrastructure equity that pay in the assets the desk also quotes.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-4.webp)

1. **Spread capture.** Two-sided quotes at HFT speed — fractions of a basis point, multiplied by millions of fills. **Paid in cash and stablecoins.**
2. **Cross-venue arbitrage.** The same token at two prices across venues or chains; latency and inventory are the moat. **Paid in cash and stablecoins.**
3. **Token market-making deals.** An issuer lends the desk tokens plus a **call option** struck near the launch price; the upside on that option is the real fee. **Paid in the issuer's token.**
4. **Principal investing.** Seed and Series A equity plus token warrants that vest on a schedule. **Paid in equity and tokens.**
5. **Infrastructure equity.** Owning the bridge (Wormhole) or the validator client (Firedancer) — effectively **an option on the whole chain**. **Paid in equity and influence.**

Only the first two are pure spread capture, cleanly separated from the assets being quoted. Channels 3, 4, and 5 pay Jump **in the assets it also makes markets in** — which means, for a growing share of its book, *revenue and inventory stop being separable, and so do liquidity and support.* When your compensation for supporting a token is a discounted position in that token, "provide liquidity" and "defend the price so my position appreciates" begin to point in the same direction. Hold that thought; it is the exact structure the SEC's Terra order describes.

### Worked example 2 — why the option, not the spread, is the real fee

Suppose a desk does a token market-making deal: the issuer lends it tokens and grants a **call option on 10 million tokens struck at $1.00**, the launch price. The desk's *quoting* profit — the spread it earns making markets — might be a few hundred thousand dollars over the deal's life. Trivial. Now suppose the token, with the desk's liquidity and the issuer's marketing, runs to **$4.00**. That option is now worth roughly (4.00 − 1.00) × 10,000,000 = **$30 million**. The spread was a rounding error; the *option* was the payday. This is why token market-making deals are so consequential: they align the desk's largest incentive not with tight two-sided quoting, but with the token going *up*. A desk paid this way is not neutral on direction — its biggest cheque depends on the very price it is quoting.

## The clean case: Wormhole, built, broken, and backstopped

Now the well-documented, admirable case — the one that shows the backstop power at its best, and also shows the vertical integration in its purest form.

![A timeline of Wormhole from Certus One's first Solana-Ethereum bridge in 2020, through Jump's acquisition, the February 2022 exploit that minted 120,000 wETH with no collateral, Jump's replacement of the funds a day later, a 2023 English court recovery order, and Wormhole's spin-out at a $2.5B valuation.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-6.webp)

The sequence is public record:

- **October 2020** — a team called **Certus One** ships the first Solana–Ethereum token bridge.
- **2021** — **Jump Crypto acquires Certus One**; Wormhole v2 launches in August 2021. Jump now owns critical cross-chain infrastructure.
- **2 February 2022** — Wormhole is **exploited**: an attacker mints **120,000 wETH** on Solana with **no collateral** backing it, draining the bridge.
- **3 February 2022** — **Jump replaces the 120,000 ETH**, worth **over $320 million** at the time, from its own balance sheet — within roughly a day.
- **21 February 2023** — an English court order moves recovered assets (120,695 wstETH and 3,213 rETH) to Jump, partially recouping the backstop.
- **29 November 2023** — Wormhole **raises $225M at a $2.5B valuation** and spins off as an independent entity.

### Worked example 3 — the anatomy of a $320M backstop

Why did Jump replace the funds at all? It was not legally obliged to; the exploit was the bridge's, not Jump's. The answer is a hard-nosed calculation as much as a noble one. If the 120,000 ETH hole were left open, every user of Wormhole would be underwater, confidence in the bridge — and in Solana's cross-chain liquidity, which Jump had invested heavily in — would collapse, and Jump's much larger positions in the *entire Solana ecosystem* would fall with it. Replacing $320M to protect a multi-billion-dollar franchise is, coldly, a good trade. And crucially, **only Jump could make it**: no fund-of-outside-money could deploy $320M of discretionary capital in 24 hours. The Wormhole rescue is the permanent-capital superpower used in the way everyone applauds — and it is *also* the clearest possible illustration of the vertical integration, because the same firm that **wrote the bridge's code** also **wrote the cheque that refilled it**. That is the model working at its best; the Terra case is the same model in a far more contested light.

It is worth dwelling on *why bridges are the choke point*, because Hat 2 — infrastructure — is the least understood of the three and the most powerful. A cross-chain bridge holds real assets on one chain and issues "wrapped" claims on another; the wrapped token is only as good as the collateral locked in the bridge. That makes bridges the largest honeypots in crypto — several of the biggest hacks in history have been bridge exploits — and it makes whoever *controls* the bridge a party to every cross-chain transaction that flows through it. Owning Wormhole is not just owning a product; it is owning a toll road between Solana and the rest of crypto, with visibility into the flow and the ability, in principle, to influence its uptime. That is what the figure means by "an option on the whole chain": infrastructure ownership converts a firm from a participant in a market into a *dependency* of it. When the bridge broke, that dependency became an obligation — Jump effectively had to make users whole to preserve the value of everything else it owned downstream. The backstop was noble and self-interested at once, and the two are not in tension; that alignment is precisely what made it credible.

The deeper takeaway is that **infrastructure ownership is a different kind of power than quoting size.** A market maker that pulls its quotes inconveniences you; a firm that controls the settlement rail can, at the limit, determine whether your transaction happens at all. Most of the time this power is dormant and benign — the rails just work, which is the point of building them well. But when you are assessing concentration, the firm that wrote the code sits at a layer beneath the price, and a risk at that layer is not something tighter spreads can offset. It is why "who wrote the code?" is its own question on the concentration map, separate from "who quotes it?"

## Four channels from a desk to the printed price

Before the Terra case, one more piece of scaffolding: the ways a market maker's activity actually reaches the candle you see on the chart. Three of these are ordinary; the fourth is the one the Terra case turns on.

![Four channels by which a market maker's desk reaches the printed price: spread and depth set slippage, inventory unwinds hit the same book, quote withdrawal gaps the spread, and directed support is deliberate buying to defend a chosen level.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-5.webp)

- **Channel 1 — spread and depth.** The desk's quotes set your slippage on every fill. Ordinary.
- **Channel 2 — inventory unwind.** When the desk hedges or works out of a position, that flow hits the same book. Ordinary.
- **Channel 3 — quote withdrawal.** The desk pulls its quotes and the spread gaps wide — often right before news. Ordinary (if unhelpful to you).
- **Channel 4 — directed support.** The desk *buys to defend a chosen price level* — not to manage inventory, but to hold a line. This is the one that stops being neutral market making.

Channels 1–3 are what every market maker does all day; they are the mechanics the rest of this series describes. Channel 4 is different in *kind*: it is buying with the intent to move or hold the price at a target, and whether it is benign smoothing or something more depends on the three tests we will reach shortly. The Terra case is a Channel-4 story.

## The contested case: May 2021, the story and the settled record

Here is where I am most careful. In **May 2021**, a year before Terra's final collapse, UST briefly **slipped under $1 and then climbed back to peg**. To the market, this looked like proof the design worked: the mint-and-burn algorithm had absorbed a shock and self-healed, unaided. Terraform publicly framed it as surviving a black-swan event. Investors, reasonably, priced UST going forward as *a design that self-heals under stress* — which made them more willing to hold it, and helped it grow to the tens of billions that made 2022's collapse so devastating.

![Two readings of the May 2021 UST repeg: what the market could see — a stablecoin that dipped and recovered on its own — against what the SEC's settled order of December 2024 describes, an arranged purchase of UST by a Jump affiliate compensated with a discounted option on LUNA.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-7.webp)

According to the **SEC's settled order dated 20 December 2024**, the picture underneath was different from the one the market saw. Per that order: Terraform and **Tai Mo Shan Limited** (an entity affiliated with Jump) agreed that Tai Mo Shan would **buy UST** to help restore the peg; Tai Mo Shan purchased **more than $20 million of UST**; and as consideration, Terraform **vested an option for Tai Mo Shan to buy LUNA at a discount**. In other words — as the settled order describes it — the repeg that the market read as the algorithm working unaided involved a **compensated purchase** by a party paid in a **discounted option on the volatile sister token**. Tai Mo Shan **settled for $123.1 million total**, resolving the SEC's charges (which also concerned unregistered sales of LUNA as a security) **without admitting or denying** the findings.

I want to state the limits of that carefully. A settled order is not a trial verdict; settling *without admitting or denying* is standard and is not a confession. What the order establishes is the SEC's account and the settlement amount, not a jury's finding of intent. But even taken purely as the regulator's documented account, it is enough to teach the mechanism, because the mechanism is the point: **the same repeg reads as "the algorithm worked" or as "a paid intervention," depending on what you were told** — and the market, in 2021, was told the first.

## Where a peg-defence package actually pays

The economic heart of the Terra case is why a firm would take such a deal, and the answer is the same as Worked Example 2: **not the spread on the defence, but the discounted option on the volatile token.**

![A payoff chart of a peg-defence package showing that the buy-and-hold defence leg is small and roughly flat, while the discounted option on the volatile token carries all the upside as that token's price rises.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-8.webp)

### Worked example 4 — decomposing the package

Split a hypothetical peg-defence package into its two legs, in the spirit of what the SEC order describes (numbers illustrative, to show the shape):

- **Leg 1 — the defence.** Buy **$20 million of UST at $0.98** while the peg is broken. If it stays broken and you mark the UST at, say, **$0.92**, this leg is *down about $1.2 million*. Small, flat, and — on its own — almost irrelevant, even a modest loss. If you were only paid this way, you would never take the deal.
- **Leg 2 — the option.** A **discounted option on LUNA**, the volatile token. If the defence succeeds and confidence returns, LUNA appreciates, and the option's payoff can run to **+$40 million, +$80 million, or more** as the token's price rises.

Add the legs and the package's entire profit lives in **Leg 2**. The defence leg is essentially the *price of admission* to the option; the option is the reward. This is why "peg defence" and "market making" are economically distinct even when they look identical on the tape: a genuine market maker is roughly flat on direction, while a firm paid in a discounted option on the defended ecosystem has a large, convex bet on that ecosystem going *up*. The payoff structure, not the buying itself, is what turns liquidity provision into something with a direction.

## Three tests: liquidity or support?

So how do you, without a subpoena, tell ordinary market making from directed price support? The honest answer is that from the outside you often *cannot* be certain — but you can apply three tests that, together, separate the two, and all three have to be true at once for "support" to be the right word.

![Three tests separating liquidity from support: whether the buying is visible to the market, whether the buyer is paid in the asset being defended, and whether the mechanism would have held without the buyer.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-9.webp)

1. **Is the buying visible?** *Disclosed* support — a named backstop the market can price in — is honest; everyone knows the floor is held by a party, not the design. *Undisclosed* buying reads on the tape as **organic demand**, and that misreading is the harm: investors attribute to the mechanism what actually came from a buyer.
2. **Is the buyer paid in the asset being defended?** Paid in **cash**, a flat fee keeps the buyer's incentives apart from the price. Paid in **tokens** (or a discounted option on them), the buyer has a reason to **defend now and distribute later** — its compensation *is* the appreciation it is helping to manufacture.
3. **Would the mechanism have held alone?** If the design would have self-corrected and the buyer only smoothed the path, the track record still belongs to the mechanism. If it would **not** have held without the buying, then the "proof the algorithm works" actually belongs to the buyer — and everyone who bought UST *because* they believed the algorithm was tested was misled about whose money did the work.

Ordinary market making fails at least one of these — it is usually visible-enough, cash-paid, or non-load-bearing. "Support" in the concerning sense is when all three line up: **undisclosed, compensated in the asset, and load-bearing.** The May 2021 repeg, as the settled order describes it, is a case where a reasonable person can see all three tests pointing the same way — which is exactly why it is the canonical teaching example, whatever one concludes about intent.

Of the three, **disclosure is the one that does the real work**, because it is the difference between a floor and a fraud. There is nothing wrong with a *disclosed* backstop — traditional finance is full of them, from central-bank liquidity facilities to a company's announced share-buyback program. When everyone knows a party stands ready to buy at a level, the market prices that in; the floor is real, but so is everyone's knowledge of who holds it. The harm in the undisclosed case is specific and identifiable: **the investor who bought UST in late 2021 because they believed the May repeg had proven the design.** That person made a decision on a false premise — that a mechanism had passed a stress test it may not have passed alone — and they could not have known otherwise, because the intervention was not visible. When the design finally failed for real in May 2022, that investor discovered the "track record" they had trusted may have belonged, at least partly, to a buyer rather than to the algorithm. Disclosure is what would have let them price the risk correctly; its absence is what left them holding a belief that was never theirs to hold.

This is why "it was just providing liquidity" is not a complete defense and "it was manipulation" is not a fair conclusion — the honest description lives in the tests. A firm can buy a falling asset for entirely ordinary reasons a thousand times a day. The concern arises only in the narrow intersection where the buying is hidden, the buyer is paid in the thing it is defending, and the buying is what holds the line. That intersection is rare, but it is exactly where the most damaging misreadings of the market get manufactured, and it is worth being able to recognize precisely because it looks, on the chart, identical to health.

## What it means if you're on the other side

You cannot audit a prop firm's balance sheet, and you will not get the settled order until years later. What you *can* do is count the hats before you size the trade — turn a vague unease about "who's behind this token" into a countable exposure.

![A six-question concentration map: who quotes the token, who wrote its code, who holds its float, whether anyone is paid in it, what is disclosed, and how many rows a single firm occupies.](/imgs/blogs/jump-crypto-and-the-terra-entanglement-10.webp)

1. **Who quotes it?** Name the market makers. If depth collapses when one desk pauses, your exit price depends on that desk's risk appetite, not the token's fundamentals.
2. **Who wrote the code?** Bridges, validator clients, and oracles are choke points. Ask who ships them and who could stop them.
3. **Who holds the tokens?** Market-making loans, options, and venture allocations are all **future supply with a known holder** — an overhang waiting to be distributed.
4. **Is anyone paid in it?** A counterparty compensated in the asset it defends has a reason to defend now and distribute later. That is the Terra pattern in one line.
5. **What is disclosed?** Compare the project's public story against filings, court records, and settled regulatory orders. The gap between the two is your risk.
6. **Score the overlap.** One firm in one row is normal and healthy — that is just a market maker doing its job. One firm in **four rows** means the token's **price, rails, and float share a single point of failure.**

The scoring becomes concrete through a contrast. One token is quoted by six independent desks, its bridge is an open-source project run by a foundation, its float is spread across thousands of holders, and no market maker is paid in it. A second token is quoted mainly by one firm, whose affiliate wrote its bridge, holds a large option position in it from a market-making deal, and sits on a venture allocation of its float. Both might trade at the same price with the same chart. But the first token's price is an emergent property of many participants, while the second's is, to a large degree, one firm's decision. If that firm's risk appetite changes — a bad quarter elsewhere, a regulatory inquiry, a strategic exit — the second token can gap in a way the first structurally cannot. Same price, entirely different fragility, and the concentration map is what lets you tell them apart before you are the one holding the position when the single participant steps back.

The purpose of this exercise is not to demonize integration — a deep-pocketed backstop genuinely saved Wormhole's users, and vertical integration can make an ecosystem more robust, not less. The purpose is to make the concentration *legible* so you can price it. When you can name a single firm that quotes the market, wrote the bridge, holds a large slice of the float, and is paid in the token, you are not looking at a diversified market with many independent participants; you are looking at one balance sheet with several roles, and your position rests on that balance sheet's continued willingness to play all of them in your favor. That might be fine. But it is a bet on a firm, not on a mechanism — and the whole lesson of Terra is that the market, in 2021, thought it was betting on a mechanism when it was, at least in part, betting on a firm.

Jump is, by most accounts, one of the most capable trading firms in the world, and the Wormhole backstop is a genuinely good deed with a hard-nosed logic behind it. This post is not an indictment of Jump; it is a use of the best-documented case available to teach a pattern that recurs across the whole market. The pattern is simply this: when revenue is paid in the asset you also quote, liquidity and support stop being separable — and the only defense available to the person on the other side is to count the hats before deciding how much to trust the price.

It is worth ending on the symmetry the two cases reveal, because it is the real lesson. The Wormhole rescue and the Terra intervention are, structurally, the *same act*: a permanent-capital firm using its own balance sheet to hold up a market it is deeply involved in. What separates them is not the mechanism but the three tests — Wormhole was disclosed, celebrated, and manifestly a backstop everyone could see; the May 2021 repeg, per the settled order, was none of those things at the time. The identical capability that let Jump do something admirable in February 2022 is what let a related structure become the subject of a $123.1 million settlement for May 2021. That is why the capability itself is neutral and the *disclosure* is everything. A firm powerful enough to save a market is, by definition, powerful enough to move one quietly — and the only thing that reliably tells the public which is happening is whether the firm, or a regulator years later, chose to say so.

## Sources & further reading

- U.S. Securities and Exchange Commission, settled order concerning **Tai Mo Shan Limited**, 20 December 2024 (the $123.1 million settlement; role in the May 2021 UST repeg and unregistered LUNA sales; resolved without admitting or denying the findings).
- Public reporting and Jump Crypto statements on the **Wormhole exploit** of 2 February 2022 and the ~120,000 ETH (>$320M) replacement on 3 February 2022.
- English High Court recovery order concerning Wormhole-related assets (2023); Wormhole's $225M raise at a $2.5B valuation and spin-out (November 2023).
- Background on **Jump Trading Group** (founded 1999), Jump Crypto, and the **Firedancer** Solana validator client.
- General references on algorithmic stablecoins, the UST/LUNA mint-and-burn mechanism, and the May 2022 Terra collapse.
- Related posts in this series: [The Loan-Plus-Options Deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid), [Wash Trading, Spoofing, and Manufactured Volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume), [Alameda Research: The Cautionary Tale](/blog/trading/crypto-players/alameda-research-the-cautionary-tale), and the hub, [Crypto VCs and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers).

*Nothing here is legal or investment advice. The Terra/Tai Mo Shan matter is described per the SEC's settled order, which was resolved without any admission or denial of the findings; nothing here should be read as asserting intent beyond that record. Dollar figures in the worked examples are illustrative and rounded to show the mechanics.*
