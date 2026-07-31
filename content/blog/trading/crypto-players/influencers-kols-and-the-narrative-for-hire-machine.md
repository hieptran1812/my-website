---
title: "Influencers, KOLs, and the narrative-for-hire machine"
date: "2026-07-29"
publishDate: "2026-07-29"
description: "How crypto narrative is manufactured and paid for: the actual commercial structures behind promotion, why a small float turns a posting window into a price move, and how to read a promotion before you act on it."
tags: ["crypto", "influencers", "kols", "tokenomics", "market-structure", "disclosure", "securities-regulation", "low-float", "retail-defense", "due-diligence", "crypto-players"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 51
---

> [!important]
> **TL;DR** — Crypto narrative is a product with a price list, and the people selling it are usually paid in the thing they are describing.
>
> - "Influencer marketing" in crypto is not one business but at least seven distinct commercial structures, ranging from a disclosed cash fee for a post to a discounted token allocation with a shorter lockup than the public got. The structures worth the most to the promoter are the ones the audience can see the least.
> - The conversion from attention to price runs through **float**, not persuasion. A token with 12% of its supply actually trading needs only a small amount of new buying to move a lot — in the worked example below, \$500,000 lifts the price 13.8%, while the same order moves a deep-float token 0.2%.
> - The asymmetry is the whole story. A promoter holding tokens at \$0.02 and an audience buying at \$0.12 are not in the same trade: at a 67% drawdown the promoter is still up 100% and the audience is down 67%.
> - The disclosure most crypto promotions omit is **already legally required in the United States** — Securities Act Section 17(b) has demanded it since 1933, and the FTC's Endorsement Guides demand it of any endorser with a material connection.
> - The number to remember: in the illustrative composite used throughout this post, a coordinated posting window generates **\$500,000** of retail buying against **\$1,500,000** of promoter supply that unlocked the same morning. The window does not create demand. It creates an exit.

There is a price for a tweet. Not a metaphorical price — an actual number, quoted in an actual currency, on an actual rate card, negotiated by an actual agency that will invoice for it. The same is true for a YouTube segment, a Telegram "call", a podcast appearance, a Discord announcement, and a piece of writing labelled "research". None of this is unusual or secret in the abstract. Every media business in history has sold attention. What makes crypto specific is that the payment is frequently made in the very asset being described, at a price the audience cannot get, with a lockup the audience does not have, on a schedule the audience cannot see.

That single fact reorganises everything. When a newspaper takes an ad, the newspaper's revenue does not depend on whether readers buy the advertised car. When a promoter is paid in tokens at \$0.02 and the audience buys at \$0.12, the promoter's revenue depends entirely on whether the audience buys — and on how many of them buy at once, and on how quickly. Attention stops being a marketing expense and becomes a supply-distribution mechanism.

This post builds that machine from zero. What the promoters actually are, what they are actually paid and in what form, how paid attention becomes a price move, why the mathematics of a small float does most of the work, what the law already requires and what the enforcement record actually shows, and — the half that matters most if you are on the receiving end — how to read a promotion and how to audit a track record honestly.

![The narrative-for-hire machine: project money buys promoter attention, attention buys retail flow, and retail flow is the liquidity that the cheapest supply sells into](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-1.webp)

The diagram above is the mental model, and it is worth sitting with before any of the detail. Money enters at the left as a promotion budget. It converts to attention in the middle. Attention converts to buying on the right. And the buying meets supply — supply held by whoever acquired it cheapest and became free to sell first. Every section that follows is a closer look at one arrow in that picture.

A note on how this post handles numbers, because the topic invites carelessness. Any figure attached to a real person, a real company, or a real regulatory action is sourced and dated, and appears only where it is a matter of public record. Everything else — the token, the allocation sizes, the order book, the track record — is an **illustrative composite** built from round numbers so you can check the arithmetic yourself. I will say "illustrative composite" every time, because the difference matters. The mechanics are real; the specific token is not.

## Foundations: the building blocks

If you have never bought a token, none of what follows will make sense without these five ideas. If you have, skim — but check the definition of float, because the entire post turns on it.

### What a "KOL" actually is

**KOL** stands for *key opinion leader*, a term borrowed from consumer marketing in Asia and now standard across crypto. Functionally it means: an account whose audience will act on what it says. That is the only property that matters commercially. It is not a claim about expertise, accuracy, or honesty — it is a claim about conversion.

The label covers at least five different businesses that happen to share a distribution channel.

![Five kinds of promoter, what each one's audience believes it is buying, how each is usually paid, and whether any of it can be verified](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-2.webp)

- **The large-account trader.** Posts entries, exits, and P&L screenshots. The audience believes it is buying a track record. In practice the track record is almost never a complete ledger — more on this in the audit section, which is the most useful part of this post.
- **The video or podcast host.** Long-form, produced, often genuinely informative. The audience believes it is buying research. This is the group most likely to take straightforward cash for a segment, and also the group where disclosure is most likely to exist in some form.
- **The group operator.** Runs a Telegram or Discord channel, sometimes paid, sometimes tiered. The audience believes it is buying early access. The commercial structure here frequently includes both an allocation and a referral arrangement, and the channel is private, so nothing is externally auditable.
- **The anonymous account.** No legal identity, no jurisdiction, no reputation that survives a name change. The audience believes it is buying insider information. There is no contract to breach and no regulator with an obvious hook.
- **The regional-language influencer.** Builds trust inside a language community — Vietnamese, Turkish, Indonesian, Portuguese, Korean. The audience believes it is buying a trusted local voice. This tier is structurally important because it reaches buyers that English-language enforcement and English-language scepticism both miss.

None of these are inherently illegitimate. A paid, disclosed sponsorship on a produced video is an ordinary media transaction. The problems begin where the payment form creates a position and the position is not disclosed.

### Float: the number that does the work

Here is the single most important concept in the post, and it takes sixty seconds to learn.

A token has a **total supply** — every token that will ever exist. It also has a **circulating supply**, usually called the **float**: the tokens that are actually free to trade right now. The rest are locked — held by the team, by early investors, by the foundation, under contracts that release them over months or years.

Those two numbers give you two different measures of size:

- **Circulating market cap** = float × price. What the tradeable portion is worth.
- **Fully diluted valuation (FDV)** = total supply × price. What the whole supply would be worth if it were all trading at today's price.

Take the illustrative composite token used throughout this post. Call it **TKN**.

- Total supply: **1,000,000,000** tokens.
- Circulating at listing: **12%**, or **120,000,000** tokens.
- Opening price: **\$0.12**.

So circulating market cap is 120,000,000 × \$0.12 = **\$14,400,000**. FDV is 1,000,000,000 × \$0.12 = **\$120,000,000**. The ratio is 8.3 to 1.

Read that ratio slowly, because it is the mechanism. The price of \$0.12 was discovered by trading against \$14.4 million of available supply. But that price is now applied to all one billion tokens. The 88% of supply that is locked is being *marked* at a price it never had to clear. And that locked supply is not gone — it is a queue, with a published release schedule, waiting to sell into whoever is buying later.

> A small float does not mean a small position. It means a small door.

### The order book, in one paragraph

When you place a market buy on an exchange, you are not buying at "the price". You are consuming the **resting sell orders** — the offers other people have already placed, stacked upward from the current price. The cheapest offers fill first. When they are exhausted, your order climbs to the next-cheapest, and so on. **Market depth** is how much sits within some distance of the current price; **slippage** is the gap between the price you saw and the average price you actually paid. A thin book means a small order climbs a long way. This is not a crypto quirk — it is how every order-driven market works, and Albert Kyle's 1985 model of price impact formalised it for equities decades before crypto existed. Crypto just runs the experiment with much thinner books.

### Vesting, cliffs, and TGE

**TGE** is the *token generation event* — the moment the token exists and usually starts trading. **Vesting** is the schedule on which locked tokens become sellable. A **cliff** is a period during which nothing releases at all, followed by a lump. "12-month cliff, then 24 months linear" means: nothing for a year, then 1/24th of the position each month.

Vesting is where a token's real power structure is written down. Everybody's tokens are worth the same price; only some people can sell them today. If you want the fuller treatment of how these schedules are designed and read, this blog's [lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) and [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) go deeper than I will here.

### Disclosure

**Disclosure**, in the sense used in this post, means telling the audience three specific things: that you were compensated, in what form and amount, and by whom. Not "I hold some", not "not financial advice", not "#ad" buried in a hashtag block. The precise legal requirements are covered later, and they are older and stricter than most people assume.

## 1. The commercial structures: seven ways attention gets paid

Ask what a promoter "charges" and you will get a number for a post. That number is the least interesting part of the arrangement, and usually the smallest.

![The promotion stack: the deals worth the most to the promoter sit at the bottom, where the audience can see the least](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-3.webp)

The stack above is the argument of this section in one picture. **Value to the promoter and visibility to the audience run in opposite directions.** A disclosed cash sponsorship is visible and comparatively cheap. A discounted allocation with a short lockup is invisible and can be worth an order of magnitude more. Anyone reasoning about promotion by looking only at posting fees is looking at the smallest line on the invoice.

Here are the seven structures, from most visible to least.

**1. The sponsored post.** A flat cash fee for a post, video, or thread. Straightforward, and the one form that most closely resembles ordinary advertising. Rates scale with audience and conversion, and vary enormously — this is a negotiated private market with no public price list, so I am not going to quote a number I cannot source. What matters structurally is that the promoter's payoff here is *independent of what the token does next*. That is the honest version of the trade, and it is the least common of the seven.

**2. The paid AMA or sponsored "research".** The project pays for a hosted conversation, a written report, or a "deep dive". The commercial relationship is the same as a sponsored post; the *framing* is editorial. This is the format where the gap between what the audience thinks it is receiving and what it is actually receiving is widest, because the container signals independent analysis.

**3. The launchpad or referral affiliate deal.** The promoter receives a share of fees or revenue generated by users they send. Exchanges and launchpads run these openly — they are ordinary affiliate programs. The structural feature to notice is that the promoter is paid on **volume**, not on outcomes. They earn whether the referred user profits or not, and they earn more when the user trades more. This blog's post on [launchpads, airdrops and the points meta](/blog/trading/crypto-players/launchpads-airdrops-and-the-points-meta) covers how these funnels are constructed.

**4. The advisor or ambassador token grant.** A grant of tokens, cost basis zero, in exchange for advice, an ambassador title, or a logo on a website. It vests over months.

**5. The KOL allocation round.** A round of the token sale reserved for promoters, priced below the public sale and — this is the operative term — often released faster.

**6. Pure undisclosed promotion.** No contract the audience sees, no disclosure, sometimes no cash at all: just a position acquired earlier and a post that does not mention it.

**7. The residual case: the promoter is the issuer.** The account launches or co-launches the asset it promotes. At this point "influencer marketing" is not really the right frame — this is a primary distribution, and the audience is the counterparty.

Two of these deserve full arithmetic, because they are where the money actually is.

#### Worked example 1: what a \$25,000 KOL allocation is worth

All numbers here are an **illustrative composite** — round figures chosen so the arithmetic is checkable. They describe a shape that is common, not a specific deal.

TKN's sale is structured in tiers. Seed investors paid **\$0.008**. A KOL round is offered at **\$0.02**. The public sale is **\$0.05**. The token opens for trading at **\$0.12**.

A promoter takes a **\$25,000** allocation at \$0.02:

- Tokens received = \$25,000 ÷ \$0.02 = **1,250,000 TKN**.
- Vesting: **25% liquid at TGE**, the rest linear over six months.
- Liquid at TGE = 1,250,000 × 25% = **312,500 TKN**.
- At the \$0.12 opening price, that day-one tranche is worth 312,500 × \$0.12 = **\$37,500**.

Stop there for a moment. The tranche that becomes sellable on the first morning is worth \$37,500 against a total cost of \$25,000. **The position has returned 1.5× its entire cost before a single locked token has released.**

![The cash flows of one \$25,000 KOL allocation: the day-one tranche alone exceeds the cost of the whole cheque](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-4.webp)

The remaining 937,500 tokens release at 156,250 per month for six months — **\$18,750 a month** at \$0.12. And the full position, if it could all be sold at \$0.12, is 1,250,000 × \$0.12 = **\$150,000**, or **6.0×** on \$25,000.

Now the honest correction, because that \$150,000 is a paper number and this post is not in the business of paper numbers. The promoter cannot sell 1,250,000 tokens at \$0.12; selling pushes the price down, exactly as buying pushes it up. The realistic figure is the one that actually clears. But note what the structure has already done: **the day-one tranche alone made the position unloseable.** Even if TKN goes to zero on day two, the promoter is ahead.

*The intuition: when the first unlocked slice exceeds the total cost, every subsequent outcome is a free option — and a free option does not create any incentive to be careful about the call.*

#### Worked example 2: the advisor grant, which costs nothing at all

The advisor structure is simpler and, per unit of effort, usually better.

An advisor grant of **0.35% of supply** on TKN's one billion tokens is **3,500,000 TKN**, at a cost basis of **\$0.00**. Vesting is 12 months linear from TGE, so **291,667 tokens release per month**.

At the \$0.12 opening price:

- Monthly release = 291,667 × \$0.12 = **\$35,000 per month**.
- Full grant = 3,500,000 × \$0.12 = **\$420,000**.

In exchange for a title, a logo, and some calls. There is no capital at risk, so there is no price at which the advisor loses money — only prices at which they make less. Compare that to the \$25,000 allocation, where at least \$25,000 was genuinely at stake for a few hours.

*The intuition: a zero-cost-basis grant has no breakeven. An adviser with no breakeven has no shared interest with anyone who bought at a price.*

### Why the payment form matters more than the amount

Put the cohorts side by side and the hierarchy becomes visible.

![Who can sell when: the KOL round is the one seat that is both cheap and liquid on day one](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-5.webp)

Seed investors paid the least but wait the longest — a 12-month cliff, then a long linear release. The team waits longer still. The public paid the most and is fully liquid immediately, which sounds like an advantage and is not, because they paid \$0.05 for something now marked at \$0.12 with everyone else's supply queued behind them.

The KOL round occupies the one genuinely privileged seat: **cheap and liquid**. Paid \$0.02, sellable on day one.

Scale it up. Suppose the round comprises **40 cheques of \$25,000** — **\$1,000,000** raised, **50,000,000 TKN** distributed, which is **5% of total supply**. The 25% day-one unlock across all 40 is **12,500,000 TKN**, worth **\$1,500,000** at \$0.12.

Now set that against the float. The entire circulating supply is 120,000,000 tokens worth \$14,400,000. So **10.4% of everything that trades on day one is held by people whose cost basis is \$0.02 and whose position has already paid for itself.** That is the structural fact the rest of this post follows from.

## 2. How it shows up in price: turning attention into buying pressure

A promotion campaign does not move price by being persuasive. It moves price by being *concentrated*, against a book that is *thin*. Those are the two variables, and the second one does most of the work.

### Why a small float is a lever

<figure class="blog-anim">
<svg viewBox="0 0 880 440" role="img" aria-label="The same 500,000 dollar market buy sweeping two order books: a thin book with 180,000 dollars resting within 2 percent of mid is lifted 13.8 percent, while a deep book with 6,000,000 dollars resting within 2 percent barely moves at 0.2 percent." style="width:100%;height:auto;max-width:880px">
<style>
.a2-eat{fill:var(--accent,#6366f1);transform-origin:left center;animation:a2eat 12s ease-in-out infinite}
.a2-e1{animation-delay:.2s}.a2-e2{animation-delay:.9s}.a2-e3{animation-delay:1.6s}
.a2-e4{animation-delay:2.3s}.a2-e5{animation-delay:3s}.a2-e6{animation-delay:3.7s}
.a2-sip{fill:var(--accent,#6366f1);transform-origin:left center;animation:a2sip 12s ease-in-out infinite;animation-delay:.2s}
.a2-markL{animation:a2markL 12s ease-in-out infinite}
.a2-markR{animation:a2markR 12s ease-in-out infinite}
.a2-order{fill:#16a34a;opacity:0;animation:a2order 12s ease-in-out infinite}
.a2-lvl{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.a2-lbl{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;font-size:13px;fill:var(--text-secondary,#6b7280)}
.a2-lblb{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;font-size:14px;font-weight:600;fill:var(--text-primary,#1f2937)}
.a2-tiny{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;font-size:11px;fill:var(--text-secondary,#6b7280)}
.a2-dash{stroke:#dc2626;stroke-width:2.5;stroke-dasharray:7 5}
@keyframes a2eat{0%,3%{transform:scaleX(0)}30%,80%{transform:scaleX(1)}94%,100%{transform:scaleX(0)}}
@keyframes a2sip{0%,3%{transform:scaleX(0)}30%,80%{transform:scaleX(.083)}94%,100%{transform:scaleX(0)}}
@keyframes a2markL{0%,6%{transform:translateY(0)}44%,80%{transform:translateY(-166px)}94%,100%{transform:translateY(0)}}
@keyframes a2markR{0%,6%{transform:translateY(0)}44%,80%{transform:translateY(-4px)}94%,100%{transform:translateY(0)}}
@keyframes a2order{0%,2%{opacity:0}10%,80%{opacity:1}92%,100%{opacity:0}}
@media (prefers-reduced-motion:reduce){.a2-eat,.a2-sip,.a2-markL,.a2-markR,.a2-order{animation:none;opacity:1;transform:none}}
</style>
<text class="a2-lblb" x="50" y="28">The same &#36;500,000 buy, two different books</text>
<text class="a2-lbl" x="50" y="48">Illustrative composite: resting sell orders stacked upward from the mid price of &#36;0.120</text>
<rect class="a2-order" x="50" y="66" width="176" height="26" fill="none" stroke="#16a34a" stroke-width="2"/>
<text class="a2-lblb" x="60" y="85" fill="#16a34a">&#36;500,000 market buy</text>
<rect class="a2-order" x="490" y="66" width="176" height="26" fill="none" stroke="#16a34a" stroke-width="2"/>
<text class="a2-lblb" x="500" y="85" fill="#16a34a">&#36;500,000 market buy</text>
<text class="a2-lblb" x="50" y="122">Thin float — 12% circulating</text>
<text class="a2-tiny" x="50" y="140">&#36;180,000 rests within 2% of mid</text>
<rect class="a2-lvl" x="50" y="316" width="42" height="20"/><rect class="a2-eat a2-e1" x="50" y="316" width="42" height="20"/>
<rect class="a2-lvl" x="50" y="286" width="52" height="20"/><rect class="a2-eat a2-e2" x="50" y="286" width="52" height="20"/>
<rect class="a2-lvl" x="50" y="256" width="60" height="20"/><rect class="a2-eat a2-e3" x="50" y="256" width="60" height="20"/>
<rect class="a2-lvl" x="50" y="226" width="68" height="20"/><rect class="a2-eat a2-e4" x="50" y="226" width="68" height="20"/>
<rect class="a2-lvl" x="50" y="196" width="76" height="20"/><rect class="a2-eat a2-e5" x="50" y="196" width="76" height="20"/>
<rect class="a2-lvl" x="50" y="166" width="84" height="20"/><rect class="a2-eat a2-e6" x="50" y="166" width="84" height="20"/>
<text class="a2-tiny" x="146" y="331">&#36;0.1212</text><text class="a2-tiny" x="146" y="301">&#36;0.1248</text>
<text class="a2-tiny" x="146" y="271">&#36;0.1284</text><text class="a2-tiny" x="146" y="241">&#36;0.1320</text>
<text class="a2-tiny" x="146" y="211">&#36;0.1344</text><text class="a2-tiny" x="146" y="181">&#36;0.1368</text>
<g class="a2-markL"><line class="a2-dash" x1="46" y1="340" x2="230" y2="340"/><text class="a2-tiny" x="234" y="344" fill="#dc2626">last</text></g>
<text class="a2-lblb" x="50" y="378" fill="#dc2626">Price ends +13.8% at &#36;0.137</text>
<text class="a2-lbl" x="50" y="398">Average fill &#36;0.126 — the buyers paid up 5.2%</text>
<text class="a2-lbl" x="50" y="418">on the way in, and the last print flatters them.</text>
<line x1="440" y1="110" x2="440" y2="420" stroke="var(--border,#d1d5db)" stroke-width="1.5"/>
<text class="a2-lblb" x="490" y="122">Deep float — 60% circulating</text>
<text class="a2-tiny" x="490" y="140">&#36;6,000,000 rests within 2% of mid</text>
<rect class="a2-lvl" x="490" y="316" width="290" height="20"/><rect class="a2-sip" x="490" y="316" width="290" height="20"/>
<rect class="a2-lvl" x="490" y="286" width="300" height="20"/>
<rect class="a2-lvl" x="490" y="256" width="308" height="20"/>
<rect class="a2-lvl" x="490" y="226" width="316" height="20"/>
<rect class="a2-lvl" x="490" y="196" width="324" height="20"/>
<rect class="a2-lvl" x="490" y="166" width="332" height="20"/>
<text class="a2-tiny" x="826" y="331">&#36;0.1212</text><text class="a2-tiny" x="826" y="301">&#36;0.1248</text>
<text class="a2-tiny" x="826" y="271">&#36;0.1284</text><text class="a2-tiny" x="826" y="241">&#36;0.1320</text>
<text class="a2-tiny" x="826" y="211">&#36;0.1344</text><text class="a2-tiny" x="826" y="181">&#36;0.1368</text>
<g class="a2-markR"><line class="a2-dash" x1="486" y1="340" x2="670" y2="340"/><text class="a2-tiny" x="674" y="344" fill="#dc2626">last</text></g>
<text class="a2-lblb" x="490" y="378" fill="#dc2626">Price ends +0.2% at &#36;0.1202</text>
<text class="a2-lbl" x="490" y="398">The order consumes 8% of the first level and</text>
<text class="a2-lbl" x="490" y="418">never reaches the second.</text>
</svg>
<figcaption>Float is the lever, not the message. The same &#36;500,000 market buy walks six price levels of a thin book and leaves the last print 13.8% higher, but consumes only 8% of the first level of a deep one and moves it 0.2% — roughly seventy times less. This is why promotion is targeted at tokens with a small circulating supply: the campaign does not have to be persuasive, only large enough relative to the book.</figcaption>
</figure>

#### Worked example 3: what \$500,000 does to a thin book

Illustrative composite again, with an explicit order book so you can follow every step.

TKN trades at **\$0.120**. The resting sell orders stack up like this:

| Price band | Distance above mid | Sell orders resting | Cumulative |
| --- | --- | --- | --- |
| \$0.1200 – \$0.1224 | 0% to +2% | \$180,000 | \$180,000 |
| \$0.1224 – \$0.1320 | +2% to +10% | \$220,000 | \$400,000 |
| \$0.1320 – \$0.1440 | +10% to +20% | \$260,000 | \$660,000 |

A coordinated posting window brings **\$500,000** of market buying. Walk it through:

1. The first **\$180,000** clears the entire 0–2% band. Price is now \$0.1224.
2. The next **\$220,000** clears the +2% to +10% band. Cumulative spend \$400,000; price is now \$0.1320.
3. **\$100,000** remains, entering a band that holds \$260,000 and spans \$0.1320 to \$0.1440. It consumes \$100,000 ÷ \$260,000 = **38.5%** of that band, lifting price 38.5% of the way across a \$0.0120 span — about **\$0.0046**.

Final price = \$0.1320 + \$0.0046 = **\$0.1366**, call it **\$0.137**. That is **+13.8%** from \$0.120.

Now the part that matters more than the headline. Compute what the buyers actually *paid* on average, not what the last print says:

- Band 1: \$180,000 at an average of about \$0.1212 → **1,485,000 TKN**
- Band 2: \$220,000 at an average of about \$0.1272 → **1,730,000 TKN**
- Band 3: \$100,000 at an average of about \$0.1343 → **745,000 TKN**

Total: roughly **3,960,000 TKN for \$500,000**, an average fill of **\$0.126** — **5.2% above** where the price started. The chart says +13.8%. The buyers, in aggregate, are up 5.2% at best, and only if they could all sell at the new last price, which they cannot, because selling walks the book the other way.

Now run the same \$500,000 into a token with a **60% float** and **\$6,000,000** resting within 2% of mid. The order consumes 8.3% of the first band and never reaches the second. Price moves about **+0.2%**.

Same order. Same "narrative". **Roughly seventy times the price move**, purely because of how much supply was free to trade.

*The intuition: promotion is aimed at low-float tokens for the same reason a lever is placed close to the fulcrum. The campaign does not have to be convincing — it only has to be large relative to the book.*

This is also why the low-float, high-FDV launch structure and the promotion economy grew up together; a later post in this series takes up the low-float/high-FDV game directly.

### The posting window

Concentration is the second variable. Ten posts spread over a month is marketing. Ten posts inside six hours is a liquidity event.

<figure class="blog-anim">
<svg viewBox="0 0 880 470" role="img" aria-label="A coordinated posting window: posts per hour build to a peak, price rises from 12.0 to 13.7 cents and then falls back to 8.1 cents, while unlocked promoter supply drains out and cumulative retail buying accumulates to only one third of that supply." style="width:100%;height:auto;max-width:880px">
<style>
.a1-bar{fill:var(--border,#d1d5db);transform-origin:center bottom;animation:a1grow 14s ease-in-out infinite}
.a1-b1{animation-delay:0s}.a1-b2{animation-delay:.35s}.a1-b3{animation-delay:.7s}.a1-b4{animation-delay:1.05s}
.a1-b5{animation-delay:1.4s}.a1-b6{animation-delay:1.75s}.a1-b7{animation-delay:2.1s}.a1-b8{animation-delay:2.45s}
.a1-b9{animation-delay:2.8s}.a1-b10{animation-delay:3.15s}.a1-b11{animation-delay:3.5s}.a1-b12{animation-delay:3.85s}
.a1-price{fill:none;stroke:var(--accent,#6366f1);stroke-width:3.5;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:760;stroke-dashoffset:760;animation:a1draw 14s ease-in-out infinite}
.a1-peak{fill:var(--accent,#6366f1);opacity:0;animation:a1pop 14s ease-in-out infinite}
.a1-supply{fill:#dc2626;transform-origin:center top;animation:a1drain 14s ease-in-out infinite}
.a1-retail{fill:#16a34a;transform-origin:center bottom;animation:a1fill 14s ease-in-out infinite}
.a1-lbl{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;font-size:13px;fill:var(--text-secondary,#6b7280)}
.a1-lblb{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;font-size:14px;font-weight:600;fill:var(--text-primary,#1f2937)}
.a1-tiny{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;font-size:11px;fill:var(--text-secondary,#6b7280)}
.a1-ax{stroke:var(--border,#d1d5db);stroke-width:1.5}
@keyframes a1grow{0%,4%{transform:scaleY(0)}22%,72%{transform:scaleY(1)}96%,100%{transform:scaleY(0)}}
@keyframes a1draw{0%,6%{stroke-dashoffset:760}62%,84%{stroke-dashoffset:0}96%,100%{stroke-dashoffset:760}}
@keyframes a1pop{0%,30%{opacity:0;r:0}40%,84%{opacity:1;r:6}96%,100%{opacity:0;r:0}}
@keyframes a1drain{0%,8%{transform:scaleY(1)}64%,84%{transform:scaleY(.12)}96%,100%{transform:scaleY(1)}}
@keyframes a1fill{0%,8%{transform:scaleY(0)}64%,84%{transform:scaleY(1)}96%,100%{transform:scaleY(0)}}
@media (prefers-reduced-motion:reduce){.a1-bar,.a1-price,.a1-peak,.a1-supply,.a1-retail{animation:none;stroke-dashoffset:0;opacity:1;transform:none}}
</style>
<text class="a1-lblb" x="60" y="28">The posting window: attention builds, price peaks, supply leaves</text>
<text class="a1-lbl" x="60" y="48">Illustrative composite token TKN — bars are coordinated posts per hour, the line is price</text>
<line class="a1-ax" x1="60" y1="330" x2="620" y2="330"/>
<line class="a1-ax" x1="60" y1="80" x2="60" y2="330"/>
<rect class="a1-bar a1-b1" x="76" y="323" width="30" height="7"/>
<rect class="a1-bar a1-b2" x="120" y="316" width="30" height="14"/>
<rect class="a1-bar a1-b3" x="164" y="232" width="30" height="98"/>
<rect class="a1-bar a1-b4" x="208" y="176" width="30" height="154"/>
<rect class="a1-bar a1-b5" x="252" y="148" width="30" height="182"/>
<rect class="a1-bar a1-b6" x="296" y="190" width="30" height="140"/>
<rect class="a1-bar a1-b7" x="340" y="246" width="30" height="84"/>
<rect class="a1-bar a1-b8" x="384" y="281" width="30" height="49"/>
<rect class="a1-bar a1-b9" x="428" y="302" width="30" height="28"/>
<rect class="a1-bar a1-b10" x="472" y="309" width="30" height="21"/>
<rect class="a1-bar a1-b11" x="516" y="316" width="30" height="14"/>
<rect class="a1-bar a1-b12" x="560" y="323" width="30" height="7"/>
<polyline class="a1-price" points="91,176 135,176 179,162 223,141 267,117 311,131 355,172 399,210 443,251 487,279 531,296 575,309"/>
<circle class="a1-peak" cx="267" cy="117" r="6"/>
<text class="a1-tiny" x="238" y="106">&#36;0.137</text>
<text class="a1-tiny" x="76" y="170">&#36;0.120</text>
<text class="a1-tiny" x="548" y="325">&#36;0.081</text>
<text class="a1-tiny" x="76" y="348">T−2h</text>
<text class="a1-tiny" x="244" y="348">peak</text>
<text class="a1-tiny" x="548" y="348">T+48h</text>
<text class="a1-lbl" x="60" y="372">Hours around the coordinated posting window</text>
<line class="a1-ax" x1="655" y1="80" x2="655" y2="345"/>
<rect x="676" y="110" width="52" height="220" fill="none" stroke="var(--border,#d1d5db)" stroke-width="1.5"/>
<rect class="a1-supply" x="676" y="110" width="52" height="220"/>
<text class="a1-tiny" x="668" y="350">promoter</text>
<text class="a1-tiny" x="672" y="364">supply</text>
<text class="a1-lblb" x="666" y="100">&#36;1,500,000</text>
<rect x="768" y="257" width="52" height="73" fill="none" stroke="var(--border,#d1d5db)" stroke-width="1.5"/>
<rect class="a1-retail" x="768" y="257" width="52" height="73"/>
<text class="a1-tiny" x="774" y="350">retail</text>
<text class="a1-tiny" x="770" y="364">buying</text>
<text class="a1-lblb" x="762" y="247">&#36;500,000</text>
<text class="a1-lbl" x="655" y="392">Unlocked day-one promoter supply is 3× the</text>
<text class="a1-lbl" x="655" y="410">buying the window generates.</text>
<text class="a1-lbl" x="60" y="410">The bars stop; the supply does not.</text>
</svg>
<figcaption>A coordinated posting window in motion. Posts per hour build to a peak and decay within about two days, dragging price from &#36;0.120 to &#36;0.137 and back to &#36;0.081. Meanwhile the day-one unlocked promoter supply — &#36;1,500,000 at the opening price in this illustrative composite — drains out into the &#36;500,000 of buying that the attention actually produced. The window is not where the money is made; it is where the exit liquidity is manufactured.</figcaption>
</figure>

The reason concentration matters is that book depth **replenishes**. Sell orders that get consumed are replaced, by market makers and by holders raising their offers, over minutes to hours. Buying spread across a month meets a book that refills between each order and barely moves price at all. Buying compressed into hours meets a book that has not had time to refill, and each order starts higher up the ladder than the last.

Layer on the reflexive part: a rising price is itself the most effective promotional content there is. The chart becomes the argument. Screenshots of the chart become the next round of posts, from accounts nobody paid, which is the cheapest attention in the funnel. This blog's game-theory series covers the general mechanism in [information cascades and herding](/blog/trading/game-theory/information-cascades-and-herding-when-rational-traders-follow-the-crowd) — the short version is that when people cannot observe fundamentals, they rationally infer information from other people's actions, and a manufactured action looks exactly like an informed one.

### Who sells into it

Return to the arithmetic. The window generated **\$500,000** of buying. The day-one unlocked KOL supply is **\$1,500,000** at the opening price.

The buying is one third of the supply that became liquid the same morning — and that is before the advisor grants, before the first monthly release, before anything the seed investors do later. There is no scenario in which \$500,000 of new demand absorbs \$1,500,000 of newly-liquid, fully-paid-for supply at a stable price. The only question is the order in which people sell and what price the last one gets.

Note carefully what this argument does **not** require. It does not require anyone to lie. It does not require a coordinated agreement to dump. Forty independent promoters, each individually rational, each holding a position that has already paid for itself, each aware that the others are also unlocked, will arrive at the same behaviour without a conversation. That is a coordination problem with an obvious solution, and it is why [crowded exits](/blog/trading/game-theory/crowded-trades-and-the-exit-game) behave the way they do. The mechanics of *manufactured* volume specifically — wash trading and the rest — are treated in [wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume).

## 3. The incentive asymmetry

Everything so far can be compressed into one chart.

![Three buyers, three breakevens: at a 67% drawdown the promoter is up 100% and the audience is down 67%](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-6.webp)

Three participants hold the identical asset. Their breakevens are \$0.02, \$0.05, and \$0.12.

At the **\$0.12** listing price:

- Promoter: +\$0.10 per token, **+500%**
- Public sale buyer: +\$0.07 per token, **+140%**
- Audience buyer: **\$0.00**, flat

Now let the price fall to **\$0.04** — a 67% drawdown that would be described everywhere as a collapse:

- Promoter: +\$0.02 per token, **+100%**
- Public sale buyer: −\$0.01 per token, **−20%**
- Audience buyer: −\$0.08 per token, **−67%**

The promoter has *doubled their money* in the scenario the audience experiences as ruin. They are not lying when they say they still believe in the project; at \$0.04 they are still up 100%, and conviction is cheap when your breakeven is six cents below the current price.

#### Worked example 4: the expected value of following the call

Let us put a number on the trade from the audience's side. The probabilities below are an **illustrative composite** — chosen to be plausible given the shape documented in the academic literature on coordinated pumps, not measured from any specific campaign. The point is the structure of the calculation, which you can re-run with your own estimates.

You buy **\$2,000** of TKN when the post lands. Three outcomes:

| Outcome | Probability | Return |
| --- | --- | --- |
| You sell inside the first hour, near the peak | 30% | +25% |
| You sell the same day, after the peak | 25% | −20% |
| You hold a week | 45% | −60% |

Expected return = (0.30 × +25%) + (0.25 × −20%) + (0.45 × −60%)
= +7.5% − 5.0% − 27.0% = **−24.5%**

On \$2,000, that is **−\$490**.

And that is before trading costs. On a thin book you pay slippage getting in *and* getting out — the same walk up the ladder that made the price move, running in reverse when you exit alongside everyone else. Add a realistic 5 percentage points of round-trip friction and the expected outcome is **−29.5%**, or **−\$590** on \$2,000.

Notice the shape of the winning branch: it requires you to sell within the first hour, into the same window the campaign created. To profit, you must be faster than the audience you are part of. That is not a strategy; it is a seat assignment.

Now run the identical trade for the promoter. They sell their 312,500-token day-one tranche into that same window at the \$0.126 average fill computed earlier:

- Proceeds = 312,500 × \$0.126 = **\$39,375**
- Total cost of the entire allocation = **\$25,000**
- Net = **+\$14,375**, *even if every remaining token goes to zero*

*The intuition: in the same event, on the same asset, over the same hours, one participant has a negative expected value and the other cannot lose. That is not a difference in skill. It is a difference in cost basis and unlock date — both of which are usually published, and neither of which is usually mentioned in the post.*

## 4. The disclosure gap: what the law already requires

The most common assumption about crypto promotion is that it occupies a legal vacuum — that the rules simply have not been written yet. For the United States that is wrong, and the correct version is more interesting: **the relevant rule was written in 1933, it is narrower and more mechanical than people expect, and it is still routinely not met.**

![What the rules already require versus what a typical promotion actually shows](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-7.webp)

### Section 17(b), the anti-touting provision

Section 17(b) of the Securities Act of 1933 makes it unlawful to publish or circulate anything that describes a security for consideration, without disclosing that consideration. It was written about stock promoters in the aftermath of the 1929 crash. It has been sitting there, unchanged in substance, the entire time.

Three features of it matter, and each one cuts against a common belief.

**It requires exactly two disclosures, not three.** The statute requires disclosure of the *receipt* of the consideration — expressly including consideration that is "past or prospective" — and "the amount thereof". That is the whole obligation. Notice what is *not* in that list: **the source**. The SEC's own press releases routinely describe the requirement as disclosing the "nature, source, and amount" of compensation, but that phrasing is the agency's gloss, not the statutory text. The bar is lower than the SEC's own summaries imply.

The "past or prospective" clause is the one worth internalising, because it closes the loophole most promoters reach for first. A promise of tokens, an allocation you have been offered but not yet received, a fee contingent on the campaign's performance — all of it is consideration that must be disclosed *before* it arrives, not after.

**There is no scienter requirement.** *Scienter* is the legal term for a guilty state of mind — intent to deceive, or reckless disregard for the truth. Most securities-fraud provisions require it, which is why fraud cases are hard to bring. Section 17(b) does not. You do not need to have intended to mislead anyone. You received consideration, you promoted the security, you did not disclose. The omission is the violation.

This is why the distinction in the next section matters so much: **a Section 17(b) case is a non-disclosure case, not a fraud case.** They are different accusations with different elements, and conflating them misdescribes what people were actually found to have done.

**It only applies to securities.** This is the genuine limit, and the one doing the most work in practice. Section 17(b) reaches "any security". Whether a given token is a security is exactly the question that a decade of litigation has failed to settle cleanly. A promoter's exposure therefore depends on a classification that nobody can reliably determine in advance — which is a poor foundation for a compliance regime and an excellent one for ambiguity.

### The FTC's Endorsement Guides

The second rule does not care whether the token is a security at all.

The Federal Trade Commission's *Guides Concerning the Use of Endorsements and Testimonials in Advertising* — codified at 16 CFR Part 255 — require an endorser to disclose any **material connection** to the thing being endorsed. A material connection is any relationship that a reasonable audience would not expect and that might affect how much weight they give the endorsement: payment, free product, an equity or token position, a family relationship, an affiliate commission.

The FTC's practical guidance for influencers is unusually concrete, and worth knowing because it defines what does *not* count:

- The disclosure must be **in the post itself**, not on a separate "disclosures" page, not in a linked bio, not in a pinned comment.
- It must be **hard to miss** — not buried at the end of a long caption, not hidden inside a block of hashtags, not requiring a "more" tap to reveal.
- It applies to **free product and unpaid perks**, not only cash.
- Ambiguous tags are not enough. "#collab", "#sp", "#thanks" and similar shorthand do not communicate a commercial relationship to an ordinary reader.

Against that standard, look at what the typical crypto promotion actually carries. "Not financial advice" is a disclaimer, and discloses nothing whatsoever — it disclaims liability for the advice, while saying nothing about who paid for it. "I'm invested" or "I hold a bag" discloses a position but not its price, its size, or its unlock date, which are the three facts that determine whether the poster is your peer or your counterparty. And "#ad" as the twenty-second hashtag in a hashtag wall is exactly the placement the FTC's guidance singles out as insufficient.

> A disclaimer protects the person who wrote it. A disclosure protects the person who reads it. Almost every crypto promotion carries the first and not the second.

### What a compliant disclosure would actually look like

It is worth writing one out, because most readers have never seen one:

*"Paid promotion. I received 1,250,000 tokens from the foundation at \$0.02 per token, a total of \$25,000 in consideration, of which 25% became transferable at listing and the remainder vests monthly over six months."*

That is two sentences. It satisfies Section 17(b) comfortably — receipt and amount, disclosed up front — and it satisfies the FTC's material-connection standard. It also, not coincidentally, gives the reader every number they need to work out that the poster's breakeven sits at a sixth of the current price and that their first tranche is already liquid.

Which is precisely why it is rare. The disclosure is not costly because it is legally onerous. It is costly because **it works**.

## 5. The enforcement record — and why it is weaker now than it was in 2023

Here is where most writing on this subject goes wrong. The usual telling ends in 2023 with a run of celebrity settlements and an implied moral: the regulators noticed, the fines landed, the problem is being handled. That story was roughly accurate when it was written. It is not accurate now, and the correction matters more to a reader than the original story did.

The enforcement history has **two eras**, and we are living in the second one.

![The Section 17(b) enforcement arc: escalation from 2017 to 2023, then retreat in 2025 and 2026](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-8.webp)

### Era one, 2017 to 2023: the warning and the escalation

On **1 November 2017**, at the peak of the initial-coin-offering boom, the SEC published a public statement warning that celebrities and others promoting token offerings could be violating the anti-touting provisions if they did not disclose the nature, scope and amount of the compensation they received. It was a warning shot, issued before any case had been brought.

Over the following five and a half years the agency brought a series of settled actions against promoters. Read as a sequence, they show a clear and deliberate escalation — not in the *sums involved*, which stayed small, but in the **penalty relative to the promoter's fee**.

That ratio is the most informative number in the entire record, so let us do the arithmetic properly.

#### Worked example 5: what the SEC charged for a promotion, as a multiple of the fee

The question a promoter's lawyer actually asks is not "what is the fine?" but "what is the fine *relative to what we were paid?*" A penalty smaller than the fee is a cost of doing business. A penalty several times the fee is a deterrent. Here is how that multiple moved:

| Settled action | Date | Civil penalty ÷ promotional payment |
| --- | --- | --- |
| Floyd Mayweather Jr. (Centra Tech and two other ICOs) | Nov 2018 | **1.0×** |
| Khaled Khaled, professionally DJ Khaled (Centra Tech) | Nov 2018 | **2.0×** |
| Steven Seagal (Bitcoiin2Gen) | Feb 2020 | **1.0×** |
| Kim Kardashian (EthereumMax) | Oct 2022 | **4.0×** |
| Paul Pierce (EthereumMax) | Feb 2023 | **≈4.7×** |

Read the column downward. In the 2018 and 2020 ICO-era cases the civil penalty was **one to two times** the promotional fee. By the 2022–2023 EthereumMax cases it was **four to nearly five times**.

Work through the arithmetic of what a 1.0× multiple actually means, because it is the key to the whole table. In those early cases the promoter also gave up the fee itself, through *disgorgement* — the surrender of ill-gotten gains, which is a return of the money rather than a punishment for taking it. So a promoter paid a fee of \$1 typically surrendered that \$1 in disgorgement, paid a further \$1 as the civil penalty, and added prejudgment interest on top. Net position: roughly **−\$1 relative to never having taken the deal**, plus legal costs and the reputational hit.

Now take the fully specified case. In the Paul Pierce settlement of **February 2023**, the Commission's order imposed a civil penalty of **\$1,150,000**, with disgorgement and prejudgment interest bringing the total to **\$1,409,565**. (The SEC's press release for the same matter states the penalty as \$1,115,000; the order's figure of \$1,150,000 is the one that reconciles to the \$1,409,565 total that both documents report, so the order is the number to cite.) At roughly 4.7× the payment, the arithmetic has changed character: the promoter is now materially worse off than if the deal had never happened.

*The intuition: disgorgement alone makes a promotion a break-even proposition — you simply give the fee back. Only the penalty multiple turns it into a losing one. Between 2018 and 2023 that multiple went from 1× to nearly 5×, and that, not the headline dollar figure, is what "the SEC got serious" actually meant.*

### The distinction that must not be blurred

Before the second era, one point of precision, because it concerns living people and it is the single easiest thing to get wrong in this subject.

**Of the promoters named above, the settlements with Mayweather, Khaled, Seagal and Kardashian were non-disclosure cases under Section 17(b), not fraud cases.** They were charged with failing to disclose payment. They were not charged with lying about the assets, with manipulating prices, or with defrauding anyone. As set out in the previous section, Section 17(b) requires no scienter — no intent to deceive — precisely so that the omission alone is actionable. Describing these settlements as "fraud" or as "pump-and-dump" would misstate what the Commission actually alleged.

Two matters in the record did include antifraud allegations: **Paul Pierce**, and **John McAfee**. Pierce settled. McAfee did not: the SEC's civil complaint of October 2020 and a separate Department of Justice indictment in March 2021 both alleged undisclosed compensation for token promotion, and neither was ever adjudicated — a Notice of Death was filed in the case, which was subsequently dismissed. **Those allegations were never tested and stand unproven.**

Everywhere else in this post where a mechanism sounds like manipulation, it is a description of how incentives and supply interact, not an accusation against any identified person.

### Era two, 2025 to 2026: the retreat

Then the direction reversed.

**On 1 May 2025, the SEC's action against Ian Balina was dismissed with prejudice.** "With prejudice" means it cannot be refiled — the matter is finally closed. The case had alleged an unregistered token offering together with a failure to disclose a bonus the promoter received on tokens he bought and then promoted.

**On 5 March 2026, the long-running Tron matter resolved.** The Commission's March 2023 complaint had charged Justin Sun and associated entities, and had separately charged eight celebrities under Section 17(b) with touting TRX and BTT without disclosing compensation; six of the eight settled at the time. The 2026 resolution concluded the matter with a **single \$10 million penalty against Rainberry for wash trading**, and dismissed all remaining claims — including the last live Section 17(b) celebrity claim, against the artist known as Soulja Boy.

The crucial detail is how the Commission characterised these outcomes. **They were framed as an exercise of enforcement discretion and a matter of policy — not as a judgment on the merits.** No court held that the disclosure obligations do not apply. No ruling narrowed Section 17(b). The statute is exactly as it was in 1933, and every element described in the previous section still stands.

What changed is the probability that anyone will bring a case about it.

### What the two eras mean if you are the audience

Put the arc together and the conclusion is uncomfortable but simple.

Between 2017 and 2023 a retail buyer could reasonably believe that undisclosed promotion carried a real and rising expected cost to the promoter, and that this cost would push disclosure toward the standard the law already required. Between 2025 and 2026 that belief stopped being reasonable. The rule did not weaken; the enforcement of it did.

So the deterrent is not where a reader might assume it is. If you are relying on the existence of Section 17(b) to mean that an undisclosed promotion in front of you is rare, or that someone is checking, the record of the last two years does not support that. **The check is not being performed on your behalf. That is the argument for performing it yourself**, which is what the remaining two sections of this post are about.

This is a description of the enforcement record as of July 2026, not a prediction. Enforcement postures change with administrations, and the statute they would be enforcing has outlasted every one of them so far.

## 6. Auditing a track record honestly

This is the section I would keep if I had to delete the rest. Almost every promotion rests on an implicit claim about past accuracy, and almost every such claim is unfalsifiable as presented. Making it falsifiable is a mechanical exercise that takes an afternoon.

![The same forty calls, described two ways: the difference is nine deleted posts and a definition of "winner" that never has to sell](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-9.webp)

Four distinct problems corrupt a public track record, and they compound.

**Survivorship.** Posts get deleted. A call that went badly can simply cease to exist, and the surviving set is then presented as the complete set. This is the single largest distortion, and it is invisible unless you go looking for the archive.

**No exit price.** "I called this at \$0.04 and it hit \$0.12" describes an entry and a high-water mark, not a trade. Without a stated exit rule, the claim is compatible with every possible outcome, including a total loss.

**"Touched" metrics.** Defining a winner as a token that *touched* +20% at any point measures whether a pump occurred, not whether a buyer made money. In a promoted low-float token a brief tick upward is nearly guaranteed — it is the mechanism from section 2. A metric that is satisfied by the promotion itself carries no information about skill.

**Denominator drift.** Calls made in private channels, deleted threads, or reply chains are excluded when convenient and included when they worked.

#### Worked example 6: auditing forty calls

Illustrative composite, but the *method* is exactly what you would run on real data.

An account advertises: **"28 winners from 31 calls — a 90% hit rate."** Winner is defined as touching +20% at any point after the post.

Now do the work.

**Step 1 — rebuild the universe.** Search an archive service for deleted posts from the account. You recover **9 additional calls** that were posted and later removed. The true universe is **40**, not 31.

**Step 2 — impose one fixed rule, chosen in advance.** Buy \$1,000 at the close of the hour the call was posted. Sell exactly 7 calendar days later. No discretion, no exceptions. The rule can be anything defensible; what matters is that it is fixed before you look and applied to every call.

**Step 3 — compute on all 40.**

- **11 up, 29 down**
- Median return **−22%**, mean return **−16.4%**
- \$40,000 deployed → **\$40,000 × (1 − 0.164) = \$33,440**. A loss of **\$6,560**.

**Step 4 — isolate each distortion.** Run the identical rule on only the 31 surviving posts: 11 up, 20 down, mean **−9.8%**. On \$31,000 that is \$31,000 × (1 − 0.098) = **\$27,962**.

So survivorship alone — nine deleted posts out of forty — moved the measured mean by **6.6 percentage points**.

**Step 5 — test the advertised metric on honest data.** Apply "touched +20% at any point" to all 40 calls: **26 of 40, or 65%**. Still a high number, on a set that lost money under every fixed exit rule. The metric was never measuring what it appeared to measure.

*The intuition: the gap between "90% hit rate" and "lost \$6,560 on \$40,000" contains no dishonest individual number. It is produced by a deleted-post rate and a metric with no exit. Both are checkable in an afternoon, and neither requires you to accuse anyone of anything.*

### A small toolkit

The audit above is four columns in a spreadsheet, or this:

Build `calls.csv` with one row per call and the columns `ticker`, `posted_at` (UTC), `entry_price`, `price_after_7d`, `max_price_7d`, and `was_deleted`. Then:

```python
import pandas as pd

calls = pd.read_csv("calls.csv", parse_dates=["posted_at"])

calls["ret"] = calls["price_after_7d"] / calls["entry_price"] - 1.0

def audit(df, label, stake=1_000):
    n = len(df)
    ending = (stake * (1 + df["ret"])).sum()
    print(
        f"{label:<28} n={n:>3}  "
        f"up={int((df['ret'] > 0).sum()):>3}  "
        f"mean={df['ret'].mean():>7.1%}  "
        f"median={df['ret'].median():>7.1%}  "
        f"${stake * n:,} -> ${ending:,.0f}"
    )

audit(calls, "All calls (true universe)")
audit(calls[~calls["was_deleted"]], "Surviving posts only")

touched = (calls["max_price_7d"] / calls["entry_price"] - 1.0) >= 0.20
print(f"'Touched +20%' rate: {touched.mean():.0%} on n={len(calls)}")
```

The last two lines are the important ones: they compute what the advertised "touched +20%" metric would report *on the honest universe*, so you can see the gap between that number and what the fixed-exit rule actually returned.

Three properties make this honest, and they are the properties to demand of any track record, including your own: the universe is fixed before you look, the exit rule is fixed before you look, and the deleted posts are in the denominator.

## 7. Reading a promotion before you act on it

The audit is retrospective. This is the version you can run in ten minutes, on a promotion that is in front of you right now.

![Five questions, each with a disqualifying answer — none of them about whether the project is good](/imgs/blogs/influencers-kols-and-the-narrative-for-hire-machine-10.webp)

None of these questions asks whether the project is good. That is a different and much harder question, and it is not the one that determines whether this particular buy is a good idea. These ask whether you can *see who is on the other side*.

**1. Is there a disclosure — and does it name the payer, the form, and the amount?** "Not financial advice" is not a disclosure; it is a disclaimer, and it discloses nothing. "I'm invested" is closer but still omits the two facts that matter: what you paid and when you can sell. A real disclosure reads like *"I received 500,000 tokens from the foundation at \$0.02, 25% unlocked at listing."* In the absence of one, the correct prior is that a promotion of a newly-launched token is compensated, because most of them are.

**2. What share of total supply is actually circulating?** Divide circulating market cap by FDV. Both numbers are on every major data aggregator, and the tokenomics are usually in the project's own documentation. Under 20% means section 2 applies to you: small flows move this price a lot, in both directions. This is not a reason never to buy — it is a reason to size as if the price you see is provisional, because it is.

**3. Who unlocks next, and at what cost basis?** Unlock schedules are published. If a cohort with a materially lower cost basis becomes liquid within your holding period, you have identified the seller you will be buying from. This is the question that most often changes a decision, and it takes about four minutes.

**4. Is the track record a complete ledger with entry *and* exit prices?** If not, it is not evidence. Section 6 is the full treatment. The short version: no exit rule, no claim.

**5. What is the promoter's cost basis relative to yours?** If you can answer questions 1 and 3, you can usually answer this. Their breakeven sits below your entry by construction. That does not make them dishonest; it makes their risk different from yours, and it means their continued enthusiasm at lower prices carries no information about your position.

If all five have answers you are comfortable with, you are making a speculation with your eyes open, which is a legitimate thing to do with money you can afford to lose. If two or more are unanswerable, the trade is not "risky" — it is *unpriced*, which is a different and worse thing. This is educational material about market mechanics, not advice about what to buy.

## Common misconceptions

**"If the disclosure is missing, the promoter must be doing something illegal."** Not necessarily, and the framing is unhelpful. Whether a given promotion is unlawful depends on whether the asset is a security, where the promoter and audience are located, whether the promoter was compensated, and what exactly was said. Plenty of undisclosed promotion sits in genuine legal grey areas, especially across borders. The useful move is not to reason about legality but to reason about *position*: what does this person hold, at what price, and when can they sell?

**"A big audience means the call moves the price."** Backwards. Float determines how much the price moves; audience size determines how much money arrives. An account with 50,000 engaged followers posting about a \$14 million-float token will move it much further than an account with 2 million followers posting about Bitcoin. This is why campaigns concentrate on small tokens — the same budget buys far more price movement.

**"The influencer is dumping on me."** Sometimes, but this framing makes the problem sound rarer and more villainous than it is. The structural problem persists even when every promoter is sincere. A grant with zero cost basis and a six-month vest creates the same supply pressure whether the holder is cynical or a true believer. Focus on the schedule, not the sentiment — the schedule is published and the sentiment is not observable.

**"Price went up right after the post, so the call was good."** The post and the price move are the same event, not cause and confirmed thesis. Section 2 is precisely the mechanism by which a coordinated window produces a price rise regardless of the underlying merits. A rise inside the window is evidence that the campaign worked, which is a claim about the campaign, not about the token.

**"I'll just be faster than everyone else."** This is the 30% branch of worked example 4, and it is the branch the entire structure is designed to make you believe you are in. You are competing on speed against people who knew the posting schedule in advance, hold a position at a fraction of your cost, and are selling into your order. Even in the branch where you are right, your edge is measured in minutes and your slippage is measured in percent.

**"Disclosure would fix this."** It would help a great deal, and it is required more often than people realise. But disclosure addresses the information asymmetry, not the supply overhang. A fully disclosed promotion of a token with 12% float and a day-one insider unlock still has \$1,500,000 of cheap supply meeting \$500,000 of demand. Disclosure lets you see the trade. It does not change the trade.

## How it shows up in real markets

Everything above is mechanism. Here are the documented episodes it was built from — all of them matters of public record, all stated as what the filings actually allege or establish, with the legal posture named in each case.

### 1. Centra Tech, and the first two touting cases

In **November 2018** the SEC settled charges against the boxer **Floyd Mayweather Jr.** and the music producer **Khaled Khaled**, professionally known as DJ Khaled, over their promotion of initial coin offerings — including Centra Tech. These were the first cases the Commission brought under Section 17(b) in the token era, and they set the template for everything that followed.

The charge was **failure to disclose payment**, not fraud. Neither was alleged to have lied about the offerings. They promoted, they were paid, they did not say so. Both settled without admitting or denying the findings, surrendering the fees through disgorgement and paying civil penalties on top — Mayweather's penalty at roughly 1.0× his promotional payment, Khaled's at roughly 2.0×.

The mechanism from this post is visible in outline: a promotional payment, an audience with no way to see it, and an offering whose economics were entirely opaque from the outside. What is notable in hindsight is the modesty of the deterrent. At a 1× multiple, a promoter who took the fee and got caught ended up roughly where they started, minus legal costs.

### 2. Bitcoiin2Gen and the actor

In **February 2020** the SEC settled with the actor **Steven Seagal** over his promotion of the Bitcoiin2Gen offering. Again the charge was Section 17(b) non-disclosure. Again it settled without admissions, with disgorgement of the payment and a civil penalty at roughly 1.0× that payment.

The detail worth carrying forward is the *form* of the promised compensation. Arrangements of this kind commonly mix cash with a quantity of the promoted token — which is exactly the structure section 1 of this post describes, and exactly the structure that makes the promoter's interest diverge from the audience's. A cash fee is neutral about what happens next. A token grant is not.

### 3. EthereumMax, and the moment the multiple changed

The two EthereumMax settlements are the most instructive pair in the record because they bracket the shift.

In **October 2022** the SEC settled with **Kim Kardashian** over a social-media post promoting the EthereumMax token. Section 17(b) non-disclosure; settled without admitting or denying; civil penalty at roughly **4.0×** the payment she received, alongside disgorgement and interest. She also agreed not to promote crypto asset securities for a period of years.

In **February 2023** the SEC settled with the former basketball player **Paul Pierce** over promotion of the same token. Pierce's matter went further than the others: it included **antifraud allegations**, not only non-disclosure. The order imposed a civil penalty of **\$1,150,000**, with disgorgement and prejudgment interest taking the total to **\$1,409,565** — roughly 4.7× the payment.

Between these two cases and the 2018 ones, the penalty multiple roughly quadrupled. That is the escalation described in section 5, and at the time it looked like a trend line.

### 4. Tron, BitTorrent, and the eight

In **March 2023** the SEC filed a civil complaint against Justin Sun and associated entities, and separately charged **eight celebrities** under Section 17(b) with promoting the TRX and BTT tokens without disclosing compensation. Six of the eight settled at the time.

Then the trend line broke. **On 5 March 2026 the matter resolved** with a single **\$10 million** penalty against Rainberry for wash trading, and the dismissal of all remaining claims — including the last live Section 17(b) celebrity claim, against the artist known as Soulja Boy. The Commission framed the resolution as an exercise of enforcement discretion rather than a judgment on the merits.

For a reader, the lesson is not about any individual. It is that the last celebrity touting claim in the SEC's docket ended in dismissal rather than in a finding, and that the reason given was policy rather than law.

### 5. The Balina matter, and what "with prejudice" means

The SEC's action against **Ian Balina**, filed in September 2022, concerned an unregistered token offering and an allegation that he failed to disclose a bonus he received on tokens he purchased and then promoted. **On 1 May 2025 it was dismissed with prejudice** — finally, with no possibility of refiling.

This case is worth knowing precisely because its allegation was the *cleanest* statement of the structure this post is about: buy cheaper than the audience, promote, do not mention the discount. That such a case ended in a with-prejudice dismissal, on discretionary grounds, is the single sharpest indicator of where enforcement now sits.

### 6. The case that was never tested

In **October 2020** the SEC filed a civil complaint against **John McAfee** alleging undisclosed compensation for promoting token offerings, and in **March 2021** the Department of Justice brought a separate indictment. Both included antifraud allegations.

Neither was ever adjudicated. A Notice of Death was filed in the SEC matter and the case was dismissed. **The allegations were never tested in court and remain unproven** — which is the correct way to describe them, and a useful reminder that a filed complaint is an accusation, not a finding. This distinction is worth applying in the other direction too: the absence of a case against a promoter is equally not a finding that anything was disclosed.

### 7. The structural episode: low float meets a promotion budget

The final "episode" is not a case but a pattern, and it is the one most likely to affect a reader directly.

Through the 2024–2025 launch cohort, a large share of new tokens came to market with a small percentage of supply circulating and a valuation implied across the whole supply — the structure section 2 dissects. That design and the promotion economy are complements, not coincidences: a small float is what makes a modest promotion budget capable of moving a price at all, and a high implied valuation is what makes the token allocations paid to promoters worth taking.

No enforcement action is required for this to work, no one needs to lie, and every schedule involved is published. That is precisely why the defensive checklist in section 7 is built out of published numbers rather than out of judgments about anyone's honesty. Two later posts in this series — on the anatomy of a token pump, and on the low-float, high-FDV game — take this pattern apart in detail.

## Sources & further reading

Primary sources behind the legal and enforcement figures in this post:

- **Securities Act of 1933, Section 17(b)** — 15 U.S.C. §77q(b), the anti-touting provision. The statutory text requiring disclosure of the receipt of consideration, "whether past or prospective", and "the amount thereof". [Cornell Legal Information Institute](https://www.law.cornell.edu/uscode/text/15/77q).
- **SEC, public statement on potentially unlawful promotion of initial coin offerings and other investments by celebrities and others**, 1 November 2017 — the Commission's warning that undisclosed token promotion may violate the anti-touting provisions. [sec.gov](https://www.sec.gov/news/public-statement/statement-potentially-unlawful-promotion-icos).
- **SEC press releases and administrative orders** in the settled touting matters: Floyd Mayweather Jr. and Khaled Khaled (November 2018); Steven Seagal (February 2020); Kim Kardashian (October 2022); Paul Pierce (February 2023). Penalty and disgorgement figures in this post are taken from the orders; where an order and its accompanying press release disagree, as they do on the Pierce penalty, the order's figure is used.
- **SEC v. Sun et al.** — civil complaint filed March 2023 charging Justin Sun, associated entities and eight celebrities; resolved 5 March 2026 with a \$10 million wash-trading penalty against Rainberry and dismissal of all remaining claims.
- **SEC v. Balina** — filed September 2022, dismissed with prejudice 1 May 2025.
- **FTC, Guides Concerning the Use of Endorsements and Testimonials in Advertising**, 16 CFR Part 255 — the material-connection disclosure standard. [ftc.gov](https://www.ftc.gov/legal-library/browse/rules/guides-concerning-use-endorsements-testimonials-advertising).
- **FTC, Disclosures 101 for Social Media Influencers** — the agency's practical guidance on placement and wording of endorsement disclosures. [ftc.gov](https://www.ftc.gov/business-guidance/resources/disclosures-101-social-media-influencers).

On market impact and the academic treatment of coordinated pumps:

- **Kyle, Albert S.**, "Continuous Auctions and Insider Trading", *Econometrica* 53(6), 1985 — the canonical model of how order flow moves price, and the source of the "lambda" price-impact coefficient referenced in the Foundations section.
- **Xu, Jiahua and Benjamin Livshits**, "The Anatomy of a Cryptocurrency Pump-and-Dump Scheme", *USENIX Security Symposium*, 2019.
- **Dhawan, Anirudh and Tālis J. Putniņš**, "A New Wolf in Town? Pump-and-Dump Manipulation in Cryptocurrency Markets", *Review of Finance*, 2023.

Related posts on this blog:

- [Follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) — how to find the allocation and vesting terms this post tells you to look for.
- [The lifecycle of a token, seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) — the full schedule from private round to public overhang.
- [How VCs move price: listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) — the supply side of the same machine.
- [Wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) — when the activity itself, not just the attention, is manufactured.
- [Information cascades and herding](/blog/trading/game-theory/information-cascades-and-herding-when-rational-traders-follow-the-crowd) — why rational people follow a crowd that knows nothing.

*Every figure in this post that is attached to a named person or a regulatory action is drawn from the public record and dated. Every figure attached to the token "TKN" is an illustrative composite, chosen for round arithmetic, and describes no specific deal.*

## When this matters to you

If you never buy a newly-launched token, this post is still useful as a template, because the structure is not specific to crypto. Paid endorsement with an undisclosed position is one of the oldest patterns in finance — the American anti-touting statute is from 1933 for a reason, and it was written about stock promoters, not tokens. What crypto changed is the speed, the reach, and above all the *form of payment*: paying a promoter in the asset, at a discount, with a shorter lockup than the audience, converts advertising into supply distribution.

The practical takeaway is small and concrete. Before acting on any promotion of a new token, spend ten minutes on five things: the disclosure, the float, the next unlock, the track-record ledger, and the promoter's cost basis. Four of the five are published somewhere public. The fifth — the disclosure — is informative precisely when it is missing.

And the reframe worth keeping: **attention is not demand.** Attention is a mechanism for concentrating demand into a window. Whether that window is a good moment to buy depends entirely on what is unlocking into it, and that is a number, not a vibe. When you cannot find the number, the honest conclusion is not "probably fine" — it is that you are being asked to take a position you cannot price.

Two later posts in this series pick up threads left open here: the anatomy of a token pump treats the coordinated event itself in detail, and narrative cycles and who sets the story steps back to ask who decides what the market is excited about in the first place. For the supply side of the same machine, [how VCs move price through listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) and [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) are the natural next reads.
