---
title: "Gold ETFs and the GLD Machine: How Paper Gold Tracks the Metal"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How a gold ETF turns a vault of bullion into a stock ticker, why its price stays glued to spot, and what you do and don't own when you buy a share."
tags: ["gold", "gold-etf", "gld", "paper-gold", "creation-redemption", "authorized-participant", "expense-ratio", "arbitrage", "custody", "how-to-own-gold"]
category: "trading"
subcategory: "Gold"
author: "Hiep Tran"
featured: true
readTime: 40
---

When you buy a share of GLD, no one mints a coin. There is no jeweller, no mint, no truck. You tap a button in a brokerage app, a number changes, and you are told you now own "gold." But somewhere in a London vault, the chain you just tugged may have ended with a bank physically wheeling a 400-ounce bar onto a pallet and signing it over to a trust. That hidden plumbing — invisible from your screen — is the entire reason the price you paid tracks the metal at all.

> [!important]
> **TL;DR** — A gold ETF turns a vault of bullion into a stock-ticker you can buy in a second, and a creation/redemption mechanism is what keeps its price glued to spot gold.
>
> - A share of GLD represents a fixed, slowly shrinking slice of real vaulted gold — roughly 1/10 of an ounce — not a promise or an IOU.
> - The price tracks spot because **authorized participants** (big banks) can swap baskets of shares for physical bars and back again, arbitraging away any gap for a riskless profit.
> - You pay for that liquidity with an **expense ratio** (0.40%/yr for GLD) that quietly sells off a sliver of the gold each year, so each share backs fewer ounces over time.
> - The trade-off is real: an ETF gives you gold's *price* with stock-like convenience, but inserts a custodian into a chain that physical metal in your hand does not have.
> - The one number to remember: **0.40%/yr** is the fee that, over a decade, converts about 3.9% of your ounces into the fund sponsor's revenue.

Most arguments about "paper gold" generate more heat than light. One camp insists that gold ETFs are an unbacked confidence trick — claims stacked on claims, with no real metal behind them. The other camp treats GLD as simply "gold, but easier," as if the vault and the bar in your safe were interchangeable. Both are wrong in instructive ways, and the way to see why is to follow the actual machinery: what happens, step by step, between your buy order and a bar in a vault.

This post is about that machinery. In [the demand map](/blog/trading/gold/who-owns-gold-the-demand-map-jewelry-bars-etfs-central-banks-industry) we treated ETF flows as one bucket of investment demand — a few thousand tonnes that swings with Western sentiment. Here we go inside the bucket. By the end you will understand exactly why a screen price tracks the metal, what you own when you own a share, and when an ETF is genuinely the right tool versus when it quietly fails the one job some people buy gold to do.

![The GLD machine: investor buys shares, an authorized participant delivers bars to the vault, shares are created, and arbitrage keeps the price equal to spot](/imgs/blogs/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal-1.png)

The spine of this whole series is that gold is not an investment that compounds — it is a monetary insurance policy, a vote of no-confidence in paper money. A gold ETF is a clever financial wrapper around that insurance policy. The wrapper is brilliant at one thing (getting you exposure to gold's price cheaply and instantly) and, by construction, weaker at another (being no one's liability). Holding both ideas at once is the key to using ETFs well.

## Foundations: what an ETF actually is, and what "backed by physical" means

Let us build this from absolutely nothing, because every later piece of machinery rests on these definitions.

An **ETF** — exchange-traded fund — is a pooled investment vehicle whose shares trade on a stock exchange like an ordinary stock. "Pooled" means many investors' money is gathered together to buy one common pile of assets. "Exchange-traded" means that, unlike a traditional mutual fund (which you can only buy or sell once a day at a price struck after the market closes), an ETF share trades continuously through the day at whatever price buyers and sellers agree on, the same way Apple or Toyota stock does.

So far that describes thousands of funds. What makes a **gold ETF** specific is the pile: instead of holding a basket of stocks or bonds, the fund holds *physical gold bars*. The largest, SPDR Gold Shares (ticker GLD), holds nothing but gold and a tiny cash buffer. When the fund is described as "backed by physical," it means precisely this: for every share outstanding, there is a corresponding, auditable quantity of real bullion sitting in a vault. The fund is not betting on gold, lending out gold, or holding a contract that pays *like* gold. It owns the metal.

A few terms we will lean on:

- **Spot price**: the price for immediate delivery of one troy ounce of gold in the wholesale market, quoted in US dollars per ounce. As of early 2026 this is around \$5,400/oz after January's all-time high near \$5,589. (A **troy ounce** is the gold-trade unit, about 31.1 grams — slightly heavier than the kitchen ounce. We covered this in [how gold is priced](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce).)
- **Net asset value (NAV)**: the per-share value of what the fund actually owns. If a fund holds 1,000,000 ounces of gold and has 100,000,000 shares outstanding, each share is backed by 0.01 oz, and at a \$5,400 spot the NAV per share is \$54. NAV is the *intrinsic* worth of a share; the *market* price is whatever it trades for on the exchange, which can briefly differ.
- **Expense ratio**: the annual fee the fund charges, expressed as a percentage of assets. GLD's is **0.40%/yr**. There is no separate bill — the fee is paid by the fund itself, by quietly selling a sliver of its gold. We will trace exactly how that erodes your ounces later.

One number anchors the intuition for the rest of the post. GLD was designed so that, at launch, **one share represented one-tenth of an ounce of gold**. That 1/10-oz design is the human-scale handle on the whole machine: a share is a small, standardized, tradeable claim on a specific slice of metal. Over time that slice shrinks slightly (the fee), but the mental model — *a share is a fraction of an ounce of real gold in a vault* — is correct and worth holding onto.

#### Worked example: what \$10,000 of GLD represents in ounces

Suppose spot gold is \$2,400/oz (a level it traded near in 2024) and a GLD share represents almost exactly 1/10 oz, so NAV per share is about \$240. You buy \$10,000 of GLD.

- Shares purchased: \$10,000 / \$240 = **41.67 shares**.
- Ounces represented: 41.67 shares × 0.10 oz/share = **4.167 oz** of gold.
- Cross-check against spot directly: \$10,000 / \$2,400 = **4.167 oz**. The two agree, because the share is just a packaged tenth-ounce.

At the early-2026 spot of \$5,400/oz, that same \$10,000 buys only \$10,000 / \$5,400 = **1.85 oz** — the metal got more expensive, so your dollars buy fewer ounces, exactly as they would at a coin dealer. The intuition: buying an ETF is buying ounces at spot, in tenth-ounce units, minus a small fee.

The reason "backed by physical" matters — and the reason this series cares — is that gold's whole job is to be *the asset that is no one's liability*. A bank deposit is a claim on a bank; a bond is a claim on a borrower; a dollar is a claim on a central bank's credibility. Gold in a vault is just gold. An ETF that genuinely owns bars preserves most of that property. An "ETF" that instead held gold *futures*, or *swaps*, or *unallocated claims on a bullion bank* would be a different and weaker animal — a claim on a counterparty wearing a gold costume. The distinction between owning the metal and owning a promise is the hinge the whole post turns on.

### A short history: why GLD existed at all

It is worth pausing on how new this machine is, because it reframes the "paper gold" suspicion. For most of history, owning gold meant owning *metal* — coins, bars, jewellery — or owning a *paper claim on a bank's gold* that, often as not, the bank could not honour. The middle ground we now take for granted, "a transparent, exchange-listed, fully-allocated claim on real bars that any retail investor can buy in a second," did not exist until the 2000s.

The first modern gold ETF launched in Australia in 2003; SPDR Gold Shares (GLD) listed in the United States in November 2004 and was, at the time, a genuine financial innovation. Before it, an American who wanted gold exposure faced a bad menu: buy coins and pay dealer spreads and storage; trade futures and manage margin and rolls; or buy gold-mining stocks and accept that you were buying a *company*, not the metal (we cover why miners disappoint in [the miners post](/blog/trading/gold/gold-miners-the-leveraged-and-disappointing-bet-on-the-gold-price)). GLD collapsed that friction. Within a few years it held hundreds of tonnes; at the 2020 peak the global ETF complex held nearly 3,909 tonnes — more gold than all but a handful of *national* reserves.

That scale is the point. Gold ETFs took a slice of demand that used to be impossible to express cleanly and turned it into a measurable, real-time flow. Part of why central banks and analysts watch ETF holdings so closely is that, before 2004, *there was no such number* — Western financial demand for physical gold was diffuse and unobservable. The ETF didn't just make gold easier to buy; it made a whole category of gold demand *visible.* Far from being "fake" gold, the physical ETF is one of the most transparent gold-ownership structures ever built — its bar list is public to the serial number, which is more than most central banks disclose about their own reserves.

## The creation/redemption mechanism: how shares come into and go out of existence

Here is the question that unlocks everything: if demand for GLD shares surges, where do new shares come from? A company can't conjure GLD shares out of thin air, because each one must be backed by real gold. And if everyone sells, where do the shares go?

The answer is a process called **creation and redemption**, run by a special class of firms called **authorized participants (APs)**.

An **authorized participant** is a large financial institution — typically a major bank or market-making firm — that has a contract with the fund allowing it to do something no ordinary investor can: hand the trust physical gold in exchange for newly created shares, or hand the trust shares in exchange for physical gold. APs are the only entities with this privilege. You and I can only buy and sell *existing* shares on the exchange; APs operate the valve that changes the *number* of shares in existence.

They don't deal in single shares. Creation and redemption happen in large blocks called **baskets** (also "creation units"). For GLD, a basket is **50,000 shares** — at a \$5,400 spot and ~0.0926 oz per share in 2026, that is roughly 4,630 oz, or about \$25 million of gold. The mechanism is **in-kind**: the AP doesn't pay cash for the new shares, it delivers the *actual gold*. This in-kind design is not a detail; it is the cleverest part of the whole structure, and we will see why when we get to taxes and tracking.

The single most useful concept to hold here is the split between **two markets** that almost everyone conflates:

- The **secondary market** is where *you* live: shares changing hands between investors on the exchange, all day, at market prices. When you buy GLD, you are buying *existing* shares from another investor. No gold moves. The total share count doesn't change. It is just shares trading owners, exactly like buying a used car — the car already exists.
- The **primary market** is where *APs* live: shares being *created and destroyed* against physical gold delivered to or taken from the trust. This is where the share count actually changes and where metal actually moves into and out of vaults — like a factory building a new car or scrapping an old one.

Your trade and the AP's trade are decoupled in time. On a quiet day, billions of dollars of GLD can change hands in the secondary market with *zero* creation or redemption — investors simply pass existing shares around. The primary market only fires when the secondary-market price drifts far enough from the gold value that an AP's arbitrage becomes profitable. So "I bought gold today" (you, secondary market) and "a bar went into the vault today" (an AP, primary market) are related but not simultaneous — the second happens only when the aggregate of the first pushes the price out of line. This is why a flood of buying *eventually* pulls metal into vaults, but not tick-for-tick.

![Creation and redemption shown as two pipelines: gold in and shares out on top, shares in and gold out below](/imgs/blogs/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal-2.png)

Walk the **creation** leg (the top row of the figure), step by step:

1. An AP sees strong demand for GLD shares — buyers are bidding the share price slightly above what the underlying gold is worth.
2. The AP buys the corresponding amount of physical gold in the wholesale market (say one basket's worth, ~4,630 oz).
3. The AP delivers those bars **in-kind** to the trust's custodian. No cash changes hands for the shares; metal goes in.
4. The trust creates **50,000 brand-new shares** and gives them to the AP.
5. The AP sells those new shares on the exchange to the public, pocketing the small premium that motivated the trade.

The net effect: gold went into the vault, the share count rose by 50,000, and the extra selling pressure from the AP nudged the share price back down toward fair value. **The fund grew because real metal was added** — not because a printing press ran.

**Redemption** (the bottom row) is the mirror image, run when shares trade *below* the value of their gold:

1. The AP buys 50,000 cheap shares on the exchange.
2. It hands that basket back to the trust.
3. The trust cancels those 50,000 shares and releases the corresponding ~4,630 oz of physical gold from the vault.
4. The AP walks away with bullion (which it can sell at the higher spot price, or keep).

The net effect: gold left the vault, the share count fell by 50,000, and the AP's buying pushed the share price back up toward fair value. **The fund shrank because real metal was removed.**

This is why the total gold held by all ETFs rises and falls in tonnes, in real time, with investor demand. When you read "gold ETFs added 90 tonnes last month," that is the sum of thousands of creation baskets — APs physically moving metal into vaults because, share by share, the public was buying. The tonnage *is* the demand, made of bars.

#### Worked example: the profit an AP captures on a 0.3% premium

Suppose GLD shares are trading at a 0.3% premium to NAV — the market wants gold exposure badly enough to overpay slightly. Take one basket of 50,000 shares with a NAV of \$500 per share (so each share is "worth" \$500 of gold).

- Fair value of the basket: 50,000 × \$500 = **\$25,000,000** of gold.
- Market value of the basket at +0.3%: \$25,000,000 × 1.003 = **\$25,075,000**.
- The AP buys \$25,000,000 of physical gold, delivers it in-kind, receives 50,000 new shares, and sells them into the premium for \$25,075,000.
- Gross arbitrage profit: \$25,075,000 − \$25,000,000 = **\$75,000** per basket, before its own trading and financing costs.

The AP is not gambling on the gold price — it is long the metal it just bought and short the shares it just sold, a hedged pair. It is harvesting the *gap*, and the act of harvesting it (buying gold, selling shares) is exactly what closes the gap. That self-erasing incentive is the engine of the next section.

### Why "in-kind" is the secret ingredient

The detail that makes the whole structure efficient is that creation and redemption are **in-kind** — gold for shares, not cash for shares. This sounds like a technicality. It is doing two heavy jobs.

First, **it offloads the work and the cost onto the AP.** When you buy a normal cash-settled mutual fund, the fund manager has to go into the market with your cash and buy assets, paying spreads and moving prices, and every other shareholder bears a slice of that cost. With in-kind creation, the *AP* sources the physical gold, bears the spread, and hands the fund finished bullion. The fund never trades to grow; it just receives metal. That keeps the fund's internal costs minimal — one reason a gold ETF can charge as little as 0.40%/yr (or less for competitors) and still track tightly.

Second, **it is highly tax-efficient in the United States** for the fund structure, because in-kind exchanges don't force the fund to *sell* gold to meet redemptions. A cash-redeeming fund that had to sell appreciated gold to pay departing holders would realize capital gains and distribute them to *everyone still in the fund* — taxing the loyal for the exits of others. In-kind redemption sidesteps that: the AP hands in shares and takes out bars, no sale occurs inside the fund, no gain is realized at the fund level. (Your *own* eventual sale of the shares is taxable to you — and note that in the US, physical-gold ETFs are typically taxed as **collectibles** at up to 28%, not the lower long-term capital-gains rate, a wrinkle worth knowing before you assume an ETF is tax-equivalent to a stock.) The point is that the in-kind plumbing keeps the *fund* clean, so tracking stays tight and costs stay low.

### Who are the APs, and what keeps them honest?

APs are not a mysterious cartel; they are the same global bullion banks and market makers that already trade physical gold in size — the firms with vault relationships, refiner access, and the balance sheet to warehouse \$25 million baskets. A fund typically has several APs under contract, and they compete: whichever AP can source gold cheapest and move a basket fastest captures the arbitrage. That competition is what compresses the tracking band toward the *lowest-cost* AP's expenses rather than the average — more APs means tighter tracking.

What keeps an AP from cheating — say, creating shares without delivering the gold? The structure forbids it mechanically: the trust will not issue a creation basket until the custodian confirms the metal is *in the vault.* Shares cannot exist ahead of their gold. This is the structural answer to the "unbacked paper" fear: there is no step in the process where an AP gets shares on credit. Gold first, then shares. The vault is the gatekeeper.

#### Worked example: why creation baskets are huge, and what that means for you

Why 50,000 shares (~\$25 million), not one share at a time? Because moving physical gold has fixed costs — assaying, transport, insured vaulting, custodian paperwork — that only make sense in size, and because the standard wholesale bar is the **400-oz Good Delivery bar** worth, at a \$5,400 spot, about \$2.16 million each.

- One basket at ~4,630 oz / 400 oz per bar ≈ **11–12 Good Delivery bars** per creation. You cannot sensibly create against a fraction of a bar, so baskets are sized in whole-bar multiples.
- This is why retail redemption is impossible: your 1.85 oz from the earlier example is **0.5% of a single bar.** There is no physical way to hand you "your" metal short of melting and re-pouring — which is exactly why the right to redeem lives only with APs trading full baskets.

The intuition: the ETF is wholesale plumbing (bars and millions) with a retail faucet (shares and dollars) bolted on; you drink from the faucet, but the pipes only move water by the tanker-load.

## Why the ETF price tracks spot: the arbitrage that closes any gap

You now have everything needed to see why a GLD share, trading freely on an exchange driven by retail sentiment, nonetheless never strays far from the price of gold itself. The answer is not regulation, goodwill, or a promise. It is **arbitrage** — riskless profit that exists only as long as a price gap exists, and which disappears the instant someone takes it.

![The arbitrage tree: when the share trades above or below spot, an authorized participant creates or redeems until the gap closes](/imgs/blogs/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal-3.png)

Read the figure top to bottom. At the top is fair value — the NAV, which is just *gold per share × spot price*. The share price can wobble away from it in two directions:

**If the share trades above spot (a premium).** Buyers are paying \$1.003 for \$1.00 of gold. An AP buys \$1.00 of physical gold, creates a share, and sells it for \$1.003 — locking in the \$0.003. Critically, *creating* a share means adding supply: more shares chasing the same demand pushes the share price down. APs keep doing this until the premium is gone and the share is back at NAV.

**If the share trades below spot (a discount).** Sellers are dumping a \$1.00 slice of gold for \$0.997. An AP buys the cheap shares, redeems them for the full \$1.00 of physical gold, and keeps the \$0.003. *Redeeming* shares means removing supply: fewer shares for the same demand pushes the share price up. APs keep doing this until the discount is gone.

Either way, the very action that earns the AP a profit is the action that erases the mispricing. This is why gold ETFs from reputable sponsors track spot almost perfectly — typical premiums and discounts are a few hundredths of a percent, well inside trading costs, and they snap shut in minutes. The peg is enforced not by trust but by greed, which is far more reliable.

There is a deep point here that links to the whole "paper vs physical" debate. The reason this arbitrage *works* is that the shares are genuinely, physically redeemable for metal by the APs. The redemption option is the leash. A "gold" product that could *not* be redeemed for real gold — a closed structure, a synthetic, a fund holding only futures — would have nothing tethering its price to the metal except sentiment, and sentiment can untether. The physical backing is not decoration; it is the mechanism. **An ETF tracks gold because, at the margin, it can always be turned back into gold.**

The result, over years, is the picture below: the ETF share value and spot gold are almost the same line, with one slowly widening difference.

![Spot gold and an ETF share value from 2015 to 2026 on a log scale, the two lines nearly identical with a small widening fee gap](/imgs/blogs/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal-7.png)

The solid blue line is spot gold, rising from roughly \$1,160 (2015 average) to \$5,414 (the 2026 spike). The dashed amber line is the value of the gold backing a share that started fully invested in 2015 and lost 0.40%/yr to fees. They sit nearly on top of each other for the whole eleven-year span — *that closeness is the tracking.* The only daylight between them is the fee: by 2026 the cumulative drag opens a gap of a couple of hundred dollars on a \$5,400 ounce. That is the single, honest cost of the wrapper, and it is the topic of the next section. Note what the chart does *not* show: any meaningful day-to-day divergence, any "the ETF wandered off from gold." It can't, because of the arbitrage — the wrapper's only failure relative to the metal is the slow, predictable fee, not tracking error.

#### Worked example: how wide can the gap get before an AP acts?

An AP only bothers when the gap exceeds its own costs: the bid-ask spread on physical gold, financing, and the operational cost of moving a basket. Say those costs total 0.15% for a given AP on a given day.

- The share would have to trade more than **0.15% away** from NAV before arbitrage is profitable.
- On a \$500 NAV share, that is a band of about \$500 × 0.0015 = **\$0.75** above or below — roughly \$499.25 to \$500.75.
- Inside that band the gap persists (no free money); outside it, APs pile in and slam it shut.

So the tracking is not literally perfect — it is "perfect to within the cheapest AP's transaction costs." For a liquid fund like GLD, that band is so tight that for any practical purpose, the screen price *is* spot. The intuition: the ETF price is leashed to gold, and the leash is exactly as long as it costs to arbitrage.

### The three things that *can* move the share off spot

The arbitrage pins *intraday market price to NAV*, but two layers can still differ, and it helps to separate them cleanly:

1. **Premium/discount to NAV** — the gap between the *share's market price* and the *value of its gold right now.* This is what arbitrage attacks, and for a liquid fund it is a rounding error (a few hundredths of a percent) that closes in minutes. It widens only when arbitrage is temporarily hard: an illiquid fund with few APs, a market dislocation when bullion itself is hard to source, or a halt in creations. The COVID-March-2020 days briefly stressed even gold's plumbing — vaults and refiners shut, flights grounded, and physical bars commanded unusual premiums — and some gold products showed wider-than-normal gaps for a few sessions until the metal could move again. That episode is the exception that proves the rule: the peg held everywhere the arbitrage could still physically run.
2. **The fee drag on NAV itself** — NAV per share falls slightly each year as the fund sells gold to pay its fee. This is *not* a market mispricing; it is the fund genuinely owning less gold per share. Arbitrage cannot fix it because there is nothing to fix — the share faithfully tracks a *slowly shrinking* slice of gold. This is the cost we quantify next.
3. **Currency, if you're not a dollar investor** — a gold ETF priced in dollars tracks *dollar* gold. If your home currency is the euro, the yen, or the dong, your return is dollar-gold *times* the currency move. That is not a flaw in the ETF; it is the same currency exposure you'd have holding any dollar-priced gold, and we unpack it in [gold and the dollar](/blog/trading/gold/gold-and-the-dollar-the-inverse-relationship-and-when-it-breaks). But it means a Vietnamese or European investor's GLD return can differ from the local gold price by the FX swing.

Keep these three straight and you will never be confused by an ETF "not matching gold": day to day it matches NAV (item 1, near-perfectly); year to year it lags the metal by the fee (item 2); and across borders it carries the dollar (item 3).

## Expense drag: why an ETF slowly loses ounces

Now the cost. The arbitrage machine is not run for charity, and neither is the fund. The sponsor charges an **expense ratio** — GLD's is 0.40% of assets per year — and the way it collects is subtle and important: **the trust sells a tiny amount of its own gold to pay the fee.** No invoice arrives. Instead, the quantity of gold backing each share drifts *down* over time.

This is why the "1/10 of an ounce per share" figure was true at launch and is no longer exactly true. At inception in 2004, one GLD share = 0.100 oz. By 2026, after two decades of fee accrual, one share represents closer to **0.0926 oz** — the missing ~7.4% of an ounce was sold, over the years, to fund the 0.40% annual charge (the cumulative drag compounds, and rebalancing rounds it).

![Expense drag: the ounces backing one ETF share erode each year at the fund's expense ratio, with 0.40 percent falling faster than 0.15 percent](/imgs/blogs/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal-5.png)

The figure makes the mechanism literal. Start with a share backed by 100 thousandths of an ounce. Each year the trust shaves off the fee, so the ounces per share follow `0.10 × (1 − fee)^years`. The amber line (0.40%/yr, GLD) sinks faster than the blue line (0.15%/yr, a hypothetical low-cost competitor). After ten years the GLD-style share is down to about 96.1 thousandths; the cheaper one is still near 98.5. The metal didn't move and the gold price is irrelevant to this chart — this is purely the fee eating ounces.

Crucially, **this is a drag on ounces, not a tracking error on price.** Your share still tracks spot tightly day to day (the arbitrage guarantees that). But the *quantity of gold each share is a claim on* slowly falls, so your ETF underperforms the metal itself by roughly the expense ratio per year. Hold gold coins in a drawer and you own the same ounces forever (ignoring the one-time buy/sell spread); hold GLD and you own slightly fewer ounces every year. That is the rent you pay for the liquidity and convenience.

#### Worked example: the dollar cost and ounce-erosion of a 0.40% fee over ten years

Put \$50,000 into GLD and hold it for a decade, assuming for clarity that the gold price stays flat (so we isolate the fee).

- Year-one fee: 0.40% × \$50,000 = **\$200**. That is the annual rent, paid invisibly by the trust selling \$200 of gold.
- Over ten years at 0.40%/yr compounding, the fraction of your ounces remaining is (1 − 0.004)^10 = 0.9607, so you keep **96.07%** of the metal and lose **3.93%** to fees.
- In dollars (flat price): you started with \$50,000 of gold and end with \$50,000 × 0.9607 = **\$48,036** — about **\$1,964** total handed to the sponsor over the decade.
- Contrast a 0.15% fund: (1 − 0.0015)^10 = 0.9851, so you'd keep 98.51% and lose only ~\$745. The 0.25%/yr difference in fee compounds into more than \$1,200 over ten years on this position.

The intuition: an ETF's fee is small per year but it nibbles your actual gold, so a non-yielding asset held in an ETF *shrinks* slightly each year — exactly the opposite of the no-yield asset's appeal, which is why fee selection matters more for gold than for almost anything else.

That last point deserves a beat. For a stock fund, the expense ratio is paid out of a stream of dividends and growth; the fee is a haircut on a *return*. Gold has no return — no coupon, no dividend, no earnings. (We dug into this in [the no-yield problem](/blog/trading/gold/the-no-yield-problem-how-a-metal-that-pays-nothing-can-be-worth-anything).) So a gold ETF's fee is not a haircut on a yield you're earning; it is a slow liquidation of the asset itself. A 0.40% fee on a stock fund returning 8% is shaving an eighth of your return. The same 0.40% on gold is shaving your principal. That asymmetry is why, for a long-term gold holding, the gap between a 0.40% fund and a 0.10–0.15% fund is worth caring about — and why some investors, for very long horizons, prefer the metal.

## What you do and don't own: the custody chain

We have established that a share is a real claim on real gold. Now the uncomfortable part: *whose* gold, held by *whom*, and what happens in the seams.

When you own GLD, you own **shares in a trust.** The trust owns the gold. You do not own specific bars; you do not have an account at the vault with your name on a stack of metal. You have an undivided beneficial interest in a pool. Trace the chain:

- **You** → own shares of
- **the trust** (a passive legal entity that exists only to hold gold and issue shares) → which holds gold via
- **the custodian** (for GLD, historically HSBC in London) → who stores the bars in
- **a vault**, in **allocated** form (specific, serial-numbered 400-oz Good Delivery bars assigned to the trust, not commingled IOUs).

That word **allocated** is the crux, and it connects directly to [the allocated-vs-unallocated discussion](/blog/trading/gold/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk) for physical metal. *Allocated* gold means specific, identified bars that legally belong to the holder — the custodian is merely storing them, and in a custodian bankruptcy they are not part of the bank's estate. *Unallocated* gold is just a claim on a bank's gold account — an unsecured creditor position. Reputable physical-gold ETFs hold **allocated** bars, and publish bar lists with serial numbers and weights, precisely so that the trust's gold is *the trust's gold* and not the custodian's balance sheet.

So what do you actually own? You own a fractional, undivided, beneficial interest in a pool of allocated bars held by a named custodian for a passive trust. That is a genuinely strong claim — much stronger than a futures contract or an unallocated bank account. But notice what it is *not*: it is not metal in your hand. The chain has links — trust, custodian, vault — and each link, however well-built, is a place where law, jurisdiction, and counterparty solvency enter. The whole reason a slice of humanity buys gold is to *escape* counterparty chains. An ETF gives you most of gold's price behavior while quietly reintroducing a (short, robust, but non-zero) chain.

It helps to rank the common gold claims by *how much you depend on someone else staying solvent and honest* — because that dependence is precisely the thing gold is supposed to minimize:

- **Coin or bar in your own possession**: zero counterparty. If every bank on earth failed tomorrow, the metal in your hand is unaffected. This is the purest form of gold's promise — and the most inconvenient and least liquid.
- **Allocated physical ETF (GLD and peers)**: low counterparty. Your gold is real, identified, and bankruptcy-remote from the custodian, but you depend on the trust being administered honestly, the custodian (and sub-custodians) storing faithfully, and your broker holding your shares — a short chain of well-regulated, audited institutions.
- **Unallocated gold account at a bank**: high counterparty. You own an *IOU* for gold, ranking as an unsecured creditor; if the bank fails, you queue with everyone else. This is "gold" in name, a bank claim in substance.
- **Gold futures**: counterparty plus leverage. You depend on the exchange's clearinghouse and your broker, and you can be forced out by a margin call before your thesis plays out, even if you're ultimately right.

Read top to bottom, that is a ladder from *pure money* to *pure promise*. The ETF sits high on the ladder — much closer to the metal than to the IOU — which is why "unbacked paper" is a slur it doesn't deserve. But it is one rung *below* metal-in-hand, and for the buyer whose entire reason for owning gold is to be on the *top* rung, that one rung is the whole point.

There is one more wrinkle worth naming: **sub-custodians.** The named custodian (HSBC, say) does not always hold every bar in its own vault; it may use sub-custodians — typically other bullion banks or the Bank of England — to store some of the metal, especially when holdings are very large. Reputable funds disclose this and the bar list still identifies the metal, but each additional storer is, strictly, another link. The chain for the careful reader is therefore: you → trust → custodian → (possibly) sub-custodian → vault. Still allocated, still bar-listed, still robust — but it is honest to acknowledge that the chain has a few more links than "a coin in your drawer," which has exactly zero.

This is the honest trade-off, and it's worth stating plainly rather than either dismissing or catastrophizing. For an investor who wants gold's *price* — as a portfolio diversifier, an inflation-regime hedge, a tactical macro position — the custody chain is a theoretical risk that has never failed for a major physical ETF, and the liquidity and cost advantages are enormous and real. For someone whose entire thesis is *"I want metal that is no one's liability, accessible if the financial system itself seizes up"* — the doomsday-insurance buyer — an ETF does not do that job, because in a true system-wide crisis your access runs through brokers, the trust, and a custodian bank, all of which are *parts of the system you're insuring against.* Same metal, different promise.

The phrase to retire here is "your gold isn't your gold," which gets thrown at ETFs as if it were a gotcha. It is *literally* true in a narrow sense — you own shares in a trust, not assigned bars — but it conflates two very different failure modes. *Unallocated* gold (a bank IOU) really can vanish in a bankruptcy: you're an unsecured creditor. *Allocated* gold held by a passive trust for shareholders is bankruptcy-remote: it is not the custodian's asset to lose. The ETF's gold *is* the shareholders' gold, collectively and undivided. What you don't have is the right to *take physical possession of a specific bar* — and for the buyer who needs exactly that, the ETF is the wrong tool, full stop. The criticism is valid against the specific job of "metal in hand," and invalid as a claim that the gold is at risk of disappearing.

## ETFs vs physical vs futures: the trade-offs

Step back and put the three main ways to own gold side by side. Each is a different bundle of liquidity, counterparty exposure, cost, deliverability, and convenience — and there is no free lunch, only a choice of which axis to optimize.

![Scorecard comparing physical gold, the ETF, and futures across liquidity, counterparty, annual cost, deliverability, and convenience](/imgs/blogs/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal-6.png)

- **Physical metal** (bars and coins): zero ongoing counterparty risk once it's in your hand, fully deliverable (you *have* it), but illiquid (you must find a buyer, often a dealer who pays under spot), expensive to transact (dealer spreads of several percent plus storage and insurance), and inconvenient (you have to store and secure it).
- **The ETF** (GLD and peers): superb liquidity (trade a million dollars in one second at a tight spread), high convenience (it sits in the same brokerage account as your stocks), and a low *ongoing* cost relative to physical's spreads — but it carries a (short, robust) custody chain, charges 0.40%/yr that erodes your ounces, and is **not deliverable** for ordinary holders (you can't ring HSBC and ask for your bar).
- **Futures** (COMEX contracts): the deepest liquidity and the cheapest way to get *leveraged* exposure, deliverable in 100-oz lots if you take it to expiry — but they carry exchange/broker counterparty risk, margin calls, and a **roll cost** as near contracts expire and you re-buy farther-dated ones. We cover the basis, contango, and the paper-vs-physical question in [the futures post](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical).

The scorecard's lesson is that the ETF occupies a specific, sensible niche: it is the **low-friction way to own gold's price** for someone who measures gold in dollars and wants to buy, hold, and rebalance it inside a normal investment account. It deliberately trades away the no-counterparty purity of metal-in-hand and the leverage of futures, in exchange for liquidity and convenience. Whether that is the right trade depends entirely on *why* you are buying gold — which is the question we keep returning to.

#### Worked example: physical's 6% round-trip vs the ETF's 0.40%/yr — the break-even horizon

Physical gold typically costs a few percent to buy (dealer premium over spot) and a few percent to sell (dealer pays under spot) — call it a **6% round-trip** all-in. The ETF costs essentially nothing to trade (a penny spread) but **0.40%/yr** to hold. When does the ETF's recurring fee overtake physical's one-time spread?

- Physical's total friction (held forever, transacted once): **6%**, paid up front and on exit.
- ETF's friction after `n` years: roughly 0.40% × n (ignoring compounding for a quick estimate), plus a near-zero trading spread.
- Break-even: 0.40% × n = 6% → n = **15 years**.

So if you intend to hold for *more than ~15 years* and never trade, the metal's one-time 6% can actually be cheaper than the ETF's accumulating fee — a real argument for physical among true long-term, buy-and-bury holders. But if your horizon is shorter, or you plan to rebalance, the ETF's frictionless trading wins easily. The intuition: pay a big one-time toll (physical) only if you'll drive the road once and stay; pay a small recurring meter (ETF) if you'll come and go.

## Common misconceptions

The "paper gold" debate is a swamp of half-truths. Three myths in particular cause people to either over-trust or over-fear ETFs. Each falls to a number.

**Myth 1: "GLD is unbacked paper — there's no real gold behind it."** This is the most common and the most wrong for a *physical* ETF. GLD publishes a daily, auditable bar list — serial numbers, refiner, weight — for thousands of allocated 400-oz bars held in London, totaling on the order of a thousand-plus tonnes of metal. That is not a promise; it is an inventory you can read. The fund's gold holdings move in measured tonnes precisely because creation and redemption put physical bars in and take them out. The confusion comes from conflating physical ETFs with leveraged or synthetic "gold" products that hold *futures or swaps* — those genuinely are claims-on-claims, and they are a different category. For a physical, allocated, bar-listed ETF, "unbacked" is simply false. The honest critique is not "there's no gold," it's "the gold is in a custody chain you don't personally control" — which is the trade-off we already named, not a fraud.

**Myth 2: "An ETF share is a claim I can always redeem for my metal."** False for ordinary investors — and this is the genuinely important one to internalize. **Only authorized participants** can redeem shares for physical gold, and only in 50,000-share baskets (~\$25 million). You, the retail holder, can never walk up to the trust and demand your 1.85 ounces in bar form. You can only *sell your shares for cash* on the exchange. The redeemability that anchors the price is a *wholesale* feature, not a retail right. This matters because it punctures the fantasy that GLD is "metal you can grab in a crisis." It is exposure to gold's *price*, redeemable for *dollars*. If physical possession is your actual goal, the ETF does not deliver it.

**Myth 3: "The ETF sets the gold price — paper drives the metal."** Backwards, for physical ETFs. The arbitrage runs the other way: APs *respond* to gaps between the share price and spot by moving physical metal, which drags the share back to spot. The causality is metal → share, enforced by redemption. The ETF is a *price-taker* that imports the wholesale gold price onto a stock exchange. (There is a longer, more legitimate debate about whether the vastly larger *futures and over-the-counter* paper markets — many multiples of annual mine supply in daily turnover — influence short-term spot. That is a real question, and we take it up in the futures post. But the *ETF* is not the tail wagging the dog; it is a faithful mirror, leashed to spot by the redemption mechanism.)

A fourth, quieter myth deserves a mention: **"All gold ETFs are the same."** They are not. Some hold allocated physical and publish bar lists (the strong kind); some hold *unallocated* gold (a bank claim); some hold futures (no metal at all); some are *leveraged* (2× or 3× *daily* moves, which decay badly if held). The label "gold ETF" spans claims of very different strength. Read what the fund actually holds before deciding what you own.

## How it shows up in real markets: the Western swing buyer

The cleanest way to see ETF mechanics in the wild is to watch total ETF gold holdings over a full cycle — because every tonne in that line is a creation basket, and every tonne out is a redemption. The line is a thermometer for *Western investment sentiment toward gold.*

![Gold-backed ETF holdings from 2015 to 2026, showing the 2020 COVID peak, the 2021 to 2024 drain, and the 2025 to 2026 return](/imgs/blogs/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal-4.png)

Three acts tell the story:

**2020 — the great inflow.** As COVID hit and central banks slashed rates to zero and launched [enormous quantitative easing](/blog/trading/finance/quantitative-easing-explained-printing-money), real yields collapsed and Western investors poured into gold for safety and as an inflation hedge. Global ETF holdings surged to an all-time peak of roughly **3,909 tonnes in October 2020** — APs created baskets en masse, physically pulling thousands of tonnes of bars into vaults. The mechanism we described, running flat out: demand for shares → APs buy gold → vaults fill.

It is worth flagging an even earlier, sharper example of the same behaviour: **2013.** After the 2011 peak near \$1,921 and two more years of grinding sideways, sentiment finally broke. In April 2013 gold suffered a violent two-day crash, and Western ETF investors fled for the rest of the year. Global ETF holdings fell by roughly 800 tonnes in 2013 alone — the largest annual redemption on record at the time — as APs handed basket after basket back to the trusts and pulled metal out. That single year is the cleanest illustration of the ETF buyer's character: when the trend turned and real yields rose under the "taper" scare, the swing buyer left *en masse*, and the redemption mechanism processed it bar by bar. The price fell from a ~\$1,668 average in 2012 to ~\$1,411 in 2013 and kept sliding to ~\$1,160 by 2015. ETF flows didn't *cause* that decline so much as *express* it — the financial buyer voting with their feet.

**2021–2024 — the long drain.** The same character repeated, more slowly. Real yields *rose* — the Fed hiked aggressively in 2022, and the 10-year TIPS real yield went from about −1% to over +2%. Suddenly a non-yielding asset competed against bonds paying a real return, and Western ETF investors sold. Holdings fell from ~3,750 tonnes (year-end 2020) to about **3,219 tonnes by 2024** — roughly 530 tonnes redeemed out, bars physically leaving Western vaults. This is the redemption leg at scale.

Here is the deep point, and the one that links straight to this series' spine. *Gold's dollar price went up over this same period* — from ~\$1,770 (2020 avg) to ~\$2,388 (2024 avg) — **even as Western ETFs were dumping it.** How? Because a different buyer had stepped in: central banks, buying over 1,000 tonnes a year from 2022. (We tell that story in [the structural-buyer post](/blog/trading/gold/central-banks-the-structural-buyer-that-changed-gold-after-2022).) The ETF line and the price *decoupled*: the metal was being passed from price-sensitive Western financial investors to price-insensitive sovereign buyers diversifying out of dollars. The ETF flows show you the *Western financial* demand — and that demand was, for once, not setting the price.

**2025–2026 — the return.** As gold broke out to new records and momentum and fear returned, Western investors came back: holdings rebuilt toward ~3,650 tonnes (2025) and ~3,950 (2026), chasing the move. The ETF buyer is, classically, a *trend-follower and swing buyer* — heavy in at peaks of fear and momentum, out when real yields make cash and bonds look better. That behavioral signature is exactly why ETF flows are a sentiment gauge and not a price oracle.

#### Worked example: turning an ETF flow headline into bars and dollars

A headline reads: "Global gold ETFs added 90 tonnes in a month." Translate it.

- 90 tonnes = 90,000 kg. At 32,150.7 troy oz per tonne, that is 90 × 32,150.7 = **2,893,563 oz**.
- In 400-oz Good Delivery bars: 2,893,563 / 400 ≈ **7,234 bars** physically moved into vaults that month.
- At a \$5,400 spot, the dollar value created: 2,893,563 oz × \$5,400 ≈ **\$15.6 billion** of new gold demand, financed by APs and ultimately by the public buying shares.

The lesson to carry away: an ETF-flow number is not a vague sentiment gauge — it is a literal count of bars that authorized participants wheeled into (or out of) London vaults, one creation basket at a time, in response to which way the public leaned.

## The takeaway: when an ETF is the right tool

Strip away the noise and the gold ETF is a genuinely elegant piece of financial engineering. It takes the world's oldest no-counterparty asset and wraps it in a stock ticker, then uses a self-interested arbitrage loop to keep the wrapper's price honest to a hundredth of a percent. For the overwhelming majority of investors who want *exposure to gold's price* — to diversify a portfolio, to hold a position through a macro regime, to rebalance against stocks and bonds — it is the correct tool: liquid, cheap to trade, transparent, and backed by real allocated bars you can read off a list.

But the elegance has a seam, and the seam is the whole point of the series. Gold's deepest job — its reason for existing in a portfolio at all — is to be *the asset that is no one's liability,* the vote of no-confidence in paper money and the institutions that issue it. An ETF, by inserting a trust, a custodian, and a broker into the chain, *re-introduces exactly the kind of counterparty dependence that pure metal exists to escape.* For 99% of holders in 99% of conditions, that seam never matters, and the convenience dominates. For the holder whose thesis is specifically *system-wide failure*, the ETF quietly fails the one test that motivated the purchase — because in that scenario, the trust, the custodian, and the broker are all parts of the very system being doubted.

So the practical rule is not "ETF good" or "physical good." It is: **match the wrapper to the job.**

- If you want gold's *price* — diversification, a macro hold, a rebalanced sleeve in a portfolio (the case made in [gold's job in a portfolio](/blog/trading/gold/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio)) — an ETF is excellent, and you should optimize for the lowest expense ratio you can find, because on a non-yielding asset the fee eats principal.
- If you want gold's *insurance property* — metal that answers to no counterparty, available if the financial plumbing itself fails — then you want allocated physical you control, and you should accept the spread and storage as the price of that purity.
- And if you want *leverage or tactical trading*, you want futures, with eyes open to roll and margin.

If you do choose an ETF, four checks separate a good one from a trap, and all four are answerable from the fund's own page in five minutes:

1. **What does it actually hold?** *Allocated physical* with a published bar list (good); *unallocated* gold (a bank claim, weaker); *futures* (no metal); *leveraged/inverse* (a daily-reset trading tool that decays — never a hold). The label "gold ETF" hides all four.
2. **What is the expense ratio?** On a non-yielding asset, the difference between 0.40% and 0.10–0.15% compounds straight out of your ounces. Larger funds and newer entrants have pushed fees down; the most liquid is not always the cheapest.
3. **Where is the gold, and who custodies it?** A named custodian, a disclosed vault location, and disclosed sub-custodians beat vagueness. Some funds let *large* holders (not retail) redeem for metal, which some buyers value.
4. **Is it liquid enough that the spread won't eat you?** A penny-wide spread on a giant fund costs nothing; a wide spread on a tiny one is a hidden tax every time you trade. Liquidity is itself a feature you are paying the fee for — make sure it's there.

Get those four right and the ETF does its job beautifully. Get them wrong — buy a leveraged product and hold it, or a high-fee fund for a decade — and the wrapper quietly works against you.

The screen price tracks the metal because, somewhere, a bank can always turn a basket of shares back into bars. That hidden plumbing is what makes paper gold *real* — and understanding it tells you precisely the one thing the plumbing cannot do: put metal in your hand when the system you're hedging against is the system holding it.

## Further reading & cross-links

- [Who owns gold: the demand map](/blog/trading/gold/who-owns-gold-the-demand-map-jewelry-bars-etfs-central-banks-industry) — where ETF flows sit among jewellery, bars, central banks, and industry; this post goes inside the ETF bucket.
- [Physical gold: bars, coins, allocated vs unallocated, and counterparty risk](/blog/trading/gold/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk) — the no-counterparty alternative the ETF trades away; the allocated/unallocated distinction the ETF relies on.
- [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — the leveraged, deliverable, roll-cost cousin, and the larger paper-vs-physical debate.
- [Gold's job in a portfolio: sizing, rebalancing, and the permanent portfolio](/blog/trading/gold/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio) — when an ETF's liquidity is exactly the feature you want.
- [Central banks: the structural buyer that changed gold after 2022](/blog/trading/gold/central-banks-the-structural-buyer-that-changed-gold-after-2022) — the price-insensitive buyer that took the metal Western ETFs sold.
- [The no-yield problem](/blog/trading/gold/the-no-yield-problem-how-a-metal-that-pays-nothing-can-be-worth-anything) — why a fee on gold eats principal rather than a yield, and why expense ratios matter more here.
- External: [SPDR Gold Shares (GLD) prospectus and daily bar list](https://www.spdrgoldshares.com/) and [World Gold Council ETF flow data](https://www.gold.org/goldhub/data/gold-etfs-holdings-and-flows).
