---
title: "Gold Futures: COMEX, Contango, Backwardation, and Paper vs Physical"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How gold futures actually work — margin, leverage, the cost-of-carry, the roll, COMEX delivery — and what the famous paper-versus-physical ratio does and doesn't mean."
tags: ["gold", "futures", "comex", "contango", "backwardation", "cost-of-carry", "paper-vs-physical", "leverage", "derivatives", "basis", "roll", "delivery"]
category: "trading"
subcategory: "Gold"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A gold futures price is just today's spot price plus the cost of carrying metal to a future delivery date, and that small gap (the *basis*) is where leverage, the roll, and the whole "paper vs physical" debate live.
>
> - A 6-month future on a \$2,400 spot at a 4.5% financing rate sits near \$2,454. That \$54 is **carry — interest plus storage — not a forecast** of where gold is going. When the curve slopes up like this it is in **contango**.
> - When deferred futures trade *below* spot — **backwardation** — it is a real-time signal that someone needs metal *now* and will pay a premium to skip the wait.
> - One COMEX contract controls 100 oz, about \$240,000 of gold, on roughly \$12,000 of margin — **~20x leverage**. That is the real reason futures move the price: a little money commands a lot of metal.
> - Open futures claims dwarf the metal sitting in deliverable COMEX vaults, often by **3:1 to 5:1**. That ratio sounds like a scandal, but ~99% of contracts cash-settle or roll before expiry, so it almost never matters — *until a delivery scramble makes it matter*.

In March 2020 something happened that gold traders had read about but almost never seen. As COVID grounded passenger flights worldwide, the small belly-hold shipments that quietly move gold bars between London, Zurich, and New York stopped. London had metal; COMEX in New York needed metal to back its June futures. For a few wild days the price of a gold *future* in New York detached from the price of gold *in London* by tens of dollars an ounce — a spread that normally lives in the single digits. Refiners scrambled to recast 400-ounce London bars into the 100-ounce and kilobars that COMEX will accept for delivery. The entire precious-metals complex seized: silver's relationship to gold blew out to the widest reading in history.

For about a week, the abstraction everyone treats as a single thing — "the gold price" — visibly split into two prices: the price of a *promise* of gold and the price of *metal you can touch*. That gap is the subject of this whole post. Most of the time it is tiny and boring, a few dollars of financing cost. But it is never zero, and understanding why is the key that unlocks both how speculators move the gold price and whether the perennial gold-bug warning — *"there are far more paper claims than there is metal, the whole thing is a fraud waiting to collapse"* — is something to lose sleep over.

We covered the basis briefly when we looked at [how gold is priced](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce). Here we go all the way down: what a futures contract actually is, how COMEX is wired, why the curve slopes the way it does, what the roll quietly costs you, and what that scary paper-to-metal ratio really tells you.

![Spot price branches into an upward contango curve and a downward backwardation curve with labeled boxes](/imgs/blogs/gold-futures-comex-contango-backwardation-and-paper-vs-physical-1.png)

## Foundations: what a futures contract is, and why a price can have a slope

Start with the most everyday version of the idea. Imagine you agree today to buy your neighbour's car in six months for a price you both fix *now*. You have not paid yet; you have not received the car yet; you have simply locked a price for a future exchange. That is a forward agreement, and a **futures contract** is the same thing, standardized and traded on an exchange so that strangers can do it with each other safely.

A gold future is a binding agreement to buy (if you are *long*) or sell (if you are *short*) a fixed quantity of gold, of a defined purity, on a defined future date, at a price agreed today. Nobody hands over money for the metal at the start. The contract just fixes the terms. On the COMEX exchange — the dominant gold futures venue, part of the CME Group in Chicago — the standard contract, ticker **GC**, is for **100 troy ounces** of gold of at least 0.995 fineness.

There is a close cousin worth naming so we don't confuse the two. A **forward** is the private, over-the-counter version of the same deal: two parties — say a mining company and a bank — agree bilaterally on a price, a quantity, and a date, with no exchange in the middle and no daily cash settlement. London's enormous gold market runs largely on forwards and spot, not futures. A **future** is the *exchange-traded, standardized* forward: identical contract terms for everyone, a central clearinghouse guaranteeing both sides, and daily marking-to-market. The standardization is what makes futures liquid — because every GC contract is interchangeable, a buyer and seller who have never met can trade in milliseconds. The forward's flexibility (any size, any date) is the OTC market's advantage; the future's fungibility is the exchange's. For the rest of this post we focus on futures, but keep the distinction in your pocket: when you read that London does the physical heavy lifting and COMEX does the price discovery, this forward-vs-future split is part of why.

Two more pieces of vocabulary clear most of the fog. **Volume** is how many contracts trade hands in a day; **open interest** is how many contracts are *currently open* — positions that have been entered and not yet closed. They are different animals. A day of frantic trading where longs and shorts open and close all afternoon can have huge volume and barely move open interest. A quiet day where new money piles in and *stays* can have modest volume but rising open interest. Traders read the pair together: rising price on rising open interest means new money is committing to the move (a stronger trend); rising price on *falling* open interest means shorts are covering (a weaker, possibly exhausting move). We will lean on open interest again when we get to the paper-vs-physical ratio, because open interest *is* the count of paper claims.

The first thing to internalize is that a futures price is *not a prediction*. This trips up almost everyone new to the market. If gold spot is \$2,400 today and the December future trades at \$2,460, that \$60 is **not** the market betting gold will rise \$60 by December. It is something far more mechanical, and once you see it you can never unsee it.

### The cost-of-carry: the engine behind the whole curve

Here is the logic. Suppose you want to *own* gold in six months. You have two ways to get there:

1. **Buy the metal today** for \$2,400, stick it in a vault, pay a small storage fee, and wait. But notice: that \$2,400 is now tied up in a brick of metal instead of sitting in a bank earning interest.
2. **Buy a six-month future** instead. You pay nothing for the metal now, so your \$2,400 stays in the bank earning interest for six months, and you only take the gold at expiry.

For the market to offer no free lunch, these two routes must cost the same. The future must be *more expensive* than spot by exactly the amount route #1 would have earned you in interest, plus the storage you saved. That difference is the **cost-of-carry**:

```
fair future = spot x (1 + r x t) + storage cost
```

where `r` is the financing (interest) rate and `t` is the time to delivery in years. The intuition in one sentence: a future costs more than spot because the seller, who *will* hold the metal for you, has to be paid for the money he ties up and the vault he rents. That premium is the **basis** — the difference between the futures price and the spot price — and it is the single most important number in this entire post.

![Pipeline from spot through added interest and storage to a fair futures price and the basis](/imgs/blogs/gold-futures-comex-contango-backwardation-and-paper-vs-physical-2.png)

#### Worked example: the basis is carry, not a forecast

Take gold spot at \$2,400 an ounce. The relevant financing rate is 4.5% per year (roughly where short-term US rates sat in 2024), and we want the six-month future, so `t = 0.5`. Gold's storage cost is genuinely tiny — a vault charges a fraction of a percent a year — so we'll fold a small amount in and round.

- Interest component: \$2,400 x 4.5% x 0.5 = \$2,400 x 0.0225 = **\$54**.
- Storage: call it a couple of dollars over six months for a 100-oz lot — negligible per ounce, so we round the carry to about **\$54**.
- Fair six-month future ≈ \$2,400 + \$54 = **\$2,454**.

So a December future quoted at \$2,454 against a \$2,400 spot is telling you *nothing* about December's gold price. It is telling you that six months of financing on \$2,400 costs \$54. If you believed the \$54 premium meant "the market expects \$2,454 in December" you would be reading a thermometer as if it were a crystal ball. **The basis is the price of time and money, not a view on gold.**

This is the reason the cost-of-carry matters so much for gold specifically. Gold pays no dividend, no coupon, no rent — a fact we explored in [the no-yield problem](/blog/trading/gold/the-no-yield-problem-how-a-metal-that-pays-nothing-can-be-worth-anything). For a stock or a bond, the carry calculation has to subtract the income the asset throws off. For gold there is nothing to subtract; the carry is almost purely the interest rate plus a sliver of storage. That makes the gold futures curve one of the cleanest expressions of the cost-of-carry you can find anywhere in finance — it is almost a pure reading of short-term interest rates projected forward.

### Margin and leverage: why a little money moves a lot of metal

Because you do not pay for the metal up front, what *do* you put down? A **margin** — a good-faith deposit the exchange holds to guarantee you can absorb a day's adverse move. Margin is not a down payment on the gold; it is collateral against your promise. The exchange sets it (and raises it in volatile periods), and it is a small fraction of the contract's full value.

This is where leverage enters, and leverage is the whole reason futures can swing the spot price around.

#### Worked example: 20x leverage from one contract

One GC contract = 100 troy oz. At a \$2,400 spot, the **notional value** — the full dollar value of the gold you control — is:

- 100 oz x \$2,400 = **\$240,000**.

The exchange's initial margin for one gold contract has typically run around \$11,000–\$13,000 depending on volatility. Call it **\$12,000**. So:

- Leverage = notional / margin = \$240,000 / \$12,000 = **20x**.

With \$12,000 you are exposed to the price moves of \$240,000 of gold. If gold rises 1% — \$24 an ounce — your contract gains 100 x \$24 = **\$2,400**, which is a 20% return on your \$12,000 margin in a single day. The flip side is symmetrical and merciless: a 1% drop is a \$2,400 loss, 20% of your stake, and if losses eat into your margin the exchange issues a **margin call** demanding more cash by the next morning or it closes your position for you. **Leverage is why a relatively small pool of speculative money can shove the gold price around far more than its dollar size suggests — and why futures traders get carried out feet-first far more often than physical holders.**

The margin system has two tiers worth knowing. **Initial margin** is what you post to open the position — the \$12,000. **Maintenance margin** is a lower floor, typically around 90% of initial, that your account must stay above as losses accrue. Drop below maintenance and you get the margin call to top back up to *initial*, not just to maintenance — the exchange wants its full buffer restored. And critically, the exchange *raises* margins when volatility spikes, exactly when traders can least afford it. In a violent gold move the CME can hike initial margin overnight, forcing leveraged longs to either find fresh cash or sell — which can accelerate the very move that triggered the hike. This is a recurring amplifier in commodity blow-offs: a margin hike into a frenzy is gasoline, not water. The 1980 gold-and-silver spike, when the Hunt brothers were cornering silver, ended partly because the exchange jacked margins and changed the rules to force liquidation — a reminder that the people who set the margin are not neutral spectators.

A subtle but important consequence of daily mark-to-market: a future and a forward with the same terms are *not* quite economically identical. Because a future settles cash every day, your gains are paid to you (and can be reinvested at interest) and your losses are drawn from you (and must be funded) along the way, whereas a forward settles only once at the end. When interest rates are correlated with the gold price, this daily-settlement timing creates a tiny pricing wedge between futures and forwards. It is a second-order effect for gold and we'll set it aside, but it is the reason the careful literature distinguishes "the forward price" from "the futures price" even though for practical purposes they track each other tick for tick.

### Expiry, the two ways out, and the basis converging

Every contract has an expiry month. As expiry approaches, you face a fork: you can **close the position** (sell the contract you bought, ending your exposure with a cash gain or loss), or you can **hold to delivery** and actually exchange metal for money. We will see in a moment that the overwhelming majority of traders take the first door.

There is one more iron law to lodge before we go deeper. **At the moment of expiry, the futures price must equal the spot price.** The basis converges to zero. Why? Because on the delivery date a future *is* spot — buying the expiring contract and taking delivery is identical to buying metal on the spot market, so the two prices cannot differ. The basis you saw months out — that \$54 — melts away as the contract ages, like an ice cube shrinking toward the delivery date. That melting is not a gain or a loss that appears from nowhere; it is the carry being earned (or paid) day by day. Hold that thought, because it is the secret cost of the *roll* we'll meet later.

This convergence is enforced, again, by arbitrage rather than rule. If a contract expiring tomorrow were trading even a dollar above spot, a trader would sell the future, buy spot metal, and deliver into the expiring contract for a near-instant locked profit — and so many would do it that the gap would vanish before you finished reading this sentence. The closer to expiry, the tighter the leash: a contract three days from delivery cannot carry a meaningful basis, because the cost-of-carry over three days is essentially nothing and any deviation is free money. Far-dated contracts can carry a large basis (lots of carry to accrue); near-dated ones cannot. So the curve isn't just *a* slope — it is a slope that must hinge down to spot at the right edge, every single month, no matter what gold itself is doing. The whole structure of contango, the roll, and the convergence is one self-consistent consequence of "you cannot earn risk-free money," which is the deepest law in all of finance.

## How COMEX works: the contract, the clearinghouse, and delivery

COMEX is not where most gold *physically* changes hands — London's over-the-counter market, which we met in the pricing post, moves far more metal. COMEX is where the world *prices* gold for hedging and speculation, and where the futures machinery lives. Understanding its plumbing demystifies most of the conspiracy theories at a stroke.

![Flow from trader to margin to the CME clearinghouse, branching to cash settlement or delivery of registered metal](/imgs/blogs/gold-futures-comex-contango-backwardation-and-paper-vs-physical-5.png)

### The contract specs

The headline product is the **GC** contract: 100 troy ounces, deliverable as one 100-oz bar or three kilobars (a kilobar is ~32.15 oz). There are smaller cousins — the 50-oz **QO** and the 10-oz **MGC** micro contract — that let smaller accounts play without the full \$240,000 notional, but GC is the benchmark whose price you see quoted everywhere. Prices are in US dollars per troy ounce, the global convention we covered when we looked at [the troy ounce and the dollar peg](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce).

Contracts are listed for many months out, but liquidity concentrates in a handful of *active* months — for gold, the February, April, June, August, and December contracts carry the bulk of the volume. The nearest active month is the **front month**, and its price is what news outlets quote as "the gold price." The further-out months are the **back months**, and the relationship between front and back is the curve we keep returning to. Because liquidity thins out in distant months, a trader who wants long-dated exposure either accepts wide spreads or builds it from the liquid front and rolls forward — which is exactly the roll mechanic we dissect later.

The delivery process runs on a fixed calendar that you must respect even if you never intend to deliver. The key date is **First Notice Day** — the first day on which a short can serve notice of intent to deliver. From that day, an unwary long who is still holding the contract can be *assigned* delivery: a warrant for metal arrives, and a bill for the full notional follows. The other anchor is the **Last Trading Day**, after which the contract can no longer be traded and goes to delivery. The practical rule for any trader who wants price exposure and not metal is iron: **be out of the front month before First Notice Day.** Funds and brokers automate this; a retail trader who forgets can wake up owning, on paper, 100 ounces of gold they have to pay \$245,000 for. The calendar is unforgiving, and it is one more reason ~99% of contracts are closed or rolled well before they reach the delivery window.

### The clearinghouse: why you never worry about who's on the other side

When you buy a futures contract, who is the seller? You don't know and you don't care, and that is the genius of the system. The **CME Clearing** house steps into the middle of every trade through a process called *novation*: the instant your buy matches someone's sell, the clearinghouse becomes the seller to you and the buyer to them. Both of you now face the clearinghouse, not each other.

This is why a futures market does not collapse when one trader blows up. The clearinghouse guarantees every contract, and it protects itself with the margin system: every position is **marked to market daily**, meaning gains and losses are settled in cash each evening, so no one can run up a huge hidden loss. If your account can't meet a margin call, the clearinghouse liquidates your position and dips into a mutualized default fund only as a last resort. The lavender box in the diagram above is the entity that makes the whole thing trustworthy — it is the *counterparty to everyone*, which is exactly why no single counterparty can poison the well.

It is worth dwelling on how robust this design is, because it directly contradicts the "house of cards" framing. The clearinghouse sits behind a layered defence — the industry calls it a *default waterfall*. First, your own margin absorbs your losses. If that's exhausted, your clearing member (your broker, who guarantees you to the clearinghouse) is on the hook from its own capital. Beyond that sits a mutualized guaranty fund contributed by all clearing members, then the clearinghouse's own capital ("skin in the game"), then assessment powers to call for more. A trader, a broker, even several brokers can fail without the contracts you hold being impaired. The failure mode of a clearinghouse is not "it runs out of gold"; it is the near-unthinkable scenario where losses blow through every layer of the waterfall at once — which has not happened to a major futures clearinghouse. When someone tells you the COMEX is one delivery request away from collapse, they are describing a market structure that does not exist. The genuine risks are real but different: a liquidity squeeze in a logistics crisis, a basis blowout, a margin spiral — *price* dislocations, not a broken promise.

### Delivery vs cash settlement: registered, eligible, and the warrant

Here is the part that fuels the most confusion. A GC contract is *physically deliverable* — unlike, say, a stock-index future that always settles in cash. If you hold a long GC contract through expiry into the delivery period, you can demand 100 ounces of real gold, and a short who is assigned to you must hand it over.

But the metal must be of a specific kind, in a specific place. COMEX-approved vaults (mostly in and around New York) classify the gold they hold into two buckets:

- **Registered** metal: bars that have a *warrant* attached — an electronic document of title that makes them deliverable against a futures contract right now. This is the pool that can actually settle delivery demands.
- **Eligible** metal: bars that meet COMEX specs and sit in approved vaults but have *no* warrant — they belong to someone who is simply storing them there, not offering them for delivery. Eligible metal can be *converted* to registered (the owner requests a warrant) but until then it is not part of the deliverable pool.

This registered/eligible distinction is the crux of the paper-vs-physical argument, so file it carefully: **only registered metal backs the deliverable promise; eligible metal is a reserve that can be summoned but is not currently on offer.**

#### Worked example: the cost of standing for delivery

Say you hold one long GC contract into the December delivery period with the future at \$2,450, and you genuinely want the metal. Settlement works like this:

- You pay the full notional: 100 oz x \$2,450 = **\$245,000** in cash (not your \$12,000 margin — the *whole* amount; the metal was never free).
- A short with registered metal is assigned to deliver you a warrant for a 100-oz bar (or three kilobars) sitting in a COMEX vault.
- You now own that warrant. The gold is yours, in the vault. If you want it in your hands you pay to withdraw and ship it, plus assay and handling — typically a few dollars an ounce, call it \$1–\$3/oz, so \$100–\$300 on the lot.

The lesson buried in this arithmetic: **taking delivery means writing a cheque for the full \$245,000.** Futures let you *speculate* on the price with \$12,000, but they do not let you *acquire* metal cheaply — to actually own the gold you pay for all of it. This is precisely why ~99% of contracts never go to delivery: almost everyone trading futures wants price exposure, not 100 ounces of metal in a New York vault.

## Contango: when the futures curve slopes up

We have already done the hard work. **Contango** is simply the normal state of the gold futures curve, where each more-distant delivery month trades a little higher than the one before, and all of them trade above spot. It is the cost-of-carry made visible: every extra month to delivery adds another month of financing, so the curve climbs.

For gold, contango is the *default* because gold's carry is almost always positive — short-term interest rates are usually above zero and gold has no income to offset them. In a 4.5%-rate world, the gold curve rises by roughly 4.5% annualized: the further out you look, the higher the futures price, in a smooth upward slope.

A subtle point that matters for traders: the *steepness* of contango moves with interest rates, not with bullishness on gold. When the Fed hikes, the cost of carry rises and the gold curve steepens; when rates fall, it flattens. This ties the gold curve directly to the macro variable we keep returning to in this series — real rates — and which we dissected in [real interest rates, the master variable](/blog/trading/gold/real-interest-rates-the-master-variable-behind-the-gold-price). The futures curve is, in a quiet way, a chart of expected short rates wearing a gold costume.

There is one more ingredient in the carry that the simple formula hides, and it is the bridge to backwardation: the **gold lease rate**. Big holders of gold — central banks, bullion banks — will *lend* their metal to others (refiners, jewelers, miners hedging) for a fee, the lease rate. If you can lease gold out and earn that fee, then holding metal is not pure dead weight; it throws off a little income, which *reduces* the net cost of carry. The market even has a name for the combined number: **GOFO**, the Gold Forward Offered Rate, was historically the rate at which dealers would lend dollars against gold collateral — essentially the dollar interest rate *minus* the gold lease rate. When GOFO is positive (the normal state), carry is positive and the curve is in contango. When GOFO goes *negative* — meaning the gold lease rate exceeds dollar interest rates, because everyone suddenly wants to *borrow metal* and few will lend it — the net carry turns negative and the curve can flip into backwardation. So the lease rate is the hinge: contango is "money is more wanted than metal," backwardation is "metal is more wanted than money."

#### Worked example: reading a contango curve correctly

Suppose you see this December-vs-spot picture:

- Spot: \$2,400.
- 3-month future: \$2,427 (basis +\$27).
- 6-month future: \$2,454 (basis +\$54).
- 12-month future: \$2,508 (basis +\$108).

The basis grows almost perfectly linearly with time — \$27, \$54, \$108 — because it is just \$2,400 x 4.5% x t for t = 0.25, 0.5, 1.0. A newcomer sees the rising curve and thinks "the market is bullish, it expects \$2,508 in a year." **Wrong.** The curve is flat in *real, carry-adjusted* terms — it is dead-level expectation plus financing. If gold spot doesn't move at all over the year, the 12-month future will simply decay from \$2,508 down to \$2,400 as it converges, and a long holder will have *lost* \$108 to carry while the metal did nothing. Contango is a *headwind* for anyone holding futures, which is the whole point of the next section.

## Backwardation: when the curve flips, and what it screams

Now the interesting case. **Backwardation** is when the futures curve slopes *down* — deferred contracts trade *below* spot. For gold this is rare and, when it happens, loud.

Think about what backwardation means in cost-of-carry terms. The fair future is spot plus carry, and carry is almost always positive, so the future *should* be above spot. For the future to trade *below* spot, the cost-of-carry has to effectively go negative — and the only thing that pushes it negative is a **convenience yield**: a premium people will pay to have the metal *in their hands now* rather than promised later.

When does anyone pay extra for immediacy? When physical metal is genuinely scarce relative to demand: a refinery needs bars *this week* to fill orders, a central bank wants delivery now, a short squeeze forces holders of paper to source real metal. In those moments, having the gold today is worth more than the interest you'd earn waiting for it, so spot trades *above* the future. **Backwardation is the futures market's way of screaming that someone needs physical metal right now and the paper promise of future metal is, briefly, less valuable than the real thing.**

The contrast with industrial commodities is illuminating and worth a beat. Crude oil and natural gas swing between contango and backwardation *all the time*, because they are expensive and awkward to store — you need tanks, pipelines, salt caverns — so the convenience yield of having barrels right now is large and volatile. When oil is in backwardation, it usually means physical supply is tight today; when it's in deep contango (as in the famous April 2020 episode when storage filled up and the front-month price briefly went *negative*), it means nobody wants the physical barrel they're contractually owed. Gold is the opposite: it is the easiest thing in the world to store — a vault, a fee, done — so its convenience yield is normally near zero and its curve sits in placid contango almost permanently. *That* is precisely why gold backwardation is such a loud signal. For a metal that is trivially cheap to hold, the only way the future can fall below spot is if the demand for *immediate physical possession* has overwhelmed the gentle pull of the financing rate. In oil, backwardation is Tuesday. In gold, it is an alarm bell.

![Two curves showing contango sloping up above spot and backwardation sloping down below spot](/imgs/blogs/gold-futures-comex-contango-backwardation-and-paper-vs-physical-3.png)

#### Worked example: what a –\$5 basis is telling you

Say spot gold is \$2,400 and the *near* future — say one month out — trades at \$2,395, a basis of **–\$5**. Run it against the cost-of-carry. Normal one-month carry on \$2,400 at 4.5% is \$2,400 x 0.045 x (1/12) ≈ **+\$9**. So the *fair* one-month future should be around \$2,409. Instead it's \$2,395 — fully \$14 below fair value.

That \$14 gap is the convenience yield: the market is paying \$14 an ounce, annualized to a huge implied rate, simply to *not wait a month* for metal. In plain numbers, holders of physical gold could sell their metal at \$2,400 today and buy it back via the \$2,395 future for delivery in a month, pocketing \$5 *plus* the \$9 of interest they'd earn on the cash in between — a near-riskless ~\$14 — yet they *don't*. The only reason a rational holder refuses that trade is that they value having the metal in hand more than \$14. **Backwardation is a tightness gauge: the deeper and more persistent it is, the more real the physical shortage.** It is one of the few genuinely forward-looking signals in the gold market that isn't just a financing rate in disguise.

## The roll: why holding futures is never free

Here is the practical consequence that catches retail traders and even some funds off guard. A futures contract *expires*. If you want to maintain gold exposure beyond the front month, you must **roll**: close your expiring contract and open the next one out. And in a contango market, the next contract is *more expensive* than the one you're leaving.

Rolling in contango means you repeatedly sell low (the expiring near contract, which has decayed toward spot) and buy high (the deferred contract, sitting above spot by a fresh slice of carry). Each roll quietly bleeds the carry out of your position. This is the same melting-ice-cube we flagged earlier, now charged to you on a schedule.

#### Worked example: a year of roll cost in contango

Suppose you want continuous one-year exposure to gold via futures, gold doesn't move at all (spot stays \$2,400 the whole year), and the curve is in a steady 4.5% contango. You roll, say, every two months into the next contract.

- Each contract you hold decays toward spot as it ages — that decay is the carry, ~4.5% per year on \$2,400 ≈ **\$108 per ounce per year**.
- Over the full year, spread across your rolls, you give up roughly that entire \$108 an ounce to carry even though the gold price never budged.
- On one 100-oz contract that's about **\$10,800** of roll cost over the year — paid out of your account, invisible on any "gold went nowhere" headline.

Now compare to physical: a holder of an actual 100-oz bar over that same flat year paid only the tiny vault fee, maybe \$50–\$150. **Contango is the structural reason a futures-based gold position underperforms physical metal over time when the price goes sideways** — and it's why long-term gold investors are usually steered toward physical or [physically-backed ETFs like GLD](/blog/trading/gold/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal) rather than rolling futures. The carry you "earn" as the seller is the carry you "pay" as the buyer; futures are a financing instrument as much as a price instrument.

The corollary is the bright side, and it's the only time roll *helps* you: in **backwardation**, the deferred contract is *cheaper* than the one you're leaving, so each roll is a small gain — you sell high and buy low. A persistently backwardated market actually pays longs to roll. That's vanishingly rare in gold but common in oil and other commodities with real storage constraints, and it's why "roll yield" is a whole sub-discipline in commodity investing.

The roll is also a *trade* in its own right, not just a chore. The mechanism for moving from one contract to the next is a **calendar spread** — simultaneously selling the expiring month and buying the next, executed as a single package so you're never momentarily un-hedged. Traders who have no directional view on gold at all will trade calendar spreads to bet purely on the *shape* of the curve: if you think contango will steepen (rates rising, financing getting dearer), you can position to profit from the front-to-back gap widening, with almost no exposure to the gold price itself. This is how the curve gets priced efficiently — there is a whole population of spread traders whose job is to keep the basis honest relative to interest rates, and they are the same force that drags the basis back to fair value whenever it wanders. When you hear that "the basis is held in line by arbitrage," these calendar-spread and cash-and-carry traders are the arbitrageurs doing the holding.

There's a related arbitrage that nails the basis to the cost-of-carry directly: **cash-and-carry**. If the future ever trades meaningfully *above* fair value (spot + carry), a trader can buy the cheap physical metal, sell the expensive future, store the metal, and deliver it at expiry — locking in a riskless profit equal to the overpricing. The reverse — *reverse cash-and-carry* — works when the future is too cheap: borrow and sell metal, buy the cheap future, take delivery later. The mere *possibility* of these trades is what keeps the basis pinned near spot + carry in normal times. Backwardation persists only when the reverse trade is blocked — when you *can't* easily borrow metal to sell, because the lease rate has spiked and holders won't lend. That blockage is the physical tightness made mechanical.

## Paper vs physical: open interest, registered metal, and the scary ratio

We now have everything we need to defuse — or rather, to *correctly calibrate* — the most famous worry in the gold world. The claim, repeated endlessly: *"There are dozens, even hundreds, of paper ounces for every real ounce. The COMEX is a house of cards. When everyone demands delivery, it defaults and the price goes to the moon."*

Let's put real numbers on it.

**Open interest** is the total number of futures contracts outstanding — every long is matched by a short, and the count of those open positions is the volume of *claims* on gold the futures market has created. In gold, COMEX open interest typically runs around 400,000–600,000 contracts. At 100 oz each, that's roughly **40–60 million ounces** of gold claimed by open futures — call it **~1,500 tonnes**.

**Registered metal** — the deliverable pool — typically sits around 15–25 million ounces, or roughly **500 tonnes**, though it swings a lot with delivery demand and dealer positioning.

![A tall open-interest bar dwarfs a small registered-metal bar with an eligible reserve between and ratio and calm annotation boxes](/imgs/blogs/gold-futures-comex-contango-backwardation-and-paper-vs-physical-6.png)

#### Worked example: the ratio, and why it usually doesn't bite

Take 500,000 contracts of open interest and 16 million ounces of registered metal.

- Paper claims: 500,000 x 100 = **50 million oz**.
- Registered metal: **16 million oz**.
- Ratio: 50 / 16 ≈ **3:1** — about three paper ounces per deliverable ounce. In tighter periods the headline "ounces of paper per registered ounce" can be quoted far higher because some analysts use a narrower metal definition; you'll see scary numbers like 100:1 thrown around, which usually compare *all open interest including far-dated months* against *only the front-month registered float*. Apples to oranges, but it makes a punchy chart.

Now the part the alarmists skip. Of those 500,000 contracts, **~99% will be closed or rolled before delivery.** To see why, look at *who actually holds futures*. The US regulator's weekly position report sorts traders into rough buckets, and the picture is always the same:

- **Commercials / hedgers** — miners locking in a future sale price for gold they will dig up, refiners and jewelers locking in a cost for metal they will buy, bullion banks running their books. These players use futures to *transfer price risk*, not to acquire metal; a miner who is short futures fully intends to deliver his *own mined* gold or, far more often, to buy back the short and sell his metal in the spot market. They are structurally short the front end as a hedge.
- **Managed money / speculators** — macro funds, CTAs, trend-followers expressing a view on the gold price with leverage. They want the price move and will be out long before any delivery window; standing for delivery would mean writing a \$245,000 cheque per contract, the last thing a leveraged macro fund wants.
- **Spread traders** — the calendar-spread arbitrageurs from the roll section, who hold offsetting longs and shorts and have essentially zero net delivery intent.

Add it up and the population that *wants metal* is a sliver. In a typical delivery month only a few thousand contracts — low single-digit percentages of open interest — actually go to delivery, and the registered pool plus the much larger *eligible* pool (which can be warranted into registered when delivery demand rises) handles them comfortably. The system is fractional *by design*, exactly like a bank: a bank holds far less cash than the total of its deposits because not everyone withdraws at once, and that is a feature, not a fraud — until a bank run.

So the honest framing is neither "it's all fake paper" nor "nothing to see here." It is: **the paper-to-physical ratio is a measure of how a delivery scramble would behave, not evidence of fraud.** In normal times the ratio is irrelevant because delivery demand is a trickle. The ratio only *bites* in the rare moment when a large fraction of longs simultaneously demand metal — and even then the exchange's first response is to let prices and margins rip until enough eligible metal gets warranted or enough longs decide the cash is good enough. The scenario where COMEX literally *runs out* and *defaults* has not happened; what happens instead is exactly what we saw in March 2020 — the *price of delivery* (the basis, the EFP spread) blows out until the market clears.

The cleanest real-world test of the "force a delivery default" thesis came not in gold but in silver, in early 2021, when a viral online campaign — *#silversqueeze* — explicitly tried to break the COMEX by getting retail buyers to demand so much physical silver that the registered pool would empty and the price would rocket. What actually happened is the textbook outcome of everything above: silver spiked about 10% for a day or two, premiums on physical coins and small bars blew out at retail dealers (the place a real shortage *does* show up), eligible metal got warranted into registered to meet the bump in demand, and within a week the price had given most of it back. The exchange did not default. The squeeze proved the system's resilience rather than its fragility: when delivery demand rose, the *price* of immediate metal rose to ration it and to pull more metal into the deliverable pool — exactly the pressure-release valve the design intends. If a coordinated campaign aimed squarely at breaking the mechanism couldn't break it, the ambient 3:1 ratio in a quiet month is not the time bomb it's sold as.

## Common misconceptions

**"Futures prove the gold price is manipulated by paper traders with no metal."** The futures price and the physical price are tied together by *arbitrage*, not by faith. If futures dropped far below physical, anyone could buy the cheap future, stand for delivery, and get metal below the spot price — a free profit that traders pounce on until the gap closes. The basis can dislocate briefly in a logistics crisis (March 2020), but it is held in line by the threat of delivery, which is why the deliverability of the contract matters even though almost no one uses it. Paper can dominate *short-term* price *discovery*; it cannot float free of physical forever because the delivery door is always there.

**"Contango is bullish — the curve is pointing up!"** No. As we computed, an upward-sloping curve is just carry. A future trading above spot tells you about interest rates, not about gold's direction. You can have a screaming bull market in *flat* contango and a brutal bear market in *steep* contango. The slope is a financing artifact; the *level* of spot is the bet.

**"Backwardation is bearish because future prices are lower."** Backwards again. Backwardation almost always coincides with physical *strength* — it means demand for metal-now is so intense it overwhelms the carry. It is a tightness signal, frequently seen near important *bottoms* in the physical market or during squeezes, not a forecast of lower prices.

**"COMEX is about to default and gold will gap to \$50,000."** The exchange has weathered every delivery surge by adjusting margins and prices and by drawing eligible metal into the registered pool. A genuine failure-to-deliver would be historic, but the realistic stress outcome is a *backwardation/EFP blowout* — the price of immediate metal spikes relative to paper — not a zero. If you're worried about delivery risk, the lesson isn't "futures will break"; it's "if you want metal, own metal," which is the argument for [allocated physical bars](/blog/trading/gold/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk).

**"Buying a future is the cheap way to own gold."** It's the cheap way to get *price exposure* on margin, but if you ever take delivery you pay the *full* notional, and if you just keep rolling in contango you bleed carry every cycle. Futures are a leverage-and-hedging tool, not a low-cost path to a gold stash.

## How it shows up in real markets

### March 2020: the EFP dislocation

The episode in the opening is the textbook case. The mechanism that normally keeps London (physical, 400-oz bars) and New York (COMEX futures, deliverable as 100-oz/kilobars) glued together is the **EFP** — Exchange for Physical — a swap dealers use to move between a London position and a COMEX position. When flights stopped and refineries (several in Switzerland shut for COVID at the same moment) couldn't recast bars, dealers who were short COMEX futures suddenly couldn't be sure they could source deliverable metal. The EFP spread, normally a couple of dollars, blew out to *tens* of dollars an ounce. COMEX responded by introducing a new 400-oz-deliverable contract to bridge to London bars. The price of *paper* gold and the price of *deliverable* gold visibly diverged — precisely the basis stress this whole post is about — and the gold/silver ratio spiked to a record as the entire complex seized.

![Bar chart of the gold-silver ratio by year with the 2020 bar highlighted at the record spike](/imgs/blogs/gold-futures-comex-contango-backwardation-and-paper-vs-physical-7.png)

The chart above uses the gold/silver ratio as the stress proxy — its March-2020 intraday reading near 125 (against a 20th-century norm around 47) marks the week the metal markets jammed. Notably, COMEX did *not* default; the system absorbed the shock by repricing the basis and creating new deliverable supply. That is the realistic shape of a "scramble": a violent move in the *spread*, not a collapse of the exchange.

#### Worked example: what the EFP blowout cost a short

Picture a bullion bank in March 2020 that is short 100 COMEX June contracts as a hedge against gold it holds in *London* as 400-oz bars. Normally that's a perfectly matched book: short paper in New York, long metal in London, and the EFP spread between them is a couple of dollars. Now the planes are grounded and the Swiss refiners are shut, so the bank cannot turn its London 400-oz bars into COMEX-deliverable 100-oz bars in time.

- Normal EFP spread: ~\$2 an ounce.
- Crisis EFP spread: it gapped to roughly **\$50–\$70 an ounce** at the worst.
- To stay hedged, the bank must either source deliverable metal or roll/close its short at the dislocated price. On 100 contracts (10,000 oz) a \$60 spread is 10,000 x \$60 = **\$600,000** of extra cost versus the normal couple-of-dollars — for a position that was supposed to be flat.

That \$600,000 is the basis stress made concrete: it is the price the market charged for the *logistics* of converting one form of gold into a deliverable form during a week the supply chain seized. No one defaulted; the shorts simply paid a brutal premium, COMEX introduced a 400-oz-deliverable contract to bridge the gap, and the spread normalized within weeks. The lesson is exactly the post's thesis — **a delivery scramble shows up as an explosion in the basis, the price of immediacy, not as a broken exchange.**

### The 2024–2026 surge and the leverage that rode it

The price context for all this is the move charted below: gold spent a decade in the \$1,200–\$1,900 range, broke out through 2024 to average around \$2,388, and then spiked toward record highs into early 2026. Throughout, COMEX futures were the venue where leveraged macro money expressed the trade. When real rates fell and central banks bought — the structural story from [the post-2022 central-bank buying wave](/blog/trading/gold/central-banks-the-structural-buyer-that-changed-gold-after-2022) — speculative longs piled into futures, and the 20x leverage amplified every dollar of conviction into outsized price impact. Futures didn't *cause* the bull market; the macro did. But futures are the megaphone through which macro conviction gets shouted at the price.

![Gold price line chart from 2010 to 2026 with the 2024 average and 2026 spike annotated](/imgs/blogs/gold-futures-comex-contango-backwardation-and-paper-vs-physical-4.png)

### April 2013: when paper led physical down

The mirror image of a delivery scramble is a *liquidation cascade*, and the clearest case is the April 2013 crash, when gold fell roughly 9% in a single session and about 14% over two days — one of the sharpest drops in its modern history. The trigger was not a sudden disappearance of physical demand; in fact, physical buyers in Asia stampeded *in* to buy the dip, and dealers ran short of coins and small bars within days. The crash happened in the *paper* market: a wave of selling — large sell orders hitting the COMEX in thin early-morning hours, stop-losses cascading, leveraged longs forced out as the price broke key levels and margin calls compounded — drove futures down, and physical followed because spot and futures are arbitraged together.

This episode is the perfect counterweight to the gold-bug framing. Yes, paper futures *led* the price, and *down*, with little physical justification in the moment — exactly the "paper dominates short-term discovery" point. But notice what it was *not*: it was not a default, not a fraud, not a permanent break from physical reality. Within months the same arbitrage that transmitted the paper selloff into spot also transmitted the surge of physical buying back the other way. Paper can shove the price violently in the short run *precisely because* of the 20x leverage — a relatively small amount of forced selling moves a huge notional — but it cannot hold the price away from physical reality indefinitely, because the delivery door and the cash-and-carry arbitrage are always standing there. The 2013 crash and the 2020 squeeze are the two faces of the same coin: leverage makes futures the loud, fast, sometimes-overshooting venue of price discovery, while delivery keeps it tethered, in the end, to metal.

### Routine backwardation as a microstructure signal

Outside crises, brief backwardation flickers show up around heavy delivery months and squeezes, and seasoned physical desks watch the front-month basis the way a sailor watches the barometer. A persistent slide of the near future below spot — even a couple of dollars beyond what carry justifies — tells the desk that real metal is getting tight and lease rates are firming. It rarely makes headlines, but it is one of the cleanest real-time reads on physical demand the market produces, far more honest than the noise of daily price ticks.

### How to read the curve in practice

For a reader who will never trade a contract, the futures curve is still worth glancing at, because it carries information the spot price alone does not. Three quick reads:

1. **The slope tells you about rates, not gold.** A steeper contango is the financing rate rising; a flattening curve is rates easing. If you want to know what the bond market thinks about short rates, the gold curve is a clean, gold-flavoured second opinion. Don't read an upward slope as bullishness — it's carry.
2. **A flicker into backwardation is a physical-tightness alarm.** When the near future slips below spot beyond what the tiny carry justifies, real metal is getting scarce — watch for it around heavy delivery months, squeezes, and crises. It's one of the few honest, hard-to-fake demand signals the market produces.
3. **If you hold gold through futures, account for the roll.** In persistent contango a rolled futures position bleeds carry every cycle; over a flat year that's the difference between a small vault fee and thousands of dollars an ounce of slippage. For buy-and-hold, that argues for physical or a physically-backed vehicle, not a rolled futures stack.

And for anyone tempted by the leverage: respect it. Twenty-to-one cuts both ways daily, margin gets *raised* exactly when volatility spikes, and the calendar will assign you metal you must pay full price for if you fall asleep past First Notice Day. Futures are a scalpel, not a savings account.

## The takeaway: futures as a tool and a signal, not a verdict

Strip away the mythology and gold futures are two things at once. As a **tool**, they are leverage: a way for hedgers to lock prices and for speculators to command \$240,000 of metal with \$12,000, marked-to-market daily, guaranteed by a clearinghouse, and almost always closed in cash before any metal moves. As a **signal**, the shape of the curve is a live readout — contango is the cost-of-carry telling you about interest rates, and the rare flip into backwardation is the market screaming that physical metal is genuinely scarce *right now*.

The paper-vs-physical debate, properly understood, dissolves into a single sober sentence: the futures market is fractional by design, like a bank, and the scary claims-to-metal ratio is a measure of how a delivery scramble would *behave*, not proof that one is coming. It bites only in the rare logistics shock, and even then it expresses itself as a blown-out basis, not a default.

Which brings us back to the spine of this whole series. Gold is monetary insurance — a vote of no-confidence in paper claims, real yields, and political stability. There is a quiet irony in that the most heavily *paper-ized* corner of the gold world, the futures market, is where you can watch that insurance get priced and traded with the most leverage and the least metal. If you want the *exposure*, futures are a precise, liquid, leveraged instrument. But if your reason for owning gold is the insurance itself — the thing that pays off precisely when paper claims are failing — then a paper claim on gold is a strange way to hold it. The basis, the roll, and the registered/eligible split all whisper the same lesson the gold bugs get right even when their arithmetic is wrong: **in the moment that gold is supposed to protect you, the only ounce that counts is the one you can actually hold.** Futures will tell you what that ounce costs. They will not, when it matters most, be that ounce.

## Further reading & cross-links

- [How gold is priced: spot, the London fix, futures, and the troy ounce](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce) — the price-mechanics primer this post drills into, including the first look at the basis.
- [Physical gold: bars, coins, allocated vs unallocated, and counterparty risk](/blog/trading/gold/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk) — the other end of the paper-physical spectrum: owning metal you can actually touch.
- [Gold ETFs and the GLD machine: how paper gold tracks the metal](/blog/trading/gold/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal) — the middle ground: a security that holds allocated metal so you don't have to roll futures.
- [Real interest rates: the master variable behind the gold price](/blog/trading/gold/real-interest-rates-the-master-variable-behind-the-gold-price) — why the cost-of-carry that shapes the futures curve is really a real-rates story.
- [Derivatives pricing: from replication to risk-neutral measures](/blog/trading/quantitative-finance/derivatives-pricing) — the formal, quant treatment of why a forward price is spot times the cost-of-carry, with the no-arbitrage proof in full.
