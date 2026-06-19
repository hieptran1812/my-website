---
title: "The Carry Trade: Getting Paid to Hold a Currency"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Borrow a cheap currency, hold an expensive one, and bank the interest-rate gap — the full dollar P&L, the leverage, and why the whole trade is a bet that interest-rate parity fails."
tags: ["forex", "currencies", "carry-trade", "interest-rate-differential", "uncovered-interest-parity", "leverage", "yen-carry", "risk-management"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The carry trade is the simplest money-making idea in currencies: borrow a currency with a low interest rate, use the proceeds to hold one with a high interest rate, and pocket the difference every single day you stay in.
>
> - The income is the **rate gap**, not a price move. Borrow yen at 0.25% and hold Australian dollars at 4.35% and you earn roughly **+4.10% a year** just for holding the position — no forecast required.
> - But you also own the **exchange rate**. Your total return is *carry income plus the spot move*, and the spot move can wipe out years of income in days.
> - **Leverage** scales both halves. A \$1,000,000 cash position earning +4.10% becomes +41% on equity at ten-times leverage — and a 5% adverse move stops being a bruise and becomes a margin call.
> - Most deeply: carry only pays because **uncovered interest parity fails**. Theory says the high-yielder must weaken by exactly the rate gap, leaving you nothing. It usually doesn't — until, violently, it overshoots and does.
> - The one number to remember: the stylised G10 carry index lost **−12%** in the single week of the August 2024 yen unwind, after grinding up for years.

## A trade that paid for years, then lost it in a week

On the third of July 2024, a hedge-fund desk in London was short the yen and long higher-yielding currencies. The position had been printing money for two years. Japan's central bank held its policy rate near zero while the US Federal Reserve had taken its rate above 5%, so the trade earned a fat interest gap every night, and on top of that the yen kept sliding — USD/JPY had walked from 115 at the end of 2021 to **161.9** that July morning. Carry income *and* a tailwind on the exchange rate. It felt like free money.

Five weeks later it was a catastrophe. The Bank of Japan hiked, the Fed signalled cuts, and the most crowded trade in the world reversed all at once. USD/JPY collapsed from **161.9 to 141.7** by the fifth of August — a 12% move in the funding currency in barely a month, most of it in a handful of sessions. The VIX, Wall Street's fear gauge, spiked to **65.7** intraday, a level seen only in 2008 and 2020. Funds that had been "getting paid to hold a currency" were suddenly getting margin calls to *not* hold it, and the rush for the exits made the move worse. Years of patient carry income evaporated in a week.

That single episode is the whole story of the carry trade in miniature: a steady, boring, daily drip of income that you collect for being patient — sitting on top of a tail risk that occasionally, suddenly, takes it all back. The figure below shows it across two decades. A stylised G10 carry index grinds relentlessly higher, then plunges in 2008, 2015, and 2024. Carry is not a free lunch. It is a paid lunch with a small, ugly chance the restaurant collapses on you.

![A G10 carry index rising steadily then crashing in 2008 2015 and 2024](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-1.png)

This post builds the carry trade from absolute zero. We will assemble it mechanically — borrow yen, buy Aussie, bank the gap — then write out the dollar profit-and-loss, separate the part you earn from the part that can hurt you, turn up the leverage, and finally connect it to the deepest idea in currency trading: that carry is, at root, a bet that *uncovered interest parity fails*. By the end you will read a yield table the way a carry trader does — not as a list of rates, but as a menu of what you pay to borrow and what you earn to hold.

## Foundations: what the carry trade actually is

Start with the spine of this whole series: **you never own a currency in isolation.** An exchange rate is the relative price of two monies, so every currency position is a pair — a bet on one currency *against* another. The carry trade takes that fact and makes it pay rent.

Here is the entire idea in one sentence. **Borrow money in a currency where interest is cheap, convert it into a currency where interest is rich, and the difference in interest rates is your income for as long as you hold the position.**

That is it. There is no forecast, no chart pattern, no clever timing. You are simply standing in the middle of two interest rates and collecting the spread between them, the same way a bank earns the gap between what it pays depositors and what it charges borrowers. To make every term concrete, we need to name the parts.

### The funding leg: what you borrow

The **funding currency** is the one you borrow. You want it cheap — a low interest rate — because the rate you borrow at is a cost you pay every day. The classic funding currencies are the yen (JPY) and the Swiss franc (CHF), because Japan and Switzerland have run near-zero interest rates for the better part of two decades. Borrowing yen at 0.25% a year is almost free money in, and that is exactly why the yen has been the world's funding currency of choice.

When you borrow yen, you owe yen. You will have to pay that loan back in yen one day, plus the small interest. That obligation — a short position in the funding currency — is the part of the trade that can bite you, and we will come back to it.

### The asset leg: what you hold

The **asset currency** (or **high-yielder**, or **target currency**) is the one you hold. You want it rich — a high interest rate — because that is the income you earn. You take the yen you borrowed, sell them, buy the high-yielder, and park the proceeds in a deposit or a short-term bond that pays the high local rate. The Australian dollar (AUD) at 4.35%, the Mexican peso at over 10%, the Brazilian real above 12% — these are asset currencies. You are long the high-yielder and you collect its yield.

### The rate gap is the income

The **rate gap** — the **interest-rate differential** — is the engine. It is simply the asset rate minus the funding rate. Borrow at 0.25%, hold at 4.35%, and the gap is 4.10%. That 4.10% is your carry: the income you earn per year just for holding the position, before the exchange rate moves a single pip. This series has a dedicated post on why the rate gap is the master variable that moves currencies in the first place — see [interest-rate differentials, the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx). The carry trade is what you do once you believe that.

A **pip** is the smallest standard increment a currency pair moves — for most pairs the fourth decimal place, so a move from 0.6500 to 0.6501 in AUD/USD is one pip. We mention it because the *spot move* — the change in the exchange rate — is measured in pips, and it is the other half of your return.

The figure below lays out the two legs as a flow. You borrow the cheap currency (money out, in red — that is the loan you must service), convert it into the expensive one, hold the expensive one (money in, in green — that is the yield you collect), and the rate gap drops into your pocket as carry income.

![Two leg structure borrow yen sell yen buy Aussie hold AUD deposit pocket the gap](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-2.png)

### Leverage: the multiplier

The last building block is **leverage** — borrowing more than your own money to put on a bigger position. Because the rate gap on a cash position is only a few percent a year, carry traders almost always lever it up. If you have \$1,000,000 of your own capital and you can borrow another \$9,000,000, you control a \$10,000,000 position. The carry income now lands on \$10,000,000 of exposure but you only put up \$1,000,000, so the *yield on your equity* is roughly ten times larger. Leverage is what turns a sleepy 4% into an exciting 40%. It is also what turns a survivable loss into a wipeout, and we will see exactly how in the leverage section.

So the carry trade has four parts: a cheap funding leg, a rich asset leg, the rate gap that pays you, and leverage that scales it. Now let us put real dollars through it.

### How the income actually reaches your account

It is worth being precise about *how* the rate gap turns into cash in your account, because it is not a cheque that arrives once a year. When you hold a foreign-currency position past the daily settlement cut-off, your broker or prime broker performs a **rollover**: it effectively rolls your borrowing in the funding currency and your deposit in the asset currency forward one more day, and credits or debits the net interest for that day. If you are long the higher-yielding currency and short the lower-yielding one, the net rollover is a *credit* — you are paid. If you flip the trade (long the low-yielder, short the high-yielder), you *pay* the rollover instead, and the carry works against you. This is why direction matters: the carry trade is specifically the *long-high, short-low* orientation, and the rollover credit is the income.

The same economics show up in the **forward market**, which is where institutions actually put carry on at scale. A currency forward is an agreement to exchange two currencies at a fixed rate on a future date. The forward rate is not a forecast — it is mechanically pinned to the spot rate adjusted by the two interest rates, an identity called **covered interest parity**. Because the high-yielder carries the higher interest rate, it trades at a *forward discount*: the forward price of the Australian dollar in yen is *below* its spot price by roughly the rate gap. A carry trader who sells the Aussie forward at that discount and lets it converge to spot earns exactly the rate gap if the spot is unchanged — the carry income, repackaged as the roll-down of the forward. The mechanics of spot, forward, and swap are covered in the sibling post on [spot, forward, and swap, the three ways to trade a currency](/blog/trading/forex/spot-forward-and-swap-the-three-ways-to-trade-a-currency); the takeaway here is that whether you express carry through deposits or through forwards, **the income is the same rate gap, and it is the market's no-arbitrage pricing that guarantees it.**

This matters for a subtle reason. Because covered interest parity *holds* (it is enforced by arbitrage), the forward already "knows" the rate gap. So when you earn carry, you are not beating the forward — you are betting that the *actual* future spot rate will land *better* than the forward predicted. The forward says the Aussie will weaken by the gap; you are betting it won't. That is the same bet against uncovered interest parity we will build to, just seen from the forward's vantage point.

## Building the trade: the dollar P&L from zero

Let us construct one concrete carry trade and follow every dollar. We will use the canonical pair of 2024: borrow yen, hold Australian dollars. The Reserve Bank of Australia held its policy rate at **4.35%** while the Bank of Japan held its at **0.25%** — a clean, real-world rate gap of **4.10%**.

#### Worked example: the annual carry income on a \$1,000,000 position

You start with \$1,000,000 of your own cash. You want a yen-funded Aussie carry, unlevered, so you borrow no extra money — you simply convert your dollars into the structure. The mechanics:

- You borrow yen worth \$1,000,000 at the funding rate of **0.25%** per year. Annual funding cost: \$1,000,000 × 0.25% = **\$2,500**.
- You sell those yen and buy Australian dollars, then park them in an Australian deposit yielding **4.35%** per year. Annual interest earned: \$1,000,000 × 4.35% = **\$43,500**.
- Your net carry income for the year is \$43,500 − \$2,500 = **\$41,000**.

That \$41,000 is exactly the rate gap — 4.10% — applied to your \$1,000,000. It arrives whether or not the exchange rate moves, accruing little by little every day the position is open (in practice through the daily rollover or swap points the broker credits you). **The carry trade's income is the rate gap times the notional, and it is yours just for being patient.**

Notice what we have *not* done yet: we have not said one word about whether the Australian dollar rises or falls against the yen. That is the second half of the return, and it is the half that is not guaranteed.

### The other half: you also own the exchange rate

When you borrowed yen and bought Aussie, you took on a position in the AUD/JPY exchange rate. You are long AUD and short JPY. If the Australian dollar rises against the yen, you gain on top of the carry. If it falls, you lose — and the loss can be far bigger than the income.

This is the crucial mental shift. A carry trade is **not** a bond. A bond pays you a coupon and returns your principal in the same currency, so your only real worry is whether the issuer defaults. A carry trade pays you a coupon (the rate gap) but your "principal" is denominated in a *foreign* currency whose price changes every second. You can be completely right about the interest rates and still lose money if the currency moves against you.

The chart below shows the rate menu of 2024 — what you pay to borrow on the left, what you earn to hold further down. The yen and Swiss franc sit at the bottom as funding currencies; the Australian dollar, Mexican peso, Brazilian real, and Turkish lira climb up the yield ladder. The taller the bar, the more carry income — and, almost always, the more risk packed into the spot move.

![The rate menu funding currencies versus high yielders policy rates 2024](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-3.png)

That menu is the carry trader's whole world. But reading it correctly means understanding that the high bars are not free yield — they are *compensation for risk*. The Turkish lira pays 50% because the lira has lost the overwhelming majority of its value against the dollar over the past decade. The market is not handing out 50% for nothing. Which brings us to the part of the P&L that hurts.

### Why the gap, not the level, is what you earn

A beginner's instinct is to chase the currency with the highest interest rate, full stop. But carry is a *spread* trade — what you earn is the gap between two rates, not the level of either one. Borrowing dollars at 4.50% to hold Australian dollars at 4.35% is a *negative* carry: you pay 0.15% a year for the privilege, even though the Aussie's 4.35% rate looks perfectly healthy in isolation. The Aussie only becomes a high-yielder *relative to a funding currency that pays less*. This is the spine of the series made operational: there is no such thing as "a high-yielding currency" in the abstract — there is only a high-yielding currency *against a specific funding currency*. The yen funds the Aussie carry not because the Aussie's rate is special but because the yen's rate is so much lower. Change the funding leg and the trade changes entirely.

This also explains why funding currencies cluster at the bottom of every rate menu and stay there. A central bank that anchors its policy rate near zero for years — Japan since the 1990s, Switzerland through the 2010s — manufactures a permanent funding currency. The market borrows it precisely because its rate is the lowest in the developed world, and the *persistence* of that low rate is what makes the carry trade a multi-year strategy rather than a one-off bet. When that persistence breaks — when the Bank of Japan finally hikes, as it did in 2024 — the funding currency's rate rises, the gap narrows, and the entire structure of trades built on the old low rate has to be rebuilt or unwound. The funding leg's stability is a load-bearing assumption, and when it cracks, it cracks for everyone at once.

## Decomposing the P&L: carry versus spot

Every carry trade's total return splits cleanly into two pieces:

> **Total return = carry income + spot move**

The **carry income** is the rate gap — known in advance, positive, banked daily. The **spot move** is the change in the exchange rate over your holding period — unknown, can be either sign, and far larger in magnitude than the carry on any given day. The figure below stacks them: a thin reliable green layer of carry income, a fat uncertain amber layer of spot move, summing to your total.

![P and L decomposition carry income plus spot move equals total return](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-5.png)

The asymmetry between the two layers is the entire psychology of carry. The carry income is small but certain; the spot move is large but uncertain. Day to day, the carry drips in and the spot wiggles around, and over a calm year the carry wins — you collect 4.10% and the currency barely moves, so you keep most of it. But the spot move has a fat, ugly left tail. Once in a while it does not wiggle; it gaps, and it gaps against you, and it takes years of carry with it.

#### Worked example: the breakeven spot move

How far can the Australian dollar fall against the yen before your carry income is fully eaten? That number is the **breakeven spot move**, and it is the single most important risk figure in the trade.

You earn **+4.10%** in carry over a year. For the trade to break even on a total-return basis, the spot move must be no worse than **−4.10%**. So:

- If AUD/JPY *rises* over the year — say AUD strengthens 3% against the yen — your total return is +4.10% carry + 3.00% spot = **+7.10%**. Carry plus a tailwind: the dream year.
- If AUD/JPY is *flat*, your total return is +4.10% + 0.00% = **+4.10%**. You kept the whole gap.
- If AUD/JPY *falls 4.10%*, your total return is +4.10% − 4.10% = **0%**. You worked all year for nothing; the spot move exactly ate the carry.
- If AUD/JPY *falls 12%* — the size of the August 2024 yen move — your total return is +4.10% − 12.00% = **−7.90%**. One bad month erased the carry and then some.

**Your breakeven is the rate gap itself: the high-yielder can fall by your annual carry before you start losing money, and not a pip more.** A 4.10% gap buys you a 4.10% cushion against an adverse move. In a market where the funding currency can move 12% in a month, that cushion is alarmingly thin.

### Why the spot move dwarfs the carry on any given day

Put numbers on the timescales and the danger becomes obvious. The carry income of 4.10% a year is about **0.011% per day** (4.10% ÷ 365). The daily standard deviation of a pair like AUD/JPY is on the order of **0.6%** — more than fifty times the daily carry. On a normal day the carry is a rounding error against the noise of the exchange rate. You are not collecting carry because it is large; you are collecting it because it is *relentless and positive* while the spot move averages to roughly zero over calm periods.

That is the carry trader's bargain: accept a tiny, certain positive every day in exchange for a small chance of a huge negative. It is, as the cliché goes, picking up pennies in front of a steamroller — a phrase so apt that this series gave it [its own post on carry crashes](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller). The pennies are real. So is the steamroller.

### The shape of carry returns: negative skew

The decomposition into a thin certain layer and a fat uncertain layer has a precise statistical signature, and it is the most important thing to internalise about carry. The return distribution is **negatively skewed**: most observations are small positives (the daily carry, plus a quiet spot), and a few observations are enormous negatives (the crash). The mean can be positive — carry makes money on average — while the *median* outcome and the worst outcome are wildly different. A strategy can show years of smooth, low-volatility, positive returns and still be sitting on a loss distribution where one bad month dwarfs all the good ones combined.

This breaks the usual risk intuition. When most people see a trade that has returned a steady few percent a quarter with low volatility for three years, they conclude it is *safe* and size it large. For carry, that conclusion is exactly backwards: the long calm stretch is not evidence of safety, it is the accumulation phase before the rare loss, and the low measured volatility *understates* the true risk because the danger is in a tail that has not printed yet. The most dangerous moment to be in a carry trade is right after a long, profitable, low-volatility run — which is, of course, precisely when it looks most attractive and is most crowded. The 2024 unwind happened after two of the calmest, most profitable years the yen carry had ever delivered. The calm was the setup.

This is why professional carry desks do not size on volatility alone. They cap leverage, they watch positioning and crowding, they buy cheap tail hedges in calm periods, and they treat a falling-volatility, rising-position environment as a *warning* rather than a green light. The math of why carry behaves like a short-volatility position — and how to manage it — is the subject of the dedicated volatility post; here the point is the shape itself: **carry pays you in small reliable pieces and bills you in rare enormous ones, and any risk model that assumes a symmetric bell curve will badly under-reserve for the bill.**

## The rate gap you ride: where carry comes from

The income on a carry trade is not magic — it is anchored in the actual gap between two countries' interest rates, and that gap moves. The cleanest real-world picture is USD/JPY against the US–Japan rate differential. When the Fed hikes and the Bank of Japan does not, the gap between US and Japanese two-year rates blows out, and the yen weakens almost in lockstep, because capital floods toward the higher US yield. The chart below shows both: the pair on the left axis (yen per dollar) and the rate gap on the right axis. They move together — the wider the gap you can earn, the harder the funding currency falls, which (until it reverses) is a tailwind on top of the carry.

![USD JPY level versus the US Japan two year rate gap 2019 to 2025](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-4.png)

This is the carry trade's home field. From 2021 to 2024 the US–Japan two-year gap widened from near zero to **4.35** percentage points, and USD/JPY weakened from 115 to 157 — the yen-funded trade earned the gap *and* rode the yen down. But look at the right edge: in 2025 the gap narrows to 3.40, and that narrowing is the warning. When the gap that pays you starts to close, the spot tailwind can flip to a headwind, because the same capital that flowed toward the high-yielder starts flowing back. The rate gap is both the source of your income and the trigger for the unwind that takes it away.

We will not re-derive *why* rate differentials move currencies — that mechanism belongs to monetary policy, and the macro series owns it. See [how monetary policy moves currencies through rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials). Here, the point is narrower and sharper: **the carry trader's income and the carry trader's risk come from the same number.** The wider the rate gap, the more you earn — and the more crowded the trade, the more violent the snap-back when the gap closes.

There is a feedback loop hiding in that chart that every carry trader must respect. A wide, widening rate gap does two things at once. First, it raises the carry income — more gap, more money per day. Second, it *attracts capital*, which pushes the high-yielder up and the funding currency down, adding a spot tailwind on top of the carry. Both effects make the trade more attractive, so more capital piles in, so the high-yielder rises further — a self-reinforcing inflow that can run for years. But the loop runs in reverse with terrifying speed. The moment the gap stops widening — when the funding central bank signals a hike or the high-yielder's central bank signals cuts — the capital that flowed in has every reason to flow back out, and because it is levered and crowded, it all tries to leave through the same door at once. The spot tailwind becomes a spot headwind, the carry cushion is too thin to absorb it, and the unwind feeds itself. The narrowing of the US–Japan gap on the right edge of the chart, from 4.35 in 2022 to 3.40 in 2025, is exactly the kind of signal that precedes an unwind.

#### Worked example: carry as a yield, compared to a bond

To feel how unusual the carry trade is, compare it to the safe alternative. Suppose you have \$1,000,000 and a one-year horizon.

- **Hold a US Treasury bill** yielding, say, 4.50%. After a year you have \$1,045,000, in dollars, essentially risk-free. Your money never left the dollar, so there is no exchange-rate risk at all.
- **Put on the yen-funded Aussie carry.** You earn the 4.10% rate gap = \$41,000, *plus or minus the spot move*. If AUD/JPY is flat you finish with \$1,041,000 — slightly *less* than the riskless T-bill, and you took currency risk to get there.

This is the uncomfortable truth the worked example exposes: **on a flat exchange rate, the carry trade barely beats the riskless rate, and it took on real currency risk to do it.** The carry trade only makes sense if you believe the high-yielder will *not* fall by the full rate gap — that the spot move will be flat or favourable often enough to compensate for the rare violent loss. That belief has a name, and it is the deepest idea in this post.

## Why carry is a bet that UIP fails

Here is the question that should be nagging you. If the Australian dollar pays 4.10% more than the yen, why doesn't *everyone* borrow yen and buy Aussie until the free money disappears? Markets are supposed to arbitrage away easy profits. What stops them?

The textbook answer is a theory called **uncovered interest parity (UIP)**. UIP says: a currency with a higher interest rate must be *expected to depreciate* by exactly that interest-rate advantage. The logic is an equilibrium argument. If the Australian dollar paid 4.10% more *and* was expected to hold its value, then everyone would pile in, and that buying pressure would push the Aussie up now and set it up to fall later — until the expected depreciation exactly cancelled the rate gap. In a world where UIP holds, the high-yielder's extra interest is *precisely* offset by its expected fall, and the expected profit from carry is **exactly zero**.

The figure below traces the fork. From the rate gap, theory (UIP) predicts the high-yielder falls by the gap, leaving expected profit at zero — parity holds. The other branch is what the data actually does, which splits again into the two faces of carry: the calm years where the currency stays flat or even rises and you bank the gap, and the rare crash where it overshoots and wipes out years of income.

![Why carry is a bet that uncovered interest parity fails branching to zero profit win or crash](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-8.png)

So if UIP held, carry would not be a strategy — it would be a coin flip with zero expected value, no matter how juicy the rate gap looked. **Carry is profitable if and only if UIP fails.** And it fails, persistently, in a very particular way: high-yielding currencies, on average, do *not* depreciate by their rate advantage. Often they barely move, and sometimes they even appreciate. This is one of the most robust empirical findings in all of finance — the **forward premium puzzle** — and it is the entire reason carry pays. This series devotes a full post to dissecting it: [uncovered interest parity and why it fails, the forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle). Read it alongside this one; the two are inseparable.

#### Worked example: the UIP-implied move versus what actually happens

Make the failure concrete. Suppose AUD/JPY is 100.00 (100 yen per Australian dollar), and the rate gap is 4.10% in the Aussie's favour.

- **What UIP predicts:** the Australian dollar should be expected to *depreciate* 4.10% against the yen over the year, ending at roughly 95.90 yen per Aussie. If that happened, your +4.10% carry on \$1,000,000 (= +\$41,000) would be exactly cancelled by a −4.10% spot loss (= −\$41,000). Net: \$0. The free lunch is a mirage.
- **What history usually does:** the high-yielder stays roughly flat or drifts *up*. Say AUD/JPY ends the year at 100.00, unchanged. Then your carry of +\$41,000 is *not* offset by anything, and you keep all \$41,000. You earned the gap because the currency refused to fall the way the theory said it must.
- **What history occasionally does:** in a crash, the move *overshoots* UIP wildly. In August 2024 the yen surged so hard that the high-yielders fell far more than any rate gap — a 12% move in weeks, three times the annual carry, in the *opposite* direction of the carry. That is UIP failing in the other direction, and it is where the years of accumulated carry go to die.

**The carry trade harvests the gap between what UIP predicts (the high-yielder falls by the rate gap) and what usually happens (it doesn't) — and pays for that harvest with the rare episode where the currency overshoots and takes it all back.** That single sentence is the whole trade. Everything else is sizing.

### Why does UIP fail? The risk-premium story

It is one thing to observe that UIP fails and another to understand *why* the failure is durable rather than a fluke that arbitrage should erase. The leading explanation is a **risk premium**. Carry is risky in a particularly nasty way — it loses exactly when investors can least afford it, in global panics, when their other assets are also falling and their need for cash is highest. An asset that pays off in good times and craters in bad times is worth *less* than its average return suggests, so investors demand extra compensation to hold it. That extra compensation is the carry. The high-yielder does not fall by the full rate gap on average *because* part of that gap is a risk premium that rational investors require for bearing crash risk, not a mispricing waiting to be arbitraged away.

This reframes the whole trade. You are not exploiting a market error; you are being *paid a wage* to hold a risk that other people want to offload. That is why carry survives decade after decade despite being one of the most-studied anomalies in finance. There is no free lunch to arbitrage — there is a risk, and a market price for bearing it, and the carry trader is the one who shows up to bear it. The corollary is sobering: if you earn the premium in the calm years, you are *contractually obligated* to pay it back in the crash. The premium and the crash are two sides of one coin. You cannot keep the wage and dodge the job.

A second strand of explanation is about *who* is on the other side. Funding-currency countries like Japan have households and institutions sitting on huge pools of low-yielding domestic savings, hungry for yield abroad — they are structural exporters of capital, persistent buyers of the carry. High-yielder countries often need that foreign capital to finance deficits. The carry flow is not just speculators; it is a structural cross-border capital flow that connects savers in low-rate countries to borrowers in high-rate ones. That structural demand is part of why the trade is so persistent and so crowded — and why, when it reverses, it reverses with the force of a structural flow slamming shut.

## Leverage: the multiplier on both income and ruin

A 4.10% return is not exciting enough to build a fund around, so carry traders lever it. Leverage means controlling a position larger than your capital by borrowing the rest. Because the carry income is a percentage of the *notional* (the full position size), levering up multiplies the income on your equity by the leverage ratio. It multiplies the losses by exactly the same ratio. There is no asymmetry in the math; the asymmetry is entirely in your ability to survive the loss.

The before-and-after below makes the point. On the left, an unlevered \$1,000,000 cash carry: +4.10% income (\$41,000), and a 5% adverse spot move is a survivable \$50,000 dent. On the right, the same trade at ten-times leverage: the income jumps to +41% on equity (\$410,000), but a 5% adverse move is now a \$500,000 hit — half your capital gone, and likely a margin call that forces you out at the worst possible moment.

![Leverage amplifies both the carry income and the spot loss ten times](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-6.png)

#### Worked example: the same trade at ten-times leverage

You have \$1,000,000 of equity. Instead of a \$1,000,000 cash position, you post that \$1,000,000 as margin and control a **\$10,000,000** yen-funded Aussie carry.

- **Carry income.** The 4.10% gap now applies to \$10,000,000 of notional: \$10,000,000 × 4.10% = **\$410,000** per year. Against your \$1,000,000 of equity, that is a **+41% return on equity** — from a 4.10% gross gap. This is why carry desks lever; the gap is small, the leverage makes it a real number.
- **The spot risk, also ×10.** A 5% adverse move in AUD/JPY now costs \$10,000,000 × 5% = **\$500,000**. That is half your equity, gone, on a move that happens routinely. The August 2024 move was 12%: \$10,000,000 × 12% = **\$1,200,000** — more than your entire capital. You are not just wiped out; you owe money.
- **The margin call.** Long before the move runs its full course, your broker marks the position to market, sees your equity evaporating, and demands more margin or liquidates you. Forced liquidation means you sell the high-yielder and buy back the funding currency *at the worst price*, locking in the loss and adding to the very selling pressure that is crushing the trade.

**Leverage turns a 4% gap into a 41% yield and a survivable 5% loss into a fatal one — and because everyone levers the same crowded trade, the forced selling on the way out is what turns a normal pullback into a crash.** The leverage is not a side detail; it is the mechanism that converts carry's small, fat-tailed risk into systemic, self-reinforcing unwinds.

This crowding-and-leverage dynamic — how a crowded, levered carry position becomes its own undoing — is the subject of a dedicated post on the relationship that governs the whole trade: [carry and volatility, the relationship that runs the trade](/blog/trading/forex/carry-and-volatility-the-relationship-that-runs-the-trade). The short version: carry sells volatility. When volatility is low and falling, carry is calm and profitable, and leverage piles in. When volatility spikes, the leverage unwinds all at once, and the spike feeds on itself.

### Sizing the trade: the discipline that keeps you alive

Given everything above, how do you actually run a carry book without blowing up? The answer is a set of disciplines that all flow from one fact — the risk is in the tail, not the average. The first discipline is **leverage caps that account for the tail, not the daily volatility.** If a 12% move can happen in a month and you are levered 10×, a 10% equity buffer is not survival, it is a fuse. Practitioners size so that a plausible *crash-sized* move — not a one-standard-deviation move — leaves them solvent and unforced. That usually means far less leverage than the smooth, low-volatility return stream tempts you to use.

The second discipline is **diversification across many carry pairs.** A single yen-funded Aussie carry is one bet on one pair. A book that funds in several low-rate currencies and holds a basket of high-yielders spreads the idiosyncratic risk — but, and this is the cruel part, it does *not* spread the crash risk, because in a panic the carry pairs all crash together. Diversification helps in normal times and fails exactly when you need it, because correlations across carry trades go to one in a crisis. So diversification reduces the everyday noise but does not save you from the tail; only sizing does.

The third discipline is **tail hedging in the calm.** Because carry is short volatility, the natural hedge is to be long a little volatility — to spend a sliver of the carry income on cheap out-of-the-money options on the funding currency (puts on the high-yielder, calls on the yen) that pay off in a crash. In quiet periods these options are cheap precisely because nobody fears the crash, which is the best time to buy them. The discipline is to keep paying that small insurance premium during the years it feels like a waste, so that you are still standing when the crash arrives.

#### Worked example: how much leverage is survivable?

Suppose you decide that no single plausible adverse move should cost you more than 20% of your equity, and you judge a *plausible* crash-sized move in your carry pair to be **15%** (smaller than 2024's 12%-in-a-month would scale to over a full unwind, but a sober planning figure). Then your maximum leverage is set by 15% × L ≤ 20%, so **L ≤ 1.33×**. On \$1,000,000 of equity that is a position of about **\$1,330,000** — a far cry from the 10× that the 41% headline yield invites.

- At that 1.33× leverage, your carry income is \$1,330,000 × 4.10% = **\$54,500** a year, about +5.5% on equity. Modest, but durable.
- A 15% crash costs \$1,330,000 × 15% = **\$199,500** — just under your 20% limit. You are bruised but solvent, and not force-liquidated.
- Compare the 10× sizing: the same 15% crash costs \$10,000,000 × 15% = **\$1,500,000**, one and a half times your entire capital. You are gone.

**The single most important decision in a carry trade is not which pair to hold — it is how much leverage to use, and the right answer is set by the size of the crash you must survive, not the size of the yield you want to earn.** Every carry blow-up in history is, at bottom, a leverage decision that assumed the tail would not arrive before the manager could get out.

## Common misconceptions

**"Carry is free money — you just collect the rate gap."** The rate gap is real income, but it is not free. You are compensated for bearing crash risk: the high-yielder can fall far more than the gap in a hurry. On a *flat* exchange rate the carry barely beats a riskless T-bill (a 4.10% gap versus a 4.50% bill, in our example), and you took currency risk to earn it. The gap is payment for a risk, not a gift.

**"A higher-yielding currency is a better carry trade."** Higher yield means *more compensation*, which means *more risk*, not more edge. The Turkish lira's 50% rate looks irresistible until you notice the lira has lost the bulk of its value against the dollar. The breakeven spot move on a 50% carry is a 50% depreciation — and the lira does that. The best carry trades historically have been *moderate* gaps in *liquid, stable* high-yielders (AUD, NZD), not the extreme gaps in fragile EM currencies where the depreciation usually swallows the yield whole.

**"If I am right about interest rates, I make money."** No. You can be perfectly right that the gap stays at 4.10% all year and still lose, because your return is carry *plus the spot move*, and the spot move is a separate bet. The carry trade is two bets bundled into one: a bet on the rate gap (usually right) and a bet on the exchange rate not falling by more than the gap (sometimes catastrophically wrong). The second bet is the one that pays the bills and the one that kills you.

**"The carry income protects me — I have a cushion."** The cushion is exactly the size of the annual carry: 4.10% for our trade. That is your entire buffer against an adverse move. In a market where the funding currency can move 12% in a month, a 4.10% cushion is gone in the first week of any real unwind. The carry softens small losses and is irrelevant to large ones.

**"Carry returns are normally distributed, so I can size it like any other trade."** Carry returns are *negatively skewed* with *fat tails* — long stretches of small gains punctuated by rare enormous losses. Standard volatility-based sizing systematically *underestimates* the risk, because the danger lives in the tail, not the everyday wiggle. Carry looks safe by the usual statistics right up until the moment it is the most dangerous trade in your book.

## How it shows up in real markets

Carry is not a textbook curiosity; it is one of the largest and most persistent flows in the entire financial system. Here is where you see it.

### The yen as the world's funding currency

For two decades, Japan's near-zero rates made the yen the default funding leg for the global carry trade. Borrow yen, sell yen, buy anything that yields more. This is why the yen behaves like a "risk-off" currency: when global markets panic and carry trades unwind, traders buy back the yen they borrowed, and the yen *strengthens* precisely when everything else is falling. The yen's safe-haven reputation is, in large part, just the mechanical reverse of the carry trade. When you see the yen surge on a bad day, you are watching carry positions being closed across the planet.

#### Worked example: the August 2024 yen unwind, in dollars

Take a desk running a \$10,000,000 yen-funded high-yield carry (10× levered on \$1,000,000 of equity) going into July 2024.

- **The good part.** From the start of 2024 the carry income accrued at roughly 4% of \$10,000,000 = \$400,000 a year, about \$33,000 a month. Plus, USD/JPY rose from 141 to 161.9 — the yen weakened, a tailwind worth tens of thousands more. The trade was up handsomely by early July.
- **The unwind.** USD/JPY fell from **161.9** on the third of July to **141.7** on the fifth of August — about a 12% move in the funding currency, against the position. On \$10,000,000 of notional that is roughly \$10,000,000 × 12% = **\$1,200,000** of loss — more than the \$1,000,000 of equity. A year of carry income (\$400,000) covered barely a third of it.
- **The forced exit.** With equity wiped, the margin call forced liquidation: buy back yen, sell the high-yielder, at the worst prices, while every other levered carry desk did the same. The crowded unwind drove the VIX to **65.7** intraday, a 2008-and-2020-only level — fear measured by the violence of everyone exiting the same trade at once.

**Years of patient carry can be erased in a single week of unwind, because leverage and crowding turn a 12% spot move into a margin call that forces you to sell into the crash.** This is not a tail you can diversify away within the carry trade; it *is* the carry trade.

### The carry crash table

Zoom out and the pattern repeats across every major unwind. The chart below collects the peak-to-trough drawdowns of a stylised carry strategy through the big episodes: −28% in the 1998 LTCM/Russia crisis, −30% in the 2008 global financial crisis, −10% in the 2011 eurozone scare, −8% in the 2015 Swiss franc shock, and −12% in the August 2024 yen unwind. Five crashes in twenty-six years — roughly one every five years, each one large enough to erase several years of the slow grind.

![Carry trade peak to trough drawdowns across major unwinds 1998 to 2024](/imgs/blogs/the-carry-trade-getting-paid-to-hold-a-currency-7.png)

Notice the timing: every one of these crashes coincided with a spike in global risk aversion. Carry does not crash on its own schedule — it crashes when *everything* crashes, when volatility spikes and leverage unwinds across all markets at once. This is why carry is sometimes described as being *short volatility* and *long the global risk cycle*. It pays you in calm and bills you in panic, and the panics cluster. The mechanics of how leverage breaks in these episodes are dissected in the macro series' study of the great unwinds: [carry-trade unwinds, 1998, 2008, 2024, when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

### The funding-currency hierarchy

Not all funding currencies are equal, and not all high-yielders carry the same risk. The choice of which currency to borrow and which to hold is itself a discipline. The yen and Swiss franc dominate the funding side because their rates are lowest *and* they tend to strengthen in a crisis, which is double trouble for a short position — but their deep liquidity makes them tradeable at scale. On the asset side, the trade-off is yield versus fragility: the Australian and New Zealand dollars offer moderate yield in liquid, stable markets, while the high-EM yielders (peso, real, lira, rand) offer fat yields wrapped around the constant threat of a sudden devaluation. Choosing your legs is a full topic of its own: [funding currencies versus high-yielders, JPY, CHF, and the rest](/blog/trading/forex/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest).

### Carry into emerging markets and the sudden stop

The richest carry — and the sharpest crashes — live in emerging markets. An EM high-yielder might pay 10% (Mexico), 12% (Brazil), or more, against a funding currency near zero. The rate gap is enormous, and in calm years the trade is spectacular. But the same capital that chases that yield is the first to flee at the first sign of stress, and EM currencies are far less liquid than the majors, so the exit is brutal. The pattern is so regular it has a name: the **sudden stop**. Foreign capital floods in chasing carry, the high-yielder appreciates and reserves build, and then a shock — a Fed hike, a commodity collapse, a domestic crisis — triggers a wholesale reversal, the currency gaps down, and the carry of three years is gone in three weeks. The 2013 "taper tantrum," when the mere hint of the Fed slowing its bond purchases sent EM currencies into a tailspin, is the textbook case.

This is also where the carry trade touches the developing world's monetary policy directly. When carry inflows pour into a small open economy, they push the local currency up and force the central bank to choose between letting it appreciate (hurting exporters) or printing local currency to buy the inflows (risking inflation). When the carry reverses, the same central bank must defend the currency by spending its hard-won reserves or hiking rates into a slowdown. Countries that run managed currencies — Vietnam's dong is a clear example, with its central reference rate and its trading band around it — feel the carry cycle as waves of inflow pressure and outflow pressure that the central bank must lean against. The connection between hot-money flows, reserves, and a managed exchange rate is explored in [Vietnam's monetary policy: the State Bank, the dong, and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling). The carry trade is not just a hedge-fund game; it is a force that shapes how small countries can run their own money.

### Carry as a permanent feature, not a fad

The deepest real-market fact is that carry has not been arbitraged away despite being known for decades. It persists because UIP keeps failing, and UIP keeps failing because someone has to bear the crash risk and demand compensation for it. Carry is the market's way of paying patient capital to absorb the risk that a high-yielding currency suddenly collapses. As long as that risk exists and someone is willing to be paid to hold it, the carry trade will exist. It is not a glitch in the system; it is a *job* the system needs done, and the rate gap is the wage.

## The takeaway: how to read a currency like a carry trader

Strip the carry trade down to its irreducible core and you are left with one of the most honest trades in finance: you are paid a known, steady income — the rate gap — in exchange for accepting a small, hidden, occasionally enormous risk that the exchange rate moves against you. There is no edge from cleverness here; the edge is from *bearing a risk other people don't want*, and getting paid the rate gap to do it.

So when you next look at a table of interest rates across currencies, read it the way a carry trader does. Each rate is not just a yield — it is a *wage for holding that currency's crash risk*. The low rates (yen, franc) are what you pay to borrow. The high rates (peso, real, lira) are what you are paid to hold, and the size of the rate tells you how much crash risk the market thinks is packed in. A 4% gap is a modest wage for modest risk; a 50% gap is a fortune offered for a currency that the market fully expects to fall apart. The gap *is* the risk premium, laid bare.

That reading reframes a lot of market behaviour you would otherwise find mysterious. Why does the yen rally on a day of global panic, when Japan's own economy is doing nothing special? Because carry trades funded in yen are being unwound worldwide, and unwinding means buying yen back. Why do emerging-market currencies that paid double-digit yields for years suddenly collapse together in a week? Because the carry that financed them reverses as one structural flow. Why does a falling, calm volatility regime so often precede a violent one? Because the calm draws in the leverage that makes the next spike explosive. Once you see the carry trade clearly, you can read these moves not as random shocks but as the predictable mechanics of a crowded, levered position being put on and taken off. The carry trade is a lens, not just a strategy.

And the practical discipline that falls out of all this is simple to state and hard to follow: respect the tail before you reach for the yield. The headline 41% looks like the reason to do the trade; it is actually the bait. The real decision is the leverage, the real risk is the crash, and the real edge — the thing you are actually being paid for — is the willingness to bear a loss that arrives rarely and all at once. If you are not sized to survive that loss, you do not own a carry trade; you own a short option that has not been exercised yet.

And remember the spine of this whole series: you never own a currency in isolation. The carry trade makes that vivid. You are always long one currency and short another, earning the gap between two interest rates, exposed to the relative price of two monies. The income comes from the rate gap; the risk comes from the spot move; the profit exists only because uncovered interest parity fails; and the whole edifice can come down in a week when leverage unwinds. Get paid to hold the currency — but never forget you are standing in front of a steamroller, collecting pennies, and know exactly how thin your cushion is before the next one rolls through.

## Further reading & cross-links

- [Uncovered interest parity and why it fails: the forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle) — the theory carry bets against, and the empirical failure that makes carry pay.
- [Carry and volatility: the relationship that runs the trade](/blog/trading/forex/carry-and-volatility-the-relationship-that-runs-the-trade) — why carry is short volatility, and how leverage and crowding govern the whole trade.
- [Carry crashes: picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller) — the fat left tail in full, and how the unwinds detonate.
- [Funding currencies vs high-yielders: JPY, CHF, and the rest](/blog/trading/forex/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest) — choosing which currency to borrow and which to hold.
- [Interest-rate differentials: the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx) — why the rate gap drives currencies in the first place.
- [Carry-trade unwinds, 1998, 2008, 2024: when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the macro anatomy of every great carry unwind.
- [How monetary policy moves currencies: rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — the policy mechanism behind the gap you earn.
