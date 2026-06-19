---
title: "Carry and Volatility: The Relationship That Runs the Trade"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a currency carry trade behaves like selling insurance — steady premium in calm, catastrophic loss in a shock — and how the carry-to-vol ratio, the Sharpe of carry, and the low-vol-invites-leverage loop drive the whole cycle."
tags: ["forex", "currencies", "carry-trade", "volatility", "carry-to-vol", "short-volatility", "leverage", "vix", "risk-management", "yen-carry"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A currency carry trade is short volatility: it collects a small, steady premium while markets are calm and pays one enormous bill when they are not. That is the same risk shape as selling insurance.
>
> - The signal that actually drives the trade is not the rate gap alone but the **carry-to-vol ratio** — the rate gap divided by the pair's volatility. A 4.5% gap is a strong signal at 8% vol and a weak one at 18% vol.
> - **Falling volatility is dangerous, not reassuring.** Low vol raises the carry-to-vol ratio, which invites leverage, which crowds the trade — quietly building the very position that crashes.
> - The number to remember: in the August 2024 yen-carry unwind, USD/JPY fell about **12% in five weeks** while the VIX spiked intraday to **65.7** — a calm-market premium of a few percent a year wiped out in days.

On the morning of August 5, 2024, traders in Tokyo woke up to a market that had stopped working the way it was supposed to. The yen, which everyone had been short for years because Japanese interest rates were near zero and you got paid to borrow yen and hold almost anything else, was screaming higher. USD/JPY had been quoted around 162 yen per dollar in early July. By the Asian open on August 5 it was trading near 142. The Nikkei fell 12% in a single session — its worst day since the 1987 crash. Wall Street's "fear gauge," the VIX, spiked to 65.7 intraday, a level seen only in the 2008 financial crisis and the 2020 pandemic crash.

Nothing fundamental had broken. There was no bank failure, no sovereign default, no war that morning. The Bank of Japan had merely nudged its policy rate from roughly 0.10% to 0.25% a few days earlier, and a soft U.S. jobs report had shifted Fed expectations. Tiny news. Yet one of the most popular, most profitable trades in global finance — borrow cheap yen, lend it out at higher rates elsewhere — went into a violent, self-reinforcing unwind that erased years of patient gains in three trading days.

This is the central paradox of the carry trade, and it is the subject of this post. The trade that paid you a steady, almost boring premium for years is the same trade that just cost a fortune in a long weekend. To understand why, you have to stop thinking of carry as "a yield play" and start thinking of it as something else entirely: **a short-volatility position**. Carry is selling insurance. You collect a premium in calm weather and you pay the claim when the storm hits. Everything that makes carry frustrating — the slow grind up, the sudden cliff, the way low volatility lures in more and more money right before the crash — falls out of that one idea.

![Carry index versus the VIX from 2007 to 2025 with crash years marked in red](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-1.png)

Study the chart above before we go further. The blue line is a stylized index of a basket carry trade in the major (G10) currencies, rebased to 100 in 2007. The amber dashed line is the VIX, the market's expectation of how much the U.S. stock market will move — a clean proxy for global "fear." Notice the shape. The blue line grinds upward, year after year, in a calm, unspectacular climb. And then, abruptly, it falls off a cliff — in 2008, in 2011, in 2015, and again in 2024 (the red dots). Every one of those cliffs lines up with a spike in the amber line. Carry goes up the stairs and down the elevator, and the elevator is volatility.

This post builds that relationship from the ground up. We will define carry from zero, show *mathematically* why it is a short-volatility trade, derive the carry-to-vol ratio that professionals actually use to size the position, explain the feedback loop by which falling volatility plants the seeds of the next crash, and then walk through two real episodes — the 2007 low-vol build-up that detonated in 2008, and the 2024 low-vol calm that detonated in August. By the end you will read a quiet, low-volatility market not as a safe one but as a loaded one.

The spine of this whole series is worth restating, because carry sits right on top of it: an exchange rate is the relative price of two monies, and what moves it is the gap between two countries' interest rates plus the flow of money across borders. Carry monetizes the *first* half of that sentence — the rate gap — and gets destroyed by the *second* half — the flow, when it reverses all at once.

## Foundations: Why carry is a short-volatility trade

Let us define every piece from scratch, because the whole argument rests on getting these primitives exactly right.

**A currency pair.** You never trade "a currency" alone. You always trade one against another — a pair. EUR/USD is the price of one euro in dollars; USD/JPY is the price of one dollar in yen. The first currency is the **base**, the second is the **quote**. When USD/JPY rises from 150 to 157, the dollar got stronger and the yen got weaker: you need more yen to buy one dollar. (If pips, base, and quote are new to you, the series post [base, quote, pips, and how to read an FX quote](/blog/trading/forex/base-quote-pips-and-how-to-read-an-fx-quote) walks the mechanics; here we just need the idea that every position is a relative bet.)

**The rate gap (the carry).** Every currency comes with an interest rate attached — roughly the rate you earn for holding it, or pay for borrowing it, set by that country's central bank and its bond market. In late 2024, holding U.S. dollars earned roughly 4.5% a year (the Fed's policy rate), while holding Japanese yen earned roughly 0.25%. That difference — about 4.25 percentage points — is the **interest rate differential**, and it is the engine of the carry trade. (The differential is the master variable of FX; the series treats it in depth in [interest rate differentials, the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx). We won't re-derive the rates themselves — that's owned by the fixed-income world, e.g. [forward rates, what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be). We just *use* the gap.)

**The carry trade.** You borrow the low-rate currency (the *funding* currency, classically the yen or the Swiss franc) and use the proceeds to hold the high-rate currency (the *target* currency — the dollar, the Australian dollar, the Mexican peso). As long as the exchange rate doesn't move against you, you pocket the rate gap. If you borrow yen at 0.25% and hold dollars at 4.5%, you earn the 4.25% spread for simply holding the position. It feels like free money. (The full anatomy of the trade — funding leg, target leg, leverage, roll — gets its own post in [the carry trade, getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency); here we focus on its *risk shape*.)

**Volatility.** Volatility is just how much a price moves around, usually measured as the annualized standard deviation of returns and quoted in percent. If USD/JPY has 8% annualized volatility, a rough rule of thumb is that over one year the pair will typically move within about ±8% of where it started; over one day it will typically move about 8% ÷ √252 ≈ 0.5%. Low volatility means small, calm daily wiggles. High volatility means large, violent swings. In FX, traders watch **implied volatility** — the volatility priced into currency options, which is the market's *forecast* of future movement — and they watch the equity VIX as a global risk barometer. Options pricing and the Greeks are owned by the options world; the post [the volatility smile and skew, why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) covers how the market prices the *shape* of that fear. We just need volatility as a number that goes up when markets get scared.

Now the key move. Why is carry *short* volatility?

A position is **"short volatility"** when it makes a little money when nothing happens and loses a lot when something big happens — in either direction, but especially to the downside. The textbook short-vol position is selling an option: you collect the premium up front, and as long as the underlying stays calm and inside a range, you keep it; if the underlying makes a huge move, you owe far more than the premium. You are *short* the thing — volatility — that the option buyer is *long*.

Carry has exactly that payoff shape, even though no option is involved. Here is the mechanism. The carry profit per period is approximately:

> **carry P&L ≈ (rate gap) − (change in the exchange rate against you)**

The rate gap is small, positive, and *steady* — you earn the 4.25% in smooth daily slivers. The exchange-rate term is the wild card. In calm markets, the target currency drifts sideways or even appreciates (because high-yield currencies attract inflows), so the second term is near zero or helps you, and you keep the whole gap. But currencies don't crash *upward* for the carry trader — the funding currency (yen) spikes *up* against the target in a panic, because in a risk-off shock everyone scrambles to buy back the cheap currency they borrowed. So the exchange-rate term, when it bites, bites hard and in one direction: a large loss. Steady small gain, occasional large loss, asymmetric to the downside. That is the short-vol signature.

#### Worked example: the rate gap collected in calm

You put on a \$1,000,000 carry position: borrow yen, hold dollars, at a 4.25% annual rate gap. In a calm year where USD/JPY barely moves, your carry income is roughly:

> \$1,000,000 × 4.25% = **\$42,500 a year**, which arrives as about \$42,500 ÷ 252 ≈ **\$169 per trading day** of smooth, drip-fed profit.

If you run the position at 5× leverage (borrowing 4 dollars of position for every 1 dollar of your own capital — utterly normal in FX, where margin requirements are tiny), your return on *your* capital is roughly 4.25% × 5 ≈ **21% a year** in a flat market. Now you see the seduction: a fifth of your money, every year, for holding a position that "does nothing." The intuition to carry forward: the premium is real and steady, but it is being paid to you *for bearing a risk that has not shown up yet*.

That last clause is everything. You are not earning 21% for free. You are earning it the way an insurance company earns premiums: by promising to absorb a disaster that hasn't happened *this* year.

One more foundational piece makes the crash direction inevitable rather than accidental: **why the funding currency spikes in a panic.** The classic funding currencies are the yen and the Swiss franc, and they share two properties. First, they are *low-yield*, which is what makes them cheap to borrow — that is why they are chosen as funding legs in the first place. Second, and less obviously, they are *safe-haven* currencies: in a global risk-off event, capital flees toward Japan and Switzerland (deep, liquid, current-account-surplus economies with large net foreign asset positions), so demand for yen and francs surges exactly when fear surges. Put those two facts together and you get the carry trader's nightmare. The currency you are *short* — the one you borrowed and owe back — is the one that rallies hardest in a crisis. A trader who borrowed yen to hold dollars is short yen; when the panic hits and the yen rockets, that short position is precisely where the loss lands. The crash is not random bad luck; it is the funding currency doing exactly what safe havens do, in the one moment a carry trader cannot tolerate it. This is why a yen-funded carry trade is *more* dangerous than the rate gap alone suggests: the funding leg is itself a coiled spring that releases on the same trigger that ends the calm.

## The short-volatility payoff: a thin premium and one cliff

Let us make the payoff shape concrete, because "short volatility" stays a slogan until you draw it.

![The short-volatility payoff of a carry trade shown as calm collect-premium versus shock pay-the-claim](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-2.png)

The figure above contrasts the two regimes the carry trade lives in. On the left, the calm regime: volatility is low and falling, risk appetite is high, you collect the 4.5% rate gap, you add leverage to amplify a thin return into a fat one, and the realized track record looks like a high-Sharpe money machine. On the right, the shock regime: volatility spikes (in August 2024, to a VIX of 65.7), every leveraged holder of the same trade reaches for the same exit at the same instant, the funding currency gaps 12% in three days, and a single loss erases years of accumulated premium. Same trade. Two faces.

This is structurally identical to the payoff of **selling a put option** on the target currency. When you sell a put, you collect a premium today. If the currency stays above the strike, you keep the whole premium — a small, bounded gain. If the currency collapses below the strike, your losses grow one-for-one with the size of the move — a large, effectively unbounded loss. Plot the profit on the vertical axis against the currency's price on the horizontal axis and you get a flat line on the right (you keep the premium) that kinks downward into a steep diagonal on the left (you eat the crash). That kinked shape — flat top, steep left tail — *is* the carry trade's economic payoff, reconstructed without ever touching an option. Carry is a synthetic short put on the high-yield currency.

Why does this matter so much? Because a short-put payoff has a property that fools almost everyone: **it produces a beautiful track record for a long time, precisely because the bad outcome is rare.** Most years, no crash happens, so the strategy posts steady positive returns with low realized volatility. The numbers look fantastic — high average return, low standard deviation, high Sharpe ratio. The strategy *appears* low-risk by every standard backward-looking statistic. And then the put gets exercised — the crash arrives — and a single observation swamps the entire history.

#### Worked example: how one bad day eats years of premium

Suppose your levered carry book earns a steady **+0.40% per week** in calm markets (that's the 21%-a-year levered return spread across 52 weeks). You run it for three years — 156 weeks — without incident. Your cumulative gain is roughly:

> 156 weeks × 0.40% ≈ **+62%** of capital. Beautiful.

Then a shock hits. The funding pair gaps 12% against you, and because you are levered 5×, your loss is:

> 12% × 5 = **−60%** of capital in a matter of days.

Three years of patient, "low-risk" 0.40%-a-week premium — gone in one week. The intuition: a short-vol strategy doesn't lose money gradually and proportionally; it gives back the whole accumulated premium (and often more) in a single event, because the premium was *compensation for that exact event all along*.

This is why carry returns are famously **negatively skewed**: lots of small wins, a few enormous losses. The average looks good; the distribution is a trap. A strategy with positive average return and negative skew is the financial-market equivalent of picking up pennies in front of a steamroller — a phrase so apt the series gives the crash mechanics their own post, [carry crashes, picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller).

## The carry-to-vol ratio: the signal that actually drives sizing

If carry is short volatility, then volatility is not a side detail — it is *the denominator of the whole trade*. The raw rate gap tells you how much premium you collect. But it tells you nothing about how much risk you are taking to collect it. To get the signal that professionals actually trade on, you divide the carry by the volatility.

![The carry-to-vol ratio stacked from raw rate gap through risk-adjustment to position size](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-6.png)

The **carry-to-vol ratio** is exactly what it sounds like:

> **carry-to-vol = (annual rate gap) ÷ (annualized volatility of the pair)**

It answers the only question that matters for sizing: *how many units of expected premium do I get per unit of risk I take?* It is the ex-ante (forward-looking) cousin of the Sharpe ratio — and in fact, for a pure carry position where the rate gap *is* the expected return, the carry-to-vol ratio is essentially the trade's expected Sharpe ratio. (The stylized historical Sharpe of a diversified G10 carry factor is around 0.5–0.6 over multi-decade samples — better than holding the stock market, which is exactly why the trade is so popular and so persistent.)

The crucial insight is what happens to this ratio as volatility moves. The numerator — the rate gap — is set by central banks and changes slowly, in 0.25% steps every few weeks at most. The denominator — volatility — can double or halve in a day. So the carry-to-vol ratio is driven *overwhelmingly by the denominator*. The same trade can look like a screaming buy on Monday and a marginal hold on Friday, with the rate gap totally unchanged, purely because volatility moved.

#### Worked example: the carry-to-vol ratio at three volatility levels

Take a fixed rate gap of **4.5%** — say, holding dollars against yen. Watch what the carry-to-vol ratio does as the pair's annualized volatility moves through three regimes:

> **Calm regime, vol = 8%:** ratio = 4.5% ÷ 8% = **0.56**. Strong. You are paid generously per unit of risk.
>
> **Normal regime, vol = 12%:** ratio = 4.5% ÷ 12% = **0.38**. Fair. A respectable but unexciting signal.
>
> **Stress regime, vol = 18%:** ratio = 4.5% ÷ 18% = **0.25**. Weak. You are taking a lot of risk for the same thin premium.

The rate gap never changed — it was 4.5% in all three cases. Yet the quality of the trade more than halved, from 0.56 to 0.25, purely because volatility rose. The intuition: **carry quality is a volatility story, not a yield story.** A high yield in a high-vol currency (think the Turkish lira at 50% rates but with 22% implied vol) can be a *worse* risk-adjusted trade than a modest yield in a calm one.

This is why volatility-targeting is the dominant way professional carry books are run. Instead of holding a fixed notional, the manager sizes the position so that the *expected risk* is constant. When volatility falls, the same risk budget allows a *larger* position; when volatility rises, the position is automatically cut. Mechanically:

> **position size ∝ (target risk) ÷ (current volatility)**

Hold that formula in your mind, because it is the hinge of this entire post. It is also, as we are about to see, the source of the instability. A rule that says "buy more when it's calm, sell when it's turbulent" sounds prudent. But run by thousands of players at once, it is a machine for buying high and selling low — for adding leverage right before the crash and dumping it right into the bottom.

There is a subtlety worth pausing on, because it is where a lot of carry blow-ups are actually born. The volatility you put in the denominator can be measured two ways, and they disagree at exactly the worst moment. **Realized volatility** is backward-looking — the standard deviation of the pair's *actual* moves over the last, say, 30 or 90 days. **Implied volatility** is forward-looking — the volatility the options market is *pricing in* for the future. In a long calm stretch, realized volatility falls faster and lower than implied, because the pair has, in fact, been quiet. A desk that sizes on realized vol will therefore size up *more aggressively* than one that sizes on implied vol, because its denominator is smaller. That is the trap: realized vol is a rear-view mirror, and the carry trade crashes through the windshield. The pair was calm yesterday; that tells you almost nothing about the gap risk tonight when the central bank meets. Sizing on trailing realized volatility is how a book ends up at its largest precisely when the latent risk is at its highest — the denominator is reporting the *past* while the position is exposed to the *future*.

It also matters *which* volatility you use because the two carry a built-in warning. When implied volatility sits well above realized volatility, the options market is charging a high price for protection even though nothing has moved yet — it is telling you the calm is fragile. A carry trader who only looks at the soothing realized number misses that signal entirely. Reading the gap between implied and realized vol is one of the few ways to see the build-up phase *in a price* rather than only in positioning data.

## Why falling volatility invites leverage — and sows the crash

Here is the part that separates people who understand carry from people who merely trade it. The dangerous moment for the carry trade is not when volatility is high. It is when volatility is *low and falling*. Low volatility is not the absence of risk; it is the accumulation of it.

![Why falling volatility sows the next carry crash, a feedback loop from calm to crowded to crash](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-4.png)

Trace the loop in the figure. It starts, innocently, with **volatility falling**. Markets are calm. Now follow the consequences:

1. **The carry-to-vol ratio rises.** Volatility is the denominator, so when it falls, the risk-adjusted attractiveness of the carry trade goes *up* even with the rate gap unchanged. The trade looks better than it did last month.

2. **Risk models permit more size.** Every serious trading desk runs a Value-at-Risk (VaR) model that estimates how much it could lose on a bad day, and that estimate is driven by *recent* realized volatility. When volatility has been low, VaR is low, so the risk manager's limit allows a bigger position. The same is true of volatility-targeting funds: the formula `size ∝ target ÷ vol` mechanically tells them to scale up. The model isn't being reckless by its own lights — it is faithfully reporting that recent risk was low. It just has no way of knowing that low *realized* risk is hiding rising *latent* risk.

3. **Traders add leverage.** Higher permitted size plus a better-looking ratio plus the pressure to hit return targets equals more leverage layered onto the same trade. A book that ran 5× in normal times might drift to 8× or 10× as the calm persists.

4. **The position gets crowded.** Crucially, *everyone is doing the same thing at the same time*, because everyone is looking at the same low volatility and the same attractive rate gaps. Carry is not a secret. The crowd piling into short-yen, long-dollar grows quietly, position by position, fund by fund.

5. **Everyone shares the same exit.** A crowded, levered, one-directional position has a fatal property: there is only one door, and everyone will reach for it at the same instant. As long as no one moves, the calm persists and feeds itself — the very stability of the trade *causes* more money to flow into it, which dampens volatility further, which invites yet more leverage. This is a self-reinforcing calm.

6. **A shock spikes volatility, and the unwind crashes.** It does not take much. A central-bank surprise, a soft data print, a single large fund deciding to cut. Volatility ticks up; VaR models flash red; volatility-targeting formulas scream "cut the position"; the most-levered players are forced to sell first; their selling moves the price, which raises volatility further, which forces the next tier of players to sell. The same feedback loop that built the position in slow motion now runs it in reverse at terrifying speed. The funding currency rockets, the carry crashes, and the premium of years evaporates in days.

The deep, uncomfortable lesson is this: **the carry trade contains the seeds of its own destruction, and low volatility is the fertilizer.** The calmer it gets, the more leverage piles in, the more crowded and fragile the position becomes, and the larger the eventual unwind. Stability is destabilizing — an idea the economist Hyman Minsky built a whole theory of financial crises around, and which the carry trade demonstrates in miniature with almost mechanical reliability.

#### Worked example: sizing up as vol falls, and the loss when it spikes

Run the volatility-targeting formula through a full cycle. You target a constant **10% annualized risk** on a carry book with a 4.5% rate gap.

> **Year 1, calm: realized vol = 8%.** Your position size is scaled to `10% ÷ 8% = 1.25×` your base notional. You feel safe — vol is low! Your expected carry income on \$1,000,000 of base notional is 1.25 × \$45,000 = **\$56,250**.
>
> **Year 2, calmer still: realized vol = 6%.** The formula now says `10% ÷ 6% = 1.67×`. Volatility fell, so you *added* size — you are now running \$1,670,000 of effective position, collecting 1.67 × \$45,000 ≈ **\$75,000**. You are more exposed than ever, precisely because it has been so quiet.
>
> **The shock: vol spikes to 30% and the pair gaps 12% against you in a week.** Your loss is on the *enlarged* position: 12% × \$1,670,000 = **−\$200,400**. That single week erases more than three years of the calm-market premium you had been collecting (\$56k + \$75k + … ).

The intuition: the volatility-targeting rule did not protect you — it *maximized your exposure at the worst possible moment*, because it sized up through the calm right into the storm. Falling volatility didn't signal safety; it signaled that the position had grown large enough to hurt.

## Carry as selling insurance: the cleanest mental model

If you remember one framing from this entire post, make it this one: **a carry trade is an insurance policy you have sold.** Everything we have derived — the steady premium, the rare catastrophic loss, the negative skew, the way calm invites more underwriting — is just the economics of being an insurer.

![Carry as selling insurance, a pipeline from collecting premium through calm years to paying the claim](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-5.png)

Walk the pipeline. You **sell the policy** by holding the high-yield currency funded in the low-yield one — you have promised to absorb a particular disaster (a risk-off shock that spikes the funding currency). You **collect the premium** every single day in the form of the rate gap. **Calm years pass** with no claim filed, and your book looks wonderfully profitable. Because the premium looks like free money, you do the thing every insurer is tempted to do in a soft market — you **grow the book**, writing more policies (more leverage, more pairs, bigger size) at ever-thinner effective premiums. And then **disaster strikes**: the volatility spike, the risk-off panic, the crowded exit. You **pay the claim**, and the claim is large, sudden, and arrives exactly when you can least afford it — because the same shock that triggers your claim is triggering everyone else's, so prices are gapping and liquidity has vanished.

The insurance analogy earns its keep because it predicts the *behavior* of carry, not just its payoff:

- **Why the premium exists at all.** Uncovered interest parity — the textbook prediction that the high-yield currency should depreciate by exactly the rate gap, leaving no profit — *fails empirically*, and it fails because someone has to be paid to bear crash risk. The carry premium is a risk premium for selling insurance against currency crashes. (The series unpacks why parity breaks in [uncovered interest parity and why it fails, the forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle); the short version is: the premium is the insurance fee.)

- **Why it is correlated across pairs.** Insurers against the same catastrophe all pay out together. Carry trades across different currency pairs look diversified in calm markets but collapse together in a crisis, because they are all short the *same* underlying risk — global risk appetite. When that one factor turns, every carry pair crashes at once. (Cross-asset correlation in a crisis is its own deep topic: [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).)

- **Why low volatility is the soft market.** In insurance, a long stretch without disasters drives premiums down and tempts underwriters to write more policies at worse terms — until a single catastrophe blows up the under-priced book. That is *exactly* the carry cycle. Low FX volatility is a soft insurance market: thin premiums, fat books, and a reckoning building.

#### Worked example: pricing the insurance you are selling

Suppose the carry trade pays you 4.5% a year, and history says a carry crash that costs you ~30% of your position arrives, on average, about once every six years. What is your expected annual return *after* accounting for the claim?

> **Premium collected:** +4.5% per year.
>
> **Expected claim cost:** a 30% loss once every 6 years is an *expected* annual cost of −30% ÷ 6 = **−5.0% per year**.
>
> **Net expected return:** +4.5% − 5.0% = **−0.5% per year**, *before* leverage.

On these (deliberately pessimistic) numbers the unlevered trade barely breaks even — the entire premium is fair compensation for the crash, with a sliver left over or even a small deficit. The intuition: carry only "works" if you (a) earn the premium for many years before a claim, (b) survive the claim without being forced to liquidate at the bottom, and (c) size it so the inevitable loss is survivable. The edge is not the headline yield; it is the *risk management of the tail*. Picking up the pennies is easy. Not getting hit by the steamroller is the whole job.

The insurance frame also explains a feature of carry returns that confuses people who only look at averages: the gap between the *arithmetic* and the *geometric* return. Arithmetically, the trade has a positive expected return — most years are wins. But what you actually compound is the geometric return, and a single −60% year is geometrically devastating: after losing 60% you need a +150% gain just to get back to where you started, which a 4.5%-a-year premium takes the better part of a decade to deliver. Negative skew quietly eats compounding. Two strategies with the same arithmetic mean but different skew end up in wildly different places after a few decades, and the negatively-skewed one — carry — ends up poorer than its glossy average return advertises. The insurer who under-prices catastrophe risk looks brilliant on an arithmetic basis right up to the year the catastrophe bankrupts the geometric path.

There is one more thing the analogy gets right, and it is the most important practical point in the whole post: **good insurers hold capital against the claim; bad ones spend the premium as it arrives.** A disciplined carry trader treats the steady premium not as profit to be levered up and consumed but as a fund to be held against the inevitable claim — running the position small enough, with enough capital in reserve, that the crash is a bruise rather than a death. The blow-ups are almost never the traders who *had* the carry trade on; they are the traders who had it on at 8× or 10× leverage, spent the premium, and had no reserve when the claim arrived. The carry trade does not kill you. Carry-trade *leverage* kills you.

## How volatility levels differ across pairs

Not every carry pair carries the same risk, because the denominator of the carry-to-vol ratio — the volatility — differs enormously across currencies. This is why the raw yield is such a misleading guide to which carry trade is "best."

![Implied volatility by currency pair from EUR USD up to USD TRY](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-7.png)

The chart shows representative one-month at-the-money implied volatilities — the market's forecast of how much each pair will move — across the spectrum from the calmest majors to the most violent exotics. EUR/USD, the most-traded pair on earth, sits at the bottom around 7%. USD/JPY runs a bit higher near 9.5% (the yen is a funding currency that spikes in risk-off, so it carries extra crash-fear). The commodity-linked AUD/USD is around 10.5%. And then the emerging-market and exotic pairs blow out: USD/MXN around 13%, USD/TRY around 22% — *three times* the volatility of EUR/USD.

Now layer the yields on top. The Turkish lira offered roughly 50% interest in 2024; the Mexican peso roughly 10%; the Australian dollar roughly 4.35%. A naive yield-chaser sees 50% and dives in. But run the carry-to-vol ratio:

#### Worked example: the high-yield pair is often the worse trade

Compare two carry trades funded in dollars, on the carry-to-vol ratio rather than the headline yield.

> **Mexican peso (MXN):** rate gap ≈ 10.25% − 4.5% = 5.75% over USD funding; implied vol ≈ 13%. Carry-to-vol = 5.75% ÷ 13% ≈ **0.44**.
>
> **Turkish lira (TRY):** rate gap ≈ 50% − 4.5% = 45.5% over USD funding; implied vol ≈ 22%. Carry-to-vol = 45.5% ÷ 22% ≈ **2.07** *on paper*.

At first glance the lira looks like the trade of the century — a ratio above 2.0. But that 22% implied vol drastically understates the *tail*: the lira does not move in tidy 22% bands, it lurches in sudden 30–40% devaluations that no symmetric volatility number captures. The true risk-adjusted return, once you account for the fat left tail and the devaluation that wipes out years of the 45% premium, is far worse than the clean number suggests. The intuition: the carry-to-vol ratio is a *first-order* guide, but for the most volatile, most crash-prone pairs you must look past the symmetric volatility to the skew — the asymmetry of the move — which the next post in this track is built around.

The general principle: a carry pair's danger is set by its volatility *and* its skew, not its yield. A modest yield in a calm pair can dominate a huge yield in a violent one on a risk-adjusted, survive-the-tail basis. This is also why diversified G10 carry baskets have historically beaten concentrated single-name EM carry on a Sharpe basis — the majors give you a cleaner premium-to-risk trade.

## The crash setup: every carry unwind has the same anatomy

Carry crashes are not random. They share an anatomy so consistent that once you have seen one, you can recognize the setup for the next. The figure of every carry crash arriving with a volatility explosion makes the point in one frame.

![VIX peaks at four crises showing every carry crash arrived with a volatility explosion](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-3.png)

Each bar is the VIX peak during a crisis that crashed the carry trade: LTCM/Russia in 1998 (VIX ~45.7), the global financial crisis in 2008 (~80.9), the COVID crash in 2020 (~82.7), and the yen-carry unwind of August 2024 (~65.7). The dashed line marks a typical calm-regime VIX around 13. The visual claim is unambiguous: **carry does not crash a little when volatility rises a little. It crashes a lot when volatility explodes by a multiple.** A move from 13 to 65 is not a 5-point change; it is a 5× change, and the carry trade — being short volatility — takes a loss proportional to *that*.

The anatomy, every time, runs in four stages:

1. **The build-up (low vol).** A long stretch of calm. Volatility grinds lower, the carry-to-vol ratio looks great, risk limits expand, leverage accretes, and the trade gets crowded. This phase is *quiet* — that is the point. Nothing looks wrong; the track record looks superb. The fragility is invisible in the price and visible only in the positioning.

2. **The trigger (a shock).** Some catalyst — often small in absolute terms — flips risk appetite. A central bank surprises (BoJ hiking in 2024). A sovereign defaults (Russia in 1998). A pandemic locks down the world (2020). The trigger does not need to be proportional to the crash; it only needs to make the most-levered, most-crowded holders start to sell.

3. **The cascade (forced deleveraging).** This is where the short-vol structure turns a small shock into a large crash. The initial selling raises volatility. Higher volatility trips VaR limits and volatility-targeting rules across the whole crowd *simultaneously*, forcing more selling. That selling moves the price further, raising volatility further, forcing the next tier to sell. The funding currency — yen, franc — spikes as borrowers scramble to buy it back. Liquidity evaporates exactly when everyone needs it. The loop that built the position in months unwinds it in days.

4. **The overshoot and snap-back.** The unwind typically overshoots fundamentals — USD/JPY hit ~142 in August 2024, well below where rate differentials "justified" — because the move is driven by forced position-covering, not by a reassessment of value. Once the weak hands are flushed, the pair often snaps part-way back (USD/JPY recovered toward 147 within days). The carry survivor who sized correctly and was not forced to liquidate at the bottom lives to collect the premium again; the over-levered holder who got margin-called at the low has a permanent loss.

The pattern is so reliable that the macro-trading series catalogs three of these unwinds side by side in [carry trade unwinds, 1998, 2008, 2024, when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks). The currencies and triggers differ; the anatomy does not. Short-vol structure plus crowding plus a vol shock equals a cascade, every single time.

#### Worked example: how a small shock becomes a large crash

Trace the cascade arithmetically to see why the loss dwarfs the trigger. Start with a crowded short-yen book sized to a 10% volatility target while realized vol sat at 6%, so the position is `10% ÷ 6% = 1.67×` base notional — call it \$1,670,000 of effective exposure on \$1,000,000 of capital.

> **Stage 1 — the trigger.** A small surprise moves USD/JPY 2% against the book. Loss so far: 2% × \$1,670,000 = **−\$33,400**. Annoying, not fatal.
>
> **Stage 2 — vol resets.** That 2% move pushes realized vol from 6% toward 12%. The volatility-target rule now says the prudent size is `10% ÷ 12% = 0.83×`. The book must be cut in half — from \$1,670,000 to \$833,000 — which means *selling* into a falling market.
>
> **Stage 3 — the crowd cuts together.** Every other volatility-targeted book runs the same arithmetic at the same instant. Their collective selling moves the pair another 6%, pushing vol higher still and forcing the next tier to dump. By the time the dust settles the pair has fallen 12% from the top.
>
> **Total loss on the original \$1,670,000:** 12% × \$1,670,000 = **−\$200,400**, a 20% hit to capital — from a trigger that, on its own, was a 2% wobble.

The intuition: the trigger does not need to be big, because the short-vol structure *manufactures* the rest of the move. The volatility-target rule that sized the book up in the calm is the same rule that forces the synchronized selling in the shock — it is an accelerant in both directions. A 2% nudge becomes a 12% crash because thousands of identical risk rules fire at once.

## Common misconceptions

**"Carry is a yield trade — pick the highest interest rate and collect."** No. Carry is a *volatility* trade wearing a yield costume. The headline rate is the numerator; the volatility (and the skew) is what determines whether the trade is good. A 45% lira yield at 22% vol and a fat devaluation tail can be a worse risk-adjusted bet than a 4.5% dollar-yen gap at 9% vol. If you size by yield instead of by carry-to-vol, you systematically over-allocate to the most crash-prone currencies — the exact opposite of what you want. The number that matters is the ratio, not the rate.

**"Low volatility means the trade is safe right now."** This is the single most dangerous misconception in carry, and it is exactly backwards. Low volatility is when leverage is highest, the position is most crowded, and the eventual unwind will be largest. The 2007 carry world had record-low volatility right before 2008; the 2024 yen-carry world had eerily low FX vol right before August. Calm is not safety; calm is the build-up phase. You should grow *more* cautious as volatility compresses, not less, because the carry-to-vol ratio that looks so attractive is attractive precisely because the denominator has fallen to a level it cannot stay at.

**"A high Sharpe ratio proves carry is low-risk."** The Sharpe ratio measures return per unit of *volatility*, and it is computed from past returns. For a short-vol strategy, past returns in a calm period have low volatility *by construction* — that's the whole point of the strategy — so the Sharpe looks spectacular right up until the crash. A high Sharpe in a strategy with strongly negative skew is not evidence of safety; it is evidence that the bad outcome simply hasn't happened *yet* in your sample. Sharpe is blind to skew and to tail risk. Two strategies can have identical Sharpes while one is a coin flip and the other is selling earthquake insurance.

**"Diversifying across many carry pairs removes the risk."** Diversification helps with *idiosyncratic* risk — one country's specific story. It does almost nothing against the *systematic* risk that all carry pairs share: global risk appetite. In calm markets the pairs look beautifully uncorrelated; in a crisis their correlation goes to one, and the diversified book crashes as hard as the concentrated one, just with more line items. You are selling insurance against the same catastrophe in ten different policies. When it strikes, all ten pay out.

**"If I just use a stop-loss, I'll be fine."** Stops sound like protection but fail in exactly the scenario that hurts carry. A carry crash is a *gap* — USD/JPY did not trade smoothly from 162 down to 142, it leapt across price levels in thin overnight liquidity. A stop-loss at, say, 155 doesn't fill at 155 in a gap; it fills at whatever the next available price is, which might be 148. And because everyone's stops cluster at the same obvious levels, the stops themselves *fuel* the cascade — triggered stops become forced sellers, which deepens the gap. Stops manage orderly trends, not short-vol cliffs.

## How it shows up in real markets

The theory is clean; the history is bloody. Two episodes show the low-vol build-up and the volatility-spike unwind with perfect clarity.

### 2007: the calmest market right before the storm

The years 2004 through mid-2007 were a golden age for carry. Global volatility was extraordinarily low — the VIX spent long stretches in the low teens and even single digits, a level of calm that, in hindsight, was the warning. Interest rate gaps were wide and stable: the Fed had hiked to about 5.25% while the Bank of Japan sat near zero, so borrowing yen to hold dollars (or even higher-yielding currencies like the Australian and New Zealand dollars, the Icelandic króna, the South African rand) paid handsomely. The carry trade was so popular that Japanese retail investors had a nickname — Mrs. Watanabe — for the housewives piling household savings into high-yield foreign currencies.

Run the carry-to-vol logic on that period and you can see the trap building. With volatility crushed to record lows, the carry-to-vol ratio looked phenomenal even though the rate gaps were ordinary. Volatility-targeting and VaR-based risk systems across the industry permitted ever-larger positions. Leverage accreted. The trade got crowded to a degree few appreciated, because so much of it lived in retail margin accounts and lightly-regulated funds. Every box on the low-vol-invites-leverage loop was being ticked, quietly, for three straight years.

Then 2008 arrived. As the financial crisis unfolded, volatility didn't rise — it *detonated*, the VIX spiking toward 80.9. The carry trade went into a historic unwind. Yen-funded positions blew up as the yen rocketed: USD/JPY, which had drifted around 120 in 2007, collapsed toward 90 by late 2008 and below 80 by 2011. High-yield carry darlings imploded — the Icelandic króna's collapse helped take the entire Icelandic banking system with it. The stylized carry index in our cover chart fell roughly 28% in that unwind (the red 2008 dot). Years of Mrs. Watanabe's patient premium were erased in months. The premium had been real; it had simply been an insurance premium, and the claim came due.

### 2024: low FX vol, then the August unwind

History rhymed almost exactly seventeen years later. Through 2023 and the first half of 2024, the dollar-yen carry trade was once again the most beloved, most crowded trade in macro. The setup was textbook: the Fed held its policy rate around 5.25–5.5% while the Bank of Japan kept rates pinned near zero, so the rate gap on USD/JPY was the widest in decades. And FX volatility was strikingly low — implied vols on the major pairs were compressed, realized volatility was minimal, and USD/JPY ground higher in a calm, one-directional climb from 130 toward 162. Every condition of the build-up phase was met: wide gap, low vol, high carry-to-vol ratio, accreting leverage, a deeply crowded short-yen position spread across hedge funds, CTAs, and Japanese retail.

![USD JPY fell about 12 percent in five weeks during the August 2024 yen carry unwind](/imgs/blogs/carry-and-volatility-the-relationship-that-runs-the-trade-8.png)

The trigger came at the end of July 2024. The Bank of Japan hiked its policy rate from roughly 0.10% to 0.25% — a tiny absolute move — and signaled more to come, just as a soft U.S. jobs report pulled forward expectations of Fed cuts. The rate gap that powered the trade was now expected to *narrow* from both ends. That was enough. As the figure shows, USD/JPY fell from its July 3 peak of 161.9 in a relentless slide: 157.8, 153.9, 150.0, 146.5, and then the August 5 crash low of 141.7 — about a **12% move in roughly five weeks, with the worst of it in three days.** The VIX spiked intraday to 65.7. The Nikkei had its worst day since 1987.

This was the cascade in pure form. The first wave of yen-buying raised volatility; higher volatility tripped VaR and volatility-targeting rules across the crowded crowd; forced sellers begat lower prices begat higher volatility begat more forced sellers. The funding currency — the yen — rocketed as everyone scrambled to buy back the yen they had borrowed. Liquidity vanished. And then, the overshoot and snap-back: USD/JPY recovered toward 147 within a week as the forced selling exhausted itself, leaving the over-levered with permanent losses and the prudent with a survivable scare. The whole event lasted barely a fortnight. The premium that the trade had paid for two years was, for the most-levered holders, gone in three days.

### 1998: the original carry cascade

For completeness, rewind to the prototype. In the summer of 1998, the dollar-yen carry trade was, once again, the crowded macro consensus — borrow yen near zero, hold higher-yielding assets, collect the gap. Volatility had been low; leverage was extreme, nowhere more than at Long-Term Capital Management, the famous fund whose models told it that recent low volatility justified enormous size. Then Russia defaulted on its debt in August, risk appetite collapsed worldwide, and the carry trade detonated. In a span of three days in early October 1998, USD/JPY fell from roughly 136 to 112 — a yen rally of nearly 18% in 72 hours — as levered players scrambled to buy back the yen they had borrowed. The VIX of the day spiked toward 45. LTCM, whose risk models had been calibrated on the preceding calm, was forced into a Fed-orchestrated rescue. Every element of the anatomy was present a full decade before 2008 and twenty-six years before 2024: low-vol build-up, a sovereign-shock trigger, a forced-deleveraging cascade, a funding-currency spike. The episode is the founding case study of "stable until it isn't," and the game-theory and finance tracks treat the speculative dynamics directly — the carry-and-vol lesson is simply that the calmest part of the cycle was the most dangerous.

The lesson the three episodes teach in unison is the thesis of this post: **the danger was visible all along, but only if you were reading volatility correctly.** The low, falling, comfortable volatility of 1998, 2007, and early 2024 was not the absence of risk. It was the build-up. The crowd read calm as safety and sized up; the survivors read calm as the loaded phase of a short-vol trade and sized *down*.

## The takeaway: read low volatility as a loaded gun

So what does all of this change about how you read a currency, and a market?

First, when you look at a carry trade, **stop reading the yield and start reading the ratio.** The interest rate gap is just the numerator. Divide it by the pair's volatility and you have the carry-to-vol ratio — the real measure of how much premium you collect per unit of risk. A fat yield on a violent currency can be a worse trade than a thin yield on a calm one. The headline rate is the bait; the ratio is the trade.

Second, **invert your instinct about calm markets.** A long stretch of low, falling volatility should make you *more* wary of carry, not less. Low vol is the soft phase of the insurance cycle: it inflates the carry-to-vol ratio, expands risk limits, invites leverage, and crowds the trade into a position where everyone shares one exit. The calmest markets are the most loaded. When you hear "FX volatility is at multi-year lows," do not hear "safe" — hear "the build-up phase of the next unwind is well underway."

Third, **respect the short-vol shape.** Carry is selling insurance: a steady premium and a rare, catastrophic claim, with returns that are negatively skewed and a Sharpe ratio that flatters the strategy right up to the crash. That shape means the entire game is tail management. The edge is not in collecting the premium — anyone can do that — it is in sizing the position so the inevitable claim is survivable, and in not being the forced seller at the bottom of the cascade. Pick up the pennies, but always know exactly how far away the steamroller is, and never stand so close that one step kills you.

A practical corollary follows from everything above: **watch the denominator, not just the level.** It is not enough to know that volatility is low; what matters is the *trajectory* and the *gap*. Volatility that is low and still falling is the most fertile build-up — the carry-to-vol ratio is climbing, leverage is accreting, the crowd is thickening. And when implied volatility starts to pull *above* realized volatility while the pair is still calm, the options market is quietly pricing a fragility that the soothing realized number hides. Those two reads — the trajectory of vol and the spread between implied and realized — are how you see the loaded phase before the gun goes off, using only prices rather than waiting for the positioning data that always arrives too late. The carry trader who survives decades is not the one with the best entry; it is the one who cuts size while everyone else is sizing up, because they are reading the denominator as a warning instead of an invitation.

And finally, connect it back to the spine of this whole series. An exchange rate is the relative price of two monies, moved by the gap between two countries' interest rates plus the flow of money across borders. The carry trade is the purest bet on that rate gap — and the carry *crash* is what happens when the flow reverses all at once. Volatility is the variable that tells you which regime you are in. When it is low and falling, the flow is one-directional and the gap is being harvested. When it spikes, the flow snaps into reverse, and the pair that paid you for years takes it all back in a weekend. The relationship between carry and volatility is not a footnote to the carry trade. It *is* the trade.

## Further reading & cross-links

Within this series (Track C, the carry trade):

- [The carry trade: getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency) — the full anatomy of the trade we have been risk-profiling here: funding leg, target leg, the roll, and how leverage turns a thin gap into a fat return.
- [Carry crashes: picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller) — the unwind mechanics in depth: the negative skew, the cascade, and how to survive the steamroller.
- [Risk reversals and the shape of fear in FX](/blog/trading/forex/risk-reversals-and-the-shape-of-fear-in-fx) — how the options market prices the *asymmetry* of a carry pair's crash risk, the skew the carry-to-vol ratio alone can't see.
- [Interest rate differentials: the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx) — the rate gap that is the numerator of every carry trade.
- [Uncovered interest parity and why it fails: the forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle) — why the carry premium exists at all: parity says it shouldn't, and the gap is the insurance fee.

Going deeper, in adjacent areas:

- [The volatility smile and skew: why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — the options-market view of the same crash asymmetry that makes carry short-vol; the skew is the price of the tail.
- [Carry trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — three unwinds side by side, showing the identical anatomy across very different triggers.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why diversifying across carry pairs fails when it matters most.
