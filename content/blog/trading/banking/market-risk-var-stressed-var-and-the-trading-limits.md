---
title: "Market Risk: VaR, Stressed VaR, and the Trading Limits That Hold a Bank Together"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank measures the risk in its trading book with VaR, stressed VaR, and expected shortfall, why VaR systematically understates the tail, and how the cascade of trading limits is supposed to stop a single desk from sinking the firm."
tags: ["banking", "market-risk", "value-at-risk", "stressed-var", "expected-shortfall", "frtb", "back-testing", "trading-limits", "risk-management", "tail-risk"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A bank's trading book makes money by holding positions that move with the market, and *market risk* is the chance those positions lose money before the bank can close them. The industry's standard yardstick, Value-at-Risk (VaR), answers one narrow question — "on a normal bad day, how much could we lose?" — and it answers it well right up until the day it matters most.
>
> - **VaR is a quantile, not a ceiling.** A 99% 1-day VaR of \$23 million says: on 99 days out of 100 you lose less than \$23m. It says *nothing* about how bad the other 1 day gets. That silence is where banks die.
> - **Three numbers, three questions.** VaR measures a normal bad day; **stressed VaR** re-runs the same book through a past crisis (the loss roughly triples); **expected shortfall** averages the losses *beyond* VaR, so it actually looks into the tail.
> - **VaR understates fat tails by design.** It assumes a bell-curve world where big moves are astronomically rare. Real markets have fat tails: the "once in a thousand years" day shows up every few years.
> - **The limit framework is the real control.** A firm-wide VaR limit cascades down to desk limits, Greek limits, and stop-losses, with escalation triggers — many small leashes, not one big rule. The number to remember: a well-calibrated 99% VaR should be breached about **2–3 times per 250 trading days**; more than that, and the model is broken.

In August 2007, the bank Goldman Sachs told investors that its funds had just lived through "25-standard-deviation moves, several days in a row." Take that sentence literally and it is absurd. A 25-sigma event, under the bell curve that risk models assume, is supposed to happen roughly once in a number of years with more zeros than there are atoms in the observable universe. To see *several in a row* is not bad luck. It is a confession that the model of risk the bank was using had almost nothing to do with the world the bank actually traded in.

That gap — between the tidy number a risk model prints every evening and the messy, fat-tailed reality of markets — is the subject of this post. It is the single most important thing to understand about how a bank measures the danger in its trading desks, because the number that looks most precise is precisely the one that lies to you on the worst day.

We are going to build the whole apparatus from the ground up: what market risk *is*, what Value-at-Risk actually measures (and what it pointedly does not), how stressed VaR and expected shortfall try to patch the holes, how regulators check the number with back-testing, what the new FRTB rulebook changes, and how the lattice of trading limits is supposed to keep one rogue desk from taking down the firm. The diagram below is the mental model for the whole piece.

![Loss distribution with the VaR cut line and the expected shortfall tail beyond it](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-1.png)

That diagram is the entire argument in one frame. VaR is the dashed line — a single point in the distribution of possible daily outcomes. Everything to the left of it, the shaded red tail, is the part VaR refuses to talk about. The bank's survival is decided in that red zone, and VaR's defining feature is that it stops measuring exactly where the danger begins.

A note before we start: this is an explanation of how banks measure and manage risk, not investment advice. Nothing here is a recommendation to buy, sell, or hold anything.

## Foundations: market risk, VaR, and the vocabulary of the trading floor

Let's define every term from zero, because the jargon in this corner of banking is dense and most of it is never explained to outsiders.

### What "market risk" actually means

A bank has two big buckets of assets. The *banking book* holds things the bank intends to keep — loans it made, bonds it bought to hold to maturity. The *trading book* holds things the bank intends to buy and sell — bonds, currencies, stocks, derivatives — either to make a market for clients or to take a position. (We covered the difference, and the markets division that runs the trading book, in [the trading book post](/blog/trading/banking/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule).)

*Market risk* is the risk that the value of the trading book falls because **market prices move** — interest rates, exchange rates, equity prices, commodity prices, credit spreads, and the volatility of all of those. It is one of the four great risks every bank runs, alongside credit, liquidity, and operational risk (the full taxonomy lives in [the four risks post](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational)). Credit risk is the chance a borrower doesn't pay you back. Market risk is different: nobody has to default for you to lose money. The price simply moves, and you are on the wrong side of it.

Here is the everyday-money version. Suppose you run a small shop that, on the side, holds a pile of foreign banknotes you bought for customers heading abroad. You didn't lend that money to anyone, so there's no credit risk. But if the exchange rate moves overnight, your pile is worth less in the morning than it was last night — and you'll have to sell it back at the new, worse price. That overnight swing, scaled up to billions of dollars of bonds and swaps, is market risk.

### What a basis point and a P&L are

Two units you'll need throughout. A *basis point* (bp) is one hundredth of a percent — 0.01%. Traders quote everything in basis points because the moves they care about are small in percentage terms but large in dollars. And *P&L* — profit and loss — is the daily change in the value of the book: if the desk's positions are worth \$5 million more this evening than last, the day's P&L is +\$5m; if they're worth \$3m less, it's −\$3m. Market-risk measurement is fundamentally the study of the *distribution* of daily P&L: how often it's positive, how often negative, and — crucially — how bad the bad days can get.

### What Value-at-Risk measures

*Value-at-Risk (VaR)* is a single number that summarizes the loss side of that P&L distribution. It has three ingredients, and you must state all three or the number is meaningless:

1. **A confidence level** — usually 95% or 99%. This is the slice of the distribution you're cutting off.
2. **A horizon** — usually 1 day or 10 days. The window over which the loss could occur.
3. **A currency amount** — the answer.

Put together: "Our **99% 1-day VaR is \$23 million**" means *we are 99% confident that we will not lose more than \$23 million over the next trading day.* Flip it around and it's clearer: *on roughly 1 trading day in 100, we expect to lose more than \$23 million.* VaR is the loss you expect to exceed only 1% of the time.

The honest one-sentence definition, the one every risk manager should tattoo somewhere visible: **VaR is the *smallest* loss you'll see on your worst 1% of days — not the *largest*, and not the *average*.** It is the threshold where the bad tail begins, not a measure of how deep the tail goes.

### Confidence and horizon, in plain terms

Why 99% and not 100%? Because 100% VaR is just "the maximum possible loss," which for a leveraged trading book is "everything," and that's useless for daily management. You need a number that reflects normal operating danger, so you pick a high-but-not-total confidence level and accept that you've deliberately ignored the very worst slice.

Why does horizon matter? Because a position you can sell in an hour is less risky than one you're stuck holding for two weeks — the longer you're exposed, the more the price can move against you. Regulators have historically required a **10-day horizon** for the capital calculation, on the theory that in a crisis it might take ten days to unwind a position, while traders watch a **1-day** number for daily management.

### Stressed VaR, expected shortfall, back-testing — a first pass

Three more terms, each a response to a specific failure of plain VaR. We'll go deep on each later, but get the one-line versions in your head now:

- **Stressed VaR (sVaR)** — the same VaR calculation, but using market data from a *historical crisis window* (typically 2007–2009) instead of recent calm data. Born after 2008, when banks discovered their VaR was tiny because the recent past had been quiet.
- **Expected shortfall (ES)**, also called Conditional VaR or CVaR — the *average* loss on the days when you breach VaR. It looks *into* the tail that VaR ignores. (There is a deep treatment in [the expected-shortfall post](/blog/trading/risk-management/cvar-expected-shortfall-and-asking-how-bad-is-bad).)
- **Back-testing** — checking, after the fact, how often actual daily losses exceeded the VaR you predicted. It's the report card that tells you whether the model is honest.

The reason a bank computes three different numbers — not one — is that each answers a genuinely different question about the same trading book, and the questions get progressively harder. VaR asks about a normal bad day. Stressed VaR asks about a repeat of a crisis you've already lived through. Expected shortfall asks how deep the loss goes once you're already past the VaR line. The matrix below lays them side by side; keep it in mind as we build each one up in detail.

![Comparison matrix of VaR stressed VaR and expected shortfall](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-2.png)

The progression is deliberate. Each measure was added to the toolkit *because* the previous one failed in a specific, documented way — ordinary VaR went quiet before 2008, so stressed VaR was bolted on; VaR ignored the depth of the tail, so expected shortfall was added to look into it. The history of market-risk measurement is a history of patches, each one a scar from a past disaster. Understanding *why* each measure exists is more useful than memorizing its formula, because it tells you which failure each one is — and isn't — protecting you against.

With the vocabulary in place, let's actually compute a VaR.

## Computing VaR: from one number to the whole distribution

The cleanest way to see VaR is to build it the simplest way first — the *parametric* method — then add realism.

### The parametric (variance-covariance) method

Parametric VaR makes one big assumption: that daily P&L follows a *normal distribution* — the symmetric bell curve. Under that assumption you only need two numbers to describe the entire distribution of outcomes: the mean (call it zero for a single day; trading desks don't expect to make or lose money on average in a day) and the *standard deviation*, written σ (sigma), which measures the typical size of a daily move. The VaR is then just a multiple of sigma:

$$\text{VaR}_\alpha = z_\alpha \times \sigma$$

where $z_\alpha$ is the number of standard deviations that cuts off the bottom $\alpha$ tail of a normal curve. The two you'll use constantly: $z_{95\%} = 1.645$ and $z_{99\%} = 2.326$.

#### Worked example: a 99% 1-day VaR from sigma

Suppose a desk's daily P&L has a standard deviation of \$10 million. That is, on a typical day the book swings about \$10m in either direction. What is the 99% 1-day VaR?

You multiply sigma by the 99% z-score:

$$\text{VaR}_{99\%} = 2.326 \times \$10\text{m} = \$23.26\text{m} \approx \$23\text{m}.$$

So the desk reports a **99% 1-day VaR of \$23 million**. The intuition: a one-percent-tail loss sits about 2.3 standard deviations out, so if a typical day is a \$10m swing, a once-in-a-hundred-days bad day is a \$23m loss — and that's the line the bank watches.

Notice how little work that took. One number (sigma), one constant (2.326), one multiplication. That's the seductive simplicity of parametric VaR — and also its fatal flaw, because the whole thing rests on the assumption that the bell curve is the right shape, and it isn't. We'll come back to that.

### The historical method

The *historical simulation* method throws out the bell-curve assumption entirely. Instead of a formula, you take the last 1–2 years of *actual* daily market moves (say 250 or 500 trading days), apply each of those historical moves to *today's* positions, and get 250–500 hypothetical P&L outcomes. Sort them from worst to best. The 99% 1-day VaR is simply the loss at the 1st percentile — for 250 days, that's roughly the 3rd-worst outcome (because 1% of 250 ≈ 2.5).

The appeal is that you don't assume any shape. If the last year happened to contain fat-tailed moves, they're right there in your data and they push up the VaR. The weakness is the mirror image: if the last year was calm, your VaR is calm, and you are blind to any kind of shock that simply didn't happen in your window. Historical VaR remembers only what it has seen.

#### Worked example: historical VaR from a sorted year of P&L

Imagine you've applied each of the last 250 days' market moves to today's book and produced 250 hypothetical P&L numbers. You sort them worst-to-best, and the bottom of the list reads: −\$41m, −\$34m, −\$28m, −\$24m, −\$21m, and so on up toward zero and into the gains. The 99% 1-day VaR is the loss at the 1st percentile. With 250 observations, 1% lands at the 2.5th worst, so you take (conventionally) the **3rd-worst** loss: **\$28m.**

Notice this is *larger* than the \$23m the parametric method gave for the same desk. That difference is the fat tail showing up in the data: the historical window contained a few moves bigger than a bell curve would have produced, and historical simulation faithfully reports them while the parametric formula smooths them away. The intuition: when historical VaR and parametric VaR disagree, the historical number is usually the more honest one, because it's reading the real distribution instead of assuming a tame one.

### The Monte Carlo method

The *Monte Carlo* method splits the difference. You build a statistical model of how the market factors move (their volatilities and how they move together), then use a random-number generator to simulate thousands — often tens of thousands — of possible next days. You revalue the book under each simulated day, collect all the P&L outcomes, and read off the 1st percentile. Its great strength is handling complex, *non-linear* positions like options, whose value doesn't move in a straight line with the underlying price — the other two methods struggle there. Its weakness is that it's only as good as the model you feed it, and it's computationally heavy. The trade-offs across all three are laid out below.

![Comparison matrix of historical parametric and Monte Carlo VaR methods](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-8.png)

Most large banks run more than one of these in parallel. A common setup: historical simulation for the official regulatory number (regulators trust the no-assumptions approach), parametric for quick intraday estimates, and Monte Carlo for the options-heavy desks. When three methods disagree sharply, *that disagreement is itself information* — it usually means the book has something unusual in it that one method captures and another misses.

### From a daily number to a capital charge

It's worth being concrete about *why* a bank computes VaR at all, because the number isn't just an internal speedometer — it directly drives how much shareholder capital the bank must lock up against its trading book. Regulators take the bank's VaR (and stressed VaR) and apply a *multiplier* of at least 3 to arrive at the market-risk capital requirement. That capital is equity that can't be lent out or paid as dividends; it sits there purely to absorb trading losses. So a higher VaR is expensive in a very direct sense — it ties up more equity — which creates a permanent, structural incentive for desks to make their reported VaR look *small.* Hold that tension in mind: the people computing the risk number also pay a price for it being large. That conflict is exactly why the measurement and the limit-setting must be done by an *independent* risk function, and why regulators back-test and floor the result rather than trusting the bank's own arithmetic. We'll see what happens when that independence breaks down in the London Whale case at the end.

This connects the trading book straight back to the series' spine. A bank runs on a thin equity cushion, and every dollar of that cushion is scarce. Market-risk capital is the slice of the cushion reserved for the trading floor's bad days. VaR is the dial that decides how big that slice has to be — which is why getting the dial right, and keeping it honest, is not a back-office detail but a question of how much of the bank's lifeblood the markets division is allowed to put at risk.

## Scaling VaR across horizons: the square-root-of-time rule

A desk computes a 1-day VaR but the regulator wants a 10-day number. How do you get from one to the other? The standard shortcut is the **square-root-of-time rule**: multiply the 1-day VaR by the square root of the number of days.

$$\text{VaR}_{T\text{-day}} = \text{VaR}_{1\text{-day}} \times \sqrt{T}$$

Why a square root and not just multiplying by $T$? Because risk, measured as standard deviation, doesn't accumulate linearly over time — it accumulates with the square root of time. If daily moves are independent and random, two days of risk isn't twice one day's risk; it's $\sqrt{2} \approx 1.41$ times. The randomness partly cancels out across days. Naively scaling by $T$ would massively *overstate* the multi-day risk.

#### Worked example: scaling a 1-day VaR to 10 days

Take our \$23m one-day 99% VaR. The 10-day VaR under the square-root rule is:

$$\text{VaR}_{10\text{-day}} = \$23\text{m} \times \sqrt{10} = \$23\text{m} \times 3.162 = \$72.7\text{m} \approx \$73\text{m}.$$

Compare that to the *wrong* linear scaling, which would give \$23m × 10 = \$230m — more than three times too big. The square-root rule says ten days of risk is only about 3.16 times one day's risk, not ten times. The intuition: losing days and winning days partly offset over a longer window, so risk grows slower than time.

![VaR scaling with the holding period showing square-root-of-time versus linear](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-3.png)

There's a catch worth flagging, because it's exactly the kind of thing that bites in a crisis. The square-root rule assumes daily moves are *independent* — that today's move tells you nothing about tomorrow's. In calm markets that's roughly true. In a crisis it breaks down badly: bad days *cluster*, volatility persists, and a sell-off feeds on itself. When moves are positively correlated across days, the true multi-day risk is *larger* than the square-root rule predicts. So even the scaling shortcut understates risk precisely when it matters most. This is a recurring theme: every convenient assumption in VaR fails in the same direction, and in the same conditions.

## Stressed VaR: re-running the book through a crisis

The 2008 financial crisis exposed the deepest structural flaw in ordinary VaR: it is **backward-looking and it has a short memory.** Banks calibrate VaR on the recent past, and the years before 2008 had been unusually calm. So in 2007, just as the most dangerous positions in banking history were being accumulated, VaR models were reporting low risk — because nothing bad had *recently* happened, the math concluded nothing bad *could* happen. The model was most reassuring at the exact moment it should have been screaming.

The regulatory fix, introduced in the "Basel 2.5" reforms of 2011, was **stressed VaR**. The idea is simple: instead of (or in addition to) calibrating VaR on recent data, calibrate it on a fixed *historical period of significant financial stress* — for most banks, a continuous 12-month window from 2007–2009. You take *today's* positions and ask: how much would this exact book have lost, day to day, if the market behaved the way it did during the worst of the crisis? Banks must now hold capital against the *sum* of ordinary VaR and stressed VaR, which roughly doubled the market-risk capital charge across the industry.

#### Worked example: ordinary VaR versus stressed VaR

Take the same desk with its \$23m ordinary 99% 1-day VaR, calibrated on a calm recent year where daily volatility averaged about 1%. Now re-calibrate on the 2008 stress window, where daily volatility for the same risk factors ran closer to 3% and — critically — assets that normally move independently all fell together (correlations went toward 1).

Volatility roughly tripled, so the loss estimate scales up. Even ignoring the correlation effect, sigma going from about 1% to about 3% would roughly triple the VaR:

$$\text{Stressed VaR} \approx \$23\text{m} \times \frac{3\%}{1\%}\text{-ish} \approx \$69\text{m}.$$

So the *same positions* carry a 99% 1-day VaR of \$23m in calm data but a stressed VaR near \$69m on the crisis window. The intuition: stressed VaR strips away the false comfort of a quiet recent past and asks the only question that matters — "what does this book do when the world breaks?"

![Before and after comparison of calm-period VaR versus a 2008-stressed VaR](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-4.png)

Stressed VaR is a genuine improvement, but it's not a cure. It protects you against *a repeat of a crisis you've already seen*. It does nothing for a *new kind* of shock — one whose pattern of correlations and volatilities looks nothing like 2008. The COVID crash of March 2020, the gilt/LDI crisis of late 2022, the regional-bank run of March 2023 — each had its own signature, and a 2008-calibrated stressed VaR captured them only by luck. The general lesson: any risk measure built from history is defending the last war.

## Expected shortfall: looking into the tail VaR ignores

Return to the cover figure. VaR is the dashed line. Everything to the left — the entire red tail — is invisible to VaR. Two books can have *identical* VaR and wildly different danger: one might lose exactly \$23m on its worst days, another might lose \$23m on its 99th-percentile day and \$200m on its 99.9th. VaR can't tell them apart. This is not a small technical quibble; it is *the* flaw, and it's why VaR famously rewards a particular kind of bad behavior — selling far-out-of-the-money options, which produce steady small gains and an occasional catastrophic loss that lives entirely in the tail VaR doesn't measure. You can lower your VaR while *increasing* your true risk.

**Expected shortfall (ES)** is the answer. Instead of asking "where does the bad tail begin?", ES asks "*how deep is the tail once I'm in it?*" Formally, ES at 99% is the *average* of all the losses that exceed the 99% VaR. It is, by construction, always at least as large as VaR, and usually larger.

#### Worked example: expected shortfall beyond VaR

For a normal distribution there's a clean formula. The 99% expected shortfall is:

$$\text{ES}_{99\%} = \sigma \times \frac{\phi(z_{99\%})}{1 - 0.99}$$

where $\phi$ is the height of the bell curve at the 99% cut-off. Plugging in $z_{99\%} = 2.326$, the bell-curve height there is $\phi(2.326) \approx 0.0267$, so:

$$\text{ES}_{99\%} = \$10\text{m} \times \frac{0.0267}{0.01} = \$10\text{m} \times 2.665 = \$26.7\text{m} \approx \$27\text{m}.$$

So while the 99% VaR is \$23m, the 99% expected shortfall is about \$27m. On the days you *do* breach VaR, your *average* loss isn't \$23m — it's \$27m, because the tail keeps going past the cut-off. The intuition: VaR tells you when the trouble starts; ES tells you how bad it gets once you're in trouble — and it's always worse than the VaR line suggests.

That \$27m figure is for a *normal* distribution, which has a thin, well-behaved tail. The real punchline is that for the fat-tailed distributions markets actually follow, ES is *dramatically* larger than VaR, because the real tail is far heavier than the bell curve. ES has another mathematical virtue worth naming: it is *sub-additive*, meaning the ES of a combined portfolio is never larger than the sum of the parts' ES. VaR lacks this property — in pathological cases, combining two books can make VaR *go up*, which is nonsensical for a risk measure and a real headache for risk aggregation. For both reasons — tail sensitivity and sub-additivity — regulators have steadily pushed the industry from VaR toward ES, which is the headline change in FRTB.

## Why VaR understates the tail: the fat-tail problem

We've gestured at "fat tails" several times. Now let's make it concrete, because it is the single most important reason VaR systematically lies, and it's the reason every other measure in this post had to be invented.

### What a fat tail is

A *fat-tailed* (or *leptokurtic*) distribution is one where extreme outcomes are far more likely than a normal bell curve would predict. The bell curve is the comfortable assumption baked into parametric VaR: it says that big moves are *astronomically* rare, and that the further out you go, the more impossibly unlikely an event becomes. A 3-sigma day under the normal model should happen about once a year; a 5-sigma day, once in about 14,000 years; a 7-sigma day, essentially never in the history of civilization.

Real market returns laugh at this. Daily stock-index returns of 5, 7, even 20 standard deviations show up not once a millennium but every few years. October 19, 1987 — "Black Monday" — was roughly a 20-sigma day under the normal assumption, an event the model says should never occur in any conceivable lifetime of the universe. The 2008 crisis served up a string of them. The honest reading is not that markets keep getting freakishly unlucky; it's that *the bell curve is simply the wrong model for the tail.*

![Normal versus fat-tailed distribution showing the same VaR cut but very different tails](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-7.png)

The figure makes the danger visible. Both distributions have the *same* VaR cut-off — the dashed amber line at the same place. If you only look at where the line sits, the two worlds appear equally risky. But look at what's *beyond* the line: the fat-tailed distribution (red) holds far more probability mass deep in the loss region. The two books have the same VaR and radically different real risk, and *VaR cannot tell them apart* because VaR only reports where the line is, never how heavy the tail beyond it is. This one figure is why parametric VaR, which assumes the thin blue curve, is dangerous: it's measuring the wrong distribution's tail.

### Why this is a feature of VaR, not a bug you can fix

You might think: fine, just use a fat-tailed distribution in the parametric formula, or use historical simulation so no distribution is assumed. Both help — but neither closes the gap, for a subtle reason. The deep tail is, by definition, the region where you have almost no data. A 1-in-1000-day event has happened maybe once or twice in your entire usable history, so *any* estimate of how heavy the tail is rests on a handful of observations. You are trying to measure the shape of a curve in exactly the region where you have the fewest points. This is why even the best tail estimate carries enormous uncertainty, and why no amount of statistical cleverness fully solves the problem. The tail is unknowable in the way that matters most.

#### Worked example: how much VaR misses under a fat tail

Suppose a desk's daily P&L truly follows a fat-tailed distribution (say a Student-t with 3 degrees of freedom), but the risk team models it as normal with the same \$10m everyday volatility. The normal model reports a 99% VaR of \$23m, as we computed. But under the true fat-tailed distribution, the actual 99% loss point sits closer to **\$33m**, and the *expected shortfall* — the average loss beyond that point — runs to roughly **\$55m**, not the \$27m the normal model implies.

So the bank thinks its tail risk is \$23–27m. Its real tail risk is \$33–55m — *double* the comfortable number, hidden entirely in the fattness of the tail. The intuition: the error isn't small or random; the normal assumption *systematically* under-counts the worst days, and the fatter the real tail, the bigger the lie. A model that's "approximately right in the middle and badly wrong in the tail" is exactly the wrong kind of wrong for risk management, because the tail is the only part that can kill you.

This is the structural defect that every patch in this post is responding to. Stressed VaR addresses it by forcing crisis-level volatility into the inputs. Expected shortfall addresses it by averaging the tail instead of ignoring it. FRTB addresses it by adopting ES and adding conservatism on top. None of them *solves* it — the deep tail stays unknowable — but each one stops pretending the tail is as thin and tame as the bell curve says.

## FRTB: the regulatory rewrite of market-risk capital

After 2008, the Basel Committee concluded that the entire framework for trading-book capital was broken, and launched a multi-year rebuild called the **Fundamental Review of the Trading Book (FRTB)**. It is the most significant overhaul of market-risk rules in a generation, and it changes four things that matter for everything above. (For how Basel as an institution sets these rules, see [the BIS and Basel post](/blog/trading/finance/bis-and-basel-bank-regulation).)

**First, expected shortfall replaces VaR as the regulatory risk measure.** The capital charge is now based on a 97.5% ES rather than a 99% VaR. (Those two are roughly calibrated to give similar numbers for a *normal* distribution — but ES is chosen precisely because it doesn't *stay* similar when the tail is fat.) This is the single biggest conceptual shift: the regulator now officially cares about the depth of the tail, not just where it starts.

**Second, varying *liquidity horizons*.** Plain VaR's 10-day horizon assumed everything can be unwound in two weeks. FRTB assigns different horizons — 10, 20, 40, 60, even 120 days — to different risk factors, depending on how long it would realistically take to exit them in a stressed market. A liquid currency might be 10 days; an exotic credit position, 120. This directly attacks the assumption that a position you *can't actually sell* should be treated as if you could.

**Third, a hard boundary between the banking book and the trading book.** Before FRTB, banks could (and did) shuffle positions across the boundary to land in whichever book carried the lower capital charge — a position with losses might migrate to the banking book to avoid mark-to-market capital. FRTB makes the boundary far more rigid and the reclassification far harder, closing an arbitrage that had real teeth.

**Fourth, a tougher relationship between internal models and the standardized approach.** Banks can still use their own internal models (the "Internal Models Approach"), but only desk-by-desk, only if each desk *passes* its back-tests and a new "P&L attribution" test, and with the standardized approach acting as a *floor* — internal-model capital can't fall below a set percentage of the standardized number. The era of banks using clever internal models to drive capital arbitrarily low is, by design, over. (The broader story of how capital rules force these trade-offs is in [the four-risks overview](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational).)

The standardized approach itself was rebuilt too, and it's worth understanding because most banks will end up using it for most desks. The new *sensitivities-based method* decomposes every position into its sensitivities to a fixed set of risk factors — delta (exposure to the underlying price), vega (exposure to volatility), and curvature (the non-linear bit options add) — then applies regulator-set risk weights to each, and combines them using prescribed correlations. The clever part is that the *correlations are dialed up in a stress scenario and down in others*, and the bank is charged the worst of three correlation regimes. In plain terms: the regulator no longer trusts the bank to estimate how its risks offset each other, so it imposes its own conservative correlation assumptions and charges for the case where diversification fails. It is the LTCM lesson — "diversification evaporates in a crisis" — written directly into the capital formula.

The through-line of FRTB is distrust. Every change assumes the bank's own model is optimistic, the recent past is misleading, and the tail is worse than it looks — and builds in conservatism accordingly. That is the correct posture, learned at enormous cost.

## Back-testing: the model's report card

How do you know if a VaR model is any good? You can't validate it by argument; you validate it by *keeping score*. **Back-testing** compares the VaR you predicted each day against the loss that actually occurred. Every day the actual loss exceeds the predicted VaR is called an **exception** (or a breach). The math is unforgiving and simple: a *well-calibrated* 99% VaR should be exceeded on about 1% of days. Over a standard 250-trading-day year, that's about **2.5 exceptions**. Get many more, and your VaR is too low — you're underestimating risk. Get zero, and your VaR is probably too *high* — you're holding too much capital and leaving money on the table.

![Back-testing chart of daily P&L against the VaR line with exceptions marked](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-5.png)

Regulators formalized this into the **Basel "traffic light" system**, which sorts a bank's exception count over 250 days into three zones and adjusts its capital accordingly:

| Zone | Exceptions in 250 days | What it means | Capital consequence |
|---|---|---|---|
| **Green** | 0–4 | Model is fine | No penalty (multiplier 3.0) |
| **Yellow** | 5–9 | Model is suspect | Rising penalty (multiplier 3.4–3.85) |
| **Red** | 10+ | Model is broken | Maximum penalty (4.0) and likely loss of model approval |

The capital "multiplier" is the punchline: a bank using its own VaR model must hold *at least three times* its VaR as capital, and that multiplier climbs as exceptions pile up. The multiplier exists precisely *because* everyone knows VaR understates the tail — it's a crude, blunt buffer bolted on top of a number the regulators don't fully trust.

#### Worked example: counting back-testing exceptions

A desk runs its 99% 1-day VaR of \$23m for a full year of 250 trading days. At year-end, the risk team counts the days where the actual loss came in worse than \$23m. They find **6 exceptions.**

Was the model okay? The expected count is about 2.5. Six is more than double that — it lands the desk in the **yellow zone (5–9)**. The statistical reading: under a correct 99% model, getting 6 or more exceptions in 250 days happens only a few percent of the time by chance, so this is a real signal, not noise. The regulator bumps the capital multiplier above 3.0, and the model owners must explain each breach. The intuition: back-testing turns "is our risk model honest?" from an opinion into a countable, falsifiable number — and 6 breaches when you expected 2 or 3 means the model is quietly running hot.

One subtlety that trips people up: there are *two* kinds of back-test. A "clean" (or hypothetical) P&L holds the positions fixed and asks what they would have lost on the day's market moves — this tests the *model*. A "dirty" (or actual) P&L includes intraday trading, fees, and new deals — this tests the *desk*. FRTB's P&L attribution test compares the two, and a desk whose clean and dirty P&L diverge too much *loses its right to use its internal model* even if its raw exception count is fine. The regulator is checking not just "is the number right?" but "does the model actually describe the book it claims to?"

## The limit framework: many small leashes, not one big rule

So far we've been measuring risk. Measurement is useless without *control*, and control is where the day-to-day life of a trading floor actually happens. A bank doesn't manage market risk with one giant VaR number; it manages it with a *cascade* of limits that gets more specific as it descends, so that a problem trips a wire long before the firm-wide number is ever in danger.

![Pipeline of the trading limit framework from firm VaR limit to forced de-risking](/imgs/blogs/market-risk-var-stressed-var-and-the-trading-limits-6.png)

Walk down the cascade:

**Firm-wide VaR limit.** At the top, the board sets a total market-risk appetite — say, a \$100m daily VaR limit for the whole trading division. This is the number the board and the regulator care about. It is rarely the binding constraint day-to-day, because it's carved up long before any one desk could threaten it.

**Desk-level VaR limits.** The \$100m is allocated down to individual desks — the rates desk gets \$30m, credit \$25m, FX \$20m, equities \$25m, and so on. Each desk head manages within their slice. This is the limit a trader actually feels.

**Greek limits.** Beneath VaR sit limits on specific *sensitivities*, named after Greek letters. *Delta* is the position's exposure to the underlying price moving. *Vega* is its exposure to volatility changing. *Gamma* is how fast delta itself changes. A desk can be within its VaR limit but carry a dangerous concentration in one Greek — a huge vega position that VaR happens to net against something else. Greek limits catch the concentrations VaR averages away. There are also plain *notional* and *concentration* limits: caps on the total size of any single position or exposure to any single name.

**Stop-loss limits.** These are not about *potential* loss but *realized* loss. A stop-loss says: once a position has actually lost, say, \$5m, it must be cut or hedged — full stop, regardless of how confident the trader is that it'll come back. Stop-losses exist because the most dangerous thing on a trading floor is a trader "averaging down" into a losing position, convinced the market is wrong. The London Whale, which we'll get to, was exactly this failure.

**Escalation and forced de-risking.** Any breach — VaR, Greek, or stop-loss — triggers *escalation*: an automatic alert that goes up the chain to the desk head, then to the independent market-risk function, then if needed to the chief risk officer. The independent risk function is critical: the people who set and police limits do *not* report to the people who run the desks. (This separation is the "second line of defense" in the risk-governance model from [the four-risks post](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational).) Persistent or large breaches force *de-risking*: the position is reduced or hedged until the book is back inside its limits, whether the trader likes it or not.

In daily life, a trader doesn't experience these as abstract numbers but as *limit utilization* — the percentage of each limit currently used. A risk report might show a desk at 70% of its VaR limit, 40% of its delta limit, and 90% of its single-name concentration limit. That last figure is the binding constraint: the trader can't add to that name without a breach, regardless of how much VaR headroom remains. Good desks are run *near* their limits — unused risk capacity is wasted return — but never *through* them, and the entire culture of a trading floor is built around the fact that a breach is a reportable event with consequences, not a number that quietly resets overnight. The limits are also reviewed and reset on a cycle: annual board-level appetite, periodic desk allocations, and *temporary* limit increases that require sign-off from the independent risk function and expire automatically. The moment temporary increases become permanent or routine, the framework has started to rot.

The design philosophy is worth stating plainly: **no single number should ever be the last line of defense.** The firm VaR limit is a backstop you hope never to approach. The real protection is the dense lattice of smaller, more specific limits below it, each of which trips early, loudly, and independently of the trader's own judgment. When this lattice works, no single desk can ever get big enough to threaten the firm. When it's overridden — when limits are raised to accommodate a star trader, when a new model conveniently lowers a desk's reported risk — the lattice fails silently, and you don't find out until the loss is already enormous.

## Common misconceptions

**"VaR is the worst that can happen."** This is the single most dangerous misreading, and it has cost banks billions. A 99% 1-day VaR of \$23m does **not** mean "the most we can lose is \$23m." It means "on our worst 1% of days, we lose *at least* \$23m" — the floor of the bad tail, not the ceiling. The actual worst case is unbounded; VaR deliberately refuses to estimate it. The whole point of the cover figure is that the danger lives in the red tail *beyond* the VaR line, which VaR never measures. Read VaR as "the threshold where things start going wrong," never as "the maximum damage."

**"A lower VaR means a safer book."** Not necessarily — and exploiting this gap is a known way to game the number. Because VaR ignores the tail, you can *lower* your VaR while *increasing* your true risk by taking positions that pay off in normal times and blow up only in the extreme tail (selling deep out-of-the-money options is the classic). VaR goes down, the desk looks safer, the risk has merely been moved into the region VaR can't see. This is exactly why regulators moved to expected shortfall, which *does* see into the tail.

**"VaR was right; we just got unlucky with a 25-sigma event."** When a bank loses far more than its VaR predicted and calls it a freak event, that's almost always a model failure, not bad luck. A true 25-sigma day under the normal model should never occur in the lifetime of the universe; observing one means the *model is wrong*, specifically that it assumed thin (normal) tails when the real tails are fat. The "unlucky" framing protects the model; the honest framing retires it. (The definitive case study of this is [2008 and the failure of VaR](/blog/trading/risk-management/2008-and-the-failure-of-var-when-the-tail-ate-the-model).)

**"More confidence is always safer — let's use 99.9% VaR."** Higher confidence sounds more conservative, but it's a trap. To estimate a 99.9% loss you need data on 1-in-1000-day events, and you simply don't have enough history to estimate that tail reliably — you're now extrapolating into a region where your data is essentially silent. A 99.9% VaR isn't more accurate; it's more *confidently wrong*. This is why the standard is 99% (or FRTB's 97.5% ES): far enough into the tail to be meaningful, not so far that you're inventing the number.

**"VaR and stressed VaR cover us; we don't need scenario analysis."** Both VaR and stressed VaR are *historical* — they're built from data that actually happened. They are structurally incapable of capturing a shock the market has never produced before. That's why banks run separate *scenario analyses* and *reverse stress tests* ("what set of moves would wipe out our capital, and how plausible is it?") alongside the statistical measures. The number that comes from the model and the scenario that comes from imagination are different tools, and a bank needs both. (More on this in [stress testing and scenario analysis](/blog/trading/risk-management/stress-testing-and-scenario-analysis-breaking-the-portfolio-on-purpose).)

## How it shows up in real banks

Theory becomes vivid when you watch it fail. Here are the episodes that wrote the modern rulebook.

### 2008: VaR reports calm into the storm

Through 2006 and into 2007, the major banks' VaR models reported steadily *declining* market risk, even as those same banks loaded up on subprime mortgage exposure. The reason was mechanical and damning: VaR was calibrated on the preceding period, which had been one of the lowest-volatility stretches in market history. Low recent volatility fed directly into low sigma, which fed directly into low VaR. The model concluded the world was safe *because* the world had recently been safe — the textbook backward-looking failure.

When the crisis hit, banks reported daily losses many multiples of their VaR, day after day. Bear Stearns and Lehman were running VaR numbers in the tens of millions while sitting on losses that ran to *billions*. The models hadn't just been a little off; they had been measuring the wrong thing entirely, because they assumed thin tails in a market that was about to deliver the fattest tail in living memory. The direct regulatory response was stressed VaR (2011) and, eventually, the whole FRTB rebuild. The deeper lesson — that a risk model calibrated on calm is most dangerous exactly when it's most reassuring — is now baked into every serious framework.

### The London Whale, 2012: limits raised to fit the trade

In 2012, a trader in JPMorgan's Chief Investment Office — Bruno Iksil, nicknamed "the London Whale" for the size of his positions — built an enormous, complex credit-derivatives book that ultimately lost the bank over **\$6 billion**. The instructive part for this post is *how the risk controls failed.*

The position kept breaching the desk's VaR limit. The response was not to cut the position — it was to *change the VaR model* to a new one that reported a lower number, and to *raise the limits* to accommodate the trade. The new model, it later emerged, contained errors (including a spreadsheet that divided by a sum instead of an average) that systematically understated the risk. Stop-loss discipline failed too: as losses mounted, the desk *added* to the position, convinced the market would revert. Every layer of the limit framework from the previous section was present — and every layer was overridden or gamed. The Whale isn't a story about VaR being wrong; it's a story about what happens when the independent control function loses the argument to the desk. JPMorgan was fined more than \$1 billion, and the episode is now the canonical teaching case for *why* risk limits must be genuinely independent and genuinely binding.

### Long-Term Capital Management, 1998: the model that priced its own collapse out of existence

Before VaR was even standard, the hedge fund LTCM — staffed by Nobel laureates — ran positions whose risk models assumed normal, thin-tailed distributions and historically stable correlations. In August 1998, when Russia defaulted, those assumptions inverted: correlations that the model treated as near-zero shot toward one, every position moved against the fund simultaneously, and a portfolio engineered to be "market-neutral" lost about **\$4.6 billion** in weeks. The fund's own analysis had described the losses it suffered as effectively impossible. The lesson that propagated into bank risk management: in a crisis, *diversification evaporates* — the very correlations your VaR relies on to net positions against each other are the first thing to break. This is precisely why stressed VaR uses a crisis window where correlations are jammed toward one.

### Société Générale, 2008: a hidden book the limits never saw

In January 2008, Société Générale revealed a loss of about **€4.9 billion** (roughly \$7.2 billion) from the unauthorized trading of Jérôme Kerviel, a junior trader on an arbitrage desk. The size is staggering, but the lesson for this post is sharper than "rogue trader." Kerviel's job was to take *small, hedged* positions that should have carried almost no net market risk. Instead he built enormous *directional* bets on European stock-index futures — at their peak, a notional position of roughly €50 billion, larger than the bank's entire market capitalization — and hid the risk by booking *fake offsetting trades* that made the book look balanced to the risk system.

This is the failure mode the limit framework is most vulnerable to: the limits, the VaR, and the back-tests all operate on *the positions the system thinks exist.* Kerviel's reported VaR was tiny because the fictitious hedges netted his real exposure to near zero on paper. Every control was functioning correctly — and every control was looking at a book that wasn't real. The bank's controls did flag dozens of anomalies over two years; each was explained away. The lesson that hardened into modern practice: risk limits are only as good as the *integrity of the position data they run on*, which is why banks now invest as heavily in trade-confirmation and reconciliation (does this trade actually exist with a counterparty?) as in the VaR engine itself. A perfect risk model fed a fake book reports perfect safety right up to the catastrophe.

### Amaranth, 2006: a single desk, a single bet, a \$6.6 billion hole

The hedge fund Amaranth Advisors lost about **\$6.6 billion** in a single week in September 2006 — at the time the largest hedge-fund collapse ever — on a wrong-way bet on natural-gas futures spreads concentrated in one trader, Brian Hunter. The position was so large relative to the market that Amaranth *was* the market in those contracts; when the spread moved against it, there was no one to sell to without crushing the price further. VaR utterly failed here for a reason it always struggles with: it assumes you can exit at the prices the model uses, but a position big enough to move the market on the way out has *liquidity risk* that ordinary VaR doesn't capture. This is precisely the gap FRTB's liquidity horizons try to close — and a reminder that concentration limits, not just VaR limits, are what stop a desk from becoming the whole market in a single contract.

### March 2020 and the dash for cash

When COVID hit in March 2020, banks' VaR numbers *spiked* — correctly, this time — as volatility exploded. But the spike created its own problem: as VaR rose, banks' risk limits automatically tightened, forcing desks to cut positions into a falling, illiquid market, which pushed prices down further and raised VaR again. This *procyclical* feedback loop — risk models forcing selling that worsens the very moves the models are measuring — is a structural weakness of any volatility-based limit system. It's one reason central banks had to step in as buyers of last resort, and one reason FRTB's liquidity horizons try to bake in the reality that you can't always exit when the model assumes you can.

### Knight Capital, 2012: when the loss isn't market risk at all

A useful counter-example to close on. Knight Capital lost about **\$440 million in 45 minutes** when a botched software deployment sent a flood of erroneous orders into the market. No market view was wrong; no VaR model failed. The loss came from a *technology* failure that *created* enormous unintended market positions in minutes. This is the boundary of market-risk measurement: VaR models the risk of the positions you *intend* to hold; they are blind to the risk of suddenly holding positions you never meant to. That gap belongs to *operational risk*, the subject of a different post — and a reminder that the trading-limit framework assumes the positions in the book are the positions you chose, which is not always true.

## The takeaway: read the number, then read what it leaves out

If you take one habit from this post, make it this: **whenever someone shows you a VaR number, immediately ask the three questions it doesn't answer.** *What confidence and horizon?* (A 95% number and a 99% number describe very different risks.) *What's the expected shortfall — how deep is the tail beyond this line?* *And how has this model back-tested — how many exceptions, in which zone?* The VaR alone is a single point on a distribution; those three follow-ups are what turn it from a comforting number into an honest one.

For understanding how a bank lives or dies, market risk fits the series' spine cleanly. A bank is a leveraged, confidence-funded machine running on a thin equity cushion. The trading book is where that leverage is most concentrated and most fast-moving — a desk can build a firm-threatening position in days, sometimes hours, in instruments whose value reprices every second. VaR and its cousins are the instruments on the dashboard that tell management how close to the edge the trading floor is running. The catastrophe in every case study is the same shape: the dashboard read "fine" — because the model assumed thin tails, or the recent past was calm, or the limits had been quietly raised to fit the trade — right up until the thin equity cushion was gone.

So the practical wisdom is not "trust VaR" or "distrust VaR" — it's *know exactly what VaR can and cannot see, and never let the number you can measure crowd out the danger you can't.* The bank that survives is the one that treats its VaR as the *beginning* of the risk conversation: the line where the bad tail starts, never the place it ends. The whole edifice of stressed VaR, expected shortfall, FRTB, back-testing, and the limit cascade exists for one reason — because the most precise number on the risk report is the one that goes quiet exactly when the firm's life is on the line.

## Further reading & cross-links

- **[The trading book: market-making, flow vs prop, and the Volcker rule](/blog/trading/banking/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule)** — what the trading book *is* and the markets business that generates the positions VaR measures.
- **[The four risks every bank runs: credit, market, liquidity, operational](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational)** — where market risk sits in the full taxonomy and the three-lines-of-defense governance that polices the limits.
- **[Value-at-Risk and exactly how VaR lies](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies)** — the deep math of VaR and a fuller treatment of its blind spots, for readers who want the full derivation.
- **[CVaR, expected shortfall, and asking "how bad is bad?"](/blog/trading/risk-management/cvar-expected-shortfall-and-asking-how-bad-is-bad)** — expected shortfall in depth: why it's coherent, sub-additive, and the FRTB measure of choice.
- **[2008 and the failure of VaR: when the tail ate the model](/blog/trading/risk-management/2008-and-the-failure-of-var-when-the-tail-ate-the-model)** — the definitive case study of VaR's structural failure in a crisis.
- **[BIS and Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation)** — how the committee that wrote stressed VaR and FRTB actually sets global capital rules.
