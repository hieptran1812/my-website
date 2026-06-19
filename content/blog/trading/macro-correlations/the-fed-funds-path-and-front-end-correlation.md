---
title: "The Fed Funds Path and the Front-End Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the 2-year yield is the market's forecast of the average Fed funds rate, why the front end correlates most tightly with the dollar and banks while the long end drives equity multiples, and how to read which end of the curve moved."
tags: ["macro", "correlation", "fed-funds", "2-year-yield", "front-end", "dollar", "rate-differential", "yield-curve", "monetary-policy", "banks", "trading"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The 2-year Treasury yield *is* the market's forecast of the average Fed funds rate over two years, so the front end of the curve is policy-anchored; it correlates most tightly with the **dollar** (rate differential, corr +0.40 with US yields) and with **rate-sensitive financials**, while the long end (10Y) drives **equity multiples** and **gold**. When the curve moves, *which end moved* tells you which correlation is live.
>
> - A front-end-led move (the 2Y jumps) is a **Fed-repricing**: the market is adding or removing hikes/cuts. Its partner trades are the dollar and the banks.
> - A long-end-led move (the 10Y jumps while the 2Y sits still) is a **growth or term-premium** story. Its partner trades are equity multiples and gold via real yields.
> - A *hawkish repricing* lifts the 2Y and the dollar together; the same hot print pushes the 2Y +9bp (vs the 10Y +7bp) and the dollar +0.35% — the front end repriced more because it is the purest forecast of the Fed.
> - The number to remember: the 2Y yield leads the Fed funds rate both up and down. In March 2022 the 2Y was already at 2.3% while funds were still 0.50%. The market priced the +525bp of hikes *before* the Fed delivered them.

## The day the front end did the Fed's job for it

On 16 March 2022, the Federal Reserve raised its policy rate for the first time in three years, a single quarter-point move that lifted the target range to 0.25-0.50%. By the standard of what was coming, it was almost nothing. Over the next sixteen months the Fed would hike at a pace not seen since the early 1980s, dragging the upper bound of the target range from 0.50% all the way to 5.50% — a cumulative \$525-basis-point tightening, the fastest hiking cycle in forty years.

Here is the strange part. On the *day* of that first tiny hike, the two-year Treasury yield was already trading at 2.28%. The Fed's overnight rate was 0.50%. The two-year was sitting almost two full percentage points *above* the rate the Fed had just set. The bond market was not waiting for the Fed to hike. It had already priced almost the entire hiking cycle, meeting by meeting, into the front end of the yield curve, months before those meetings happened. By the time the Fed actually arrived at 5.50% in July 2023, the two-year had been there and was already drifting back down, anticipating the cuts that would not begin until September 2024.

This is the central fact of the front end, and almost every beginner gets the causality backwards. They think the Fed sets a rate and the bond market reacts. The truth is closer to the reverse: the bond market continuously forecasts the path of the Fed, and the *front end of the curve is that forecast made into a price*. The two-year yield is not a separate number that happens to track the Fed. It is, to a very good approximation, the market's expected average Fed funds rate over the next two years. Understanding that one identity unlocks an entire correlation structure — why the front end moves with the dollar and the banks, why the long end moves with equity multiples and gold, and why naming *which end* of the curve led a move tells you exactly which cross-asset trade is alive.

The reason this matters for *trading* — not just for understanding — is that the curve is constantly telling you two different stories at its two ends, and they have *different partners*. If you treat "yields went up" as a single fact, you will reach for the wrong cross-asset trade half the time. A rate trader who saw the two-year jump in 2022 and bought the dollar made money. A trader who saw the ten-year jump in late 2023 and bought the dollar got chopped up, because that move had no front-end repricing behind it and so no rate-differential support for the dollar. The discipline this post installs is mechanical: before you act on a yield move, find out which end led, and you will know which correlation you are actually trading. That habit is worth more than any single forecast, because it is right across every regime.

![Before-after diagram contrasting the front end driving the dollar and banks with the long end driving equity multiples and gold](/imgs/blogs/the-fed-funds-path-and-front-end-correlation-1.png)

This post sits in the rates track of the series. If the [bond yield is the master correlation with every asset](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset), this post zooms into the *short* end of that master variable and shows that the curve is really two correlation engines wearing one coat. We will build the Fed funds rate from zero, show how Fed funds futures turn it into a forecastable path, derive why the two-year *is* that path, and then split the cross-asset correlations cleanly into a front-end family (dollar, banks, short duration) and a long-end family (multiples, gold, long-duration growth). By the end you will be able to look at a curve move and say, in one sentence, which correlation just switched on.

## Foundations: the Fed funds rate, futures, and the two-year as a forecast

Before any correlation, we need three building blocks: what the Fed funds rate actually is, how Fed funds futures price its future path, and why the two-year Treasury yield equals that path. Build these from zero and everything else follows.

### What the Fed funds rate is

The **Fed funds rate** is the interest rate at which banks lend reserves to each other overnight. That sounds technical, so here is the everyday picture. Every bank in the United States keeps an account at the Federal Reserve — think of it as the bank's own checking account at the central bank. At the end of each day, some banks have a little extra cash in that account and some are a little short. The ones with extra lend it overnight to the ones who are short, and the interest rate on those overnight loans is the Fed funds rate.

The Fed doesn't *literally* dictate this rate to each bank. Instead, the Federal Open Market Committee (FOMC) sets a **target range** — for example 5.25-5.50% — and then uses its tools (paying interest on reserves, draining or adding cash) to keep the actual overnight rate inside that band. When people say "the Fed hiked rates," they mean the FOMC moved the target range up at one of its eight scheduled meetings a year.

Why does this single overnight rate matter so much? Because it is the *price of the shortest, safest loan in the economy*, and every other interest rate is built on top of it. A one-month loan is roughly the overnight rate compounded for a month, plus a tiny premium. A one-year loan is roughly the *expected average* overnight rate over the year, plus a premium. This is the key idea, and we will lean on it heavily: **a longer rate is, to first approximation, the expected average of the overnight rate over its life.** That is why the Fed, by controlling one overnight number, exerts gravity over the entire structure of interest rates — and through them, as the macro-trading series argues, over [the price of money itself](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

### How Fed funds futures price the path

If a longer rate is the expected average of the overnight rate, then to forecast longer rates we need to forecast the Fed's *path* — what it will do at each meeting. The market does this continuously, and it does it through a specific, tradeable instrument: the **30-day Fed funds futures contract**, traded at the CME.

A Fed funds future is a bet on the *average daily Fed funds rate during a specific calendar month*. The quoting convention is delightfully simple: the price equals `100 minus the expected average rate`. So if the December contract trades at 95.25, the market is pricing an average Fed funds rate of `100 - 95.25 = 4.75%` for December. If the contract trades at 94.75, the implied rate is `100 - 94.75 = 5.25%`. Higher implied rate, lower price.

Because there is a separate contract for each month, the *strip* of these contracts traces out the market's entire expected path of the Fed funds rate, month by month, a year or two into the future. Traders read the strip directly: "the curve has 3.5 cuts priced for next year," "the market expects the first hike in June," and so on. The strip is the rawest, most direct measure of the market's Fed forecast that exists — and it is what every other rate is built from.

![Pipeline showing Fed funds rate priced by futures into a path, averaged into the expected policy rate, which equals the 2-year yield plus a term premium](/imgs/blogs/the-fed-funds-path-and-front-end-correlation-7.png)

#### Worked example: pricing the path from Fed funds futures

Suppose it is early 2023 and you are reading the Fed funds futures strip. The contracts imply the following month-average rates:

```
Mar 2023:  4.75%   (price 95.25)
Jun 2023:  5.10%   (price 94.90)
Sep 2023:  5.30%   (price 94.70)
Dec 2023:  5.25%   (price 94.75)
```

Read it like a story. The strip says the market expects the Fed to keep hiking through the spring and summer, peaking around 5.30% in the third quarter, then *plateauing* and maybe nudging down by December. To turn the strip into the "expected average rate over the next nine months or so," you average the implied rates: `(4.75 + 5.10 + 5.30 + 5.25) / 4 = 5.10%`. That \$5.10% average is the market's best single-number forecast of where the overnight rate will sit, on average, over that window. The intuition: a Fed funds future is a price you can read as a probability-weighted forecast of policy, and the strip of them is the path you build longer rates from.

### Why the two-year yield is the expected average policy rate

Now we can state the identity that anchors this entire post. The **two-year Treasury yield** is, to a very good approximation, the market's expected *average* Fed funds rate over the next two years, plus a small **term premium** (extra compensation for tying your money up for two years and bearing the risk that rates move against you).

In symbols, treating it loosely:

```
2Y yield  ~=  average expected Fed funds rate over 2 years  +  term premium
```

The logic is an arbitrage. If you could earn 5% by rolling overnight loans for two years (because the Fed is expected to average 5%), you would refuse to buy a two-year note yielding 3% — you would just roll overnight and earn more. So buyers push the two-year yield up toward the expected average of the overnight path. Conversely if the Fed is expected to *cut* aggressively, rolling overnight will earn you a *falling* average, so locking in a two-year note at today's higher rate is attractive, and buyers push the two-year yield *down* below today's overnight rate. The two-year yield is continuously dragged toward the average of the expected path. It *is* the path, expressed as a single price.

The term premium is usually small at the two-year tenor — a few tenths of a percent, sometimes negative — because two years is short enough that the risk of being badly wrong about the average is limited. That is why the two-year is treated as an almost-pure expectation of policy. The *ten*-year yield, by contrast, blends the expected path of the *first* couple of years with long-run inflation and growth expectations and a *much larger* term premium, because a lot can happen over a decade. Hold onto that distinction: it is the seam along which the curve splits into two correlation engines.

It is worth being precise about what "the path" contains, because it is the difference between the two-year as a *forecast* and the funds rate as a *fact*. At any moment the two-year embeds the market's probability-weighted average over every plausible future: a path where inflation stays hot and the Fed grinds to 6%, a path where the economy cracks and the Fed slashes to 2%, and everything in between. The two-year yield is the expected value across that whole distribution of paths. This is why it can move *without the Fed doing anything at all*: a hot data point shifts probability mass toward the high-rate paths, the expected average rises, and the two-year rises — even on a day with no FOMC meeting. The Fed funds rate is a single realized number; the two-year is the expectation over all the numbers the Fed *might* realize. That gap between expectation and realization is the entire reason the front end leads.

### The dot plot versus market pricing

There are, in fact, *two* published forecasts of the Fed path, and the difference between them is one of the most-watched signals in macro. The first is the market's — the Fed funds futures strip, which *is* the front end. The second is the Fed's own: four times a year, in its Summary of Economic Projections (SEP), each FOMC participant submits a forecast of where they think the appropriate policy rate will be at the end of each of the next few years. Plotted as a scatter of dots, this is the famous **dot plot**.

When the dot plot and the market's pricing *agree*, the front end is calm — everyone is on the same page about the path. When they *disagree*, the front end is a coiled spring. In late 2023, the market was pricing far more cuts for 2024 than the dot plot showed; the gap was the trade. As each data point arrived, the front end repriced toward whichever forecast the data validated, and the dollar moved with it. The discipline of reading the dot plot against market pricing — and trading the convergence — is the subject of the event-trading post on [the dot plot and the SEP](/blog/trading/event-trading/cpi-the-report-that-moves-the-world); for our purposes the point is that the front end is the *market's* path, and its distance from the *Fed's* path is a measure of how much repricing risk is loaded into the two-year.

![Step chart of the Fed funds target with the 2-year yield overlaid, 2020 to 2026, showing the 2Y leading the funds rate up and down](/imgs/blogs/the-fed-funds-path-and-front-end-correlation-2.png)

The figure above makes the identity visible. The gray step line is the Fed funds target — it only changes at meetings, so it moves in discrete jumps. The blue line is the two-year yield. Notice three things. First, the two-year *leads* the funds rate up: by March 2022 it was at 2.3% while funds were still 0.50%, because it had already priced the hikes the Fed was about to deliver. Second, the two-year *leads* the funds rate down: it peaked at 5.05% in October 2023 and was falling well before the Fed's first cut in September 2024. Third, the funds rate eventually *converges* to where the two-year said it would go. The forecast (blue) leads; the realization (gray) follows. The two-year is the market doing the Fed's arithmetic for it, in real time.

### What "correlation" means in this post

One last foundation, since this is a series about correlation. A **correlation** between two things is a number, between −1 and +1, that summarizes how they move together. +1 means they move in lockstep; −1 means they move in exact opposition; 0 means no linear relationship. For the relationships in this post we also care about a **beta**: the *size* of one variable's move per unit of the other's. "The dollar's correlation with US yields is +0.40" tells you they tend to rise together; "the 2Y repriced +9bp per +0.1pp core-CPI surprise" tells you the size. Sign tells you direction; beta tells you magnitude. The series' deeper point — argued in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — is that both can change with the regime. Keep that in the back of your mind; we will see the front-end correlations strengthen and fade.

## The front-end family: the dollar and the banks

Now the payoff. The front end is policy-anchored, so it correlates with the things that key off *policy*: the dollar and rate-sensitive financials. Let us take each in turn and build the mechanism, not just the number.

### Why the front end drives the dollar: the rate differential

The single tightest, most reliable correlation the front end has is with the **US dollar**. To see why, you have to understand what makes a currency go up. A currency is just a claim on a country's money market — to hold dollars is to be able to earn the dollar interest rate. If the US offers a higher short-term interest rate than Europe, then, all else equal, global money wants to sit in dollars to earn that higher rate. That demand bids the dollar up. The driver is the **rate differential**: the gap between US short rates and foreign short rates.

And what *is* the US short rate, in a forward-looking sense? It is the expected Fed path — the front end. So when the front end reprices higher (the market adds hikes), the US rate differential widens, and the dollar firms. When the front end reprices lower (the market adds cuts), the differential narrows, and the dollar softens. The front end *is* the rate-differential signal. This is the same mechanism the macro-trading series develops in detail in [how monetary policy moves currencies](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy); here we are pinning down the *correlation* it produces.

The data file's dollar correlations make the structure crisp. The dollar (DXY) has a positive correlation of **+0.40 with US 10-year yields** — higher US rates pull the dollar up — and it is *negatively* correlated with almost everything else it touches: gold (−0.55), oil (−0.45), copper (−0.50), emerging-market equities (−0.55), Bitcoin (−0.35). The dollar is "cross-asset gravity": when it rises, it presses down on commodities and risk assets priced against it. But the *source* of a dollar rally, when it is rate-driven, is the front end firming.

![Horizontal bar chart of the dollar's cross-asset correlations, positive with US yields and negative with gold, oil, copper, EM, Bitcoin](/imgs/blogs/the-fed-funds-path-and-front-end-correlation-4.png)

The one green bar — the dollar's *positive* correlation with US yields — is the front-end channel. Every red bar is the gravity that channel exerts on everything else. When you see the dollar surging, the first question is "is the front end driving this?" If the two-year is jumping at the same time, the answer is yes, and you are in a rate-differential regime where commodities and EM are likely under pressure.

There is an important refinement that keeps you from mis-reading the dollar: the dollar actually has *two* drivers, and the front end is only one of them. This is the "dollar smile." On the left of the smile, the dollar rallies in a *risk-off panic* — a flight to the safest, most liquid asset on earth, regardless of rates. On the right of the smile, the dollar rallies on *US growth and a firming front end* — the rate-differential channel this post is about. In the *middle* — calm, synchronized global growth — the dollar tends to *sag* as money flows out to riskier, higher-returning corners of the world. The front-end correlation lives on the *right* side of the smile. So when you see the dollar rising, your first diagnostic is *which side of the smile*: if the two-year is rising with it, you are on the right side (a rate story, and the rate-differential trades apply); if equities are crashing and the two-year is *falling*, you are on the left side (a haven story, and the rate-differential logic does not apply). Confusing the two is the most common way the front-end-to-dollar correlation appears to "break" — it didn't break, you were on the wrong side of the smile.

#### Worked example: the rate differential and the carry it pays

Suppose the US two-year yields 5.0% and the German two-year yields 3.0%. The rate differential is 2.0 percentage points in favor of the dollar. A global investor who borrows euros at ~3% and parks the proceeds in US two-year notes at ~5% earns the 2pp gap as **carry** — the income from holding the higher-yielding currency. On a \$1,000,000 position, that is `\$1,000,000 × 2.0% = \$20,000` per year, before any move in the exchange rate.

That carry is exactly what bids the dollar up: investors chase the differential, buying dollars to capture it. Now suppose hot US data lifts the US two-year to 5.5% while the German two-year stays at 3.0%. The differential widens to 2.5pp, the annual carry on the same position rises to `\$1,000,000 × 2.5% = \$25,000`, and the dollar firms further as more money chases the wider gap. The intuition: the front end *is* the carry signal, and a widening rate differential is both a higher income stream and an upward force on the dollar — which is why a hawkish repricing of the Fed path and a stronger dollar are the same event seen from two angles.

### Why the front end drives the banks: net interest margin

The second member of the front-end family is **rate-sensitive financials** — banks above all. A bank's core business is borrowing short and lending long: it takes in deposits (which it pays a short-term rate on, or sometimes near zero) and makes loans (which earn a longer-term rate). The gap between what it earns on assets and pays on liabilities is its **net interest margin** (NIM), and NIM is the engine of bank profits.

When the Fed hikes — when the front end rises — banks can immediately charge more on floating-rate loans and on new lending, while deposit rates lag (depositors are slow to demand more, and checking accounts pay little regardless). So a rising front end *widens* NIM in the early innings and lifts bank earnings. That is why bank stocks are positively correlated with a rising front end, especially at the start of a hiking cycle. The relationship is not perfectly clean — if the Fed hikes the economy into a recession, credit losses can swamp the NIM benefit — but the first-order correlation is real: front end up, bank margins up.

There is a second-order subtlety worth flagging, because 2023 taught it the hard way. A bank doesn't just lend; it also *holds* bonds. When the front end (and the whole curve) screams higher, the bonds a bank already owns lose value — that is duration risk, and we will price it below. A bank that funded long-dated bonds with flighty deposits can take a mark-to-market hit large enough to threaten solvency, which is precisely what felled Silicon Valley Bank in March 2023. So the front end's effect on banks is two-sided: it widens the margin on *new* business while marking down the *old* bonds on the balance sheet. The healthy, deposit-rich bank loves a rising front end; the duration-mismatched bank can be killed by it.

There is also a *funding-cost* dimension that the simple "NIM widens" story glosses over, and it is why the bank correlation fades late in a hiking cycle. Early in a cycle, deposit rates lag badly — depositors are slow and sticky, so the bank captures the full benefit of charging more on loans while paying little more on deposits. But as rates stay high, two things happen. Depositors wake up and demand higher rates (or move cash to money-market funds, draining the cheap deposit base), and the *deposit beta* — the fraction of each Fed hike the bank must pass through to depositors — rises from near zero toward 50% or more. So the NIM tailwind is strongest at the *start* of a hiking cycle and erodes as it matures. This is why the bank-to-front-end correlation is itself regime-dependent: positive and strong in the early innings of a tightening, weaker or even negative late, when deposit competition and recession-risk credit losses start to bite. The correlation has a *shape over the cycle*, not a constant value.

There is one more rate-sensitive corner of the front-end family worth naming: **money-market funds and the short-duration cash complex**. When the front end rises, the yield on Treasury bills, repo, and money funds rises with it almost one-for-one, because those instruments roll over in days or weeks and reprice immediately to the new policy rate. This is the most *direct* front-end correlation of all — there is barely any forecast or term-premium component, just the current overnight rate plus a sliver. It is why, in a hiking cycle, cash stops being trash: in 2023, with the front end above 5%, a money-market fund paid roughly \$5,000 a year on \$100,000 of idle cash, for almost no risk, simply because it sat at the very front of the curve. The flip side is that when the Fed *cuts*, that yield evaporates just as fast — which is why investors scramble to lock in longer maturities (extend duration) right before a cutting cycle, to keep earning the old high rate after the front end has fallen.

#### Worked example: a short-duration trade on a front-end move

Say you expect a hawkish repricing — a hot CPI print that makes the market add hikes to the Fed path. You want to express it cleanly through the front-end family. One trade: buy a basket of large, deposit-rich banks and fund it by shorting a long-duration growth name (whose multiple the *long* end will compress — more on that below). Suppose your \$100,000 long-banks leg rises 2.5% on the repricing as NIM expectations widen, gaining `\$100,000 × 2.5% = \$2,500`, while your \$100,000 short-growth leg falls 3.0% (long-duration multiple compression), gaining `\$100,000 × 3.0% = \$3,000` on the short. The paired trade nets `\$2,500 + \$3,000 = \$5,500` on \$100,000 of gross exposure per side, and it is roughly market-neutral — it is a pure bet that the *front end* leads this move. The intuition: when you believe a move is front-end-led, the cleanest expression is long the rate-sensitive winners (banks) against the long-duration losers, because that isolates the policy-repricing correlation from the overall direction of the market.

## The long-end family: equity multiples and gold

Now the other side of the seam. The long end (the 10-year) is *not* a clean forecast of the next few Fed meetings. It blends the path with long-run inflation, long-run growth, government bond supply, and a large term premium. So it correlates with a different family: equity *multiples* and gold.

### Why the long end drives equity multiples

A stock is worth the present value of its future earnings. To compute a present value you **discount** future cash flows by an interest rate — and the rate you discount by is anchored to the long end, the 10-year yield, because that is the closest proxy for the "risk-free rate over the long horizon" that valuation models use. When the 10-year rises, the discount rate rises, and the present value of *far-future* earnings falls hardest. That is why a rising long end compresses **equity multiples** (the price you pay per dollar of earnings), and why it hits long-duration growth stocks — companies whose value is mostly in distant, fast-growing cash flows — far harder than it hits value stocks with near-term earnings.

This is the discount-rate channel, and it is genuinely different from the front-end channel. A move in the *front* end repriced the dollar and the banks via the policy/differential mechanism. A move in the *long* end reprices the entire equity market's valuation via the discount-rate mechanism. The 2022 bear market in growth stocks was overwhelmingly a *long-end* story: the 10-year went from 1.5% to over 4%, the discount rate roughly tripled, and the Nasdaq fell 32.5% as multiples compressed. The same year, the front end's surge was busy driving the dollar to a twenty-year high — two correlations, two ends of the curve, one tightening cycle.

### Why the long end drives gold

Gold's master correlation is not with inflation directly; it is with the **real** (inflation-adjusted) yield, which lives on the long end. Gold pays no interest. So the *opportunity cost* of holding it is whatever you could have earned, safely, after inflation — the 10-year real yield. When real yields rise, holding gold gets more expensive relative to holding an inflation-protected bond, and gold tends to fall. When real yields fall, gold rallies. That correlation has historically been strongly negative (around −0.8 over 2007-2021). The full story — including how it *broke* in 2022-24 as central-bank buying overwhelmed the real-yield signal — is the subject of [inflation and gold, the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story); for our purposes here the point is structural: **gold keys off the long end (real yields), not the front end (policy path).**

So the seam is clean. Front end → dollar, banks, short duration, via policy and the rate differential. Long end → equity multiples, growth stocks, gold, via the discount rate and real yields. The same curve, two correlation engines. The whole-market version of this map is the [macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix); this post is the rates-specific lens on it.

There is a clean way to *quantify* the discount-rate channel so you can see why the long end hits growth stocks hardest. The value of a stock can be written, in a simplified growth model, as roughly `earnings ÷ (discount rate − growth rate)`. The discount rate is built on the long end. For a value stock with modest growth — say a 3% growth rate and a 9% discount rate — the denominator is `9% − 3% = 6%`. For a high-growth stock — say 7% growth and the same 9% discount rate — the denominator is `9% − 7% = 2%`, a much *smaller* number, so the stock is far more valuable per dollar of earnings (a higher multiple). Now lift the long end so the discount rate rises by 1pp, to 10%. The value stock's denominator goes from 6% to 7%, a `1/6 ≈ 17%` widening, so its value falls about 14%. The growth stock's denominator goes from 2% to 3%, a *50%* widening, so its value falls about 33%. Same 1pp rise in the long end; the growth stock falls more than twice as hard. That asymmetry — long-duration cash flows are far more sensitive to the discount rate — is the precise reason a *long-end-led* move is a growth-stock and multiple-compression story, distinct from the front-end-led dollar-and-banks story.

### Why the front end leads: the lead/lag of forecast over realization

It is worth pausing on *why* the front end leads the funds rate, because the lead/lag is itself a measurable feature of the correlation and a tradeable one. The front end leads because it is a *forecast* and the funds rate is the *realization* of that forecast. A forecast, by definition, moves the moment new information arrives; the realization moves only when the FOMC actually meets and votes. Between meetings — and the Fed meets only eight times a year, roughly every six weeks — the funds rate is frozen, but the two-year is repricing continuously on every data point. So the two-year *front-runs* every Fed move by weeks to months, with the lead determined by how far ahead the data lets the market see.

This lead is why a trader watching the front end has, in effect, a preview of the Fed's next several moves. When the two-year sits well above the funds rate, the market is telling you hikes are coming and the Fed is "behind the curve" — it will likely confirm the front end's forecast at upcoming meetings. When the two-year sits well *below* the funds rate, the market is forecasting cuts, and the Fed will likely follow. The front end converging *to* the funds rate (rather than the funds rate driving the front end) is the correct mental model, and it inverts the naive causality almost everyone starts with. The funds rate is the lagging, confirmed value; the two-year is the leading, forecasted value. In the language of [the lead-lag post](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators), the front end is a *leading* indicator of policy and the funds rate is the *coincident-to-lagging* confirmation.

#### Worked example: 2-year duration P&L on a front-end move

To trade the front end you need to know how much money a yield move actually makes or loses on a note. The relevant number is **modified duration**: the percentage change in a bond's price per 1 percentage point change in its yield. A two-year Treasury note has a modified duration of roughly 1.9. So if the two-year yield rises by 50 basis points (0.50pp) on a hawkish repricing, a two-year note's price falls by approximately:

```
price change  ~=  -modified duration  x  yield change
              ~=  -1.9  x  0.50%
              ~=  -0.95%
```

On a \$1,000,000 two-year position, that is a loss of `\$1,000,000 × 0.95% = \$9,500`. Small, by design — that is the *point* of the front end. Compare it to a ten-year note, with modified duration around 8.5: the same 50bp move would cost `-8.5 × 0.50% = -4.25%`, or `\$42,500` on the same notional. The intuition: the front end carries far less duration risk per dollar, which is exactly why traders express *pure Fed-path views* through it — you get the cleanest policy signal with the smallest exposure to the growth-and-term-premium noise that swamps the long end.

## The split, in one move: a hawkish repricing

The cleanest way to *see* the two engines is to watch a single hawkish data surprise hit the curve. A hawkish surprise — a hotter-than-expected inflation print, say — does its work primarily by revising the *expected Fed path*. That revision lands hardest on the front end, because the front end *is* the path. The long end moves too, but less, because a single print is a smaller fraction of what the ten-year prices.

![Two-panel bar chart showing the 2Y repricing more than the 10Y and the dollar rising while gold falls on a hawkish surprise](/imgs/blogs/the-fed-funds-path-and-front-end-correlation-6.png)

The figure quantifies it with the event-study betas from the 2022-23 inflation-fear regime. For a +0.1pp upside surprise in core CPI: the two-year yield rises **+9bp**, the ten-year only **+7bp** — the front end repriced more. And on the *same* print, the dollar rises **+0.35%** while gold falls **−0.80%**. Read the whole reaction as one event: the hot print revised the Fed path up, which (a) lifted the front end most, (b) widened the rate differential and firmed the dollar, and (c) pushed real yields up, which knocked gold. The 2Y and the dollar moved *together* — they are the same event. This is the signature of a hawkish repricing, and it is why traders watch the two-year and the dollar in the same glance after an inflation print. (The full cross-asset beta table — equities, Nasdaq, Bitcoin — lives in [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises), the post that explains *why* you correlate the surprise rather than the level.)

The deep rule behind the +9bp-vs-+7bp pattern is this: **the closer an instrument is to being a pure expectation of the thing your data informs, the larger its beta to that data's surprise.** A CPI surprise directly revises the near-term Fed path, and the two-year is almost a pure expectation of that path, so it has the biggest beta. The ten-year blends the path with long-run views, so a single print is diluted. The dollar inherits the front end's move through the rate differential. Gold inherits the long end's real-yield move. One print, two engines, a whole cross-asset reaction that snaps into focus once you know which end of the curve each asset hangs off of.

#### Worked example: turning a 2-year move into a count of Fed hikes

The most useful translation a front-end trader does is converting a two-year yield move into a *number of Fed meetings repriced*, because that is the unit the market thinks in. The arithmetic rests on the path identity. A single 25bp Fed hike, if it is expected to *persist* for the full two years, lifts the average expected funds rate by the full 25bp and so lifts the two-year by roughly 25bp. But many repricings are about hikes that happen *partway* through the window or are only *probable*, so the two-year move is a fraction of 25bp per "increment."

Take the hawkish CPI print above: the two-year rose +9bp. As a rough read, `9bp ÷ 25bp ≈ 0.36` — the market just added a bit more than a *third of one extra hike* to the expected path, or equivalently raised the probability of an already-expected hike by about a third. Now scale it up: suppose a much hotter print sends the two-year up 50bp in a session. That is `50 ÷ 25 = 2` increments — the market repriced the path by roughly *two extra quarter-point hikes*. On a \$1,000,000 two-year position, that 50bp move costs about `1.9 × 0.50% × \$1,000,000 = \$9,500` (the duration math from the next example), and it tells you, in one number, that the Fed is now expected to go two hikes further than the market thought yesterday. The intuition: a front-end yield move *is* a repricing of the Fed's path in units of meetings, and translating basis points into "extra hikes" is how you read what the bond market just learned about the Fed.

## Common misconceptions

A handful of beliefs about the front end are widespread and wrong. Each correction comes with a number.

**Myth 1: "The Fed sets interest rates, so the bond market follows the Fed."** Backwards. The Fed sets *one* overnight rate. The bond market sets *everything else* by forecasting the Fed's future path, and the front end *leads* the funds rate, not the other way around. In March 2022 the two-year was at 2.3% while funds were 0.50% — the market had priced almost the entire +525bp cycle before the Fed delivered any of it. By the time the Fed hit 5.50%, the two-year was already falling. The Fed *confirms* the path the front end has forecast; surprises happen only when the Fed deviates from that priced path.

**Myth 2: "A higher 10-year yield means the Fed is hiking."** Not necessarily. The ten-year can rise because of *term premium* or *growth/supply* with no change in the expected Fed path at all — a "bear steepener," where the long end rises faster than the front end. Late 2023 saw exactly this: the ten-year jumped toward 5% on a wave of Treasury supply and term-premium repricing while the front end (the Fed path) was roughly steady. The lesson is the whole thesis of this post: *which end* moved determines the story. A front-end-led rise is a Fed repricing (watch the dollar); a long-end-led rise is a growth/term-premium move (watch equity multiples).

**Myth 3: "A strong dollar is good for US stocks."** Usually the opposite, and the reason is the front-end link. The dollar most often rallies *because* the front end is firming (a hawkish Fed repricing), and that same repricing is hitting equity multiples through the long end. So a rate-driven dollar rally typically coincides with equity weakness, not strength. The data shows the dollar's correlation with the S&P at −0.20 — modestly negative — and far more negative with the rate-sensitive corners (EM equities −0.55, gold −0.55). A strong dollar is a *symptom* of the tightening that is pressuring risk assets.

**Myth 4: "The 2-year and the 10-year are basically the same trade with more or less risk."** No — they are *different correlations*, not the same correlation scaled. The two-year is a Fed-path/dollar instrument; the ten-year is a growth/multiple/gold instrument. A position in the two-year is a bet on policy; a position in the ten-year is a bet on growth and term premium. They can even move in *opposite* directions in the same session (a "twist"), which is impossible if they were the same trade. Treat them as two distinct exposures.

**Myth 5: "Banks always benefit from higher rates."** Only the well-funded, deposit-rich banks, and only on the *margin* side. A rising front end widens NIM on new business, but it simultaneously marks down the bonds a bank already holds. A bank that mismatched long-dated bonds against flighty deposits can be destroyed by the same rate rise that fattens a healthier bank's margin — Silicon Valley Bank, March 2023, is the cautionary number: a duration hit large enough to trigger a deposit run and a failure inside a hiking cycle. "Higher rates help banks" is true on average and fatal in the tails.

## How it shows up in real markets

Three dated episodes show the front-end correlation working — and the one place it gets subtle.

### 2022: the fastest hiking cycle, with the 2Y and the dollar surging together

2022 is the textbook case of a front-end-led regime. The Fed hiked from 0.25-0.50% to 4.25-4.50% over the year (and on to 5.50% by mid-2023), the fastest pace in four decades. The two-year yield, which had begun the year near 0.73%, surged through 4.4% by December — pricing the path as fast as the Fed could deliver it. And the dollar did exactly what the rate-differential mechanism predicts: the DXY climbed from a 2021 close of 95.7 to a 2022 *intraday* peak of **114.8** in late September, a twenty-year high, before closing the year at 103.5.

![Line chart of the US Dollar Index by year with the 2022 intraday peak of 114.8 marked at a twenty-year high](/imgs/blogs/the-fed-funds-path-and-front-end-correlation-3.png)

The figure shows the surge. The dollar peaked in late September 2022 — precisely when the front-end repricing was at its most violent and the rate differential against Europe and Japan was widest. (The yen, with the Bank of Japan still pinning its own front end at zero, blew through 150 to the dollar that autumn — the rate differential at its starkest.) Then, as the hiking cycle approached its end in 2023 and the market began to price *cuts*, the differential started to narrow and the dollar eased. The whole arc of the 2022 dollar is the front-end correlation, drawn large: up with the repricing, down as the repricing faded. Note the symmetry with the early years — the 2020 low of 89.9 came when the Fed had cut to zero and the rate differential had collapsed.

What makes 2022 the *clean* case is that, for once, both sides of the dollar smile pointed the same way. The front end was firming hard (the right side), *and* risk was selling off as the rate shock hit equities and bonds together (a touch of the left side). With both drivers pulling the dollar up at once, the move was enormous and one-directional — a 20% rally peak to peak in the broad dollar. It is the rare year where you didn't have to disentangle the two drivers because they agreed. That is also why 2022 is the year every cross-asset correlation went haywire: a single dominant variable — the front-end-led tightening — drove the dollar up, gold sideways, equities and bonds down *together*, and commodities up, all from the same root. When one macro force is this dominant, the front end is its leading edge, and the dollar is its clearest scoreboard.

### 2024: the first-cut repricing

If 2022 was the front end pricing hikes, 2024 was the front end pricing *cuts* — the same engine in reverse. Through 2024, the market repeatedly repriced *when* and *how fast* the Fed would cut. Each dovish data point (a cooler CPI, a softer jobs report) pulled the two-year down as the expected path eased; each hawkish point pushed it back up. The two-year, which had peaked at 5.05% in October 2023, fell to 3.66% by September 2024 as the first cut arrived, then bounced to 4.25% by December as the market trimmed the number of cuts it expected for 2025. The dollar tracked it: soft into the first cut, then firmer as the easing path was repriced shallower — DXY closing 2024 at 108.5, *higher* than 2023, because the market decided the Fed would cut *less* than feared. The macro-trading series develops the mechanics of pricing the easing path in [terminal rate and rate-cut cycles](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession); the correlation lesson here is that the front end and the dollar move together in *both* directions — a dovish repricing softens the dollar, a "less dovish than feared" repricing firms it.

### The carry trade: living on the rate differential

The purest expression of the front-end-to-dollar correlation is the **carry trade**: borrow a low-yielding currency, invest in a high-yielding one, and pocket the rate differential. For most of 2022-2024, the dollar (front end at 5%+) against the yen (front end pinned near zero) was the marquee carry trade. The differential was enormous — over 5pp — and the trade printed money as long as the dollar stayed firm. The mechanics are exactly the carry from the worked example earlier, scaled across the whole global investor base: borrow yen at ~0%, hold US bills at ~5%, collect the 5pp gap, and ride the dollar higher as everyone else does the same. The trade is self-reinforcing on the way up, because the act of putting it on (selling yen, buying dollars) pushes the dollar up further, which adds an exchange-rate gain on top of the carry.

But carry trades carry a tail: when the differential suddenly *narrows* (the funding-currency central bank hikes, or the high-yield central bank signals cuts), the trade unwinds violently. The unwind is the mirror image of the build-up — everyone scrambles to buy back yen and sell dollars at once, and the same self-reinforcement that drove the trade up now drives it down in a cascade. In early August 2024, a Bank of Japan hike plus a soft US jobs report compressed the US-Japan front-end differential, and the yen carry trade unwound in days — the VIX spiked to 65.7 and global risk assets convulsed. The lesson: the front-end correlation with the dollar is the *engine* of the carry trade, and the *speed* at which the front end can reprice is the carry trader's biggest risk. A carry position is, underneath, a leveraged short-volatility bet on the rate differential staying wide — which is why it earns a steady drip for months and then loses years of profit in a single week when the front ends converge.

### Where it gets subtle: the long-end-led move that fakes a Fed story

The trap is a curve move that *looks* like a Fed repricing but isn't. Late 2023 is the case. The ten-year yield surged toward 5% — its highest since 2007 — and headlines blamed "higher for longer." But the *front* end barely moved; the Fed path was roughly stable. The ten-year was rising on **term premium** and **Treasury supply** (a flood of new issuance), not on a hawkish Fed repricing. A trader who read it as a front-end move and bought the dollar got the correlation wrong: the dollar's reaction was muted because the *rate differential* (the front-end signal) hadn't actually widened. The fix is the discipline this whole post is building: before you trade a curve move, ask *which end led*. A long-end-led move is a growth/term-premium/supply story, and its partner trades are equity multiples and gold — not the dollar and the banks.

## How to read it and use it

Here is the playbook, distilled to a decision you can run on any curve move.

![Two-by-two matrix mapping which end of the curve led and whether yields rose or fell to the active regime and trade](/imgs/blogs/the-fed-funds-path-and-front-end-correlation-5.png)

**Step 1 — identify which end led.** Look at the two-year and the ten-year side by side. Did the two-year move *more* than the ten-year (front-end-led) or *less* (long-end-led)? This single read is the fork in the road. A quick proxy: watch the 2s10s spread *and* the level of the two-year. If the two-year is doing the work, it is a Fed-path story.

**Step 2 — read the direction.** Yields up or down? Combine with step 1 using the matrix above:

- **Front end up (hawkish repricing):** the market is adding hikes or removing cuts. Active correlation: 2Y up *with the dollar up*. Partner trades: long dollar, long banks/financials, fade gold and long-duration growth. The 2022 trade.
- **Front end down (dovish repricing):** the market is adding cuts. Active correlation: 2Y down, dollar softens, short-duration rallies. The 2024 first-cut trade.
- **Long end up (growth/supply/term premium):** equity multiples compress, gold pressured via real yields, but the *dollar may not react* because the rate differential hasn't moved. Watch growth stocks, not the dollar. Late-2023 trade.
- **Long end down (growth scare or flight to safety):** recession or risk-off bid; duration and gold rally, equities wobble. Risk-off.

**Step 3 — confirm with the partner asset.** The cross-check that keeps you honest: if you think a move is front-end-led, the *dollar* should be moving in agreement. If the two-year jumps and the dollar yawns, either the move isn't really a Fed repricing or the differential against the rest of the world didn't widen — re-examine. A front-end story without a dollar confirmation is a story you should distrust.

**What invalidates the front-end-to-dollar correlation.** Three things. First, when *foreign* front ends move too: if the Fed hikes but the ECB hikes more, the US two-year can rise while the dollar *falls*, because the differential narrowed. Always think in *relative* rates, not absolute US rates. Second, when the dollar trades on *risk* rather than rates: in a true panic, the dollar rallies as a safe haven even with the front end falling (a flight to cash). Third, when *fiscal/supply* dominates: a debt-ceiling scare or a supply glut can move the dollar and the long end without any front-end repricing. The correlation is a *regime* — it holds when the Fed path is the dominant macro variable and weakens when something else takes over the wheel.

**A note on time horizon.** The front-end correlations are sharpest at the *event* and *days-to-weeks* horizon — the window in which a data surprise reprices the path and the dollar and banks respond. Over *months and quarters*, slower forces (the business cycle, fiscal supply, foreign central banks, positioning unwinds) layer in and the simple "2Y up, dollar up" relationship gets noisier. So use the front-end read to interpret and trade *moves*, especially around data and FOMC meetings, and lean on the slower frameworks (the [business-cycle clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock), the whole-market matrix) for the multi-month picture. A correlation that is +0.6 over a week of repricing can be +0.2 over a full year that contained three different regimes — the regime-dependence the [opening post of the series](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) is built around. Match the tool to the horizon.

**The most common practical error**, beyond mistaking the long end for the front end, is forgetting that *foreign* front ends exist. Traders anchored to the US data calendar watch the US two-year and the dollar religiously and forget that the differential is a *difference*. A US two-year that rises 20bp while the German two-year rises 30bp is a *narrowing* differential and a *bearish* signal for the dollar — even though "US rates went up." Always ask: did the US front end move *relative to* the rest of the world? The correlation is with the *differential*, never with the absolute level of US rates alone. Build the habit of glancing at the German and Japanese front ends before you conclude a US move was dollar-positive.

**The one-sentence skill.** Look at a curve move and finish this sentence: "The ___ end led, yields went ___, so the active correlation is ___, and the partner trade is ___." If you can fill those four blanks, you have read the move correctly. The front end up with the dollar means a hawkish Fed repricing; the long end up without the dollar means a growth/supply story. *Which end moved tells you which correlation is live* — that is the whole post in nine words.

## Further reading and cross-links

Within this series:

- [Bond yields: the master correlation with every asset](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset) — the parent post; this one zooms into the short end of that master variable.
- [The dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) — the full map of the dollar as cross-asset gravity, of which the front-end channel is the engine.
- [The yield curve as a growth signal and its asset correlation](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) — the *slope* (2s10s) and what inversion forecasts; the companion to this post's *level* lens.
- [The surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — why and how the front-end repricing betas in this post are estimated.

The mechanism (why policy moves the asset) and the release-day reaction (the intraday print) live in two complete sibling series:

- macro-trading: [terminal rate and rate-cut cycles, pricing the path](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) and [the dollar system, why USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) for the rate-differential mechanism.
- event-trading: [the jobs report (NFP), the king of data](/blog/trading/event-trading/the-jobs-report-nfp-the-king-of-data) and [CPI, the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world) for the dot-plot-versus-market-pricing read and how a print repriced the front end intraday.
