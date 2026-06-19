---
title: "The Macro-Correlation Playbook: From Indicator to Position"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The capstone that ties the whole series together: read the regime, look up which correlations it activates, size the position to the beta, and watch for the flip."
tags: ["macro", "correlation", "regime", "stock-bond-correlation", "real-yields", "portfolio", "risk-management", "asset-allocation", "playbook", "capstone"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A macro correlation is not a number you memorize; it is a function of the regime you are in, and the entire job is to read the regime, look up which correlations it turns on, size to those, and watch for the flip.
>
> - Every correlation has four properties: a **sign**, a **strength**, a **lead/lag**, and a **regime in which it flips**. Master those four and you have decoded the whole series.
> - The **growth x inflation quadrant** you are in is the master switch: it decides whether bonds hedge stocks (stagflation: r = +0.55, the hedge fails) or offset them (deflation: r = -0.55, the strongest hedge).
> - **Size to the correlation, not to the view.** A position you think is two bets is often one bet wearing two tickets — and in a crisis the average pairwise correlation rises to about 0.80, so "diversified" becomes one trade.
> - The one number to carry out of the whole series: the stock-bond correlation went from about **-0.50 to +0.60** in 2022. The relationship did not "break." It did exactly what regime theory said it would.

In October 2022 a \$1,000,000 portfolio built the way every textbook said to build it — 60% stocks, 40% bonds — was down roughly **\$210,000** on the year, and the part that was supposed to cushion the fall, the bonds, had *added* to the loss. Long Treasuries had their worst year in over a century. The investor had not picked bad stocks or bad bonds. They had relied on a correlation — "bonds go up when stocks go down" — that had been true for twenty years and was now, suddenly, false.

Here is the thing that should bother you: nothing was broken. The negative stock-bond correlation that powered the 60/40 portfolio for two decades did not malfunction. It flipped, on schedule, because the regime changed. When inflation is the dominant risk in the system, the Federal Reserve hikes rates, and rising rates crush stocks *and* bonds at the same time — they fall together, the correlation goes positive, and diversification evaporates exactly when you reach for it. A trader who understood that the correlation was a *regime, not a constant* did not need to predict 2022. They needed to read the inflation regime, recognize which correlations that regime activates, and stop relying on the bond hedge before it failed.

Step back and notice how strange this is. The investor did everything the industry told them to do. They diversified across two asset classes that had moved in opposite directions for a generation. They were not greedy or reckless. And the very tool that was supposed to protect them — the negative correlation between stocks and bonds — turned around and bit them, not because of a freak event, but because of a *predictable* response to a *visible* change in the macro regime. The whole tragedy of 2022 is that it was forecastable from the regime alone, with no edge in predicting headlines, by anyone who understood that the stock-bond correlation is a function of what is driving the discount rate. When growth drives it, bonds hedge. When inflation drives it, bonds amplify. Inflation took over in 2022, and the correlation flipped on cue. The people who got hurt were not unlucky; they were running a portfolio built for one regime into a different one.

That is the entire series in one sentence, and this post is where it all comes together. Across forty-four other posts we built the pieces: what a correlation actually measures, how each indicator maps to each asset, how the four macro regimes rewire the whole map, and how to measure and trade it without fooling yourself. This capstone assembles those pieces into one operating system — a repeatable loop that takes you **from an indicator to a position** — and then runs the full loop end to end on a real \$1,000,000 portfolio in a 2022-style inflation shock. If you read only one post in the series, read [the spine](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) and then this one.

![Macro correlation operating system from indicator to position](/imgs/blogs/the-macro-correlation-playbook-capstone-1.png)

The figure above is the whole playbook as a loop. An **indicator** (a CPI surprise, a jobs number, an ISM print) feeds a **regime read** — which growth-inflation quadrant are we in? The regime **selects the live correlations** — which sign-and-strength relationships are switched on right now. Those correlations **size the position** — you set risk to the correlation, not to your opinion. And a **flip-watch** closes the loop — you monitor the one input that would invert the sign, and you cut before it does. Read the regime, look up the correlations, size, watch for the flip. Everything else is detail.

## Foundations: the four properties of a correlation

Before we run the playbook, we need to agree on what a correlation *is*, because almost everyone gets this wrong in the same way: they treat it as a fixed fact about two things ("stocks and bonds are negatively correlated"), when it is really a *measurement over a window* that changes as the world changes.

A **correlation coefficient** — usually written *r* — is a single number between -1 and +1 that summarizes how two things move together. If *r* = +1, they move in perfect lockstep: when one is up 1%, the other is up 1%. If *r* = -1, they move in perfect opposition: one up 1%, the other down 1%. If *r* = 0, knowing one tells you nothing about the other. The full mechanics — Pearson versus Spearman, and how correlation relates to **beta** (the *size* of the move, not just the direction) — are in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta). For the playbook you need four properties of every relationship, and only four.

**1. The sign.** Does the asset rise or fall when the indicator rises? A hot CPI print pushes stocks *down* (negative) and the dollar *up* (positive). The sign is the first thing you read and the thing that flips.

**2. The strength.** How tight is the link? A correlation of -0.95 (10-year yields versus long bond prices) is almost mechanical — they are nearly the same trade. A correlation of -0.20 (the dollar versus the S&P 500) is real but loose; it will fail you often. Strength tells you how much to *trust* the relationship.

**3. The lead/lag.** Does the indicator move *before* the asset, *with* it, or *after* it? The yield curve inverts about **14 months** before a recession on average; credit spreads widen about **3 months** before an equity drawdown; initial jobless claims lead the unemployment rate by about **2 months**. Lead/lag is what turns a correlation into a *signal* you can act on instead of a coincidence you notice too late. The full taxonomy is in [leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators).

**4. The flip.** In which regime does the sign reverse? This is the property the series is built around and the one nobody on TV mentions. Stocks and bonds are negatively correlated in a growth-driven world and positively correlated in an inflation-driven world. Gold is negatively correlated with real yields — until central banks start hoarding it and the sign decouples. A correlation without its flip condition is a loaded gun pointed at your portfolio.

It is worth pausing on *why* these four properties are the right four, because the framework does a lot of quiet work. Notice that the first three — sign, strength, lead/lag — are the properties a statistician would list. They are what you can compute from a clean dataset. The fourth, the flip, is the one a *trader* adds, and it is the one that turns the other three from a description of the past into a tool for the future. A statistician computes a correlation and reports it. A trader computes a correlation, asks "under what conditions was this measured, and what would change them," and only then decides whether to trust it. The flip property is the humility that keeps you alive: it is the admission that your -0.96 is a fact about a sample, not a law of nature.

There is also a subtle relationship between strength and the flip that beginners miss. You might assume the *strongest* correlations are the safest — surely an *r* of -0.96 is more reliable than an *r* of -0.20. But strength measures only how tightly the two series tracked *within the sample regime*; it says nothing about how durable that regime is. Gold versus real yields had one of the strongest correlations in all of macro (-0.96) and still flipped hard. The dollar versus the S&P, by contrast, has a weak -0.20 correlation precisely *because* it averages across many regimes — it is loose because it is robust to a wide range of conditions. A weak, stable correlation can be more *tradeable* than a strong, fragile one, because you can size to it without fear of a sudden inversion. The series teaches you to ask not just "how strong" but "how strong, and how stable" — two different questions that beginners conflate into one.

Finally, the four properties interact with each other in ways that matter for sizing. Sign and strength together give you the *expected move* (the beta). Lead/lag tells you *when* you can act (a leading indicator gives you a tradeable window; a lagging one only confirms what already happened). And the flip tells you *how long the trade is good for* before the regime invalidates it. A complete trade specification is all four: "this asset moves -0.70% per +0.1pp CPI surprise (sign and strength), the surprise arrives at 8:30am on the print so I can position the day before (lead/lag), and the sign holds until inflation rolls over (the flip)." Three of those four come from statistics; the fourth comes from understanding the economy. The whole series is really an argument that you need both.

Hold all four in your head at once and you can read any cell of the master map. Let me show you the map.

![The whole series in one correlation matrix heatmap](/imgs/blogs/the-macro-correlation-playbook-capstone-2.png)

This heatmap is the entire series compressed into one picture, and it deserves a slow look. The **rows are macro drivers** — a rise in the 10-year yield, a rise in real yields, a hot CPI surprise, a stronger dollar, higher oil, a rising ISM, a wider credit spread. The **columns are assets** — the S&P 500, the Nasdaq, the long bond, gold, Bitcoin, EM equities, the dollar. Each cell is the sign-and-strength of that driver hitting that asset: deep **green** is strongly positive, deep **red** is strongly negative, pale is weak. The deep-dive on building and reading this grid is [the macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix).

Three patterns jump out, and they are the spine of the whole series. First, look at the **"10Y yield (rise)" row**: it is red across almost every risk asset (S&P -0.45, Nasdaq -0.60, bonds -0.95, gold -0.55, EM -0.50) and green only for the dollar. Rates are the master variable — when the price of money rises, the price of nearly everything else falls. Second, the **"real yield (rise)" row** is even redder on gold (-0.80) than the nominal row, because gold is fundamentally a bet against real yields, not against inflation. Third, the **"ISM/PMI rise" row** is green where the rate rows were red — because growth, on its own, is good for risk. The whole map is really two forces, rates and growth, pulling in opposite directions, and the regime decides which one is in charge.

There is a fourth pattern that is easy to miss but is the deepest one: read *down* the columns instead of *across* the rows. Look at the **dollar column** — it is green almost everywhere (a stronger dollar means a stronger dollar, +1.00 with itself, and the dollar strengthens with rising US yields, +0.40-0.45). Now look at the **gold and EM columns** — they are red against nearly every driver. Gold and EM equities are, in effect, the *anti-dollar*: they are the assets that hurt most when the dollar gravity intensifies. This is why the dollar earns the nickname "cross-asset gravity," explored in [the dollar's cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) and [the dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity). When you cannot decide what a regime will do to a basket of risk assets, ask first what it does to the dollar — the rest of the column often follows from that one sign.

And notice what the matrix does *not* show, which is just as important: it shows the *typical-regime* sign and strength, the long-run average. Every one of these cells has a flip condition that the static heatmap cannot display. The "ISM/PMI rise" row is green on the S&P (+0.55) in a normal expansion, but in a "good-news-is-bad" inflation regime that very cell would flip toward red, because strong growth then means more rate hikes. The matrix is the *map*; the regime selector that comes next is the *legend* that tells you which version of the map is currently in force. Never read a static correlation matrix without first asking which regime it was averaged over — that is the single most common way smart people misuse these numbers.

#### Worked example: reading one cell

Take the cell where "hot CPI surprise" meets "US 10Y bond": the value is **-0.75**. Translate it. A hotter-than-expected inflation print is strongly *negatively* correlated with the bond's price — bonds fall hard. How hard? From the surprise betas, a +0.1 percentage-point upside surprise in core CPI moved the 10-year *yield* up about **7 basis points** in the 2022-23 regime, and a 7bp yield rise on a long bond with a duration near 17 is a price drop of roughly 17 x 0.07% = **1.2%**. So a single hot print could knock about **\$12,000** off a \$1,000,000 long-bond position in one session. The -0.75 is not abstract: it is twelve thousand dollars per surprise. *The cell is a dollar figure once you attach a beta and a position size.*

## The regime selector: which correlations are switched on

A map is useless if you do not know where you are standing. The single most important skill in this series is locating yourself on the **growth x inflation grid** — a 2x2 of growth (up or down) and inflation (up or down) that produces four regimes, each of which switches a different set of correlations on. This is the master switch; the full treatment is in [the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants), and the mechanism behind why policy responds differently in each is in [the business cycle's four phases](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders).

![Regime selector quadrant returns heatmap with stock bond correlation sign](/imgs/blogs/the-macro-correlation-playbook-capstone-3.png)

The four quadrants are:

- **Goldilocks** (growth up, inflation down): the dream regime. Stocks lead (+18% real on average), and crucially the stock-bond correlation is **-0.40** — bonds hedge, the 60/40 sings.
- **Reflation** (growth up, inflation up): the recovery-with-heat regime. Commodities lead (+16%), stocks still do well (+12%), and the stock-bond correlation drifts toward neutral (**-0.10**) — the hedge is weaker.
- **Stagflation** (growth down, inflation up): the nightmare for diversified portfolios. Gold (+12%) and commodities (+14%) lead; stocks *and* bonds both fall; the stock-bond correlation goes **+0.55**, so your hedge becomes a second copy of your risk. This was 2022.
- **Deflation / deleveraging** (growth down, inflation down): the recession regime. Bonds lead (+12% real) and the stock-bond correlation is **-0.55**, its most negative — this is when Treasuries are the best hedge you can own. This was 2008.

The whole point of the figure is that **the quadrant selects the sign on the right margin**. You do not choose your correlations; the regime chooses them for you. Standing in deflation and standing in stagflation, you own the *same* assets — stocks and bonds — but you own a completely different *portfolio*, because the relationship between those assets has inverted. A 60/40 in deflation is genuinely diversified. A 60/40 in stagflation is a leveraged bet on the same factor twice.

It helps to understand *why* each quadrant produces the sign it does, because then you can reason about regimes the data has not seen yet rather than memorizing four cases. The engine in every quadrant is the discount rate, and what differs is *which way it moves and why*. In **deflation**, growth is collapsing, so the central bank slashes rates and yields fall — that lifts bond prices (lower yield, higher price) at the very moment stocks are falling on collapsing earnings. Bonds up, stocks down: a strong *negative* correlation, the best hedge you can own. In **stagflation**, the central bank is forced to *raise* rates to fight inflation even as growth weakens — so yields rise, which crushes bond prices *and* compresses equity valuations through the same discount-rate channel. Bonds down, stocks down: a strong *positive* correlation, the hedge fails. The two nightmare-versus-dream cases are mirror images driven by the same variable moving in opposite directions for opposite reasons.

The two growth-up quadrants sit in between. In **Goldilocks**, falling inflation lets the central bank stay easy while growth lifts earnings — stocks soar, bonds do fine, and because the discount rate is calm the two are only loosely linked (a mild negative). In **reflation**, growth and inflation rise together, so yields drift up (hurting bonds) while earnings rise (helping stocks) — the two forces roughly cancel and the correlation sits near zero, which is why reflation is the trickiest regime to hedge: bonds neither reliably help nor reliably hurt. The lesson is that the stock-bond correlation is essentially a *barometer of what is driving the discount rate*. When growth drives it, the correlation is negative. When inflation drives it, the correlation is positive. The regime selector is just that one idea, drawn as a grid.

How do you tell which quadrant you are in, in real time? You read the lead indicators. **Growth** you read from ISM/PMI new orders, jobless claims, and the yield curve (covered in [ISM/PMI the leading correlation](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals) and [the yield curve as a growth signal](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation)). **Inflation** you read from CPI, core CPI, PPI upstream, and breakevens (covered in [CPI and asset prices](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation) and [PCE, breakevens, and forward inflation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation)). You do not need to predict the regime. You need to recognize the one you are *in* and respect the correlations it has switched on. The rotation across all four phases over a cycle is the subject of [the business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock).

#### Worked example: locating yourself in mid-2022

In June 2022 the readings were: core PCE inflation near **5.6%** (its peak), ISM new orders rolling over from the low-60s toward 50, the 2s10s curve about to invert, and the Fed hiking 75 basis points a meeting. Growth: turning down. Inflation: very high and rising. That is the **stagflation** corner. The regime selector immediately tells you: the stock-bond correlation is about **+0.55**, gold and commodities are the leaders, and your bond allocation is *not* a hedge — it is a duplicate of your equity duration risk. A trader who simply located themselves on the grid in June 2022 knew, with no forecast required, that a standard 60/40 was about to behave like a 100/0. *Reading the quadrant is worth more than predicting the news, because the quadrant tells you what the news will do.*

## The master transmission: how one print reaches every asset

Now connect the regime to the assets. The reason a single map can cover seven assets is that most macro drivers act through one shared channel: the **discount rate**, the rate at which future cash flows are converted into today's price. Raise the discount rate and every future dollar is worth less today — so every asset priced off future cash flows falls. That is the mechanism; it is derived in detail in [interest rates, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map). Here we care about the *measured* version: how big is the move, per unit of surprise?

The most important refinement in the whole series is this: **markets trade the surprise, not the level.** A 4% inflation print is not bullish or bearish in itself — what matters is whether 4% was higher or lower than the consensus expected. If everyone expected 4.2% and it printed 4.0%, that is a *dovish* surprise and risk assets rally, even though inflation is "high." The asset's reaction is its **beta to the surprise**, and the surprise framework is the foundation of both this series and the [event-trading surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework). The deep-dive here is [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).

![Asset beta to one hot CPI surprise across every asset](/imgs/blogs/the-macro-correlation-playbook-capstone-5.png)

This bar chart is the master transmission, measured. It shows how every asset reacted to a **+0.1 percentage-point upside surprise** in core CPI during the 2022-23 inflation-fear regime. Read it left to right by magnitude: Bitcoin -1.60% (the highest-beta risk asset), the Nasdaq -1.00% (long-duration tech), gold -0.80% (real yields up), the S&P -0.70%, the dollar +0.35% (the only winner among prices), and yields up — the 2-year **+9bp**, the 10-year **+7bp**. One print, one surprise, seven assets, all repriced through the same discount-rate channel. The sign pattern is exactly the "hot CPI" row of the master matrix, now with magnitudes attached.

Notice the hierarchy of betas. The longer an asset's "duration" — the further out its cash flows, or in Bitcoin's case the *absence* of any cash flow — the bigger its reaction to a rate surprise. This is why a single liquidity or rate shock lands hardest on the Nasdaq and on crypto, the subject of [crypto as a macro asset](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation) and [global liquidity and the everything correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation).

#### Worked example: turning a CPI surprise into a P&L

Suppose you are long \$1,000,000 of the Nasdaq-100 and CPI prints +0.3pp hot on core (a 0.3pp surprise). The beta is -1.00% per +0.1pp, so the expected move is -1.00% x 3 = **-3.0%**, or about **-\$30,000** in one session. If instead you held \$1,000,000 of the S&P (beta -0.70%), the hit would be -2.1%, about **-\$21,000**. The same news, the same dollar exposure, a \$9,000 difference — purely because the Nasdaq's higher beta to rate surprises makes it a bigger bet on the same regime. *Position size in dollars is not your exposure; dollar size times beta is.*

The two cleanest single-driver correlations in the whole series — the ones with the highest *r* — are bonds versus yields and gold versus *real* yields. The bond one is almost mechanical (price and yield are two sides of one coin; see [bond yields, the master correlation](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset)). The gold one is the cleaner *macro* relationship and the best teaching case for why a strong correlation is still not a constant.

![Gold versus 10 year real yield scatter with regime break](/imgs/blogs/the-macro-correlation-playbook-capstone-6.png)

The scatter plots the gold price against the 10-year TIPS real yield, year by year. The **green dots (2007-2021)** trace a beautifully tight downward line: as real yields fall, gold rises, with a correlation of about **r = -0.96** and a fitted relationship of roughly gold ≈ \$1,489 − \$354 x (real yield). That is one of the cleanest macro correlations that exists — gold is, mechanically, a long-duration bet against real yields, because it pays no income and competes directly with the real return on a TIPS. The full mechanism is in [inflation and gold, the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story) and [real yields, the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation).

Then look at the **red dots (2022-2025)**. The correlation does not just weaken — it *inverts* to about **r = +0.80**. Real yields surged from -0.95% to +2.0% over those years, which on the old line should have crushed gold below \$1,000. Instead gold rallied to \$2,650. The relationship decoupled because a *new driver* — relentless central-bank gold buying — overwhelmed the real-yield channel. This is the perfect capstone lesson: a correlation can have an *r* of -0.96 over fourteen years and still flip on you, because *r* measures the past under a particular regime, and the regime changed. When you see a correlation break like this, it is a signal that a new force has entered the system; the framework for that is [structural shifts](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays).

The practical move when a strong correlation breaks is not to abandon it but to *diagnose* it. Ask: did the old mechanism stop working, or did a new, larger force temporarily dominate it? For gold, the real-yield mechanism did not break — gold is still, all else equal, a long-duration bet against real returns. What happened is that a new buyer with a different objective (central banks diversifying reserves away from the dollar for geopolitical reasons, not chasing real yields) entered at a scale that swamped the marginal real-yield trade. That is a *structural* shift, and it changes how you size the position: you can no longer mechanically short gold when real yields rise, because there is now a price-insensitive buyer underneath. The same diagnostic applies to the stock-bond flip — the negative-correlation mechanism (growth shocks dominating) did not break; a different mechanism (inflation shocks) took over. In both cases the discipline is identical: a broken correlation is data, not noise. It is the market telling you the regime has a new driver, and your job is to identify the driver and re-derive the live correlations around it.

#### Worked example: re-sizing gold after the break

In 2021 you ran a rule: short gold when the 10-year real yield rises 50bp, because the -0.96 correlation and the fitted slope (gold ≈ \$1,489 − \$354 x real yield) implied a 50bp rise should drop gold by about \$354 x 0.50 = **\$177/oz**. In 2022 real yields rose far more than 50bp — and gold *rose*. Had you kept the rule on a \$150,000 gold short, you would have been run over for a double-digit-percent loss, perhaps **\$30,000-plus**, as gold climbed to \$2,650. The flip-watch on this correlation was "watch for a price-insensitive structural buyer." Once central-bank buying data confirmed the new driver, the correct move was to *cut the short and flip to long*, because the dominant force had reversed. *A trading rule built on a correlation must die the moment the correlation's driver changes — keeping a dead rule alive is how strong historical relationships produce the largest losses.*

## Lead, lag, and the canaries that warn you early

A correlation that is *coincident* — that moves at the same instant as the asset — is interesting but not very useful, because by the time you see it the move has already happened. The correlations that pay are the ones with a **lead**: the indicator that moves first, giving you a window to position before the asset catches up. The series spends real time on these because they are where a correlation becomes a tradeable signal rather than a rear-view-mirror observation. The full taxonomy is in [leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators); here is the synthesis you carry into the playbook.

The three most important *leading* macro correlations, ranked by how reliably they lead:

- **The yield curve leads recessions by about 14 months.** When the 2-year yield rises above the 10-year (an "inversion"), a recession has historically followed roughly a year or more later — the 1989 inversion led by 18 months, 2000 by 13, 2006 by 22. The mechanism is in [the yield curve as a growth signal](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) and [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession). The catch — and it is a big one — is that the lead time is long and variable, and the 2022 inversion had *not* produced a recession at the time of writing, fueling the "is the signal broken or just slow" debate. A long lead is a warning, not a trade trigger.

- **Credit spreads lead equity drawdowns by about 3 months.** When high-yield bond spreads widen — investors demanding more compensation to hold risky debt — equity selloffs tend to follow within a quarter, with a correlation around -0.70 that *tightens toward -1 in stress*. Credit is the "smart money" canary: bond investors, who sit ahead of equity holders in the capital structure, smell trouble first. This is the most actionable of the three because the lead is short enough to trade and the signal is precise. The deep-dive is [credit spreads, the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).

- **ISM new orders lead S&P earnings growth by about 6 months, and initial jobless claims lead the unemployment rate by about 2 months.** These are the growth canaries. A rolling-over ISM new-orders index warns that the earnings cycle is about to turn; a rising trend in weekly jobless claims warns that the labor market — and with it the recession-versus-expansion call — is deteriorating before the monthly unemployment rate confirms it. These feed directly into your regime read. The detail is in [ISM/PMI, the leading correlation](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals) and [unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation).

There is one *coincident* signal that deserves special mention because it is the purest real-time read on whether the whole correlation structure is about to change: the **VIX**, the market's expected volatility. The VIX does not lead — it spikes *with* the crisis — but its spike is the unmistakable signal that you have entered the regime where *all* correlations converge toward one. When the VIX jumps from its long-run average near 19.5 to 65 or 80 (it hit 82.7 in March 2020, 65.7 in the August 2024 yen-carry unwind), diversification has just failed, because the average pairwise correlation across risk assets has jumped from about 0.25 to about 0.80. The VIX is your "the map just changed" alarm, covered in [the VIX, risk-on/risk-off and the correlation spike](/blog/trading/macro-correlations/the-vix-risk-on-risk-off-and-the-correlation-spike).

#### Worked example: trading the credit-spread lead

You manage a \$1,000,000 portfolio that is 70% equities (\$700,000). Your credit canary fires: high-yield spreads widen from 4.0% to 5.7% over six weeks — a meaningful move with a 3-month lead on equity drawdowns. The historical pattern says an equity selloff of roughly 10-15% may follow within a quarter. You do not need to predict the catalyst; the credit market has already voted. You cut equity from 70% to 50% (\$200,000 de-risked) ahead of the drawdown. If the S&P then falls 12%, the \$200,000 you de-risked avoids about **\$24,000** of loss, and the cash is dry powder to redeploy when spreads peak (wide spreads have historically preceded *high* forward returns — the 10.8% spread of late 2008 preceded a +35% forward year). *The leading indicator does not tell you what will happen; it tells you the smart money already thinks it will, and that is enough to act.*

## Measure it honestly: the traps and the toolkit

The four-property framework and the master map are only useful if you measure correlations *honestly*, and the most expensive mistakes in this whole field come from measuring them lazily. Three traps recur, and the toolkit posts in the series exist to defuse them.

**Trap one: the full-sample number lies.** If you compute the stock-bond correlation over 1990-2025 you get a mild positive number that describes *no actual regime* — it is the average of a strongly negative two decades and a strongly positive few years, and the average is true of nothing. Always condition on the regime. The fix is the **rolling correlation** — recomputing *r* over a moving window — covered in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) and operationalized in [rolling-correlation regimes and change-point detection](/blog/trading/macro-correlations/rolling-correlation-regimes-and-change-point-detection).

![Stock bond rolling correlation 1990 to 2025 the regime spine](/imgs/blogs/the-macro-correlation-playbook-capstone-4.png)

This is the spine chart of the entire series — the rolling 24-month stock-bond correlation from 1990 to 2025 — and it is the single most important picture to internalize. The green region (below zero) is where bonds hedged stocks: roughly 1998 through 2021, the **great diversifying era** that made the 60/40 famous, with the correlation often near **-0.50**. The red region (above zero) is where they fell together: the high-inflation early 1990s and, dramatically, **2022**, when the correlation spiked to **+0.60**. A full-sample average would smear these two regimes into a meaningless middle. The deep-dive on this exact flip is [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) and the post-mortem on the 2022 break is [when correlations break](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip).

**Trap two: correlation is not causation, and spurious correlations abound.** Two series can correlate beautifully because both follow a third thing (the business cycle), or by pure chance over a short sample, or because of a structural overlap that will not persist. The discipline of separating real economic linkages from coincidences is [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data). A correlation you cannot explain mechanically is a correlation you should not size to.

**Trap three: the relationship is non-linear, so a single *r* misleads.** Inflation and stocks are the classic case. People say "stocks hedge inflation" or "stocks hate inflation" — both are half-true. Real equity returns are best in *moderate* inflation (about +10% real in the 2-3% band), and negative in *both* deflation (-2%) and high inflation (-5% above 5%). It is a U-shape, not a line, so any single correlation number is an artifact of which part of the curve your sample sat on. The correlation of equities with inflation surprises is mildly *positive* (+0.10) when inflation is low and stable but strongly *negative* (-0.45) when it is high and rising — the sign literally flips with the level. This is the canonical "correlation that flips," covered in [inflation and stocks](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips).

The toolkit to do this measurement properly — rolling windows, event-study surprise betas, change-point detection, and an honest backtest — is the subject of the Python track: [building a correlation dashboard](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python), [measuring beta to data surprises](/blog/trading/macro-correlations/measuring-beta-to-data-surprises-an-event-study-in-python), [from correlation to signal](/blog/trading/macro-correlations/from-correlation-to-signal-building-a-macro-overlay), and the crucial guardrail, [backtesting a correlation without fooling yourself](/blog/trading/macro-correlations/backtesting-a-correlation-without-fooling-yourself).

![The series in one map the eight tracks](/imgs/blogs/the-macro-correlation-playbook-capstone-7.png)

The figure above is the series itself as a map: the four tracks of learning, with the regime at the center because it is what selects everything else. Foundations teach you what a correlation *is* (sign, strength, lead-lag, flip). The driver track gives you every indicator's beta. The regime track shows how the four quadrants rewire the map and when it breaks. And the measurement-and-running track turns it into a live process. Each cell carries its key number, so the map doubles as a cheat sheet.

## Common misconceptions

Five myths cause more real-money damage than any others. Each is corrected with a number.

**"My portfolio is diversified because I own stocks and bonds."** Only in some regimes. In deflation the stock-bond correlation is **-0.55** (genuinely diversified); in stagflation it is **+0.55** (one bet, two tickets). Diversification is a property of the *regime*, not of the *holdings*. The classic framing of this "free lunch" and its limits is in [the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch).

**"Gold is an inflation hedge."** No — gold tracks **real yields**, not inflation. From 2007-2021 its correlation with real yields was **-0.96**, far tighter than its link to CPI. Gold rises when inflation rises *only if* real yields fall as a result. When real yields rose in 2022 despite high inflation, gold's old relationship inverted entirely. Gold is a bet against the real cost of money, dressed up as an inflation story.

**"Diversification protects you in a crash."** It protects you least when you need it most. The average pairwise correlation across major risk assets is about **0.25** in calm markets, **0.45** in a normal selloff, and **0.80** in a crisis deleveraging. Everything sells off together when leverage unwinds, because investors sell what they *can*, not what they want to. This is the brutal lesson of [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) and [correlation during crises](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails).

**"Good economic news is good for stocks."** Only in a growth-led regime. In 2022-23's "good-news-is-bad-news" regime, a +100k upside surprise in non-farm payrolls moved the S&P **-0.50%** (strong jobs meant more Fed hikes), where in a normal expansion the same surprise moves it **+0.35%**. The sign of the jobs-to-stocks correlation flips with the regime — the deep-dive is [NFP and asset prices](/blog/trading/macro-correlations/nfp-and-asset-prices-the-king-of-data-correlation).

**"A high correlation means a reliable trade."** A high *historical* correlation only tells you the relationship held in the *past regime*. Gold versus real yields had an *r* of -0.96 and still flipped to +0.80. The number is necessary, not sufficient — you also need to know *why* it holds and *what would break it*. That is the difference between [common correlation mistakes](/blog/trading/macro-correlations/common-correlation-mistakes-and-how-to-avoid-them) and a real edge.

## How it shows up in real markets: the playbook in action

Now let us run the whole loop end to end on a real \$1,000,000 portfolio. This is the synthesis the series was building toward: regime → live correlations → position → size → flip-watch.

### Step 1 — Read the regime

The date is mid-2022. The readings: core PCE at **5.6%** and rising, headline CPI at 9%, the Fed hiking 75bp a meeting with more promised, ISM new orders rolling toward 50, the 2s10s yield curve inverting. Inflation: extreme and rising. Growth: decelerating. That is the **stagflation quadrant** — the upper-left nightmare of the regime selector. We did not forecast it; we located it from the lead indicators in about thirty seconds.

### Step 2 — Look up the live correlations

The stagflation quadrant switches on a specific set of correlations, and we read them straight off the master map and the quadrant table:

- **Stock-bond correlation: +0.55.** Bonds will *not* hedge stocks; they will amplify the loss. Diversification is off.
- **Inflation-equity: -0.45.** With inflation high and rising, hot prints are bearish for stocks — every upside CPI surprise hurts.
- **Gold vs real yields:** real yields are rising, so the *textbook* says gold should fall — but watch the central-bank-buying override that decoupled it.
- **DXY vs EM: -0.55.** The dollar is surging (DXY hit 114 in September 2022), which is a wrecking ball for EM equities and commodities.
- **Credit spreads:** high-yield spreads widening from 3% toward 6%, a -0.70 correlation with equity that tightens toward -1 in stress — the canary, covered in [credit spreads, the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).

### Step 3 — Build the position

Given those live correlations, a naive 60/40 (\$600,000 stocks, \$400,000 bonds) is a *concentrated* bet on the rate factor, because in this regime both legs move the same way. We rebuild around what the regime rewards. A defensible stagflation-aware \$1,000,000 portfolio:

| Sleeve | Allocation | Why this regime rewards it |
|---|---|---|
| Stocks (cut, defensive tilt) | \$350,000 | Stock-bond hedge is off; reduce equity duration |
| Short-duration bonds / cash | \$300,000 | Avoid the long-bond rate hit; earn the rising front-end yield |
| Commodities / energy | \$150,000 | Quadrant leader (+14% real in stagflation) |
| Gold | \$150,000 | Quadrant leader (+12%); watch the real-yield decoupling |
| Long Treasuries (small) | \$50,000 | Only as a recession-flip option, not as the main hedge |

The total equity-plus-duration risk has been cut, and the hedge has been moved out of long bonds (which fail in this regime) and into commodities and gold (which lead it).

### Step 4 — Size to the beta, not the view

Here is where most people go wrong: they size by *conviction* ("I really like gold") instead of by *beta* (how much the position will actually move per unit of the driver). Suppose your risk budget is a 1% portfolio move (**\$10,000**) per +0.1pp CPI surprise. Using the surprise betas:

- Gold's beta is -0.80% per +0.1pp surprise, so a \$150,000 gold position contributes 0.80% x \$150,000 = **\$1,200** of move per surprise.
- The \$350,000 equity sleeve at -0.70% beta contributes 0.70% x \$350,000 = **\$2,450**.
- The \$50,000 long-bond option, via the -7bp-per-surprise yield move on a 17-duration bond, contributes about 1.2% x \$50,000 = **\$600**.

Add the sensitivities (they all point the same way in this regime — that *is* the +0.55 correlation showing up in your P&L) and a single +0.1pp surprise moves the portfolio roughly **\$4,250**, well inside the \$10,000 budget. If you wanted to use the *full* budget, you could scale the risk sleeves up by about 2.3x — but you would do it knowing the correlation has stacked the bets, not pretending they offset.

#### Worked example: sizing to a beta target

You want your gold sleeve to contribute exactly \$2,000 of move per +0.1pp CPI surprise (its beta is -0.80% per surprise). Solve: position x 0.80% = \$2,000, so position = \$2,000 / 0.0080 = **\$250,000**. To hit a \$2,000 surprise-sensitivity from gold you size it at \$250,000, not by feel. If the regime later calms and gold's beta to CPI falls toward zero (the real-yield channel reasserts), the *same* \$250,000 contributes far less, so you would *increase* notional to hold the risk constant. *You size to the beta, and you re-size when the beta changes — that is the whole discipline.*

### Step 5 — Set the flip-watch

The position is built and sized; now you protect it. The flip-watch is the single input whose change would invert the live correlations and invalidate the whole position. In the stagflation regime, that input is **inflation rolling over**. The day core CPI and PPI clearly decelerate, the regime shifts toward deflation/disinflation, and the live correlations *reverse*: the stock-bond correlation falls back toward -0.55 (bonds become a hedge again), gold's real-yield headwind eases, and "bad news" (weak growth) becomes *good* for bonds. The flip-watch is concrete: when **PPI upstream** rolls (it leads core goods CPI by about a month) and **breakevens** fall, you begin rotating the long-bond option up and trimming the commodity leader. This watch is what saved the careful investor in late 2022, and it is the heart of [building a weekly correlation monitor](/blog/trading/macro-correlations/building-a-macro-correlation-monitor-the-weekly-routine).

### A second scenario: the flip-watch that saved a 60/40

Run the same loop for an investor who *did* hold a standard 60/40 into 2022 but ran the flip-watch. Their \$1,000,000 was \$600,000 stocks, \$400,000 long bonds. As inflation surprised hot all spring, the flip-watch fired: the rolling stock-bond correlation crossed from negative through zero to positive (the spine chart), signaling the hedge had failed. They cut the long-bond sleeve to short duration in May. By year-end, the long bond was down about 30% — a \$400,000 long-bond sleeve would have lost roughly **\$120,000**, of which moving to short duration salvaged the lion's share, perhaps **\$90,000**. The flip-watch did not require predicting the war or the Fed; it required noticing that the *correlation itself* had changed sign and acting on it. *The signal was not in the news; it was in the relationship.*

### A fourth scenario: the imported correlation (Vietnam)

The playbook is not US-only. An emerging market like Vietnam imports US macro correlations through two channels — the dollar and global risk appetite — and then layers its own domestic regime on top. The VN-Index has a correlation of about **+0.45** with the S&P 500 and **+0.55** with broad EM equities, so when US risk is on, Vietnam tends to rise with it. But the sharper relationship is to the **dollar**: the VN-Index correlates about **-0.40** with DXY and about **-0.25** with US 10-year yields. A surging dollar — the same 2022 dollar that hit EM everywhere — pulls foreign capital out of Vietnam (foreign net buying on the HOSE correlates about +0.50 with the index), and the currency feels it directly: USD/VND drifted from about 22.83 thousand at end-2021 to about 24.27 at end-2023 as the Fed-SBV rate differential widened and the dollar strengthened.

Then there is the *domestic* leg. Vietnam runs its own monetary regime through the State Bank of Vietnam (SBV), and the home correlations matter: the VN-Index correlates about **-0.45** with changes in the SBV policy rate and about **-0.30** with domestic CPI. In 2022 the SBV *hiked* its refinancing rate from 4.0% to 6.0% to defend the currency against the strong dollar — importing the global tightening regime — and the VN-Index fell hard. In 2023 the SBV *cut* back toward 4.5% as the dollar eased, and the domestic regime turned supportive. An investor running the playbook on Vietnam reads two regimes at once: the imported one (US dollar and risk appetite, via the same master map) and the domestic one (SBV and VN inflation). The position is sized to whichever is dominant — in 2022 the imported dollar regime overwhelmed everything; in calmer years the domestic credit cycle leads.

#### Worked example: sizing a Vietnam position to the dollar

You hold the equivalent of \$200,000 in a VN-Index ETF and want to know your dollar-regime risk. The VN-Index's correlation to DXY is about -0.40, and in a strong-dollar episode DXY can rise 10% (it rose from roughly 96 to 114 in 2022, about 19%). With a beta consistent with that correlation, a 10% DXY surge has historically pulled the VN-Index down on the order of 15-25% in dollar-stress episodes — call it -20%, or about **-\$40,000** on your position, before any domestic SBV effect. If the SBV is simultaneously hiking (the -0.45 domestic correlation firing the same direction), the two regimes stack and the drawdown deepens. *An EM position is never just a local bet; it is a local bet plus a leveraged short on dollar strength, and you size it knowing both legs.*

### A fifth scenario: the cross-asset exposure audit

A final discipline the playbook enforces: periodically audit whether your positions are *really* independent. An investor held \$300,000 of the Nasdaq, \$200,000 of Bitcoin, and \$200,000 of EM equities and believed they had three different bets. The master map says otherwise — in a liquidity-driven regime all three load on the same factor (real yields and global liquidity), with pairwise correlations of 0.5-0.7. When the Fed drained liquidity in 2022, all three fell together: the Nasdaq about -33%, Bitcoin about -64%, EM about -20%. Their \$700,000 of "diversified risk assets" was really one \$700,000 bet on liquidity, and it lost about **\$280,000** in concert. The audit — running the pairwise correlations of your actual holdings — would have revealed the concentration before the regime punished it. This is exactly the trap that [global liquidity, the everything correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation) and [crypto as a macro asset](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation) warn about.

## How to read it and use it: the playbook on one page

Here is the operating system distilled to a routine you can run every week.

**1. Locate the regime (5 minutes).** Read growth (ISM new orders, jobless claims, the yield curve) and inflation (CPI, core, PPI, breakevens). Place yourself in one of the four quadrants. You are not forecasting; you are locating.

**2. Pull the live correlations (2 minutes).** Look up the stock-bond sign, the inflation-equity sign, the gold-real-yield relationship, the DXY-EM link, and the credit-spread signal for *that* quadrant. The master matrix and the quadrant table are your cheat sheet.

**3. Stress your portfolio against them (10 minutes).** Run the pairwise correlations of your *actual* holdings, not the labels you gave them. The Nasdaq, Bitcoin, and EM equities have different names and different stories, but in a liquidity-driven regime they load on the same factor and move together — three tickets, one bet. Ask the two questions that catch every concentration mistake: are any two of my "different" bets really one bet in *this* regime? And is my hedge actually hedging, or is it a duplicate of my risk wearing the costume of protection? The dashboard that automates this is [building a correlation dashboard in Python](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python).

**4. Size to the beta (10 minutes).** For each position, compute dollar exposure times beta to the dominant driver. Set total surprise-sensitivity to your risk budget. Re-size when betas change, not when your opinion changes.

**5. Set the flip-watch (5 minutes).** Identify the one input whose change would invert the live correlations. Define the concrete trigger (e.g. "PPI upstream rolls and breakevens fall"). When it fires, rotate. The flip-watch is the difference between surviving a regime change and being run over by one.

Run this loop weekly, not daily — regimes change on the timescale of months, and watching correlations tick around intraday is how you talk yourself into noise. The point of the weekly cadence is to catch the *slow* turn, the one that took the 60/40 down in 2022 over the course of a spring, not to react to every print. And keep a written log: the regime you read, the live correlations you pulled, the trigger you set. When the flip eventually comes, you want to discover that your past self already wrote down exactly what to watch for — that is the discipline compounding. The full weekly routine, with the specific dashboards and triggers, is the operational payoff of [from correlation to signal, building a macro overlay](/blog/trading/macro-correlations/from-correlation-to-signal-building-a-macro-overlay).

The deepest lesson of the whole series is also the simplest: **a correlation is not a fact about two assets; it is a fact about the regime those assets are sitting in.** The same two assets are diversified in one quadrant and identical in another. Your job is not to memorize a number — it is to read the regime, look up the relationship that regime activates, size to it, and watch for the flip. Do that, and 2022 stops being a disaster that happened to you and becomes a regime you recognized, positioned for, and watched.

Let me leave you with the shape of the whole journey, because seeing it as one arc is the point of a capstone. You started by learning what a correlation *is* — a number between -1 and +1 with four properties, of which the fourth, the flip, is the one that matters most and the one statistics alone cannot give you. You learned how to *measure* it honestly, defeating the three traps: the full-sample number that describes no regime, the spurious correlation with no mechanism, and the non-linear relationship that a single coefficient flattens into a lie. You mapped *every* driver to *every* asset, and found that under the surface there are really only two forces — rates and growth — fighting over the discount rate, with the regime deciding the winner. You learned to locate yourself on the growth-by-inflation grid in thirty seconds, and to read off which correlations that quadrant switches on. You learned to read the canaries — the yield curve, credit spreads, jobless claims — that lead the turn, and the VIX that screams when the whole map collapses to one. And finally you learned to run the loop: regime, correlations, position, size, flip-watch. None of it required a crystal ball. All of it required the discipline to treat a correlation as the conditional, perishable, regime-bound thing it actually is. That discipline is the edge. The number on the screen is just where the edge starts.

One last way to hold all of this: the difference between an amateur and a professional in macro is not that the professional knows more correlations. It is that the amateur thinks the correlation is the answer, while the professional knows the correlation is the *question* — "under what regime was this measured, and is that regime still in force?" Carry that question into every chart, every backtest, every position, and you will have absorbed the one idea this entire series was built to deliver. The market will keep changing the regime. Your job is simply to keep reading it.

## Further reading and cross-links

This post is the hub. Here is the whole series, organized as a path through the four tracks plus the cross-asset and mechanism foundations it leans on.

**Start here — foundations (what a correlation is):**
- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the thesis of the series.
- [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta)
- [Rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters)
- [Lead, lag: leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators)
- [The surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises)
- [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data)
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the master map.

**The drivers — inflation:**
- [CPI and asset prices: the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation)
- [Core CPI, shelter, and supercore](/blog/trading/macro-correlations/core-cpi-shelter-and-supercore-what-actually-correlates)
- [PPI: the upstream inflation correlation](/blog/trading/macro-correlations/ppi-the-upstream-inflation-correlation)
- [PCE, breakevens, and the forward inflation correlation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation)
- [Inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips)
- [Inflation and gold: the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story)

**The drivers — growth and labor:**
- [NFP and asset prices: the king of data](/blog/trading/macro-correlations/nfp-and-asset-prices-the-king-of-data-correlation)
- [Unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation)
- [ISM/PMI: the leading correlation with cyclicals](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals)
- [GDP, retail sales, and the consumer correlation](/blog/trading/macro-correlations/gdp-retail-sales-and-the-consumer-correlation)
- [The yield curve as a growth signal](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation)
- [The business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock)

**The drivers — rates and bonds:**
- [Bond yields: the master correlation with every asset](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset)
- [Real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation)
- [The Fed funds path and front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation)
- [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime)
- [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary)
- [The dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation)

**The drivers — cross-asset:**
- [Oil prices, CPI, and the energy-equity correlation](/blog/trading/macro-correlations/oil-prices-cpi-and-the-energy-equity-correlation)
- [Copper-gold and the growth-inflation signal](/blog/trading/macro-correlations/copper-gold-and-the-growth-inflation-signal)
- [Crypto as a macro asset: the liquidity correlation](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation)
- [The VIX, risk-on/risk-off, and the correlation spike](/blog/trading/macro-correlations/the-vix-risk-on-risk-off-and-the-correlation-spike)
- [Global liquidity and the everything correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation)

**The regimes (where the playbook lives):**
- [Correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants)
- [When correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip)
- [Correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails)
- [Structural shifts: why today's correlations aren't yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays)

**The toolkit (measure it in Python):**
- [Building a macro-asset correlation dashboard in Python](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python)
- [Measuring beta to data surprises: an event study in Python](/blog/trading/macro-correlations/measuring-beta-to-data-surprises-an-event-study-in-python)
- [Rolling-correlation regimes and change-point detection](/blog/trading/macro-correlations/rolling-correlation-regimes-and-change-point-detection)
- [From correlation to signal: building a macro overlay](/blog/trading/macro-correlations/from-correlation-to-signal-building-a-macro-overlay)
- [Backtesting a correlation without fooling yourself](/blog/trading/macro-correlations/backtesting-a-correlation-without-fooling-yourself)

**The imported correlation — Vietnam:**
- [VN-Index and US macro: the imported correlation](/blog/trading/macro-correlations/vn-index-and-us-macro-the-imported-correlation)
- [USD/VND, DXY, and the rate-differential correlation](/blog/trading/macro-correlations/usdvnd-dxy-and-the-rate-differential-correlation)
- [VN inflation, SBV, and the domestic asset correlation](/blog/trading/macro-correlations/vn-inflation-sbv-and-the-domestic-asset-correlation)

**The playbook (put it to work):**
- [Building a macro-correlation monitor: the weekly routine](/blog/trading/macro-correlations/building-a-macro-correlation-monitor-the-weekly-routine)
- [Common correlation mistakes and how to avoid them](/blog/trading/macro-correlations/common-correlation-mistakes-and-how-to-avoid-them)

**The mechanism and cross-asset foundations this series builds on (don't re-derive — read these):**
- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable)
- [Real vs nominal: real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal)
- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map)
- [The business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders)
- [The stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine)
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis)
- [Real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything)
- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework)
