---
title: "ISM and PMI: The Leading Correlation with Cyclicals"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Business surveys like the ISM and PMI turn before GDP and before earnings. This post builds the diffusion index from zero and shows why its correlation with cyclicals, copper, small caps and yields is strong and positive, while its link to defensives is weak or negative."
tags: ["macro", "correlation", "ism", "pmi", "diffusion-index", "cyclicals", "leading-indicators", "business-cycle", "copper", "new-orders"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The ISM and S&P Global PMI are *diffusion indices*: surveys of how many businesses are getting better versus worse. Because they read the order book, they turn *before* the hard data and *before* earnings, which makes their correlation with cyclical assets (industrials, copper, small caps, yields) strongly **positive** (about +0.45 to +0.65) and their correlation with defensives weak or **negative**.
>
> - **The number to remember:** ISM new orders lead reported S&P 500 EPS growth by about **6 months** — so a rising survey is a forward read on the cyclical trade, not a confirmation of it.
> - **The 50 line is the switch.** Above 50, more firms are expanding than contracting (risk-on); below 50, the reverse (risk-off). Crossing 50 flips the cyclical-versus-defensive tilt.
> - **Direction beats level.** A reading of 48 that is *rising* is more bullish for cyclicals than a reading of 52 that is *falling*. The rate of change carries the signal.
> - **Be honest:** the survey is noisy, the lead time wobbles by regime, and in a "good-news-is-bad-news" regime a hot survey can mean *more rate hikes* and a falling stock market — the sign of the equity reaction itself can flip.

In the spring of 2020 the world's factories went dark almost overnight. By April the ISM Manufacturing PMI — a monthly survey of purchasing managers at hundreds of US industrial firms — had collapsed to **41.5**, deep below the 50 line that separates a growing economy from a shrinking one. The hard data was even uglier and still getting worse: industrial production was cratering, unemployment was spiking past 14%, and GDP was about to print the steepest quarterly contraction on record. To anyone reading the *backward-looking* numbers, the economy was in free fall and there was no bottom in sight.

But the people who watched the *survey* saw something the hard data could not yet show. By December 2020 the ISM had not merely recovered — it had rocketed to **60.5**, then to **63.7** by March 2021, one of the strongest readings in decades. The purchasing managers were telling you, in real time, that order books were refilling fast. And the assets that key off the industrial cycle moved with the survey, not with the lagging hard data: copper roughly doubled off its 2020 low, small-cap stocks ripped higher, industrial shares led the market, and the 10-year Treasury yield climbed as the bond market priced a real recovery. The survey turned first. The cyclical trade turned with it. The earnings beats that "confirmed" the recovery did not arrive until two and three quarters later.

That is the entire thesis of this post compressed into one episode. A business survey is the earliest honest read on the cycle you can get, because it asks the people placing the orders. And the assets whose fortunes ride on the cycle — the *cyclicals* — are correlated with that survey strongly and positively, while the assets people hide in when growth fails — the *defensives* — are not. If you learn to read one number well, the ISM is arguably the highest-information-per-character release in the entire macro calendar.

![Why the survey leads from the order book to the cyclical trade](/imgs/blogs/ism-pmi-the-leading-correlation-with-cyclicals-1.png)

This post sits inside the "Macro Correlations" series, where every relationship has a sign, a strength, a lead/lag, and a regime in which it flips. For the cross-correlation machinery — what it means for one series to *lead* another, and how you measure the lag — read [lead, lag, or coincident: the time axis of every correlation](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) first; this post is the deep dive on the single most important *leading* indicator in that taxonomy. For where the survey fits in the broader rotation, see [the business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock).

## Foundations: what a diffusion index actually is

Before any correlation, you have to understand the *thing* being correlated, because the ISM is not measured the way most economic data is, and that unusual construction is the source of both its power and its quirks.

Start with the everyday version. Suppose you run a delivery company with 100 drivers, and every Friday you ask each one a single question: "Is your route busier, the same, or quieter than last week?" You do not ask *how much* busier — just the direction. At the end of the day you tally the answers. Maybe 55 say busier, 30 say the same, 15 say quieter. How would you summarize the week in one number?

A diffusion index does it like this: count the share saying "better," add half the share saying "same," and ignore the rest. So the index is `55 + (30 × 0.5) = 55 + 15 = 70`. If instead 50 said better, 0 said same, and 50 said worse, the index would be `50 + 0 = 50`. **Fifty is the magic number**: it is the reading you get when exactly as many respondents are improving as are deteriorating. Above 50, the *balance* of the economy is tilting toward growth; below 50, toward contraction. The index can only ever sit between 0 (everyone worse) and 100 (everyone better).

That is the whole construction. A diffusion index measures the **breadth** of a move — *how many* firms are improving — not its **magnitude** — *how much* they are improving. This is a deliberate and clever design choice. Magnitude data (like "industrial production rose 0.4%") is precise but slow: a firm has to actually produce the output, count it, and report it weeks later. Breadth data is fast and forward-looking: a manager *knows today* whether next month looks busier, because she is the one signing the purchase orders right now.

There is one more reason the breadth design is so useful for a *correlation* signal, and it is worth spelling out. Because the index is bounded between 0 and 100 and pivots on a fixed reference (50), it is **stationary** — it does not wander off to ever-higher levels the way a price or a GDP level does. A series that trends forever is treacherous to correlate: two series that both rise over decades will show a high correlation that means nothing (the classic spurious-correlation trap covered in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data)). A diffusion index oscillates around 50 instead, so when you correlate it with an asset's *returns* (which also oscillate around zero), you are comparing two well-behaved, mean-reverting series. That is a big part of why the ISM produces such clean, stable correlations with cyclical assets while a raw level series would not.

#### Worked example: computing a diffusion index from raw answers

Make it fully concrete with a tiny survey. Say 200 purchasing managers respond about new orders this month: 96 report orders *up*, 64 report them *the same*, and 40 report them *down*. Convert to shares: up = `96 / 200 = 48%`, same = `64 / 200 = 32%`, down = `40 / 200 = 20%`. The diffusion index is `up% + 0.5 × same% = 48 + (0.5 × 32) = 48 + 16 = 64.0`. A reading of 64 — well above 50 — says the *breadth* of improvement is strong: nearly half of all firms are growing and only a fifth are shrinking. Now suppose next month the "up" camp shrinks to 70 and the "down" camp grows to 66, same = 64: up = 35%, down = 33%, same = 32%, index = `35 + 16 = 51.0`. The economy is *still* expanding (above 50) — but the index fell 13 points, and *that fall* is the tradeable signal, not the still-positive level. The intuition: the absolute level tells you the regime, but the month-over-month change is what moves cyclical assets, because markets price the change in conditions, not the level of conditions.

### The two surveys you will hear about

In practice there are two families of these surveys for the US, and a global ecosystem beyond.

- **The ISM reports.** The Institute for Supply Management publishes two headline diffusion indices on the first business days of each month: the **Manufacturing PMI** (released on the 1st business day) and the **Services PMI** (the 3rd). These are the oldest and most market-moving. The manufacturing report is the one traders quote when they say "the ISM" without qualification, even though manufacturing is now a small slice of the economy — more on that tension below.
- **The S&P Global PMI** (formerly Markit). A parallel set of surveys, released a touch earlier as "flash" estimates mid-month, covering manufacturing and services across the US, the eurozone, the UK, Japan, and dozens of other economies. Because it is global and early, traders use it as a first read before the ISM lands. The methodology differs slightly but the spirit is identical: a 50-line diffusion index.

Both report a **headline index** and a set of **sub-indices**: new orders, production, employment, supplier deliveries, prices paid, backlog, inventories. The headline is a weighted blend of several sub-indices. For our purposes, one sub-index towers above the rest, and we will return to it constantly: **new orders**.

### The global PMI ecosystem and why it matters

It is worth zooming out, because the cyclical signal is not just a US story. The S&P Global PMI publishes a manufacturing diffusion index for nearly every major economy, plus regional composites: a **eurozone PMI**, a **China PMI** (alongside China's own official NBS PMI), a UK PMI, a Japan PMI, and a **global manufacturing PMI** that aggregates them. Because global industrial demand is what drives globally-traded commodities, the *global* manufacturing PMI is often a better correlate of **copper** and **oil** than any single country's survey — copper does not care whether the new orders are in Ohio or Guangdong, only that someone, somewhere, is building. When traders talk about "the global cycle turning," they very often mean the global manufacturing PMI crossing 50.

This also gives you a timing edge. The S&P Global *flash* PMIs land mid-month, roughly two weeks *before* the ISM's first-of-the-month print, and the eurozone and Asia flashes land before the US one (time zones help). So a disciplined reader gets a *preview* of the cycle's direction from the flash and the overseas surveys before the headline ISM confirms it. The surveys form a small relay: overseas flash → US flash → ISM → hard data → earnings. Each link leads the next, and the whole relay leads the cyclical trade.

### Beyond new orders: the other sub-indices and what they distort

New orders is the star, but a professional reads the supporting cast, because the headline can lie about *why* it moved.

- **Supplier deliveries** measures how long it takes suppliers to deliver — and counterintuitively, *slower* deliveries *push the headline up*, on the logic that suppliers are slow because demand is overwhelming them. That logic broke spectacularly in 2021–22, when deliveries were slow because of *supply-chain breakage*, not booming demand. A headline propped up by lengthening deliveries is a weak signal; strip it out and look at new orders.
- **Prices paid** tracks input cost inflation. It is a useful inflation read (it can lead [PPI](/blog/trading/macro-correlations/ppi-the-upstream-inflation-correlation) and goods CPI), but it has *nothing* to do with the breadth of real demand, and a spike in prices-paid driven by, say, an oil shock can move the narrative without moving the cyclical signal. Keep the growth read (new orders, production) separate from the inflation read (prices paid).
- **Employment** is the laggiest sub-index — firms hire after demand firms up and fire after it fades — so it confirms rather than leads.
- **Backlog of orders** is an underrated forward gauge: a rising backlog means firms have *committed future work* they have not yet produced, which is genuinely leading. New orders plus backlog together are the cleanest forward read in the report.

The takeaway: when the headline surprises, do not trade it blind. Decompose it. A headline up on new orders and backlog is a real cyclical green light; a headline up on slow deliveries and high prices paid is a supply-and-cost story dressed up as a demand story.

### What "correlation" means here, in one paragraph

Across this series, a correlation has three coordinates: a **sign** (do the two things move together or oppositely?), a **strength** (a number between −1 and +1 — Pearson's r — where ±1 is a perfect line and 0 is no linear relationship), and a **lead/lag** (does one move first?). When this post says "the ISM has a +0.6 correlation with copper," it means: in months when the survey is rising, copper has tended to rise too, and the relationship is strong but far from perfect. When it says copper has a "beta" to the ISM, it means how *many* percent copper tends to move per unit of survey change. Sign, strength, lead — keep all three in mind, because the ISM's signature feature is the third one. (If any of this is unfamiliar, [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) builds Pearson, Spearman and beta from zero.)

## Why surveys lead the hard data

Here is the mechanical reason the ISM turns before GDP, before industrial production, before earnings — and it is not magic, it is the order of operations in a real economy.

Think about the life of a single widget. First a manager *decides* to buy materials and place an order with a supplier (that decision is captured by **new orders** in the survey, today). Weeks later the supplier ships and the factory *produces* the widget (that shows up in **industrial production**, a hard number, later). The widget is *sold* and *booked as revenue*, and eventually that revenue shows up in **GDP** and in the company's **reported earnings** (later still). The chain runs: decision → production → revenue → reported profit. Each link happens strictly after the one before it.

A survey samples the *first* link. The hard data samples the middle. Earnings sit at the very *end*. So when you watch the order survey, you are watching the leading edge of a wave that will roll through production a few weeks later, through GDP a quarter or two after that, and through reported earnings about six months out. The survey does not predict the future in some mystical sense — it simply observes an *earlier* point in a causal chain that everything else lies downstream of.

This is why the formal indicator taxonomy puts surveys squarely in the **leading** bucket. The Conference Board's Leading Economic Index — a composite designed to turn before the cycle — explicitly includes ISM new orders for exactly this reason. By contrast, the unemployment rate, corporate earnings, and the final GDP print are **lagging**: they confirm a turn the economy already made. (For the full three-class taxonomy and the cross-correlation function that measures the lead, see [lead, lag, or coincident](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators).)

### New orders: the leading part of a leading indicator

Even within the survey, not all sub-indices are equally forward-looking. The headline blends new orders, production, employment, supplier deliveries, and inventories. But **production** is already half-coincident — by the time a factory is producing, the decision was made weeks ago. **Employment** is laggier still. The sub-index that captures the earliest, most decision-shaped link is **new orders**, because an order is a commitment to *future* output. So new orders is the leading part of an already-leading indicator. When professionals dissect an ISM print, the very first thing they look at after the headline is whether new orders is above 50 and which way it moved. A headline that ticks up but is driven by lengthening supplier deliveries (a supply bottleneck, not real demand) is a weaker signal than one driven by new orders.

The lead time is concrete and measurable. Slide the new-orders series forward in time against reported S&P 500 earnings growth and ask at which shift the two line up best — the cross-correlation peaks at roughly a **six-month lead**. That is the single most actionable fact in this post: the survey is telling you today what the earnings line will be doing about two quarters from now.

![Lead time how many months each indicator leads its target](/imgs/blogs/ism-pmi-the-leading-correlation-with-cyclicals-4.png)

The chart sorts a handful of documented leads. The inverted yield curve leads recessions by the longest stretch (peak cross-correlation near 14 months), building permits lead GDP by about 9, and our protagonist — ISM new orders — leads S&P EPS growth by about 6. Notice how *short* some leads are: credit spreads lead equity drawdowns by only about 3 months, and CPI essentially moves *with* PCE (a ~0 lead). A long lead gives you time to position; a short one barely gives you time to react. The ISM's six-month lead is the sweet spot: long enough to act on, short enough to be reliable.

#### Worked example: turning the six-month lead into a timing rule

Suppose it is the end of a quarter and ISM new orders prints **54.0**, up from **49.0** three months ago — a clean cross back above 50 and rising. The six-month lead says reported S&P 500 EPS growth should be inflecting *upward* roughly two quarters out. So if you are positioning a \$100,000 cyclical sleeve, the rule is: **add the cyclical beta now, on the survey, not later on the earnings beat.** If you instead wait for the earnings confirmation, you are buying six months late — and by then the cyclicals have, on average, already done much of their move. Concretely, if industrials historically run up about 15% in the two quarters between a new-orders inflection and the earnings confirmation, waiting to "be sure" costs you roughly \$15,000 of that \$100,000 sleeve's move. The intuition: in a leading indicator, the confirmation is the part you pay for by being late.

## The centerpiece: the correlation with cyclical assets

Now the heart of the matter. If the survey reads the cycle early, then assets whose returns ride on the cycle should be correlated with the survey — and the data says they are, strongly and positively.

![Correlation of each asset with a rising ISM PMI](/imgs/blogs/ism-pmi-the-leading-correlation-with-cyclicals-2.png)

Read this chart top to bottom and a clean story emerges. The assets most positively correlated with a rising ISM are exactly the **cyclicals**: industrial-sector stocks (about +0.65), copper (+0.60), the broad S&P 500 (+0.55), small caps (+0.55), and the 10-year Treasury yield (+0.45, meaning yields *rise* — bond *prices* fall — as growth firms). At the bottom, the assets people own *instead* of cyclicals when they fear a slowdown — gold (about −0.05, essentially uncorrelated) and consumer-staples defensives (−0.10) — barely respond, and **long Treasuries** are the mirror image at about −0.45: when growth accelerates, the long bond is the asset you least want to hold.

Let me define the two camps precisely, because the whole rotation hinges on them.

- **Cyclicals** are companies and commodities whose revenue swings hard with the economy: industrials, materials, energy, semiconductors, autos, banks (loan demand and credit losses track the cycle), and broadly *small caps* (smaller firms have more operating and financial leverage, so they amplify the cycle). **Copper** is the commodity poster child — its nickname "Dr. Copper" exists because it goes into everything that gets built, so its demand is a near-pure read on global industrial activity.
- **Defensives** are companies whose revenue barely moves with the cycle because they sell things people buy in any economy: consumer staples (toothpaste, food), utilities (you pay the electric bill in a recession too), and much of healthcare. **Gold** and **long-duration Treasuries** are the "hide here when growth fails" assets — they tend to do *well* when the cycle rolls over, which is precisely why their ISM correlation is flat-to-negative.

So the ISM is not just "an indicator that correlates with stocks." It is a *cross-sectional* signal: it tells you which *part* of the market to own. A rising survey says own the cyclicals; a falling survey says own the defensives. The headline index level barely moves the *whole* market — it moves the *rotation within* it.

### Why each asset has the beta it does

The correlations are not arbitrary; each one traces back to a mechanism. Understanding the mechanism is what lets you trust the correlation when it is working and recognize when it is about to break.

- **Industrials and materials (+0.65, +0.60 via copper).** These firms' revenue *is* the cycle. An industrial company sells capital equipment, machinery, freight, and basic materials — demand that swings hard with how much the economy is building and producing. When the survey says order books are filling, these are the firms whose order books are filling. The correlation is high because the link is nearly mechanical.
- **Small caps (+0.55).** Smaller companies carry more **operating leverage** (a higher share of fixed costs, so profits swing more for a given revenue change) and more **financial leverage** (more debt relative to equity, so they are more sensitive to the cycle and to credit conditions). Both amplify the cycle: a small-cap index is, roughly, a high-beta version of the broad market's cyclical exposure. That is why small caps both outperform hardest in early-cycle rallies (as in 2020–21) and underperform hardest when the survey rolls over.
- **The S&P 500 (+0.55).** The broad index is a *blend* of cyclicals and defensives, so its correlation sits between the two camps. The positive sign reflects that, on net, the index has meaningful cyclical content — but this is exactly the correlation most vulnerable to the "good-news-is-bad-news" sign flip, because the broad index is also the most rate-sensitive aggregate.
- **The 10-year yield (+0.45).** This one trips people up, so it deserves care: a *rising* yield means a *falling* bond price, so the +0.45 says **bond prices fall as the survey rises.** The mechanism: when the cycle accelerates, the bond market prices stronger growth and (usually) higher future inflation and central-bank rates, all of which push yields up. So a strong survey is *bad* for long bonds — which is the same statement as the long-Treasury correlation being −0.45. Growth firming and the long bond selling off are two sides of one coin. (For the master mechanism, [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) is the reference.)

#### Worked example: the yield leg of the cyclical trade

Suppose the ISM crosses decisively from 47 to 54 over two months and the bond market reprices the growth outlook. A move of that size has historically coincided with the 10-year yield rising on the order of 40–60 basis points. Put numbers on the bond side: a 10-year Treasury has a duration near 8, so a +50 bp yield move means a price loss of roughly `8 × 0.50% = 4%`. On a \$250,000 long-Treasury position, that is a loss of about **\$10,000** — and the *same* growth acceleration is simultaneously making your cyclical equity sleeve money. This is why the rotation is symmetric and powerful: a rising survey pays you on the cyclical long *and* on a short-duration (or long-cyclical-versus-long-bond) stance at the same time, because both legs key off the one signal. The intuition: the ISM does not just rank stocks against each other — it ranks the whole risk spectrum, from long bonds at the bottom to industrials and copper at the top.

#### Worked example: copper's beta to the survey

Copper carries a +0.60 correlation with the ISM and a high *beta* — it moves a lot per unit of survey change. Suppose the ISM jumps from 48 to 53 over a quarter, a decisive cross above 50. Historically a move of that size has coincided with copper rallying on the order of 12–15% over the following months as the market prices a genuine industrial upswing. On a \$50,000 copper position (via futures or a miner ETF), a 13% move is about **\$6,500** of P&L. Now flip it: if the survey instead slides from 53 to 48, copper has historically given much of that back. The intuition: copper is a near-pure bet on the same global industrial demand the survey measures, so it is one of the highest-beta expressions of the ISM signal — which is exactly why it is also one of the most painful to hold when the survey rolls over.

### The ISM's row in the master correlation matrix

It helps to see the survey's signature inside the bigger map of "every driver versus every asset." In [the macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix), each macro driver gets a row of correlations across the major asset classes. The "ISM/PMI rise" row reads, roughly: S&P 500 **+0.55**, Nasdaq **+0.50**, US 10-year *bond* **−0.25** (bonds fall as growth firms — the same statement as yields rising), gold **0.00** (flat — gold does not key off the growth cycle, it keys off real yields and the dollar), Bitcoin **+0.40** (a high-beta risk asset that rides risk-on), EM equity **+0.55** (emerging markets are heavily cyclical and commodity-linked), and the US dollar **−0.10** (a stronger cycle is mildly dollar-negative as capital rotates into risk).

The shape of that row *is* the thesis: positive for risk and cyclicals, near zero for gold, negative for bonds. Compare it to the "real yield rise" row, which is negative almost everywhere (including a brutal −0.80 on gold): real yields are a *discount-rate* shock that hurts nearly everything, while the ISM is a *growth* signal that helps the risk side and hurts the safe side. Two drivers, two completely different correlation fingerprints — and reading the fingerprint tells you which driver is in charge of the tape on any given day.

### The 50 line is a switch, and direction beats level

Two refinements separate amateurs from professionals in reading this number.

First, **the 50 line is a regime switch, not a smooth dial.** The correlation with cyclicals is asymmetric around it. When the index is comfortably above 50 and climbing, cyclicals are in their element and the positive correlation is at its strongest. When the index falls below 50, the *defensive* trade takes over and the cyclical correlation can even invert in the short run as the market rotates out. Crossing 50 — in either direction — is the moment the rotation flips. That is why the financial press makes such a fuss about a "sub-50 print": it is not just a slightly lower number, it is a change of regime.

Second, and more subtly: **the rate of change matters more than the level.** Markets price the *second derivative* — not where the survey is, but where it is going. A reading of **48 that is rising** (the economy is still contracting, but less than last month — the trough is forming) is *more bullish* for cyclicals than a reading of **52 that is falling** (still expanding, but decelerating — the peak is in). This is one of the most counterintuitive truths in cyclical trading: the best returns in cyclicals historically come not when the survey is highest, but when it is **lowest and turning up**. By the time the survey prints a strong 58, the cyclical move is mature and you are late.

![Reading a diffusion print level on one axis direction on the other](/imgs/blogs/ism-pmi-the-leading-correlation-with-cyclicals-5.png)

The matrix lays out the four states you can be in. The **best quadrant** for cyclicals is above 50 *and rising*. The **worst** is below 50 *and falling* — contraction deepening, time to own defensives and long Treasuries. But the two off-diagonal cells are where the money is made: **below 50 but rising** is the early-cycle buy (the trough), and **above 50 but falling** is the warning to trim (the peak rolling over). Read both axes — level *and* direction — and you have most of what the survey can tell you.

## The path through 50: a decade in one chart

Theory is cleaner with the actual history in front of you. Here is the ISM Manufacturing PMI walking through the 50 line over 2020–2025.

![ISM Manufacturing PMI through the 50 line 2020 to 2025](/imgs/blogs/ism-pmi-the-leading-correlation-with-cyclicals-3.png)

Three eras stand out, and each is a textbook correlation case.

1. **The 2020 collapse and V-recovery (41.5 → 63.7).** The survey cratered in April 2020, then snapped back above 50 and surged to a multi-decade high of 63.7 by March 2021. The cyclical trade tracked it tick for tick: copper, small caps, industrials and yields all ripped. This is the cleanest demonstration of the positive correlation in the dataset — survey up, cyclicals up.
2. **The 2021–2022 rollover (63.7 → 53.0 → 48.4).** From the peak, the survey ground steadily lower through 2021 and finally crossed *below* 50 in late 2022. Watch the *direction* signal here: the survey was still above 50 (still "expansion") for most of 2022, but it was *falling* the whole time — the "peak rolling over" quadrant. The cyclical correlation was already working against you well before the sub-50 print.
3. **The 2022–2025 contraction stretch.** The index spent an unusually long run *below* 50 — manufacturing was in a rolling soft patch even as the broader economy (and especially services) kept growing. This is a crucial real-world caveat we will unpack: manufacturing is a shrinking slice of a services-dominated economy, so a sub-50 *manufacturing* print did not mean recession.

That last point is important enough to make its own section. A naive reader would have seen the ISM stuck below 50 from late 2022 onward and shorted everything. The economy kept growing anyway. Why?

### Manufacturing versus services in a services economy

Manufacturing is now roughly **10–12% of US GDP**; services are the rest. So a contracting *manufacturing* survey, by itself, is a read on a small (if highly cyclical and bellwether-ish) corner of the economy. The **Services PMI** covers the dominant slice, and through 2023–2024 the services survey stayed comfortably above 50 even while manufacturing languished below it. An investor who only watched manufacturing badly misjudged the overall cycle.

So why does the manufacturing ISM still get the headlines and still move markets? Two reasons. First, **manufacturing is the most cyclical part of the economy** — it amplifies the cycle, turning earlier and harder than services, which makes it a sensitive early-warning gauge even when it is small. Second, **history and habit**: the manufacturing series is the longest-running and the one most embedded in trading models and the Conference Board's LEI. The practical lesson: read *both*. Use the manufacturing survey (and its new-orders sub-index) as your sensitive cyclical thermometer, but cross-check the services survey before you conclude anything about the *whole* economy. A manufacturing-only recession (as in 2022–24) is real but partial, and it correlates with a *narrow* cyclical underperformance, not a broad bear market.

There is a deeper reason manufacturing keeps its bellwether status despite its small GDP share: it is the **most globally-traded, inventory-intensive, capital-cycle-driven** part of the economy. Services like a haircut or a restaurant meal are consumed locally and roughly when produced, so they barely cycle. A factory, by contrast, builds to a forecast, holds inventory, ships across borders, and makes lumpy capital-equipment decisions — all of which means its activity *swings* far more than its size suggests and *turns earlier* than the consumer does. The amplification is the point: a 5-point move in the manufacturing survey reflects a sharper turn in the cyclically-sensitive economy than the same 5-point move in services would. So manufacturing remains the canary not because it is large but because it is *sensitive* — it is the part of the economy where the cycle shows up first and hardest, which is exactly what you want from an early-warning correlate of cyclical assets.

#### Worked example: a cyclical-versus-defensive rotation P&L on an ISM turn

This is the trade the whole post is building toward, so let us put dollars on it. You manage a \$1,000,000 equity book and you run it market-neutral on the cyclical signal: long cyclicals, short defensives, in equal dollar amounts. The ISM new-orders sub-index has just crossed from 47 up to 51 — below-50-but-rising tipping into above-50-and-rising, the early-cycle buy.

You put **\$500,000 long industrials/materials (cyclicals)** and **\$500,000 short consumer staples (defensives)**. Over the next two quarters, as the survey keeps climbing and the cyclical correlation does its work, suppose cyclicals return **+14%** and defensives return **+2%** (defensives still rise in an up market, just far less). Your long leg makes `0.14 × \$500,000 = \$70,000`. Your short leg *loses* `0.02 × \$500,000 = \$10,000` (you are short something that rose). Net P&L: **\$70,000 − \$10,000 = \$60,000**, or **+6%** on the \$1,000,000 book — earned not by calling the market up, but by calling the *rotation* correctly off the survey.

Now the mirror trade. The survey rolls from 53 *down* to 49 — above-50-falling tipping into below-50-falling. You flip: short cyclicals, long defensives. If cyclicals now *fall* 12% and defensives are flat, the long-defensive/short-cyclical book makes about `0.12 × \$500,000 = \$60,000` on the short leg and roughly breaks even on the long leg. The intuition: the ISM does not just tell you whether to be in the market — it tells you which *half* of the market to own, and the spread between the two halves is the tradeable edge.

![The cyclical defensive rotation flips with the ISM direction](/imgs/blogs/ism-pmi-the-leading-correlation-with-cyclicals-6.png)

The figure makes the rotation literal. The green bars are how each asset correlates with a *rising* ISM (risk-on); the red bars are the *falling*-ISM regime, where the signs flip. Cyclicals, small caps and copper love a rising survey and hate a falling one; long Treasuries are the exact opposite — the asset you rotate *into* when the survey turns down. Defensives sit near zero in both regimes, which is the whole point of owning them: they are the low-cyclical-beta ballast.

## The survey and the investment clock

Step back and the ISM is really a hand on the *cycle clock*. The classic "investment clock" framework sorts the economy into four phases and assigns a winning asset class to each, and the survey's level-and-direction tells you which phase you are in.

- **Early / recovery** (survey below 50 but rising — the trough). Stocks lead hard; in representative cycle data, equities have returned around +20% real in this phase, commodities +8%, while cash loses to inflation. This is the cyclicals' moment: small caps and industrials lead the bounce because the market is pricing the recovery the survey just flagged. 2020 H2 is the archetype.
- **Mid / expansion** (survey above 50 and stable-to-rising). Growth is confirmed and broad; equities still win (around +12% real) but the leadership broadens and the easy cyclical gains are behind you. Bonds do little.
- **Late / overheat** (survey above 50 but falling — the peak rolling over). This is where *commodities* lead (around +14% real in representative data) as capacity tightens and inflation runs hot, while bonds start to lose. The cyclical-equity trade gets dangerous here: the survey is still "good" on a level basis but the *direction* has turned, and that is the signal to trim. 2021 is the archetype.
- **Recession** (survey below 50 and falling — contraction deepening). *Bonds and cash* win (government bonds around +10% real) while stocks and commodities fall hard. This is the defensive quadrant: long Treasuries, staples, utilities. The survey's sub-50-and-falling reading is the clock striking this hour.

The mapping is not a coincidence — it is the same causal chain seen from a different angle. The survey reads the cycle first, the cycle determines which assets win, so the survey *is* the early read on the asset rotation. (For the full rotation framework, see [the business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) and the within-series [business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock).) The reason professionals obsess over the *direction* of the survey rather than its level is precisely that the clock turns on the direction: the move from "rising" to "falling" is the move from late-cycle (own commodities, trim cyclicals) to recession (own bonds), and missing that turn is how cyclical investors give back a whole cycle's gains in a few quarters.

## How it shows up in real markets

Three dated episodes, each a clean correlation case, two confirming and one cautionary.

### 2020: the V-recovery and the cyclical rally

We have referenced it throughout; here it is as a case study. From the April 2020 trough at 41.5, the ISM crossed back above 50 by June and reached 60.5 by December, 63.7 by March 2021. The cyclical complex tracked the survey with almost embarrassing fidelity: copper rallied from roughly \$2.10/lb in March 2020 to over \$4.20/lb by 2021 (a double); the small-cap Russell 2000 outran the S&P by a wide margin into early 2021; energy and industrials led every sector; and the 10-year yield climbed from 0.5% to over 1.7% as the bond market priced the recovery the survey had already flagged. An investor who bought the cyclical rotation on the *survey's* cross above 50 — rather than waiting for the GDP and earnings confirmation that came two and three quarters later — captured nearly the whole move.

The timing detail is the lesson. The survey crossed back above 50 in June 2020. Reported S&P 500 earnings did not actually bottom and inflect higher until late 2020, and the *strong* earnings beats that "proved" the recovery did not arrive until the first and second quarters of 2021 — roughly the six-month lead, as advertised. An investor who insisted on waiting for the earnings confirmation bought cyclicals near the *top* of the early-cycle move, after copper had already doubled and small caps had already led. The survey did not merely correlate with the cyclical trade; it *front-ran* it by about two quarters, which is precisely the window the new-orders-to-EPS lead predicts. This is the single best argument for trading the leading indicator: in 2020 the confirmation was the expensive part.

#### Worked example: reading a live print in real time

Walk through the discipline on a single hypothetical release. The headline ISM prints **52.3**, up from **50.8** last month — above 50 and rising, so far so bullish. But you do not stop at the headline. You open new orders: **54.5**, up from 51.0 — strong and accelerating, a real demand green light. You check supplier deliveries: roughly flat, so the headline is *not* being propped up by bottlenecks. You check prices paid: elevated but stable, so this is a demand story, not a fresh cost shock. Finally the services PMI: **53.0** and steady, so the move is broad, not a narrow manufacturing blip. Verdict: this is a clean above-50-and-rising, demand-led, broad print — add cyclical beta. Now size it: on a \$200,000 risk budget for the rotation, you might put \$200,000 into a long-cyclicals/short-defensives spread, expecting the cyclical half to outperform the defensive half by perhaps 8–12 percentage points over the next two quarters, for a target of roughly **\$16,000 to \$24,000**. The intuition: the edge is not in seeing the headline — everyone sees that — it is in decomposing it into new orders, deliveries, prices and services before you commit, so you are not fooled by a headline that moved for the wrong reason.

### 2021–2022: the rollover before the cyclical underperformance

The more instructive case, because it is the one that pays attention to *direction*. Through 2021 the ISM was still high — above 55 — but it was *falling* from the 63.7 peak. The "above-50-but-falling" quadrant. A reader fixated on the *level* would have seen a number in the high 50s and stayed long cyclicals. A reader watching the *rate of change* would have seen the survey decelerating for a year and trimmed. And the cyclical/defensive spread bore the second reader out: through 2022, energy (the most cyclical sector) was the only winner (+65.7%), but the high-beta cyclicals tied to the *consumer* and to *rates* were crushed — consumer discretionary −37.0%, technology −28.2%, materials −12.3% — while the defensives held up far better — consumer staples −0.6%, utilities +1.6%, healthcare −2.0%. The rotation out of cyclicals and into defensives that the falling survey foreshadowed was exactly the trade that worked in 2022.

Notice how the two failure modes of a naive reader compound here. The level-watcher stayed long the wrong thing. The investor who *also* ignored the regime — who assumed "the economy is still expanding, so risk-on" — got the additional 2022 surprise that this was a *good-news-is-bad-news* tape: the Fed was hiking aggressively into the inflation it had let run, so even decent growth data could not save the rate-sensitive cyclicals. The disciplined ISM reader got two things right that the naive one missed: the *direction* of the survey said trim cyclicals, and the *regime check* said the equity sign had flipped. Both pointed the same way — toward defensives and into the long bond once the survey crossed under 50 in late 2022 — and that combination was one of the cleaner macro trades of the decade. The cyclical correlation did not break in 2021–22; it worked perfectly, in the *negative* direction, for anyone reading the rate of change rather than the level.

It is also worth naming the long pedigree behind why any of this is trustworthy. The ISM manufacturing survey has been running since 1948, through every post-war business cycle, which is exactly why it earns a place in the Conference Board's Leading Economic Index and in the trading models that key off the cycle. A correlation backed by seventy-plus years of cycles and an unbroken causal chain (orders lead production lead revenue lead profit) is a far sturdier thing to trade than a relationship that merely showed up in a five-year backtest. That pedigree does not make it infallible — nothing in macro is — but it is the difference between a leading indicator with a mechanism and a statistical coincidence dressed up as a forecast.

### 2022–2024: the sub-50 print that did *not* signal recession

The cautionary tale. The manufacturing ISM crossed below 50 in late 2022 and stayed there, off and on, for roughly two years — the longest sub-50 stretch in a decade. A mechanical reading said "contraction → recession → sell." But no recession came: GDP grew about 2.9% in 2023 and 2.8% in 2024, and the broad equity market rose strongly. The signal "failed" — but it failed for a *knowable* reason. It was a *manufacturing* contraction in a *services* economy: the services PMI stayed above 50 the whole time, and the parts of the market tied to services and to AI-driven tech boomed. The lesson is not "the ISM is useless"; it is "the ISM manufacturing survey correlates with the *cyclical, manufacturing-sensitive* slice of the market, and you must check the services survey and the breadth of the contraction before extrapolating to the whole economy." A correlation is a regime, not a constant — and the regime here was "narrow industrial soft patch," not "broad recession."

## When the correlation breaks

A correlation is a regime, not a constant, and the ISM-cyclicals relationship has three documented failure modes. Knowing them is what separates a signal you can size a position on from one that will blow up your book at the worst moment.

**Break 1 — the narrow-contraction divergence.** This is the 2022–24 case in full. The manufacturing survey can fall below 50 while the *economy* keeps growing, because manufacturing is only ~10–12% of GDP and services can power on independently. In that regime, the manufacturing ISM correlates only with the *manufacturing-sensitive* slice of the market (industrials, materials, goods-makers), not with the broad index, which was being carried by services and mega-cap tech. The fix is breadth: confirm the manufacturing signal against the services PMI and the breadth of the sub-indices before extrapolating to the whole market. A divergence between the two surveys is itself information — it says "this is a sector story, trade it as one."

**Break 2 — the good-news-is-bad-news sign flip.** When the market's dominant worry switches from *growth* to *inflation and rates*, the equity reaction to a strong survey can invert. In 2022–23, a hot ISM meant the economy was running too warm for the Fed's comfort, which meant *more* tightening, which the rate-sensitive parts of the equity market took as bad news. Crucially, even in this regime the *cross-sectional* rotation usually survives — cyclicals still beat defensives on a relative basis — but the *directional* call (strong survey → market up) can be flat-out wrong. This is why expressing the ISM as a long-cyclicals/short-defensives *spread* is more robust than expressing it as outright long the index. (The mechanism behind the inflation-regime flip is the subject of [inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips).)

**Break 3 — the correlation-to-one shock.** When a large non-cyclical shock hits — a banking scare, a sovereign crisis, a geopolitical spike, a pandemic — *every* risk asset falls together regardless of the survey, and diversification fails exactly when you need it. In those windows the ISM signal is drowned out: a great survey print means nothing if the market is in a liquidity scramble. This is the general truth that [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) is built around. The defense is not in the signal — it is in the risk management: size positions so a correlation-to-one episode is survivable, because the survey will not warn you about a shock that originates outside the real economy.

The common thread across all three breaks: the ISM is a clean read on the *industrial cycle*, and it correlates with cyclicals *to the extent that the industrial cycle is what is driving markets*. When something else takes the wheel — a services-led economy, an inflation-fear regime, a financial shock — the survey keeps measuring its slice faithfully, but its slice is no longer the thing that matters. A good correlation user does not abandon the signal; they ask, every time, "is the industrial cycle in charge of the tape right now?" and weight the signal accordingly.

## Common misconceptions

Five myths, each corrected with a number.

**Myth 1: "The ISM measures how fast the economy is growing."** No — it measures *breadth*, not *speed*. A diffusion index counts *how many* firms are improving, not *how much*. You can have a reading of 55 (a solid majority improving) with very modest actual output growth, or a 52 with a few firms booming. The number tells you the *direction and breadth* of the cycle, which is exactly what you want for a leading rotation signal, but it is not a GDP nowcast. Treat the level as "majority expanding/contracting," not "GDP is X%."

**Myth 2: "A reading of 48 is bearish, full stop."** Only if it is *falling*. A 48 that rose from 45 is the early-cycle buy — the trough forming — and historically precedes some of the best cyclical returns. The rate of change carries the signal. A static 48 tells you little; a 48-and-rising and a 48-and-falling are nearly opposite trades.

**Myth 3: "Strong survey is always bullish for stocks."** Not in a "good-news-is-bad-news" regime. In 2022–23, when the market's reaction function keyed on inflation and rate hikes, a *strong* survey meant the economy was running hot, which meant *more Fed tightening*, which the equity market took as a *negative*. The ISM-versus-cyclicals correlation is robust (cyclicals still beat defensives), but the ISM-versus-*broad-index* sign can flip depending on whether the market is worried about growth or about rates. (This is the same "regime flip" mechanism explored in [inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips).)

**Myth 4: "The manufacturing ISM tells you about the whole economy."** Manufacturing is only ~10–12% of GDP. The 2022–24 sub-50 stretch did *not* signal recession precisely because services — the other ~88% — kept expanding. Always cross-check the services survey before extrapolating from a manufacturing print.

**Myth 5: "The ISM is a precise, clean signal."** It is noisy, it is revised, the lead time wobbles by regime, and the "prices paid" sub-index can drag the headline around for reasons unrelated to real demand. The six-month lead to earnings is an *average* peak, not a metronome — in some cycles it is four months, in others nine. Use it as a high-information *prior*, confirmed by new orders and the services survey, not as a deterministic clock.

## How to read it and use it

Here is the playbook, distilled. The figure below is the decision flow; the bullets are the reasoning.

![The ISM playbook from one print to a portfolio tilt](/imgs/blogs/ism-pmi-the-leading-correlation-with-cyclicals-7.png)

**Step 1 — read the level *and* the direction.** Where is the headline versus 50, and which way did it move from last month? Plot it on the two-axis matrix: above/below 50 × rising/falling. The off-diagonal cells (below-50-rising, above-50-falling) are the high-value turning-point signals.

**Step 2 — look straight at new orders.** The headline can be flattered by lengthening supplier deliveries (a bottleneck, not demand) or distorted by inventories. New orders is the cleanest forward-looking sub-index and the one with the documented six-month lead to earnings. If the headline and new orders disagree, trust new orders for the *forward* read.

**Step 3 — set the cyclical tilt.** Rising survey → add cyclical beta (industrials, materials, copper, small caps), lighten long Treasuries. Falling survey → rotate toward defensives (staples, utilities, healthcare) and long Treasuries, cut copper. The trade is the *spread* between the two halves of the market, not a directional bet on the index.

**Step 4 — run the regime check.** Before you assume "strong survey = stocks up," ask what the market is worried about. In a growth-fear regime, a strong survey is unambiguously risk-on. In an inflation/rate-fear regime ("good news is bad news"), a strong survey can mean more hikes and a *falling* broad index even as cyclicals still beat defensives. The cross-sectional rotation signal is more robust than the directional one. (For the intraday mechanics of how a survey *release* hits each market, see the event-trading companion, [ISM and PMI: the business surveys that lead](/blog/trading/event-trading/ism-pmi-the-business-surveys-that-lead).)

**Step 5 — cross-check breadth and services.** Is the manufacturing move broad (many sub-indices and the services survey agreeing) or narrow (manufacturing-only, as in 2022–24)? A broad turn correlates with the whole cycle; a narrow one only with the manufacturing-sensitive cyclicals.

**What invalidates the signal.** The cyclical correlation breaks down when (a) the move is a narrow manufacturing-only soft patch while services power on, (b) the market is in a "good-news-is-bad" regime so the equity *sign* flips, or (c) a non-cyclical shock (a banking scare, a geopolitical spike) dominates and drives correlations toward one, drowning out the survey signal. In all three, the *rotation* (cyclicals vs defensives) usually still holds even when the broad-index call does not — which is why expressing the ISM as a *relative* trade is more robust than expressing it as a *directional* one.

One more practical habit ties the whole playbook together: **track the survey as a series, not as a single print.** A single month is noisy and gets revised; the signal lives in the three-month direction and in the cross-confirmation between manufacturing and services, between new orders and the headline, between the US ISM and the global PMI. Build the simple two-axis read (level versus 50, direction over three months), overlay new orders, and you have a forward read on the cyclical rotation that updates monthly, leads earnings by about two quarters, and tells you which half of the market to own. That is a remarkable amount of usable signal from one free, public, first-of-the-month number — and it is why the ISM remains the single most-watched line in the macro calendar despite measuring a shrinking slice of the economy.

A final piece of honesty. The ISM is one of the best leading indicators we have, but it is still a survey of opinions about the near future, and opinions can be wrong together. Its correlation with cyclicals is strong (+0.45 to +0.65) but not perfect; its six-month lead to earnings is an average, not a guarantee; and the sign of the equity reaction depends on a regime that itself can shift. The right way to hold all this is the way you should hold every relationship in this series: a correlation is a *regime*, with a sign, a strength, a lead, and a set of conditions under which it flips. The ISM gives you the earliest read on the cycle in the macro calendar — and if you respect its construction (breadth, not speed), its leading sub-index (new orders), and its switch (the 50 line, direction over level), it is as close to a usable crystal ball as macro data gets.

## Further reading and cross-links

Within this series:

- [Lead, lag, or coincident: the time axis of every correlation](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) — the cross-correlation function and the leading/coincident/lagging taxonomy the ISM sits in.
- [The business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — how the cyclical/defensive rotation rotates through the four phases of the cycle.
- [NFP and asset prices: the king of data correlation](/blog/trading/macro-correlations/nfp-and-asset-prices-the-king-of-data-correlation) — the other great growth-and-labor release, and how its sign flips by regime.
- [GDP, retail sales and the consumer correlation](/blog/trading/macro-correlations/gdp-retail-sales-and-the-consumer-correlation) — the coincident hard-data counterpart the survey leads.
- [Copper, gold and the growth-inflation signal](/blog/trading/macro-correlations/copper-gold-and-the-growth-inflation-signal) — copper as the purest commodity expression of the ISM's growth signal.

Companion series:

- [ISM and PMI: the business surveys that lead](/blog/trading/event-trading/ism-pmi-the-business-surveys-that-lead) — the intraday *release-day* reaction to a survey print.
- [The business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — the mechanism behind the cyclical rotation, phase by phase.
