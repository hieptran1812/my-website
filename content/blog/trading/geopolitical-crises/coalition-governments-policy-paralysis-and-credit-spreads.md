---
title: "Coalition governments, policy paralysis, and credit spreads: when hung parliaments become market risk"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "How coalition formation delays in Italy, Germany, Israel, and France translate directly into measurable sovereign credit spread widening, and how traders position around the political uncertainty premium."
tags: ["geopolitics", "sovereign-bonds", "credit-spreads", "coalition-governments", "political-risk", "italy", "france", "germany", "israel", "btp-bund", "fixed-income", "political-uncertainty"]
category: "trading"
subcategory: "Geopolitical Crises"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — When a parliamentary election produces no clear majority, the weeks or months of coalition negotiations translate directly into basis-points on the sovereign spread, and the pattern is systematic enough to trade.
>
> - Italy 2018: BTP-Bund spread surged from ~130bp to over 300bp in eight weeks as the 5-Star/Lega coalition stalled, pushing Italy's 10-year borrowing cost from 1.8% to 3.4%.
> - France 2024: OAT-Bund spread widened +38bp in just eight trading days after Macron called a snap election.
> - Israel's five elections in four years created a persistent "political uncertainty discount" on the TASE versus global peers.
> - The compression trade is real: spreads begin narrowing 2-4 weeks before a coalition deal is formally announced, as options market positioning and newspaper reporting signal probable resolution.
> - The key number: on average, 50-60% of the political uncertainty premium compresses within eight weeks of a coalition deal being signed.

On the evening of May 27, 2018, Italian President Sergio Mattarella rejected the coalition government's proposed economy minister — a euroskeptic economist named Paolo Savona — and instead asked a former IMF official to form a technocratic administration. What happened in the next 72 hours is one of the most instructive episodes in post-2010 financial markets: the spread between Italy's 10-year government bond yield and Germany's Bund rose by over 100 basis points in a single session. Italy's two-year bond yield leaped from near zero to over 2.7% in a single day. Markets had priced in a possible Italian exit from the euro.

This was not a war, a sanctions shock, or a natural disaster. It was a political negotiation about who would sit in a cabinet. And yet the financial damage — measured in Italy's sovereign borrowing costs, bank stock losses, and contagion to Spain and Portugal — rivaled many physical crises. The episode crystallized something that practitioners had observed for decades but academics had only recently started to measure: coalition government formation is itself a form of political risk, and that risk is directly quantifiable in sovereign credit spreads.

![Coalition uncertainty to sovereign spread premium transmission pipeline](/imgs/blogs/coalition-governments-policy-paralysis-and-credit-spreads-1.png)

This post maps the mechanism from parliamentary arithmetic to basis points. We will walk through Italy 2018, Germany's 2021 traffic-light coalition, Israel's five-election cycle, and France's 2024 snap vote. We will build the "political uncertainty premium" model in formal terms, show when it peaks and when it compresses, and lay out a concrete playbook for positioning around it.

## Foundations: how sovereign credit spreads work and why political uncertainty moves them

Before we can understand why a hung parliament widens the spread, we need to understand what the spread is and what drives it in the first place.

A sovereign bond spread is the difference in yield between one country's government debt and a reference country — for Europe, this is almost always the German Bund. If Italy's 10-year bond yields 3.0% and Germany's Bund yields 2.0%, the BTP-Bund spread is 100 basis points (1 basis point = 0.01 percentage point). If the spread widens to 200bp, Italy must now pay 4.0% on its 10-year debt while Germany still pays 2.0%. That 1 percentage point difference compounds over years: for a country with Italy's debt-to-GDP ratio (around 140%), a 100bp spread widening adds roughly 0.6% of GDP per year to its interest bill once the debt rolls over.

Why do spreads widen? The bond market is fundamentally an exercise in probability assessment. When you lend money to a government by buying its bond, you face three questions: Will this government repay the principal? Will inflation erode the real value of repayment? Will there be enough liquidity in the market for me to exit this position if I need to? Political instability touches all three.

A coalition government that has not yet formed cannot commit to a fiscal path. Without a fiscal path, the market cannot assess whether the primary balance (revenue minus non-interest spending) will be sufficient to stabilize the debt-to-GDP ratio. The Italian bond market in May 2018 was not primarily worried about default in the legal sense — it was pricing the probability that a populist government, once in office, would abandon the commitments made to the European Commission on deficit targets and force a confrontation with the EU. In that scenario, Italian debt could be restructured or Italy could theoretically exit the euro and redenominate debt, introducing currency risk that had not existed in the eurozone era.

This is the crux of the political uncertainty premium: it is not the probability of catastrophe times the damage of catastrophe. It is the probability of a policy regime change times the market-implied damage of that regime change. The Bloom (2009) policy uncertainty model — the Economic Policy Uncertainty (EPU) index — measures this formally by counting newspaper references to uncertainty about economic policy, expiring tax provisions, and disagreement among economic forecasters. The EPU index spikes reliably during coalition formation periods. Baker, Bloom, and Davis (2016) showed that a 1-standard-deviation EPU shock reduces GDP by 0.4% and investment by 1% over a year, and that financial markets price this in approximately contemporaneously — spreads move at the same time as the news, not with a lag.

The formal model runs as follows. Let S(t) denote the sovereign spread at time t. The spread can be decomposed into:

S(t) = credit_risk_premium(t) + liquidity_premium(t) + redenomination_risk_premium(t) + political_uncertainty_premium(t)

During stable government periods, the political uncertainty premium is close to zero and the spread is dominated by the structural credit risk of the country. During coalition formation episodes, the political uncertainty premium becomes the dominant driver — it can contribute 100-200bp to a spread that was previously only 130bp wide, effectively tripling or quadrupling the risk premium in a matter of weeks.

The speed of the move matters as much as the magnitude. Unlike a sovereign debt crisis driven by slow deterioration of fiscal fundamentals (Greece 2010-2012 unfolded over two years), political uncertainty premiums move at the speed of political news. A single tweet from a coalition negotiator, a presidential statement, or a leaked policy document can move a spread by 20-30bp in a trading session. This creates both the risk and the opportunity.

## The political moves: what happens during coalition formation and why it matters to markets

To understand the market dynamics, we need to understand the political dynamics that drive them. Coalition formation in parliamentary systems is not a single event — it is a sequential process with multiple stages, each of which can break down and create new uncertainty.

When no single party wins a parliamentary majority, the country enters a formation period that typically proceeds through these stages: first, the president or head of state charges the leader of the largest party (or coalition bloc) with forming a government; second, exploratory talks begin between potential coalition partners to identify shared policy ground; third, formal coalition negotiations start on a coalition agreement document; fourth, the proposed government is presented to parliament for a confidence vote; finally, the government takes office.

Each stage can fail. In Italy 2018, the critical failure came at the very last stage: even after 5-Star and Lega had agreed on a coalition document and submitted their government list, the president exercised his constitutional prerogative to reject a minister nomination. That created a constitutional crisis that had not been priced in — markets had assumed the formation process was essentially complete when it suddenly was not.

The political logic that matters for markets is specifically about fiscal and structural policy credibility. Bond markets are forward-looking instruments. They price the present value of all future coupon payments and principal repayment, discounted at a rate that incorporates default risk. The fiscal credibility question is: given the proposed coalition's stated policies, will the government run deficits that stabilize, reduce, or increase the debt-to-GDP ratio? And is the coalition's stated policy position credible — will they actually implement it once in office?

Credibility is the operative word. A coalition between two parties with divergent economic views inherently has lower credibility than a single-party government, because any policy commitment must survive future disagreements between the coalition partners. Markets embed a "commitment discount" into coalition government spreads — a premium that reflects not just the stated policy but the probability that the coalition will fracture on a key vote and revert to uncertainty.

The German 2021 coalition (SPD, Greens, FDP — the "traffic-light" coalition) illustrates this clearly. The three parties agreed on a coalition document, but their views on fiscal policy were genuinely incompatible: the SPD and Greens wanted substantial public investment and a flexible interpretation of the constitutional debt brake (Schuldenbremse), while the FDP insisted on strict adherence. Markets assigned a "policy delivery discount" to German assets, particularly to sectors dependent on green investment policy. The DAX underperformed the S&P 500 by approximately 12 percentage points in the six months following the election, partly driven by this policy uncertainty. The debt brake crisis eventually came to a head in 2023 when the Federal Constitutional Court struck down a budget maneuver, forcing exactly the kind of coalition crisis that markets had partially priced in two years earlier.

## The financial channels: how political moves become spread moves

There are four transmission channels through which coalition formation uncertainty flows into sovereign spreads. Understanding which channel is dominant in any given episode determines the likely magnitude and persistence of the spread widening.

**Channel 1: Fiscal path uncertainty.** When a coalition has not formed, there is no budget. Without a budget, markets cannot assess the trajectory of government revenue and spending. The Italian constitutional crisis of 2018 was fundamentally a dispute about whether Italy would comply with EU fiscal rules — a deficit above 3% of GDP would trigger EU sanctions and potentially constrain Italian borrowing. Markets priced the probability of rule-breaking multiplied by the damage of an EU confrontation.

**Channel 2: Monetary-fiscal interface.** In the eurozone, the ECB can act as an implicit backstop for sovereign debt through its Transmission Protection Instrument (TPI) — but only for countries that are "in compliance with EU fiscal framework requirements." A government that is likely to breach EU fiscal rules faces a credible threat that the ECB backstop will not be available, which dramatically increases the tail risk of the bond and therefore the required risk premium.

**Channel 3: Redenomination/exit risk.** For eurozone members with high debt and populist coalition partners, markets embed a nonzero probability of euro exit or redenomination. In Italy 2018, the five-year credit default swap (CDS) spread — which prices the cost of insuring against Italian default — rose from 110bp to over 280bp in the crisis weeks. The CDS market was pricing not just fiscal slippage but structural redenomination risk.

**Channel 4: Foreign investor positioning.** Italy's government bond market has roughly 2.3 trillion euros of outstanding debt. Foreign investors hold approximately 30% of this — a significant share. When political uncertainty rises, foreign investors do not wait for the outcome; they reduce exposure preemptively. This creates a self-reinforcing dynamic: foreign selling pushes yields higher, which raises the probability of a fiscal crisis, which induces more selling. The initial political trigger becomes amplified by the market microstructure.

![Italy 2018 and France 2024 sovereign spread during coalition uncertainty](/imgs/blogs/coalition-governments-policy-paralysis-and-credit-spreads-3.png)

The interaction between channels matters enormously for the spread trajectory. In 2018, Italy had all four channels active simultaneously — fiscal uncertainty, ECB backstop doubt, redenomination fear, and foreign selling pressure — which is why the spread moved so dramatically. France 2024, by contrast, had a lower debt-to-GDP ratio, a government that remained in place during the election period, and a market perception that the ECB backstop was still available; accordingly, the OAT-Bund spread widened by only 38bp peak-to-pre-election-announcement, rather than the 170bp Italy experienced.

## The four case studies: Italy 2018, Germany 2021, Israel's cycle, and France 2024

### Italy 2018: the template

The March 4, 2018 Italian general election produced no majority coalition. The 5-Star Movement won 32% of the vote; the center-right bloc (led by Lega) won 37%; the center-left (PD) was reduced to 19%. After weeks of failed negotiations, 5-Star and Lega agreed to form a coalition government — the "government of change."

The critical market-moving event was not the election itself. The BTP-Bund spread was around 130bp on election day, close to its recent historical range. The spread began moving meaningfully in mid-April as markets realized that the coalition would likely be openly confrontational with EU fiscal rules. But the crisis peak came in the last week of May, when President Mattarella rejected the proposed economics minister and triggered a constitutional standoff.

Between May 10 and May 31, the BTP-Bund spread rose from approximately 175bp to over 300bp. On the day of the worst session (May 29), Italian two-year yields rose by more than 100bp in a single day — one of the largest single-day moves in eurozone history outside of the 2010-2012 debt crisis. Italian bank stocks fell 5-10% as their sovereign bond holdings were marked down.

The government eventually formed on June 1 when President Mattarella backed down and accepted a modified cabinet list. Spreads began compressing immediately — the BTP-Bund fell from 310bp on May 31 to 255bp within a week and continued compressing as the new government began its term.

#### Worked example:

Suppose you are managing a European fixed-income portfolio in early May 2018. Italy's BTP 10-year bond is trading at a spread of 160bp over Bund, with a yield of approximately 1.95% (the Bund is yielding ~0.55%). You hold \$10 million face value of Italian BTPs.

Political risk is rising: the coalition is forming, but 5-Star/Lega statements suggest fiscal expansion inconsistent with EU rules. You estimate a 30% probability that the spread widens to 300bp (crisis scenario), and a 70% probability that a government forms without confrontation and the spread stays around 180bp.

Expected spread: 0.30 × 300 + 0.70 × 180 = 90 + 126 = 216bp.

Current spread: 160bp. Expected spread: 216bp. The market is not fully pricing the crisis probability. Duration on a 10-year Italian bond is approximately 8.5 years.

If the spread widens from 160bp to 300bp (a 140bp widening), the price impact on a duration-8.5 bond is approximately:

Price change ≈ -Duration × Yield_change = -8.5 × 0.014 = -11.9%

On \$10 million face value: \$10M × -0.119 = -\$1.19M loss.

A hedged position (long BTPs but short Italian CDS protection) would cost approximately 1.10% per year in premium (the CDS spread in May 2018 was 110bp). To protect \$10M of BTPs, you buy \$10M of 5-year CDS protection for an annual cost of \$110,000. If the spread spikes to 300bp and CDS widens to 280bp, the CDS position gains \$10M × (0.028 - 0.011) × 5 ≈ \$850,000, partially offsetting the \$1.19M bond loss.

### Germany 2021: the traffic-light coalition and DAX underperformance

Germany's September 2021 election produced its most complex coalition since reunification. The SPD won 25.7%, CDU/CSU fell to 24.1% (their worst result since 1949), the Greens won 14.8%, and the FDP took 11.5%. Coalition formation took 73 days — from September 26 to December 8, 2021.

Germany's institutional credibility and AAA rating meant that Bund spreads versus other safe havens did not blow out — Germany's borrowing costs did not face the same stress as Italy's. The market expression of uncertainty was instead in equity markets. The DAX index underperformed the S&P 500 by approximately 12 percentage points from election day to the end of 2021. Sector rotation was pronounced: industrials and utilities outperformed (perceived as coalition-agnostic), while healthcare and financials underperformed (perceived as more exposed to potential coalition policy changes on pricing and bank regulation).

The debt brake crisis that fully materialized in 2023 can be seen as the delayed realization of the risk that bond markets were already embedding in 2021. The market was correct that the coalition would face irreconcilable fiscal differences — it just took two years for the fault line to break into the open.

### Israel: five elections in four years and the political uncertainty discount

Between April 2019 and November 2022, Israel held five parliamentary elections. No party or bloc succeeded in forming a stable majority government until Benjamin Netanyahu's December 2022 coalition. The frequency of electoral cycles meant that the Israeli economy spent a significant portion of three years in political uncertainty.

The TASE (Tel Aviv Stock Exchange) showed a clear and measurable "political uncertainty discount" during this period. The TA-35 index (Israel's blue-chip benchmark) underperformed the MSCI World index by approximately 18 percentage points between the start of the second election cycle (September 2019) and the end of the fourth election cycle (April 2021). Part of this underperformance was attributable to sector composition (TASE is heavily weighted in financials and real estate), but event studies controlling for sector effects still find a statistically significant political uncertainty component of approximately 8-10 percentage points.

The Israeli shekel also showed measurable sensitivity. In the weeks surrounding each election, EUR/ILS and USD/ILS volatility (implied by options) rose 2-4 volatility points above baseline, reflecting uncertainty about the economic policy direction of a future government. Foreign direct investment flows slowed during the most intensive uncertainty periods.

![Israel TASE vs MSCI World during political uncertainty cycles](/imgs/blogs/coalition-governments-policy-paralysis-and-credit-spreads-5.png)

What makes Israel's case particularly instructive is the persistence question. Unlike a one-time shock (Italy 2018's spread widening was partially resolved when the government formed), Israel's repeated elections created a situation where the "political uncertainty premium" became embedded in valuation multiples. Analysts covering Israeli equities began building "political risk haircuts" of 10-15% into their target prices for companies that were domestically focused. The uncertainty was not a temporary spike but a sustained discount.

#### Worked example:

An asset manager is running a \$500 million emerging-market equity portfolio in March 2020, just after Israel's third election produced another hung parliament. The manager wants to assess the "political uncertainty cost" embedded in the TASE position.

Historical TASE beta to MSCI EM (emerging markets) is 0.85. Under normal circumstances, if MSCI EM gains 10%, TASE should gain approximately 8.5%.

The manager estimates: (a) 40% probability that a fourth election is called within six months; (b) 60% probability that a coalition forms and uncertainty normalizes. Under scenario (a), historical data suggests TASE underperforms MSCI EM by ~8 percentage points over the subsequent six months. Under scenario (b), TASE reverts to beta-normal performance.

Expected relative performance of TASE vs MSCI EM: 0.40 × (-8%) + 0.60 × (0%) = -3.2%.

On a \$50 million TASE allocation within the portfolio: \$50M × -0.032 = -\$1.6 million expected underperformance. The manager can reduce the TASE allocation to \$35 million (cutting exposure by \$15M) and allocate the \$15M to a MSCI EM ex-Israel ETF, neutralizing the political uncertainty drag while maintaining emerging market exposure.

### France 2024: speed of spread widening in a core eurozone member

France's June 2024 snap election was distinctive because it affected a core eurozone member — not a peripheral country — and because the OAT-Bund spread had previously been remarkably stable at 40-50bp for years. The announcement of the snap election on June 9, 2024 (following the European Parliament elections, in which Marine Le Pen's Rassemblement National came first) produced one of the fastest core-eurozone spread moves on record.

The OAT-Bund 10-year spread widened from 47bp on June 7 to 85bp on June 17 — a 38bp move in eight trading days. French equity markets fell 5-6% in the two weeks following the election announcement. French bank stocks (BNP Paribas, Société Générale, Crédit Agricole) fell 10-15% as their large OAT holdings were marked down and their exposure to a potentially looser French fiscal stance was repriced.

The French episode illustrates a key asymmetry in how markets price political uncertainty: the speed of widening is much faster than the speed of compression. The OAT-Bund spread widened 38bp in 8 trading days, but it took approximately six weeks to compress back to its pre-election level even after the election produced a hung National Assembly with no absolute majority. The persistence of elevated spreads reflected ongoing uncertainty about governance — with no coalition, France operated under a minority government that could pass few major policy changes, leaving the fiscal trajectory ambiguous.

## The political uncertainty premium model in formal terms

The Bloom (2009) Economic Policy Uncertainty framework provides the most widely used formal model. The EPU index for a country is constructed as a weighted average of three components: (1) the count of newspaper articles mentioning economic policy uncertainty; (2) the number of expiring tax code provisions; (3) dispersion among economic forecasters' projections for key fiscal variables like government spending.

For sovereign spread modeling, the relevant innovation is the extension by Bordo et al. (2016), who showed that EPU is directly causal to sovereign spread widening, not merely correlated. Using a panel of OECD countries from 1985-2014, they found that a 1-standard-deviation increase in the country-specific EPU index is associated with a 12-18bp widening of the sovereign spread over the following month, controlling for fiscal fundamentals (debt-to-GDP, deficit, growth).

The transmission is not linear. At baseline EPU levels, the effect is small. But EPU is bounded below (you cannot have negative uncertainty) and unbounded above. Coalition formation episodes move EPU to 2-3 standard deviations above baseline, which places the spread effect in a non-linear range where the marginal impact of additional uncertainty is larger. This non-linearity explains why spread moves during coalition crises seem disproportionately large relative to the "size" of the political event.

A practical signal: the EPU index for the relevant country (available at policyuncertainty.com) typically peaks 2-4 weeks before the sovereign spread peaks. This is consistent with the observation that markets price news about political negotiations contemporaneously with news reporting, while the full spread impact takes time to work through as foreign investor positioning adjusts. Monitoring EPU in real time (the newspaper component updates daily) can provide a leading indicator for when spread widening is likely to peak.

![Matrix of political uncertainty premium by scenario type](/imgs/blogs/coalition-governments-policy-paralysis-and-credit-spreads-4.png)

## How long does the premium persist and when to fade it

One of the most practically important questions for a trader is: when does the political uncertainty premium compress, and how fast?

The evidence from the cases above suggests three phases:

**Phase 1: Widening (election to peak, typically 4-8 weeks).** Spreads widen as the probability of a functional coalition is unclear. The widening is front-loaded — most of the move happens in the first 2-3 weeks after the election as positioning is established. Late-stage widening in Phase 1 tends to be smaller and driven by specific political events (a breakdown in negotiations, a rejected proposal, a constitutional challenge).

**Phase 2: Peak consolidation (2-4 weeks before deal).** Spreads stop widening and trade in a range. This phase can be prolonged if negotiations are genuinely uncertain. Options markets show elevated implied volatility without a clear directional drift. This is the highest-risk period to be short the spread (i.e., betting on compression), because a negative shock (negotiation collapse) can produce a rapid additional widening.

**Phase 3: Compression (deal announcement to 8 weeks post-deal).** Once a coalition deal is formally announced, spreads compress sharply. Historically, 40-60% of the total widening from pre-election to peak compresses within two weeks of a deal, and 75-85% compresses within eight weeks. The compression is not monotonic — a minor political crisis in the new coalition's first weeks can cause a brief re-widening — but the directional move is clearly toward tighter spreads.

The critical signal: spread compression often begins before the formal deal announcement. In Italy 2018, BTP-Bund spreads started narrowing approximately 10 days before the Conte government was formally sworn in, as media reports indicated a compromise was close. In France 2024, the OAT-Bund spread started compressing before the second round results were confirmed. The options market is typically ahead of the cash market in pricing coalition resolution: a decline in implied volatility on Italian or French government bonds is an early signal that market participants are becoming more confident in a resolution.

![Sovereign bond market: stable government vs coalition paralysis states](/imgs/blogs/coalition-governments-policy-paralysis-and-credit-spreads-6.png)

#### Worked example:

It is May 20, 2018. The BTP-Bund spread is 235bp, having widened from 130bp since the election. You are a hedge fund manager who wants to position for eventual spread compression.

You execute a "spread compression trade" by buying Italian BTPs (10-year, yield 2.8%) funded by selling German Bunds (10-year, yield 0.45%). This is called a "long BTP, short Bund" position, or equivalently a "short spread" position.

Notional: \$20 million in BTPs purchased, \$20 million in Bunds sold short. Duration of each position: approximately 8.5 years (10-year bond). Current spread: 235bp. Target spread (post-deal): 160bp (partial compression back toward pre-crisis levels). Maximum adverse move you tolerate: spread widens to 350bp (stop-loss).

Expected gain if spread compresses to 160bp: 235bp - 160bp = 75bp spread compression. Duration-adjusted P&L ≈ \$20M × 8.5 × 0.0075 = \$1.275 million profit.

Potential loss if spread widens to 350bp: 350bp - 235bp = 115bp additional widening. Duration-adjusted P&L ≈ \$20M × 8.5 × (-0.0115) = -\$1.955 million loss.

Risk-reward ratio: 1.275 / 1.955 = 0.65 before deal probability weighting. If you assign 65% probability to a deal being struck within 6 weeks (reasonable at this point in the Italian negotiations), expected value = 0.65 × 1.275 + 0.35 × (-1.955) = \$0.829M - \$0.684M = +\$145,000 expected value on a \$20M trade.

The sizing is moderate but the carry is negative in the interim — you are paying the BTP-Bund spread differential (2.8% minus 0.45% = 2.35% per annum on the net position). On a \$20M notional, that is \$470,000 per year or about \$39,000 per month. This means the trade must resolve within 3-4 months before negative carry erodes the expected value significantly.

![Spread compression pattern before and after coalition deal announcement](/imgs/blogs/coalition-governments-policy-paralysis-and-credit-spreads-7.png)

## The Italy 2018 timeline in detail

The BTP-Bund spread's journey in 2018 is worth examining in granular detail because it contains almost every pattern that repeats in subsequent episodes.

March 4: Election day. No majority produced. Spread: ~130bp. Markets were prepared for this outcome — polling had indicated no clear majority for weeks. The initial reaction was muted; the spread widened only to about 140bp in the first week.

April: Exploratory talks between parties failed repeatedly. The spread drifted to 160bp as markets priced in a longer-than-expected formation period but remained calm about the ultimate outcome — most analysts expected either a 5-Star/PD deal or a 5-Star/Lega deal, both of which would be acceptable to markets.

May 1-10: Coalition talks between 5-Star and Lega accelerated. A draft coalition document was leaked that included provisions for a possible euro exit mechanism and debt restructuring. The spread moved to 175-200bp on these reports. Bond markets priced a higher probability of an anti-establishment government with genuinely confrontational fiscal views.

May 16-27: The coalition document was finalized and a cabinet list was submitted to the president. The proposed economy minister, Paolo Savona, had publicly advocated for a plan B for euro exit. The spread moved to 240-280bp as markets priced the possibility that Savona's appointment would signal a genuine intent to challenge EU fiscal rules.

May 27-31: President Mattarella rejected Savona and charged a technocrat (Cottarelli) with forming a caretaker government. The constitutional crisis was now fully apparent. Markets priced the probability of new elections under extreme uncertainty. The spread hit 310bp on May 29, Italian two-year yields rose above 2.7% (from negative territory in February), and contagion spread to Spain and Portugal.

June 1: President Mattarella backed down and accepted a revised cabinet list with Savona in a less powerful role. Conte was sworn in as Prime Minister. Spreads compressed from 310bp to 255bp within a week. By end-June the spread was around 230bp — still significantly wider than pre-election, but the crisis premium had partially faded.

The second peak came in September-October 2018 when the new Italian government unveiled a 2019 budget with a deficit of 2.4% of GDP, above the 1.8% target agreed with the EU. The BTP-Bund spread re-widened to 310-320bp. This second widening demonstrated that political uncertainty premiums do not disappear with coalition formation — they transform into "policy delivery uncertainty" as markets assess whether the new government's fiscal commitments are credible.

## France 2024: the snap election mechanics

France's episode differed from Italy's in a critical structural way: France has a two-round electoral system for its National Assembly, and the President of France (Macron) had broader emergency dissolution powers than Italy's constitution allows. When Macron called a snap election on June 9, markets immediately had to assess two scenarios:

Scenario A (probability ~35% in initial market pricing): Rassemblement National wins an absolute majority, forming a government that would implement fiscal expansionism outside EU rules, potentially triggering a confrontation with the European Commission.

Scenario B (probability ~65%): A hung parliament with RN as the largest party but no absolute majority, requiring either a minority government or a "republican front" coalition to block extremes.

The 38bp spread widening reflected the probability-weighted value of these scenarios. In the event, Scenario B occurred — France ended up with a fragmented National Assembly in which no party or bloc had a majority. This produced a different kind of political uncertainty than Italy 2018: not the formation risk of an anti-establishment government, but the governance risk of a legislature unable to pass major legislation.

The OAT-Bund spread compressed from 85bp back to around 65-70bp after the election results clarified that an outright RN majority was avoided, but it did not return to the pre-announcement level of 47bp. The residual 20bp reflects the ongoing governance uncertainty of a hung parliament — the "coalition formation discount" replaced by a "policy paralysis discount."

## Common misconceptions

**Misconception 1: Spread widening during elections is a buying opportunity.** Sometimes it is, but the timing matters enormously. Many investors bought Italian BTPs in early May 2018 assuming the coalition uncertainty was already priced. They were wrong — the spread widened another 100bp over the following three weeks. The correct entry point for a compression trade is when coalition formation is highly probable (Phase 2 to early Phase 3), not when uncertainty is at its peak.

**Misconception 2: Coalition governments are always bad for bond markets.** Germany has been governed by coalitions for its entire modern history and maintains AAA credit ratings with some of the tightest sovereign spreads in the world. The spread impact is not about coalition governments per se but about the specific combination of (a) fiscal credibility risk, (b) formation delay, and (c) foreign investor concentration. A coalition between fiscally conservative parties in a country with low debt-to-GDP can actually be positive for bond markets if it signals political stability relative to a fragile single-party government.

**Misconception 3: Political uncertainty premiums persist for years.** The empirical evidence suggests they do not, outside of genuinely chronic political dysfunction. In Italy 2018, the political uncertainty premium (the component of the spread above what fiscal fundamentals would imply) dissipated within 3-4 months of the government's formation, even though the government itself was controversial. In France 2024, the residual premium after the election was in the range of 15-20bp — measurable but modest. The market correctly distinguishes between acute formation uncertainty (large, temporary premium) and chronic governance uncertainty (moderate, persistent premium).

**Misconception 4: The ECB always intervenes to compress spreads.** The ECB's Transmission Protection Instrument (TPI) is available "to countries in compliance with the EU fiscal framework." A government that openly violates EU fiscal rules (Italy's 2018 budget confrontation) may face a situation where the ECB's backstop is conditional on fiscal adjustment — which is itself a source of spread widening, not compression. Markets priced this conditionality explicitly in October 2018 when the Italian budget confrontation with the EU produced a second spread spike.

**Misconception 5: Smaller countries' political crises don't matter to larger markets.** Greece's political crises in 2011-2015 produced significant contagion effects in Spain, Portugal, and even Italy — countries with much larger GDP and bond markets. The contagion mechanism works through two channels: direct sovereign spread correlation (risk-off positioning simultaneously reduces demand for all peripheral eurozone debt) and banking system interconnection (European banks held peripheral debt and faced mark-to-market losses simultaneously). Israel's political uncertainty produced modest contagion to neighboring Middle Eastern markets through risk-off flows.

## How it shows up in real markets

**The basis: CDS versus cash spread.** During political uncertainty episodes, the CDS spread and the cash bond spread often diverge. CDS (credit default swap) spreads respond faster because they are derivatives with no funding requirement — you can buy CDS protection without owning any bonds. The "basis" (CDS spread minus bond spread) tends to widen during political crises as CDS is the instrument of choice for fast-moving hedgers and speculators. When the basis is negative (bonds imply wider credit risk than CDS), this often signals that real-money investors are selling the bonds while derivatives traders have yet to fully price the risk — a leading indicator of further spread widening.

**Options market signals.** Implied volatility in the bond options market (swaptions and bond futures options) rises sharply during political uncertainty periods. In Italy 2018, BTP options implied volatility rose from 4-5% to over 15% at the peak of the crisis. Falling implied vol is one of the clearest early signals that the market believes political resolution is approaching — often 1-2 weeks before the actual coalition deal.

**Auction bidding.** Italy, France, and Germany hold regular government bond auctions. The bid-to-cover ratio (total bids / amount sold) is an important signal of foreign demand. During the May 2018 crisis, Italy's auction bid-to-cover ratios fell to 1.1-1.2x (versus a normal range of 1.5-1.8x), indicating weak demand from foreign investors. Auction data is published immediately and is a real-time indicator of market stress.

**Bank stock performance.** European banks hold large sovereign bond portfolios as a regulatory requirement and for liquidity purposes. When sovereign spreads widen, bank stocks fall because the mark-to-market losses on their bond holdings reduce their regulatory capital. The correlation between BTP-Bund spread widening and Italian bank stock performance is very high (correlation around -0.85 during crisis periods). Bank stocks therefore function as a leveraged proxy for sovereign spread risk — a useful instrument for hedging or expressing a view with more liquidity than the bond market itself.

#### Worked example:

It is June 10, 2024, the day after Macron announces the snap election. The OAT-Bund spread has widened from 47bp to 57bp overnight. French bank stocks (BNP, SocGen) are down 4-5% at the open. You manage a long-only European equity fund and want to hedge your French bank exposure.

You hold \$30 million in French bank stocks (approximate beta to OAT-Bund spread changes: -1.2, meaning every 10bp of spread widening costs approximately 1.2% in equity price).

If the spread widens to 85bp (scenario consistent with the eventual peak), the expected loss on your bank stock position: (85 - 57) × 0.1% × 1.2 × \$30M = 28 × 0.001 × 1.2 × \$30M = \$1.008 million.

You can hedge by buying OAT put options (options that profit if OAT prices fall / yields rise). Alternatively, you can buy French CDS protection or sell French equity index futures (CAC 40 futures).

CAC 40 hedging: the index has a beta of approximately 0.7 to French bank stocks. To hedge \$30M of bank stocks, you need to short CAC 40 futures worth approximately \$30M / 0.7 = \$42.9M. CAC 40 futures have a contract value of approximately \$55,000 per contract (varies with index level). Number of contracts: \$42.9M / \$55,000 = 780 contracts. The hedge reduces your French bank exposure significantly but at the cost of capping your upside if political uncertainty resolves quickly.

## How to trade it: the playbook

The political uncertainty spread trade follows a four-stage playbook based on the timing analysis above.

**Stage 1: Pre-election risk assessment (4-6 weeks before election).** Poll tracker analysis, historical precedents for the specific country, and an assessment of which coalition scenarios are possible and which would be fiscal-credibility-positive versus negative. Identify the specific spread level at which you want to initiate a hedging position (typically, a widening of 50-80bp from baseline for peripheral eurozone members). Set stop-loss levels.

**Stage 2: Post-election positioning (election day to peak, 4-8 weeks).** If the election produces the expected hung parliament, initiate the trade per your risk parameters. In Phase 1, the position is typically a hedge rather than an aggressive bet — the uncertainty is high and the spread can widen further. Use options (buying protection via bond puts or CDS) rather than directional outright shorts, which have unlimited adverse movement potential.

**Stage 3: Peak identification and compression trade initiation (Phase 2).** The key signals for identifying the peak: (a) EPU index stops rising or begins falling; (b) options implied volatility peaks and begins declining; (c) auction bid-to-cover ratios stabilize; (d) media reports shift from "negotiations in crisis" to "coalition close to deal"; (e) spread has widened more than 1.5 standard deviations above its historical relationship with fiscal fundamentals. At this point, initiate a compression position — long the spread through bonds or by selling CDS protection.

**Stage 4: Post-deal compression management (Phase 3).** Take partial profits at the first sharp compression (typically 40-50% of the total widening) and hold a residual position for further normalization. The stop-loss for the remaining position is typically a re-widening above the post-deal close — if spreads re-widen above where they were when the deal was announced, something has gone wrong with the new government and the thesis needs to be re-evaluated.

**What invalidates the thesis:** (a) coalition collapse after formation — the government loses a confidence vote or a coalition partner exits, returning to formation uncertainty; (b) rating agency downgrade — if the uncertainty period results in a credit rating cut, the spread ceiling rises structurally and the compression target must be revised downward; (c) ECB policy change — if the ECB removes its backstop commitment or tightens its TPI conditionality, the risk premium for peripheral debt rises structurally; (d) global risk-off shock — a global recession or financial crisis concurrent with the coalition uncertainty will dominate the political premium and compress it via a different channel.

**The early signal — spread compressing before the deal.** In Italy 2018, the BTP-Bund spread started narrowing approximately 10 days before June 1, when coalition negotiations concluded. In France 2024, OAT-Bund began compressing before the second-round results. The mechanism: options traders and bond dealers who maintain close contact with political sources begin adjusting positions as they receive signals that a deal is close. The options market IV starts falling first, then the cash spread follows. Monitoring BTP swaption implied vol daily — available through Bloomberg and Reuters — provides a real-time leading indicator.

## The Bloom model in practice: reading EPU as a spread predictor

Baker, Bloom, and Davis maintain the Economic Policy Uncertainty index at policyuncertainty.com. The index is updated monthly for most countries and quarterly for some. During coalition formation periods, the newspaper component (which updates more frequently) is the most useful.

For Italy, the EPU index rose from approximately 120 (baseline = 100) in January 2018 to over 280 in May 2018. This 2.3-standard-deviation move was consistent with the historical relationship between EPU and BTP-Bund spreads — each standard deviation move in EPU has historically been associated with 60-80bp of spread widening in Italy, after controlling for debt-to-GDP and growth. The May 2018 peak EPU reading would have implied a spread of approximately 300bp under the historical regression, which was close to the actual peak.

For Germany 2021, the German EPU rose from 90 to 165 during the coalition formation period — a 1.25-standard-deviation move — consistent with the more modest equity market reaction (no Bund spread widening but about 12pp DAX underperformance).

The key limitation of EPU as a trading signal is its frequency: the full index updates monthly, and the relevant newspaper component updates daily but with a one-day lag. For fast-moving political crises (France 2024's 38bp in 8 days), EPU is too slow to serve as a timing signal. In those cases, real-time monitoring of political news flow, auction data, and options implied volatility provides more timely signals.

## Cross-asset effects and second-order impacts

The sovereign spread is the primary financial channel, but coalition uncertainty produces measurable effects across multiple asset classes.

**Equity markets.** Political uncertainty reduces the denominator of equity valuations (the discount rate rises as sovereign yields rise) and the numerator (fiscal uncertainty reduces the expected profitability of domestically oriented businesses). The sector most directly affected is banking (through sovereign bond holdings), followed by utilities and real estate (through long-duration discount rate sensitivity), and then consumer discretionary (through fiscal policy uncertainty affecting household confidence).

**Currency markets.** For eurozone members, political uncertainty does not primarily affect the EUR/USD rate (since it is a shared currency) but produces intra-eurozone basis effects. EUR/CHF typically weakens modestly during Italian or French political crises as Swiss franc safe-haven flows increase. The JPY also tends to strengthen modestly. For non-eurozone members (Israel), the domestic currency weakens during political uncertainty — USD/ILS tends to rise 1-2% during election uncertainty peaks.

**Volatility markets.** The VSTOXX (European equity volatility index) rises during eurozone political crises, but typically by less than the VIX does during global risk events. The BTP basis volatility (implied vs. realized) is the most sensitive financial indicator of European political uncertainty. Relative-value volatility trades (long BTP vol, short Bund vol) provide a precise way to express a political uncertainty view without directional exposure to sovereign spreads.

**Credit market contagion.** Italy's crisis in 2018 produced measurable spread widening in Spain (+30bp peak) and Portugal (+40bp peak), even though those countries had clear single-party or stable coalition governments. The contagion was driven by two channels: risk-off positioning (investors selling all peripheral eurozone debt) and banking system interconnection (European banks with exposure to multiple peripheral sovereigns faced simultaneous mark-to-market pressure). Portugal, which had just emerged from its own financial assistance program, was particularly vulnerable to contagion.

## Further reading and cross-links

For the mechanical details of how sovereign credit works — duration, convexity, yield curve construction — see the [fixed income](/blog/trading/fixed-income) series, particularly the posts on sovereign credit risk and yield curve mechanics.

For the macro transmission of fiscal policy uncertainty into growth and inflation outcomes — the "policy channel" that underlies the spread move — see the [macro trading](/blog/trading/macro-trading) series, particularly the posts on fiscal multipliers and debt sustainability analysis.

For the quantitative tools used to measure political uncertainty — EPU index construction, event study methodology, and panel regressions of EPU on spreads — the [quantitative finance](/blog/trading/quantitative-finance) series covers the time-series econometrics framework.

For historical context on how debt crises have resolved or failed to resolve — including the Greek restructuring, the Argentinian defaults, and the Turkish lira crises — see the [geopolitical crises](/blog/trading/geopolitical-crises) series posts on debt crises and financial contagion.

The Caldara-Iacoviello Geopolitical Risk Index (GPR) provides a complementary political risk signal to the Baker-Bloom-Davis EPU index, focusing on war, terrorism, and geopolitical confrontation rather than policy uncertainty. During coalition crises, the EPU index is the more relevant measure; during war-related political shocks, the GPR is more directly applicable.

For practitioners seeking real-time data, the relevant Bloomberg functions include: BVWD (BTP-Bund spread history), WDCI (sovereign CDS history), MOVE (bond volatility index), and EUVI (VSTOXX). Reuters Eikon provides similar functionality. The EPU indices are freely available at policyuncertainty.com and update daily for major economies.

The political uncertainty premium in sovereign bonds is one of the more tractable forms of geopolitical risk for financial practitioners: it has a defined trigger (election producing hung parliament), a measurable quantity (spread widening), a clear resolution event (coalition formation), and a historically consistent compression pattern. Mastering its mechanics — understanding when the premium is peaking, when to fade it, and how to size the position for the risk-reward available — is one of the more durable edges available in European and global fixed-income markets.
