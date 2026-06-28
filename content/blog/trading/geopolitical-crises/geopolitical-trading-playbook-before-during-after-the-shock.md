---
title: "The Geopolitical Trading Playbook: Before, During, and After the Shock"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The complete three-phase practitioner playbook for trading geopolitical shocks: pre-positioning signals (satellite imagery, diplomat recalls, options skew), day-1 shock management, and the five-signal framework for distinguishing tail risks from mean-reversion opportunities."
tags: ["geopolitics", "trading-playbook", "risk-management", "volatility", "safe-havens", "kelly-criterion", "pre-positioning", "tail-risk", "mean-reversion"]
category: "trading"
subcategory: "Geopolitical Crises"
author: "Hiep Tran"
featured: true
readTime: 42
featured_image: "/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-1.png"
---

> [!important]
> **TL;DR** — Geopolitical shocks have three distinct trading phases, each requiring different strategies: pre-positioning captures the risk premium buildup; shock management on day 1 requires deliberately small sizing; and the five-signal framework determines whether you fade the spike or size for tail risk.
>
> - Pre-positioning signals with the longest lead time (satellite imagery 2-8 weeks, diplomat recalls 1-4 weeks) are the highest-quality alerts; options skew is the shortest-lead but partly priced signal.
> - Day 1 rule: "Everyone is wrong about magnitude." Size positions at 30-50% of normal on the day of the shock; add on the retest once the initial panic has been absorbed.
> - Five signals that determine tail vs. mean-reversion: nuclear risk, alliance cascade activation, economic decoupling, domestic political change in attacker, and whether the shock was priced in advance.
> - The modified Kelly criterion for geopolitical tail risk: use quarter-Kelly (never full Kelly), cap any single geopolitical bet at 10% of portfolio, use historical base rates not forecasts.

At 5:47 AM Eastern Time on February 24, 2022, Russian tanks crossed the Ukrainian border. The invasion that NATO intelligence services had been warning about for months — and that the market had partially priced as a "boy who cried wolf" scenario after multiple false alarms — was real. Brent crude opened up 8%. Gold was up 2.5%. European stocks fell 4%. The VIX spiked 31%.

And then, over the next 48 hours, the trading question became: is this a spike or a structural shift?

The answer was: both. Russian oil came partially off the market (not immediately — the initial Ukraine market reaction assumed Western sanctions would be severe; they were, but it took weeks). The initial 8% crude spike was partly right but partly excess. But the structural shift — the end of European energy dependence on Russia, the beginning of a multi-year defense spending surge, the acceleration of deglobalization — was real and permanent.

The traders who got rich on February 24 were not the ones who correctly predicted the invasion (though some did). They were the ones who had been pre-positioned based on the mounting signals (the satellite imagery of Russian troop buildups was public from mid-January; the US intelligence declassification of Russian invasion planning in early February was unprecedented; the Ukrainian bond market was already selling off sharply). And the ones who avoided ruin were those who on day 1 sized their positions at 30-50% of where their conviction suggested — because the first 48 hours of any geopolitical shock are characterized by maximum uncertainty and minimum information quality.

This post is the practitioner's complete playbook: how to read the signals before, how to trade the shock, and how to determine whether to fade or press after.

![Three-phase geopolitical trading playbook before during and after the shock with phase transitions](/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-1.png)

## Foundations: The Three-Phase Framework

Every significant geopolitical event follows the same three-phase market structure. Understanding this structure is more valuable than trying to predict which events will occur.

**Phase 1: Pre-Event (weeks to months before the shock)**

Markets rarely price geopolitical risk efficiently in advance. The reason is structural: professional analysts face the asymmetric incentive that false positives (predicting a war that doesn't happen) are more reputationally costly than false negatives (failing to predict a war that does happen). This creates systematic under-pricing of geopolitical tail risk in the months before major events.

The trader's edge in Phase 1 comes from reading the observable signals that precede geopolitical shocks — signals that are publicly available but rarely systematically synthesized into market positions. This includes satellite imagery (commercially available from Planet Labs, Maxar, ICEYE), diplomatic signaling (ambassador recalls, UN Security Council emergency sessions), economic leading indicators (reserve drawdowns, capital flow data), and market-based signals (options skew, CDS spread divergence, bond market moves in the target country).

**Phase 2: The Shock (day 0 to day 3)**

This is the phase of maximum uncertainty and minimum quality information. Prices move on headlines, social media, and fragmentary intelligence. The initial market moves are almost always wrong in magnitude (too large if the event is mean-reverting, too small if the event is structurally transformative). The professional's advantage in Phase 2 is psychological: the ability to maintain the 72-hour framework (see the Soleimani post), resist the urge to add to positions on day 1, and preserve dry powder for the retest.

**Phase 3: After (day 3 to weeks/months)**

This is where the five-signal framework determines the correct position. By day 3, enough information has usually emerged to distinguish a spike-and-revert from a structural shift. The Phase 3 trade is either a mean-reversion capture (if 0-1 cascade signals have fired) or a regime-change positioning (if 3+ signals have fired). Phase 3 trades have much higher conviction and can be sized more aggressively.

## Phase 1: Pre-Positioning — Reading the Signals

### The Eight Leading Indicators

![Pre-positioning signal checklist eight leading indicators with lead times reliability and action steps](/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-3.png)

**Signal 1: Satellite Imagery (lead time: 2-8 weeks)**

Commercial satellite imagery has become one of the most powerful tools for geopolitical intelligence since the 2010s. Companies like Planet Labs, Maxar Technologies, and ICEYE provide daily or near-daily coverage of virtually any point on Earth. What to look for:

- Unusual military vehicle concentrations near borders (the Russia-Ukraine invasion had visible buildup from December 2021)
- Hospital construction or blood bank mobilization near potential conflict zones
- Unusual construction activity at known nuclear facilities (DPRK enrichment expansions are regularly tracked this way)
- Port congestion in sanctioned countries (Iran oil loading patterns)

**The trading application:** When satellite imagery shows unusual military movement near a potential conflict zone, begin building small hedge positions: gold calls (3-month expiry, 5% OTM), oil futures (long), CDS on sovereign debt of potentially affected countries. These are insurance positions — cheap in calm periods, valuable if the event materializes.

**Signal 2: Diplomat Recalls (lead time: 1-4 weeks)**

Governments rarely publicly recall ambassadors without serious underlying cause. The withdrawal of diplomatic staff from a country signals a deterioration in bilateral relations severe enough that governments are either: (a) following protocol that precedesvotes/military action, or (b) protecting their diplomatic personnel from potential hostility. The US withdrawal of diplomats' families from Ukraine in January 2022 was a clear and public signal.

**Signal 3: Central Bank Reserve Drawdown (lead time: 4-12 weeks)**

Central banks in countries expecting conflict or sanctions often pre-position their foreign exchange reserves:
- Moving reserves from US-held accounts to gold or Yuan accounts (Russia did this aggressively in 2021, reducing its exposure to potential US Treasury freezes)
- Accelerating reserve drawdown before expected currency pressure
- Borrowing in USD before expected sanctions cutoff

The IMF's International Reserves data, released monthly with a 6-week lag, tracks aggregate reserves. More timely data comes from central bank weekly statistical releases. Russia's pre-invasion reserve shuffle was visible in the data.

**Signal 4: Leadership Reshuffle in Target Country (lead time: 2-6 weeks)**

When potential attack or crisis countries suddenly shuffle key posts — defense ministers, central bank governors, intelligence chiefs — it often signals internal preparation for a crisis. The Ukraine government's emergency "unity government" decisions in early 2022, while not fully public, were visible to well-networked intelligence analysts.

**Signal 5: Bond Market Move in Target Country (lead time: 1-3 weeks)**

Domestic bond markets in countries facing geopolitical risk tend to move before equity markets. Domestic investors, who are closer to the political reality, sell bonds as uncertainty rises. The Ukrainian hryvnia bond market sold off sharply in January-February 2022, a clearer signal than the equity market which was partly supported by foreign speculative buyers.

**Signal 6: Options Skew (lead time: days to 1 week)**

When sophisticated options traders begin expecting a geopolitical event, they buy protective puts on affected assets — crude oil, gold, equity indices of affected countries. This shows up as a steepening of the put-call skew. Before the Russia-Ukraine invasion, VIX term structure had inverted (near-term vol higher than far-term) and the crude oil skew was heavily tilted toward upside calls. These were market-based confirmations of the satellite/diplomatic signals.

**Signal 7: CDS Spread Widening on Sovereign Debt (lead time: 1-2 weeks)**

Credit default swap spreads on sovereign debt — which essentially price the probability of default — widen when smart money believes a country is facing economic pressure from potential conflict or sanctions. Russian sovereign CDS widened from approximately 90bp to 250bp in the weeks before the Ukraine invasion. This was a clear market signal.

**Signal 8: Nationalist Media Messaging (lead time: 4-12 weeks)**

State-controlled media in authoritarian countries often escalates nationalist rhetoric before geopolitical action. This is partly to build domestic public opinion support, partly to test international reaction. Chinese state media's rhetoric on Taiwan, Russian state television's messaging on Ukraine, and Iranian press coverage of nuclear issues all showed escalation patterns before major events. The signal is noisy (nationalist rhetoric doesn't always precede action) but adds context to other signals.

### Pre-Positioning Trade Construction

When multiple signals align (3+ of the eight are flashing), the pre-positioning trade should be:

**Size:** Small — 1-3% of portfolio. This is a probability-weighted position, not a conviction trade.

**Instruments:** Options preferred (defined downside, unlimited upside) over futures (unlimited downside). Specifically:
- Brent crude calls (1-3 month expiry, 5-10% OTM): benefits from supply disruption premium
- Gold calls (1-3 month expiry, 3-5% OTM): safe haven demand
- VIX calls (1-month expiry, 30% strike): volatility expansion trade
- Put protection on likely-to-be-affected equity indices

**Invalidation:** If the signals reverse (satellite imagery shows troop withdrawal, diplomats return, CDS tightens), close the pre-positioning trade. The option premium is lost, but it was sized for this outcome.

## Phase 2: The Shock — Day 1 and the Day 3 Rule

### Day 1: Everyone Is Wrong About Magnitude

The most reliable insight about day-1 geopolitical shock trading is this: no one has full information, and the initial market moves are driven by a combination of algorithmic reactions (stop-loss triggers, volatility-targeting mandates that force de-risking), retail panic, and professional uncertainty.

This creates a systematic pattern: the initial move is almost always larger than the "correct" move based on fundamental analysis — in either direction. In spike scenarios (Soleimani, Arab Spring incidents, Crimea), the initial spike is larger than the ultimate resting level. In structural scenarios (Russia-Ukraine invasion), the initial move is often smaller than the ultimate level because markets underestimate the duration.

**Day 1 rule: Size at 30-50% of target position.**

If your analysis suggests a crude oil long of \$2M notional, enter \$600K-1M on day 1. The remaining \$1-1.4M is available for the retest. This approach:
1. Ensures participation if the move is real
2. Preserves dry powder for the more information-rich retest
3. Limits loss if the initial move fully reverses (as in Soleimani)

**Day 1 instruments:** Liquid, exchange-traded instruments only. Geopolitical shock days are characterized by illiquidity in OTC markets, wide bid-ask spreads in corporate bonds and EM assets, and credit facility drawdowns by leveraged players. Stay in futures, listed ETFs, and major currency pairs.

**The "Day 1 errors" to avoid:**
- Going full size immediately
- Buying illiquid assets on day 1 (you may not be able to exit)
- Assuming the initial move direction is correct (be willing to reverse if the 72-hour watch shows opposite signals)
- Trading on emotion (the news is dramatic; the position should be mechanical)

### Day 3: The Fade or Add Decision

By day 3, the 72-hour framework has produced enough information to make a higher-confidence decision.

![VIX spike versus recovery time scatter plot showing the spike-fade versus tail risk quadrant split](/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-2.png)

The scatter plot above reveals the market structure: events in the upper-left quadrant (high VIX spike, fast recovery) are the systematic fade trades. Events in the lower-right quadrant (moderate VIX spike, slow recovery) require more patience. The COVID pandemic outlier (VIX +82%, 180-day recovery) sits far off this chart — it was not a geopolitical shock but a structural global event.

**Day 3 decision tree:**

Evaluate the five cascade signals. If 0-1 signals have fired:
- **Fade the spike:** Sell oil/gold/volatility positions from pre-positioning. Add long positions in the assets that were sold off on day 1 (S&P 500, EM equities in non-affected regions, risk assets generally). Enter the mean-reversion trade.

If 3+ signals have fired:
- **Add to positions:** The structural scenario is materializing. Add to pre-positioning positions. Consider longer-dated options (3-6 month expiry) to avoid theta decay while waiting for the full repricing.

If 2 signals have fired (ambiguous):
- **Hold and monitor.** Don't fade aggressively. Add minimally. Wait for signal clarification, which typically comes within 24-48 additional hours.

## Phase 3: After — Distinguishing Tail from Mean Reversion

### The Five-Signal Framework

![Five signal framework tail risk versus mean reversion decision tree for geopolitical events](/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-5.png)

**Signal 1: Nuclear Risk or Posture Change**

This is the binary gate. If any nuclear-capable state changes its nuclear posture — alters alert levels, moves weapons, changes command authority structures, changes declaratory policy — the tail risk scenario is real regardless of other signals. Nuclear posture changes are extremely rare (last significant one: 1983 Able Archer crisis) and have not occurred in recent events (Soleimani, Arab Spring, Ukraine initial invasion).

**How to check:** US Strategic Command and NATO regularly comment on nuclear posture publicly when it changes. Nuclear threat initiative reports, Arms Control Association updates, and analyst networks (Carnegie Endowment for International Peace) are reliable sources.

**Signal 2: Alliance Cascade Activation**

Has any country with formal mutual defense obligations begun military mobilization? Not verbal condemnation — mobilization. Mobilization orders are publicly announced (countries need to explain to their own populations why reserves are being called up) and often visible in transportation/logistics changes (military aircraft movements, ship deployments).

**How to check:** Defense ministries publish mobilization orders. Military aviation tracking (FlightRadar24 has military mode-S data). Port authority data. Congressional Research Service reports.

**Signal 3: Economic Decoupling**

Are countries implementing trade restrictions, asset freezes, financial sanctions, or supply chain decoupling actions that are structural rather than symbolic? The distinction: Russia's oil export ban from the EU (structural, required multi-year supply chain adjustment) vs. the US freezing Afghan government assets (symbolic gesture with minimal economic impact on the US). Structural economic decoupling creates lasting repricing of supply chains, commodity prices, and bilateral trade flows.

**How to check:** OFAC (US Treasury Office of Foreign Assets Control) sanction announcements. EU Official Journal sanction decisions. WTO dispute filings. Import/export data in affected sectors.

**Signal 4: Domestic Political Change in the Attacker Country**

Does the geopolitical action create a domestic political crisis for the country that initiated it? In democratic systems, significant casualties or costs from geopolitical actions can create legislative pushback, public opinion reversal, and electoral consequences. This is an important check because it determines whether the attacker is likely to escalate further or de-escalate to manage domestic costs.

**How to check:** Legislative reactions (has the party in power lost a parliamentary vote?). Polling data (approval ratings in the days after the event). Protest size and location. Elite defections (former officials publicly criticizing the action).

**Signal 5: Was the Shock Priced?**

If markets had significantly priced in the event before it happened, the aftermath will be "sell the news" even if the event is structurally significant. The Russia-Ukraine invasion was partly "priced" (Russian stocks had been underperforming, MOEX CDS was at 250bp) but the magnitude was understated. The initial crude spike of 8% partially reflected the amount of "not priced" in the event.

**How to check:** Compare pre-event levels of crude oil, gold, VIX, and target-country assets to their 90-day average. Large deviations from average indicate pre-event pricing. CDS levels in the weeks before the event.

### Geopolitical Risk Index and Market Overlay

![Geopolitical Risk Index GPR by major event showing spikes above 200 as extreme risk zone](/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-6.png)

The Caldara-Iacoviello Geopolitical Risk Index (GPR) provides a monthly academic measure of geopolitical risk based on newspaper text analysis. It's valuable for two purposes:

1. **Regime identification:** When GPR is above 200 (extreme risk zone), geopolitical risk has a measurable drag on EM equity returns of approximately 3-5% per month above the GPR-calm baseline. This is the environment to reduce EM equity weights systematically.

2. **Mean reversion:** After GPR spikes above 200 and then falls below 150, the historical pattern shows EM equity outperformance in the following 6-12 months as the fear premium dissipates. This creates a systematic "geopolitical risk mean reversion" alpha.

## Safe Haven Asset Performance: The Playbook Data

![Safe haven asset average 7-day returns after major geopolitical events showing gold JPY USD performance](/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-4.png)

The safe haven performance data across 10+ major geopolitical events since 2001 confirms the systematic patterns:

**Gold (+2.3% average 7-day return)** is the most consistent geopolitical hedge. It benefits from: flight-to-safety demand, USD weakness (when geopolitical risk undermines US policy credibility), and store-of-value demand in countries near the crisis. Gold's hedge effectiveness is partly behavioral (investors globally think of gold as "safe" in crises) and partly fundamental (gold cannot be sanctioned, frozen, or devalued by any single government).

**Japanese Yen (+1.4% average 7-day return)** is the most reliable currency safe haven. Japan runs persistent current account surpluses, so its investors are natural buyers of domestic yen in risk-off environments. The yen also benefits from "carry unwind" — yen-funded carry trades are unwound (JPY bought back) when volatility spikes.

**US Dollar (+1.1% average 7-day return)** benefits from flight to the deepest, most liquid financial markets in the world. However, the USD safe-haven effect has been weakening as geopolitical shifts increase reserve diversification away from dollar assets.

**US Treasuries (yield falls -0.18pp average)** — the flight-to-quality bid. Note this is the 10-year Treasury yield *falling*, meaning bond *prices rising*. Long-duration Treasuries are a geopolitical hedge, but the correlation is lower than gold or JPY because rate expectations dominate the treasury market in high-inflation regimes.

**EM Equities (-3.8% average 7-day return)** — the consistent loser in geopolitical shocks. EM assets suffer from direct exposure risk, capital outflow as investors de-risk to developed markets, and commodity cost shocks (for commodity-importing EMs).

## Position Sizing: The Modified Kelly Framework

![Kelly criterion adaptation for geopolitical tail risk with quarter-Kelly and portfolio caps](/imgs/blogs/geopolitical-trading-playbook-before-during-after-the-shock-7.png)

The Kelly Criterion provides a mathematical framework for optimal position sizing. The standard formula is:

f* = (p × b - q) / b

Where p = probability of winning, q = 1-p, b = ratio of win to loss size.

For geopolitical trades, the inputs are:

**For spike-and-revert trades (Soleimani type):**
- p = 0.90 (90% historical base rate for mean reversion within 10 days)
- Win = +5% (selling the spike, capturing volatility premium)
- Loss = -10% (cascade materializes, tail risk)
- b = 0.5
- f* = (0.90 × 0.5 - 0.10) / 0.5 = (0.45 - 0.10) / 0.5 = 0.70

Full Kelly says 70% of portfolio — obviously wrong for a trade with genuine tail risk. This illustrates why full Kelly is never appropriate for geopolitical risk.

**Quarter-Kelly = 17.5% of portfolio.** This is still too concentrated for a single geopolitical event. Apply an additional "concentration cap" of 10% maximum.

**For tail risk scenarios (Russia-Ukraine type):**
- p = 0.60 (you have some evidence this is structural but significant uncertainty remains)
- Win = +30% (long oil and short EUR over 6 months)
- Loss = -15% (de-escalation materializes, unwind losses)
- b = 2.0
- f* = (0.60 × 2.0 - 0.40) / 2.0 = (1.20 - 0.40) / 2.0 = 0.40

Full Kelly says 40% — again, cap at 10% given estimation uncertainty.

**The practitioner rule:** Use quarter-Kelly as your starting point, cap at 10% of portfolio, phase into the position over 3 days (day 0, day 2, day 5 retest), and review after each of the five signal checks.

## The Complete Playbook: Worked Through Russia-Ukraine

#### Worked example:

**Russia-Ukraine Invasion: Three-Phase Playbook (Retrospective)**

*Phase 1: Pre-Positioning (January 15 - February 23, 2022)*

Signals active by January 15:
- Satellite imagery: Russian military equipment concentrations at Belgorod, Kursk, Bryansk (visible on Maxar/Planet images shared publicly)
- Diplomat signals: UK advised all British nationals to leave Ukraine; US ordered embassy family departures
- Bond market: Ukrainian hryvnia bonds yielding 15%+, 3-month highs for yields
- CDS: Russian sovereign CDS at 250bp, Ukraine at 900bp (from 200bp baseline)
- Options skew: Brent crude upside calls being bought aggressively, skew steepening

Signal count: 4 of 8 → Build pre-positioning.

Pre-positioning trade (entered January 15):
- Long Brent crude calls (March expiry, \$80 strike, crude at \$83): \$400,000 notional = 2% of \$20M portfolio
- Long gold (March futures, gold at \$1,820): \$200,000 = 1% of portfolio
- Short MOEX (Russian equity ETF) put to offset cost: -\$100,000

*Phase 2: Day of Invasion (February 24, 2022)*

Pre-positioning positions at open:
- Crude call value: +8% oil move → call position approximately +40% = +\$160,000
- Gold: +2.5% = +\$5,000
- MOEX short: +20% (MOEX halted, ETF tracking MOEX down 20%) = +\$20,000

Day 1 action:
- Add to crude long: Brent at \$105, add \$1,000,000 futures long (5% of portfolio) — total crude now 7% of portfolio
- Add gold: \$400,000 (2% more) — total gold 3% of portfolio
- Short EUR/USD: \$500,000 (European energy import cost headwind) — 2.5% of portfolio
- Total Phase 2 additions: 9.5% of portfolio at 50% of conviction sizing

Remaining dry powder: approximately 40% in cash-equivalents for Phase 3 decisions

*Phase 3: Five-Signal Check (February 24-27, 2022)*

Signal 1 — Nuclear: No posture change confirmed → NO
Signal 2 — Alliance cascade: NATO mobilizing but Article 5 NOT triggered (Ukraine not a member) → PARTIAL (1 point)
Signal 3 — Economic decoupling: EU energy sanctions announced, SWIFT disconnection of major Russian banks → YES (1 point)
Signal 4 — Domestic political change: Putin public approval actually rising in Russian polling → NO
Signal 5 — Was it priced?: Partially (Russian CDS at 250bp, MOEX at -25% from highs) → PARTIALLY PRICED

Score: 2/5 signals → Tail risk, not full cascade. This is a structural repricing trade, not a "fade on day 3" trade.

Phase 3 additions (February 28):
- Hold crude long (structural supply disruption confirmed)
- Hold short EUR/USD (energy dependency confirmed)
- Add long gas futures (TTF European gas) — \$500,000 = 2.5% of portfolio
- Add short Bund (German government bonds) — European fiscal expansion for defense → \$300,000

Total portfolio geopolitical positioning: approximately 25% of portfolio, diversified across oil, gold, FX, bonds.

*Outcome through June 2022:*
- Crude long: Brent peaked at \$128 in March → approximately +30% from average entry of \$96 = +\$300,000 on \$1M position
- Gold: peaked at \$2,050 → approximately +12% = +\$48,000 on \$400K
- EUR/USD short: EUR fell from 1.132 to 0.985 → approximately +13% = +\$65,000 on \$500K
- TTF Gas: 75 → 310 EUR/MWh → approximately +313% on the futures = +\$1,565,000 on \$500K
- Short Bund: German 10yr from 0.2% to 1.7% → approximately -12.5 price points on Bund futures = +\$37,500 on \$300K

**Total geopolitical playbook P&L: approximately \$2,015,000 on \$20M portfolio = +10.1% from geopolitical positioning alone over 4 months.**

Key to success: the pre-positioning captured the initial spike; the Phase 2 additions at moderate size captured the structural repricing; and the Phase 3 five-signal analysis correctly identified this as a structural shift rather than a spike-and-revert.

## Common Misconceptions About Geopolitical Trading

**Misconception 1: "Geopolitical events are random and unpredictable."**
They are not. As the eight leading indicators show, most major geopolitical events have observable precursors that are publicly available. The Russia-Ukraine invasion had 6+ weeks of public satellite imagery, explicit US intelligence warnings, and clear market signals. The challenge is synthesis, not information access.

**Misconception 2: "You need insider information to trade geopolitical events."**
Insider information is illegal. Everything described in this playbook uses publicly available data: commercial satellite imagery, diplomatic announcements, central bank reserve data, options market data, CDS spreads. The alpha comes from synthesis and systematic application, not from inside knowledge.

**Misconception 3: "Safe haven trades always work in geopolitical crises."**
Gold and JPY work reliably in geopolitical crises. US Treasuries work *except* when the crisis is specifically about US fiscal credibility (the Truss crisis showed this; a US fiscal crisis would not drive Treasury rallies). The USD safe-haven effect is weakening as de-dollarization gradually reduces the reflex bid for dollar assets in crises.

**Misconception 4: "The biggest geopolitical risks are the ones that move markets most."**
Not necessarily. The 9/11 attacks — the most significant geopolitical event since WWII in terms of historical consequence — caused the S&P 500 to fall approximately 12% and recover within 3 months. The European sovereign debt crisis of 2010-2012 — a "boring" fiscal negotiation with no bombs or assassinations — created far longer-lasting market impacts across European assets. Market impact depends on economic transmission channels, not on dramatic footage.

**Misconception 5: "Once a geopolitical event is 'over,' the trade is over."**
The tail effects of geopolitical events can last years. The Russia-Ukraine invasion restructured European energy markets for the 2020s. China-US tech decoupling is creating decade-long supply chain changes. The phase 3 "structural shift" positions are often held 2-4 years, not days.

#### Worked example:

**The Geopolitical Overlay Portfolio: Systematic Implementation**

For a macro fund with \$100M AUM that wants to systematically capture geopolitical risk premium:

**Annual budget allocation:**
- Pre-positioning options premium: 1.5% of AUM = \$1,500,000 per year
- Tactical geopolitical positions (Phase 2-3): Up to 15% of AUM at any time = \$15,000,000 maximum

**Systematic process:**

Monthly: Run the eight-signal scoring matrix across 20 potential geopolitical hotspots (Taiwan Strait, Korean Peninsula, Iran nuclear, Russia NATO border, India-Pakistan, South China Sea, Middle East, EM political crises, etc.)

When a hotspot scores 3+ on the eight-signal checklist: Begin building pre-positioning (0.5-1.5% of AUM) using the cheapest available options

When a hotspot scores 5+ signals: Full pre-positioning build (1.5% AUM)

On event materialization: Phase 2 rules — size at 30-50% of target, use the five-signal day-3 framework

**Expected annual contribution of geopolitical overlay:**
Based on historical data of 2-4 significant geopolitical events per year, average pre-positioning gain of 15-25% when events materialize (options purchased 2-6 weeks before events), and successful Phase 2-3 trades on 60-70% of material events: estimated contribution of 1.5-3.0% annual alpha above benchmark.

This is a systematic, process-driven approach to capturing geopolitical risk premium — not a prediction business, but a systematic reading and positioning business.

#### Worked example:

**Sizing the Ukraine Pre-Positioning Trade with Modified Kelly**

Entry date: January 15, 2022. Portfolio size: \$20M. Signal count: 4/8.

Step 1 — Assess probability of material event by Feb 24:
- US intelligence declassified briefings had publicly warned of "imminent" invasion risk
- Satellite imagery showing 100,000+ troops was public
- Market-based signals confirmed (CDS, bond yields)
- Estimated p(invasion by March 1, 2022) = 60%

Step 2 — Assess win/loss:
- If invasion occurs: crude oil calls +40-60%, total position gain +\$180,000-240,000
- If no invasion: options expire worthless, total loss -\$400,000 (options premium spent)
- b = 200,000 / 400,000 = 0.5

Step 3 — Kelly calculation:
f* = (0.60 × 0.5 - 0.40) / 0.5 = (0.30 - 0.40) / 0.5 = -0.20

**Negative Kelly!** This means the expected value is negative with these inputs. Should we not trade?

Step 4 — Sensitivity check: The Kelly result is sensitive to the probability estimate. At p = 70%:
f* = (0.70 × 0.5 - 0.30) / 0.5 = (0.35 - 0.30) / 0.5 = 0.10 = 10% of portfolio

At p = 65% (splitting the difference): f* = 0.05 = 5%.

Given the uncertainty in the probability estimate, use 2-3% of portfolio (conservative position size justified by signal count of 4/8, not 6/8). This is consistent with the risk budget allocation of 1.5% AUM maximum for pre-positioning.

**Decision:** Enter at 2% of portfolio (\$400,000 notional in crude calls), acknowledging this is a probability-weighted insurance position, not a conviction trade.

## Further Reading and Cross-Links

This playbook integrates concepts from across the *Geopolitical Crises & Markets* series:

- **[Political assassination and market reaction](./political-assassination-and-market-reaction-soleimani-khashoggi)** — the Soleimani case study is the canonical Phase 2-3 fade-the-spike trade
- **[Political business cycle](./the-political-business-cycle-and-fiscal-timing)** — the presidential cycle creates predictable periods of higher geopolitical risk-taking (election year foreign policy assertiveness)
- **[South China Sea and Vietnam risk premium](./the-south-china-sea-dispute-and-vn-market-risk-premium)** — applying pre-positioning to a structural geopolitical risk
- **[Series capstone: Complete playbook](./geopolitical-crises-and-markets-the-complete-playbook-capstone)** — the full 6×5 asset sensitivity matrix that provides the quantitative framework for this playbook

For volatility trading mechanics (VIX options, variance swaps), see the **[Options and Volatility series](../options-volatility/)**. For systematic portfolio construction with geopolitical overlays, see the **[Risk Management series](../risk-management/)**.

The three-phase geopolitical trading framework is not a prediction machine — no one can reliably predict which specific events will occur or when. It is a systematic process for reading observable signals, sizing positions appropriately for uncertainty, and applying a consistent decision framework across the chaos of day-1 shock periods. The traders who consistently outperform during geopolitical crises are not those who called the event in advance. They are those who had a process, sized appropriately, and maintained discipline when the headlines were at their most frightening.
