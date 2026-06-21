---
title: "The Geopolitical Risk Premium: What Markets Price In and What They Miss"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How financial markets discount geopolitical uncertainty, why they consistently get it wrong, and the three-phase cycle every investor must understand before placing a single trade."
tags: ["geopolitics", "risk-premium", "gpr-index", "vix", "safe-havens", "ukraine", "sanctions", "event-trading", "geopolitical-crises", "market-structure"]
category: "trading"
subcategory: "Geopolitical Crises"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Markets price *expected* geopolitical outcomes, not worst-case ones — and they do it badly when outcomes are binary and timelines are uncertain.
>
> - The "geopolitical risk premium" is the extra return investors demand for holding assets exposed to political and conflict uncertainty.
> - The GPR Index (Caldara-Iacoviello) hit 345 after Russia invaded Ukraine in February 2022 — its second-highest reading since 9/11's 326.
> - Most geopolitical shocks are *spikes*: VIX surges 8–45% and recovers within days to weeks. Ukraine's VIX recovered in 45 days. But Ukraine's energy regime change was permanent — European TTF gas went from 25 EUR/MWh in 2021 to 310 EUR/MWh peak in August 2022.
> - Markets systematically under-price slow-building geopolitical risks and over-price fast visible shocks — the opposite of rational discounting.
> - The three-phase cycle (shock → discount → fade or regime lock-in) is the master framework for timing a geopolitical trade.

On the morning of February 24, 2022, Brent crude opened at \$96.40 a barrel. By the end of the week it was at \$110. By June it had touched \$120. But here is the more interesting number: European natural gas (the TTF benchmark) traded at roughly 25 EUR per megawatt-hour in early 2021. By August 2022 it hit 310 EUR/MWh — a 12x increase. Today, in mid-2026, it sits at 39 EUR/MWh, a fraction of that peak, but European LNG import capacity has permanently tripled, Russian pipeline flows to Europe have dropped from 201 billion cubic meters per year to under 8, and the continent's entire energy infrastructure has been restructured in 36 months.

The VIX — Wall Street's "fear gauge" — spiked from 28 to nearly 34 in the days after the invasion. Then it recovered. Fully. In about 45 days. Equity markets absorbed the political shock of the worst war in Europe since 1945 in roughly six weeks. The energy market did not. That divergence is the entire lesson of geopolitical risk: two things happened simultaneously, in the same event, and they were completely different animals. One was a spike. The other was a regime change. And most investors treated them identically.

![Geopolitical Event to Asset Repricing: The Transmission Chain](/imgs/blogs/geopolitical-risk-premium-what-markets-price-in-1.png)

The diagram above is the master mental model for this entire series. A geopolitical shock does not go directly from "event" to "asset price." It travels through a chain: political actors decide how to respond, those decisions activate specific economic channels (sanctions, tariffs, energy flows, capital controls), and only then do asset prices reprice to a new equilibrium. Each stage in that chain adds uncertainty, adds delay, and adds the possibility of partial or no transmission. Understanding which stage you are watching — and how far along the chain the market has already priced — is the job.

## Foundations: How Markets Price Uncertainty

Before we get to geopolitics specifically, we need to understand how financial markets price *any* uncertain future outcome. This is not obvious, and getting it wrong is the most common beginner mistake.

### The expected value principle

Markets do not price the worst-case scenario. They price the *probability-weighted average* of all scenarios. This is called expected value pricing, and it is the foundation of everything.

Here is a simple example. You hold a stock worth \$100 today. There is a 20% chance that a tariff announcement will knock it to \$80, and an 80% chance that nothing happens and it stays at \$100. What should it trade for?

Expected value = (0.20 × \$80) + (0.80 × \$100) = \$16 + \$80 = \$96.

So the stock should trade at \$96 *right now*, before the announcement, reflecting the 20% probability of the bad outcome. The market has already "priced in" a \$4 discount.

This matters enormously for geopolitical events, because:

1. If the bad scenario is 30% probable, the discount is larger (\$94).
2. If the bad scenario is 50% probable, the discount is even larger (\$90).
3. If the bad scenario has already happened, there is no more discount to price — everything is already known.

The classic punchline: "buy the rumor, sell the news." When a geopolitical event is well-flagged in advance, by the time it actually occurs, markets have already partially or fully priced the expected outcome. The *event* itself is not the price signal — the *probability shift* around the event is.

### What a "risk premium" actually is

A *risk premium* is the extra return that investors demand above a risk-free rate to compensate for bearing a specific risk. If a 10-year US Treasury bond (essentially risk-free, backed by the US government) yields 4%, and a 10-year Argentine government bond yields 14%, the 10 percentage-point difference is Argentina's risk premium — the extra compensation investors require for the possibility that Argentina might default (again).

A *geopolitical risk premium* is the same concept applied to political and conflict uncertainty. When geopolitical risk rises, investors require higher expected returns from exposed assets. To deliver those higher returns without assets magically generating more cash flow, prices must fall. Lower prices today → higher expected returns going forward → geopolitical risk premium embedded.

The premium shows up differently across asset classes:

- **Equities**: lower P/E multiples (investors pay less per dollar of earnings because future earnings are more uncertain)
- **Bonds**: higher yield spreads (lenders charge more because repayment is less certain)
- **Currencies**: weaker exchange rates (capital flees countries with elevated political risk)
- **Commodities**: higher spot prices for supply-disruption-sensitive goods (oil, wheat, critical minerals)

### The discounting mechanism in practice

Financial markets are *forward-looking* aggregating machines. Every trade reflects a buyer's and seller's best estimate of the future, incorporating all publicly known information. When a geopolitical risk emerges — say, military buildups on a border — professional investors immediately start asking:

1. What is the probability of escalation? (20%? 50%? 80%?)
2. If it escalates, what happens to the specific assets I hold? (Which sectors, which currencies, which commodities get disrupted?)
3. What is the *expected* impact, probability-weighted across all scenarios?
4. Has that expected impact already been priced, or is there a discovery gap?

The market is often reasonably good at pricing known risks with continuous distributions (a flood might cause 5–15% damage to property values; the market prices the midpoint). It is historically bad at pricing *binary* risks — situations where the outcome is either relatively benign or catastrophically bad, with little middle ground. War is the canonical binary risk: either the invasion happens or it doesn't; either the capital is captured or it isn't. Markets struggle with these because the probability estimate itself is deeply uncertain.

## The GPR Index: Measuring Geopolitical Fear

In 2022, economists Dario Caldara and Matteo Iacoviello published a measure called the Geopolitical Risk (GPR) Index. They built it by counting the frequency of news articles about geopolitical risks — wars, terrorism, tensions between nations — in major newspapers, then normalizing the count so the average value across the 20th century equals 100.

The index is important for three reasons:

1. It is *systematic* — not based on anyone's subjective judgment, just the observable pace of negative geopolitical news coverage.
2. It goes back to 1899, giving over 120 years of history.
3. It has been empirically linked to investment slowdowns, higher equity risk premia, and increased safe-haven demand.

![GPR Index 2001-2025: Geopolitical Risk Spikes and Baselines](/imgs/blogs/geopolitical-risk-premium-what-markets-price-in-2.png)

Look at the GPR history since 2001. Three peaks dominate:

- **September 2001**: GPR = 326. The 9/11 attacks were unprecedented in the modern era — the US homeland had been struck, and nobody knew whether a larger series of attacks was coming.
- **March 2003**: GPR = 254. The Iraq War invasion, preceded by months of elevated diplomatic tension, UN inspections, and explicit US war preparation.
- **February 2022**: GPR = 345. Russia's full-scale invasion of Ukraine — the largest land war in Europe since 1945.

Notice that the Crimea annexation of 2014 barely registers at 161, and the Gaza war of October 2023 peaked around 188. These numbers carry a message: markets assess geopolitical risk based on *scope of disruption*, not just on the moral or human weight of an event. A high-casualty conflict that is geographically contained and not connected to major trade or energy routes produces a modest GPR spike. A conflict that threatens a key energy corridor or involves nuclear-armed great powers produces a massive one.

After every spike in the GPR Index, there is a decay. Risk fades from headlines. New equilibria establish themselves. Investors refocus on earnings, interest rates, and economic data. The GPR's natural tendency to mean-revert is one of the most useful facts in geopolitical trading: most crises feel worse at the moment than they turn out to be, and the *ex-post* asset impact is usually smaller than the *ex-ante* fear.

### What GPR tells us about expected returns

The academic work by Caldara and Iacoviello, and subsequent studies, found a consistent empirical relationship: when the GPR Index doubles (say, from 100 to 200), the forward equity risk premium rises by roughly 0.7–0.9 percentage points. This sounds small. Here is what it means in price terms.

#### Worked example:

Suppose a stock market index trades at a P/E multiple of 20x (meaning investors pay \$20 for every \$1 of current earnings). The equity risk premium is 5.0% — the extra return above the risk-free rate that equities are expected to deliver. Now the GPR Index doubles from 100 to 200 due to an escalating conflict.

If the risk premium rises by 0.8 percentage points (from 5.0% to 5.8%), what happens to the P/E multiple?

The P/E multiple is roughly the inverse of the required rate of return. At a 5.0% equity risk premium plus a 4.0% risk-free rate, the total required return is 9.0%, implying a P/E of approximately 1 ÷ 0.09 ≈ 11.1 on a "earnings yield" basis. But in practice, P/E incorporates growth expectations, so let's use the Gordon Growth Model simplification:

P/E = 1 ÷ (required return − growth rate) = 1 ÷ (0.09 − 0.04) = 1 ÷ 0.05 = 20x

Now with the elevated risk premium: P/E = 1 ÷ ((0.09 + 0.008) − 0.04) = 1 ÷ (0.058) ≈ 17.2x

The market falls from 20x to 17.2x earnings — a decline of about 14%. On a \$10,000 investment, that is a \$1,400 loss, purely from the geopolitical risk premium expansion, before any actual change in corporate earnings.

**Intuition**: The GPR Index doubling doesn't mean catastrophe is certain. It means uncertainty has doubled. Investors demand more compensation for that uncertainty. Higher required returns mean lower prices today. The math is mechanical, not emotional.

## Safe Havens: Where Money Goes When Geopolitics Bites

When geopolitical risk rises sharply, capital doesn't just sit still — it moves. Predictably. Investors sell risky assets and buy safe assets, a pattern so consistent it has its own name: the *flight to safety*.

The data across the top 10 geopolitical shocks since 2001 shows clear patterns:

![Safe Haven vs Risk Asset Reaction to Geopolitical Shocks](/imgs/blogs/geopolitical-risk-premium-what-markets-price-in-3.png)

Reading the average 7-day moves:

**Safe havens (positive moves)**:
- Gold: +2.3% — the canonical geopolitical hedge; no counterparty risk, 5,000-year history as a store of value
- JPY (Japanese yen): +1.4% — Japan is a net creditor nation; when global uncertainty rises, Japanese investors repatriate overseas assets, boosting the yen
- USD Index (DXY): +1.1% — the dollar benefits from global risk-off moves, as USD is the world's reserve currency and settlement currency for most commodity trades
- CHF (Swiss franc): +0.9% — Switzerland's political neutrality, large current account surplus, and bank secrecy tradition make CHF a classic safe haven

**Risk assets (negative moves)**:
- Emerging market equities: -3.8% — EM is doubly exposed: both equity risk and currency risk
- S&P 500: -2.4% — the global equity benchmark sells off, though less than EM
- EM FX index: -1.9% — currencies of emerging economies weaken as capital flees to safe havens

**Oil (positive, for a different reason)**:
- Brent crude: +3.1% — note this is NOT a safe haven — oil rises because geopolitical events often threaten supply routes

Understanding *why* each asset moves is more important than knowing *that* it moves. Safe havens rise because they represent lower risk, lower counterparty dependency, or established stores of value. Risk assets fall because their cash flows are exposed to economic disruption. Oil rises because it is the commodity most sensitive to supply-side geopolitical disruptions.

These are *averages* across diverse events. Individual events deviate sharply — the direction and magnitude depend on the specific geographic and economic exposure. An attack on a Middle Eastern oil facility spikes oil more and equities less than a conflict in a commodity-poor region.

## The Three-Phase Cycle: How Geopolitical Events Play Out in Markets

The most useful framework for navigating geopolitical shocks is what I call the three-phase cycle. Understanding which phase you are in changes what trades make sense.

![The Three-Phase Geopolitical Market Cycle](/imgs/blogs/geopolitical-risk-premium-what-markets-price-in-7.png)

### Phase 1 — The Shock (Days 0–3)

The event happens. Markets reprice instantly, without full information, under high uncertainty. Volume surges. Bid-ask spreads widen. VIX spikes 15–45%. Safe havens are bought regardless of valuation. Risk assets are sold regardless of fundamentals.

This is *panic pricing* — not analytical pricing. The market is not computing expected values carefully; it is moving capital away from uncertainty as fast as possible. Prices overshoot their rational equilibrium in both directions: safe havens become too expensive, risk assets become too cheap.

For a practitioner, Phase 1 is usually not the time to trade *toward* the event. Spreads are wide, liquidity is thin, and you are competing against market makers who are also managing their own risk. The edge is small. The transaction costs are high.

### Phase 2 — The Discount (Days 3–30)

Information accumulates. The initial panic fades. Markets begin to calculate more carefully: What is the actual probability of escalation? Which specific supply chains are disrupted? What is the policy response? How fast can substitutes emerge?

VIX begins to fall. Spreads narrow. Investors who bought safe havens in Phase 1 start to question whether they overpaid. Risk assets begin to recover — but unevenly, based on actual exposure. A German auto manufacturer is more exposed to a European energy crisis than a US tech company. A wheat importer is more exposed to a Black Sea conflict than a domestic US retailer.

Phase 2 is also when the most important analytical call happens: *is this a spike, or the beginning of a regime change?*

If the disruption is containable, temporary, or can be substituted away, it is a spike — and Phase 3A (fade) follows.
If the disruption is structural — it has permanently destroyed a supply chain, changed an alliance structure, or locked in a new policy equilibrium — it is a regime change, and Phase 3B (regime lock-in) follows instead.

This is the hardest call in geopolitical analysis. It requires deep knowledge of the specific domain: energy infrastructure, supply chain geography, political coalition dynamics. A general investor reading headlines cannot make this call reliably. A specialist can.

### Phase 3A — Fade (Weeks to Months)

Most geopolitical shocks fade. The VIX returns to baseline. The safe-haven premium in gold and CHF unwinds. Risk assets recover most of their spike losses. The news cycle moves on. Investors refocus on earnings and interest rates.

The statistical reality is that this is the *most common outcome*. Out of 12 major VIX-spike geopolitical events between 2001 and 2024, 10 saw the VIX recover to near-baseline within 60 days. The London bombings in 2005 saw a VIX spike of only 8%, recovering in 3 days. The Soleimani killing in January 2020 — widely feared at the time as a potential trigger for a US-Iran war — saw a VIX spike of 7%, recovering in 4 days.

The market's base case, for any geopolitical shock, is fade. This base case is right more often than not.

### Phase 3B — Regime Lock-In (Permanent)

Occasionally — rarely, but consequentially — a geopolitical event triggers a *permanent structural change* in how a market functions. The supply chain shift is not temporary. The alliance restructuring is not reversible. The sanctions architecture is not lifted.

When this happens, you are not looking at a temporary risk premium that will unwind. You are looking at a new, permanently higher cost of capital for affected assets, a permanently restructured supply chain, or a permanently lower valuation multiple for a sector.

Ukraine 2022 is the clearest recent example. The VIX spike recovered (spike). But Russian pipeline gas to Europe went from 201 bcm/year in 2018 to under 8 bcm/year by 2025. European LNG import capacity nearly doubled in three years, from 163 bcm to over 310 bcm. German industrial competitiveness, built on cheap Russian gas, is structurally impaired. None of this reverts when the war ends. That is a regime change.

## Spike vs. Regime Change: The Most Important Distinction in Geopolitical Trading

This distinction deserves its own section. Getting it wrong is expensive. Getting it right is the basis of a serious geopolitical trading approach.

![Spike vs. Regime Change: Two Very Different Market Outcomes](/imgs/blogs/geopolitical-risk-premium-what-markets-price-in-4.png)

A **spike event** has these characteristics:
- The disruption is geographically contained
- Substitution is feasible within months (another supplier, another route, another technology)
- No major political coalition has permanently altered its foreign policy
- The event is a one-time shock, not the beginning of a new equilibrium

A **regime-change event** has these characteristics:
- The disruption destroys an infrastructure, relationship, or supply chain that took decades to build
- No credible substitute exists at scale within years
- Major powers have permanently repositioned their foreign and economic policy
- The event is the beginning of a new normal, not a deviation from the old one

The difficulty is that at the moment of the shock, *almost every event looks like a potential regime change*. The market's fear is always "what if this is the big one?" That fear is almost always wrong. But when it is right, the magnitude is enormous.

#### Worked example:

Ukraine Feb 24, 2022. Let's separate the spike from the regime change with real numbers.

**What was a spike**: The VIX rose from 28.1 to a peak of 33.6 within a week. The S&P 500 fell roughly 3% in the days after the invasion. Gold rose from \$1,898 to \$1,988 by early March — a 4.7% move. The EUR/USD fell from 1.132 to a low around 1.072, a 5.3% move. All of these largely reversed within 3–6 months as it became clear that (a) NATO would not directly intervene militarily, and (b) US and European equities' underlying earnings were not directly disrupted by the war.

**What was a regime change**: The TTF gas benchmark went from 83 EUR/MWh before the invasion to 112 in March, 130 in June, and 310 in August 2022 — a 3.7x peak increase. Russian pipeline gas to Europe fell from roughly 155 bcm in 2021 to 85 bcm in 2022 to 27 bcm in 2023 to under 15 bcm in 2024. Germany launched an emergency program to build five Floating Storage Regasification Units (FSRUs) — LNG import terminals — in under a year, a feat of infrastructure construction that normally takes 5–10 years. By 2025, European LNG import capacity had grown from 163 to over 310 bcm per year. None of this is temporary. None of it reverts.

The investor who classified everything as a "spike" and bought European energy stocks in March 2022 would have been partially right (equities recovered) and partly very wrong (European energy prices and industrial energy costs were permanently restructured, which hurt energy-intensive industries for years).

## The Known Unknown Problem: Binary Geopolitical Risk

Markets are bad at pricing binary geopolitical risks. This is not a character flaw — it is a structural problem with the discounting mechanism. Let's understand why.

For most economic risks, the outcome distribution is *continuous*. GDP growth might be 1%, 2%, or 3%; there are infinite values between those numbers, and the actual outcome is a point within a range. Markets can price these reasonably well by forming a probability distribution over outcomes.

But many geopolitical risks are *binary*: either the invasion happens or it doesn't. Either the sanctions include SWIFT exclusion or they don't. Either the nuclear deal is signed or it isn't. Binary outcomes have no "middle ground" to price into — the market must assign a single probability to the bad outcome, and that probability estimate is almost always deeply uncertain.

This creates two problems:

**Problem 1: The probability estimate is itself unknowable.** Before February 24, 2022, what was the probability of a full-scale Russian invasion of Ukraine? CIA analysts, defense experts, and market strategists had estimates ranging from 5% to 95%. The dispersion of expert opinion was enormous. When there is no reliable base rate and no continuous information flow, the market cannot form an accurate probability. It guesses, anchoring on prior episodes that may not be comparable.

**Problem 2: Binary events cannot be "partially" priced.** For a continuous distribution, the market can gradually incorporate new information — each new data point moves the probability estimate slightly. For a binary event, the market often stays at one probability estimate until the event either happens or clearly isn't going to happen, then jumps. This creates discontinuities — the famous "gap open" — that cannot be traded through.

The result: markets systematically under-price the probability of *actual* binary geopolitical events until they are nearly certain, then over-react at the moment of confirmation. The Ukraine invasion was well-telegraphed months in advance; US intelligence was unambiguous. Yet market prices did not fully price the invasion scenario until the tanks crossed the border. This is not irrationality — it is the rational response to uncertain binary probabilities.

### The discovery-period arbitrage

When markets cannot trade for a period after a known event, the "discovery period" creates a specific type of pricing opportunity. The most famous historical example is September 11, 2001.

The US stock market closed on September 11 and remained closed through September 17 — four full trading days. When it reopened on September 17, markets had to price not just the original shock (the attacks) but four additional days of information: the scope of casualties, the policy response, which sectors were actually affected, and which businesses were going to survive.

#### Worked example:

The S&P 500 fell 11.6% in its first week back. But within that aggregate, the moves were extreme and divergent. Airlines lost approximately 40% — United Airlines fell from \$30 to under \$18 in one day. Defense stocks rose 10–15% — Lockheed Martin, Raytheon, Northrop Grumman all jumped sharply. The gap between "the news happened" (September 11) and "the price updated" (September 17) was four full days.

What does this mean for the analytical investor? The key insight is that when the market reopens after a forced closure, it must price a *known* event. There is no probability uncertainty — the event happened. The question is purely about impact. And impact analysis requires sector-level, company-level knowledge that most investors don't do in real time.

An investor who, on September 14 (during the closure), did the work to estimate that airlines would lose 35–45% of their revenue for 6–12 months (based on SARS data, historical terrorism impacts on travel, and the specific economics of US airlines) and simultaneously estimated that defense budgets would increase by \$50–100bn per year (based on the US government's stated intent to go to war) had actionable information — but no market to trade it in until September 17.

The discovery period is a forced laboratory for expected-value analysis. The investor who was ready with a pre-formed view on sector impacts was positioned to trade the opening. The one who was still processing the shock was not. This is the core skill that geopolitical trading rewards: doing the impact analysis *before* the price updates.

**Intuition**: The edge in geopolitical trading is not knowing what will happen — it is knowing *what it means* faster and more precisely than the market.

## How Markets Get Geopolitical Risk Wrong: The Systematic Biases

Markets are not simply "wrong" about geopolitical risk randomly. They are wrong in predictable, systematic ways. Here are the four main biases.

### Bias 1: Over-pricing fast, visible shocks; under-pricing slow, invisible buildups

A bomb going off in a capital city produces a VIX spike because it is immediate, vivid, and makes headlines. A gradual restructuring of global semiconductor supply chains — which is geopolitically driven and ultimately far more consequential to corporate earnings — produces almost no immediate VIX response.

This is a form of *availability bias*: humans (and the markets they create) overweight vivid, immediate, concrete events relative to slow-moving, abstract ones. A war beginning is more psychologically salient than a decade-long deglobalization trend, even if the latter has larger long-run economic effects.

The practical implication: the real geopolitical risk premiums worth tracking are often *not* the ones in the headlines. The headline events are already being priced by 10,000 analysts. The slow-building restructurings — critical mineral supply chain dependency, tech nationalism, debt-trap diplomacy, demographic shifts in key democracies — are often ignored until they suddenly become crises.

### Bias 2: Anchoring on the prior shock

After 9/11, investors were primed for terrorism. After the Iraq War, they were primed for Middle East conflict. After Ukraine, they were primed for energy disruption. This anchoring causes markets to *over-price* the risk of a repeat of the previous crisis, while *under-pricing* a qualitatively different crisis in a different domain.

In 2023, after years of energy security fears, oil markets were hyper-sensitive to any Middle East conflict. The Gaza war in October 2023 briefly pushed Brent from \$88 to \$95 before fading — because the supply impact was minimal (Gaza doesn't produce oil). The market was pricing a replay of a risk that wasn't actually present. Meanwhile, semiconductor supply chain fragility — a different kind of geopolitical risk — was relatively under-priced.

### Bias 3: Mis-estimating path dependency

Many geopolitical risks are path-dependent: the outcome depends not just on *whether* a crisis occurs, but on *how* it unfolds. Russia could have achieved its objectives in Ukraine in days (as many analysts initially expected), or it could have been repelled (as actually happened), or it could have been a grinding multi-year war (actual outcome). Each path has radically different market implications.

Markets are poor at pricing path-dependent scenarios because they require sequential probability assessments — "what is the probability of path A in month 2 given that path B occurred in month 1?" — that are genuinely hard to estimate and change rapidly as new information arrives. Most investors simplify by pricing just two scenarios: "invasion succeeds quickly" vs. "no invasion," missing the drawn-out grinding war scenario entirely.

### Bias 4: The "known unknown" under-discounting problem

Here is the deepest one. When investors *know* that they don't know the outcome of a geopolitical risk, they often discount it by less, not more, than they should. The logic runs: "I don't know what will happen, so I'll wait and see before adjusting my position." But waiting to see is itself a bet — a bet that the current price is approximately right. In a binary event, the current price can be far from right.

The rational response to a binary outcome where you have high uncertainty about the probability is *more* caution, not less — larger discount, wider confidence interval, smaller position size. The behavioral response is often the opposite: paralysis ("I don't know, so I'll do nothing") that effectively means holding the pre-event price until the event itself forces a reprice.

## Historical Case Studies: Four Events, Four Lessons

### Case 1: 9/11 — The Discovery-Period Arb

As discussed above, 9/11 was a spike event. The S&P 500 fell 11.6% in its first week, then recovered all of it by the end of 2001. The VIX spiked to 45%, recovered over 60 days. For the equity market as a whole, September 11 is almost invisible in the long-run return series.

But specific sectors were regime-changing events: US airlines never fully recovered their pre-9/11 profitability model. The combination of security costs, reduced business travel demand, and higher insurance premiums permanently changed the economics of US aviation.

**Lesson**: Even events that are "spikes" at the index level can be "regime changes" at the sector level.

### Case 2: Iraq War 2003 — The Well-Telegraphed Invasion

The Iraq War was announced months in advance. US troop buildups, UN weapons inspections, Colin Powell's Security Council presentation in February 2003 — the invasion was one of the most forewarned military operations in history. The GPR Index was at 254 when the invasion began on March 20, 2003 — up from a baseline of around 100.

What happened to markets? Brent crude rose from roughly \$30 to \$37 before the invasion — then *fell* to \$25 within a month of the invasion beginning. This is the "buy the rumor, sell the news" in its purest form: the geopolitical risk premium built up during the pre-invasion period, then rapidly unwound once the invasion began and it became clear that (a) oil infrastructure was not destroyed, and (b) a quick regime change was occurring.

The VIX fell from 34 to 22 in the first two weeks of the war. Markets had been *overpricing* the Iraq risk premium during the uncertainty period, and that excess premium rapidly unwound once the uncertainty resolved.

**Lesson**: For well-telegraphed geopolitical events, the highest risk premium is often at the moment of maximum uncertainty — before the event, not after. The resolution of uncertainty, even if the event is bad, often triggers a relief rally.

### Case 3: Ukraine Feb 24, 2022 — Spike Plus Regime Change

This was the rare event that combined both. The VIX spiked from 28 to 34 and recovered in about 45 days — a spike. European gas prices went from 83 EUR/MWh to 310 EUR/MWh — a regime change. The ruble collapsed 47% before recovering substantially as Russia implemented capital controls. MSCI Russia — the index of Russian stocks — went from 1,390 to 210 within a week, and then to effectively zero as the Moscow Exchange halted trading.

The discernment required: within the single "Ukraine" label, there were at least four separate risk stories playing out simultaneously:
1. European energy (regime change — long-term supply restructuring)
2. Global equities (spike — VIX recovered in 45 days)
3. Russian assets (regime change — permanently impaired by sanctions)
4. Wheat and agricultural commodities (partial-permanent — Ukrainian exports declined but alternative supplies emerged within a year)

Treating all of these the same — as one "Ukraine trade" — would have been a costly mistake.

#### Worked example:

Suppose you are managing a \$1 million portfolio on February 24, 2022. You assess three scenarios:

- **Scenario A (35% probability)**: Quick Russian victory, ceasefire within 30 days. European gas spike is temporary, supplies rerouted within a year. S&P 500 falls 8% and recovers by year-end. European energy-intensive stocks fall 15% then recover.

- **Scenario B (45% probability)**: Prolonged war, years of attrition. European gas permanently restructured. Russia permanently isolated from Western financial system. S&P 500 falls 12% and takes 18 months to recover. European energy-intensive industrials fall 30% and don't recover.

- **Scenario C (20% probability)**: NATO direct involvement, extreme escalation. S&P 500 falls 30%+. This is the tail risk.

Expected S&P 500 move = (0.35 × -8%) + (0.45 × -12%) + (0.20 × -30%) = -2.8% - 5.4% - 6.0% = -14.2%.

On a \$1 million portfolio, that is an expected loss of \$142,000. How you choose to hedge — and whether you hedge the S&P (spike exposure) or the European industrials (regime-change exposure) or both — depends on which scenarios you assign higher probability to.

Scenario B turned out to be closest to reality. The investors who recognized the regime-change component early — particularly for European energy infrastructure and Russian asset exposure — significantly outperformed those who either (a) sat through everything or (b) sold everything and missed the equity recovery.

### Case 4: Gaza October 7, 2023 — Spike That Mostly Faded

On October 7, 2023, Hamas launched an unprecedented attack on Israel. The initial market reaction was textbook Phase 1: VIX spiked 19%, gold rose, oil rose (Brent from \$85 to \$92 within days). Every historical analogy suggested Middle East conflict = oil price spike = sustained risk-off.

It didn't happen. Brent peaked around \$95, then fell back below \$80 by the end of October, and continued declining to the mid-\$70s range in 2024. The VIX recovered in 21 days. The reason: the conflict, while horrific in human terms, did not disrupt any significant oil supply infrastructure. Gaza and Israel are not oil producers. The Houthi attacks on Red Sea shipping (which began in November 2023) created some supply chain disruption, but it was manageable.

The GPR Index rose to 188 in October 2023 — elevated, but far below Ukraine's 345. The market's assessment was essentially correct: a high-intensity regional conflict that was unlikely to escalate into a region-wide war involving Iran, Saudi Arabia, or Turkey directly. That probability assessment — while not certain — turned out to be roughly right.

**Lesson**: Not all Middle East conflicts are 1973. The geographic and economic specifics of *which* infrastructure and *which* actors are involved determine whether a regional conflict translates into global market disruption. The map of energy infrastructure matters more than the map of combat operations.

## The GPR-to-P/E Translation: Connecting Risk to Valuation

Let's formalize the link between geopolitical risk and equity valuation, building on the worked example earlier.

The equity risk premium (ERP) is the extra return investors require above the risk-free rate to hold equities. It fluctuates based on how risky the future looks. In calm periods, the ERP might be 4–5%. In crisis periods, it might spike to 7–8%.

Research (Caldara-Iacoviello 2022, updated studies through 2024) finds that a one-standard-deviation increase in the GPR Index — roughly doubling from the long-run average of 100 to around 200 — corresponds to an increase in the equity risk premium of approximately 0.7–0.9 percentage points. This is a *sustained* increase, not a one-day spike.

#### Worked example:

Situation: S&P 500 trades at 22x forward earnings. Risk-free rate (10-year Treasury yield) = 4.5%. Long-run earnings growth expectation = 2%. This implies an equity risk premium of approximately:

ERP = Earnings yield − Risk-free rate = (1/22) − 4.5% = 4.55% − 4.5% = 0.05%, which seems too low. In practice, earnings growth adjustment raises this to roughly 5.5% by a more complete model. Let's use 5.5%.

GPR Index doubles from 100 to 200 (a sustained elevated geopolitical environment). ERP rises by 0.8 percentage points to 6.3%.

New implied P/E = 1 / (ERP + risk-free rate − growth) = 1 / (6.3% + 4.5% − 2%) = 1 / 8.8% ≈ 11.4x

Wait — this seems extreme. A P/E compression from 22x to 11.4x? The key nuance: this formula ignores that interest rates also move during geopolitical crises. In a typical flight-to-safety episode, the risk-free rate (Treasury yield) *falls* as investors buy Treasuries. If the 10-year falls from 4.5% to 4.0% during the crisis:

New P/E = 1 / (6.3% + 4.0% − 2%) = 1 / 8.3% ≈ 12.0x

Still a large compression from 22x. But this is a *sustained elevated GPR at 200* scenario — a prolonged period of very high geopolitical uncertainty, not a brief spike. The actual S&P 500 impact of a short-duration spike is much smaller, because the P/E expansion/compression is proportional to *duration*: a 30-day elevated GPR has 1/12 the impact of a 12-month elevated one.

The practical takeaway: brief geopolitical spikes have small long-run P/E implications. Sustained elevated geopolitical environments — like 2022–2023 with the Ukraine war ongoing — have meaningful P/E compression effects that rational investors must build into their valuation models.

## The Geopolitical Event Type Matrix

Not all geopolitical events are created equal. The matrix below organizes event types by their historical market outcome profile.

![Geopolitical Event Types vs. Market Outcome: A Classification Matrix](/imgs/blogs/geopolitical-risk-premium-what-markets-price-in-5.png)

Key readings from the matrix:

- **Economic sanctions almost never produce only a spike.** The very nature of sanctions — which reorder supply chains, restrict access to financial systems, and impose long-duration changes on trade relationships — means they are always at least partial-permanent in their market impact. Investors who treat sanctions announcements as "noise" are consistently wrong.

- **Trade wars produce medium-term structural changes even when ostensibly "resolved."** The US-China trade war of 2018–2025 nominally saw a "Phase One" deal in January 2020 and a "90-day pause" in May 2025. But the *underlying tariff baseline* went from 3.1% before the war to 19.3% after Phase One, to 35% or more in 2025. Each "resolution" preserved most of the structural change.

- **Most coups are spikes.** The Turkey coup attempt of July 2016 sent the Turkish lira down 4.7% overnight. Within 14 days, most of the move had reversed. This is the historical norm for failed coups in large, institutionalized economies. Successful coups in small, commodity-dependent economies are more likely to be regime changes (Myanmar 2021 saw a -26% currency devaluation that persisted for years).

- **Wars along major energy infrastructure corridors are regime-change risks.** The Strait of Hormuz, the Black Sea grain shipping lane, the South China Sea — these are the geographic chokepoints where conflict translates most quickly and irreversibly into commodity supply disruption. Wars in commodity-poor, trade-light regions produce spikes. Wars near these chokepoints produce regime changes.

## Common Misconceptions About Geopolitical Risk

### Misconception 1: "Markets always sell off on geopolitical risk"

The direction of market response to geopolitical events depends entirely on what the event actually disrupts. The S&P 500 rose 2.1% on the day Russia's invasion of Crimea began in March 2014. Why? Because the market assessed that the event was contained, that NATO would not escalate militarily, and that US corporate earnings were not directly affected. The "geopolitical risk" was priced as small and contained because the assessment was small and contained — and that assessment was roughly correct.

Moreover, certain geopolitical events are *positive* for specific markets. A major military conflict involving the Middle East is bullish for oil producers. A sanctions regime that excludes a major supplier is bullish for that supplier's competitors. A trade war that diverts manufacturing from China is bullish for Vietnam, Mexico, and India. Blanket "sell everything" geopolitical responses systematically miss these cross-sectional opportunities.

### Misconception 2: "If the situation is bad, the market hasn't priced it in"

This is backwards. The *worse* a situation is being reported in the news, the more likely it is to already be priced. By the time CNN and Bloomberg are running 24/7 coverage of a geopolitical crisis, every institutional investor in the world is aware of it and has adjusted their portfolio. The *un-priced* geopolitical risks are the ones that are not in the news cycle — slow-building structural shifts, second-order consequences of events that happened years ago, risks in geographies that don't get Western media coverage.

The corollary: the best geopolitical trades are usually available in the *pre-announcement* or *early-escalation* phase, before mass media coverage, not in the full-crisis phase when everyone is already repositioned.

### Misconception 3: "The VIX tells you how serious a geopolitical event is"

The VIX measures *implied equity volatility* for US S&P 500 options. It tells you how much US equity market participants are paying for downside insurance. It does NOT measure:
- The severity of the human or geopolitical situation
- The risk to non-US markets
- The probability of escalation
- Whether the event is a spike or regime change

The VIX spiked 82% during COVID — which had a genuine global economic impact — and only 8% during the London bombings of 2005 — which were horrific in human terms but had minimal global economic impact. The VIX is a real-time measure of US equity fear, not a general measure of geopolitical seriousness.

### Misconception 4: "Safe havens always work in geopolitical crises"

In typical fast geopolitical shocks, gold and USD do rise. But there are important exceptions:

- **Very large global shocks**: In March 2020 (COVID pandemic), the USD rose sharply but gold initially fell — because investors needed to sell *everything* liquid to meet margin calls, including gold.
- **USD-originating crises**: In April 2025, when the US announced Liberation Day tariffs, the dollar *fell* against the euro and yen — because the tariff shock raised concerns about US economic policy credibility and the status of US assets as safe havens. This was unusual and marked an important regime shift: for the first time in decades, a US policy shock caused capital to flee *from* the dollar, not to it.

Safe havens are not mechanical. They depend on the specific nature of the crisis and what triggers the flight-to-safety demand.

### Misconception 5: "If the market recovered, there was no lasting damage"

The S&P 500 fully recovered from the Ukraine invasion within 6 months. European equity indices also recovered. But this aggregate recovery masked enormous sectoral damage: German chemical companies, whose margins depended on cheap energy inputs, faced years of margin compression. European energy-intensive manufacturers restructured or relocated. The aggregate index recovered because US tech and consumer stocks (which had minimal Ukraine exposure) drove the rebound, offsetting the still-damaged European industrials.

Index-level recovery after geopolitical events masks sub-index-level regime changes constantly. The careful investor looks through the index at the sectors and geographies that were specifically exposed.

## VIX Spike Patterns: What History Tells Us

![VIX Spike Magnitude vs Days to Recovery for Geopolitical Events](/imgs/blogs/geopolitical-risk-premium-what-markets-price-in-6.png)

Reading the scatter chart carefully reveals several patterns:

**Pattern 1: Small spikes recover in days, large spikes take weeks to months.** Events producing a VIX spike below 15% (London bombings, Soleimani, Paris attacks) recovered in 3–7 days. Events producing 15–30% spikes (Ukraine Crimea, Arab Spring, Gaza) took 21–30 days. The two largest non-COVID spikes (9/11 at 45%, and Russia-Ukraine 2022 at 31%) took 60 and 45 days respectively.

**Pattern 2: Iraq War is the outlier.** The Iraq War produced a 22% VIX spike that recovered in only 14 days — faster than events with smaller spikes like Arab Spring or Ukraine Crimea. The reason: the invasion was the resolution of a years-long uncertainty. The spike was a *pre-invasion* accumulation finally releasing, not a true surprise.

**Pattern 3: COVID is the complete outlier.** An 82% VIX spike taking 180 days to recover. COVID is excluded from the chart because including it would compress the scale so much that the geopolitical events become invisible. But it serves an important reminder: when a geopolitical risk also has large, uncertain, global *economic* consequences (not just the risk premium from uncertainty), the recovery is much slower. COVID was not just a geopolitical risk — it was a genuine global economic shock with massive real-sector consequences.

**Pattern 4: The relationship is not linear.** Going from a 15% spike to a 30% spike doesn't just double the recovery time — it extends it from ~21 days to ~45+ days. The largest spikes appear to be qualitatively different events, not just bigger versions of small spikes.

## How to Read Geopolitical Risk as an Investor: A Practical Framework

Here is the decision tree for any new geopolitical event:

**Step 1: Characterize the event.**
- Is this a fast shock (military action, terrorist attack) or a slow-building event (sanctions escalation, trade war tariff increase)?
- Fast shocks trigger Phase 1 immediately. Slow events may have been partially priced over weeks or months.

**Step 2: Map the economic channels.**
- Which specific supply chains, energy routes, trade flows, or financial connections does this event threaten?
- Be specific: not "energy" but "European LNG supply from Russia" or "Middle East strait chokepoint for Asian oil imports."

**Step 3: Spike or regime change?**
- Is the disrupted infrastructure physically destroyed or politically severed in a way that takes years to reverse?
- Are there viable substitutes at scale within 6–12 months?
- Have major powers permanently repositioned their foreign policy as a result?

**Step 4: What has already been priced?**
- If the event was well-telegraphed, the risk premium may already be embedded. The VIX may be elevated before the event.
- If the event is a surprise, Phase 1 repricing has just begun.

**Step 5: What is the base case vs. the tail?**
- Price the expected outcome, not the worst case.
- Keep a tail-risk hedge (usually options, gold, or USD exposure) sized to the probability and magnitude of the tail scenario.

This framework does not guarantee profits — geopolitical forecasting is genuinely hard, and the information advantage of professional analysts is real but limited. What it does guarantee is a *disciplined analytical process* that prevents the most common mistakes: selling everything on every headline, ignoring slow-building structural changes, or failing to distinguish spike risk from regime-change risk.

## How It Shows Up in Real Markets: Practitioner Observations

**Energy markets and geopolitical tails**: Oil traders systematically price a small "geopolitical premium" above supply-demand fundamentals. This premium was estimated by the IEA at roughly \$5–8 per barrel in calm periods and \$15–25 per barrel during elevated Middle East tensions. When the premium is high and no disruption materializes, it reverses. When the premium is low and a disruption materializes, you get a sudden spike. Positioning for the spike-to-mean-reversion in the geopolitical premium — by monitoring the spread between spot and futures and comparing it to historical crisis analogues — is a real trading strategy used by commodity funds.

**Emerging market bond spreads**: Sovereign spreads (as shown in the EMBI data for Argentina) are essentially continuous real-time markets for geopolitical and political risk. Argentina's EMBI spread went from 777 basis points in January 2019 to 2,300 basis points in August 2019 after the PASO election result — in a single day. When Milei won in November 2023, it fell from 2,100 to 1,560 in weeks. Spread traders who correctly assessed the Milei election as a regime change (toward orthodox policy) rather than a spike captured enormous returns.

**FX and regime-change signals**: The yen's behavior during the Ukraine war illustrates how safe-haven currencies encode geopolitical risk assessments. JPY appreciated initially (flight to safety), then depreciated sharply through 2022 as Japan's energy import bill — priced in dollars — exploded due to the LNG price surge. The yen was simultaneously benefiting from safe-haven demand and being hurt by its terms-of-trade deterioration. Investors who understood both effects could neutralize one and express the other.

**US defense stocks as geopolitical barometers**: Defense stocks (Lockheed Martin, Raytheon, Northrop Grumman) have consistently outperformed the S&P 500 during periods of elevated geopolitical risk. After 9/11, the defense sector rose 10%+ while the S&P fell 11.6%. After Ukraine in 2022, European defense stocks surged 20–40% in weeks as European governments announced major defense spending increases. These moves are not speculative — they track actual increases in government defense budgets, which are highly observable and relatively predictable once a geopolitical event occurs.

## The Playbook: Translating This Framework Into Action

Given everything above, here is how a thoughtful investor applies the geopolitical risk premium framework in practice.

**Before the event**:
- Track the GPR Index and cross-reference it against your portfolio's geopolitical exposures.
- Map which of your holdings are most sensitive to: energy supply disruption (European industrials, airlines, chemical companies), sanctions exposure (companies with significant Russia/Iran/China revenue), trade route disruption (shipping companies, export-oriented manufacturers in politically exposed countries).
- Pre-establish hedging playbooks for the most likely scenarios. You cannot make perfect decisions in Phase 1; make your Phase 1 playbook in Phase 0 (calm periods).

**During Phase 1 (shock)**:
- Do not trade the index — trade the specific exposures.
- Buy protection (gold, USD, long volatility) only if you don't already have it and the tail risk justifies the cost.
- Resist the impulse to extrapolate Phase 1 prices as the new normal — they almost never are.

**During Phase 2 (discount)**:
- Make the spike vs. regime-change call.
- Analyze the specific economic channels with real-world data: What is actually disrupted? What substitution is possible? How long does it take?
- Size positions based on your conviction about the Phase 3 resolution, not based on how scary the news feels.

**During Phase 3A (fade)**:
- Be the patient seller of safe-haven premium that overshot.
- Re-enter cyclical/risk assets that sold off without specific economic exposure to the disruption.

**During Phase 3B (regime lock-in)**:
- Accept the new structural reality.
- Position for the second-order effects: who benefits from the restructuring? (European LNG terminal operators, US LNG exporters, alternative energy sources.) Who is structurally impaired? (Energy-intensive European manufacturers, companies with irreplaceable Russia/China supply chains.)

## Further Reading and Cross-Links

This post is the opening framework for the series. The subsequent posts build on this foundation by examining specific cases in detail:

- **Ukraine 2022** — A full deep-dive on the energy regime change: which exact assets moved, why, by how much, and what the long-run market implications were. See [Macro Trading: Central Banks and Energy Markets](/blog/trading/macro-trading/) for the macroeconomic transmission.
- **Sanctions mechanics** — How financial sanctions work, which channels they activate, why they are always structural. See [Fixed Income: Sovereign Spreads](/blog/trading/fixed-income/) for the bond market implications.
- **Trade Wars and Tariffs** — The US-China tariff escalation from 3.1% to 145%, the supply chain restructuring, and the Vietnam/Mexico beneficiary trade. See [Macro Correlations: Trade Policy and FX](/blog/trading/macro-correlations/) for the FX transmission.
- **Geopolitical Risk and Gold** — Why gold rises in some geopolitical crises and not others; the historical record is more nuanced than "gold always goes up." See [Gold: From First Principles](/blog/trading/gold/) for the complete gold framework.
- **Event Trading** — Specific strategies for trading around geopolitical events using the three-phase framework. See [Trading the News: How Markets React to Geopolitical Events](/blog/trading/event-trading/) for the tactical layer.
- **GPR Index methodology** — Caldara, D., and M. Iacoviello (2022), "Measuring Geopolitical Risk," American Economic Review, 112(4): 1194-1225. Available at matteoiacoviello.com/gpr.htm. The full dataset is publicly available and updated monthly.
- **Geopolitical risk and equity premiums** — Hassan, T.A., Hollander, S., van Lent, L., and Tahoun, A. (2019), "Firm-Level Political Risk: Measurement and Effects," Quarterly Journal of Economics. Uses firm-level earnings call transcripts to measure geopolitical risk exposure at the company level — a more granular tool than the GPR Index.


*This post is part of the series "Geopolitical Crises & Markets: How Nations Move, How Markets React." It is educational in intent and does not constitute investment advice. Past geopolitical market patterns do not guarantee future outcomes; every crisis has unique characteristics. Data sourced from: Caldara-Iacoviello GPR Index (matteoiacoviello.com), Bloomberg, MSCI, IEA, as of June 2026.*
