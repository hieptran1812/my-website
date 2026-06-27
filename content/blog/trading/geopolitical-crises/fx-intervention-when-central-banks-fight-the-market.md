---
title: "FX Intervention: When Central Banks Fight the Market"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A deep-dive into how and why central banks intervene in currency markets, when those interventions succeed or fail, and how traders position around them."
tags: ["geopolitics", "fx", "currency", "central-bank", "intervention", "boj", "snb", "erm", "speculative-attack", "reserve-management", "forex", "macro"]
category: "trading"
subcategory: "Geopolitical Crises"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Central bank FX intervention is a political act with a financial mechanism: it buys time when fundamentals are close, but burns through reserves when they are not.
>
> - Sterilized intervention changes the exchange rate signal without altering monetary conditions; unsterilized intervention does both, and is far more powerful.
> - The Bank of Japan spent roughly \$42.8 billion in October 2022 alone to defend the yen near 145 — a record single-month intervention — yet the yen hit a 38-year low of 161 in July 2024.
> - The Swiss National Bank held the EUR/CHF 1.20 floor for four years (2011–2015) at a cost exceeding CHF 450 billion in reserves before abandoning it in a single morning, causing a 15% CHF appreciation that bankrupted several FX brokers.
> - The UK's 1992 Black Wednesday cost roughly \$3.3 billion in reserve spending in one afternoon before sterling crashed out of the European Exchange Rate Mechanism — Soros made an estimated \$1 billion in profit.

On the morning of September 6, 2011, the Swiss National Bank did something that central banks almost never do: it announced a hard floor on its currency. The Swiss franc (CHF) had appreciated so violently against the euro during the eurozone debt crisis that Swiss exporters — watch manufacturers, machinery builders, pharmaceutical firms — were screaming. EUR/CHF had briefly touched 1.00, meaning one Swiss franc bought one euro, a level that made Swiss goods catastrophically expensive for European buyers. The SNB drew a line: EUR/CHF would not go below 1.20, and it would buy euros "in unlimited quantities" to defend that line. Markets tested it immediately. The SNB spent billions in the first few days and held the line. For four years.

Then, on January 15, 2015, with no public warning, the SNB removed the floor. EUR/CHF dropped from 1.20 to 0.86 in minutes — a move of 28% in a currency pair that had not moved by more than 0.1% in months. FX brokers with leveraged client positions on the wrong side of the trade were wiped out overnight. Alpari UK went insolvent. FXCM, one of the largest retail FX brokers in the world, required an emergency \$300 million loan to survive. The lesson encoded in that single morning: when a central bank fights the market for long enough, the peg becomes a trap — not just for traders, but for the central bank itself.

![FX intervention transmission chain from political trigger to market outcome](/imgs/blogs/fx-intervention-when-central-banks-fight-the-market-1.png)

This post dissects the entire intervention toolkit: how FX intervention actually works mechanically, why sterilization matters, what the "three Ps" of intervention design look like in practice, and how to read the signals that precede, accompany, and follow a central bank entering the market. We will work through three of history's most instructive case studies — Japan in 2022, Switzerland in 2011–2015, and the UK in 1992 — and finish with a practitioner playbook for trading around interventions.

## Foundations: How FX Intervention Works

### The basic mechanism

A currency exchange rate is, at its simplest, a price set by supply and demand in the foreign exchange market. When there are more buyers of the Japanese yen (JPY) than sellers, the yen rises in value. When there are more sellers — maybe investors are selling yen to fund positions in higher-yielding US assets — the yen falls.

The foreign exchange market is the largest financial market in the world, with daily trading volume exceeding \$7.5 trillion (BIS Triennial Survey 2022). No single buyer or seller, not even a G10 central bank, can dominate it indefinitely. But a central bank does have one weapon no private actor has: the ability to create its own currency in unlimited quantities. A country's central bank can print as much of its own currency as it chooses. It can then sell that newly created currency into the market to weaken it. Conversely, to strengthen its currency, it must spend its foreign exchange reserves — the stockpile of dollars, euros, and other foreign currencies it holds — to buy its own currency back. And those reserves are finite.

This asymmetry is fundamental to understanding intervention. Weakening your currency is (theoretically) unlimited — you can always print more. Strengthening it is limited by how many foreign exchange reserves you hold. This is why interventions to defend a currency that is falling tend to be far more fraught than interventions to prevent a currency from rising too fast.

### Sterilized versus unsterilized intervention

Here is where central bank operations get precise, and the distinction matters enormously for how long an intervention can last and how much market impact it has.

When a central bank intervenes in the FX market, it creates or destroys domestic currency. If the Bank of Japan buys yen and sells US dollars, it is taking yen out of the financial system — tightening domestic money supply. That has real effects: it raises Japanese interest rates, which could harm the domestic economy. A central bank that wants to intervene in FX without triggering those domestic monetary effects will sterilize the intervention.

Sterilization means offsetting the monetary impact with an opposite operation in the domestic bond market. If the BOJ sells \$10 billion of US dollar reserves and buys \$10 billion worth of yen, it has tightened domestic yen supply by that amount. To sterilize, it simultaneously buys \$10 billion worth of Japanese government bonds from banks — injecting those same yen back into the banking system. The net effect on domestic money supply: zero.

![Sterilized versus unsterilized intervention: mechanism and monetary impact comparison](/imgs/blogs/fx-intervention-when-central-banks-fight-the-market-2.png)

Sterilized intervention works through the "portfolio balance channel" and the "signaling channel." The portfolio channel says: by changing the relative supply of assets denominated in different currencies, the central bank shifts risk premia in ways that move exchange rates. The signaling channel says: even if the balance sheet impact is sterilized, the act of intervention sends a message to markets about the central bank's intentions and its willingness to defend a level — and that signal alone can move prices.

The academic research on whether sterilized intervention works is genuinely mixed. The consensus view among economists is that sterilized intervention has limited lasting impact unless it is backed by the credibility of a larger policy shift — a rate hike, capital controls, or a clear communication of how far the central bank intends to go. Unsterilized intervention, by contrast, is real monetary policy and has clear, lasting effects. But it comes with domestic side effects: higher interest rates, tighter credit, slower growth.

### The foreign exchange reserve war chest

The size of a country's foreign exchange reserves is the single most important determinant of how long it can defend its currency. Japan's reserves — roughly \$1.1 trillion at the end of 2022 — are among the largest in the world, second only to China's. But even Japan can run up against the limit when the market is moving against it.

Switzerland's reserves grew from CHF 250 billion in 2012 to over CHF 700 billion by 2017 as a direct result of the EUR/CHF floor defense. At its peak, the SNB's balance sheet was roughly equivalent to Switzerland's entire annual GDP — an extraordinary situation that no other central bank in a developed economy has ever approached. The cost of holding that floor eventually became politically untenable.

Emerging market central banks face a much tighter constraint. Turkey's net reserves fell below zero in 2021 at various points when off-balance-sheet FX swap commitments were accounted for. Brazil's central bank has historically held \$350-380 billion in reserves but also carries significant domestic debt obligations. When a country's reserves are thin relative to the daily trading volume of its currency, a speculative attack can drain them in days.

### The three Ps of intervention design

Practitioners who watch central bank FX operations closely have identified three factors that determine the design and likely success of an intervention. Call them the three Ps:

**Price** — what level is the central bank defending? Is the chosen level economically justifiable given purchasing power parity, current account dynamics, and interest rate differentials? A level that makes sense for the fundamentals is far easier to defend than one that is purely politically convenient. Switzerland's 1.20 floor was defensible as long as eurozone economic weakness depressed the EUR relative to its long-run value. When the ECB moved toward quantitative easing in late 2014, the fundamentals began pointing EUR/CHF structurally lower, and the floor became untenable.

**Pace** — how fast is the currency moving, and is the intervention timed to slow a disorderly move versus reverse a long-term trend? Central banks are far more successful at smoothing short-term volatility — calming a currency that has overshot its fair value on panic flows — than at reversing a trend driven by real economic differentials. The BOJ's October 2022 intervention was aimed at stopping the velocity of yen weakening, not necessarily the yen level per se. When the fundamental drivers (the US-Japan interest rate differential) were still pointing to yen weakness, each intervention bought time rather than changing direction.

**Proximity to elections** — central banks nominally operate independently of political cycles, but the empirical record shows that intervention intensity correlates with political pressure. The BOJ's September 2022 intervention came with Japanese general elections in the recent past and the political cost of a weak yen — through import price inflation — becoming a major domestic issue. The UK's defense of the ERM band in September 1992 was partly driven by the political costs Prime Minister John Major would face if sterling devalued before the Conservative Party had stabilized its economic credibility after the 1992 election.

## The Political Calculus Behind Intervention

### Why governments push central banks to intervene

Currency intervention is almost always a response to a political problem that has found its expression in an exchange rate. A central bank executive looking at the exchange rate sees a price. A finance minister looking at the same number sees votes.

When the yen weakens significantly — as it did through 2022, falling from 115 to 151 JPY per USD — Japanese households feel it as higher import prices. Japan imports virtually all of its energy. A weak yen that coincides with global commodity price spikes (as happened with Russia's invasion of Ukraine driving oil and gas prices up) translates directly into higher gasoline prices, higher utility bills, and higher food import costs. Japanese consumer confidence fell sharply through 2022, and the governing Liberal Democratic Party faced real domestic pressure. The line between "independent central bank decision to intervene" and "politically driven intervention" was thin.

Switzerland's situation in 2011 was the reverse: the franc was too strong, not too weak. The problem for Switzerland was its export sector and the threat of deflation. Switzerland's inflation rate was running negative by late 2011 — prices were falling — and a rising franc would have deepened that deflationary spiral. Swiss exporters, representing a disproportionately large share of the economy, were losing competitiveness by the month. The SNB's floor was simultaneously monetary policy (preventing deflation) and exchange rate policy (protecting exporters).

The UK in 1992 was fighting a different battle: maintaining the political credibility of European integration. The UK had joined the Exchange Rate Mechanism in October 1990 at a rate of DEM 2.95 per pound — a rate widely regarded by economists at the time as overvalued. Maintaining ERM membership was a cornerstone of the Conservative government's European strategy and part of its anti-inflation credentials. Exiting the ERM would have been seen as an embarrassing admission that the entry rate was wrong and that UK inflation had been higher than the Bundesbank's.

### The decision calculus: what alternatives did each bank reject?

Understanding why a central bank chose intervention over other options reveals the constraints it was operating under.

Japan in 2022 faced three realistic options: (1) intervene directly in FX markets, (2) raise interest rates to narrow the US-Japan rate differential that was driving yen weakness, or (3) do nothing and accept yen weakness. Option 2 — rate hikes — was rejected because Japan's economy was still fragile, its real estate market needed support, and the BOJ under Governor Kuroda had staked its entire institutional credibility on maintaining yield curve control (YCC) at near-zero rates. Option 3 was politically impossible given import inflation. Option 1 was the only remaining tool.

Switzerland in 2011 faced the choice between the floor, capital controls, or negative interest rates. The SNB had already cut rates to near zero. Capital controls were seen as inconsistent with Switzerland's role as a global financial center — the SNB rejected them explicitly. Negative interest rates were introduced only later (in 2014) when additional pressure built. The floor was the most transparent, market-friendly signal the SNB could send.

The UK in 1992 had one option that the Conservative government refused to consider: devaluing sterling before Black Wednesday by negotiating a lower ERM band with its European partners. That would have been an orderly exit with some political cover. Instead, the Treasury and the Prime Minister committed publicly and repeatedly that the pound would not devalue — a commitment that turned the government's credibility into a target. By making sterling's defense a matter of political honor rather than economic logic, they created the conditions for Soros to exploit.

## The Financial Channels: How Intervention Moves Markets

### The direct order flow channel

When a central bank enters the FX market, it does so as a large buyer or seller. In the yen market, which trades roughly \$1 trillion per day, a BOJ intervention of \$5-10 billion in a single session is significant but not overwhelming. The immediate price impact depends on how the intervention is sized relative to the day's volume and whether other market participants know or suspect intervention is happening.

Central banks typically intervene through their primary dealer banks — the same institutions that participate in government bond auctions. The bank contacts the dealer, places a large order to buy or sell the currency, and that order hits the market as a block trade. The initial impact is a sharp move in the spot rate, often 1-2% in a matter of minutes. Algo traders immediately pull back their quotes (wider spreads, lower volume) because they cannot tell whether more intervention is coming. Thin markets amplify the move.

#### Worked example:

On September 22, 2022, the BOJ confirmed it had intervened for the first time since 1998. The yen had been trading at 145.9 JPY/USD. The intervention was estimated at \$19.7 billion in a single session. The yen moved from 145.9 to 140.3 in roughly two hours — a move of approximately 4% or 560 pips. For a trader holding \$100 million of long USD/JPY positions — a bet that the yen would weaken further — a 4% adverse move at 145 implied a loss of approximately \$4 million on that position in two hours. At standard FX leverage of 10:1, that 4% move on a \$100 million notional position (funded with \$10 million of margin) would have wiped out 40% of margin. Forced liquidations amplified the move, as leveraged long USD/JPY positions were stopped out, adding to the selling pressure on the dollar.

### The signaling and expectations channel

The more durable mechanism of FX intervention is not the order flow itself but the signal it sends about the central bank's intentions and the policy environment. A credible signal can shift the market's equilibrium — the price that speculators expect the currency to settle at — without the central bank having to keep spending reserves to maintain it.

Japan's September 2022 intervention failed the signaling test because the underlying policy — BOJ's commitment to yield curve control and near-zero rates — had not changed. Every yen that the BOJ bought through FX intervention to strengthen the yen was simultaneously being partially offset by the BOJ's bond-buying program to suppress interest rates. The market understood the contradiction: you cannot simultaneously suppress interest rates (which weakens the currency by reducing the yield advantage of holding yen) and defend the yen's exchange rate through intervention. One policy had to give. Because the BOJ was unwilling to let the rate policy give, the FX intervention was always going to be temporary.

Switzerland's 2011 floor succeeded for four years precisely because the signal was credible in context. The floor was economically sensible (it prevented deflation), fiscally consistent (Switzerland had the reserves to defend it), and structurally grounded (the Swiss franc was genuinely overvalued relative to purchasing power parity). The SNB's commitment was unlimited in principle and backed by the actual mechanical ability to print francs — something it could do without limit, unlike a country defending its currency against depreciation, which is limited by its reserve stock.

![Japanese yen per USD with Bank of Japan intervention zones 2022 to 2024](/imgs/blogs/fx-intervention-when-central-banks-fight-the-market-3.png)

## The Three Case Studies

### Case Study 1: Bank of Japan 2022 — the \$60+ Billion Yen Defense

The BOJ's 2022 interventions are the most expensive single-year FX intervention program in history. The yen weakened relentlessly through the year, driven by the widening US-Japan interest rate differential: the Federal Reserve was raising rates aggressively to fight 8%+ US inflation, while the BOJ was holding rates near zero and buying unlimited government bonds to cap 10-year yields at 0.25%.

By late September, USD/JPY had reached 145.9 — the weakest yen level since 1990. The BOJ intervened on September 22 for the first time since 1998, spending approximately \$19.7 billion. The yen rallied back to 140.3 briefly. Then the market started weakening it again.

October was the crisis month. USD/JPY pushed through 150 in early October and hit 151.9 by late October — a 32-year low. The BOJ intervened multiple times through October, with the largest single-day operation on October 21 estimated at \$42.8 billion — the largest single-day FX intervention ever recorded. Total October spending: approximately \$42.8 billion. Total calendar year 2022 intervention spending: approximately \$60.2 billion based on Japan's Ministry of Finance official data.

The yen strengthened back to 138 by late November, not primarily because of the interventions, but because US inflation data began showing signs of softening — reducing the expectation of further aggressive Fed rate hikes. The interventions bought time. They did not change the underlying driver. By 2024, with the Fed still restrictive and the BOJ still hesitant, USD/JPY hit 161.7 in July 2024 — a new 38-year low, surpassing even the 2022 trough.

#### Worked example:

The "three P" analysis of the BOJ 2022 intervention: Price — 145 was not economically defensible on purchasing power parity terms (which suggested fair value around 100-110 JPY/USD), but the BOJ was not defending a specific level so much as trying to slow the pace of depreciation. Pace — the yen had weakened 30% in less than a year, which was unprecedented in the post-Plaza Accord era and causing genuine import cost distress. Proximity to elections — Japan had held elections in July 2022 (won by the LDP), and consumer price inflation was becoming a political liability. All three factors pointed to intervention becoming likely when 145 broke. A practitioner watching these signals could position for intervention by holding a small yen-long position as insurance against a sudden reversal, then fading the initial spike post-intervention rather than chasing it.

### Case Study 2: The SNB EUR/CHF Floor — A Peg That Held and Then Snapped

The Swiss National Bank's decision to set a floor at EUR/CHF 1.20 on September 6, 2011, was a direct response to the eurozone debt crisis. Greek sovereign debt was in distress, Italian and Spanish bonds were selling off, and investors were flooding into Swiss francs as a safe haven. EUR/CHF had fallen from 1.35 in June 2011 to briefly touch 1.00 in August — a 26% appreciation of the franc in two months.

For Switzerland, whose economy runs a current account surplus of roughly 8-12% of GDP and whose exports represent about 50% of GDP, this was an existential threat to the export sector. The SNB set the floor and enforced it through massive EUR purchases. In the process, it accumulated what became the world's second-largest foreign exchange reserve stockpile as a proportion of GDP.

![SNB EUR/CHF floor 2011 to 2015: four years held then sudden abandonment](/imgs/blogs/fx-intervention-when-central-banks-fight-the-market-6.png)

The floor held for four years and three months. During that period, EUR/CHF traded in a remarkably tight range just above 1.20. Volatility in EUR/CHF collapsed to near zero — options market implied volatility on EUR/CHF dropped to some of the lowest readings ever seen in a major currency pair. This low volatility made EUR/CHF a popular "carry corridor" for options sellers and structured product issuers who collected premium by selling options on a "boring" cross.

Then came January 15, 2015. The ECB was preparing to announce quantitative easing on January 22. The SNB had been warning markets privately that the floor was becoming increasingly costly. But the timing and lack of warning of the removal was a shock. EUR/CHF moved from 1.20 to 0.86 in minutes — a move larger than most FX pairs see in a year, delivered in a quarter-hour.

#### Worked example:

The carry trade math inside the EUR/CHF floor: in 2013, a EUR/CHF risk reversal (buying a CHF call / selling a CHF put, both one-month 25-delta) was trading at near zero — meaning the options market assigned roughly equal probability to CHF strengthening or weakening beyond the 1.20 level. The SNB floor made CHF strengthening below 1.20 seem impossible, so many options sellers were receiving premium on the CHF call (a bet that EUR/CHF stays above 1.20) with what felt like zero probability of being exercised. A fund selling CHF calls at 1.20 with one-month expiry and collecting \$50,000 per week in premium was running what appeared to be an almost riskless income stream. In January 2015, when EUR/CHF moved to 0.86, those short CHF call positions triggered at-the-money — a move of roughly 28% through the strike. A \$10 million notional exposure to short CHF calls with a 1.20 strike would have faced losses on the order of \$2.8 million in a single afternoon. For brokers with client accounts margined at 2-3% of notional, the losses far exceeded the margin posted.

### Case Study 3: Black Wednesday 1992 — Soros Breaks the Bank of England

The story of Black Wednesday, September 16, 1992, is the most famous speculative attack in FX history. To understand it, you need to understand the ERM.

The European Exchange Rate Mechanism, established in 1979, required member currencies to trade within 2.25% bands around a set of central rates against the German deutschmark (DEM). The UK joined in October 1990 at a central rate of DEM 2.95 per pound. The UK's inflation rate at the time was running around 10% — far above Germany's. To maintain the parity, the UK needed either to reduce its inflation quickly (difficult) or to accept that the market would pressure the pound toward the bottom of its ERM band.

By 1992, the UK was in recession. Unemployment was rising. Interest rates had been cut but not by enough to stimulate the economy. Meanwhile, German reunification had required the Bundesbank to keep rates high to contain the inflationary pressures from absorbing East Germany. The UK was locked into a rate structure set by German domestic needs, not UK domestic conditions. The pound was overvalued at 2.95 against the mark. Economists at the major investment banks were openly saying so. The question was not whether the pound would devalue, but when.

George Soros and other macro hedge fund managers looked at this situation and ran the numbers. The cost of borrowing pounds, selling them against the DEM, and holding the position until devaluation was manageable. The probability of devaluation — within ERM rules or through exit — was very high. The potential profit from a 10-15% realignment of sterling to its equilibrium level was enormous relative to the borrowing cost of the short position.

![GBP/USD around Black Wednesday 1992 with ERM floor and peg break annotated](/imgs/blogs/fx-intervention-when-central-banks-fight-the-market-5.png)

On September 16, 1992, the UK government desperately tried to defend sterling. The Bank of England spent approximately \$3.3 billion in foreign exchange reserves in a matter of hours — buying pounds with its dollar and DEM reserves. The Treasury twice raised interest rates in a single day (from 10% to 12%, then announcing 15% — the second hike was never actually implemented as the situation became hopeless). None of it worked. By the evening, the UK announced its withdrawal from the ERM. Sterling fell 15% over the following weeks.

Soros's Quantum Fund reportedly made approximately \$1 billion in profit. The UK Treasury's loss — in reserves spent, in the premium paid on sterling's interest rate commitment, and in the disruption to UK monetary policy credibility — was in the billions.

#### Worked example:

The speculative attack arithmetic: suppose a macro fund borrows \$1 billion worth of British pounds at the prevailing UK interest rate of 10% per annum, or approximately 0.83% per month (\$8.3 million per month in borrowing costs). The fund sells those pounds against DEM at the market rate. If the pound devalues by 15% against the DEM within 3 months, the fund repays the pound borrowing at the lower rate, pocketing the 15% currency gain minus 3 months of borrowing costs at 0.83% per month = approximately 2.5%. Net profit on a \$1 billion notional position: approximately \$125 million. The actual trade was more sophisticated — Soros used options to reduce his cost of carry — but the basic arithmetic shows why the attack was rational: expected profit was enormous relative to the cost of the bet, as long as devaluation probability was high, which it was once UK economic fundamentals made the ERM rate unsustainable.

## The Speculative Attack Model

Economists have developed a formal model of speculative attacks that explains why currency pegs so often end in crisis rather than gradual adjustment. The core insight is that a speculative attack is self-fulfilling once reserves drop below a critical threshold.

![Speculative attack mechanism from weak fundamentals through reserve exhaustion to peg collapse](/imgs/blogs/fx-intervention-when-central-banks-fight-the-market-7.png)

Here is the logic: if markets believe a peg will hold, no one attacks it, and it holds. If markets believe a peg will break, everyone attacks it, the reserves are spent, and it breaks — confirming the belief. The equilibrium shifts from "stable" to "crisis" based on a change in beliefs, not necessarily a change in fundamentals. This is why pegs often appear stable for years and then collapse suddenly: the fundamental weakness was there all along, but markets had priced it as low probability until some event — a data release, a political statement, a global risk-off shock — shifted the equilibrium.

The critical variables that determine which equilibrium prevails are: (1) the size of reserves relative to the daily flow of pressure, (2) the credibility of the government's commitment to the peg (if they say "we'll defend it no matter what," does the market believe them?), (3) the domestic political cost of raising rates to defend the currency, and (4) the availability of alternative policy tools (capital controls, new reserve borrowing, IMF support).

## When Intervention Succeeds vs. Fails

The empirical record across 50 years of floating exchange rates gives us a reasonably clear picture of when interventions work.

![Intervention success rate by reserve size and fundamental alignment](/imgs/blogs/fx-intervention-when-central-banks-fight-the-market-4.png)

**Interventions tend to succeed when:**

The fundamental value of the currency is close to the defended level. The SNB in 2011 was defending a level that was not wildly out of line with Switzerland's real exchange rate, given that the safe-haven inflows were distorting the currency beyond its equilibrium. Japan's interventions in 1998 were defending a yen that had overshot its fundamentals due to the Russian default and LTCM crisis — the fundamentals genuinely supported a stronger yen.

The intervention is accompanied by a broader policy shift. When the Thai baht was defended in 1997, the interventions failed because they were not accompanied by rate hikes or IMF support that would have been large enough to change the underlying dynamics. When the Czech koruna was defended in 2013 with a weaker-is-better peg (analogous to the SNB floor), it worked because the CNB stood ready to sell unlimited korunas — defending currency weakness rather than strength, so it could create as much supply as needed.

Market positioning is already extreme in one direction. Soros's edge in 1992 came from the clear imbalance: the pound was overvalued, rates were wrong, and the market had not yet fully priced the devaluation probability. By the time a currency has already moved significantly in the direction the intervention is opposing and speculative positions are already extreme, intervention faces the headwind of all those short positions needing to be covered — which provides fuel for the reversal.

**Interventions tend to fail when:**

The defended level is far from fundamental equilibrium. Venezuela's currency interventions, Argentina's managed peso regime, Turkey's various reserve drawdowns in 2018-2021 — all failed because the currencies were genuinely overvalued relative to domestic inflation and productivity.

The reserves are insufficient. Emerging market central banks with \$20-30 billion in reserves defending currencies with \$5-10 billion of daily trading volume face a mathematical problem: the market can simply outlast them. When the reserve drawdown becomes visible in weekly or monthly data releases, it accelerates the attack by signaling that the defense is becoming unsustainable.

Interest rate rises needed to defend the currency are politically or economically impossible. The UK in 1992 could not raise rates to 15% without triggering a wave of mortgage defaults and a deep recession. The political cost was too high, making the defense unconvincing. Speculators read this correctly.

## Common Misconceptions

**Misconception 1: "Central bank intervention always fails in the long run."**

The empirical record shows this is not true. Japan's interventions in 1998 and 2003–2004 successfully stabilized the yen when the fundamentals supported a stronger yen. The SNB held the EUR/CHF floor for four years. The Czech National Bank held a EUR/CZK floor from November 2013 to April 2017 — over three years — at a cost of approximately EUR 50 billion in reserves accumulated. The Swiss franc is still, as of 2025, roughly 15% stronger than the pre-2011 level but not at the catastrophic 1.00 level the SNB feared. Intervention bought real years of time.

**Misconception 2: "Sterilized intervention is pointless."**

While the academic literature is skeptical of sterilized intervention's lasting price impact, the signaling channel is real and can be material. The act of intervening — even if sterilized — communicates that the central bank is watching the rate and is willing to act. This can shift the expected path of monetary policy (markets may conclude that intervention is a precursor to rate hikes). For short-term volatility smoothing, sterilized intervention reliably works.

**Misconception 3: "The reserves data tells you exactly when intervention will stop."**

The reserves figures published by central banks are monthly, often with a one-month lag, and exclude off-balance-sheet commitments (FX swaps, IMF drawing rights, bilateral currency swaps from other central banks). Japan can also borrow from the US through the bilateral currency swap agreement. The "true" firepower available to a central bank defending its currency is always larger — and harder to measure — than the reported reserves alone. However, the rate of reserve drawdown over multiple months remains a valid signal that the defense is under pressure.

**Misconception 4: "Intervention is only about the spot rate."**

Major central banks also intervene in the options market. Buying near-term options on the domestic currency can influence implied volatility and effectively set a ceiling on speculative positions by making them more expensive. Japan's MOF reportedly uses FX options as well as spot intervention. The SNB in 2011 set an explicit price floor that effectively made any EUR/CHF put option below 1.20 worthless — wiping out hundreds of millions of dollars of outstanding options positions in a single announcement.

**Misconception 5: "Only weak currencies require intervention."**

The SNB's 2011–2015 episode was an intervention to prevent the franc from strengthening too much. The Czech National Bank's 2013 floor was similarly aimed at preventing excessive koruna strength. Interventions to weaken a currency are mechanically easier (you can create unlimited domestic currency) and often aim at a different policy objective: preventing deflation, protecting export competitiveness, or managing the pass-through of global asset reflation into the domestic economy.

## How It Shows Up in Real Markets

### The options market as an early warning system

The most sophisticated early warning signal of an impending intervention — or a peg break — comes from the options market. Specifically, the risk reversal: the relative cost of puts versus calls on a currency at the same delta distance from the current spot rate. When puts become expensive relative to calls (a negative risk reversal), the market is paying a premium for protection against currency weakness. That premium reflects the market's asymmetric expectation.

In early 2022, before the BOJ's first intervention, USD/JPY one-month 25-delta risk reversals moved sharply in favor of USD calls (yen puts) — traders were paying up for the right to sell more yen. This was an observable signal that the market expected further yen weakness and was positioning for it. When intervention came, it was against this positioning — which is why the initial move was sharp.

For the SNB in January 2015, the options market missed the peg removal: EUR/CHF one-month implied volatility was trading around 0.5% — essentially zero — just days before the floor was removed. The market had priced in near-zero probability of a break. The lesson: when options markets are pricing extreme calm on a pair with known political support, the risk of a discontinuous move is mispriced.

### Reserve drawdown data as the exhaustion signal

Japan's Ministry of Finance publishes monthly foreign exchange intervention data with a one-month lag. More timely is the Bank of Japan's daily statements about reserve account balances, which can be read against baseline expectations to infer whether intervention occurred. When daily reserves account balances are significantly higher than predicted by normal bond settlement flows, intervention occurred that day.

More broadly, emerging market central banks publish reserve figures monthly (IMF Data Standards). Watching the trajectory: a central bank that has spent 15% of its reserves in three months defending a level is signaling its limit is approaching. The speed of drawdown is as important as the absolute level.

#### Worked example:

Reading the Turkey exhaustion signal in 2018: Turkey's central bank began the year with approximately \$100 billion in gross reserves. Through the spring and summer, as the lira collapsed (from 3.80 TRY/USD in January to 7.24 in August 2018), the CBRT was conducting swap operations that consumed reserves at a rate of roughly \$3-5 billion per month. By August, net reserves (adjusting for swaps) had fallen to approximately \$50 billion. The pace of drawdown — visible monthly in the CBRT's published reserve figures — showed the defense was becoming fiscally unsustainable. The lira ultimately fell 45% peak-to-trough in 2018. A practitioner tracking the monthly reserve data alongside the monthly TRY/USD move had a quantifiable signal: each 5% TRY depreciation was drawing down approximately \$6 billion in reserves, implying roughly 8 more months of defense at the current pace before exhaustion. That pace was politically unsustainable before the mathematical exhaustion point, given Turkish domestic inflation was running above 15%.

## How to Trade It: The Intervention Playbook

Trading around FX interventions is one of the few macro strategies where the setup is observable in advance and the mechanics are well-understood. The key rules are:

### Rule 1: Never front-run the central bank

If you know intervention is coming, the instinct is to buy the currency the central bank is about to buy. This is dangerous. Central banks intervene in size, and their timing is deliberately unpredictable. Being long a currency into intervention is not a bad trade if the intervention is already happening — but predicting the exact entry point is essentially impossible from publicly available information. Position sizing should reflect this uncertainty.

### Rule 2: Fade the initial intervention spike

The immediate price reaction to confirmed intervention is often 2-4% in a matter of hours. This initial move is frequently a tradable overreaction, particularly when the intervention is purely sterilized and the underlying policy has not changed. The classic trade: when a central bank confirms it has intervened (either through statement or through MOF data disclosure), the short-term overreaction creates an opportunity to fade the spike — taking a small position against the intervention-induced move, with a tight stop.

The math: if the yen moves 4% in two hours on intervention, and historical data shows that 60% of the initial spike is given back within two weeks when the BOJ has not changed its YCC policy, then a small mean-reversion position at the spike extreme has positive expected value. Position size should be small relative to capital: FX intervention trades have fat tails in the adverse direction if the central bank decides to double down.

### Rule 3: Watch the reserve drawdown for exhaustion

As detailed above, the monthly reserve data is your structural signal. When reserves have fallen 20%+ from peak levels and the currency has not stabilized, the intervention is losing its effectiveness. Positioning to benefit from currency weakness at that point — through options or outright short positions — becomes better-risk-adjusted.

### Rule 4: Trade the peg break asymmetry

When a currency is pegged and the peg is under stress, the payoff profile is asymmetric: if the peg holds, you lose a small carry; if the peg breaks, you gain a large directional move. The SNB floor trade in reverse — buying EUR/CHF puts at 1.20 strike through 2013 and 2014 — was cheap because implied volatility was near zero and the options were far out-of-the-money (EUR/CHF was trading at 1.22-1.25). Those puts became extraordinarily valuable on January 15, 2015. The cost of running this trade for two years (two years of small option premiums) was less than 0.5% of notional. The potential payoff if the floor broke was 10-15% of notional.

This trade only works when you can clearly identify: (a) a currency with an explicit or implicit peg or floor, (b) where the fundamental value is diverging from the defended level, (c) where the central bank's cost of defense is rising visibly. All three criteria were met for EUR/CHF through 2014.

### Rule 5: The three-phase intervention trading framework

Practitioners who trade frequently around intervention episodes think in three phases:

**Phase 1 — Pre-intervention accumulation:** The currency is moving in one direction, volume is rising, options risk reversals are skewed, and reserve drawdown data suggests the central bank may act. The trade is to be positioned for continued directional movement but with reduced size and a mental stop at levels where intervention becomes likely. Position in the direction of the trend but expect a temporary reversal.

**Phase 2 — Intervention reaction:** The central bank acts. Volatility spikes. The currency reverses sharply. Reduce or cover directional positions immediately. Wait for the initial spike to exhaust — typically 2-4 hours of sharp reversal followed by gradually returning trend pressure.

**Phase 3 — Post-intervention fade:** Once the intervention spike is partially reversed and new data (reserve reports, BOJ statements, MOF disclosures) confirms the intervention was sterilized and the policy unchanged, re-establish the directional trade but with smaller size and wider stops. The 2022 BOJ playbook: sell the yen each time it rallied post-intervention, with a stop above the intervention entry point. This trade worked multiple times through the year, with each intervention rally fading back toward new lows.

## Further Reading and Cross-Links

The mechanics of FX markets — how spot, forwards, and cross-rates relate — are covered in depth in the forex series at [trading/forex](/trading/forex). The broader macro backdrop of central bank policy divergence and its effect on currency trends is explored in [trading/macro-trading](/trading/macro-trading). The Turkish lira and Argentine peso currency crises — where reserve exhaustion led to full-scale financial crises — appear as detailed case studies in [trading/geopolitical-crises](/trading/geopolitical-crises). The speculative attack model connects to the broader literature on currency crises and balance-of-payments dynamics covered in [trading/capital-markets](/trading/capital-markets). For practitioners looking at options-based approaches to trading intervention asymmetry, the options and volatility series at [trading/options-volatility](/trading/options-volatility) covers risk reversals, skew, and event-driven vol strategies in full detail.

The three case studies here — Japan 2022, Switzerland 2011-2015, and the UK 1992 — represent the full range of outcomes: costly-but-time-buying (Japan), years-long success then sudden failure (Switzerland), and rapid failure with lasting policy consequences (UK). Understanding which category a current intervention falls into — by watching the three Ps, the reserve trajectory, the options skew, and the fundamental alignment — is the core skill in trading around central bank currency defense.
