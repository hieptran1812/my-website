# Geopolitical Crises & Markets — SERIES ROADMAP

**Series title:** *Geopolitical Crises & Markets: How Nations Move, How Markets React*
**Folder:** `content/blog/trading/geopolitical-crises/`
**subcategory:** `"Geopolitical Crises"`
**Skill:** finance-writer (deep-dive). Kit: `.cache/finance-writer/_geopolitical-crises/_kit.md`. Data: `data_geopolitics.py`. Render: reuse `_macro/render-post.sh`. Chartkit: reuse `_macro/chartkit.py`.
**Scope:** 86 posts, 15 tracks (A–O). finance-writer, ≥8,800 words + ≥7 figs/post.

---

## THE SPINE
**Geopolitical Event → Political Moves by Nations → Policy & Economic Channels → Asset Price Reaction → the Trade.**

Every post treats a geopolitical event or crisis type as a *case study*:
1. **The event** — what triggered it, why it matters
2. **The moves** — what each major actor (US, China, EU, regional powers, the target country) *chose* to do politically, economically, militarily — and why that specific move, not another
3. **The channels** — which economic/financial levers those moves pulled (sanctions, tariffs, energy, FX, credit, capital flows)
4. **The market impact** — which asset classes moved, how fast, how far, and whether the move was permanent or a spike
5. **The trade** — what signals to watch, how a practitioner positions, what invalidates the view

**Different from law-and-geopolitics:** That series = legal/regulatory lens (rules move markets). This series = event/crisis lens (nation-level political decisions move markets). Cross-links out to law-and-geopolitics for legal mechanics; never re-derives them.

---

## CROSS-LINK PHILOSOPHY
This series cross-links OUT to:
- `trading/law-and-geopolitics/` — for legal/regulatory mechanics (sanctions law, tariff authority, CFIUS)
- `trading/macro-trading/` — for macro transmission (how the Fed, oil prices, the dollar work)
- `trading/event-trading/` — for reaction patterns (how fast markets reprice specific data releases)
- `trading/fixed-income/`, `trading/gold/`, `trading/forex/`, `trading/options-volatility/` — asset-class mechanics
- `trading/cross-asset/` — correlation regimes in risk-off events

It OWNS the "who moved what politically and why" layer that the other series assume.

---

## BUILD MECHANICS (per wave)
- Parallel background agents (one post each). Kit = single source of truth.
- Main session: `render-post.sh <slug>` → `npm run optimize-blog-images` → `verify-finance-post.sh <post.md> <slug> deep-dive` → fix → commit wave's md+webp (explicit paths) → push main.
- Gates: ≥8,800 words, ≥7 figures, ≥4 worked examples, TL;DR-at-top, Foundations heading.
- `.png` embeds in markdown (NOT `.webp`). Escape `\$` in body.

---

## STATUS TRACKING
- [ ] Wave 1 — Track A: Framework (8 posts: A1–A8)
- [ ] Wave 2 — Track B: War & Armed Conflict Part 1 (5 posts)
- [ ] Wave 3 — Track B: War & Armed Conflict Part 2 (5 posts)
- [ ] Wave 4 — Track C: Trade Wars & Tariffs (8 posts)
- [ ] Wave 5 — Track D: Sanctions & Economic Warfare (7 posts)
- [ ] Wave 6 — Track E: Energy Geopolitics (6 posts)
- [ ] Wave 7 — Track F: Currency & Capital Wars (6 posts)
- [ ] Wave 8 — Track G: Elections & Political Transitions (5 posts)
- [ ] Wave 9 — Track H: Vietnam & ASEAN (5 posts)
- [ ] Wave 10 — Track J: Coups & Political Instability (6 posts)
- [ ] Wave 11 — Track K: Sovereign Debt Crises (6 posts)
- [ ] Wave 12 — Track L: Treaties & Agreements (5 posts)
- [ ] Wave 13 — Track M: Pandemics & Black Swans (5 posts)
- [ ] Wave 14 — Track N: Cyber & Tech Nationalism (5 posts)
- [ ] Wave 15 — Track O: Resources & Climate + Track I Capstone (7 posts)

---

## TRACK A — Framework: How to Read Geopolitical Risk (Wave 1, 5 posts)

**A1** `geopolitical-risk-premium-what-markets-price-in`
> The geopolitical risk premium: how markets discount uncertain futures, the VIX-of-geopolitics concept, spike vs. regime repricing, the 3-phase cycle (shock → discount → fade or lock-in). Cover fig = the three-phase market response cycle.

**A2** `asset-sensitivity-matrix-which-assets-move-in-which-crisis`
> The 6×5 matrix: crisis type (war, sanctions, trade war, coup, pandemic, debt crisis) × asset class (equities, bonds, gold, oil, USD, EM FX) → direction+magnitude+duration. Why gold and short-term Treasuries behave differently than oil.

**A3** `political-moves-decoded-how-governments-choose-their-response`
> The political science layer: why governments choose sanctions vs. tariffs vs. military vs. diplomacy; the domestic-politics constraint; the alliance calculus; how to read a press conference for the real signal; the "policy toolkit matrix" every investor needs.

**A4** `safe-havens-flight-to-quality-and-the-dollar-in-a-crisis`
> Safe-haven mechanics: USD, CHF, JPY, gold, Treasuries — when each works and when it doesn't; the dollar-smile in risk-off; how fast safe-haven premiums fade; 2022 vs. 2020 vs. 2008 comparisons.

**A5** `reading-a-geopolitical-shock-spike-vs-regime-change`
> The most important distinction in geopolitical trading: a spike (recovers in days/weeks) vs. a regime change (permanent repricing). Historical hit rates, the signals that separate them (supply disruption length, alliance shifts, sanctions permanence), how to position for each.

**A6** `how-to-read-a-countrys-political-signals-before-it-acts`
> The early-warning intelligence layer: how to detect that a country is about to make a political/economic/military move BEFORE it announces. Six signal categories: (1) diplomatic language changes (UN speeches, foreign minister statements shift from "concern" to "unacceptable"), (2) military positioning (satellite imagery of troop movements, naval exercises near flashpoints), (3) legislative/regulatory telegraph (laws enabling asset seizure, emergency powers drafted), (4) economic pre-positioning (FX reserve drawdown, import stockpiling, state-enterprise restructuring), (5) leadership reshuffles (hawkish appointments to finance/defense/intelligence), (6) state-media framing (when a domestic narrative shifts from neutral to hostile toward a target country). Case studies: how all six signals appeared before Russia's 2022 invasion, the US 2018 Section 301 investigation, China's 2020 Australian barley/wine tariffs. How a practitioner builds a "political dashboard" to monitor these signals. What false positives look like (military exercises that don't escalate) and how to weight them. Market positioning: when signal count crosses threshold, adjust portfolio before the event — not after.

**A7** `forecasting-geopolitical-events-frameworks-base-rates-and-scenarios`
> How to think probabilistically about future political events — the intelligence community's structured approach made investable. Key tools: (1) Base rates — historically, how often do trade disputes escalate to tariffs (60%), how often does military posturing lead to armed conflict (15%), how often do sanctions actually change a target country's policy (30%); (2) Structured analogy — find the 3-5 closest historical cases and use their outcomes as your probability distribution; (3) Scenario analysis — define 3 scenarios (baseline/escalation/de-escalation), assign probabilities, map each to asset-class outcomes, then size positions by expected value; (4) Prediction markets as calibration (Polymarket, Metaculus geopolitical forecasts — how to use them without over-relying on them); (5) Red flags vs. green flags — specific observable events that would shift the probability of each scenario. Applied examples: current Taiwan scenario (what specific signals shift invasion probability from 5% to 20%); the Iran nuclear threshold (what IAEA report language changes the market's risk premium); the Russia de-escalation path (what specific diplomatic steps would unlock a rally in Russian-exposed assets). Common mistakes: anchoring on the most recent event, ignoring base rates, confusing "shocking if it happens" with "likely to happen."

**A8** `anticipating-the-next-move-escalation-ladders-in-geopolitical-crises`
> Once a crisis starts, what comes next? The escalation-ladder framework applied to modern geopolitics: (1) Herman Kahn's 44-rung ladder (1965) mapped to contemporary instruments — diplomatic protest → economic pressure → targeted sanctions → comprehensive sanctions → proxy conflict → limited direct military action → strategic escalation; (2) Actor-specific move repertoires — the US toolkit (OFAC sanctions, Section 232/301 tariffs, export controls, military posture changes, NATO activation, intelligence sharing), China's toolkit (economic coercion/boycotts, gray-zone maritime harassment, cyber operations, rare-earth controls, diplomatic isolation campaigns, Taiwan strait signaling), Russia's toolkit (energy weaponization, information warfare, nuclear signaling levels, proxy forces, grain/food leverage); (3) The "tit-for-tat" pattern — empirically, ~70% of retaliatory moves in trade/sanctions disputes match the opponent's move in scale within ±20% and land within 30 days; military proxy responses typically come in 14–60 days; (4) Red-line identification — what specific observable action by which actor triggers the next rung (Taiwan scenario: ADIZ intrusion vs. naval blockade vs. missile test over the island vs. amphibious assault; Iran nuclear: enrichment to 90% vs. weapon assembly vs. detonation test; US-China tech war: FDPR extension vs. full TSMC ban vs. semiconductor import controls); (5) De-escalation pathway anatomy — the "face-saving exit ramp" each major actor structurally needs and what its observable signals look like (back-channel diplomat activations, ambiguous press statements, third-party mediator emergence, domestic political framing shifts); (6) The investor's anticipation framework: (a) locate current rung on the ladder, (b) enumerate the 3–5 most probable next rungs with probability weights, (c) map each rung to its primary asset impact (e.g., Rung 7→oil +12%, Rung 8→gold +8%, USD +3%), (d) compute probability-weighted expected move, (e) size position accordingly with defined invalidation events. Live case study walkthroughs: current US-China semiconductor war (where are we on the ladder, what are Beijing's remaining coercive moves, what signals a TSMC ban is imminent), Russia's residual coercive toolkit (which levers still unplayed, what crossing each would do to European gas and EM spreads), Iran's nuclear escalation options (the three credible paths and each path's oil-price implications). Common mistakes: anchoring on the last move made instead of the residual toolkit; ignoring the domestic political constraint that makes certain rungs harder than they look; treating de-escalation as binary (either full war or full peace).

---

## TRACK B — War & Armed Conflict (Waves 2–3, 10 posts)

**B1** `russia-invades-ukraine-the-first-72-hours-in-markets`
> Feb 24, 2022: real-time asset reactions in the first 72 hours; why markets had partly priced it; the 11% VIX spike; what crude, wheat, MSCI EM, and EUR/USD did; how to read the fog of war for positioning.

**B2** `russia-ukraine-war-the-energy-shock-playbook`
> Europe's dependence on Russian gas (40%+ of supply), the pipeline-to-LNG pivot, TTF gas price surge to €300+/MWh, energy inflation transmission to CPI, German industrial recession. Political moves: Russia weaponizes gas, EU scrambles for alternatives, US unlocks LNG exports.

**B3** `russia-ukraine-war-sanctions-swift-and-the-ruble-collapse`
> The sanctions architecture: who hit what, why the ruble initially crashed then recovered, the reserve freeze (Russia's \$300B frozen), secondary sanctions, SWIFT exclusion; what the ruble V-shape tells us about the limits of financial sanctions.

**B4** `russia-ukraine-war-food-wheat-fertilizer-and-the-global-south`
> Ukraine = the world's breadbasket (30% of global wheat); the Black Sea export blockade; fertilizer sanctions (Russian potash, Belarus); food inflation in MENA/Africa; the political fallout in countries that needed grain, not geopolitics.

**B5** `iran-israel-shadow-war-drone-strikes-and-the-oil-risk-premium`
> The shadow war: proxy conflict (Hezbollah, Houthi, Syrian proxies), the April 2024 direct exchanges, why oil markets priced a smaller premium than you'd expect; the Strait of Hormuz threat; what it takes to actually move Brent sustainably.

**B6** `iran-israel-nuclear-timeline-and-market-pricing`
> How markets price nuclear escalation risk (binary but tail); the JCPOA signing vs. withdrawal reactions; what "Iran goes nuclear" would actually mean for oil, regional EM equities, and Saudi Arabia; historical analogies (India/Pakistan 1998).

**B7** `taiwan-strait-tensions-semiconductor-risk-and-supply-chain-repricing`
> The Taiwan scenario: TSMC as geopolitical hostage (60%+ of advanced chips from one island); the "silicon shield" thesis; how markets priced the Pelosi visit (Aug 2022); China's "reunification pressure" toolkit vs. invasion; the supply-chain insurance trade (onshoring premium, ASX, CHIPS Act).

**B8** `north-korea-escalations-how-markets-price-a-non-event`
> Why NK missile tests barely move markets: the "boy who cried wolf" premium, the Seoul/Tokyo buffer, the China-NK relationship as a circuit breaker; the moments that *did* move markets (2017 H-bomb); the difference between signaling and escalation.

**B9** `the-gaza-war-middle-east-contagion-risk-and-what-it-took-to-spread`
> Oct 7, 2023: why global markets mostly shrugged; the contagion test (Hezbollah involvement, Houthi Red Sea disruption, US-Iran); the 3-month Houthi attack and shipping repricing; the regional escalation scenario and what specific assets it would move.

**B10** `every-major-war-and-what-markets-did-the-historical-playbook`
> The full dataset: Korea, Vietnam, Gulf War (1991), Kosovo, Afghanistan/Iraq post-9/11, Libya, Syria, Russia-Georgia 2008, Ukraine 2014 vs. 2022. Pattern: initial shock → recovery vs. structural change. When war is bullish for equities (stimulus) and when it's bearish (supply shock). The WW2 case.

---

## TRACK C — Trade Wars & Tariffs (Wave 4, 8 posts)

**C1** `how-a-trade-war-starts-trigger-escalation-negotiation`
> The anatomy: the initial grievance, the legal instrument (Section 301/232/IEEPA), the escalation ladder, the negotiation endgame; the five forces that decide who blinks first; historical escalation timelines.

**C2** `us-china-trade-war-2018-19-tariff-rounds-and-market-scorecards`
> The full timeline: 7 tariff rounds, average tariff 3%→21%, market reaction per round, the yuan depreciation offset, the soybean/rare-earth retaliation, the Phase One deal (and what it did not fix); sector winners (Vietnam, Mexico) and losers (US agriculture, Chinese tech).

**C3** `the-2025-trump-tariff-shock-repricing-the-global-supply-chain`
> Liberation Day 2025: the 10% baseline + reciprocal tariff architecture, the 90-day pause, what countries negotiated and what they couldn't; how the S&P, dollar, and Treasuries moved; the "tariff put" thesis and when it expires.

**C4** `tariffs-and-the-dollar-why-trade-wars-are-currency-wars`
> The currency offset mechanism: tariffs → dollar appreciation → offset trade pain → forces trading partners to devalue; the RMB-USDCNH link; the 7.00 line in USD/CNH and what breaking it signals; how to trade the currency side of a tariff announcement.

**C5** `trade-war-winners-vietnam-mexico-india-and-the-rerouting-trade`
> The supply-chain exodus: FDI flows to Vietnam (Samsung, Intel, Apple suppliers), Mexico's nearshoring boom, India electronics push; how to pick the winner before the FDI data confirms it; VN-Index reaction to tariff news as a "safe harbor" signal.

**C6** `chinas-retaliation-playbook-soy-boeing-rare-earths-tourism`
> China's toolkit: agricultural tariffs (soy, pork), Boeing cancellations, rare-earth export controls (germanium, gallium, graphite 2023), tourism bans, consumer boycotts; why China's retaliation is calibrated, not maximized — the hostage logic.

**C7** `semiconductor-export-controls-as-a-geopolitical-weapon`
> The chip war: Oct 2022 BIS rule (A100/H100 cut-off), the FDPR extension, ASML EUV ban; how Nvidia's China revenue collapsed from 25%→8%; the stock reactions; the "Swiss cheese" in enforcement and what fills the gap; the TSMC, Samsung, Intel beneficiary thesis.

**C8** `wto-dispute-settlement-why-markets-ignore-it-but-shouldnt`
> The WTO's Appellate Body crisis (US blocked appointments 2017→deadlock); why Panel rulings take 3–7 years; the "GATT-illegal but politically stable" tariff; the moment WTO wins actually matter (carbon border adjustment, digital services taxes); when the rule of law bites back.

---

## TRACK D — Sanctions & Economic Warfare (Wave 5, 7 posts)

**D1** `how-sanctions-work-primary-secondary-and-the-enforcement-gap`
> The legal anatomy of a sanction: OFAC, SDN list, secondary sanctions (extraterritorial reach), the enforcement chain; why sanctions leak (workarounds: third countries, crypto, barter, shadow fleets); the 40% effectiveness estimate and what it means for markets.

**D2** `russia-sanctions-anatomy-swift-reserves-oil-price-cap`
> The most comprehensive sanctions package in history: the \$300B reserve freeze, SWIFT cut for 7 banks, the G7 oil price cap at \$60/bbl, the shipping and insurance ban; what actually hurt Russia (reserve freeze, chip imports) vs. what didn't (oil flows rerouted to India/China).

**D3** `iran-sanctions-40-years-of-adaptation-and-what-oil-markets-learned`
> 1979→2024: the full sanctions timeline; Iran's adaptations (ghost tanker fleet, barter with China, crypto for oil); the JCPOA price reaction vs. the 2018 withdrawal reaction; the enforcement gap (India/China exemptions); the oil-market signal: Iranian barrels appear when sanctions enforcement softens.

**D4** `the-weaponization-of-the-dollar-swift-as-a-strategic-tool`
> Why the dollar's reserve status gives the US a "financial WMD"; the SWIFT exclusion architecture (not SWIFT itself but correspondent banking); the "use it too much and lose it" paradox; de-dollarization demand post-2022 Russia; what reserve-currency history says about the dollar's shelf life.

**D5** `dedollarization-real-trend-or-geopolitical-theater`
> The data: USD share of global reserves fell from 72%→59% over 20 years; CNY share stuck at 2%; the BRICS payment alternatives; the "petrodollar" mythology and the actual Saudi-China oil-in-yuan pilot; why dollar alternatives face the liquidity trap; what would actually move the needle.

**D6** `sanctioned-commodity-markets-shadow-fleets-price-gaps-arbitrage`
> The infrastructure of sanctions evasion: Russia's shadow tanker fleet (600+ vessels, 40% of crude moves), the Urals-Brent discount (peak \$30/bbl), Indian refinery "laundering" (Russian crude → Indian product → Europe), the arbitrage trade and the enforcement risk.

**D7** `how-sanctions-fail-and-what-that-means-for-asset-prices`
> The 5 failure modes of sanctions: (1) third-country circumvention, (2) commodity rerouting, (3) domestic adaptation, (4) political resistance hardening, (5) coalition fragmentation. Historical: Cuba, Iran, Russia, North Korea. The market read: when failure signals buy the target's assets.

---

## TRACK E — Energy Geopolitics (Wave 6, 6 posts)

**E1** `opec-plus-as-a-political-actor-production-cuts-and-market-power`
> OPEC+'s evolution from cartel to Saudi-Russia geopolitical instrument; the anatomy of a production cut decision (Riyadh-Moscow calls, quota allocation, cheating); the 2022 Oct cut (snub to Biden), the 2023 1.66mb/d cut, the 2024 voluntary cuts; market reaction per decision; the "OPEC put" price floor concept.

**E2** `the-oil-weapon-when-energy-becomes-foreign-policy`
> 1973 Arab oil embargo as the template: the political trigger (Yom Kippur War), the embargo mechanism, the 4× oil price spike, global stagflation; the lessons: why pure oil embargos fail against the US/Europe now but still work against exposed importers (Japan, South Korea, Germany).

**E3** `nord-stream-sabotage-and-the-european-energy-repricing`
> Sep 2022: Nord Stream 1&2 destroyed; 77bn cubic meters of capacity offline; the TTF spike, German industrial competitiveness cliff, the political whodunit (unsolved), the long-term structural shift from Russian pipeline gas to LNG; who gained (US LNG exporters, Norwegian gas, nuclear revival).

**E4** `lng-rerouting-how-europe-replaced-russian-gas-in-18-months`
> The fastest energy pivot in history: Europe went from 40% Russian pipeline gas to near zero; the new LNG terminal build-out (Germany, Netherlands, Italy), the US LNG contract surge, the TTF→Henry Hub spread arbitrage; the price: European industry paying 3× pre-crisis gas for years.

**E5** `rare-earths-and-critical-minerals-chinas-geopolitical-leverage`
> China controls 85% of rare-earth processing; the 2010 Japan embargo (Senkaku dispute); the 2023 germanium/gallium controls; the 2024 graphite restrictions; the EV battery supply chain (lithium in Chile/Australia, cobalt in DRC, nickel in Indonesia); what the US/EU diversification timeline actually looks like.

**E6** `the-green-energy-race-as-geopolitical-strategy`
> The IRA (2022) as an industrial policy weapon (EV/battery/solar subsidies); EU reaction (net-zero industry act, carbon border adjustment); China's solar dominance (80% of panels); the "subsidy war" in EVs; how energy transition policy moves commodity prices (lithium, copper, cobalt cycles).

---

## TRACK F — Currency & Capital Wars (Wave 7, 6 posts)

**F1** `competitive-devaluation-when-countries-race-to-the-bottom`
> The currency war taxonomy: unilateral devaluation (China 2015), FX intervention (Japan 2022), interest-rate-driven depreciation (Turkey 2021-23); the "beggar thy neighbor" logic and the retaliatory cycle; how to detect a currency war in the data (intervention reserves, real effective exchange rate).

**F2** `capital-controls-as-a-crisis-response`
> The political economy of capital controls: Iceland 2008 (complete closure, worked), Cyprus 2013 (bank deposit freeze), Argentina (multiple), China (QDII limits); when they work vs. backfire; what the announcement does to equities and FX overnight; the cross-asset signal when controls are *lifted*.

**F3** `fx-intervention-when-central-banks-fight-the-market`
> The intervention toolkit: sterilized vs. unsterilized, the "three Ps" (price, pace, proximity to elections), the Bank of Japan's 2022 \$60B+ defense of 145; when intervention succeeds (SNB 2011 peg) vs. fails (UK Black Wednesday 1992); the speculative attack model and how Soros won.

**F4** `the-yen-carry-trade-and-geopolitical-stress-events`
> Why the yen is the world's ultimate safe-haven-carry paradox: low rates fund global risk; geopolitical shocks trigger carry unwind (yen rallies when stocks fall); the Aug 2024 Bank of Japan rate hike unwind (global equity selloff in 2 days); how to hedge/exploit the yen carry in a crisis.

**F5** `emerging-market-currency-crises-triggered-by-geopolitics`
> The external-shock template: geopolitical risk → capital flight → currency collapses → EM central bank forced to hike → recession. Case studies: TRY (Turkey post-coup 2016, Erdogan rate policy), ARS (Argentina default cycles), ZAR (South Africa load-shedding + political risk), VND context.

**F6** `brics-and-the-alternative-payment-system-gamble`
> The de-dollarization project: BRICS+, the mBridge CBDC corridor, bilateral local-currency trade (India-Russia oil in rupees), the Yuan-petrostate deals; why the SWIFT alternative is still 5–10 years away; what a genuine reserve-currency shift would do to gold, Treasuries, and the dollar index.

---

## TRACK G — Elections & Political Transitions (Wave 8, 5 posts)

**G1** `trading-the-us-election-sectors-dollar-rate-expectations`
> The US election playbook: sector rotation by policy agenda (defense, energy, healthcare, infrastructure), the fiscal expansion bet, dollar reaction (Republican = strong, Democrat = weaker historically?); the "uncertainty premium" that rises in the 60 days before; post-2024 election case study.

**G2** `brexit-the-complete-market-timeline-from-referendum-to-trade-deal`
> Jun 23, 2016 → Jan 1, 2021: the full sterling and FTSE timeline; the "shock pricing" on Jun 24 (GBP -8% in hours), the 4-year uncertainty premium on UK equities (FTSE vs. MSCI World), the Jan 2021 Trade Deal → brief relief → the reality of non-tariff barriers; what the 8-year return of GBP tells you.

**G3** `emerging-market-political-risk-turkey-argentina-brazil-case-studies`
> Three EM political-risk templates: Turkey (Erdogan fires 3 central bank governors → TRY -45%, -30%, -20%), Argentina (Kirchner → Macri → Fernandez → Milei — what each transition did to spreads, the peso, and Merval), Brazil (Lula vs. Bolsonaro and BOVESPA swings). The generic EM election signal: local bond spreads widen 60 days before.

**G4** `coalition-governments-policy-paralysis-and-credit-spreads`
> When hung parliaments = market risk: Italy 2018 (5-Star/Lega coalition, BTP-Bund spread +300bp), Israel's revolving governments, Germany's post-2021 coalition uncertainty; why coalition formation delay is measurable in sovereign spreads; the "political uncertainty premium" on credit.

**G5** `the-political-business-cycle-and-fiscal-timing`
> The classic model (Nordhaus 1975): governments boost growth before elections, tighten after. The evidence in US data (fiscal impulse by year of presidential cycle); the S&P presidential cycle pattern (year 3 strongest); how to trade the infrastructure/defense spending wave of an election year.

---

## TRACK H — Vietnam & ASEAN in the Great Game (Wave 9, 5 posts)

**H1** `vietnams-strategic-positioning-friend-to-all-enemy-to-none`
> Vietnam's "bamboo diplomacy": US strategic partner + China's largest trading partner + Russia arms buyer + ASEAN chair; how Hanoi navigates great-power competition to stay off the sanctions list; the economic dividend (FDI inflows, export growth) and the political cost.

**H2** `how-us-china-tensions-move-the-vn-index`
> The VN-Index geopolitical beta: event-study of US-China tension spikes → VN-Index reaction (initial risk-off → recovery as supply-chain winner narrative takes hold); the "China +1" premium in FDI → VND FX → stock sectors (industrial parks, logistics, tech assembly).

**H3** `asean-geopolitical-beta-which-countries-win-a-decoupled-world`
> The decoupling winners matrix: Vietnam (electronics, textiles), Malaysia (semiconductors, data centers), Indonesia (nickel, EV battery materials), Thailand (auto, food), Philippines (BPO, remittances); the loser: countries too dependent on China's supply chain (Cambodia, Laos).

**H4** `vietnam-as-a-supply-chain-rerouting-destination-fdi-and-exports`
> The numbers: Vietnam FDI went from \$15B/year pre-2018 to \$35B+ post-tariff war; electronics exports now >30% of GDP; Samsung makes 50% of its phones in Vietnam; Apple's Foxconn shift; the bottlenecks (infrastructure, logistics, skilled labor) and the political risk (Vietnam's own regulatory whims).

**H5** `the-south-china-sea-dispute-and-vn-market-risk-premium`
> The SCS dispute: UNCLOS, the 2016 Arbitration (China refuses), the fishing/oil-rig incidents; what an escalation would mean for VN shipping (80% of trade through contested waters), the VN-Index "SCS discount", and how to trade a spike (sell logistics, buy defense adjacent).

---

## TRACK J — Coups, Instability & Political Shocks (Wave 10, 6 posts)

**J1** `coups-and-market-crashes-thailand-turkey-myanmar-anatomy`
> The 24-hour coup playbook: asset reactions (equity circuit breakers, currency freefall, sovereign spread widening), the recovery timeline (weeks for reversals, years for genuine instability). Three case studies: Thailand 2014 (SET -4%, recovered fast), Turkey 2016 attempted coup (TRY -5%, Erdogan consolidates), Myanmar 2021 (total collapse, KYK off market).

**J2** `regime-change-and-asset-repricing-venezuela-nationalization`
> Chavez's nationalization wave 2007-12: oil, steel, telecoms — how sector-by-sector expropriation moved sovereign spreads from 400bp to 2,000bp+; the ExxonMobil/ConocoPhillips battles; why Venezuela went from richest LatAm to hyperinflation — and what it tells investors about political risk in resource-rich EM.

**J3** `political-assassination-and-market-reaction-soleimani-khashoggi`
> Jan 3, 2020 Soleimani killing: oil +4%, S&P futures -1.5% overnight — then full reversal in 48h; Nov 2018 Khashoggi: Saudi equities -7%, TASI selloff, but oil recovered; the pattern — geopolitical assassination is a spike, not a regime change, unless it triggers a war; how to trade the overreaction.

**J4** `failed-referendums-catalonia-scotland-the-political-risk-premium`
> Oct 2017 Catalonia: IBEX -2%, Spanish sovereign spreads +30bp, Catalan banks' stocks -20% in 3 days; the actual resolution (direct rule, referendum voided); Sep 2014 Scotland: GBP most volatile before/after, UK index flat — markets learn not to price tail probability of secession from stable democracies. The signal: credit spreads on sub-national entities.

**J5** `civil-unrest-and-capital-flight-hong-kong-chile-france`
> Three unrest templates: Hong Kong 2019-20 (Hang Seng -25% over 6 months, capital flight \$40B, relocation to Singapore), Chile 2019 (peso -10%, IPSA circuit break), France Gilets Jaunes (CAC -8% in Q4 2018, government concedes). Common signal: VIX local analogs spike 2-3 weeks before the worst volatility.

**J6** `democratic-backsliding-and-sovereign-spread-widening`
> Hungary (Orban's constitutional capture, HUF weakest in EU bloc), Poland (PiS judiciary dismantling, PLN premium), Turkey (Erdogan's erosion of central bank independence → 10-year spread from 400bp→900bp→back). The model: rule-of-law index decline predicts sovereign spread widening with 6-12 month lag.

---

## TRACK K — Sovereign Debt Crises with Political Dimensions (Wave 11, 6 posts)

**K1** `greece-2010-15-when-sovereign-debt-becomes-a-political-crisis`
> The full arc: Greek bond yields 4%→30%, the Troika (ECB/EU/IMF), three bailouts, Syriza's election and the "Grexit" standoff (bank closures, capital controls, ATM queues), Tsipras's climb-down, the haircut. What the Greek crisis says about the limits of democracy inside a monetary union.

**K2** `argentinas-perpetual-default-cycle-the-politics-of-debt`
> 2001, 2014, 2020, 2024: four defaults in 23 years; the politics of every restructuring (holdout funds, the NY court battles, "vulture" funds), the peso-dollar parallel market, the political economy of IMF conditionality; why Argentina keeps defaulting and what the pattern means for bond investors.

**K3** `lebanon-collapse-currency-peg-bank-freeze-political-paralysis`
> The most complete state collapse in a middle-income country: LBP peg to USD for 22 years → September 2019 Eurobond default → October 2019 protests (WhatsApp tax) → bank freeze → 90%+ currency devaluation → Beirut port explosion Aug 2020 → three years of no functioning government. A case study in "political debt": when entrenched elites prevent the restructuring that would save the economy.

**K4** `sri-lanka-2022-imf-protests-and-the-president-who-fled`
> April-July 2022: Sri Lanka runs out of fuel, medicine, and foreign exchange; protesters storm the Presidential Palace; the President flees to Maldives; IMF bailout negotiated; debt restructuring with China (largest creditor). What happened: COVID wiped tourism + organic farming disaster + \$3B Chinese debt overhang → the perfect political-economic storm. Cross-asset: USD bonds from 100→10 cents, then recovery to 45 cents after restructuring.

**K5** `turkeys-unorthodox-rate-policy-when-politics-overrides-central-banks`
> Erdogan's "interest rates cause inflation" belief, the firing of 3 central bank governors 2019-2021, the TRY losing 80% of value 2018-2022, the eventual U-turn (Simsek/Erkan orthodox team 2023), the TRY stabilization — and what the episode says about central bank independence as an asset-price driver.

**K6** `the-imf-playbook-what-countries-must-do-and-market-reactions`
> The IMF toolkit: standby arrangement, extended fund facility, the conditionality menu (fiscal consolidation, rate hikes, FX liberalization, privatization); how markets price an "IMF program under negotiation" (spreads tighten on approval, equities rally on stability signal); the 2022-24 wave (Pakistan, Egypt, Ghana, Sri Lanka) and the political cost of austerity.

---

## TRACK L — Treaties & Agreements: Signed and Broken (Wave 12, 5 posts)

**L1** `jcpoa-the-iran-nuclear-deal-oil-markets-and-geopolitical-chess`
> July 2015 JCPOA signing: Brent -7% in 6 months as Iranian barrels returned; Iranian oil goes from 1mb/d→2.5mb/d; May 2018 Trump withdrawal: Brent +10%, Iranian oil cut again; 2021 Biden re-entry talks (failed); the "nuclear-deal premium/discount" in oil: currently ~\$5/bbl risk premium.

**L2** `paris-agreement-us-in-us-out-us-in-climate-and-energy-markets`
> Jun 2017 Trump withdrawal: coal stocks +5%, solar -3% (market read: US policy reversal); Jan 2021 Biden rejoin: clean energy ETF +40% in 6 months; Jan 2025 Trump re-withdrawal: the market barely flinched (IRA subsidies still law, states still committed); the lesson: US treaty participation ≠ US climate policy because the IRA is not a treaty.

**L3** `nafta-to-usmca-renegotiating-trade-rules-and-the-auto-peso`
> 2017-2019 NAFTA renegotiation: MXN most volatile EM currency on Trump tweets about termination; the auto industry content rules (75% North American content for duty-free), the digital trade chapter; peso settled once USMCA signed (Jul 2020); the lesson: renegotiation threatens more than the outcome.

**L4** `tpp-withdrawal-and-the-asian-trade-architecture-shift`
> Jan 2017 Trump TPP withdrawal: AUD -1%, Japan's Abe scrambles to salvage; CPTPP formed without US (10 countries, 13.5% of global GDP); what the US missed (Asia-Pacific market access, setting labor/environmental standards, counterweighting China in trade rules); China pivots to RCEP.

**L5** `nato-credibility-and-article-5-doubt-as-a-defense-premium`
> The "Article 5 put" in European defense spending: when Trump questioned NATO commitment (2016, 2024), European defense stocks rallied (BAES, Rheinmetall, Thales); the formula: doubt about US commitment → European NATO members forced to self-insure → defense budget increase → defense equity premium. The 2022 Vilnius summit commitment: Germany's 2% GDP goal, the Rheinmetall trade.

---

## TRACK M — Pandemics & Black Swan Political Responses (Wave 13, 5 posts)

**M1** `covid-19-comparing-political-responses-and-market-outcomes`
> The natural experiment: US (fiscal bazooka + reopening pressure) vs. EU (slower fiscal + closer epidemiological control) vs. China (zero-COVID → strict → abrupt end Dec 2022) vs. Sweden (no lockdown). Market outcomes: S&P +28% in 2021, CSI 300 -22% in 2022, STOXX 600 +23% in 2021. The political signal: speed of fiscal response explains 60% of equity recovery variance.

**M2** `sars-2003-vs-covid-2020-political-speed-and-market-v-shapes`
> SARS: MSCI EM -8% in 3 months then full V-shape recovery (China contained it, WTO membership required transparency); COVID: -34% in 5 weeks, then the fastest-ever recovery (Fed/fiscal response); why the V-shape is a *political* artifact (unlimited QE + helicopter money) not a biological one.

**M3** `lockdown-politics-zero-covid-china-vs-open-sweden`
> CSI 300 underperformed MSCI World by 30% in 2022 (zero-COVID lockdowns); the Dec 2022 reversal (zero-COVID abandoned overnight → China reopening trade); Sweden's "no lockdown" model — OMXS30 outperformed Europe in 2020-21; the PM Löfven political fallout vs. the economic outcome.

**M4** `vaccine-nationalism-how-hoarding-supply-moved-em-markets`
> 2021: US/EU hoarded vaccines (COVAX underfunded); India banned exports despite being the world's vaccine factory (Delta wave hit hard); Africa got vaccines 12 months late. Market consequence: EM/DM divergence widened in 2021 as slow vaccination = slower reopening = lower growth; the "vaccine gap trade" in EM sovereign spreads.

**M5** `post-pandemic-political-scars-inflation-supply-chains-fiscal`
> The post-COVID political legacy: inflation as a political crisis (Biden approval -15pp with inflation, UK PM Truss 44-day tenure), the supply-chain onshoring imperative (political cover for industrial policy), the fiscal deficits that won't close (structural deficit 5%+ of GDP in US/UK); long-term asset implication: structurally higher neutral rates.

---

## TRACK N — Cyber Warfare & Tech Nationalism (Wave 14, 5 posts)

**N1** `cyberattacks-as-market-events-solarwinds-colonial-pipeline`
> Dec 2020 SolarWinds: Russia SVR infiltrates 18,000 networks including Treasury/State — cybersecurity stocks +15% in 6 months; May 2021 Colonial Pipeline: Brent +3%, East Coast fuel panic, Colonial paid \$4.4M ransom; the market signal: cyber = a new geopolitical weapon AND a new sector growth driver (CISA spending, CIBERSEGURIDAD ETFs).

**N2** `huawei-blacklist-tiktok-ban-and-tech-nationalism`
> The Huawei entity-list (May 2019): Huawei phone market share collapsed, ARM/Google supply chain terminated; TSMC stopped shipping in Aug 2020 (FDPR enforcement); TikTok forced divestiture debates (2023-24); the pattern — US tech nationalism moves individual stocks ±30% and creates entire new investment theses (US alternatives, Indian competition).

**N3** `chinas-tech-crackdown-2021-the-1-trillion-disappearing-act`
> July 2021: DiDi forced to delist days after NYSE IPO; Alibaba, Meituan, Tencent each lose \$100B+ in weeks; the "common prosperity" regulatory campaign; the CSAR (Chinese company SEC audit requirement); Nasdaq Golden Dragon Index -65% peak to trough; the "uninvestable China" narrative and whether it was right.

**N4** `digital-yuan-cbdc-as-geopolitical-strategy`
> The e-CNY pilot (2020→2024, 200M+ wallets): not just a payment system but a dollar-circumvention tool, an economic-activity surveillance tool, and a capital-control precision instrument; how 120+ countries' CBDC programs relate to the dollar's future; mBridge (BIS+HK+UAE+Saudi+China) as the plumbing for a new settlement system.

**N5** `internet-shutdowns-capital-controls-and-em-market-access-risk`
> 2023: Iran, Russia, Myanmar, Ethiopia each deploy internet shutdowns; the correlation with capital flight (USD buying spikes in unofficial markets during shutdowns); Russia's RuNet isolation experiment; the investor implication: when the government can cut market access, what's the "political VPN premium" in EM liquidity?

---

## TRACK O — Resources, Climate & Great-Power Competition (Wave 15, 4 posts)

**O1** `rare-earths-wars-china-controls-85-percent-and-the-supply-chain-bet`
> China controls 85% of rare-earth processing; the 2010 Japan embargo (Senkaku); 2023 germanium/gallium; 2024 graphite; the US IDA (Defense Production Act mining push) and MP Materials/Lynas/Energy Fuels thesis; the timeline to non-China supply and the "strategic stockpile premium."

**O2** `water-scarcity-as-a-geopolitical-risk-factor`
> The Nile River dam dispute (Ethiopia GERD vs. Egypt/Sudan — Egypt threatened military action); the Colorado River water rights political crisis in the US; Cape Town Day Zero 2018; how water stress moves agriculture commodity futures, EM sovereign risk in arid regions, and the "water infrastructure" investment theme.

**O3** `carbon-border-adjustment-the-new-trade-weapon`
> The EU CBAM (2023-2026): a carbon tariff on steel, aluminum, cement, fertilizers, power from non-carbon-priced countries; WTO legality question; Chinese/Indian protests (it's protectionism in green clothing); the market: European steel +5% on CBAM announcement, EM steel exporters -8%; the carbon price convergence trade.

**O4** `arctic-sovereignty-russia-canada-and-the-melting-resource-race`
> The Arctic paradox: climate change that markets ignore is opening the world's largest untapped resource frontier; Russia's Arctic military build-up (Murmansk nuclear icebreakers, new bases); the Northwest Passage as a shipping route (30% shorter than Suez for some routes); Arctic rare earths, gas, and the sovereignty disputes — the long-duration geopolitical trade.

---

## TRACK I — Capstone (Wave 15, 3 posts)

**I1** `building-a-geopolitical-risk-monitor-data-indicators-signals`
> The practitioner toolkit: GPR index (Caldara-Iacoviello), VIX geopolitical spike detection, commodity-FX triangulation (oil/gold/USD ratio), EMBI spread divergence, satellite data (ship tracking, military movements), social media sentiment as an early-warning system. How to build a real geopolitical dashboard.

**I2** `geopolitical-trading-playbook-before-during-after-the-shock`
> The three-phase playbook: (1) Pre-positioning (signals that an event is approaching: satellite imagery, intelligence-community language, diplomat recalls); (2) During the shock (size small, fade the first move in spikes, add on the re-test); (3) After (distinguish the tail vs. the regime change — hold or exit). Position sizing for tail risk.

**I3** `geopolitical-crises-and-markets-the-complete-playbook-capstone`
> The capstone: the master mental model, the full asset-sensitivity matrix with every crisis type, the political-move decoder table (sanction = X, tariff = Y, coup = Z in markets), cross-links to all 82 posts in the series, the geopolitical risk framework an investor can use tomorrow.

---
**TOTAL: 89 posts**
Tracks A (8) + B (10) + C (8) + D (7) + E (6) + F (6) + G (5) + H (5) + I (3) + J (6) + K (6) + L (5) + M (5) + N (5) + O (4) = **89**
*(A6 + A7 added 2026-06-21: "how to read political signals before a country acts" + "forecasting geopolitical events"; A8 added 2026-06-22: "escalation ladders + next-move anticipation")*
