# FX / Currency Trading — "Trading Currencies, From Carry to Crisis"

Series of **42 deep-dive posts** in a NEW folder **`content/blog/trading/forex/`** (subcategory `Forex`).
finance-writer voice, English, ≥ 8,500 words, 7 figures each, `.png` embeds + `optimize-blog-images`.
Audience: smart curious reader, **no finance background**. Global spine + a dedicated **Vietnam / EM-FX track**.
Kit: `.cache/finance-writer/_forex/_kit.md` (reuse the `_macro` chartkit + a cited `data_forex.py`). Roadmap: this file.
Commit + push **each wave** (explicit paths, never `git add -A`); scope to `trading/forex/` slugs + their webp.

## Positioning (do NOT duplicate the sibling series — cross-link instead)
This series is **the FX discipline itself**, built from zero to practitioner depth, around one running thesis:
an exchange rate is **the relative price of two monies** — you can never own "a currency" in isolation, so
**every position is a pair, a spread, a relative bet**. The master variables are **rate differentials and flows**.
- **fixed-income** owns rates / duration / the curve → link out (FX rides rate differentials; don't re-teach rates).
- **macro-trading** owns monetary policy & transmission → link out, don't re-derive the policy mechanism.
- **cross-asset** owns allocation / correlation / the dollar as one asset → link out.
- **options-volatility** owns vanilla option Greeks & the vol surface → link out (the FX smile is a track here, but cite the Greeks).
- **game-theory** owns the formal speculative-attack / coordination models → link out for the model, keep the narrative.
- **gold** touches the dollar → link out where the dollar overlaps.

## The relative-price spine (woven through every track, owned by Tracks B & H)
Every post should connect its mechanics back to the spine: *a price is always two monies, and what moves it is the
gap between their rates plus the flow of money across borders.* Tracks B (what moves a currency) and H (the playbook)
make the spine explicit; the dollar (Track D) is the one money that is on the other side of everyone else's trade.

---

## Track A — Foundations: What an Exchange Rate Really Is (6) — Wave 1
A1 `why-currencies-are-different-fx-trading-introduction` — Series intro + the relative-price thesis: you never own a currency, only a pair; FX is the price of one money in another.
A2 `base-quote-pips-and-how-to-read-an-fx-quote` — The quote line by line: base vs quote currency, bid/ask, pips and pipettes, lots, why EUR/USD up means the dollar is weaker.
A3 `the-biggest-market-on-earth-inside-the-interbank-fx-market` — The $7.5T/day market: dealers, the interbank ladder, prime brokerage, ECNs, why FX is OTC and (mostly) 24/5.
A4 `spot-forward-and-swap-the-three-ways-to-trade-a-currency` — Spot (T+2) vs outright forwards vs FX swaps; what each is for and how the forward is just spot plus a rate gap.
A5 `who-trades-fx-corporates-central-banks-funds-and-retail` — The players and their motives: hedgers, reserve managers, macro funds, CTAs, and the retail tail — and who is on the other side.
A6 `majors-minors-and-exotics-the-map-of-currency-pairs` — The pair taxonomy: majors, crosses, EM/exotics; liquidity tiers, spreads, sessions (Tokyo/London/New York), and how an FX trade actually settles.

## Track B — What Moves a Currency (6) — Wave 2
B1 `interest-rate-differentials-the-master-variable-of-fx` — Why the rate gap between two countries is the gravitational center of an exchange rate; covered interest parity from first principles.
B2 `uncovered-interest-parity-and-why-it-fails-the-forward-puzzle` — UIP, the forward as a (bad) forecast, the forward-premium puzzle, and why the failure of UIP is the carry trade's whole reason to exist.
B3 `purchasing-power-parity-and-the-real-exchange-rate` — PPP, the Big Mac index, the law of one price, the real vs nominal rate, and why PPP anchors decades but not days.
B4 `the-balance-of-payments-and-the-current-account` — How a country pays its way: current account, capital account, and why a persistent deficit pulls a currency down (or doesn't).
B5 `terms-of-trade-and-the-commodity-currencies` — AUD, CAD, NOK, BRL: how export prices drive a currency; the terms-of-trade shock and the petro-currency link.
B6 `capital-flows-and-the-dornbusch-overshoot` — Why money chasing yield moves currencies faster than trade does; sticky prices, the overshooting model, and why FX overreacts.

## Track C — The Carry Trade (5) — Wave 3
C1 `the-carry-trade-getting-paid-to-hold-a-currency` — Carry from zero: borrow low, lend high, pocket the rate gap; the worked dollar P&L and why it is a bet that UIP fails.
C2 `carry-and-volatility-the-relationship-that-runs-the-trade` — Why carry is short volatility; the carry-to-vol ratio, the Sharpe of carry, and why low vol invites leverage.
C3 `funding-currencies-vs-high-yielders-jpy-chf-and-the-rest` — The two sides of carry: JPY/CHF as funding legs, the high-yield EM and commodity longs, and how the whole basket is built.
C4 `carry-crashes-picking-up-pennies-in-front-of-a-steamroller` — The fat left tail: why carry pays steadily then craters; 1998, 2008, and the August-2024 yen unwind.
C5 `fx-as-a-factor-zoo-carry-value-momentum-and-dollar` — FX styles as systematic factors: carry, value (PPP reversion), momentum/trend, and the dollar factor; how funds harvest them.

## Track D — The Dollar System (5) — Wave 4
D1 `the-dxy-and-the-dollars-special-role-in-the-world` — What the dollar index is and isn't; the dollar as the unit of account, invoicing currency, and the other side of ~88% of all trades.
D2 `the-dollar-smile-why-the-dollar-wins-in-boom-and-in-panic` — The dollar-smile framework: USD strong in US outperformance and in global risk-off, weak in the calm middle.
D3 `eurodollars-and-the-offshore-dollar-system` — Dollars created outside the US: the eurodollar market, offshore credit, and why the Fed is the world's central bank whether it likes it or not.
D4 `the-global-dollar-shortage-and-central-bank-swap-lines` — Why the world is structurally short dollars; the funding squeeze in a crisis and how Fed swap lines put out the fire.
D5 `the-dollar-as-a-wrecking-ball-for-emerging-markets` — The strong-dollar doom loop: dollar debt, tightening financial conditions, and why a rising DXY is an EM stress gauge.

## Track E — Central Banks & Intervention (6) — Wave 5
E1 `how-central-banks-intervene-in-the-currency-market` — The intervention toolkit: spot buying/selling, sterilized vs unsterilized, verbal intervention, and why size and surprise matter.
E2 `pegs-bands-and-crawling-pegs-the-spectrum-of-fx-regimes` — From free float to hard peg: bands, crawls, currency boards, dollarization; the trade-offs of giving up your exchange rate.
E3 `the-impossible-trinity-pick-two-of-three` — The trilemma: fixed FX, free capital flows, independent monetary policy — you get two; the framework that explains every FX regime choice.
E4 `currency-crises-and-the-anatomy-of-a-speculative-attack` — How a peg breaks: reserves, the one-way bet, self-fulfilling runs; the narrative of the attack (the formal model lives in game-theory).
E5 `soros-and-black-wednesday-breaking-the-bank-of-england-1992` — The canonical attack: the ERM, the pound's misalignment, the short, and the day the Bank of England lost.
E6 `the-snb-2015-peg-break-when-a-central-bank-blinks` — The franc floor, the surprise abandonment, the 30% gap, and the brokers it bankrupted; the lesson about pegs you cannot defend forever.

## Track F — Plumbing & Instruments (6) — Wave 6
F1 `fx-forwards-and-swaps-the-real-workhorses-of-the-market` — Pricing the forward points from the rate gap; the FX swap as a collateralized loan in two currencies; why most "FX" volume is swaps, not spot.
F2 `non-deliverable-forwards-trading-uninvestable-currencies` — NDFs: how the offshore market prices currencies you cannot freely deliver (CNY, KRW, INR, VND); fixing, settlement in USD, and the onshore/offshore gap.
F3 `fx-options-and-the-volatility-smile` — FX vanilla options, the at-the-money straddle, and why the FX market quotes vol not price (the Greeks live in options-volatility; cite them).
F4 `risk-reversals-and-the-shape-of-fear-in-fx` — The 25-delta risk reversal as the market's directional fear gauge; butterflies, the smile's wings, and how skew encodes crash risk.
F5 `the-cross-currency-basis-when-covered-parity-breaks` — Why the rate gap and the forward stop lining up; the cross-currency basis as a dollar-funding stress gauge and a real arbitrage that won't close.
F6 `settlement-risk-and-cls-how-fx-actually-clears` — Herstatt risk, the day-the-other-side-fails problem, and how CLS payment-versus-payment removed the biggest hidden risk in FX.

## Track G — Vietnam / EM-FX Track (5) — Wave 7
G1 `usd-vnd-and-the-managed-float-how-the-sbv-runs-the-dong` — The dong's regime: the central rate, the +/- band, the reference rate, and how the State Bank steers USD/VND between policy and the market.
G2 `the-offshore-ndf-market-for-the-dong-and-asian-currencies` — Where the dong and its neighbors trade beyond the band; NDF pricing, the onshore/offshore spread, and what it signals about devaluation pressure.
G3 `fx-reserves-and-intervention-in-an-emerging-market` — Why reserves are an EM's war chest; months-of-imports, the reserve-adequacy yardstick, and how a managed-float central bank actually defends a level.
G4 `remittances-fdi-and-the-current-account-that-funds-the-dong` — The flows that hold an EM currency up: remittances, FDI, the trade surplus; Vietnam's balance of payments as a worked example.
G5 `the-carry-into-emerging-markets-and-the-sudden-stop` — Hot money chasing EM yield, the carry-into-EM trade, and the sudden stop: how a flow reversal becomes a currency crisis (1997, 2013 taper tantrum).

## Track H — Crises & the Playbook (5) — Wave 8
H1 `the-erm-crisis-of-1992-when-europes-currencies-broke` — The full 1992 story: the ERM strains, the lira and the pound, and what a misaligned peg does to a region.
H2 `the-1997-asian-crisis-thb-idr-and-krw-in-freefall` — The Asian crisis: the dollar pegs, the dollar debt, the contagion from Thailand outward, and the IMF aftermath.
H3 `1998-russia-ltcm-and-the-fx-tremor-that-shook-the-world` — The ruble default, the carry unwind, LTCM's leverage, and how an FX shock became a global deleveraging.
H4 `2015-to-2022-the-franc-the-yen-and-the-em-stress-cycle` — The modern stress map: the 2015 CHF shock, the 2022 yen collapse and the BoJ's defense, the gilt/sterling crisis, and EM under a strong dollar.
H5 `building-an-fx-view-the-currency-trading-playbook-capstone` — The capstone: a repeatable process for forming an FX view from rate gaps, flows, positioning, and vol — and the playbook that links the whole series.

---

## Cross-links the kit will hand to agents (real, existing posts)
- Rates / curve lens (don't re-teach rates): `/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be`,
  `/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance`,
  `/blog/trading/fixed-income/emerging-market-and-sovereign-debt-yield-with-country-risk`,
  `/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates`
- Macro / policy lens (don't re-derive policy): `/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials`,
  `/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry`,
  `/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile`,
  `/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy`,
  `/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks`,
  `/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable`,
  `/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance`
- Allocation / correlation lens: `/blog/trading/cross-asset/fx-currencies-the-relative-value-layer`,
  `/blog/trading/cross-asset/the-dollar-cross-asset-gravity`,
  `/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis`,
  `/blog/trading/cross-asset/correlation-by-regime-growth-and-inflation`
- Options / vol lens (cite the Greeks, don't re-derive): `/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more`,
  `/blog/trading/options-volatility/trading-skew-risk-reversals-collars-and-the-shape-of-fear`,
  `/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear`,
  `/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol`
- Game-theory lens (link for the formal model, keep the narrative): `/blog/trading/game-theory/bank-runs-as-coordination-games-diamond-dybvig-and-svb`,
  `/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed`,
  `/blog/trading/game-theory/crowded-trades-and-the-exit-game`,
  `/blog/trading/game-theory/case-study-vietnam-foreign-flows-room-limits-and-manipulation-games`
- Gold / dollar overlap: `/blog/trading/gold/gold-and-the-dollar-the-inverse-relationship-and-when-it-breaks`,
  `/blog/trading/gold/sbv-gold-policy-auctions-the-2024-intervention-and-anti-goldization`,
  `/blog/trading/gold/smuggling-the-gray-market-and-the-gold-dollar-fx-triangle-in-vietnam`
- Finance / case lens: `/blog/trading/finance/soros-bank-of-england-1992-black-wednesday`,
  `/blog/trading/finance/asian-financial-crisis-1997`,
  `/blog/trading/finance/ltcm-1998-when-genius-failed`,
  `/blog/trading/finance/petrodollar-and-dollar-dominance`,
  `/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling`,
  `/blog/trading/finance/swift-and-the-weaponization-of-payments`
- Vietnam-sector lens: `/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam`,
  `/blog/trading/vietnam-stocks/the-four-macro-dials-rates-credit-fx-commodities`
- Within series: link sibling posts by `/blog/trading/forex/<slug>`.

## Wave / status tracker
- [x] Wave 1 — Track A (6) — Foundations
- [x] Wave 2 — Track B (6) — What moves a currency
- [x] Wave 3 — Track C (5) — The carry trade
- [ ] Wave 4 — Track D (5) — The dollar system
- [ ] Wave 5 — Track E (6) — Central banks & intervention
- [ ] Wave 6 — Track F (6) — Plumbing & instruments
- [ ] Wave 7 — Track G (5) — Vietnam / EM-FX
- [ ] Wave 8 — Track H (5) — Crises & the playbook
