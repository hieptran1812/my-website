---
title: "The Currency Channel: Rate Differentials and Carry"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How a gap between two central banks' interest rates moves the exchange rate, fuels the carry trade, and reprices imports, exporter earnings, and foreign-currency debt across borders."
tags: ["currency-channel", "carry-trade", "rate-differentials", "interest-rate-parity", "exchange-rate", "monetary-policy", "imported-inflation", "dollar", "emerging-markets", "asset-valuation"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When one central bank sets a higher interest rate than another, the gap moves the exchange rate, and that move reprices imports, exporter earnings, and foreign-currency debt on real balance sheets around the world.
>
> - The higher-rate currency tends to be **bid up today** (everyone wants the yield) but is **expected to weaken later** — the two facts coexist through interest-rate parity, and the difference between them is the carry trade's reward and its risk.
> - **Carry** means borrowing a low-yield currency to fund a high-yield one and pocketing the rate gap. It pays a little every day for years, then can lose a year's carry in a single day when it unwinds.
> - A currency move is not abstract: a weaker home currency imports inflation, lifts an exporter's earnings, lightens an importer's bill, and makes dollar debt heavier — all at once, in opposite directions, on different balance sheets.
> - **The one number to remember:** on 5 August 2024 a tiny Bank of Japan hike plus one soft US jobs print snapped the yen-carry trade — the yen ripped, the Nikkei fell ~12% in a day, and risk assets convulsed worldwide. A rate differential moved every asset.

On the morning of 5 August 2024, the Tokyo stock market had its worst single day since 1987. The Nikkei 225 fell about 12% — a one-day crash bigger than anything in the 2008 crisis or the 2020 COVID panic. By the time the sun reached New York, the VIX (Wall Street's "fear gauge") had spiked above 65, a level seen only in 2008 and March 2020, and bitcoin, Nvidia, the Mexican peso, and a dozen other unrelated-looking assets were all falling together. Nothing about the economy had changed overnight. No bank had failed. No war had started.

What had happened was almost laughably small on paper. A week earlier the Bank of Japan (BOJ) had raised its policy interest rate by 0.15 percentage points — from roughly 0.10% to 0.25%. A rounding error by the standards of the US Federal Reserve, which had moved in 0.75-point steps in 2022. Then, on 2 August, the US jobs report came in soft, which made traders think the Fed would *cut* sooner. So one central bank nudged its rate up a hair while another looked like it would bring its rate down. The *gap* between Japanese and US interest rates — which had been enormous, and had been the engine of one of the largest trades on Earth — started to close.

That tiny narrowing was enough. It set off a stampede out of the **yen-carry trade**, and the unwind of that one trade reached across every market on the planet in 48 hours. This post is about why. It is the story of the **currency channel** — the pathway by which a central bank's interest-rate decision travels through the exchange rate and lands as a concrete change in what assets are worth, in country after country, often far from where the policy was set.

![Rate differential travels through the currency channel to reprice imports exporter earnings and dollar debt](/imgs/blogs/the-currency-channel-rate-differentials-and-carry-1.png)

The mental model above runs through the whole post. A central bank pulls a lever — it sets an interest rate. When that rate sits *above* another country's rate, a **rate differential** opens up, and that differential is a price. It travels through the **currency channel**: capital chases the higher yield, bids the high-rate currency up in the spot market, piles into the carry trade — and then, when the gap narrows, the whole thing unwinds violently. And it lands as a **change in what assets are worth**: imported inflation reprices bonds, the exporter-importer split reprices equities, and a currency move rewrites the burden of foreign-currency debt. One note before we start: this post is the *channel* — how the rate gap moves the currency and how the currency reprices assets. The mechanics of FX as a *policy lever* in its own right — pegs, bands, reserves, the impossible trinity, direct intervention — live in the sibling post [Currency Policy and FX Intervention](/blog/trading/policy-and-markets/currency-policy-and-fx-intervention). I will lean on it rather than repeat it.

## Foundations: how an interest rate moves an exchange rate

Start from zero, because the whole chain rests on one idea and that idea is genuinely counterintuitive.

An **exchange rate** is just a price: the price of one currency in another. USD/JPY = 150 means one US dollar buys 150 Japanese yen. A bigger USD/JPY number means a *stronger* dollar and a *weaker* yen (it takes more yen to buy a dollar). Keep that straight and half the confusion evaporates.

An **interest rate**, for our purposes, is what you earn for holding money in a currency. Park cash in US dollars when the Fed's rate is 5% and, very roughly, you earn about 5% a year. Park it in Japanese yen when the BOJ's rate is 0.25% and you earn about 0.25%. So if you are simply choosing where to keep money, the dollar pays you twenty times more. Money is mobile and money is greedy, so the natural first guess is: *capital should flood into the high-rate currency, and that demand should push it up.* That first guess is half right, and the other half is where all the interesting physics lives.

### The naive story: money chases yield

Here is the half that is right. When a central bank raises its rate above the rest of the world's, foreign capital does flow in to capture the yield, and that flow does tend to push the currency up *in the spot market* — the price for delivery today. This is the most reliable short-run relationship in macro: **rate up → currency up.** It is why, through 2022–2023, as the Fed hiked the fastest in 40 years while the Bank of Japan held its rate near zero, the dollar climbed and the yen sank to multi-decade lows. The gap between the two rates — call it the **rate differential** — is the fuel.

![US-Japan policy rate gap widens then narrows as USD/JPY climbs then falls](/imgs/blogs/the-currency-channel-rate-differentials-and-carry-4.png)

Look at the figure. The amber bars are the gap between the Fed's policy rate and the BOJ's: it blew out from a fraction of a point in 2021 to about 5.6 points by the end of 2023, as the Fed went to a 5.25–5.50% target range while the BOJ stayed pinned near zero. The red line is USD/JPY, the price of a dollar in yen. It tracked the gap almost exactly — climbing from about 115 to 157 as the differential widened, then turning back down in 2024–2025 as the gap began to close (the BOJ finally hiked; the Fed began cutting). The relationship is not perfect and it is not instant, but the direction is unmistakable: **a wider rate gap, a stronger high-rate currency.** For the statistical version of this — the betas, the rolling correlations, how tight the link actually is — see [The Dollar (DXY) Cross-Asset Correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation). Here I care about the *mechanism* and what it does to value.

### The other half: interest-rate parity

Now the counterintuitive part, and it is the single most important idea in all of currency economics. If the high-rate currency simply rose and stayed risen, the trade would be free money: borrow the cheap currency, hold the expensive one, collect the rate gap *and* the currency gain, forever, with no offsetting cost. Free money does not survive in markets. So there must be a catch, and the catch is built into the structure of how forward exchange rates are priced.

The catch is called **interest-rate parity**, and it comes in two forms.

**Covered interest-rate parity (CIP)** is the airtight, no-arbitrage one. A *forward* exchange rate is the price you lock in *today* for exchanging currencies at a *future* date. CIP says the forward rate must be set so that you cannot make a riskless profit by combining a spot trade, a forward trade, and the two interest rates. The consequence: **the higher-yielding currency must trade at a forward discount.** In plain words, the market quotes a future exchange rate at which the high-rate currency is *worth less*. The forward market literally prices the high-rate currency to weaken, by almost exactly the rate differential. If it did not, a bank could borrow the low-rate currency, convert at spot, invest at the high rate, and lock the conversion back with a forward — pocketing a guaranteed profit. Banks do exactly this arbitrage all day, which is what keeps CIP holding to the basis point.

**Uncovered interest-rate parity (UIP)** is the economic theory, and it is *not* airtight — it is a tendency, not a law. UIP says that, on average, the high-rate currency should *actually* weaken over time by enough to wipe out the interest advantage, so that an investor is indifferent between currencies. The intuition: if a currency paid more *and* never depreciated, everyone would pile in until it could not pay more for free.

Here is the crucial real-world fact: **UIP fails, persistently, in the short and medium run.** High-rate currencies do not depreciate as fast as UIP predicts; they often *appreciate* for years. That persistent failure is not a footnote — it *is* the carry trade. The carry trade is, precisely, a bet that UIP keeps failing: that you can collect the rate gap and the high-rate currency will *not* fall enough to offset it. It works until, suddenly and violently, UIP reasserts itself all at once. That sudden reassertion is the unwind.

So hold both halves at the same time, because they only look contradictory: the high-rate currency is **bid up spot** (the naive story, real and observable) *and* it is **priced to weaken forward** (parity, equally real). The gap between those two prices is the carry — your reward for taking the risk that the forward price is right and the spot strength is temporary.

### Why does UIP fail? The "forward premium puzzle"

It is worth pausing on *why* uncovered interest parity fails so persistently, because the answer is the foundation of the carry trade and one of the most famous unsolved puzzles in finance — economists literally call it the **forward premium puzzle** or the **UIP puzzle**. The theory predicts high-rate currencies should depreciate; the data show they tend to *appreciate*, or at least not depreciate nearly enough, for long stretches. Researchers have run this test across decades and dozens of currencies, and the result is robust: betting *against* UIP — that is, holding the high-rate currency — has been profitable on average. The free lunch the theory says cannot exist appears to exist most of the time.

There is no fully-agreed explanation, but the leading one is a *risk premium*. The carry trade is not actually free money; it is compensation for bearing **crash risk**. Investors demand the extra return precisely because, every so often, the trade detonates and they lose a fortune in days. In that view, the average carry profit is the *insurance premium* the high-rate currency pays to lenders for holding a currency that occasionally collapses. The carry "works" on average for the same reason selling insurance works on average — you collect a steady premium for taking on a rare, catastrophic risk. The puzzle is only a puzzle if you ignore the tail; once you account for the rare, devastating unwinds, the "excess" return starts to look like fair pay for a nasty risk. Either way, the practical lesson is the same: **the carry trade's reliability is an illusion created by the rarity of the disaster, not its absence.** Keep that sentence in mind every time a strategy shows a beautiful, smooth, upward-sloping track record built on a rate differential.

#### Worked example: the covered interest parity forward rate

Let me make CIP concrete with numbers, because the formula sounds abstract until you see it move money.

Take USD and JPY. Suppose the spot rate is USD/JPY = 150 (one dollar buys 150 yen). The US one-year interest rate is 5.0%; the Japanese one-year rate is 0.5%. What is the one-year *forward* rate that CIP forces?

The no-arbitrage condition is: starting with one dollar, you can either (a) invest it in the US for a year, or (b) convert to yen, invest in Japan, and convert back at the forward rate. Both routes must end with the same number of dollars, or there is free money.

- **Route (a):** \$1 grows to \$1 × (1 + 0.05) = **\$1.05**.
- **Route (b):** \$1 converts to ¥150, grows to ¥150 × (1 + 0.005) = ¥150.75, then converts back at the forward rate `F` (yen per dollar): you get ¥150.75 ÷ `F` dollars.

Set them equal: 150.75 ÷ `F` = 1.05, so `F` = 150.75 ÷ 1.05 ≈ **¥143.6 per dollar**.

The forward rate is ¥143.6, *below* the spot of ¥150. A lower USD/JPY means a *weaker dollar* and a *stronger yen*. So CIP forces the forward market to price the high-rate dollar to *fall* about 4.3% over the year — almost exactly the 4.5-point rate gap (5.0% − 0.5%). **The forward market gives back, on paper, exactly the interest advantage the dollar offers — which is why locking in the carry with a forward earns you nothing, and why the carry trade only pays if you stay *unhedged* and bet the spot doesn't fall that far.**

## How the carry trade actually works

We have arrived at the engine. The **carry trade** is the most important application of the currency channel, the source of the 2024 crash, and the cleanest illustration of how a rate differential reprices the whole world.

The recipe is three steps. **Borrow** in a low-interest-rate currency (the *funding currency* — historically the yen and the Swiss franc, more recently sometimes the euro). **Convert** the borrowed money into a high-interest-rate currency or asset (the *target* — the US dollar, or higher still, the Brazilian real, the Mexican peso, the Australian dollar). **Invest** at the high rate. You pay the low rate on what you borrowed and earn the high rate on what you bought; the difference is your **carry**, and you keep it as long as the exchange rate doesn't move against you.

![Carry trade flow from borrowing yen to investing dollars with the unwind below](/imgs/blogs/the-currency-channel-rate-differentials-and-carry-2.png)

The figure traces it. Borrow yen at 0.5% (red — this leg costs you). Sell the yen and buy dollars at the spot rate — and notice, that is where all the risk lives, because you now have a yen liability and a dollar asset, so a stronger yen hurts you. Invest the dollars at 5.0% (green — this leg pays you). The net is a +4.5% annual carry while markets are calm. And underneath, in the danger zone, is the unwind: when the rate gap narrows or risk spikes, everyone holding this trade tries to buy back yen to repay their loans *at the same time*, the yen rips higher, and a single FX move can erase more than two years of carry in days.

Why is the unwind always so violent? Three reasons, and they compound. First, the trade is **crowded** — it is so simple and so profitable for so long that everyone from hedge funds to Japanese retail investors (the famous "Mrs. Watanabe") to corporate treasuries piles into the same side. Second, it is **levered** — because the carry on an unlevered position is small (a few percent), traders borrow heavily to amplify it, so a small adverse FX move triggers margin calls. Third, it is **one-sided and self-reinforcing** — when the yen starts rising, the carry traders' losses force them to buy yen to cut risk, which pushes the yen *higher*, which forces *more* of them to buy, a doom loop in fast-forward. The trade that paid a smooth trickle for years pays its bill in one terrifying lump.

#### Worked example: carry-trade P&L — the 4.5% carry versus the FX risk

Now the numbers that make the 2024 crash inevitable in hindsight. Suppose you run a \$10 million yen-carry trade. You borrow ¥1.5 billion (worth \$10 million at USD/JPY = 150) at 0.5% and invest the \$10 million in US assets at 5.0%.

**The carry, per year (if the exchange rate never moves):**
- You earn on the dollar leg: \$10,000,000 × 5.0% = **+\$500,000**.
- You pay on the yen leg: ¥1,500,000,000 × 0.5% = ¥7,500,000, which at 150 is **−\$50,000**.
- Net carry = \$500,000 − \$50,000 = **+\$450,000 per year**, a clean +4.5% on your \$10 million. Smooth, boring, lovely.

**Now the yen strengthens 10% in the unwind** — USD/JPY falls from 150 to about 136 (a stronger yen). Your problem: you owe ¥1.5 billion, and yen just got more expensive in dollars.
- To repay ¥1,500,000,000 you now need \$1,500,000,000 ÷ 136 ≈ **\$11,029,000**.
- But your dollar assets are worth about \$10 million (plus a bit of the year's carry). So the FX move alone costs you ≈ \$11,029,000 − \$10,000,000 = **−\$1,029,000**.
- A 10% yen move wiped out **about 2.3 years of the +\$450,000 carry in days** — and in August 2024 the yen moved roughly that much in about three weeks.

**A carry trade collects pennies in front of a steamroller: years of small, steady gains, then one move that takes it all back — and with leverage, far more than all of it.** That is why the August 2024 unwind crashed markets that had nothing to do with Japan: forced sellers raise cash by selling whatever is liquid, so the yen ripping reprices Nvidia, bitcoin, and the peso in the same hour.

### The carry trade's return profile: why it looks safe until it isn't

There is a deeper statistical reason the carry trade is so seductive and so dangerous, and it is worth naming because it explains why smart, well-paid people keep getting blown up by it. The carry trade's return distribution is *negatively skewed* with *fat tails* — in plain terms, it produces many small positive returns and a few enormous negative ones. Plotted over time, a carry strategy's equity curve climbs in a smooth, gentle, almost boringly reliable line for years, which makes it look like a low-risk, high-Sharpe-ratio strategy by every standard backward-looking measure. Then it falls off a cliff.

Economists describe this as "picking up pennies in front of a steamroller" or, more formally, as the carry trade being *short volatility* — it is, in effect, selling insurance against a currency crash. Most of the time no crash comes, and you pocket the premium (the carry). Occasionally the crash comes, and you pay out far more than every premium you ever collected. This is the same payoff shape as selling earthquake insurance: steady income, then ruin. Any risk model that looks only at the smooth historical returns will *understate* the true risk by an order of magnitude, because the catastrophe is rare enough to be absent from most samples. That is precisely why so many funds, over so many decades, have re-discovered the carry trade, declared it a free lunch, levered it up, and then been carried out on a stretcher when the steamroller finally arrived.

### The funding currencies and the cross-currency basis

Two technical wrinkles deepen the picture. First, *which* currency becomes the funding currency is not random — it is whichever credible, liquid currency has the lowest interest rate. For two decades that was overwhelmingly the **Japanese yen** (the BOJ held rates near zero from the late 1990s), with the **Swiss franc** as the other classic funder. When the European Central Bank pushed rates negative in the 2010s, the euro joined the funding club for a while. The funding currency is the one the carry trade is *short*, so it is the one that rips higher in an unwind — which is exactly why a *strengthening yen* is now read by traders as a global risk-off signal regardless of anything happening in Japan.

Second, the perfectly-arbitraged world of covered interest parity occasionally *breaks*, and the size of the break is itself a watched indicator. The **cross-currency basis** is the measure of how far CIP is violated — how much *extra* it costs to borrow dollars synthetically (via the FX swap market) versus directly. In a calm world the basis is near zero, because banks arbitrage any gap away. But when dollar funding gets scarce — at quarter-ends when banks shrink their balance sheets for regulatory reporting, or in a crisis like 2008 and March 2020 — the basis blows out, sometimes dramatically, signaling that the world is short of dollars and willing to pay a premium for them. A widening cross-currency basis is a flashing warning light on the dashboard of every central bank: it means the dollar-funding plumbing is clogging, which is the early stage of the dollar-shortage doom loop that breaks emerging markets. So even the *failure* of interest-rate parity is information — it tells you the currency channel is under stress before the prices in it have fully moved.

## The currency channel reprices everything: the four transmission paths

A carry unwind is the dramatic case, but the currency channel is working *all the time*, quietly, through four paths. Each one turns an exchange-rate move into a change in what an asset is worth.

![One FX move reprices an importer exporter multinational and EM borrower in opposite directions](/imgs/blogs/the-currency-channel-rate-differentials-and-carry-5.png)

The figure makes the central, easily-missed point: **one currency move reprices everyone, but in opposite directions.** A strong dollar that delights the US consumer (cheaper imports) is the *same move* that crushes the emerging-market borrower (heavier dollar debt) and squeezes the US multinational (foreign earnings worth fewer dollars) while gifting the foreign exporter (its goods got cheaper abroad). The exchange rate is not one bet; it is the same bet wearing four jerseys. Let me take the four paths one at a time.

### Path 1 — Imported inflation: the currency reprices bonds

When a country's currency weakens, everything it *imports* gets more expensive in local-currency terms — oil priced in dollars, electronics, food, machinery. That feeds straight into consumer prices. A weaker currency is, mechanically, an *inflationary* force; a stronger one is *disinflationary*. This is the currency channel's most direct hit on **bonds**, because bond prices live and die by inflation: higher inflation means the central bank holds rates higher for longer, and the fixed coupons a bond pays are worth less in real terms — so the bond reprices *down* (its yield rises). The chain is: weaker FX → imported inflation → higher-for-longer policy rate → bonds fall. This is exactly why emerging-market central banks defend their currencies so fiercely — a falling currency is an inflation problem they cannot ignore. For how the rate-inflation link reprices a bond from first principles, see the companion [Discount-Rate Channel](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows).

#### Worked example: a 20% currency fall and imported inflation

Suppose a small open economy imports 30% of what it consumes, and its currency falls 20% against the dollar (the currency it pays for imports in). Hold everything else constant.

- Imported goods now cost 20% more in local currency. If imports are 30% of the consumption basket, the *direct* first-round hit to the price level is 0.30 × 20% = **+6 percentage points** of inflation, before any second-round effects (wages chasing prices, domestic firms raising prices under cover).
- If inflation was running at 3%, this shock alone could push it toward **9%**, well above any sane central bank's target.
- The central bank's likely response: hike rates to defend the currency and cap inflation. Say it raises its policy rate by 3 points. A 10-year government bond yielding 5% might see its yield jump to 8% — and a 10-year bond loses roughly its *duration* (≈8 years) times the yield change in price, so ≈ 8 × 3% = **−24% on the bond's price**.

**A currency that falls 20% does not just make your imported coffee dearer — it can knock a quarter off the value of your country's government bonds through the inflation it imports and the rate hikes it forces.** The currency channel and the discount-rate channel are the same hose pointed at the bond.

### Path 2 — Exporter earnings versus importer costs: the currency reprices equities

Stocks reprice through the currency channel because a firm's *earnings* depend on the exchange rate. The split is clean: a **weaker home currency helps exporters and firms with foreign revenue** (their foreign sales convert into *more* home-currency profit, and their goods get cheaper and more competitive abroad), and it **hurts importers and firms with foreign costs** (their inputs get more expensive). A **stronger** home currency flips every sign.

This is why, in a weak-yen world, Japanese exporters like Toyota and Sony post record profits — their dollars and euros convert into far more yen — while a strong yen squeezes them. It is why a strong-dollar episode is a headwind for the big US multinationals in the S&P 500, which earn roughly 40% of their revenue abroad, and a tailwind for European and Japanese exporters. The equity market is not one bet on the currency; it is a portfolio of opposite bets, and a currency move quietly reshuffles which companies are winning.

#### Worked example: a 10% dollar move on an S&P company with 40% foreign revenue

Take a US multinational — call it a large-cap with \$100 billion in annual revenue, of which 40% (\$40 billion) is earned abroad in foreign currencies and 60% (\$60 billion) is domestic. The dollar strengthens 10%. What happens to reported earnings?

- The \$40 billion of foreign revenue was earned in, say, euros and yen. A 10% stronger dollar means those same foreign-currency sales convert into **10% fewer dollars**: \$40 billion becomes ≈ \$40 billion ÷ 1.10 ≈ \$36.4 billion. That is a **−\$3.6 billion** hit to reported revenue, purely from translation — no fewer units sold.
- The domestic \$60 billion is unaffected.
- So reported revenue falls from \$100 billion to ≈ \$96.4 billion, **−3.6%**. If margins are roughly stable, reported earnings (EPS) fall a similar ≈ 3.6%.
- At a constant price-to-earnings multiple, the stock is worth ≈ 3.6% less — *before* the second-round effect that a strong dollar also makes the firm's exports less competitive, eroding *real* foreign sales on top of the translation hit.

**A 10% dollar move can quietly cut a multinational's reported EPS by ~4% with not a single unit of demand changing — which is why "currency headwinds" is the most common excuse on US earnings calls in a strong-dollar year.** This is the equity leg of the currency channel, and it is the reason a Fed decision in Washington shows up in Apple's and Caterpillar's quarterly numbers. For how the discount rate *also* reprices these same equities, see [How Monetary Policy Moves Stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors).

### Path 3 — Foreign-currency debt: the currency reprices the borrower's solvency

Here is the path that turns a currency move into a *crisis*, and it is the heart of emerging-market vulnerability. When a company or government borrows in a currency that is not its own — almost always the **US dollar** — it has created a dangerous mismatch: its *revenues* are in local currency (a Turkish firm sells in lira, a Brazilian utility collects reais) but its *debt* is in dollars. As long as the local currency is stable, fine. But if the local currency falls, the dollar debt does not shrink — it gets *heavier* in local-currency terms, because the borrower now needs more local money to buy each dollar it owes.

This is **the** reason a strong dollar is a wrecking ball for emerging markets. The dollar is the world's reserve currency and its dominant funding currency: a huge share of cross-border debt, trade invoicing, and commodity pricing is in dollars. So when the dollar strengthens, the real burden of dollar debt rises simultaneously across dozens of countries — a synchronized tightening of financial conditions that no single emerging central bank chose. The dollar is **global collateral**, and when collateral gets scarce and dear, the most leveraged borrowers break first.

#### Worked example: an EM borrower with \$1 billion of dollar debt when its currency falls 20%

Take an emerging-market company that earns its revenue in local currency but borrowed \$1 billion in dollars (cheaper than borrowing locally — that was the temptation). Set the starting exchange rate at 20 local units per dollar.

- **Before:** the \$1 billion debt is worth \$1 billion × 20 = **20 billion local units**. Manageable against, say, local revenue of 8 billion units a year — the debt is 2.5× annual revenue.
- **The local currency falls 20%** against the dollar — now it takes 25 local units to buy a dollar (20 ÷ 0.80 = 25).
- **After:** the same \$1 billion debt is now worth \$1 billion × 25 = **25 billion local units**. The debt grew by **5 billion local units — a 25% jump — without the company borrowing a single extra dollar.**
- Against the same 8 billion of local revenue, the debt-to-revenue ratio jumps from 2.5× to 3.1×. If the company's earnings cannot cover the suddenly-larger local-currency interest bill, it is staring at default — purely because of an exchange-rate move it did not control.

**A 20% currency fall makes a dollar borrower's debt 25% heavier overnight in the money it actually earns — which is why "the dollar is going up" is, for a leveraged emerging market, a solvency event, not a market quote.** This is the mechanism behind the 1997 Asian crisis, the 1998 Russian default, Argentina's repeated blowups, and the 2022 emerging-market stress. The fix is to *match* the currency of your debt to the currency of your revenue — but cheap dollar funding is a temptation governments and firms keep falling for.

#### The dollar as the world's collateral

To understand why the dollar specifically wrecks emerging markets — rather than, say, the euro or the yen — you have to understand that the dollar is not just *a* currency; it is the *plumbing* of the global financial system. Three roles compound:

First, the dollar is the dominant **invoicing currency**: an estimated half of all global trade is priced in dollars even when neither the buyer nor the seller is American — a Korean shipyard selling to a Norwegian firm will often invoice in dollars. So when the dollar rises, the *price* of traded goods rises for everyone at once, tightening global financial conditions without any central bank choosing it.

Second, the dollar is the dominant **funding currency**: a vast share of cross-border bank lending and bond issuance is in dollars, and the Bank for International Settlements tracks *trillions* of dollars of dollar-denominated debt owed by borrowers *outside* the United States. Every one of those borrowers has the currency mismatch from the worked example above, so a strong dollar tightens the screws on all of them simultaneously.

Third, the dollar is the world's **reserve and collateral asset**: central banks hold their reserves mostly in dollars (US Treasuries), and dollars and Treasuries are the collateral that greases global funding markets. When stress hits, everyone scrambles for the *same* safe asset — dollars — which makes the dollar rise *more* precisely when emerging markets can least afford it. This is the cruel feedback loop of a "dollar shortage": fear → dash for dollars → stronger dollar → heavier dollar debt → more fear. In March 2020, this loop got so severe that the Federal Reserve had to open emergency *dollar swap lines* with other central banks — effectively lending dollars to the rest of the world — to stop a global dollar squeeze from cascading into defaults.

The practical upshot is that the dollar is **global collateral**, and the price of collateral is the most important price in the system. When it gets scarce and dear, the most leveraged borrowers break first, and they are almost always in emerging markets. This is why traders watch the DXY as a master risk dial: a rising dollar is not just a currency move, it is a *tightening of the financial conditions of the entire planet*, transmitted through the currency channel into the solvency of borrowers who never touched US policy.

### Path 4 — The carry channel and risk sentiment

The fourth path is the one we opened with: the carry trade ties the currency channel to *global risk appetite itself*. Because the funding currencies (yen, franc) are borrowed to buy risk assets everywhere, the funding currency becomes a *risk barometer*. When markets are calm and greedy, capital flows *out* of the funding currency into risk — the yen weakens, and risk assets rise. When fear hits, the carry unwinds — the funding currency is bought back, it strengthens, and risk assets fall *together*, regardless of fundamentals. This is why a rising yen is now a *symptom* of global risk-off, not just a Japan story: the yen and the VIX move together because the same trade connects them. For how this same risk-sentiment plumbing reaches crypto and FX liquidity, see [How Policy Prices Crypto and FX](/blog/trading/policy-and-markets/how-policy-prices-crypto-and-fx-liquidity-and-divergence).

The reason this path is so important — and so easy to miss — is that it makes the currency channel a *transmitter of contagion* rather than just a repricer of fundamentals. In the first three paths, the currency move changes something real: an inflation rate, an earnings number, a debt burden. In the fourth path, the currency move changes *nothing real* and yet reprices everything, because the carry trade is the wire that connects unrelated assets. When the yen-carry trade unwinds, a Japanese household's mortgage rate, a Mexican government bond, a US technology stock, and a bitcoin position all move together — not because they have anything to do with each other, but because the *same leveraged traders* hold all of them funded by the same short-yen position, and when they are forced to delever they sell all of it at once. This is correlation manufactured by *positioning*, not by fundamentals, and it is the single most underappreciated source of cross-asset risk. It is why a strategy that looks beautifully diversified on paper can lose money in every line item on the same day: the diversification was an illusion, because the positions shared a hidden common factor — the carry — that only reveals itself in the crash.

#### Worked example: the carry's risk-adjusted reality across a cycle

Let me put a number on why the carry trade fools so many professionals. Suppose a yen-carry position pays a steady +5% per year (slightly higher than our earlier example, with a bit of leverage) for *seven* calm years, then loses 30% in a single bad year-eight when the carry unwinds.

- **Seven good years:** +5% each. Cumulative simple gain ≈ +35% over seven years.
- **Year eight, the unwind:** −30% in one year.
- **Net over eight years:** roughly +35% − 30% = **+5% cumulative**, or about +0.6% per year — a *terrible* return for the leverage and risk involved.
- But here is the trap: for the *first seven years*, the strategy showed +5% per year with *tiny* measured volatility (the returns were smooth), producing a gorgeous Sharpe ratio of maybe 2 or 3. Any investor or risk committee looking at years one through seven would have called it a fantastic, low-risk strategy and *added* leverage and capital — right before year eight.

**The carry trade's headline return is a lie told by the calendar: it looks like a high-Sharpe machine for years, then a single unwind reveals that all those smooth gains were an advance against a loss you hadn't taken yet.** The risk was always there; it just hadn't shown up in the sample. This is why position-sizing for a carry trade should be governed by the size of the *crash*, never by the smoothness of the *carry* — a discipline almost no one keeps when the carry has paid reliably for years.

## Common misconceptions

**"Higher interest rates always strengthen a currency."** Half-true and dangerous. A rate *hike* usually lifts a currency in the short run (the naive yield-chasing story), but interest-rate parity says the high-rate currency is *priced to weaken* over time, and that is the bet the forward market makes. Worse, if a country raises rates because it is in a panic — defending a collapsing currency or fighting runaway inflation — the hike can *coincide* with a *falling* currency, because the market reads the hike as a sign of distress, not strength. Turkey raising rates into a lira crisis is not the same as the Fed raising rates from confidence. Rates move currencies, but always read *why* the rate is moving.

**"A strong currency is good and a weak currency is bad."** There is no "good" direction — there are only winners and losers, and they sit on opposite sides of the same move. A strong dollar is great for a US tourist and a US importer and terrible for a US exporter and an emerging-market borrower. A weak yen is wonderful for Toyota's profits and brutal for Japanese households buying imported food and energy. Governments that want a "strong currency" for prestige are usually choosing to help consumers and hurt exporters, whether they admit it or not. The right question is never "strong or weak" but "for whom, and what does it do to the assets I hold?"

**"The carry trade is a clever niche strategy."** It is one of the largest forces in global markets, and its unwind has caused or amplified multiple crises — 1998, 2008, and 2024. The yen-carry trade alone has been estimated, by various measures, in the *trillions* of dollars. It is not a niche; it is a structural feature of a world with persistent rate differentials, and its unwinds are systemic events, not isolated trades.

**"Covered interest parity is just theory — real banks don't trade off it."** The opposite: CIP is one of the most tightly enforced relationships in finance precisely *because* banks arbitrage it continuously. When it does break — as it did during the 2008 crisis and at quarter-ends when bank balance-sheet space is scarce (the "cross-currency basis") — it is a flashing red signal of dollar funding *stress*, watched closely by every central bank. A CIP deviation is not a free lunch lying on the floor; it is a measure of how broken dollar funding has become.

**"A central bank can set its rate and its exchange rate independently."** No — this is the *impossible trinity*, and a country with open capital flows must choose: control the rate *or* control the currency, not both. Raising your rate to fight inflation strengthens your currency whether you want that or not; cutting it to support growth weakens the currency. The exchange rate is the *price* of your monetary-policy choice, paid in the currency channel. (The trinity itself is worked through in the sibling [Currency Policy and FX Intervention](/blog/trading/policy-and-markets/currency-policy-and-fx-intervention).)

## Case studies: when a rate gap moved every asset

Theory is cheap. Here are three dated, real episodes where the currency channel did exactly what the model says — at scale, with numbers.

### Case 1 — August 2024: the yen-carry unwind

Set the scene. For years, the world's most reliable trade was simple: borrow yen at roughly zero and buy literally anything that yielded more — US tech stocks, Mexican bonds, Indian equities, bitcoin. The BOJ held its rate at or below zero from 2016, kept it there while the Fed hiked to 5.25–5.50% in 2022–2023, and the resulting ~5.5-point rate gap (see the rate-differential figure above) drove USD/JPY from about 115 to over 160 by mid-2024 — a 38% weaker yen, and an enormous, crowded, levered carry trade built on top of it.

Then the gap started to close from both ends. On 31 July 2024 the BOJ raised its rate to 0.25% and signaled more, while soft US data on 2 August (a weak jobs report) made the market price Fed *cuts*. The carry's entire premise — a permanently wide, stable rate gap — cracked. The unwind was mechanical and merciless:

- The yen surged: USD/JPY fell from about 160 in early July to about 142 by 5 August — roughly a **12% stronger yen in weeks**, exactly the kind of move our P&L example showed wipes out years of carry.
- The Nikkei 225 fell about **12% on 5 August 2024**, its worst day since 1987, as the strong yen crushed Japanese exporter earnings and forced carry traders to dump everything.
- It went global because the trade was global: the VIX spiked above 65, the S&P 500 fell sharply, Nvidia and the rest of US tech sold off, bitcoin dropped, and high-carry emerging currencies like the Mexican peso fell hard — all in the same 48 hours, all because forced sellers buying back yen had to raise cash by selling whatever they held.

What made the episode so instructive is how *little* fundamental news drove such an enormous move. No bank failed; no recession arrived; corporate earnings were fine. The entire convulsion was *mechanical* — a positioning unwind. Years of one-way carry had built up an enormous, crowded, levered short-yen position spread across thousands of accounts, from sophisticated macro hedge funds to leveraged retail traders. The moment the yen started rising, the first wave of losses forced the most-levered players to buy yen to cut risk; that buying pushed the yen higher; the higher yen triggered the next wave of margin calls; and the loop accelerated into a 48-hour stampede. Crucially, the forced sellers did not only sell yen-funded positions — they sold *whatever was liquid* to raise cash, which is how an unwind that started in the yen reached Nvidia, the S&P 500, bitcoin, and the Mexican peso, none of which had anything to do with Japanese monetary policy.

The speed of the round-trip was as telling as the crash. Within about two weeks the worst-positioned traders had been flushed out, the BOJ's deputy governor gave a soothing speech promising not to hike into market instability, and most of the losses reversed — the Nikkei recovered, the VIX collapsed back toward normal, and US indices made new highs by September. That V-shaped recovery is itself a signature of a *positioning* shock rather than a *fundamental* one: when the cause is forced selling rather than a real economic deterioration, prices snap back once the forced sellers are done. But the snap-back should not obscure the lesson.

The lesson is the whole post in one event: a **0.15-point rate move** plus one soft data print, by narrowing a *rate differential*, reached through the currency channel and convulsed every risk asset on Earth. It is the cleanest demonstration ever recorded of how a rate gap is wired to global asset prices — and a permanent reminder that the most dangerous risks are the ones that look smoothest right before they break. For the positioning playbook around these events — how to see them coming, how leverage breaks — see [Carry-Trade Unwinds: 1998, 2008, 2024](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

### Case 2 — 2022: the wrecking-ball dollar

In 2022 the currency channel ran in the other direction, and it ran over almost everything. The Fed hiked 525 basis points in 16 months — from near-zero to a 4.50% upper bound by December (peaking at 5.25–5.50% in mid-2023) — the fastest tightening since Volcker. With the US out-hiking nearly everyone, the rate differential turned massively in the dollar's favor, and the US Dollar Index (DXY) ripped to about **114 in late September 2022**, a two-decade high. A strong dollar plus rising global rates is the textbook setup for cross-asset carnage, and that is what happened.

![2022 cross-asset returns showing the dollar up and stocks bonds and 60/40 all down](/imgs/blogs/the-currency-channel-rate-differentials-and-carry-7.png)

The bar chart tells the story: the dollar (DXY) was up while the S&P 500 fell about **19%**, the US aggregate bond index fell about **13%**, and the classic 60/40 stock-bond portfolio lost about **16%** — its worst year in modern history, because for once stocks *and* bonds fell together. And outside the US the damage was worse, transmitted by the currency channel: the strong dollar made every other country's imports (especially dollar-priced energy) more expensive, forced emerging-market central banks to hike to defend their currencies, and made dollar debt heavier across the developing world all at once. The yen fell to 32-year lows, the pound briefly approached parity with the dollar in the [UK gilt crisis](/blog/trading/policy-and-markets/the-fiscal-toolkit-spending-taxes-and-deficits), and the phrase "**dollar wrecking ball**" entered the financial vocabulary.

The single image to carry away: in 2022, the only asset class that reliably worked was *holding dollars*. The currency channel meant that the Fed's domestic inflation fight was exported to the whole world as a dollar shock, and the strength of the dollar — not any one country's fundamentals — was the dominant macro variable for the year.

### Case 3 — 2025: the dollar *decline* as a credibility move

If 2022 showed the dollar as a wrecking ball, 2025 showed something rarer and more instructive: a dollar that *fell* even though US rates stayed high — because the move was about **credibility**, not the rate differential. This is the most subtle case, and the most important for understanding the *fourth* channel of all policy: expectations.

![DXY falls from 110 to a 97-98 low across 2025 H1 the worst since 1973](/imgs/blogs/the-currency-channel-rate-differentials-and-carry-3.png)

The chart shows the move: the DXY started 2025 around 110 and fell to roughly 97–98 by April — a decline of about **10.7% in the first half of the year, the worst first-half for the dollar since 1973** (the year the postwar Bretton Woods fixed-exchange-rate system finally collapsed). What makes this case so important is *why* it happened. US interest rates were still high — the Fed held at 4.25–4.50% for most of the first half before cutting later in the year — so the naive "rate gap" story would have predicted a *strong* dollar. Instead the dollar fell, because the currency channel was carrying a different signal: the 2025 tariff shock (the "Liberation Day" reciprocal tariffs of 2 April), erratic policy, and questions about US institutional credibility made global investors *less willing to hold dollar assets at any rate*.

This is the channel's deepest lesson: a rate differential moves a currency *only if the market trusts the issuer*. When credibility cracks, capital leaves *despite* the yield — exactly the opposite of the naive story, and exactly what UIP failing in reverse looks like. The repricing was everywhere: a weaker dollar mechanically lifted dollar-priced commodities (gold ran to record highs above \$3,500/oz on the way to even higher levels), helped emerging markets by lightening their dollar-debt burden, and squeezed any investor who had assumed the dollar's reserve status made it unconditionally safe. The 2025 dollar decline is the proof that the currency channel is, at its root, about *trust* — the rate is just the price tag, and the price tag only matters if buyers believe the seller.

### A note on the managed case: USD/VND and the slow crawl

Not every currency floats freely and lets the channel run wild. Many emerging economies *manage* the exchange rate to keep the currency channel under control — and Vietnam is a clean example of the trade-offs.

![USD/VND rises steadily year by year showing a managed currency crawling weaker](/imgs/blogs/the-currency-channel-rate-differentials-and-carry-6.png)

The chart shows the dong (VND) doing exactly what a managed currency does: crawling *steadily* weaker against the dollar — from about 23,100 per dollar at end-2020 to about 26,100 by end-2025, roughly a 13% depreciation, but spread smoothly over five years rather than snapping in a panic. The State Bank of Vietnam (SBV) runs the dong inside a ±5% band around a daily central rate, deliberately choosing a slow, predictable slide over the violent moves a free float would deliver. The cost of this choice is the impossible trinity: by managing the currency with open trade flows, Vietnam gives up some monetary independence — when the Fed hikes and the dollar strengthens, the SBV must tighten too (or spend reserves) to keep the dong from falling out of its band. The benefit is that exporters and dollar-borrowers get *predictability*; the currency channel still operates, but it is throttled to a manageable trickle instead of a flood. For the legal machinery behind this — the SBV's credit quotas and FX tools — see [SBV Monetary and Banking Law](/blog/trading/law-and-geopolitics/sbv-monetary-and-banking-law-credit-quotas-and-the-dong).

## What it means for asset values: the playbook

Pull the four paths together into something you can actually use. The currency channel is always running; the question is which direction it points and which of your assets it reprices.

**Read the rate-differential first.** The single most useful gauge of where a major currency is headed over months is the *gap* between its central bank's rate path and the other side's. Widening gap in your currency's favor → your currency tends to firm (and the carry into it builds). Narrowing gap → it tends to soften (and any carry unwinds). Watch *expected* rate paths, not just current rates — the market prices the future, which is why a *soft data print* that changes the *expected* path can move a currency before any rate actually changes. That is the entire 5 August 2024 story.

**Map the move to your assets by sign.** A *weaker home currency* is, roughly: bad for your bonds (imported inflation, higher-for-longer), good for your exporters and foreign-revenue stocks, bad for your importers, and dangerous for any borrower with foreign-currency debt. A *stronger home currency* flips all four. Before you ask "is this currency move good or bad," ask "good or bad *for which asset I hold*."

**Respect the carry's asymmetry.** A carry trade — in any form, including just being long high-yield emerging-market assets funded implicitly in dollars — pays a smooth premium most of the time and takes it all back in a crash. Size it for the crash, not the calm. The signal that a carry unwind is brewing: a crowded, levered trade (everyone is on the same side) *plus* a narrowing rate gap or a rising risk gauge (the VIX, credit spreads). When both line up, the funding currency (yen, franc) is the thing to watch — when it starts ripping higher for no fundamental reason, the unwind has started.

**For emerging markets, watch the dollar above all.** A strengthening dollar is a synchronized tightening for the entire developing world through the dollar-debt channel — it is the single biggest risk-off macro variable for EM assets. The cleanest tell is the DXY: a sustained, broad dollar rally is a headwind for *everything* priced off cheap dollar funding, and a falling dollar (as in 2025) is the green light for EM, commodities, and gold.

**What would invalidate the read.** The currency channel's naive "rate up → currency up" arrow breaks when *credibility* is the issue (2025): if a country's institutions, fiscal path, or policy reliability come into doubt, capital can leave *despite* high rates, and the currency falls when the rate model says it should rise. So the differential is necessary but not sufficient — always sanity-check it against whether the market still *trusts* the currency's issuer. When rates say "strong" but the currency is falling, credibility is the missing variable, and that is usually the more important story.

The thread of the whole series holds here: a central bank pulls a lever (a rate), it travels through a channel (the currency), and it lands as a change in what assets are worth (bonds, equities, the solvency of a borrower an ocean away). The currency channel is the one that makes a decision in one capital reprice assets in every other — which is exactly why a 0.15-point move in Tokyo can be felt on every trading floor on Earth.

## Further reading & cross-links

**Within this series — Policy & Markets:**
- [Currency Policy and FX Intervention](/blog/trading/policy-and-markets/currency-policy-and-fx-intervention) — the FX-as-a-policy-*lever* companion: pegs, bands, reserves, the impossible trinity, and direct intervention mechanics.
- [How Policy Prices Crypto and FX: Liquidity and Divergence](/blog/trading/policy-and-markets/how-policy-prices-crypto-and-fx-liquidity-and-divergence) — how the same risk-sentiment and liquidity plumbing reaches crypto and FX.
- [The Discount-Rate Channel: How Rates Reprice Cash Flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) — the sibling channel that reprices bonds and equities through the rate, not the currency.

**Macro-trading — the positioning playbook:**
- [How Monetary Policy Moves Currencies: Rate Differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — the trader's-eye view of trading the rate-differential-to-FX relationship.
- [Carry-Trade Unwinds: 1998, 2008, 2024 — When Leverage Breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the anatomy of how crowded, levered carry trades break.

**Macro-correlations — the statistics:**
- [The Dollar (DXY) Cross-Asset Correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) — the empirical betas and rolling correlations between the dollar and every other asset class.

**Law & geopolitics — the rulebook:**
- [SBV Monetary and Banking Law: Credit Quotas and the Dong](/blog/trading/law-and-geopolitics/sbv-monetary-and-banking-law-credit-quotas-and-the-dong) — the legal machinery behind Vietnam's managed dong.
