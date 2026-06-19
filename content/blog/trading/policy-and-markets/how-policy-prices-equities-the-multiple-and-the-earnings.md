---
title: "How Policy Prices Equities: The Multiple and the Earnings"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A stock's price is its multiple times its earnings — and policy moves each leg through a different channel. How rates set the P/E, how taxes and tariffs set EPS, and why sectors rotate."
tags: ["monetary-policy", "fiscal-policy", "equities", "asset-valuation", "interest-rates", "earnings", "sector-rotation", "tariffs", "tax-policy", "equity-risk-premium", "central-banks", "stock-market"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A stock's price is two numbers multiplied together — the **multiple** (its P/E) and the **earnings** (its EPS) — and policy reaches each through a different door. Rates and liquidity move the *multiple*; taxes, tariffs, and the demand cycle move the *earnings*. Read any equity move by asking which leg policy just touched.
>
> - The discount-rate channel sets the multiple: P/E is roughly 1/(r − g), so when the Fed cuts and yields fall, the same dollar of earnings is worth more, and the P/E rises. 2020-21 zero rates lifted the S&P P/E above 22x; the 2022 hiking cycle compressed it back toward 16-17x even though earnings rose — a textbook multiple-driven move.
> - The demand channel sets the earnings: fiscal stimulus and tax cuts lift EPS directly. The 2017 TCJA cut the corporate rate from 35% to 21% and added roughly \$10-12 to S&P EPS in a single year. The 2025 tariffs threatened the earnings leg while rate cuts supported the multiple — opposite forces on the same price.
> - Different sectors are different *durations*: long-duration tech and rate-sensitive REITs/utilities fall most when rates rise; banks and energy hold up or benefit. That is why leadership rotates through the policy cycle.
> - **The one number to remember:** a +1 percentage-point rise in the discount rate, with growth unchanged, cuts a fair P/E of 25x to about 20x — a ~20% hit to price from the multiple alone, before earnings move a cent.

In 2020 and 2022 the S&P 500 was made of almost exactly the same companies — the same Apples, Microsofts, JPMorgans, ExxonMobils, earning broadly similar profits. Yet the index behaved as though it were two different markets. In 2020 the Federal Reserve cut its policy rate to zero and bought bonds without limit; by late 2021 the index traded at more than 22 times its earnings and kept setting records. In 2022 the same Fed raised that rate by 525 basis points in sixteen months — the fastest hiking cycle since Paul Volcker — and the index fell more than 19% on the year *even as corporate earnings held up and even grew*. Same firms. Same profits. Opposite policy. Opposite price.

That is the puzzle this post exists to solve, and the solution is almost embarrassingly simple once you see it. A stock's price is not one number. It is **two** numbers multiplied together: how much profit the company makes per share (its **earnings**), and how many dollars investors will pay for each dollar of that profit (its **multiple**). Write it out and the whole thing fits on a napkin:

> **Price = Multiple × Earnings**, or in the language traders use, **P = (P/E) × EPS.**

Once you hold those two numbers apart, every confusing equity move resolves into a clean question: *which leg did policy just touch?* The Fed and the bond market move the **multiple**. Congress, the Treasury, the tax code, and tariffs move the **earnings**. The 2020-vs-2022 puzzle was a multiple story — rates went up, the P/E came down, earnings barely moved. The 2025 tariff shock was, at heart, an earnings story dressed up as a multiple story. Tell them apart and you can read the policy newspaper like a pricing sheet.

![Diagram showing a policy lever splitting into a discount-rate channel that sets the multiple and a demand channel that sets the earnings, the two combining into the stock price.](/imgs/blogs/how-policy-prices-equities-the-multiple-and-the-earnings-1.png)

This is the first post in the per-asset valuation track of the series. We have already built the two machines it relies on — [the discount-rate channel, how rates reprice cash flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) and [the fiscal toolkit of spending, taxes, and deficits](/blog/trading/policy-and-markets/the-fiscal-toolkit-spending-taxes-and-deficits). Here we point both of them at one asset class — stocks — and show exactly where each one lands.

## Foundations: what a stock is actually worth

Start with the most basic question, the one every valuation rests on: *why is a share of a company worth anything at all?* A share is a claim on a slice of all the cash the company will ever hand back to its owners — dividends today, buybacks tomorrow, and ultimately whatever is left if the firm is sold or wound down. A stock is, in the end, a stream of future cash flows, just like a bond. The difference is that a bond's cash flows are fixed and contractual, while a stock's are uncertain and (usually) growing.

So a stock's fair value is the present value of all those future cash flows. "Present value" is the idea that a dollar arriving years from now is worth less than a dollar in your hand today, because you could invest today's dollar and earn a return. We **discount** future cash by a rate `r` that reflects what else you could earn for the same risk. The further out the cash and the higher the discount rate, the less it is worth today. (If that sentence felt fast, the [discount-rate channel post](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) builds it from zero.)

There is a famous shortcut. If a company's cash flows grow forever at a steady rate `g`, and you discount them at rate `r`, the whole infinite stream collapses to one clean fraction — the **Gordon growth model**:

```
Value = next year's cash flow / (r − g)
```

Divide both sides by earnings and you get the single most important equation in equity valuation:

```
Multiple (P/E) = payout / (r − g)
```

If the company pays out all its earnings, this is just **P/E ≈ 1 / (r − g)**. That little fraction is the entire discount-rate channel in one line. The numerator is about *growth and payout*; the denominator's `r` is the **discount rate that policy controls**. Push `r` down and the gap `(r − g)` shrinks, so the multiple explodes; push `r` up and the gap widens, so the multiple collapses. The multiple is a *lever on growth versus rates*, and the central bank has its hand on the rate.

Now define the two numbers cleanly, because the whole post lives in the space between them:

- **Earnings (EPS — earnings per share):** the company's actual profit, divided by the number of shares. This is a fact about the *business* — how much it sells, at what margin, after what tax rate. Demand, costs, and the tax code live here.
- **The multiple (P/E — price-to-earnings):** how many dollars the market pays for each dollar of EPS. This is a fact about *the market's mood and the discount rate* — how cheap money is, how much growth investors expect, how nervous they feel. Rates and liquidity live here.

Multiply them and you have the price. That is the whole skeleton.

![A three-column decomposition showing price equals the P/E multiple times earnings per share, with policy arrows pointing at each leg.](/imgs/blogs/how-policy-prices-equities-the-multiple-and-the-earnings-3.png)

#### Worked example: how a 1-point rate move reprices the multiple

Take a fast-growing company — call it a long-duration growth stock — whose cash flows the market expects to grow at `g = 6%` a year. Suppose the appropriate discount rate is `r = 10%`. The fair multiple is:

P/E ≈ 1 / (r − g) = 1 / (0.10 − 0.06) = 1 / 0.04 = **25x**.

If the company earns \$4 of EPS, the fair price is 25 × \$4 = **\$100**.

Now the Fed hikes and the discount rate rises by **one percentage point**, to `r = 11%`, with growth unchanged. The new fair multiple is:

P/E ≈ 1 / (0.11 − 0.06) = 1 / 0.05 = **20x**.

The fair price falls to 20 × \$4 = **\$80**. A single percentage point on the discount rate — *with earnings untouched* — knocked **20% off the price**. Notice the leverage: because `(r − g)` was a small number (0.04), a small absolute change in `r` is a large *proportional* change in the denominator. **The smaller the gap between the discount rate and the growth rate, the more violently the multiple reacts to rates** — which is exactly why the highest-growth, longest-duration stocks get hit hardest when the Fed tightens. That single fraction explains the 2022 tech crash and the 2020 tech melt-up in one stroke.

### Why "later" cash flows lose the most to rates

The single fact that makes the whole post work is that *not all of a stock's value sits at the same point in time.* Picture a company's value as a row of envelopes, one for each future year, each holding the cash that year is expected to deliver back to owners. The discount rate is the toll you pay to drag each envelope back to the present — and the toll compounds, so the further out the envelope, the more the toll eats.

Use everyday numbers. A dollar arriving **next year**, discounted at 5%, is worth \$1 / 1.05 = **\$0.95** today — you lose a nickel. The same dollar arriving in **ten years**, discounted at 5%, is worth \$1 / 1.05¹⁰ = **\$0.61** — you lose almost forty cents. Now raise the rate from 5% to 6%. The near dollar barely flinches: \$1 / 1.06 = \$0.943, a 0.7% haircut. But the ten-year dollar drops to \$1 / 1.06¹⁰ = **\$0.558**, a **9% haircut** — more than ten times the damage. *The same one-point rate rise costs a far-future cash flow far more present value than a near one.* That is the entire intuition behind "duration": a stock whose value lives mostly in distant envelopes (a young, high-growth company) is a long-duration asset, and the discount rate is a wrecking ball aimed straight at it. A stock whose value lives in this year's and next year's envelopes (a mature dividend payer) barely feels the same blow. Hold this picture — it is the hinge between the multiple, the growth-versus-value trade, and the sector clock, all of which are the same fact seen from three angles.

### What "earnings" really means — and why buybacks matter

We have been loose with "earnings" and "the cash a company hands back," so tighten it. The **E** in P/E × E is accounting profit, but what owners ultimately receive is *cash returned* — dividends plus buybacks. Dividends are obvious: the company mails you a check. Buybacks are subtler but, for U.S. equities, now larger: the company uses its profit to buy its own shares in the open market and retire them, which shrinks the share count and therefore *raises EPS even if total profit is flat* (the same pie, fewer slices). For the S&P 500, buybacks have run on the order of \$800 billion to \$1 trillion a year in recent years — a bigger return of cash than dividends. Policy reaches *this* too: the corporate tax rate sets how much after-tax profit is available to return; a tax cut funds bigger buybacks (much of the 2018 buyback surge was the TCJA windfall), and a buyback *excise tax* (a 1% levy introduced in 2023) nicks the channel at the margin. The takeaway for valuation: when you move the *E*, remember it flows to owners as dividends-plus-buybacks, and that buybacks let a tax cut lift *per-share* earnings even beyond the rise in total profit — a quiet multiplier on the earnings channel.

## The discount-rate channel: how policy moves the multiple

The Fed does not set a "stock multiple" anywhere. It sets one number — the overnight federal funds rate — and lets that number ripple out through the entire structure of interest rates. The two-year yield, the ten-year yield, mortgage rates, corporate borrowing costs: all of them move off the policy rate and the market's expectation of where it is going. The discount rate `r` that prices a stock is built out of those yields. Mechanically, `r` is roughly the **risk-free rate** (think the ten-year Treasury yield) **plus an equity risk premium** — the extra return investors demand for bearing stock-market risk rather than holding a safe government bond. We will come back to the risk premium; for now, the headline is that *when the Fed moves rates, it moves the risk-free anchor under every multiple in the market.*

Here is the causal chain, link by link:

1. **The Fed changes the policy rate** (or, just as powerfully, changes where the market expects it to go — *guidance* moves yields before any actual hike). For the toolkit of rates, QE, QT, and forward guidance, see [the central-bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).
2. **Treasury yields reprice.** A higher expected path for the funds rate lifts the whole curve; the ten-year yield — the risk-free anchor for equities — rises.
3. **The discount rate `r` rises**, because `r` = risk-free + equity risk premium.
4. **The multiple `1/(r − g)` falls.** Every dollar of future earnings is now discounted harder, so each dollar of *current* earnings is worth fewer dollars of price.
5. **The price falls — even if earnings haven't moved.** This is the move that confuses people: the company is doing fine, but it is worth less, because money got more expensive.

Balance-sheet policy works through the same door from a different angle. When the Fed buys bonds (QE), it pushes yields *down* and floods the system with reserves that have to go somewhere — much of it into risk assets, compressing the equity risk premium and lifting multiples. When it sells or lets bonds roll off (QT), it drains that tide. For the mechanics of how the balance sheet moves markets, see [QE vs QT](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets); for the positioning playbook, [how monetary policy moves stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors). What *this* series owns is the valuation arithmetic and the case studies — the *why the number changed*, not the *how to trade it*.

There is a second, subtler way QE lifts multiples beyond just pushing the ten-year yield down: it works on the *equity risk premium* directly. When the Fed buys trillions in bonds, it takes safe, yield-bearing assets *out* of private hands and replaces them with reserves earning almost nothing. Investors who wanted that safe yield are pushed out the risk curve — out of Treasuries into corporate bonds, out of corporates into stocks — a process strategists call the "portfolio-balance" or "reach-for-yield" effect. More dollars chasing the same pool of risk assets bids their prices up and their *required return* down, which is the same as a *lower discount rate* and therefore a *higher multiple*. The Fed's balance sheet went from \$0.9 trillion before 2008 to a peak of nearly \$9 trillion in April 2022; that eightfold expansion is a large part of why multiples spent the 2010s and 2020-21 structurally elevated. So the discount-rate channel has two sub-levers the Fed can pull: the *rate* itself (the risk-free anchor) and the *quantity* of safe assets (the risk premium). QT reverses both — and that is why 2022 was so punishing for multiples: the Fed hiked the rate *and* started draining the balance-sheet tide at the same time, hitting both sub-levers at once.

#### Worked example: how QE-driven risk-premium compression lifts the index

Hold S&P earnings fixed at **\$200** per index share and the ten-year yield fixed at **2.0%**, and watch *only* the equity risk premium move as QE floods the system with reserves. Before QE, investors demand an ERP of **4.0%**, so the discount rate is `r` = 2.0% + 4.0% = **6.0%**. With long-run growth `g` = 4.0%, the fair multiple is 1/(0.06 − 0.04) = 50x in the raw model; scale it to a realistic payout-adjusted **18x**, putting the index at 18 × \$200 = **3,600**.

Now QE pushes investors out the risk curve and the ERP compresses by **one point**, to 3.0% — *with the bond yield and earnings unchanged*. The discount rate falls to `r` = 5.0%, the gap `(r − g)` narrows from 2.0% to 1.0%, and the fair multiple roughly *doubles* in the raw model; in the realistic version call it a move from 18x to **22x**, lifting the index to 22 × \$200 = **4,400** — a **+22%** gain with no change in rates, earnings, or growth. *That is QE's gift to stocks: it does not need to move the headline yield or earnings at all; squeezing the risk premium is enough to re-rate the whole market.* And it runs in reverse under QT — which is the quiet second engine behind 2022's multiple compression.

![Step chart of the Fed funds upper bound from 2019 to 2026 with shaded bands marking the zero-rate, hiking, and cutting regimes and the matching P/E story.](/imgs/blogs/how-policy-prices-equities-the-multiple-and-the-earnings-4.png)

The figure above is the discount-rate channel as it actually played out. The flat zero band from 2020 to early 2022 is the period the S&P traded above 22 times earnings. The staircase up through 2022-23 — 525 basis points of hikes — is the period the multiple compressed toward 16-17x. The descent that began in late 2024 and continued through 2025 is the multiple re-expanding back toward 22x. One policy rate; three multiple regimes. (Note: there is no published S&P P/E *time series* in this series' curated data, so the P/E figures here are stated qualitatively from contemporaneous market reporting and anchored to the index-level chart that follows — we do not invent a P/E array.)

#### Worked example: the 2021-to-2022 move was almost pure multiple

Let us put real numbers on the headline puzzle. At the end of 2021 the S&P 500 closed near **4,766**, and S&P 500 trailing earnings were roughly **\$208** per index share. That implies a multiple of:

4,766 / 208 ≈ **22.9x**.

Through 2022 the Fed hiked 525 basis points. S&P earnings did *not* fall — full-year 2022 operating EPS came in *higher*, around **\$219** per index share, as nominal sales rose with inflation. Yet the index *fell* to about **3,840** at the end of 2022. The new multiple was:

3,840 / 219 ≈ **17.5x**.

Decompose the move. Earnings went *up* about 5% (208 → 219), which on its own should have *lifted* the index by ~5%, to roughly 5,000. Instead the index fell ~19%. The entire decline — and then some — came from the multiple collapsing from ~22.9x to ~17.5x, a **24% compression** driven almost entirely by the discount rate rising. In one line: *earnings were a small tailwind; the multiple was a large headwind; the multiple won.* That is what a hiking cycle does to stocks. It rarely needs to break earnings — it just makes every dollar of earnings worth fewer dollars of price.

![Two side-by-side stacked columns comparing 2021 and 2022: earnings roughly flat while the multiple compresses from 22x to 17x, dragging the price down.](/imgs/blogs/how-policy-prices-equities-the-multiple-and-the-earnings-8.png)

## The demand channel: how policy moves the earnings

The other door is the earnings door, and the government — not the central bank — holds the key. Earnings are profits, and profits depend on three things the fiscal system touches directly: **how much customers buy** (demand), **how much it costs to make and sell** (input costs, including tariffs), and **how much of the profit the company keeps after tax** (the corporate tax rate). Pull a fiscal lever and you move EPS — sometimes immediately, sometimes through the slower grind of the business cycle.

There are four distinct ways fiscal and trade policy reach the earnings line:

- **The corporate tax rate — the cleanest, fastest lever.** Cut the rate companies pay on profits and after-tax earnings jump *mechanically*, with no change in the business itself. A firm earning \$100 pre-tax keeps \$65 at a 35% rate but \$79 at a 21% rate — a 21.5% increase in after-tax profit, overnight, from a single line of the tax code. This is the [TCJA](/blog/trading/law-and-geopolitics/tax-law-as-a-market-force) story, and we work it below.
- **Fiscal stimulus — the demand lever.** Government spending and transfer payments (stimulus checks, expanded benefits, infrastructure outlays) put money in customers' pockets, lifting sales and therefore earnings. For the trader's view of this, see [fiscal policy for traders](/blog/trading/macro-trading/fiscal-policy-for-traders-spending-deficits-demand).
- **Tariffs — a tax on inputs that *cuts* earnings.** A tariff is a tax on imports. For a company that buys parts or finished goods abroad, it raises costs; unless the firm can pass every cent on to customers (it usually can't), the gap comes out of margins — straight off EPS. This is the 2025 story.
- **The cycle itself — policy's slow earnings lever.** Beyond direct taxes and spending, the entire stance of policy steers the business cycle, and earnings rise and fall with it. Easy policy → expansion → rising sales and margins → rising earnings. Tight policy → slowdown → the reverse. This is why a hiking cycle that goes too far eventually hits *both* legs: first the multiple (fast), then the earnings (slow, via recession).

The crucial asymmetry: **the multiple reprices in seconds, earnings reprice over quarters.** When the Fed surprises the market, the multiple gaps the same afternoon — discount rates are a market price and move instantly. But a tax cut takes a quarter or two to show up in reported EPS, and a tariff's margin damage shows up only as inventories turn over and contracts reset. So policy that hits the multiple is a *fast* market mover; policy that hits earnings is a *slow* one — which is exactly why tariff headlines first hit stocks through fear (a multiple move) and only later through actual margin compression (an earnings move).

#### Worked example: the 2017 corporate tax cut lifted S&P EPS by double digits

In December 2017 the Tax Cuts and Jobs Act cut the U.S. statutory corporate tax rate from **35% to 21%**. Hold the *pre-tax* profit of the S&P 500 constant and just change the tax the companies pay. A simplified index-level calculation:

- Suppose the S&P 500's *pre-tax* earnings imply roughly **\$160** of after-tax EPS at the old 35% rate. Pre-tax EPS would then be \$160 / (1 − 0.35) = **\$246** per index share.
- Re-tax that same \$246 of pre-tax profit at the new 21% rate: after-tax EPS = \$246 × (1 − 0.21) = **\$194**.
- The mechanical lift is \$194 − \$160 = **\$34** of EPS — a **~21% increase in after-tax earnings** purely from the rate change, before a single extra unit is sold.

In reality the full effect was diluted — not every company paid the full 35% (many used deductions and foreign structures), so the *effective* rate fell less than the *statutory* rate, and the real-world S&P EPS bump from tax reform is usually estimated at roughly **\$10-12** per index share, lifting 2018 operating EPS from the high-\$130s toward ~\$160. Still: a single tax law added on the order of **8-10% to index earnings** in one year, and the market priced it in advance — much of the late-2017 rally was the tax cut being capitalized into EPS expectations. *The tax rate is the most direct line policy has into the earnings leg of a stock's price.* The catch worth filing away: when the bill that extends those cuts in 2025 — the [OBBBA](/blog/trading/policy-and-markets/the-fiscal-toolkit-spending-taxes-and-deficits), signed July 4, 2025 — *extends* an existing rate rather than cutting a fresh one, it removes a feared *headwind* (a snap-back to higher rates) rather than delivering a new tailwind. Avoiding a cut is worth less to EPS than making one.

#### Worked example: a tariff cutting a retailer's EPS and its share price

Take a concrete, simplified retailer. It imports goods, marks them up, and sells them. Per share:

- Sales: **\$100** of revenue per share.
- Cost of imported goods: **\$60** per share.
- Other costs (wages, rent, overhead): **\$30** per share.
- Pre-tax profit: \$100 − \$60 − \$30 = **\$10**; after a 21% tax, EPS = **\$7.90**.

Now a **20% tariff** lands on its imported goods. The \$60 of imports now costs \$60 × 1.20 = **\$72** — an extra **\$12** per share in costs. Suppose the retailer can pass *half* of it to customers by raising prices (lifting revenue by \$6) and eats the other half. New numbers:

- Sales: \$106. Cost of goods: \$72. Other costs: \$30.
- Pre-tax profit: \$106 − \$72 − \$30 = **\$4**; after 21% tax, EPS = **\$3.16**.

Earnings per share fell from \$7.90 to \$3.16 — a **60% collapse in EPS** from a 20% tariff, because the tariff hit a cost line (\$60) that was *six times* the profit line (\$10). This is the brutal arithmetic of tariffs on thin-margin, import-heavy businesses: **a tariff that is small relative to revenue can be enormous relative to profit.** If the stock holds its 18x multiple, the price falls from 18 × \$7.90 = **\$142** to 18 × \$3.16 = **\$57** — a 60% drop driven entirely by the earnings leg. In practice the multiple *also* compresses on the uncertainty, which is why import-exposed retailers and manufacturers were among the worst performers in April 2025. *Tariffs are an earnings shock first and a multiple shock second.*

![Grouped bar chart comparing the April 2025 headline reciprocal tariff rates with the lower negotiated rates that settled later in the year, by trading partner.](/imgs/blogs/how-policy-prices-equities-the-multiple-and-the-earnings-7.png)

The chart above is the 2025 tariff shock in one frame. In April the headline "reciprocal" rates were extraordinary — China 34%, Vietnam 46%, the EU 20%, Japan 24% — the highest U.S. tariffs since the 1930s. Those numbers are an earnings *threat* the size of which the worked example just showed. Then the negotiations: China settled near 30%, Vietnam at 20% (down from 46%), the EU and Japan at 15%. Every notch lower on those bars is earnings handed *back* to import-exposed companies — which is precisely why the S&P could recover even as a baseline tariff stayed in place. The market was repricing the *earnings* leg in real time as the policy uncertainty resolved.

## Growth versus value: a duration trade in disguise

Here is the deepest single idea in equity-policy analysis, and it is worth slowing down for. **Growth stocks and value stocks are not really two styles of company — they are two different *durations* of cash flow, and rates reprice durations.**

A **value stock** earns most of its cash *now* — a mature bank, a utility, a consumer-staples giant throwing off dividends today. A **growth stock** earns most of its cash *later* — a young software or biotech firm whose profits are mostly a promise about the 2030s. Recall from the foundations that *the further out a cash flow, the harder the discount rate bites it.* A dollar promised in 2035 loses far more present value from a 1-point rate rise than a dollar paid in 2026. So:

- **Growth = long-duration cash flows = highly rate-sensitive.** When rates fall, growth multiples explode (2020). When rates rise, growth multiples crater (2022).
- **Value = short-duration cash flows = less rate-sensitive.** Value holds up better when rates rise because its cash is close, and its multiples were never stretched to begin with.

This is *exactly* the same mathematics as bond duration. A 30-year bond moves far more for a given yield change than a 2-year bond; a growth stock is a "long-duration" equity and a value stock a "short-duration" one. When you hear "rotation from growth to value," translate it instantly to: *the discount rate is rising, and the market is shortening the duration of what it owns.* When you hear "growth is leading again," translate it to: *rates are falling (or expected to), and the market is extending duration to capture the convexity.* The whole growth/value cycle is the discount-rate channel acting on the *time profile* of earnings rather than their level — and it is the bridge between [how rates reprice cash flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) and the sector story we turn to next.

## Sector rotation through the policy cycle

If growth-versus-value is the discount-rate channel sorting stocks by duration, **sector rotation** is the same logic applied to the eleven industry groups, plus the earnings cycle layered on top. Different sectors have different rate sensitivities (different durations) *and* different sensitivities to the economic cycle, so as policy moves through its phases, market leadership rotates.

Three forces sort the sectors:

1. **Rate sensitivity (the multiple channel).** Long-duration, no-dividend tech behaves like a long bond — it soars when rates fall and gets crushed when they rise. **Rate-sensitive yield sectors** — utilities, REITs (real-estate trusts), telecoms — are bond *proxies*: investors buy them for the dividend, so when bond yields rise those dividends look less attractive and the sectors fall, almost mechanically. (REITs are doubly exposed: they borrow heavily, so higher rates hit both their multiple *and* their earnings.)
2. **Earnings-from-rates (the special cases).** A few sectors *earn more* when rates rise. **Banks** make money on the spread between what they lend at and what they pay depositors; a higher rate and a steeper curve widen that spread, lifting bank *earnings* even as the multiple channel pressures everyone else. **Energy and materials** are tied to commodity prices and the inflation that often accompanies a hiking cycle.
3. **Cyclicality (the demand channel).** **Cyclical** sectors — consumer discretionary, industrials, materials — live and die with the economy, so they lead in the early-cycle recovery and lag in the slowdown. **Defensive** sectors — consumer staples, health care, utilities — sell things people buy in any economy (food, medicine, electricity), so their earnings are stable and they outperform when growth is falling.

![A four-quadrant sector-rotation clock showing which sectors lead in each phase of the policy and rate cycle, rotating clockwise from early easing to recession.](/imgs/blogs/how-policy-prices-equities-the-multiple-and-the-earnings-6.png)

Read the clock above clockwise. **Early easing** (cuts begin, recovery ahead): cheap money and a brightening outlook favor long-duration growth — tech, consumer discretionary, small caps. **Mid-cycle into early hikes** (growth firm, rates rising): the steepening curve and strong demand favor financials, industrials, energy, materials. **Late cycle at peak rates** (inflation high, policy tight): you want short-duration, inflation-hedged, cash-rich names — energy, staples, health care; you avoid anything long-duration. **Slowdown into recession** (growth falling, cuts coming): pure defensives lead — utilities, staples, health care — and, crucially, *bonds often beat stocks entirely* as the discount-rate channel reverses. This clock is a map of where the discount-rate and demand channels point at each phase; for the cross-asset version of the same idea, see [asset rotation across the business cycle](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants).

The 2022-2025 cycle traced the clock almost perfectly. **2022** (hikes accelerating): energy was the *only* S&P sector to finish positive — up over 50% — as oil spiked and its short-duration cash flows shrugged off rising rates, while long-duration tech and consumer discretionary fell 25-35%. **2023-24** (rates peaking, then AI): the multiple channel reversed for the mega-cap "Magnificent Seven" tech names as the market sensed the top in rates and priced an AI earnings boom — long-duration growth led again. **Late 2024-2025** (cuts begin): rate-sensitive and cyclical sectors found a bid as the funds rate came down, even as the tariff shock scrambled the earnings outlook sector by sector. Rotation is not noise; it is the policy cycle expressed in industry weights.

![Horizontal bar chart of illustrative equity sensitivity by sector to a one-point rise in long rates, from long-duration tech most negative to banks most positive.](/imgs/blogs/how-policy-prices-equities-the-multiple-and-the-earnings-5.png)

The bar chart above sketches the rate-sensitivity ordering (note: this is *illustrative* — the sign and rough magnitude of each sector's reaction to a +1-point rise in long rates, not a fitted statistical beta, which is not in this series' data). Long-duration tech and rate-sensitive REITs and utilities sit deepest in the red; banks and energy sit on the green side. For the *measured* statistical relationships — actual rolling betas and the way the stock-bond correlation itself flips with the inflation regime — cross to [inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips). This series owns the *mechanism*; macro-correlations owns the *numbers*.

#### Worked example: a bank, a utility, and a tech stock meet a 1-point rate rise

Put three stocks side by side and raise long rates by one percentage point. Watch the two channels work in opposite directions across them.

- **The tech stock (long duration).** Most of its value is far-future cash. Using P/E ≈ 1/(r − g) with `g = 6%`, the multiple falls from 25x (at r = 10%) to 20x (at r = 11%) — a **−20% price hit** from the multiple, and its earnings are unaffected by rates. Net: **roughly −20%.**
- **The utility (rate-sensitive, short duration, a bond proxy).** Its cash flows are near and stable, so the duration hit is milder — say a multiple move from 18x to 17x, about **−6%**. But it is a *yield* play competing with bonds, and it carries heavy debt, so higher rates also lift its interest costs and nick its earnings by a couple of percent. Net: **roughly −7% to −8%.**
- **The bank (earnings benefit from rates).** The multiple channel still pressures it slightly — say **−2%** on the multiple. But a higher rate widens its lending spread: suppose net interest income rises enough to lift EPS by **+5%**. The two legs partly cancel and the bank comes out **roughly +2% to +3%** — *up* on the same rate move that sank the tech stock.

One policy action, three opposite outcomes, all explained by which leg it touched: the tech stock was hit on the multiple, the utility on both legs mildly, the bank *helped* on the earnings leg. *Sector rotation is not a personality test for stocks — it is the two channels landing differently on different durations and different business models.* This is the single most useful frame for trading a Fed decision at the sector level.

## The equity risk premium and the Fed put

We said the discount rate `r` is the risk-free rate *plus* an **equity risk premium** (ERP) — the extra annual return investors demand for holding risky stocks instead of safe Treasuries. The ERP is the part of `r` the central bank does *not* set directly but powerfully influences, and it is where "the Fed and stocks" gets interesting.

A rough, widely-used estimate of the ERP is the **earnings yield minus the bond yield**. The earnings yield is just the inverted P/E — if the S&P trades at 20x, its earnings yield is 1/20 = 5%. Subtract the ten-year Treasury yield and you get a crude ERP:

```
ERP ~ earnings yield - 10Y yield = (1 / P/E) - 10Y
```

When the S&P traded at 22x in 2021 (a 4.5% earnings yield) against a 1.5% ten-year, the implied ERP was about **3%** — historically thin, meaning stocks were richly priced relative to bonds, leaning on cheap money. By late 2023, with the S&P near 18-19x (a ~5.3% earnings yield) against a ~4.9% ten-year, the implied ERP had collapsed toward **0.4%** — stocks were barely compensating investors more than risk-free bonds. That compression is the discount-rate channel at its most uncomfortable: when bond yields rise toward the earnings yield, *every* dollar in stocks is being asked to justify itself against a suddenly-attractive safe alternative.

This is the machinery behind two famous phrases.

**The "Fed put"** is the market's belief that if stocks fall far enough, the Fed will ease — cut rates or restart QE — to stop the bleeding, the way a put option limits your downside. The belief is self-reinforcing: because investors expect support, they demand a *lower* risk premium (they feel protected), which *raises* multiples. The Fed put is, in valuation terms, a structural *compression of the equity risk premium* — a gift to multiples that lasts exactly as long as the Fed is willing to honor it. In 2020 the Fed honored it spectacularly (zero rates, unlimited QE, even corporate-bond backstops); in 2022 the Fed *revoked* it (it needed stocks weaker to cool demand and inflation), which is half of why multiples compressed so hard.

**"Good news is bad news"** is the regime where strong economic data makes stocks *fall*. It sounds backwards until you hold the two channels apart. In a normal regime, good news (strong jobs, rising sales) lifts the *earnings* leg and stocks rise. But when inflation is the Fed's enemy, good news means the economy is running hot, which means the Fed will keep rates *higher for longer*, which lifts the *discount rate* and compresses the *multiple*. If the multiple hit outweighs the earnings benefit, good news sinks stocks. This dominated 2022-2023: a hot payrolls number would send the S&P *down* because traders read it as "more hikes." The regime flips back to "good news is good news" once the Fed's fight with inflation is won and the earnings leg is allowed to lead again — which is roughly what happened into 2024. *Knowing which regime you are in is knowing which leg the market is currently pricing.*

#### Worked example: how the equity risk premium reprices the whole index

Suppose S&P earnings are fixed at **\$240** per index share, and the ten-year Treasury yields **4.0%**. Investors demand an equity risk premium of **2.0%**, so the discount rate is `r` = 4.0% + 2.0% = **6.0%**. With long-run earnings growth `g` = 4.0% (roughly nominal GDP), the fair multiple is 1/(0.06 − 0.04) = **50x** — unrealistically high, which tells you real-world ERPs and growth assumptions are more conservative, but the *direction* is what matters. Use a more realistic payout-adjusted setup and call the fair multiple **20x**, so the index is 20 × \$240 = **4,800**.

Now a *credibility shock* — say a chaotic policy announcement — makes investors demand a **higher** risk premium, +1 point to 3.0%, even though the Fed hasn't moved and earnings haven't changed. The discount rate rises to 7.0%, the gap `(r − g)` widens from 2.0% to 3.0%, and the fair multiple falls by a third, from 20x toward ~13-14x in the simple model (less in a realistic one, but sharply). The index could fall **15-30%** *with no change in either the policy rate or earnings* — purely because the *price of risk* repriced. This is what a self-inflicted policy shock does: it widens the equity risk premium directly. It is the cleanest way to understand the UK gilt crisis and the sharpest leg of the April 2025 sell-off — *the market raised the risk premium on its own, and the multiple paid for it.*

## Common misconceptions

**"Earnings drive the stock market."** Over a *decade*, mostly true — earnings compound and the multiple mean-reverts, so long-run returns track earnings growth. But over the horizons that make headlines — a year, a Fed cycle — the *multiple* usually does the heavy lifting, and the multiple is a rate story. In 2022 earnings *rose* ~5% and the index fell ~19%; the multiple was the entire move. Confusing the two is the most common equity-policy error: people look for an earnings reason for a move that was pure discount rate.

**"Lower rates are always good for stocks."** Only if rates are falling for the *right* reason. Rates falling because inflation is beaten and the Fed is normalizing (a multiple tailwind with earnings intact) is the dream — that is 2024. Rates falling because the economy is collapsing into recession (a multiple tailwind, but the earnings leg is caving) can come with a *falling* market: the discount-rate channel says "buy," the demand channel says "sell," and in a hard recession the earnings collapse wins until the Fed gets ahead of it. *Always ask why rates are moving, not just which way.*

**"A strong economy is good for the stock market."** Only in a "good news is good news" regime. When the Fed is fighting inflation, a strong economy means higher-for-longer rates, a higher discount rate, and a compressed multiple — good news becomes bad news. The same payrolls number can rally stocks in 2024 and sink them in 2022. *The sign depends entirely on which leg the market is currently pricing.*

**"Tariffs are a tax on foreign countries, so they don't hurt our stocks."** A tariff is a tax collected at *our* border, largely paid by *our* importers and consumers. As the retailer worked example showed, a tariff on a large cost line can devastate the *earnings* leg of domestic, import-exposed companies — which is why U.S. retailers, automakers, and manufacturers led the April 2025 sell-off, not foreign ones. The cross-border [trade-war](/blog/trading/law-and-geopolitics/the-2018-19-us-china-trade-war) history is unambiguous on who pays.

**"The Fed controls the stock market."** The Fed controls *one leg of one channel* — the risk-free anchor under the multiple — powerfully but not completely. It does not set earnings (that is the cycle and fiscal policy), it does not set the equity risk premium (that is sentiment and credibility), and it does not set growth expectations (that is technology and demographics). The Fed put is real but conditional; in 2022 the Fed deliberately let stocks fall. *The Fed has a hand on the multiple, not a hand on the price.*

### The 40-year tailwind under every multiple

Zoom out and one fact dominates equity valuation for an entire generation. When Paul Volcker broke inflation, the ten-year Treasury yield peaked near **15.8%** in September 1981. Over the next forty years it ground all the way down to roughly **0.5%** in 2020 — a four-decade bond bull market. Run that through P/E ≈ 1/(r − g) and the implication is enormous: the *risk-free anchor* under every multiple fell by more than fifteen percentage points, which is a colossal, secular *tailwind* to the multiple leg. A large share of the long-run rise in equity valuations from the early 1980s to 2021 was not better businesses or faster growth — it was the discount rate falling for forty years. That is the deep version of "policy prices the multiple": the *trend* in rates set the *trend* in multiples across a generation. It also frames the worry of the 2020s: if the forty-year decline in rates is over — if we have entered a regime of structurally higher yields driven by big deficits, deglobalization, and tariffs — then the great multiple tailwind has become, at best, a flat road, and possibly a headwind. In that world, *earnings have to do more of the work*, because the multiple can no longer count on an ever-falling discount rate to lift it. Keeping the two legs separate is not just a tool for reading a single Fed meeting; it is how you read whether a whole decade will be a multiple decade or an earnings decade.

## Case studies: policy pricing equities, 2017-2025

### 2017 — the TCJA capitalized a tax cut into earnings

The cleanest earnings-channel case in modern history. The Tax Cuts and Jobs Act, signed December 2017, cut the statutory corporate rate from **35% to 21%** (the [tax-law-as-a-market-force](/blog/trading/law-and-geopolitics/tax-law-as-a-market-force) cross-link has the legal mechanics). As the worked example showed, this added roughly \$10-12 to S&P operating EPS — on the order of 8-10% to index earnings — purely from the tax line, with no change in the underlying businesses. The market did what markets do: it priced the cut in *advance*. The S&P rallied through late 2017 and into January 2018 as the EPS upgrade was capitalized. This is the earnings channel in its purest form — a single line of the tax code lifting the *E* in P/E × E, with the multiple roughly stable. Contrast with 2025's OBBBA, which *extended* those cuts rather than cutting fresh, and so removed a feared headwind rather than delivering a new tailwind — a much quieter market event for exactly that reason.

### 2020-21 vs 2022 — one multiple, expanded then crushed

The defining multiple case of the era, and the puzzle we opened with. In March 2020 the Fed cut to zero and launched unlimited QE; combined with the CARES Act's fiscal bazooka, this was *both* channels firing at once — the discount rate collapsed (multiple up) *and* fiscal transfers propped up demand (earnings cushioned). The S&P, which had fallen 33.9% in 33 days, doubled off its low by 2021 and traded above **22x** earnings — a multiple only zero rates and a flood of liquidity could justify. The [QE explainer](/blog/trading/finance/quantitative-easing-explained-printing-money) covers the liquidity mechanics.

Then the reversal. Through 2022 the Fed delivered **525 basis points** of hikes — the fastest cycle since Volcker — to fight 9% inflation. As the worked example computed, earnings *rose* ~5% (to ~\$219) while the index *fell* 19.4% (to ~3,840), because the multiple compressed from ~22.9x to ~17.5x. Bonds fell *with* stocks (the Bloomberg Agg lost ~13%, a 60/40 portfolio ~16%) because the same discount-rate channel that crushed equity multiples crushed bond prices — there was nowhere to hide. This is the discount-rate channel at full force: *the Fed didn't need to break earnings; it just made every dollar of earnings worth fewer dollars of price.* For the trader's positioning view of this exact period, see [how monetary policy moves stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors).

### 2025 — tariffs threaten earnings while cuts support the multiple

The richest recent case, because the two channels pointed in *opposite directions* on the same index — a live experiment in multiple-versus-earnings.

The S&P 500 set an all-time high of **6,144** on February 19, 2025. Then, on April 2 — "Liberation Day" — the administration announced a 10% universal tariff plus "reciprocal" rates on 57 partners, some extraordinary: China 34%, Vietnam 46%, the EU 20%, Japan 24%. This was a direct hit to the *earnings* leg of every import-exposed company in the index, and the *threat* of recession (a broader earnings hit) plus the policy chaos *also* compressed the multiple via a wider equity risk premium. The result was violent: the S&P fell about **12% in four trading days** and bottomed at **4,983** on April 8 — a **−18.9%** drawdown from the February high, the fastest correction of its kind in years.

Crucially, the *bond market did not behave like a safe haven*: the ten-year yield *spiked* about 50 basis points in the week of the crash (from ~3.99% to ~4.49%) rather than falling, and the dollar *fell* over 8% in the first half of the year — its worst start since 1973. When stocks, bonds, *and* the dollar all fall together, the market is not pricing a normal growth scare; it is repricing the *credibility* of U.S. policy itself — a risk-premium shock, the ugliest version of the discount-rate channel. (For the safe-haven mechanics and the war/credibility thread, this series' bond and currency posts go deeper.)

Then the resolution, and it is the multiple-versus-earnings story in miniature. On April 9 the administration paused the reciprocal rates for all but China — and the S&P ripped **+9.5% in a single day**, one of the largest one-day gains in history. That move was *almost pure risk-premium relief*: nothing about earnings had changed in 24 hours, but the *tail risk* to earnings was suddenly capped, so the multiple snapped back. Over the following months negotiations cut the rates further — Vietnam to 20%, the EU and Japan to 15%, China toward 30% — handing earnings *back* to import-exposed firms. Meanwhile the Fed, which had held all year to see whether tariffs hit inflation or jobs, began **cutting in September 2025** (three cuts, to a 3.50-3.75% range by December), supporting the *multiple*. With both legs improving — earnings threat receding, discount rate falling — the S&P erased its loss by mid-May and closed 2025 at **6,680**, an all-time high.

The 2025 case is the whole post in one year: a tariff shock to the *earnings* leg, a credibility shock to the *multiple* via the risk premium, then both reversing — negotiations restoring earnings, rate cuts restoring the multiple. *To read 2025 you had to hold the two legs apart and watch each one move.*

### 2022 — the UK gilt crisis: a credibility shock straight to the risk premium

A smaller but razor-sharp case that isolates the *risk-premium* leg. On September 23, 2022, the UK government announced £45 billion of *unfunded* tax cuts — fiscal stimulus with no plan to pay for it, into an inflation already raging. The market did not wait for the Bank of England. It repriced UK assets immediately and brutally: thirty-year gilt yields jumped about **130 basis points** in days, the pound fell to a record low near **\$1.035**, and a doom-loop in pension-fund hedging (LDI) forced the BoE into a £65 billion emergency bond-buying intervention on September 28. The government collapsed; the prime minister lasted **49 days**.

In our framework, this was a pure *self-inflicted widening of the risk premium*. Earnings hadn't changed and the policy *rate* hadn't yet moved — but the *credibility* of UK policy cracked, so investors demanded a far higher return to hold UK assets, which is exactly a higher discount rate via a wider risk premium. The multiple on UK assets compressed and the currency fell *together* — the same triple signal (stocks, bonds, currency all weak) that flashed in the US in April 2025. The lesson is the one the [equity-risk-premium worked example](#the-equity-risk-premium-and-the-fed-put) made arithmetically: *a government can raise the discount rate on its own stocks without the central bank touching a rate, simply by spending its credibility.* Markets disciplined a G7 government in days. It is the cleanest real-world proof that the risk-premium leg of the multiple is as real as the rate leg.

### Sector rotation, 2022-2025 — the clock in motion

Lay the rotation clock over those years. **2022** (hikes accelerating, peak-rate quadrant): energy was the lone positive S&P sector, up over 50%, its short-duration commodity cash flows immune to the discount-rate channel that sank long-duration tech (consumer discretionary and communications fell 30%+). **2023-24** (rates peaking, the easing-anticipation quadrant): the multiple channel reversed for mega-cap tech as the market sensed the rate top and priced an AI earnings boom — the "Magnificent Seven" drove the index while breadth stayed narrow. **2025** (cuts begin, but tariffs scramble earnings): rate-sensitive sectors found support from the falling funds rate, while the tariff shock sorted winners and losers by *import exposure* rather than by duration alone — a reminder that when the earnings channel dominates, the rotation map is drawn by *supply chains*, not just by rates. The lesson across the four years: *the same rotation clock governs leadership, but which channel is dominant — discount rate or demand — decides which axis of the clock matters most that year.*

## What it means for asset values: the playbook

Reduce the whole post to a procedure you can run on any policy headline.

1. **Identify which leg the policy touches.** Rates, QE/QT, forward guidance, real yields, the risk premium → the **multiple**. Corporate taxes, fiscal stimulus, tariffs, the demand cycle → the **earnings**. Some shocks (a deep recession, a credibility crisis) hit *both* — name which one dominates.
2. **Translate to direction and rough magnitude.** Multiple leg: a +1-point rise in the discount rate cuts a 25x fair P/E to ~20x — about −20% on price, more for long-duration growth, less for short-duration value. Earnings leg: a corporate-tax cut of ~14 points (35%→21%) lifts after-tax EPS by ~20% statutory / ~8-10% effective; a tariff on a large cost line can cut EPS by far more than its headline rate suggests.
3. **Sort the sectors by duration and business model.** Long-duration growth (tech) and rate-sensitive yield (utilities, REITs) suffer most when rates rise; banks and energy can *benefit*. Defensives (staples, health care, utilities) lead in slowdowns; cyclicals (discretionary, industrials) lead in recoveries. Place the cycle on the rotation clock and read off the leaders.
4. **Watch the regime, not just the data.** In a "good news is bad news" regime the multiple leg dominates and strong data sinks stocks; in a normal regime the earnings leg leads and strong data lifts them. The flip happens when the Fed's inflation fight is judged won.
5. **Respect the equity risk premium.** When the earnings yield sits barely above the bond yield (a thin ERP), the multiple is fragile — any rise in yields, or any credibility shock that widens the premium, reprices the whole index hard, regardless of earnings.

The same two-leg frame travels across borders, and an emerging market makes it even sharper. Vietnam's VN-Index is a vivid example: an export-led economy where roughly 29% of exports go to the United States, so a tariff lands directly on the *earnings* leg of its exporters, while the State Bank of Vietnam's credit-growth quota and policy rate move the *multiple* leg at home. In 2022 the VN-Index fell about **33%** as a domestic bond-and-property crackdown, SBV rate hikes, and global tightening hit both legs at once. In April 2025 the threat of a **46%** reciprocal tariff (later negotiated to 20%) was a direct earnings shock to Vietnamese exporters — the same Price = Multiple × Earnings logic, with the earnings leg exposed to *another country's* trade policy. The lesson generalizes: the more a market's earnings depend on external demand, the more its *earnings* leg is hostage to foreign policy, even as its *multiple* leg answers to its own central bank.

**The signals to watch:** the ten-year Treasury yield (the risk-free anchor under every multiple), the *direction of the next Fed move* and its *reason* (normalization vs recession-fighting), the corporate-tax and tariff legislative calendar (the earnings-leg events), and the earnings-yield-minus-bond-yield spread (the ERP cushion). **What would invalidate the read:** a multiple that keeps expanding while real yields rise — that means the market is pricing an earnings acceleration (an AI-style productivity boom) large enough to overpower the discount-rate channel, and you should switch your attention from the rate leg to the growth leg. The model is not "rates down, stocks up." The model is *Price = Multiple × Earnings, and policy moves each through its own door* — name the door, and the move stops being a mystery.

## Further reading & cross-links

**Within this series — Policy & Markets:**
- [The discount-rate channel: how rates reprice cash flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) — the engine behind the multiple leg, built from present value.
- [The fiscal toolkit: spending, taxes, and deficits](/blog/trading/policy-and-markets/the-fiscal-toolkit-spending-taxes-and-deficits) — the levers behind the earnings leg, including the TCJA and OBBBA.
- [How policy sets the bond market: the yield curve](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve) — the risk-free anchor that the equity discount rate is built on.

**Macro-trading — the positioning playbook (we link, we don't re-derive):**
- [How monetary policy moves stocks: discount rates and sectors](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors).
- [Asset rotation across the business-cycle quadrants](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants).

**Macro-correlations — the statistical relationships (the numbers):**
- [Inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) — the measured stock-bond and inflation-stock betas behind the regimes described here.
