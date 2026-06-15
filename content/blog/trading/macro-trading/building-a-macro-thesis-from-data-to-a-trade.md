---
title: "Building a Macro Thesis: From Data to a Tradeable View"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The capstone of the Macro for Traders series: how to turn a wall of economic data into one falsifiable view, express it through the cleanest instrument, size it to conviction, and pre-commit the print that kills it — the full end-to-end global-macro playbook from data to a trade you can defend and kill."
tags: ["macro", "macro-trading", "global-macro", "trading-thesis", "position-sizing", "invalidation", "monetary-policy", "regime-analysis", "risk-management", "trade-construction", "falsifiability", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A great macro trade is not a hunch. It is a *falsifiable thesis* — a one-sentence view of the world that implies a specific position — built by synthesizing six inputs (cycle, policy, liquidity, flows, valuation, positioning) into one coherent read, then expressed through the cleanest instrument, sized to conviction, and guarded by a pre-committed invalidation that tells you exactly when you are wrong.
>
> - A **thesis** is a view that *can be proven wrong by data*. "I feel bearish on bonds" is a mood; "the Fed will hike more than the market prices, so the 2-year yield rises" is a thesis — a specific CPI or jobs print would kill it.
> - **The cleanest instrument wins.** The exact same correct view ("Fed hikes more than priced") pays handsomely through the 2-year note and can *lose money* through stocks, because stocks bury the rate signal under earnings, buybacks, and sentiment.
> - **Size is the opinion, not the entry.** On a \$1,000,000 book, a high-conviction view (all six inputs aligned) might risk \$200,000; a low-conviction view risks \$50,000. The number you risk *is* how strongly you believe.
> - **The one rule to remember:** set the invalidation — the data print and the dollar stop that kill the trade — *before* you enter, never after you are losing. Being right and making money are two different skills, and the gap between them is the invalidation.

In the first quarter of 2022, the data was screaming. US consumer prices were rising at **7.5%** year-over-year in January and accelerating — by June they would hit **9.06%**, a 40-year high. And yet the Federal Reserve's policy rate, the most powerful lever in global finance, was pinned near zero: the target range topped out at just **0.25%**. A central bank whose entire job is price stability was sitting on a rate of essentially nothing while inflation ran at four times its 2% target. That gap — between a problem that was enormous and a policy response that was almost non-existent — was not a contradiction to be confused by. It was the trade.

A trader who saw it clearly could write the whole thing on an index card. Inflation is high and rising. The Fed is humiliatingly far behind and will be *forced* to hike hard — harder, and for longer, than the market currently believes. As it hikes, two things follow almost mechanically: bond prices fall (so short duration), and capital floods toward the higher-yielding dollar (so go long the dollar). That is a complete macro thesis: a read of the world, a view about what must happen, and a position that expresses it. Over the next eighteen months, the Fed hiked from 0.25% to **5.50%** — the fastest tightening in four decades — the 2-year Treasury yield rose from under 1% to over 5%, and the dollar index spiked to a two-decade high of nearly **115**. The thesis paid, and it paid because it was a thesis, not a hunch.

This is the capstone of the *Macro for Traders* series. Across the series we built the individual instruments of the macro orchestra: the [business cycle](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders), [interest rates](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable), [global liquidity](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide), [the flows](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging). This post is about *conducting* — how to take all of that and turn it into one trade you can defend and, just as importantly, one you know exactly how to kill. We build the whole craft from zero.

![Funnel from a wall of data narrowing to a regime read, a falsifiable view, one instrument, a conviction-based size, and a pre-set invalidation](/imgs/blogs/building-a-macro-thesis-from-data-to-a-trade-1.png)

## Foundations: what a macro thesis actually is

Before any data, we need to define the thing we are building, because the word "thesis" gets thrown around loosely. Most of what passes for a macro view in trading chat rooms is not a thesis at all. It is a vibe with a chart attached.

### A thesis is a falsifiable view of the world

A **macro thesis** is a *claim about how the world will behave* that is specific enough to be proven wrong. That last part is everything. The philosopher Karl Popper argued that what separates a scientific statement from a non-scientific one is **falsifiability**: a real claim makes a prediction that could fail. "The market will do what it does" cannot fail, so it says nothing. "The Fed will hike to at least 5% by mid-2023 because core inflation is above 4% and the labor market is tight" *can* fail — if core inflation collapses or unemployment spikes, the claim is dead. That is a thesis.

Concretely, a macro thesis has three parts welded together:

- **A read of the world** — what regime are we in? "Inflation is high and rising, growth is still positive, and the central bank is behind the curve." This is a description, built from data.
- **A view about what must happen next** — the prediction. "The Fed will be forced to hike more than the market currently prices in." This is the falsifiable claim.
- **An implied position** — the trade the view demands. "If rates rise more than priced, then short the front end of the bond curve and go long the dollar." This is the part you can actually put on.

If any of the three is missing, you do not have a thesis. A read with no view is just commentary. A view with no position is just an opinion you can't be paid for being right about. A position with no view is just a gamble. The discipline of macro is forcing all three to connect.

### Being right is not the same as making money

Here is the single most important idea in this entire post, and it is the one that takes traders years to internalize: **being right about the world and making money are two different skills.** You can have a perfectly correct macro view and lose money on it. You can have a half-right view and make a fortune. The reasons are mechanical, and we will return to each one:

- You can be right about the *direction* but wrong about the *timing* — and a position that's "early" is, in the market's accounting, simply wrong, because you get stopped out or run out of capital before the world catches up.
- You can be right about the world but express it through the *wrong instrument* — one so contaminated by other forces that the move you predicted never shows up in your P&L.
- You can be right but *sized so large* that a normal wobble forces you out before the payoff, or *sized so small* that being right doesn't matter.
- The market may have *already priced in* your brilliant insight, so even if it comes true, the price doesn't move — you were right about the world and wrong about the surprise.

A thesis that ignores these is half a thesis. The job is not to predict the world. The job is to *get paid for predicting the world*, and that requires building the prediction into a position that can actually capture the move and survive long enough to do it.

### The synthesis: six inputs, one view

So where does the read of the world come from? Not from one indicator. The amateur's mistake is to find a single signal — an inverted yield curve, a hot CPI print, a liquidity chart — and build a whole trade on it. The professional reads *several* inputs and synthesizes them into one coherent view, paying attention to which inputs agree and which conflict.

There are six inputs that, together, cover the macro world. Memorize them; this is the skeleton of every thesis you will ever build:

1. **The cycle** — where are we in the business cycle? Expanding, peaking, contracting, recovering? Read from GDP growth, the ISM/PMI surveys, employment.
2. **Policy** — what is the central bank doing and what will it do? Read from the policy rate, the [dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot), the Fed's reaction function.
3. **Liquidity** — is the tide of money rising or draining? Read from central-bank balance sheets, [net liquidity](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga), credit growth.
4. **Flows** — how is everyone positioned, and who is offside? Read from the [CoT report, dealer hedging](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging), fund flows.
5. **Valuation** — is the asset cheap or expensive *relative to the regime*? Read from real yields, equity risk premia, multiples.
6. **Positioning / consensus** — what does the market already believe and have priced in? Read from forward curves, survey expectations, the consensus narrative.

A thesis is the single view that *survives weighing all six together*. In early 2022, five of the six pointed the same way: the cycle was late, policy was absurdly behind, liquidity was about to drain (the Fed had announced it would stop buying bonds and start shrinking its balance sheet), flows showed bonds still over-owned, and the consensus had not yet accepted how high rates would go. Only valuation was ambiguous. When five of six align, you have a high-conviction thesis. When they conflict, you have either no trade or a small one.

![Six macro inputs — cycle, policy, liquidity, flows, valuation, positioning — feeding a weighing step that produces one falsifiable view](/imgs/blogs/building-a-macro-thesis-from-data-to-a-trade-2.png)

## The six inputs and how to weigh them

The six inputs are not equal. Their importance shifts with the regime, and learning *which one is in charge right now* is the core skill of macro. Let me walk through each, what it tells you, and when it dominates.

### Input 1 — the cycle (where growth is)

The business cycle is the slow oscillation of an economy between expansion and contraction. It sets the backdrop for everything: in an expansion, earnings rise and risk assets tend to grind higher; in a contraction, the opposite. You read the cycle from a handful of series — real GDP growth (the rear-view confirmation), the ISM manufacturing PMI (a leading gauge where 50 separates expansion from contraction), and the labor market (jobs, unemployment, wage growth).

The cycle matters most when you are trading the *medium-term direction* of equities and credit, and when you are trying to anticipate the central bank, because the cycle drives the Fed's reaction. The cycle is a weak input for a fast trade around a single data release and a powerful one for a six-month allocation. We mapped the full cycle-to-asset relationship in [asset rotation across the business cycle quadrants](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants); for thesis-building, the cycle tells you which way the wind is blowing before you try to sail.

### Input 2 — policy (where the central bank is)

Policy is, in the modern era, the master variable. The central bank sets the price of money — the risk-free rate that anchors the valuation of *every* asset on earth. When the Fed moves the policy rate, it moves the discount rate applied to all future cash flows, the cost of leverage, the attractiveness of cash versus risk, and the relative value of currencies. We unpacked the mechanics in [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [the Fed's reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).

For thesis-building, the key question is never just "where is the policy rate?" but "where is it *relative to where it should be*?" A policy rate that is far below what inflation and growth justify is a coiled spring — the central bank is behind the curve and must catch up, which is itself a tradeable view. In 2022, the entire thesis hinged on this gap: the rate was at 0.25% when a textbook reaction function would have put it at 4% or higher. Policy dominates the weighing whenever inflation or the central bank is the headline story — which, since 2021, has been most of the time.

### Input 3 — liquidity (the tide of money)

Liquidity is the total quantity of money-like balances sloshing through the financial system. When central banks expand their balance sheets, liquidity rises and floats *all* risk assets higher together — the "everything rally." When they drain it (by shrinking balance sheets or letting the Treasury rebuild its cash account), liquidity falls and the tide goes out under everything at once. We built this idea fully in [global liquidity, the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide).

Liquidity dominates the weighing when the *correlation across assets is high* — when stocks, crypto, and credit are all moving together, the cause is usually the tide, not any individual fundamental. In 2022, liquidity was draining (the Fed had begun quantitative tightening, and the reverse-repo facility was sucking cash out of the system), which reinforced the short-duration thesis: less money in the system, higher yields, lower asset prices.

### Input 4 — flows and positioning (who is offside)

Flows are about *how everyone else is positioned*. Markets are zero-sum at the margin; for you to make money on a move, someone has to be on the other side and forced to react. The Commitments of Traders report shows you whether speculators are crowded long or short a futures market; dealer hedging tells you whether option dealers will *amplify* or *dampen* a move. When a position is extremely crowded, it is fragile — a small adverse move forces a cascade of stop-outs that becomes a large move.

Flows are a *timing and risk* input more than a direction input. They rarely tell you *what* will happen, but they tell you *how violent* a move could be and *whether the easy money is gone*. A thesis where you are with the crowd is a thesis where the payoff may already be spent; a thesis where the crowd is offside and will be forced to chase you is a thesis with fuel behind it. We covered the mechanics in [following the flows](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging).

### Input 5 — valuation (cheap or rich for the regime)

Valuation asks whether an asset is priced attractively *given the regime*. A 4% bond yield is rich in a 1% inflation world and cheap in a 5% inflation world — valuation is always relative to the macro backdrop, never absolute. For bonds, the cleanest valuation gauge is the **real yield** (the nominal yield minus expected inflation); for equities, the earnings yield versus the risk-free rate. We argued in [real vs nominal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) that real yields are the master valuation signal across assets.

Valuation is a *weak short-term, strong long-term* input. Cheap things can get cheaper and rich things richer for a long time, so valuation rarely triggers a trade by itself. But it tells you where the *risk/reward* is asymmetric — buying something already cheap means a smaller downside if you're wrong. In thesis-building, valuation sets the size of the prize and the margin of safety, not the timing.

### Input 6 — consensus (what is already priced)

The last input is the most subtle and the one beginners most often forget: **what does the market already believe?** Markets are forward-looking discounting machines. By the time a recession is obvious in the data, stocks have usually already fallen to price it. The question is never "will X happen?" but "will X happen *differently from what's priced?*" Your edge is not in being right about the world; it is in being right about the world *in a way the market hasn't already accepted*.

You read consensus from forward curves (what hike path is priced into rates futures), from sell-side surveys, and from the dominant narrative. In early 2022, the consensus believed inflation was "transitory" and the Fed would hike gently. The thesis — that the Fed would hike *more* than priced — was an explicit bet against consensus, which is precisely why it paid. Consensus is the input that converts a correct view into a *profitable* view: you must be right *and* non-consensus.

#### Worked example: building the 2022 thesis from the data

Let us build the actual 2022 thesis the way a professional would, input by input, and watch it become a trade.

**Read the inputs.** It is March 2022. The data on the table:

- **Cycle:** Growth is still positive (2021 real GDP was +5.9%) but clearly past peak; the recovery is mature. *Late-cycle.*
- **Policy:** The Fed funds upper bound is **0.25%**. Core PCE inflation is **5.6%** (its peak), CPI is heading to **9.06%**. A reaction function that responds to inflation says the rate "should" be around 4-5%. The Fed is roughly **4-5 percentage points behind**. *Wildly behind the curve.*
- **Liquidity:** The Fed has announced the end of bond-buying and the start of balance-sheet shrinkage. The reverse-repo facility is draining cash. *Liquidity turning negative.*
- **Flows:** Bonds are still over-owned after a decade of falling yields; speculators are not yet short duration. *The crowd is offside, fuel for the move.*
- **Valuation:** The 10-year real yield is around **−0.14%** — investors are paying to lend to the government in real terms, an absurd valuation that only makes sense if inflation stays low. It won't. *Bonds are historically rich.*
- **Consensus:** The market prices the Fed funds rate peaking around 2.5-3% and believes inflation is transitory. *Consensus underestimates the hikes.*

**Weigh them.** Five of six point the same way; only valuation is "ambiguous" (and even it argues bonds are dangerously rich). Policy and liquidity are the dominant inputs in this regime. The synthesis is unambiguous.

**Write the view (falsifiable):** *The Fed will be forced to hike to at least 5% — more than the ~3% the market prices — because core inflation above 5% and a tight labor market leave no choice. As it does, the front end of the yield curve rises and the dollar strengthens.*

**Derive the position:** Higher front-end yields mean bond prices fall, so **short duration** (specifically, short the 2-year Treasury, the most policy-sensitive point on the curve). A hiking Fed pulls capital into dollars, so **long the dollar**.

The numbers that followed: the Fed hiked to **5.50%**, the 2-year yield rose from ~2.3% in March 2022 to over **5.05%** by October 2023, and the dollar index spiked to nearly **115**. The intuition: when five of six macro inputs align against a complacent consensus, you don't have a guess — you have a thesis with the wind at your back.

![CPI inflation rising to a 9.06% peak while the Fed funds upper bound is forced from 0.25% up to 5.50% on a shared timeline](/imgs/blogs/building-a-macro-thesis-from-data-to-a-trade-3.png)

The chart above *is* the 2022 thesis. The amber line is the problem — inflation towering to 9.06%. The blue staircase is the response — a policy rate forced to climb in the fastest hiking cycle in four decades. The shaded gap between them, where inflation ran far above the policy rate, is the window in which the short-duration, long-dollar trade did its work. A thesis is, in the end, a story about a gap between what *is* and what *must be* — and the gap here was visible to anyone reading the data instead of the narrative.

## From view to trade: choosing the instrument

You have a view. Now comes the step that separates the trader who *understands* macro from the trader who *makes money* from it: choosing the instrument. The same view can be expressed through a dozen different assets, and they are wildly different in how cleanly they capture the move.

### The principle: pick the cleanest beta to your view

Every asset is moved by many forces at once. A stock moves on earnings, on sentiment, on the sector, on buybacks, on the broad market — *and* a little bit on interest rates. A 2-year Treasury note moves on almost nothing *except* the expected path of the policy rate. If your view is about the policy rate, the 2-year note is an almost-pure expression of it, while a stock buries that same view under a mountain of noise.

The principle: **express your view through the instrument whose price is most directly driven by the thing you have a view about, and least driven by everything else.** We want the highest *signal-to-noise ratio* — the cleanest "beta" (sensitivity) to the specific thesis. A correct view expressed through a noisy instrument can fail to pay even when you are right, because the noise swamps the signal.

### Ranking instruments for the 2022 view

Take the 2022 view — "the Fed hikes more than priced" — and consider the ways to express it:

- **The 2-year Treasury note (short it).** The 2-year yield is, almost by definition, the market's expectation of the average policy rate over the next two years. If you think the Fed hikes more than priced, the 2-year yield *must* rise and its price *must* fall. There is almost no other force moving it. This is the cleanest possible expression.
- **Short-term interest rate (STIR) futures — SOFR or fed funds futures (sell them).** These price the *exact* expected path of the policy rate, contract by contract, and they are leveraged (low margin), so they give the most P&L per dollar of capital for a pure rate view. Slightly more contaminated by term premium and thin liquidity in far contracts, but extremely clean and sharp.
- **The US dollar (buy it, e.g. via DXY or long USD/JPY).** A hawkish Fed pulls capital toward the higher-yielding dollar — a clean *second leg* of the same thesis. But the dollar is a *relative* price: it also depends on what the ECB, BoJ, and others do. A good expression, but one that introduces a second variable. We covered the drivers in [what moves exchange rates](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry).
- **Gold (short it).** Higher real yields hurt gold (it pays no coupon, so it competes with real yields). But gold is *also* driven by fear, the dollar, and central-bank buying — several forces that can fight your rate view. Noisy.
- **The S&P 500 (short it).** Higher rates do weaken stocks (a higher discount rate compresses valuations). But stocks are dominated by earnings, sentiment, buybacks, and the AI narrative. In 2022 the rate view was *correct* and stocks did fall — but the relationship was weak and slow, and you could easily have been right about rates and chopped up trying to express it through equities. The worst vehicle for this particular view.

![Matrix ranking five instruments for the same Fed-hikes-more view by how directly each tracks the view and how much other noise moves it](/imgs/blogs/building-a-macro-thesis-from-data-to-a-trade-4.png)

The lesson is stark: **the same correct view ranges from "cleanest trade of the year" to "coin flip" depending purely on the instrument.** This is why two traders with identical, correct macro views can have opposite P&L. The one who shorted the 2-year compounded; the one who shorted stocks got whipsawed.

#### Worked example: instrument selection — the 2-year versus stocks

Let us make the cost of a bad instrument choice concrete with the 2022 view.

**Express it via the 2-year note.** In March 2022 the 2-year yield was about **2.28%**; by October 2023 it reached **5.05%** — a rise of **2.77 percentage points**. A 2-year note has a duration of roughly 2, meaning its price falls about 2% for every 1 percentage point rise in yield. So the price fell roughly **2 × 2.77 ≈ 5.5%**. If you shorted \$10,000,000 face of the 2-year, you captured roughly **\$550,000** of price decline, almost entirely from the thing you had a view about. The trade did what your thesis predicted, cleanly, with no surprises.

**Express it via stocks.** The S&P 500 fell from about **4,500** in March 2022 to a low near **3,500** in October 2022 — a 22% drop — *but then rallied back above 4,300 by mid-2023* even as the Fed kept hiking, because the AI narrative and resilient earnings took over. If your thesis was "rates rise, so short stocks," you were *right about rates* and yet the S&P was *higher* eighteen months later. The rate signal was real but it was a small, slow input swamped by earnings and sentiment. You could have been stopped out repeatedly in the 2023 rally despite your view being correct.

Same view, same eighteen months. The 2-year made you \$550,000 cleanly; the short-stock expression of the *identical correct view* likely chopped you to pieces. The intuition: choosing the instrument is not a detail after the analysis — it *is* half the analysis, because a view only pays through a vehicle that actually tracks it.

### When the "clean" instrument isn't available

Sometimes the cleanest expression is impractical — you can't trade the exact instrument, or its liquidity is poor, or its cost of carry is punitive. Then you build a *proxy* and pay attention to the *basis risk*: the gap between what your proxy tracks and what your view is about. A retail trader who can't easily short the 2-year might use an inverse Treasury ETF or a rate futures contract, accepting some tracking error. The discipline is to *know* the contamination you are accepting, not to pretend it isn't there. Every step away from the cleanest beta is a step toward "being right and not getting paid."

## Conviction, sizing, and risk

You have a view and an instrument. Now: how big? This is where most of the actual money is made or lost, and where the discipline of a thesis pays off most concretely.

### Size is the real expression of your opinion

A trader's true opinion is not what they say; it is *how much they risk*. You can describe a view as "high conviction" all day, but if you put on a tiny position, you don't really believe it, and if you put on a huge position on a flimsy view, you are gambling. **The position size is the quantitative statement of your conviction.** A coherent process maps conviction directly to risk.

The cleanest way to do this is to think in terms of **risk per trade as a fraction of the book**, not in terms of notional. What matters is not "how many dollars of bonds did I short" but "how much do I lose if I'm wrong" — the distance from entry to the invalidation, times the position size. We size the *risk*, then back into the notional.

### Mapping conviction to risk on a fixed book

Suppose you run a **\$1,000,000** book. Define a few conviction tiers and the risk each is allowed:

- **High conviction** (five or six of the six inputs aligned, clear non-consensus edge): risk up to **\$200,000** — 20% of the book — though most professionals would cap a single thesis lower. This is your best idea of the year.
- **Medium conviction** (four inputs aligned, some ambiguity): risk **\$100,000** — 10%.
- **Low conviction** (the inputs are mixed but the risk/reward is attractive): risk **\$50,000** — 5%.

The risk number is the *maximum loss if the invalidation triggers* — not the notional. From the risk budget and the distance to your stop, you compute the position size: position = risk budget ÷ (distance to invalidation). A wide stop forces a small position; a tight stop allows a large one. This single discipline — sizing from a risk budget rather than from a notional — is what keeps one wrong thesis from ending your career.

#### Worked example: sizing to conviction on a \$1,000,000 book

Two versions of the same 2022 short-duration trade, differing only in conviction.

**High-conviction version.** It's March 2022; all six inputs scream the same thing and the consensus is complacent. You assign **high conviction** and a **\$200,000** risk budget (20% of the \$1,000,000 book). You short the 2-year, planning to exit if the thesis is invalidated by a sharp move *against* you — say a 0.40 percentage-point fall in the 2-year yield from your entry (which would itself signal the market no longer believes in more hikes). A 0.40 point yield move on the 2-year is roughly a 0.8% price move. To risk \$200,000 at a 0.8% adverse price move, your position size is \$200,000 ÷ 0.008 = **\$25,000,000** face of the 2-year. That's a big position — justified *only* because conviction is high and the stop is defined.

**Low-conviction version.** Same trade, but it's a regime where the inputs are mixed and you're less sure. You assign **low conviction** and a **\$50,000** risk budget (5%). With the same 0.8% adverse move as your invalidation, your position is \$50,000 ÷ 0.008 = **\$6,250,000** face — one quarter the size. If the trade goes wrong, you lose \$50,000 (a 5% drawdown, easily survivable). If it goes right, you make one quarter of what the high-conviction version makes — which is *correct*, because you were less sure.

The discipline is that you decided the \$200,000 versus \$50,000 *from the alignment of the inputs*, not from how exciting the chart looked. The intuition: sizing is not a feeling applied after entry — it is the arithmetic translation of how many of your six inputs agree, and it is the single biggest determinant of whether your thesis survives long enough to pay.

![Two columns contrasting a low-conviction view risking fifty thousand dollars with a high-conviction view risking two hundred thousand dollars, both with a pre-set stop](/imgs/blogs/building-a-macro-thesis-from-data-to-a-trade-6.png)

### Risk is about survival, not just sizing

Two more risk principles round out the sizing discipline:

- **Correlation across positions.** If you put on the 2022 thesis through *both* short 2-year *and* long dollar, those are not two independent bets — they are two expressions of the *same* view, and they will lose together if the view is wrong. Size them as one thesis, not two, or you have secretly doubled your bet. The biggest blow-ups come from traders who thought they were diversified across five positions that were really one macro bet.
- **The asymmetry of drawdowns.** A 20% loss requires a 25% gain to recover; a 50% loss requires a 100% gain. Large losses are mathematically punishing, which is why even a high-conviction thesis caps its risk well short of the whole book. Surviving to put on the *next* thesis is worth more than maximizing any single one.

### Optionality: when to express a view with options instead

There is one more tool in the expression kit that deserves its own note, because it changes the risk math entirely: **options**. A macro view expressed through a futures or cash position has symmetric risk — you make money if you are right and lose money if you are wrong, roughly in proportion to how far the market moves. A view expressed through an option (buying a put to bet on a fall, a call to bet on a rise) has *asymmetric* risk: your loss is capped at the premium you paid, while your upside stays open-ended. For a high-conviction-but-uncertain-timing macro thesis, that asymmetry can be worth paying for.

The 2022 short-duration thesis is a good illustration. You could express "rates go up a lot" by shorting bond futures (symmetric — great if right, brutal if the Fed blinks) or by buying puts on long bonds (asymmetric — you pay a premium, but if the Fed blinks you lose only the premium, not a runaway amount). The trade-off is that options cost money: the premium is a drag if the move is slow or never comes, and you have to be right on *timing*, not just direction, because options expire. The rule of thumb is to use the linear instrument when you are confident on both direction and timing and want maximum capital efficiency, and to use options when your conviction on *direction* is high but the *timing or the tail risk* is uncertain and you want to cap the cost of being early or wrong. The best macro traders mix both — a core linear position sized to conviction, with options used to define the worst case or to add cheap leverage to the highest-conviction tail.

## The invalidation: pre-committing to being wrong

Here is the part that separates a thesis from a hope. A thesis is falsifiable, which means *there exists a specific observation that would prove it wrong*. The **invalidation** is you writing that observation down, in advance, and committing to act on it.

### Why the invalidation must come before the entry

The human mind is extraordinarily good at rationalizing a losing position. Once you are down money and emotionally committed, every adverse data point becomes "noise," every bounce becomes "the turn," and the stop you vaguely intended drifts ever lower. The only defense is to **decide what would prove you wrong before you have any money on the line** — when you are calm, objective, and not yet attached.

The invalidation has two forms, and a complete thesis specifies both:

- **The data invalidation** — the *fundamental* observation that kills the view. For the 2022 thesis, the whole edifice rested on inflation staying high enough to force the Fed's hand. So the data invalidation is: *a clear, sustained cooling in core inflation.* A core PCE print falling decisively below, say, 3% and trending down would mean the Fed no longer *must* hike — the thesis is dead, regardless of price.
- **The price invalidation (the stop)** — the *market* observation that kills the position. Even before the data confirms you're wrong, the price can tell you the market has stopped believing your view. A 2-year yield that *falls* hard despite hot inflation is the market pricing out hikes — the thesis is being rejected. You set a dollar stop tied to this level.

The data invalidation protects you from being *wrong*; the price invalidation protects you from being *early* and running out of capital. A thesis with only a price stop gets shaken out of correct views by noise; a thesis with only a data invalidation can ride a correct view straight into a margin call. You need both.

### What the invalidation is NOT

The invalidation is not "I'll exit when I feel like the trade isn't working." That is not a rule; it is the absence of one. It is not "if the loss gets uncomfortable." It is a *specific, pre-written, observable condition* — a number on a screen or a print in a release — that requires no judgment in the moment to recognize. The whole point is to remove in-the-moment judgment, because in-the-moment judgment, on a losing position, is precisely when judgment is worst.

#### Worked example: the invalidation that kills the 2022 thesis

Make the invalidation concrete for the short-duration trade.

**The data invalidation.** The thesis is "the Fed must hike more than priced because core inflation is too high." The killing print: *core PCE inflation falls below 3% and prints lower for two consecutive months.* Why that threshold? Because below 3% and falling, the Fed gains the room to *stop* hiking — the "must" in your thesis evaporates. (In reality core PCE didn't reach 3.0% until December 2023, well after the trade had worked; the data invalidation never triggered during the trade's life, which is *why it kept working*.)

**The price invalidation (the dollar stop).** You're short \$25,000,000 face of the 2-year (the high-conviction sizing from earlier), risking \$200,000. Your entry is at a 2-year yield of 2.28%. The price stop: *exit if the 2-year yield falls 0.40 points to 1.88%*, which on a 2-duration note is roughly a 0.8% price move against you, i.e. about \$200,000 — your full risk budget. If the yield hits 1.88%, you are out, no debate, because the market is pricing *out* the hikes your whole thesis depends on.

Now watch the trade actually unfold: the 2-year yield never fell to 1.88%. It went the *other* way — to 2.96% by June, 4.22% by September, 5.05% a year later. The price invalidation was never hit, the data invalidation was never hit, so you held a correct thesis to its full payoff. The intuition: the invalidation is not a prediction that you'll be wrong — it is a pre-written instruction for *if* you are, so that being wrong costs you \$200,000 and a shrug instead of your whole book and your confidence.

### Killing a thesis cleanly

When an invalidation triggers, the discipline is to *kill the trade with no story*. No "let me give it one more day." No "averaging down to improve my entry." No reframing the thesis to keep the position. The view was falsifiable; the falsifying observation arrived; the view is dead; the position closes. This is emotionally hard and it is the entire game. The traders who survive decades are not the ones with the best views — they are the ones who kill their wrong views fastest and cheapest. A thesis you cannot bring yourself to kill was never a thesis; it was an identity.

## Common misconceptions

### Misconception 1: "A thesis is a prediction of what will happen."

A thesis is *not* a prediction that you are confident is correct. It is a *falsifiable structure*: a view, an expression, a size, and an invalidation. The prediction is only one component, and it's allowed to be wrong — that's what the invalidation is for. Treating a thesis as a prediction makes you defend it when it fails; treating it as a falsifiable structure makes you *test* it and kill it cleanly. In 2022, plenty of people *predicted* high inflation and still lost money because they had a prediction, not a thesis — no clean instrument, no sizing discipline, no invalidation. The prediction was the easy part.

### Misconception 2: "More data makes a better thesis."

There is a powerful instinct to gather more indicators, more charts, more confirmation before acting. But past a point, more data makes a *worse* thesis, not a better one. Each additional indicator you can find that "confirms" your view is usually correlated with the ones you already have — it feels like independent confirmation but adds nothing, while *increasing your conviction falsely*. Worse, the search for confirmation is biased: you stop looking when you've found enough "yes," not when you've honestly weighed the "no." A great thesis rests on *six well-chosen inputs honestly weighed*, including the ones that disagree — not on forty cherry-picked charts that all say the thing you already wanted to believe. The 2022 thesis was strong because of *one honest input that disagreed* (valuation was ambiguous), which kept the sizing sane.

### Misconception 3: "Being right means making money."

We have said it throughout, and it is the misconception that costs the most. You can be perfectly right about the macro world and lose money — by being early, by choosing a contaminated instrument, by sizing wrong, or by trading a view the market already priced. In 2022, "rates will rise" was *correct* and a short-S&P expression of it could still have lost you money through the 2023 rally. The world being as you predicted is *necessary but not sufficient* for profit. The other half — clean instrument, right size, surviving the timing, and a real edge over consensus — is what converts a correct view into P&L. Internalizing this is the difference between a smart commentator and a paid trader.

### Misconception 4: "A bigger position shows more conviction."

Conviction is not loudness. A bigger position than your invalidation and book can support isn't conviction — it's a death wish, because a normal wobble forces you out of a correct view before it pays. *True* conviction is sized so that you can *survive the noise* and hold the view to its payoff. The high-conviction 2022 trade was \$200,000 of risk on a \$1,000,000 book — large, but bounded, with a pre-set stop. A trader who put 80% of the book on the "obvious" trade would have been right about the world and stopped out by the first violent counter-trend rally. Conviction is the *quality* of the edge times the *survivability* of the size, not the size alone.

### Misconception 5: "If the thesis is good, I don't need an invalidation."

The better you think your thesis is, the *more* you need a written invalidation — because high conviction is exactly when you'll rationalize a losing position the hardest. The invalidation is not an admission of doubt; it is the structural feature that *makes* the view a thesis rather than a faith. A view you refuse to attach an invalidation to is a view you've decided is unfalsifiable, which means it's not a thesis at all.

### Write the thesis down — the discipline that separates pros

One practice underlies everything above and is worth stating plainly: **write the thesis down before you put it on.** A macro thesis you hold only in your head is infinitely flexible — and that flexibility is the enemy, because it lets you quietly move the goalposts when the trade goes against you, rationalize a loser into a "long-term hold," and forget what you originally believed. A thesis written down — the view, the six inputs that support it, the instrument, the size, and above all the invalidation — is a contract with your past self. When the market moves, you check it against the written thesis, not against your hopes. Did the invalidation trigger? Then you are out, no debate. Did the supporting inputs change? Then you re-underwrite from scratch. The single habit that most separates disciplined macro traders from gamblers is not better forecasting; it is the boring act of writing the thesis and its kill-switch down before risking a dollar, and then actually obeying what you wrote. Everything in this post — the synthesis, the sizing, the invalidation — only works if it is committed to paper before the emotion of an open position can corrupt it.

## How it shows up in real markets

### The 2022 short-duration, long-dollar thesis

We have traced this thesis throughout, so let us see it whole, as a single coherent trade from data to payoff. The read: late cycle, policy 4-5 points behind, liquidity draining, bonds over-owned and historically rich, consensus underestimating the hikes. The view: the Fed hikes to 5%+, more than priced. The instrument: short the 2-year (cleanest), long the dollar (clean second leg). The size: high conviction, \$200,000 risk on a \$1,000,000 book. The invalidation: core PCE below 3% and falling, or the 2-year yield breaking down to 1.88%.

The result: the Fed hiked from 0.25% to **5.50%**, the 2-year yield rose from ~2.3% to **5.05%**, the dollar index spiked to nearly **115**, and neither invalidation triggered until the trade had done its work. This is what a complete thesis looks like end to end — not a lucky call, but a structure that captured a real macro move cleanly and could have been *defended at every step* and *killed at a defined point* if wrong.

### The 2023 disinflation pivot

The harder, more instructive case is knowing when a winning thesis is *over*. By late 2023, the 2022 thesis was decaying. Core PCE, which had peaked at **5.6%** in February 2022, had fallen to **3.0%** by December 2023 and kept dropping. CPI had cooled from **9.06%** to the mid-3s. The data that had powered the short-duration thesis — towering, sticky inflation — was disappearing. The Fed's last hike was July 2023, to 5.50%; from there it held, and the market began pricing *cuts*.

A trader running the framework would have seen the original thesis *approaching its data invalidation*. Core PCE below 3% and falling was precisely the condition that killed the "Fed must hike more" view. The disciplined move was not to ride the short-duration trade into the ground but to recognize that the regime was turning — from *hiking* to *holding* to eventually *cutting* — and to build a *new* thesis for the new regime. The disinflation pivot thesis: with inflation falling toward target and the Fed done hiking, the *front end peaks and the next move is down*, so the trade flips from short duration to *long* duration (buying bonds to capture falling yields as cuts get priced). We worked the cut-cycle mechanics in [terminal rate and rate-cut cycles](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path).

The lesson is not which way the bond trade went in 2024 (it was choppy — yields rose again in late 2024 even as cuts began, a reminder that the next thesis is never a gift). The lesson is *the discipline of letting the data retire the old thesis*. The same six-input framework that built the 2022 short told you, in late 2023, that its core fuel was gone. A trader who fell in love with the winning short-duration call and held it through 2024 gave back gains; a trader who ran the framework continuously rotated to a new view when the data invalidated the old one. **The framework is not a single trade. It is a process you re-run every time the macro picture changes.**

![Three-panel dashboard of real GDP growth, CPI inflation, and ISM manufacturing PMI as the regime snapshot](/imgs/blogs/building-a-macro-thesis-from-data-to-a-trade-5.png)

The dashboard above is the regime snapshot you'd read to know *which* thesis the world currently supports. Growth (left) tells you the cycle phase; inflation (center) tells you the price-pressure regime and how far it sits from the Fed's 2% target; the ISM PMI (right) is the leading gauge of whether the manufacturing economy is expanding or contracting (50 is the line). Read together, in March 2022 they said *late cycle, inflation far above target, momentum still positive* — the regime that supported the short-duration thesis. By late 2023 the inflation panel was collapsing toward target, the signal that the regime — and the thesis — had to change. Note the right edge of the inflation panel: the **2026 re-acceleration to 4.25%** is the live story today, the data that would *rebuild* an inflation-fighting thesis if it persists. The dashboard is never finished; the thesis is never permanent.

## How to trade it: the full macro playbook

Here is the entire craft, distilled into the loop you run every time the macro picture moves. This is the capstone of the series — everything we built, in one repeatable process from a wall of data to a trade you can defend and kill.

**Step 1 — Read the regime from the six inputs.** Pull up the dashboard: cycle (GDP, PMI, jobs), policy (rate, dot plot, reaction function), liquidity (balance sheets, net liquidity), flows (CoT, dealer hedging, fund flows), valuation (real yields, risk premia), and positioning/consensus (forward curves, the dominant narrative). Write the regime in one sentence — "late cycle, inflation far above target, policy behind, liquidity draining, consensus complacent." If you can't write it in one sentence, you don't understand it yet.

**Step 2 — Write the view as a falsifiable claim.** Convert the regime read into a specific, killable prediction. Not "I'm bearish bonds" but "the Fed hikes to 5%+, more than the ~3% priced, because core inflation above 5% and a tight labor market force it." The test: can you state the exact observation that would prove this wrong? If not, it's a mood, not a thesis — go back to step 1.

**Step 3 — Pick the cleanest instrument.** Of all the ways to express the view, choose the one whose price is most directly driven by the thing you have a view about and least driven by everything else. A policy-rate view → the 2-year note or STIR futures, not stocks. Rank the candidates by signal-to-noise; accept basis risk knowingly if the cleanest vehicle is impractical. *Half your edge is here.*

**Step 4 — Size to conviction.** Count how many of the six inputs align. Five or six aligned against a complacent consensus → high conviction → risk up to \$200,000 on a \$1,000,000 book. Four aligned → medium → \$100,000. Mixed but attractive risk/reward → low → \$50,000. Size the *risk*, then back into the notional from the distance to your invalidation. The number you risk *is* your opinion, stated quantitatively.

**Step 5 — Set the invalidation before you enter.** Write down both kills: the *data invalidation* (the fundamental print that retires the view — "core PCE below 3% and falling") and the *price invalidation* (the dollar stop that closes the position if the market rejects the view — "2-year yield down to 1.88%, ~\$200,000 loss"). Decide these while you are calm and uncommitted. No invalidation, no trade.

**Step 6 — Let the data decide.** Put the trade on and then *watch the invalidations, not the P&L swings*. If the view confirms, hold or add within your risk budget. If a kill print hits, close with no story — no "one more day," no averaging down, no reframing. The view was falsifiable; it was falsified; it's done. The discipline of step 6 is the whole game.

**Step 7 — Re-run on the turn.** When the regime changes — when the data that powered the thesis starts to disappear, as inflation did in late 2023 — retire the old view and start again at step 1. Being right *once* is luck. Running the loop *continuously*, letting each regime build and each turn retire its own thesis, is the edge that compounds across decades. The framework is the asset, not any single trade.

That is the global-macro playbook, end to end. A thesis is not a hunch you defend; it is a falsifiable view you build from the six inputs, express through the cleanest instrument, size to your honest conviction, guard with a pre-committed invalidation, and kill the moment the data says you're wrong. Master the loop and you stop seeing a hundred unrelated headlines and start seeing one move — the way the trader who shorted the 2-year in March 2022 saw the entire year coming as a single, defensible, killable trade.

![The seven-step macro playbook loop from reading the regime to letting the data confirm or kill the trade and re-running on the turn](/imgs/blogs/building-a-macro-thesis-from-data-to-a-trade-7.png)

## Further reading & cross-links

This post is the capstone; each step draws on a dedicated post in the series:

- **The regime read** — [The business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) and [asset rotation across the business-cycle quadrants](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants) build the cycle input.
- **Policy** — [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [the Fed's reaction function and the dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) build the policy input.
- **Liquidity** — [Global liquidity: the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide) and [the central-bank balance sheet, net liquidity, reserves, RRP, TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga) build the liquidity input.
- **Flows** — [Following the flows: positioning, CoT, dealer hedging](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging) builds the flows input.
- **Valuation** — [Real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) and [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) build the valuation input.
- **Expressing the view** — [What moves exchange rates: rates, flows, carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) and [terminal rate and rate-cut cycles](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path) for instrument selection in rates and FX.
- **Foundations** — [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) and [the Volcker 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) for the historical playbook of fighting inflation with policy.
