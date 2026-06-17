---
title: "The Research Workflow in Production: From Idea to Live Signal"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The real end-to-end pipeline a trading signal travels from a researcher's hypothesis to live capital — and the gate it must pass at every stage, the failure mode that kills most ideas there, and roughly how many survive. The thesis: production research is a brutal filter, and the discipline lives in the gates, not the idea."
tags: ["quant-careers", "quant-finance", "careers", "quant-research", "research-pipeline", "backtesting", "purged-cv", "capacity", "signal-decay", "slippage", "paper-trading", "alpha-research"]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — In production, a trading idea is not a flash of insight; it is a candidate that has to pass a long chain of gates — hypothesis, data, signal, backtest, risk and capacity, paper trade, small live, ramp, monitor — and the discipline is in the gates, not the idea.
>
> - **Each stage exists to pass exactly one gate, and each gate has a characteristic way of killing ideas.** A vague hypothesis dies for lack of a mechanism; a leaked dataset fakes a whole edge; an overfit signal collapses out-of-sample; a real edge that costs too much to trade or has no capacity dies anyway. The skill is building the gates and being willing to lose your idea at any of them.
> - **The funnel is brutal.** A workable rough shape: **~100 hypotheses** become **~15** that survive a careful backtest, **~6** that survive costs and capacity, **~3** that track in paper, **~2** that match P&L live at small size, and **~1** still earning a year later. Most of your work is being wrong cheaply.
> - **A great backtest is not a great strategy.** The backtest quietly assumes mid-price fills, zero latency, unlimited size, and clean data; live trading charges you the spread, slippage, latency, partial fills, and market impact. The gap between the two is where signals that "passed everything" die in week three.
> - **The one number to remember:** a signal with a small-size Sharpe near **2.0** can have a *useful capacity* of only about **20M USD** and a *hard capacity* near **80M USD** — push more money through it and market impact drives the net edge to zero. Capacity, not the backtest, is what decides whether an idea is worth funding.

It is a Thursday in the third week of a new live signal, and **Wei** is watching it die in real time.

Six months ago this was the best idea he had produced all year. He framed a clean hypothesis about a cross-sectional pattern in mid-cap equities, pulled point-in-time data, built a tidy signal with only three parameters, and ran it through the firm's backtester with the full discipline his lead had drilled into him: purged cross-validation, a held-out final year he touched exactly once, realistic costs. It survived. The out-of-sample Sharpe came in near **1.6**, the information coefficient was a believable **0.04**, turnover was manageable, and the capacity analysis said it could hold a few tens of millions before the edge thinned. It passed peer review. It went into a paper-trading book, where for two months its simulated P&L tracked the backtest almost exactly. The committee approved a small live allocation — **1%** of the eventual target size — and it went live with real money.

And now, in week three, the live P&L is sitting *below* the paper book, and the gap is widening a little every day. Nothing is broken. There is no bug. The signal is doing exactly what it was designed to do. It is just that the backtest, and even the paper book, were quietly charging him the wrong price for every trade. The spread is wider than the model assumed on the names he trades most. His orders, small as they are, are nudging the price against him. The data feed lands a few hundred milliseconds late, and in those milliseconds the easy fills are gone. None of these costs existed in the simulation. All of them exist now. Wei does the arithmetic and realizes that at the size the firm wants to deploy, the net edge is not 1.6 Sharpe; it might be 0.4, and it might be zero.

This is not a failure. This is the **pipeline working**. Figure 1 is the chain Wei's idea traveled — and the thing to notice is that it is not a straight line from "smart idea" to "rich." It is a sequence of gates, each one designed to kill the idea, with a loop at the bottom that sends the dead ones back to the start. The discipline of production research is not having good ideas. It is building the gates honestly and being willing to lose your idea at any of them — including, like Wei, at the very last one, with real money already on the line.

![A pipeline diagram showing the full production research chain from hypothesis through data, signal build, backtest, risk and capacity, paper trade, small live, ramp, monitor, and decay-and-retire, with a loop edge sending dead ideas back to the start](/imgs/blogs/the-research-workflow-in-production-from-idea-to-live-signal-1.png)

This post is the systems-level companion to [a day in the life of a quant researcher](/blog/trading/quant-careers/a-day-in-the-life-quant-researcher). That post is about the rhythm of one researcher's day; this one is about the *machine* the work feeds into — the production pipeline a signal must survive to touch live capital, the gate it faces at every stage, the specific way most ideas die there, and roughly how many survive. The technical craft of each stage already lives in the quant-research deep-dives, and I will point you to them rather than re-derive the math. The thesis here is simpler and harder: **production research is a brutal filter, and the discipline is in the gates, not the idea.**

## Foundations: what "production" means for a signal

Before any of the stages make sense, you need a small, precise vocabulary. None of it is difficult, but every term is load-bearing, and most people's picture of "having a profitable signal" is wrong precisely because they are missing one or two of these ideas. Assume you are brilliant but have never worked inside a trading firm — we will build each concept from zero.

**A signal, also called alpha.** A signal is a number you compute for each asset at each point in time that you believe predicts that asset's future return. The bet is that assets with a higher signal value will, *on average*, slightly outperform those with a lower value — not every time, just on average, with a tiny edge repeated across thousands of positions. "Alpha" is the finance word for return that is not explained by simply taking market risk; a signal is your attempt to manufacture it. The craft of building one — normalizing, neutralizing, controlling decay — is the subject of [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).

**A backtest.** A backtest is a simulation. You take your signal, apply a rule that turns signal values into positions ("go long the top decile, short the bottom decile, rebalance daily"), and replay history to see what profit and loss — "P&L" — the rule *would have* produced. The output is an equity curve and a handful of summary statistics. The backtest is the researcher's laboratory, and like any laboratory it can be contaminated in ways that turn the result into a beautiful lie. Doing it correctly is hard enough to deserve its own deep-dive: [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research).

**The numbers that summarize a signal.** Three statistics carry most of the evaluation, and you should know them cold:

- **Information coefficient (IC).** The IC is the correlation between your signal's value today and the asset's realized return over the next period, measured across all assets. It answers "does a higher signal actually predict a higher return?" Real cross-sectional equity signals live in a *low* range: a per-period IC of **0.02 to 0.05** is genuinely useful, and anything above about 0.1 should make you suspect you have leaked future information into the signal.
- **Sharpe ratio.** The Sharpe ratio is the strategy's average excess return divided by its volatility, annualized — how much return per unit of risk. A Sharpe of **1.0** is a respectable single signal, **2.0** is excellent, and a *combined book* of many signals at a top fund might run 2 to 4. A *single* raw signal showing a backtest Sharpe above ~2 is almost always overfit.
- **Turnover.** Turnover is how much of your book you trade per period. It matters because *every trade costs money*, so a signal that looks great before costs can be dead after them if it trades too fast. The full evaluation toolkit is in [evaluating alpha signals: IC, Sharpe, turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research).

**A gate.** This is the central concept of the whole post. A gate is a *pass-or-fail check* that an idea must clear to advance to the next stage. The hypothesis gate asks "is this falsifiable, with a real mechanism?" The backtest gate asks "does the edge survive honest validation, net of costs?" The capacity gate asks "is the edge big enough at fundable size?" Each gate is a place where the idea can die — and a well-run research process is, more than anything, a *well-designed set of gates*. The naive researcher has one gate ("does the backtest look good?") and ships everything that passes it. The disciplined researcher has nine, and most of their ideas die at one.

**Capacity.** Capacity is the amount of capital you can deploy in a signal before your own trading destroys its edge. This is the single most misunderstood idea in retail and student-level quant thinking. When you buy, you push the price up; when you sell, you push it down. At tiny size this *market impact* is negligible. As you scale, it grows — roughly with the square root of the size you trade — and it eats directly into your edge. A signal with a gorgeous Sharpe at 1M USD can have *zero* net edge at 100M USD. The backtest, run on historical prices as if your trades didn't move them, is blind to this. Capacity is what turns "a real edge" into "a fundable strategy," and the two are not the same.

**Decay.** A signal's edge is not permanent. Markets adapt: as more capital chases the same pattern, the inefficiency that created the edge gets traded away. An edge that ran at 0.04 IC for two years thins to 0.02, then 0.01, then noise. Decay is not a bug — it is the market doing its job. The production question is not "will this signal decay?" (it will) but "how fast, and how do we detect it before it costs us more than it made?"

**The difference between a backtest and live capital.** Hold this one firmly, because it is the post's recurring lesson. A backtest is a *story about the past told with optimistic assumptions*: it assumes you filled at the mid-price, instantly, at any size, on perfectly clean data. Live trading is *the present charging you real prices*: you cross the spread or miss the trade, your fills lag your signal, your size moves the market, your data arrives late and gets revised, and you pay fees, financing, and borrow costs the simulation never modeled. The gap between the backtest and the live result is not noise — it is structural, it is mostly in one direction (against you), and it is exactly where signals that "passed everything" go to die.

**The funnel, the gates, and the loop.** Three facts about the *shape* of the work close out the foundations. First, production research is a **funnel**: many hypotheses enter, almost all die, a tiny fraction reach live capital — we will do the funnel math as the first worked example. Second, the funnel is made of **gates**: each stage is there to enforce one specific check, and the value of the process is in how honestly those checks are built. Third, research is a **loop**: a dead idea does not end the day; it returns you to the top with one more thing learned, and even a *live* signal eventually loops back when it decays and is retired. Funnel, gates, loop — hold those three, and the rest of this post is detail.

#### Worked example: the idea-mortality funnel

Let us make the funnel concrete with Wei's year. The numbers are illustrative round estimates — the *shape* is the real lesson, not the exact counts.

Wei logs **100 hypotheses** over the year. Many are sparks from reading a paper, a few come from the desk, some from re-examining decaying signals. Of those, about half — **50** — pass a quick smell test: there is a plausible mechanism, the data exists, and they are worth the cost of coding up. The other 50 die on the whiteboard, which is exactly where you want ideas to die, because it is free.

Of the 50 he codes and backtests, roughly **30%** survive a *careful* backtest with purged cross-validation — that is **15**. The other 35 either showed no edge, or showed an edge that evaporated out-of-sample once he stopped letting himself peek at the test data. This gate, [overfitting, purged CV, and the deflated Sharpe](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research), is where the largest *absolute* number of ideas die.

Of the 15 that survive the backtest *gross*, about **40%** still have a positive edge after realistic transaction costs and at a fundable capacity — **6**. Costs and capacity together are the most underrated killers: an idea can be genuinely predictive and still be worthless because it trades too fast or holds too little money to matter.

Of those 6, roughly half — **3** — track their backtest once they hit a paper-trading book on live data. Of those 3, about two-thirds — **2** — match their simulated P&L when they go live at small size with real fills. And of those 2, about half — **1** — are still earning a year later; the other decayed and was retired.

So Wei's year: **100 hypotheses → ~1 durable live signal.** That is a 1% conversion from idea to lasting edge, and it is *normal*. Figure 2 is this funnel. Notice the shape: the cliff is in the middle, at the backtest and cost gates, and the survivors at the right are a sliver.

![A bar chart showing the idea-mortality funnel, with steep drop-off bars from one hundred logged hypotheses through fifty that pass a smell test, fifteen surviving backtest, six surviving costs and capacity, three surviving paper trading, two surviving the small-live ramp, and one still live after a year](/imgs/blogs/the-research-workflow-in-production-from-idea-to-live-signal-2.png)

*The intuition: you are not paid to find signals — you are paid to run a filter that kills 99 ideas cheaply so the firm only ever risks real money on the one that survives.*

## Stage 1 — Hypothesis: the gate of the falsifiable why

The pipeline starts with an idea, but an idea is not yet a candidate. The first stage's job is to turn a vibe into a *hypothesis* — a testable, falsifiable claim with a mechanism behind it.

**What happens.** The researcher writes down, before touching the data, exactly what they expect to find and why. "Momentum works" is not a hypothesis; it is a vibe. "In US mid-cap equities, stocks in the top decile of trailing 12-month return, skipping the most recent month, outperform the bottom decile over the next 20 trading days, because institutional rebalancing is slow" *is* a hypothesis — it specifies the universe, the rule, the horizon, and a mechanism. The mechanism is the part beginners skip and the part that matters most.

**The gate.** Can you state, in advance, a single falsifiable claim *and* a reason the edge should exist? If you cannot say what result would make you abandon the idea, you do not have a hypothesis — you have a hope, and hope is what gets you fooled into mistaking noise for signal. The "why" matters because the market is enormous and randomness is generous: with enough patterns tested, *something* will look predictive by pure chance. A mechanism is your prior. An edge with no plausible story behind it is far more likely to be a coincidence that will not survive contact with fresh data.

**The failure mode.** Ideas die here from being *vibes without mechanisms* — too vague to test, or "data-mined" backwards from a chart the researcher already stared at. An idea born from looking at the data is *born contaminated*: you have already let the data shape the hypothesis, so an in-sample test of it proves nothing.

**Survival.** Roughly **half** of logged ideas clear this gate. The rest die on the whiteboard — the cheapest, healthiest place for an idea to die.

This is also where the connection to the rest of your toolkit begins: the statistical discipline of framing a claim you can actually reject is the same discipline tested in [the research case and take-home](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it), and it draws directly on the methods in [statistics and ML for alpha research](/blog/trading/quant-careers/statistics-and-ml-for-alpha-research-the-researchers-toolkit).

## Stage 2 — Data: the gate of point-in-time truth

Once a hypothesis exists, the longest and least glamorous stage begins. In practice, data work is the majority of a researcher's time, and it is where the most dangerous failures hide.

**What happens.** The researcher sources, cleans, and aligns the data the signal needs: prices, volumes, fundamentals, alternative data, corporate actions. They handle splits and dividends, survivorship (does the dataset include companies that went bankrupt or delisted, or only the ones that survived?), and — most importantly — *point-in-time* correctness: every value must be tagged with the date it was *actually available*, not the date it *refers to*. A company's Q1 revenue is not knowable on the last day of Q1; it is knowable weeks later when the company reports it. Using the earlier date is a time machine.

**The gate.** Is every input point-in-time, survivorship-corrected, and free of leakage? Leakage — *look-ahead bias* — is using information in your signal that you would not have had at the moment of the trade. Using a closing price to decide a trade you place at the open. Using a revenue figure on the date it refers to rather than the date it was published. Using a "current constituents" index list to backtest a strategy ten years ago.

**The failure mode.** A single look-ahead bug **manufactures a whole edge out of nothing**. This is the deadliest failure in the entire pipeline because it does not announce itself — it produces a *gorgeous* backtest. A strategy trading on tomorrow's newspaper will show a Sharpe of 5 and an equity curve that looks like a stairway to heaven, and a researcher who has fallen in love with the result will defend it instead of hunting the bug. Most of the data work in production exists precisely to prevent this one failure.

**Survival.** This stage does not so much filter ideas as *gate their integrity* — most ideas pass through it, but the ones that pass with an undetected leak will die spectacularly later (or worse, lose money live). The discipline here is what separates a researcher from a tourist, and it is covered in the data-side methods of [statistics and ML for alpha research](/blog/trading/quant-careers/statistics-and-ml-for-alpha-research-the-researchers-toolkit).

## Stage 3 — Signal construction: the gate of parsimony

With clean data in hand, the researcher builds the actual signal — the number, per asset, per period, that encodes the hypothesis.

**What happens.** The raw idea becomes a computed score. The researcher decides how to normalize it (so a few extreme values don't dominate), whether to *neutralize* it against known factors like sector, size, or market beta (so you are betting on your edge and not accidentally on "tech went up"), and how to handle the signal's decay horizon (how fast the prediction goes stale). The output is a clean cross-sectional panel of signal values aligned to the return panel.

**The gate.** Is the signal *simple* — few parameters, stable construction — and does it hold up when you vary those choices? The enemy here is the number of knobs. Every parameter you tune is a chance to fit noise. A signal with one lookback window and one threshold is far more trustworthy than one with seven interacting parameters, each "optimized."

**The failure mode.** Ideas die here from **overfitting to a lucky parameter**. The researcher tries 40 variations, picks the best-looking one, and reports its backtest as if it were the only thing they tried. That best-of-40 result is contaminated by the *search itself*: the more configurations you try, the better the best one looks by chance alone. This is why the deflated Sharpe ratio exists — it docks your reported Sharpe for the number of trials you ran. A signal that needs a precise parameter to work is usually a signal that does not work.

**Survival.** Most signals that reach this stage produce *a* number; the filter is really the next gate, where parsimony gets tested against held-out data. But the discipline you bring here — keeping the signal simple, counting your knobs honestly — determines whether you pass that next gate or fail it.

## Stage 4 — Backtest with purged CV and costs: the gate of the honest edge

This is the stage everyone pictures when they think of quant research, and it is where the single largest number of ideas die.

**What happens.** The researcher runs the signal through history with a position rule, but does it *honestly*. That means three things that the naive backtest skips. First, **purged cross-validation**: instead of one in-sample fit and one out-of-sample check, you split the timeline into folds, train on some and test on others, and *purge* the data around each test fold so that information from the training period cannot leak into the test through overlapping return windows. Second, **realistic costs**: every simulated trade pays an estimated spread, commission, and slippage. Third, **the deflated Sharpe**: the reported Sharpe is discounted for how many ideas you tried before this one.

**The gate.** Does the edge survive purged cross-validation, *net of costs*, after deflating for the number of trials? A signal that shows Sharpe 2.6 in-sample and 0.3 out-of-sample fails. A signal that shows a clean 1.2 across every fold, holds up net of costs, and stays positive after deflation, passes.

**The failure mode.** Two ways to die here, and they are the most common deaths in the whole pipeline. The first is the **out-of-sample collapse**: the beautiful in-sample Sharpe evaporates on data the researcher did not get to peek at. The second is **costs eating the edge**: a signal genuinely predictive on paper trades too fast, so once you charge it realistic transaction costs, the net edge is gone. A signal that needs to trade 200% of its book per day to capture a 0.03 IC will pay more in costs than the IC is worth.

**Survival.** Of the ideas that reach a careful backtest, only about **30%** survive — and of those, only about **40%** survive the cost-and-capacity test that follows. Together these two gates kill roughly **85%** of everything that makes it this far. The full treatment of doing this right is [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) and [overfitting, purged CV, and the deflated Sharpe](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research).

## Stage 5 — Risk and capacity: the gate of fundable size

A signal can pass an honest backtest and still be worthless. This is the stage that decides whether a real edge is a *fundable* edge, and it is the most underrated gate in the entire pipeline.

**What happens.** The researcher asks two questions the backtest cannot answer. First, the *risk* question: what are the drawdowns, the tail losses, the factor exposures, the correlation to the firm's existing book? A signal that is great alone but perfectly correlated with five signals the firm already runs adds no diversification and may add concentrated risk. Second, the *capacity* question: how much capital can this signal hold before the researcher's own trading destroys the edge?

**The gate.** Is the edge large enough, at a size worth funding, at acceptable risk and acceptable correlation to the existing book? A firm running billions does not care about a signal that tops out at 2M USD of capacity, no matter how high its small-size Sharpe — the edge it generates is rounding error against the firm's costs of having a researcher, a slot, and a risk budget tied up in it.

**The failure mode.** Ideas die here because the **edge is real but too small to fund** — it has no capacity, or it is too correlated with what the firm already trades to add anything. This is the heartbreak gate: the signal *works*, the researcher did everything right, and it still gets killed because the market cannot absorb enough of it to matter.

**Survival.** Of the signals that survive the backtest gross, this gate (combined with costs) leaves roughly **40%**. The capacity analysis is mathematical, and it is worth doing explicitly.

#### Worked example: the capacity curve

Wei's surviving signal shows a Sharpe near **2.0** at tiny size — a few hundred thousand dollars, where his trades do not move any prices. The question is: what happens as the firm scales it up?

The driver is **market impact**. When Wei deploys more capital, he has to push larger orders into the market, and larger orders move the price against him before they fill. A standard rule of thumb is that impact cost grows roughly with the *square root* of the size you trade relative to the market's daily volume — double your order and the cost per share rises by about 40%, not 100%, but it never stops rising. So the net Sharpe is the small-size ceiling minus an impact drag that grows like the square root of deployed capital.

Suppose the small-size ceiling is Sharpe **2.0**, and Wei calibrates the impact so that the net edge hits zero at about **80M USD** of deployed capital. A simple model that captures this is `net Sharpe = 2.0 − 0.224 × sqrt(C)`, where `C` is the capital deployed in millions and the constant `0.224` is chosen so the curve passes through zero at `C = 80` (because `0.224 × sqrt(80) ≈ 2.0`). Plug in numbers:

- At **\$1M**: `2.0 − 0.224 × sqrt(1) = 1.78`. Almost the full edge — at this size his trades are a drop in the ocean.
- At **\$20M**: `2.0 − 0.224 × sqrt(20) = 1.0`. Still a genuinely useful signal — this is the *useful capacity*, the point below which the signal carries its own weight against the firm's threshold.
- At **\$45M**: `2.0 − 0.224 × sqrt(45) = 0.5`. Marginal — the firm might run a little here, but the edge is thinning fast and the marginal dollar is barely earning.
- At **\$80M**: `2.0 − 0.224 × sqrt(80) = 0`. The *hard capacity* — past here, market impact eats the entire edge and you are trading for free, or for a loss.

So the same signal is "excellent" at \$1M and "worthless" at \$80M, and *nothing changed about the signal* — only the size. Figure 3 is this capacity curve. The decision the firm actually makes is not "is this a good signal?" but "how much can we deploy before the Sharpe drops below our threshold?" If their threshold is Sharpe 1.0, this signal's real capacity is \$20M, and its contribution to the firm is the P&L that \$20M at Sharpe 1.0 generates — meaningful, but a fraction of what the backtest's headline Sharpe might have suggested. A second, sharper way to read the curve: the *marginal* dollar's Sharpe falls even faster than the average. The average Sharpe over the first \$20M is around 1.4, but the Sharpe on the last dollar deployed at \$20M is exactly 1.0, and at \$40M the marginal dollar is already earning a Sharpe near 0.6. Smart desks size to where the *marginal* edge is still worth the risk, not where the average still looks good — which is why two firms can run the identical signal and one deploys \$10M while the other deploys \$30M, depending entirely on their hurdle.

![A declining curve showing realized net Sharpe ratio on the vertical axis against capital deployed in millions of dollars on the horizontal axis, starting near two at tiny size and falling to zero near eighty million, with the useful-capacity line at twenty million where Sharpe equals one and the hard-capacity line at eighty million where the edge hits zero](/imgs/blogs/the-research-workflow-in-production-from-idea-to-live-signal-3.png)

*The intuition: a signal's value is not its peak Sharpe — it is the area under the capacity curve up to the size the market will let you trade, and that ceiling is usually far lower than beginners assume.*

## Stage 6 — Paper / simulated trading: the gate of live-data tracking

The backtest used historical data the researcher could study at leisure. The next gate tests the signal against *live, forward-moving data* — but without risking real money yet.

**What happens.** The signal is wired into a paper-trading (or "sim") system that consumes the real-time data feed, computes the signal each period as the data arrives, generates the orders the strategy *would* send, and records the simulated fills and P&L. Crucially, this is *forward* data — data that arrives after the backtest ended, that the researcher has never seen and could not have fit to. It is the cleanest out-of-sample test there is, and it also surfaces the first reality-checks: is the data feed actually delivering the inputs on time? Does the production code compute the same signal the research code did? Are there real-world quirks — halts, holidays, bad ticks — that the historical data smoothed over?

**The gate.** Does the live-data simulated performance track the backtest? If the backtest said Sharpe 1.6 and IC 0.04, the paper book over a couple of months should be in the same neighborhood. A paper book that diverges immediately from the backtest is a red flag that something — leakage, a code mismatch, a regime change — was hiding.

**The failure mode.** Ideas die here when **fills and latency break the backtest's assumptions** before any real money is even at stake, or when the live signal simply does not behave like the simulated one — exposing a leak or an implementation bug the historical backtest could not catch. Paper trading is the first place the *engineering* reality of production touches the *research* idea, and the seam between research code and production code is where bugs hide.

**Survival.** Of the signals that pass risk and capacity, roughly **half** track well enough in paper to advance. The rest reveal a divergence that sends them back to the loop. This stage is also where the difference between a backtest and live capital, Figure 4, first becomes visible — even though no real money is on the line, the paper book runs on real fills and real latency.

![A before-and-after comparison contrasting backtest assumptions on the left, including mid-price fills, instant execution, unlimited size, flat costs, and clean data, with live-trading reality on the right, including crossing the spread, signal-to-fill lag, market impact at size, full costs, and late or revised data](/imgs/blogs/the-research-workflow-in-production-from-idea-to-live-signal-4.png)

## Stage 7 — Live and ramp: the gate of small-size P&L

Now real money goes in. This is the most psychologically demanding gate, because for the first time the firm's actual capital is exposed, and the researcher's name is attached to the P&L.

**What happens.** The signal goes live at a *deliberately small* size — often around **1%** of the eventual target allocation. The point is to risk almost nothing while learning the things only real money teaches: actual fills (not simulated ones), actual market impact, actual borrow costs on the shorts, actual behavior of the strategy when the firm's *other* signals are trading the same names. If the small-live P&L tracks the paper book, the signal earns a *ramp*: capital is added in steps, and at each step the realized P&L is checked against the capacity curve's prediction. If it keeps tracking, it ramps toward full size. If it diverges, the ramp halts.

**The gate.** Does the small-size live P&L match the simulation, and does the realized P&L keep matching the capacity curve as size is added? The ramp is a sequence of small gates, not one big one — each increment of capital is a fresh out-of-sample test of the capacity model.

**The failure mode.** This is where Wei's signal is dying. The failure mode is **market impact at size killing the net edge** — the small-size fills were fine, but as capital ramps, the impact the backtest never modeled shows up and drags the net P&L below the sim. Sometimes the divergence appears at the very first dollar of real money (a sign the backtest's cost model was simply wrong); sometimes it appears only at size (a sign the capacity estimate was too optimistic). Either way, the live-versus-paper gap is the signature.

**Survival.** Of the signals that track in paper, roughly **two-thirds** match their P&L at small live size and earn a meaningful ramp. The rest, like Wei's, reveal that the costs were underestimated and get pulled back or retired.

#### Worked example: paper-versus-live divergence from costs

Let us put numbers on Wei's week-three problem. His paper book and his live book trade the *exact same signal* and generate the *exact same target positions*. The only difference is what each one is charged per trade.

In the paper simulation, the modeled gross daily return is about **+0.060%** of capital — consistent with the backtest's Sharpe near 2 at the sim's assumed (low) costs. But live, the real costs are higher: the spread he actually crosses, the slippage between his signal price and his fill price, the small market impact of his orders, and the fees and borrow he pays add up to roughly **0.042%** of capital *per day* that the sim ignored. So the live book earns about **+0.018%** per day — the same gross edge, minus the costs the simulation never charged.

Compound these forward. Starting both books at an index of 100 on go-live day, each one grows by a factor of `(1 + daily return)` every trading day:

- The paper book grows at `1.00060` per day. After **60 trading days** it sits at `100 × 1.00060^60 ≈ 103.7` — up about 3.7%.
- The live book grows at `1.00018` per day. After 60 days it sits at `100 × 1.00018^60 ≈ 101.1` — up only about 1.1%.

The gap is about **2.6%** of starting capital in three months — and it widens every single day, because it compounds. The two lines start together on day one and fan apart, exactly as in Figure 5. Nothing is broken; the signal is doing what it was built to do. It is just that *the costs the backtest waved away are the whole difference between a 2-Sharpe paper book and a barely-positive live book*.

Now scale Wei's predicament up. At 1% of target size the dollar loss from this drag is trivial — a rounding error on a tiny book. But the *rate* is what matters, because it does not improve as you add capital; if anything, the impact term grows it. Project the same 0.042%-per-day drag onto the full target allocation the firm had in mind, and the annualized cost is enormous — roughly `0.042% × 250 trading days ≈ 10%` of capital a year handed straight to the market in spread, slippage, and impact. A signal whose *gross* edge is 15% a year nets 5%; a signal whose gross edge is 11% nets 1% and is not worth the risk. This is precisely why the small-live gate exists and why it is sequenced *before* the ramp: it is far cheaper to discover this divergence at 1% of target size, where the dollars lost are negligible, than at full size with the firm's capital committed and a 10%-a-year cost leak running live.

![A two-line chart showing cumulative equity indexed to one hundred at go-live, with a dashed paper or simulated book line climbing steadily and a solid live book line climbing more slowly, the shaded gap between them representing cost, slippage, and impact drag widening over sixty trading days](/imgs/blogs/the-research-workflow-in-production-from-idea-to-live-signal-5.png)

*The intuition: paper and live trade the same signal, so any persistent gap between them is pure cost — and because it compounds, a small daily drag the backtest ignored becomes the entire reason the strategy fails.*

## Stage 8 — Monitor, decay, and retire: the gate of persistent edge

A signal that reaches full size has not "won." It has earned the right to be *watched*, because every edge eventually fades, and the final discipline of production research is killing your own live signal at the right moment.

**What happens.** Once live and ramped, the signal is monitored continuously. The desk tracks its realized IC, its Sharpe over trailing windows, its drawdown against the historical worst case, its slippage versus the model, and its correlation to the rest of the book. They watch for two things: a *break* (something suddenly wrong — a data feed change, a regime shift, a bug) and a *decay* (the slow thinning of the edge as the market adapts and more capital crowds the trade).

**The gate.** Does the edge persist, and does the signal stay within its risk budget? This gate runs forever, re-evaluated every period. The hard part is distinguishing decay from noise: a single bad month is not decay, it is variance. You need a *trailing* measure — say, a rolling three-month IC — and a pre-committed *kill threshold* so that the decision to retire is a rule, not an emotional flinch.

**The failure mode.** Signals die here by **decaying below the kill threshold** — the edge thins until, net of costs, it no longer pays for the risk and the capital it ties up. The subtle failure is *not* retiring on time: a researcher attached to their once-great signal keeps it running on hope, paying costs and risk for an edge that is gone, occupying a slot a fresh idea could use. The discipline of killing a live signal is the same discipline as killing a backtest, covered in the research-write-up craft — being willing to declare your own work dead.

**Survival.** Of the signals that ramp to full size, roughly **half** are still earning a year later. The rest decay and are retired — and retirement is not failure; it is the loop closing, freeing the slot and the risk budget for the next candidate.

#### Worked example: detecting and retiring a decaying signal

Wei has a *different* signal that did make it to full size and earned well for over a year. Now the desk needs to decide when to retire it. They monitor its information coefficient month by month against a pre-committed **kill threshold of IC = 0.015** — the level below which, net of costs at its capacity, the signal no longer pays for itself.

At launch the signal ran an IC near **0.040**. Month-to-month the realized IC is noisy — it bounces between 0.02 and 0.04 in the early months purely from variance — so the desk does not watch the raw monthly number. They watch the **trailing three-month average**, which smooths the noise and reveals the trend. Over the following year and a half, that trailing average drifts down: 0.038, 0.034, 0.030, then through the 0.020s, as more capital crowds the trade and the inefficiency thins.

The kill rule is pre-committed and unemotional: *retire the signal the first time the trailing three-month IC closes below 0.015 and shows no sign of recovering.* In Wei's case that happens around **month 16**. Before that point, individual months had dipped below 0.015, but the trailing average had not — that is the rule protecting the desk from retiring on a single noisy month. After month 16, the trailing IC sits durably under the line, the net-of-cost P&L turns marginal, and the signal is pulled. Figure 6 shows the decay and the kill point.

The discipline is in the *pre-commitment*. Had the desk waited "just one more quarter" each time, they would have paid costs and risk for a dead edge and kept a slot occupied. The kill threshold, set in advance, turns an emotional decision into an arithmetic one.

![A bar-and-line chart of signal decay showing monthly realized information coefficient as gray bars and a trailing three-month average as a blue line, both drifting down from about 0.04 at launch, with a red dashed kill threshold at 0.015 and a vertical kill-point line at month sixteen where the trailing line crosses below the threshold](/imgs/blogs/the-research-workflow-in-production-from-idea-to-live-signal-6.png)

*The intuition: you do not retire a signal because it had a bad month — you retire it when a smoothed, trend-revealing measure crosses a line you committed to in advance, so the decision is arithmetic instead of emotional.*

## How to think about the whole pipeline

Step back from the individual stages and the pipeline reveals a few organizing truths.

**The discipline is in the gates, not the idea.** A beginner thinks the scarce resource is the idea — the clever insight nobody else has had. In production, ideas are *cheap*; Wei logs a hundred a year. The scarce resource is a *trustworthy filter*. Two researchers with the same hundred ideas will get wildly different results depending on how honestly they build their gates: the one with a leaky backtest and no capacity analysis will ship three garbage signals that lose money; the one with purged CV, realistic costs, a capacity model, and a kill threshold will ship one signal that earns. The value you add is not the idea. It is the *killing*.

**Each gate filters for a different failure, and the failures are not interchangeable.** The hypothesis gate catches vibes. The data gate catches leakage. The backtest gate catches overfitting and cost-blindness. The capacity gate catches edges too small to fund. The paper gate catches implementation and tracking bugs. The live gate catches impact at size. The monitor gate catches decay. A pipeline missing any one of these gates has a *specific* blind spot, and that blind spot is exactly where its losses will come from. Figure 7 lays out every stage against its gate and its characteristic failure mode — it is the map worth keeping on your wall.

![A matrix figure with eight rows for the pipeline stages from hypothesis through monitor, and two columns showing the gate each stage must pass and the characteristic way ideas die at that stage, with the gates colored by their role and every failure cell marked in red](/imgs/blogs/the-research-workflow-in-production-from-idea-to-live-signal-7.png)

**The costs move in one direction, and they compound.** Every gap between the backtest and reality — spread, slippage, latency, impact, fees, borrow, decay — runs *against* you, and several of them compound over time. This asymmetry is why production results are *systematically* worse than backtests, not just noisier. A researcher who treats the backtest as the answer instead of the optimistic ceiling will be disappointed every single time. The professional builds the costs in early and treats the backtest Sharpe as an *upper bound* the live result will fall below.

**Capacity, not Sharpe, sizes the prize.** The headline Sharpe sells the idea; the capacity curve decides what it is worth. A 1.2-Sharpe signal that holds 200M USD is worth far more to a large firm than a 2.5-Sharpe signal that tops out at 5M. This is why the same backtest excites a student and bores a multi-billion-dollar fund — they are reading different numbers off the same chart.

**The gates are sequenced cheapest-first, on purpose.** Notice the order: the cheapest gate to fail — the whiteboard hypothesis — comes first, and the most expensive — live capital at full ramp — comes last. This is not an accident; it is the economic logic of the whole pipeline. Each gate is more expensive to reach than the one before it: the hypothesis costs an hour, the backtest costs days of compute and a researcher's week, the paper book costs infrastructure and weeks of waiting, and the live ramp costs *real money and real risk*. A well-designed pipeline kills as many ideas as possible at the cheapest gates, so that by the time an idea reaches the expensive gates it has already survived the cheap filters. A pipeline that lets weak ideas slip through the early gates pays for that laziness later, in dollars, exactly where the failures hurt most. The discipline of being ruthless early — killing a vibe on the whiteboard, hunting a leak in the data before you fall in love with the backtest — is what makes the late gates affordable.

**The loop never ends.** Even your best live signal eventually decays and is retired, sending you back to the top of the funnel. There is no "done." The career, like the pipeline, is a probabilistic edge played repeatedly: you cannot win every idea, so you build a process that wins *on average across many ideas*, kills the losers cheaply, sizes the winners by capacity, and recycles the slot when an edge fades. That EV-under-uncertainty mindset — applied to ideas, to costs, to capital, to your own attachment — is the whole job, and it is the same mindset the firm hired you for.

## Common misconceptions

**"A great backtest means a great strategy."** No. A great backtest means a great *story about the past told with optimistic assumptions*. Between the backtest and a great strategy stand the cost gate, the capacity gate, the paper gate, the live gate, and the decay gate — and most beautiful backtests die at one of them. The most beautiful backtests of all are often the ones with a leakage bug, because trading on tomorrow's data produces flawless equity curves. A high backtest Sharpe should raise your suspicion, not your confidence.

**"Capacity is unlimited — if it works at 1M, just scale it to 100M."** This is the single most expensive misconception in quant. Your own trading moves prices, and the impact grows with size, so a signal's edge *shrinks as you deploy more capital* — that is the entire message of the capacity curve. A signal can be excellent at 1M and worthless at 80M with nothing about the signal changed. Capacity is a property of the *market's depth*, not your cleverness, and it caps the prize far below where beginners expect.

**"Live performance equals backtest performance."** It does not, and the gap is structural, not random. Live trading charges you the spread, slippage, latency, partial fills, impact, fees, and borrow that the backtest waved away — and these costs run against you and compound. A signal that backtests at Sharpe 2 routinely lives at Sharpe 0.5 to 1 once it pays real-world prices. The professional expects the live result to land *below* the backtest and is pleasantly surprised when it lands close.

**"Signals last forever — once you find alpha, you're set."** Every edge decays. Markets adapt: as more capital chases the same inefficiency, the edge thins and eventually disappears. The question is never "will it decay?" but "how fast, and will I detect it before it costs me?" A live signal is not a possession; it is a melting asset that must be monitored against a kill threshold and retired on schedule, not on hope.

**"The smartest researcher wins."** The most *disciplined* researcher wins. Raw cleverness generates ideas, but ideas are cheap. The researcher who builds honest gates, counts their trials, models their costs and capacity, and kills their own darlings on a pre-committed rule will out-earn a more brilliant colleague who falls in love with beautiful backtests. The edge is in the process, not the IQ.

## How it plays out in the real world

At a systematic fund like Two Sigma or D. E. Shaw, or in a research pod inside Citadel, this pipeline is not a metaphor — it is *infrastructure*. There is a literal research platform that enforces the gates: point-in-time data stores that make leakage hard, backtesting frameworks with purged cross-validation built in, automated cost and capacity models, paper-trading harnesses on the live feed, and a deployment process that ramps capital in pre-set increments with automated kill switches. WorldQuant's [alpha-factory model](/blog/trading/quant-careers/worldquant-and-the-alpha-factory-model) industrializes this further: thousands of researchers feed candidate signals into a shared pipeline that does the filtering at scale, and the platform itself owns the gates.

The recruiting consequence is direct. The [research case and take-home](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it) is a miniature of this pipeline: the firm hands you data and a few days, and watches whether you frame a falsifiable hypothesis, avoid leakage, validate out-of-sample, account for costs, and — the thing they care about most — whether you can *kill your own idea* honestly. A candidate who submits a Sharpe-4 backtest with no cost analysis and no mention of capacity has failed the case before a human reads it, because they have demonstrated they would ship the very ideas the pipeline exists to kill. A candidate who submits a modest, honestly-validated edge with a clear-eyed discussion of where it would fail has demonstrated the exact discipline the job requires.

The day-to-day, covered in [a day in the life of a quant researcher](/blog/trading/quant-careers/a-day-in-the-life-quant-researcher), is mostly the early gates: hypothesis framing and the long, unglamorous data work, punctuated by the rare backtest that survives. The late gates — capacity, paper, live, ramp, monitor — are more collaborative, involving the desk, risk managers, and quant developers who own the production code. As you grow from junior to senior, your center of gravity shifts *rightward* along the pipeline: a junior spends most of their time on data and signal construction; a senior spends more of theirs on capacity, portfolio construction, and the kill decisions, because those are where the firm's real money is decided. The pay reflects it — variable compensation is tied to the live P&L your signals contribute, which means it is tied to how well your ideas survive the late gates, not how clever they looked in the backtest.

And the honest reality, the one this whole series keeps returning to: *most of your ideas will die, and that is the process working.* A researcher whose ideas never die is not a genius — they are someone whose gates are broken, and the firm's money will find the holes. The researchers who last are the ones who internalize that being wrong, cheaply and quickly and honestly, ninety-nine times out of a hundred, is not failure. It is the job.

## When this matters / Further reading

This systems view matters the moment you stop thinking about quant research as "having good ideas" and start thinking about it as "running a filter." If you are preparing for research interviews, it tells you what the [research case](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it) is really testing — not whether you can find an edge, but whether you can build the gates and kill your own idea. If you are a student building a portfolio of projects, it tells you that a project showing a *modest, honestly-validated, cost-aware* signal with a capacity discussion will impress a quant far more than a flashy backtest with a Sharpe of 5. And if you are already on a desk, it is the map of where you are spending your time and where the firm's money is actually decided.

To go deeper on the individual stages, follow the links out to the methods series:

- The craft of building the signal itself: [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).
- Doing the backtest without fooling yourself: [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research).
- The statistical machinery that keeps you honest: [overfitting, purged CV, and the deflated Sharpe](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research).
- The numbers that summarize and monitor a signal: [evaluating alpha signals: IC, Sharpe, turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research).

And for the human side of the same machine — the daily rhythm, the toolkit, and how the pipeline shows up in hiring — read the companion posts in this series: [a day in the life of a quant researcher](/blog/trading/quant-careers/a-day-in-the-life-quant-researcher), [statistics and ML for alpha research](/blog/trading/quant-careers/statistics-and-ml-for-alpha-research-the-researchers-toolkit), and [the research case and take-home](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it). The pipeline is the same in all of them; what changes is whether you are living it, learning it, or being tested on it. In every version, the lesson holds: the discipline is in the gates, not the idea.
