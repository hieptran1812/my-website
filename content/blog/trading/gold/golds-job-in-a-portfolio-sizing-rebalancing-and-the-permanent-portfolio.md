---
title: "Gold's Job in a Portfolio: Sizing, Rebalancing, and the Permanent Portfolio"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Gold barely beats inflation over the long run, yet a 5-15% sleeve can raise a portfolio's risk-adjusted return. This is the allocation math: why gold diversifies, how to size it, the rebalancing bonus, and the Permanent Portfolio case."
tags: ["gold", "portfolio", "asset-allocation", "diversification", "rebalancing", "permanent-portfolio", "risk-parity", "correlation", "sharpe-ratio", "60-40"]
category: "trading"
subcategory: "Gold"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Gold's value to a portfolio is not its return. It is its near-zero correlation to stocks and bonds and its tendency to pay off in exactly the regimes where both of them fail. A small gold sleeve, rebalanced, can raise a portfolio's *risk-adjusted* return even though gold by itself barely beats inflation.
>
> - **Gold is a diversifier, not an engine.** Over 1971-2024 gold returned roughly 3% per year after inflation versus about 7% for stocks. You do not hold it to get rich; you hold it because it zigs when your stocks and bonds zag.
> - **The free lunch is correlation, not return.** Mixing an asset with low or negative correlation into a portfolio lowers the *whole portfolio's* swings by more than it lowers the return — so the return per unit of risk goes up. That is the only free lunch in finance, and gold is one of its cleanest sources.
> - **Size it 5-15%, then rebalance.** A 5% sleeve cushioned a 60/40 portfolio's worst year and a fixed-weight rebalancing rule mechanically sells gold high and buys stocks low for you — no forecast required.
> - **The one number to remember:** in 2022, when stocks fell about 18%, long bonds fell 31%, and a 60/40 portfolio fell 16%, **gold finished -0.3%** — flat, while everything else broke.

In January 2022, almost every retirement account on Earth was about to have its worst year in a generation, and almost no one saw it coming. The standard advice — the bedrock of pension funds, robo-advisors, and a million "set it and forget it" portfolios — was the **60/40**: 60% stocks, 40% bonds. The logic was a half-century of received wisdom: stocks grow your money, bonds steady the ride, and crucially, when stocks fall, bonds usually *rise*, cushioning the blow. For decades it had worked. Then 2022 happened. Inflation roared back, the Federal Reserve hiked interest rates at the fastest pace in forty years, and the two halves of the 60/40 — which are *supposed* to offset each other — fell *together*. Stocks dropped about 18%. Long-term government bonds, the supposed safe ballast, dropped a stunning 31%. The 60/40 portfolio lost about 16% — its worst calendar year since 1937.

In that same year there was one boring, ancient, yield-less asset that did almost nothing. It did not soar. It did not crash. Gold finished the year down **0.3%** — essentially flat, while the stock-and-bond machine that runs the world's savings was breaking in both gears at once. An investor who had carved a small slice of their 60/40 into gold would have ended 2022 with a smaller loss, a smoother ride, and — this is the part that matters — a better return for every unit of risk they took.

That is the whole case for gold in a portfolio, and it is *not* the case you usually hear. The usual case is some version of "gold goes up." This post argues the opposite is fine: gold can barely beat inflation over decades and *still* be one of the most valuable things you can own — because its job is not to win, it is to **not lose when everything else does**. That sounds like a paradox. The resolution is a piece of math called diversification, and once you see it, the question stops being "will gold go up?" and becomes "how big a position, and rebalanced how?" This post is that allocation math, end to end.

![Two by two of growth and inflation regimes with the winning asset in each, gold winning the stress quadrants](/imgs/blogs/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio-1.png)

The cross-asset companion to this series, [gold as money, insurance, or just a rock](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock), frames *whether* an allocator should own any gold at all. This post takes that as settled and goes one level deeper into the mechanics: how much, why rebalancing matters, and what the famous all-weather portfolios actually do with the metal.

## Foundations: correlation, diversification, and risk-adjusted return

Before we can talk about *how much* gold to hold, we have to build four ideas from zero. If you already trade for a living you can skim; if you have never thought about a portfolio as a system rather than a pile of bets, read slowly, because everything later rests on these four words: **return, risk, correlation, and rebalancing.**

### Return and risk are two different things

**Return** is how much an asset grows your money, expressed as a percent per year. If you put in \$100 and a year later have \$107, your return was 7%. Easy.

**Risk**, in the portfolio sense, is *not* "the chance you lose money" in some vague way. It has a precise meaning: it is how much the return *bounces around* from year to year. The technical name is **volatility**, and the technical measure is **standard deviation** — but you can hold the intuition without the statistics. An asset that returns +7% every single year, like clockwork, has zero volatility. An asset that returns +30% one year, -25% the next, +15%, -10% — averaging maybe the same 7% — has *high* volatility. Both might end up in the same place, but the second one is a far rougher ride, and a rough ride matters for real reasons: you might need the money in a down year, you might panic and sell at the bottom, and (as we will see) volatility quietly *eats* compound returns.

Here is the single most important fact in this whole post: **stocks have high return and high risk. Gold has modest return and high-ish risk. Bonds have low return and low-ish risk.** If risk and return were all that mattered, gold would look like a bad deal — it has stock-like volatility but bond-like returns. So why would any sensible person hold it? The answer is the third word.

### Correlation: do two assets move together or apart?

**Correlation** measures whether two assets tend to move in the *same direction* at the *same time*. It is a number between -1 and +1.

- **+1** means they move in perfect lockstep: when one is up, the other is always up, by a proportional amount. Two S&P 500 index funds have a correlation near +1 — they are basically the same thing.
- **0** means they move *independently*: knowing what one did tells you nothing about the other. Gold and the stock market are close to this — over the long run, gold's correlation to stocks hovers around zero, sometimes a touch positive, sometimes negative.
- **-1** means they move in perfect *opposition*: when one is up, the other is always down. Nothing in real markets is reliably -1, but gold during a crisis can lurch sharply negative — stocks crash, gold rallies.

Why does this matter? Because **the risk of a portfolio is not the average of its parts' risks — it is *less*, and how much less depends entirely on correlation.** This is the hinge of the entire argument, so let me make it concrete. Picture two assets, each of which bounces around a lot on its own. If they are perfectly correlated (+1), holding both is no smoother than holding one — when one drops, so does the other, and your combined value drops just as hard. But if they are uncorrelated (0) or, better, negatively correlated, then on a day when one drops the other is often flat or up, and the two partly *cancel out*. The combined portfolio is calmer than either piece. You have reduced your risk *without* reducing your average return. That reduction-for-free is what people mean by the **diversification "free lunch"** — and it is the only free lunch in all of finance. The cross-asset series explains the general principle in [correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch); here we are applying it to one specific, unusually well-suited asset.

The mathematics behind that claim is worth one paragraph, because once you see it the whole post follows. When you combine two assets, the *return* of the blend is just the weighted average of their returns — boring, linear, no magic. But the *risk* of the blend is **not** the weighted average of their risks. The formula for the combined volatility has a third term inside it that depends on the correlation, and that term *subtracts* from the total whenever the correlation is below +1. The lower the correlation, the bigger the subtraction, and the calmer the blend. At a correlation of zero the combined volatility is meaningfully *below* the weighted average of the two pieces; at a negative correlation it is lower still. This asymmetry — returns average linearly, risks combine *sub*-linearly when correlation is low — is the entire engine of diversification. It is why you can take two volatile assets, mix them, and get something calmer than either. And it is why an asset's *correlation* to what you already own can matter more than its *own* return: a low-correlation asset gives you that subtraction term, and the subtraction term is free risk reduction.

### Risk-adjusted return: the Sharpe ratio

Once you accept that risk and return are separate, you need a way to compare portfolios that have *both* a different return *and* a different risk. A portfolio that returns 8% with wild swings is not obviously better than one that returns 7% with a gentle ride. We need a single score that rewards return and penalizes risk.

That score is the **Sharpe ratio**, named after economist William Sharpe. It is almost embarrassingly simple:

```
Sharpe ratio = (portfolio return - cash return) / portfolio volatility
```

In words: take your return *above what you could have earned risk-free in cash*, and divide by how much your portfolio bounced around. A higher Sharpe ratio means more reward per unit of white-knuckle risk. Two portfolios can have the same return, but the one with lower volatility has the higher Sharpe — it got there more safely.

Why subtract the cash return? Because you can always earn *something* in a savings account or Treasury bill with no risk at all. Only the return *above* that risk-free rate is your reward for taking risk, so that is what we measure against the risk you took. Throughout this post, when I say a gold sleeve "raises risk-adjusted return," I mean precisely this: it raises the Sharpe ratio — usually by *lowering the denominator* (portfolio volatility) more than it lowers the numerator (return). Gold does not need to *add* return to help. It just needs to subtract more risk than it costs you in return.

### Rebalancing: snapping back to your targets

The fourth idea is the most mechanical and, surprisingly, one of the most powerful. **Rebalancing** is the discipline of periodically buying and selling to return your portfolio to its *target weights*.

Say you decide your portfolio should be 55% stocks, 40% bonds, 5% gold. You buy exactly that. A year passes. Stocks have a great year and gold is flat, so now your mix has *drifted*: stocks are 60%, bonds 35%, gold 4%. You are no longer holding the portfolio you designed — you are holding a riskier one, because the winner grew into a bigger share. **Rebalancing** means selling some stock and buying some bonds and gold until you are back at 55/40/5.

Notice what that forces you to do: it makes you **sell whatever just went up and buy whatever just went down or lagged.** It is a rule that *automatically* sells high and buys low, with no judgment, no forecast, no nerve required. That sounds trivial. It is not — for an asset like gold that swings around a flat-ish trend, this mechanical buy-low-sell-high is a genuine source of return all on its own, a phenomenon we will quantify later as the **rebalancing bonus**.

Those are the four pillars: **return** (growth), **risk** (volatility), **correlation** (do they move together?), and **rebalancing** (snap back to target). With them in hand, gold's strange portfolio behavior stops being mysterious.

## Why gold diversifies when almost nothing else does

The defining feature of a great diversifier is not high return — it is *low correlation* to the things you already own, *especially* in the moments those things crash. Gold has three properties that make it close to ideal on this measure, and each one is the subject of a deeper post in this series. Here is how they stack up into a portfolio argument.

### Near-zero correlation to stocks

Over the long run, gold's correlation to the stock market sits close to zero. That is unusual. Most "alternative" assets you can buy turn out, when the panic comes, to be stocks wearing a costume — corporate bonds, real estate, private equity, high-yield credit, even most commodities are all ultimately bets on the economy doing well, so they all fall together in a recession. Gold is not a bet on the economy. It is a bet on the *opposite* — on the system's plumbing failing, on real interest rates falling, on confidence in paper money eroding. That orthogonal driver is exactly what makes its correlation to stocks hover around zero rather than springing positive when you least want it to.

A correlation near zero is the workhorse case. It means that, year in and year out, gold's ups and downs are *unrelated* to your stocks' ups and downs, so it smooths the combined ride without you needing gold to do anything heroic. The heroics are a bonus, and they show up in crises.

It is worth being precise about what "near zero on average" hides, because the average is the *least* interesting part. Gold's correlation to stocks is not a fixed constant — it drifts with the regime. In calm, risk-on years it can be mildly *positive* (everyone is buying everything, gold included). In the regimes that matter — a deflationary crash, a stagflation, a currency scare — it swings sharply *negative*. The long-run average of those swings lands near zero, but an allocator does not experience the average; they experience the regimes. And gold's correlation is negative in exactly the regimes where a stock portfolio is hemorrhaging. This is the opposite of what happens with most so-called diversifiers, whose correlations *rise toward +1* in a crisis (when "everything falls together") right when you needed them to fall. Gold's correlation behaves the way you want a hedge to behave: low when you do not need it, negative when you do. That conditional behavior is worth far more to a portfolio than the bland average number suggests, and it is the single best argument for holding the metal.

### A crisis payoff: correlation that turns negative when it matters

The truly valuable thing about gold is that its low correlation is not symmetric — it *gets better in exactly the bad moments*. When stocks crash and central banks respond by slashing rates and printing money, gold tends to rally. So gold's correlation to stocks, near zero in normal times, lurches *negative* in the regimes that hurt you most. A diversifier that helps a little all the time but helps *a lot* precisely when your portfolio is on fire is worth far more than its average correlation suggests. This crisis behavior — and its tricky two-phase shape, where gold can fall *first* in a liquidity scramble before it soars — is the subject of the dedicated crisis post, [fear and the safe-haven trade](/blog/trading/gold/fear-and-the-safe-haven-trade-how-gold-behaves-in-a-crisis). For our purposes the headline is enough: gold's diversification is *front-loaded into the disasters*, which is when diversification is the only thing that matters.

### The dollar and real-rate hedge

Gold is priced in dollars and tends to move opposite the dollar's strength, and it tends to do well when *real* interest rates (the interest rate after subtracting inflation) fall. Those are two more drivers that have nothing to do with whether your stocks are having a good quarter. Stocks and bonds together are essentially a leveraged bet that real growth is fine and money is sound. Gold is the hedge against the scenario where money is *not* sound — where the currency is being debased or real rates are deeply negative. The master-variable story is in [real interest rates: the master variable behind the gold price](/blog/trading/gold/real-interest-rates-the-master-variable-behind-the-gold-price); the macro-policy transmission is in [how monetary policy moves commodities, real rates, and gold](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold). The portfolio takeaway: gold's drivers are *different drivers*, and different drivers are the raw material of diversification.

Now look at the cover figure again. It splits the economy into four climates by two axes — is growth rising or falling, is inflation rising or falling — and names the asset that wins in each. Stocks win in the calm, growing, low-inflation quadrant (the 1990s). But in the two *stress* quadrants — stagflation (growth down, inflation up) and the deflation bust (growth down) — stocks and bonds are both losing or unreliable, and gold is the asset that pays. **Gold's correlation profile is not random: it is structurally tilted toward the regimes the rest of your portfolio cannot handle.** That is why it diversifies when almost nothing else does.

#### Worked example: why a flat asset can lower portfolio risk

Forget exact statistics for a moment and watch the mechanism with round numbers. Suppose in a *bad year* your stocks fall 20% and your gold rises 5% (a typical crisis split). In a *good year* your stocks rise 20% and your gold falls 5%. Gold's average return across the two years is 0% — it goes nowhere. A naive investor says: "gold added nothing, drop it."

But watch the *portfolio*. Say you hold 90% stocks, 10% gold.

- **Bad year:** `0.90 × (-20%) + 0.10 × (+5%) = -18.0% + 0.5% = -17.5%`. Your all-stock neighbor lost 20%; you lost 17.5%.
- **Good year:** `0.90 × (+20%) + 0.10 × (-5%) = +18.0% - 0.5% = +17.5%`. Your neighbor made 20%; you made 17.5%.

Your *average* return barely changed (the all-stock portfolio averages 0% over the two years; so do you). But your *swing* shrank from ±20% to ±17.5%. You took less risk for the same return — your Sharpe ratio went *up* — and the asset that did it returned *zero*. **A zero-return asset that moves opposite your main holding is not dead weight; it is portfolio insurance you are paid in smoothness to hold.** That is the free lunch, and gold is one of the few assets that reliably serves it.

## Gold barely beats inflation — and that is fine

Let me now puncture the most natural objection before it grows: *"if gold's whole appeal is that it goes nowhere, why hold an asset that goes nowhere?"* Because going-nowhere is the *job*, and the long-run numbers confirm it is doing the job, not failing at it.

![Bar chart of real annual returns 1971 to 2024 with US stocks 7 percent, gold 3 percent, bonds 2.5 percent, cash 0.4 percent](/imgs/blogs/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio-3.png)

Across the whole post-gold-standard era — from the 1971 Nixon shock that cut the dollar's last link to gold, through 2024 — the real (after-inflation) returns line up like this: **US stocks roughly 7% per year, gold roughly 3%, bonds roughly 2.5%, cash roughly 0.4%.** Gold's 3% real is real money over decades — it is not the "useless rock" the cynics claim; it has preserved and slowly grown purchasing power across more than fifty years. But it is *less than half* of what stocks delivered. If you are trying to *grow* a portfolio, gold is not your engine. Stocks are.

Two things hide inside that 3% average, and both matter. First, the average conceals enormous regime variation: from 1971 to 2000, gold's real return was actually slightly *negative* (-0.5% per year) — three decades of dead money — while from 2001 to 2024 it ran about +7.5% real, rivaling stocks. Gold does not deliver its return smoothly; it delivers it in violent multi-year bursts separated by long winters, which is why [gold's behavior comes by the decade, not the month](/blog/trading/gold/gold-and-inflation-the-hedge-that-works-by-the-decade-not-the-month). Second — and this is the key portfolio point — **even an asset whose long-run return only matches inflation can raise a portfolio's risk-adjusted return, as long as its correlation is low enough.** The worked example above proved it with a zero-return asset. Gold's real-world 3% real is *better* than that hypothetical zero, so the case is only stronger. The return is the consolation prize. The correlation is the product.

This is why the framing in this series' spine matters: **gold is not an investment that compounds — it is a monetary insurance policy.** You do not evaluate insurance by asking "did it return more than stocks?" You evaluate it by asking "did it pay out when I needed it, and what did it cost me to hold?" By that test, gold's modest standalone return *is the premium you pay*, and a flat-to-modest year is insurance you bought and (thankfully) did not need. The chart above is not an indictment of gold. It is the price tag.

## How much gold to hold: the 5-15% range

So gold diversifies, and a flat return is acceptable for a diversifier. How big should the slice be? The professional consensus, across decades of allocation studies and the practices of large institutions, lands in a remarkably tight band: **5% to 15% of a portfolio.** Let me explain where that range comes from, because the *reasoning* matters more than the number.

### Why not zero?

Below about 5%, gold is too small to move the needle. A 2% sleeve that rallies 30% in a crisis adds only 0.6% to your portfolio — a rounding error against a 20% stock loss. If you are going to hold gold at all, the position has to be large enough that its crisis payoff is *felt*. Most studies find the diversification benefit becomes meaningful starting around 5%.

### Why not 50%?

This is the more important boundary, and it is where most gold enthusiasts go wrong. Above roughly 15-20%, gold starts to *dominate* your portfolio's behavior, and now its low return becomes a real drag while its high volatility becomes your problem rather than your hedge. Remember gold has *stock-like* volatility. A portfolio that is half gold is a portfolio that swings like stocks but returns like a savings account over the long run — the worst of both worlds. The diversification benefit of adding gold has **diminishing returns**: the first 5% does most of the smoothing, the next 5% does less, and past 15% you are mostly just dragging down your expected return without buying much additional safety. The free lunch is real, but you cannot eat infinite servings of it.

### What the size depends on

The right number *within* the 5-15% band depends on three things about *you*:

1. **How much of your portfolio is in risk assets you need to hedge.** A portfolio that is 90% stocks needs more insurance than one that is 50% bonds and 50% stocks. More stock exposure to hedge → bigger gold sleeve, toward the 10-15% end.
2. **Your view on the monetary regime.** If you believe we are in a debasement era — large deficits, structural inflation, falling real rates, central banks buying gold — you lean toward the high end. If you think real rates will stay high and positive and the dollar strong, gold will *drag*, and you lean toward the low end (or zero). The structural-buyer story behind the high-end case is in [central banks: the structural buyer that changed gold after 2022](/blog/trading/gold/central-banks-the-structural-buyer-that-changed-gold-after-2022).
3. **Whether you will actually rebalance.** A gold sleeve you never touch is worth far less than one you rebalance, for reasons the next section makes precise. If you know you will let it ride, size it conservatively.

There is a deeper reason the sweet spot is a *range* and not a single magic number, and it is worth dwelling on because it is where allocation stops being a formula and becomes a judgment. The "optimal" gold weight that a backtest spits out depends entirely on the *period* you feed it. Run the optimizer over 1971-1980 and it will tell you to hold 40% gold; run it over 1980-2000 and it will tell you to hold zero. Neither answer is right going forward, because both are *overfit* to a regime that has already happened. The 5-15% band is robust precisely because it does *not* try to nail the optimum — it is wide enough to be roughly right across many possible futures and narrow enough to keep gold a sleeve rather than a bet. An allocator who insists on a single precise number is fooling themselves about how knowable the future is; the band is an admission of honest uncertainty, and that honesty is a feature.

#### Worked example: the diminishing return of each extra slice of gold

Watch how the *benefit* of adding gold fades as the sleeve grows, using the 2022 cushion as the yardstick. Start from a 60/40 portfolio (-16.1% in 2022) and add gold by trimming stocks, one 5-point slice at a time. The 2022 result improves like this: 0% gold → -16.1%; 5% gold → -15.2% (a 0.9-point cushion); 10% gold → -14.3% (1.8 points); and if you pushed to 20% gold you would get roughly -12.5% (about 3.6 points).

Now look at the *marginal* gain from each slice. The first 5% bought 0.9 points of cushion. The second 5% bought another 0.9. The jump from 10% to 20% bought about 1.8 points across *ten* points of gold — only 0.9 points per 5-point slice, same as before, *but* you have now given up 20% of your equity engine, which in a good year costs you dearly. The cushion per slice is roughly constant in a crash year, but the *opportunity cost* per slice rises the more you starve your growth assets — so the *net* benefit per extra slice falls. That is what "diminishing returns" means in concrete numbers: each additional ounce of gold buys the same crisis insurance but costs progressively more forgone growth. *The 5-15% band is where the crisis insurance is still cheap; past it, you are paying rising premiums for the same coverage.*

#### Worked example: a 5% gold sleeve and the 2022 cushion

Let us make the cushion concrete with the real 2022 numbers. The 2022 returns chart below shows the carnage: stocks (S&P 500) -18.1%, long Treasuries -31.2%, investment-grade bonds -15.4%, the 60/40 portfolio -16.1% — and gold, alone, **-0.3%**.

![Horizontal bar chart of 2022 asset returns with gold near zero and stocks bonds and sixty forty all sharply negative](/imgs/blogs/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio-2.png)

Now take a \$1,000,000 portfolio and compare two versions through 2022. The 60/40 implies a broad-bond leg of about -13.1% (that is the bond return that, combined with -18.1% stocks at a 60/40 weight, produces the -16.1% the index reported).

**Version A — plain 60/40:**
- Return = -16.1%. Your \$1,000,000 becomes **\$839,000**. You are down \$161,000.

**Version B — 55/40/5 (carve the 5% gold out of the stock sleeve):**
- `0.55 × (-18.1%) + 0.40 × (-13.1%) + 0.05 × (-0.3%)`
- `= -9.96% - 5.24% - 0.015% = -15.2%`
- Your \$1,000,000 becomes **\$848,000**. You are down \$152,000.

The gold sleeve saved you about **\$9,000** in a single bad year — a cushion of roughly **0.9 percentage points** — for a 5% position. Double the sleeve to 10% (a 50/40/10 mix) and the loss shrinks to about -14.3%, an **\$18,000** cushion, **1.8 points**. The figure below shows that monotonic improvement directly: same brutal year, more gold, smaller hole.

![Bar chart showing 2022 portfolio loss shrinking from minus sixteen percent at zero gold to minus fourteen percent at ten percent gold](/imgs/blogs/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio-6.png)

Nine thousand dollars on a million does not sound like a fortune — and in a single year it is not. But notice three things. First, this was the *cost-free* version: we funded gold by trimming stocks, so in good years we give up a little stock upside, but the give-up is small because gold's own return is positive. Second, the cushion *compounds*: a smaller hole means less ground to recover, and avoiding a deep drawdown is worth far more than the arithmetic suggests (a portfolio that falls 50% must rise 100% just to break even). Third, and most important, **the cushion arrives in the worst year**, when you are most likely to panic and sell — and a smaller loss is the difference between holding your plan and abandoning it at the bottom. *A 5% gold sleeve does not change your good years much; it changes your worst year, which is the year that decides whether you stay invested at all.*

## The rebalancing bonus: getting paid to be disciplined

Here is where gold earns more than its standalone return suggests, and where most beginners leave money on the table. We met **rebalancing** in the foundations: snapping back to target weights. Now I want to show that for an asset like gold, rebalancing is not just risk control — it is a *source of return*, often called the **rebalancing bonus** or **diversification return**.

The mechanism is the figure below. Walk through it left to right.

![Pipeline showing set targets then crisis then weights drift then rebalance then locked in swing](/imgs/blogs/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio-4.png)

You set a target — say 55% stocks, 40% bonds, 5% gold. A crisis hits. Stocks fall; gold, with its low correlation, rallies. Now your weights have *drifted*: gold, which rose while stocks fell, is now an oversized share of your (shrunken) portfolio — say 8% instead of 5% — and stocks are an undersized share. Your rebalancing rule kicks in: **sell 3% of the portfolio in gold, buy 3% in stocks**, to get back to 55/40/5. Look at what you just did. You **sold gold after it rose** (sold high) and **bought stocks after they fell** (bought low) — mechanically, unemotionally, by rule. When the crisis passes and stocks recover, you own *more* of them than you would have, bought at the discount. You harvested the swing.

The deep insight is that the rebalancing bonus is *largest* for asset pairs with **low or negative correlation and high individual volatility** — which is gold-and-stocks almost by definition. Two assets that move together (high correlation) rarely drift apart, so there is little to rebalance between them. But two assets that move *oppositely* are constantly drifting apart and snapping back, and each snap-back is a forced sell-high-buy-low. Gold, precisely because it is volatile *and* uncorrelated, is one of the best rebalancing partners a stock-heavy portfolio can have. **The volatility you were taught to fear becomes a feature: a volatile, uncorrelated asset gives a rebalancing rule more to work with.**

#### Worked example: the rebalancing bonus in dollars

Take a \$500,000 portfolio at the start of a crisis year, held 95% in a stock fund and 5% in gold, with an annual rebalance.

- **Start:** stocks = \$475,000 (95%), gold = \$25,000 (5%).
- **Crisis year:** stocks fall 20% → \$380,000. Gold rises 20% → \$30,000. Total = \$410,000.
- **Drift:** gold is now \$30,000 / \$410,000 = **7.3%** of the portfolio; stocks are 92.7%. You have drifted off your 5% target.
- **Rebalance:** target gold = 5% of \$410,000 = \$20,500. You **sell \$9,500 of gold** (down to \$20,500) and **buy \$9,500 of stocks** (up to \$389,500). You sold gold near its high and bought stocks near their low.
- **Recovery year:** stocks rebound 25% → \$389,500 × 1.25 = **\$486,875**. Gold gives back its gain, falling 17% → \$20,500 × 0.83 ≈ \$17,015. Total ≈ **\$503,890**.

Now compare to the investor who held the *same assets* but never rebalanced. Their stocks went \$475,000 → \$380,000 → \$475,000 (×1.25). Their gold went \$25,000 → \$30,000 → \$24,900. Total ≈ **\$499,900**. The rebalancer ended with about **\$4,000 more** on the same assets over the same path — and they did it by following a rule, not by predicting anything. *The rebalancing bonus is the market paying you to be disciplined when everyone else is emotional; gold, being volatile and uncorrelated, hands you more of those disciplined trades than almost any other holding.*

There is a second, subtler reason rebalancing into a volatile diversifier pays, and it has a name: **variance drain**. A volatile asset's *compound* return is always lower than its *average* return, because losses hurt more than equal-sized gains help — a 50% loss needs a 100% gain to recover, not a 50% gain. The rougher the ride, the wider that gap, and the more your real, compounded wealth lags the simple average. Now here is the elegant part: by *lowering the portfolio's volatility*, a low-correlation sleeve like gold *narrows the variance-drain gap* — so the rebalanced portfolio compounds closer to its average return than an undiversified one does. You get a quiet, structural boost to long-run compounding *on top of* the sell-high-buy-low harvest, and both of them flow from the same source: gold's volatility paired with its low correlation. **The very combination that makes gold look scary as a standalone holding — volatile and going nowhere — is what makes it valuable inside a rebalanced portfolio.**

A practical note: rebalancing is not free. Selling gold can trigger taxes in a taxable account, and trading costs something. So most investors rebalance on a schedule (once a year) or on a *threshold* (whenever a weight drifts more than, say, 5 percentage points off target), not constantly. Over-frequent rebalancing can actually *hurt* — it cuts your winners too early and racks up costs — so the discipline is to rebalance enough to capture the drift, not so often that you are churning. But the principle stands: a rebalanced gold sleeve is worth materially more than a buy-and-hold one, and the worse-behaved (more volatile, less correlated) the sleeve, the more the rebalancing rule extracts from it.

## The Permanent Portfolio and risk parity: owning every regime

If gold's superpower is paying off in the regimes where stocks and bonds fail, you might ask: *why not build a portfolio explicitly around the idea that you never know which regime is coming?* Two famous strategies do exactly that, and gold is central to both.

### Harry Browne's Permanent Portfolio: 25/25/25/25

In the 1980s, the libertarian writer and investment advisor **Harry Browne** proposed one of the most elegant ideas in personal finance. He reasoned that the economy is *always* in one of four climates, and crucially, **you cannot reliably predict which one is coming next.** So instead of forecasting, hold one asset that thrives in each climate, in equal weight. The result is the **Permanent Portfolio**: 25% each in stocks, long-term bonds, gold, and cash.

The figure maps each quarter to its climate.

![Two by two matrix of economic climates mapping prosperity to stocks disinflation to bonds inflation to gold recession to cash](/imgs/blogs/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio-5.png)

- **Prosperity** (growth, loose money): **stocks** boom. The business cycle is expanding, earnings rise, equities lead.
- **Disinflation / tight money with growth**: **long bonds** win. When inflation and rates fall, the price of long-dated bonds rises sharply.
- **Inflation**: **gold** wins. As the currency loses purchasing power, the asset that is no one's liability and cannot be printed holds its value.
- **Recession / deflation**: **cash** is king. In a deflationary bust, cash gains purchasing power as prices fall, and — just as important — it is *dry powder* to buy the other three assets at their lows.

The beauty is that *something is always working*. In any given year, one or two of the four are carrying the portfolio while the others lag, and you never have to guess which. The Permanent Portfolio's historical return is modest — it will badly trail an all-stock portfolio in a long bull market — but its *worst years are extraordinarily shallow*, and that shallowness is the whole point. It is built for people who want to sleep at night and never be wrecked by a regime they did not see coming. Gold's 25% weight is heavy by the standards we discussed (well above the 5-15% band), but it is justified here because the *entire portfolio* is engineered around regime coverage rather than return maximization — gold is not a sleeve on top of a growth portfolio, it is one of four equal pillars.

#### Worked example: the Permanent Portfolio in 2008 and 2022

Take a \$200,000 Permanent Portfolio: \$50,000 each in stocks, long bonds, gold, cash.

**2008 (the global financial crisis):** stocks (S&P 500) fell about 37%; long Treasuries *rose* about 26% in the flight to safety; gold rose about 5%; cash returned roughly 2%.
- Stocks: \$50,000 × 0.63 = \$31,500
- Long bonds: \$50,000 × 1.26 = \$63,000
- Gold: \$50,000 × 1.05 = \$52,500
- Cash: \$50,000 × 1.02 = \$51,000
- **Total = \$198,000 — a loss of about 1%**, in a year the S&P 500 lost 37% and a 60/40 lost roughly 22%.

**2022 (the rate shock):** stocks -18%; long bonds -31% (the one climate where two pillars fail together); gold -0.3%; cash roughly +1.5%.
- Stocks: \$50,000 × 0.82 = \$41,000
- Long bonds: \$50,000 × 0.69 = \$34,500
- Gold: \$50,000 × 0.997 ≈ \$49,850
- Cash: \$50,000 × 1.015 = \$50,750
- **Total ≈ \$176,100 — a loss of about 12%.**

Look at the contrast. In 2008, a textbook deflationary crisis, the Permanent Portfolio was *almost perfectly protected* — long bonds and gold both rose, offsetting the stock collapse, and it lost ~1% while the world burned. In 2022 it had a genuinely *bad* year (-12%), because 2022 was the rare regime where *two* of its four pillars (stocks and bonds) failed at once and only gold and cash held — but even -12% beat the 60/40's -16%, and gold and cash were the only reason it was that shallow. *The Permanent Portfolio is not magic; it can still lose in a two-pillar regime like 2022. But it converts catastrophic years into merely bad ones, and gold is the pillar that covers the inflation climate the other three cannot.*

### Risk parity: weighting by risk, not by dollars

The Permanent Portfolio weights by *dollars* — 25% each. A more modern cousin, **risk parity** (popularized by Ray Dalio's "All Weather" approach), weights by *risk* instead: it sizes each asset so that every one contributes roughly the *same amount of volatility* to the portfolio. Because bonds are far less volatile than stocks, risk parity ends up holding *more* bonds (often leveraged) and a meaningful gold and commodity sleeve, so that no single asset dominates the portfolio's swings. The goal is identical to Browne's — own every regime, depend on no forecast — but the math is more refined. The full treatment is in [all-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime). What matters for *gold* is that in both frameworks, gold is not an afterthought or a speculation. It is a **structural pillar**, held specifically to cover the inflation-and-currency-debasement regime that is the blind spot of every stock-and-bond portfolio ever built.

## When gold helps and when it just drags

Honesty demands the other side. Gold is not always a free lunch; sometimes it is a slow tax. The figure lays out the two columns.

![Before after comparison of regimes where gold drags such as high real rates versus regimes where gold helps such as crisis and debasement](/imgs/blogs/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio-7.png)

**Gold drags when:**

- **Real interest rates are high and positive.** When you can earn a *real* 2-3% in cash or short bonds with no risk, the opportunity cost of holding a yield-less metal is steep. Every year you hold gold instead of a real-yielding asset, you forgo that yield. This is why gold did almost nothing from 1980 to 2000 — Volcker's high real rates made the metal's zero yield unbearably expensive. The story is in [the twenty-year winter: 1980 to 2000](/blog/trading/gold/the-twenty-year-winter-1980-to-2000-when-gold-did-nothing).
- **Stocks are compounding in a calm bull market.** In a long Goldilocks expansion — growth fine, inflation low, dollar trusted — stocks compound at 7%+ real and gold lags far behind. Your gold sleeve is a persistent drag, and the *only* compensation is the insurance it would provide if the regime changed. In the 1990s, holding gold felt like paying for an alarm system in a neighborhood with no crime.
- **The dollar is strong and trusted.** No debasement fear, no safe-haven bid, nothing to push gold up.

**Gold helps when:**

- **Real rates are negative.** When a "safe" bond is losing to inflation, gold's zero yield is suddenly *competitive* — you are not giving up anything by holding it, and its scarcity becomes attractive. This is the 2020-2021 setup that drove gold to records.
- **Crisis and crash.** Stocks fall, central banks cut and print, real yields collapse, and gold soars — the 2008 and 2020 pattern.
- **Debasement and deficits.** When money is being printed and the currency is doubted, gold is the anti-currency — the asset that wins when fiat is diluted. This fiscal story is in [debasement, debt, and gold as an anti-currency](/blog/trading/gold/debasement-debt-and-the-fiscal-story-gold-as-an-anti-currency).

The honest framing is this: **the years gold "drags" are the premium you pay; the years it "helps" are the claim paying out.** You will have more drag years than help years — that is what it means for insurance to be cheap. The mistake is to look at a string of drag years (like the 1990s, or 2013-2018) and conclude gold is useless, then sell it right before the regime that makes it priceless. Insurance you cancel the year before the fire is worse than no insurance, because you paid all the premiums and got none of the payout.

## Common misconceptions

Three myths cause more bad gold decisions than anything else. Each one is a half-truth that becomes wrong the moment you think in portfolio terms.

**Myth 1: "Gold has no place in a portfolio because it yields nothing."** This is the most common and the most confused. It treats gold as if it were competing with bonds on yield — but gold's job is *not* to yield, it is to be *uncorrelated*. A bond pays you interest *and* moves with the rate cycle (it fell 31% in 2022). Gold pays nothing *and* moves on a different axis entirely (it was flat in 2022). The yield is not the point; the diversification is. The worked example earlier proved a *zero-yield, zero-return* asset can still raise risk-adjusted return purely through correlation. The correct statement is not "gold yields nothing, so avoid it" but "gold yields nothing, so don't hold *too much* of it" — which is why the band is 5-15%, not 50%.

**Myth 2: "More gold is always safer."** No — gold has *stock-like volatility*. A 5% sleeve smooths your portfolio; a 50% sleeve makes your portfolio swing like gold, which means swinging like a volatile asset with a mediocre long-run return. The diversification benefit has sharply diminishing returns: the first 5% does most of the smoothing, and past ~15-20% you are *adding* risk and *subtracting* return. "Safe" is a property of the *portfolio*, not of any single asset — and the safest portfolios hold gold in moderation, not to the hilt. The gold maximalist who holds 80% bullion is not safe; they have simply traded stock risk for gold risk.

**Myth 3: "Gold's low return makes it useless."** This conflates a *standalone* judgment with a *portfolio* judgment. Standalone, yes, 3% real is unexciting. But you do not hold a diversifier standalone — you hold it *inside* a portfolio, where its value is measured by what it does to the *whole portfolio's* Sharpe ratio. By that measure, an asset that returns 3% real with near-zero correlation to your main holdings is extraordinarily *useful*, precisely because it does its smoothing without dragging the return to zero. Judging gold by its solo return is like judging a seatbelt by how fast it makes the car go. Wrong metric entirely.

## How it shows up in real markets

Theory is cheap. Here is how gold's portfolio contribution actually played out across the three big stress events of the modern era — the cases that built (and tested) every diversification argument above.

**2008 — the global financial crisis.** This was gold's textbook scenario: a deflationary collapse met with massive monetary easing. The S&P 500 fell about 37% on the year and roughly 57% peak-to-trough. Long Treasuries rallied hard (the classic flight to safety), and gold rose about 5% for the year while almost everything risky was crushed. A portfolio with a gold sleeve had that sleeve doing exactly its job — flat-to-up while stocks halved. The Permanent Portfolio, as the worked example showed, lost about 1%. Gold's only stumble was a brief liquidity-scramble dip in late 2008 (the two-phase pattern), but it recovered and then ran for three more years.

**2020 — the COVID crash.** Gold showed both of its crisis faces in eight weeks. In the March dash-for-cash, gold *fell* about 12% alongside stocks as leveraged investors sold their best, most-liquid assets to raise cash — a reminder that gold's diversification can briefly *fail* in the opening days of a liquidity panic. But once the Fed slashed rates to zero and launched unlimited QE, real yields collapsed and gold soared to a record above \$2,075, finishing 2020 up about 25%. A rebalancer who *added* to gold in the March dip, or simply held through it, was richly rewarded. The lesson for a portfolio: gold's payoff is in *phase two* of a crisis, after the policy response — which is why you size it in advance and hold through the scary first act rather than chasing it.

**2022 — the rate shock.** The case that opened this post, and the one that *re-sold the world on gold*. This was the regime that breaks the 60/40: inflation and rates rising together, so stocks *and* bonds fell in lockstep. The 60/40 lost 16%, its worst year in decades. Bonds — the supposed ballast — were the *second-worst* major asset, down 31% at the long end. Gold finished -0.3%. It did not soar (real rates were *rising* in 2022, which normally pressures gold — the structural central-bank bid and currency-debasement fears held it up), but *flat in a year everything else broke* is precisely the diversification you pay for. Every worked example in this post used the real 2022 numbers because 2022 is the cleanest demonstration in living memory of *why* you hold an asset that does nothing: in the year the rest of your portfolio needed help the most, gold was the one thing not bleeding.

There is a deeper lesson buried in 2022 that is easy to miss. Notice that gold *did not even need to rise* to do its job. The entire benefit came from gold *not falling* while everything correlated to it collapsed. We have been trained to think a hedge "works" only when it spikes up dramatically. But in portfolio terms, an asset that stays flat while your other holdings drop 16-31% is delivering enormous relative value — it is *outperforming* the rest of your portfolio by 16 to 31 points, which is exactly what a diversifier is supposed to do. The mistake is to judge gold against zero ("it only returned -0.3%, so it did nothing") instead of against *your other holdings* ("it returned -0.3% versus -16% for the portfolio it was protecting"). Relative performance, not absolute, is the lens that makes a diversifier's value visible. And 2022 also quietly demolished the lazy assumption that bonds are *the* diversifier you need — for forty years bonds and stocks had moved opposite each other, so a generation of investors assumed the 60/40 was permanently safe. The stock-bond correlation is *not* a law of nature; it flips positive in inflationary regimes, and when it does, the 60/40's only hedge is whatever *third* asset has a genuinely different driver. Gold is the most accessible such asset, which is why 2022 sent allocators who had ignored it for a decade scrambling back to it.

#### Worked example: the cost of a sleeve in calm years vs the payoff in a crash

Put a dollar value on the whole trade-off with a \$500,000 portfolio and a 10% (\$50,000) gold sleeve, funded by trimming stocks.

**The cost, in a calm bull year:** suppose stocks return 12% and gold returns 2%. Without the sleeve, your \$50,000 of stock would have made \$6,000. As gold, it made \$1,000. The sleeve *cost* you about **\$5,000** in forgone upside that year — roughly 1% of the portfolio. Real, but modest. Across, say, eight calm years, the cumulative drag might be \$40,000 of forgone gains — the total premium paid.

**The payoff, in a -35% crash year:** now a real bear market hits. Your stock holdings fall 35%. But gold, in a flight-to-safety-plus-rate-cut crash, rises 25%.
- Without the sleeve (all \$500,000 in stocks/bonds at a 35% drawdown on the risk portion): the risk assets fall hard; say the portfolio drops to about \$340,000.
- With the 10% gold sleeve: the \$450,000 of risk assets fall 35% to \$292,500, but the \$50,000 gold sleeve *rises* 25% to \$62,500. Total ≈ **\$355,000** — about **\$15,000 more** than without the sleeve, *in a single crash year*.

That \$15,000 crash-year payoff roughly offsets three calm years of premium — and it arrives at the exact moment you most need ballast and are most tempted to sell at the bottom. *Over a full cycle, the gold sleeve is close to a wash on return, but it converts a portion of your terrifying years into merely bad ones — and that conversion, not the return, is what you are buying.*

## The takeaway: gold as a position size, not a bet

Step back, because the whole point of this post is a reframe. Most people approach gold as a *bet*: "is gold going up or down?" That is the wrong question, because it makes gold a forecast you can get wrong, a trade you can be early or late on, a position you will second-guess every time it lags. Framed that way, gold is exhausting to hold and easy to abandon at the worst moment.

The allocator's reframe is to treat gold as a **position size, not a bet.** You decide — once, in calm times, based on how much risk you carry and your view of the monetary regime — that gold will be, say, 7% of your portfolio. Then you stop forecasting. You hold it, you rebalance it back to 7% when it drifts, and you let it do its structural job: zigging when your stocks zag, paying off in the regimes that break everything else, and handing your rebalancing rule a stream of disciplined sell-high-buy-low trades. You are not trying to be right about gold's price. You are buying *insurance and a rebalancing partner*, sized so its drag in good years is tolerable and its payoff in bad years is felt.

This is why the series' spine — **gold is not an investment that compounds, it is a monetary insurance policy** — resolves into such clean portfolio advice. You do not buy insurance hoping it appreciates. You buy the right *amount* of it, you do not over-insure (no 50% sleeves), you do not cancel the policy after a long claim-free stretch (the 1990s, 2013-2018), and you certainly do not judge it by whether it "beat the market." The 5-15% sleeve, rebalanced, is the financial equivalent of a well-sized insurance policy on a house you love: a small, steady premium in the calm years, and the thing that keeps you whole in the fire.

The deepest insight, the one that turns all of this from theory into discipline, is the one the math has been whispering the whole way through: **an asset that goes nowhere can be worth more to your portfolio than an asset that goes up.** Gold's flatness is not a flaw to be apologized for; it is the *uncorrelated* flatness that smooths your ride, the crisis-tilted flatness that pays when you bleed, and the volatile flatness that feeds your rebalancing rule. Stop asking whether gold will go up. Decide how much of it you should own, rebalance to that number, and let the only free lunch in finance do its quiet work.

## Further reading & cross-links

- [Gold: money, insurance, or just a rock?](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) — the allocator's-eye view of gold as one line in a portfolio; this post is the deeper sizing-and-rebalancing build of that framing.
- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) — the general principle behind why a low-correlation asset lowers portfolio risk; gold is the cleanest single example.
- [All-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) — the modern, risk-weighted cousin of the Permanent Portfolio, where gold is a structural pillar.
- [Fear and the safe-haven trade: how gold behaves in a crisis](/blog/trading/gold/fear-and-the-safe-haven-trade-how-gold-behaves-in-a-crisis) — the two-phase crisis shape (fall first, soar second) that the 2008 and 2020 case studies above depend on.
- [Real interest rates: the master variable behind the gold price](/blog/trading/gold/real-interest-rates-the-master-variable-behind-the-gold-price) — why high positive real rates make gold drag and negative real rates make it help.
- [How monetary policy moves commodities, real rates, and gold](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) — the policy transmission that drives the regime gold is built to hedge.
