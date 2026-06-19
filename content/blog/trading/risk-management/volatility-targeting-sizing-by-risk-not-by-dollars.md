---
title: "Volatility Targeting: Sizing by Risk, Not by Dollars"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why a fixed dollar position is secretly a moving risk bet, and how setting a volatility target and a leverage of target divided by realized vol pins your risk steady, tames your worst drawdowns, and makes every position contribute equal risk."
tags: ["risk-management", "volatility-targeting", "position-sizing", "leverage", "vol-scaling", "turnover", "risk-parity", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **The one idea:** size a position by the **risk** it carries, not by the **dollars** it costs — pick a target portfolio volatility and set your leverage to target volatility divided by the asset's realized volatility, so exposure rises when markets are calm and falls when they are stormy.
> - A **fixed dollar** position is a **moving risk bet** in disguise: when volatility doubles, the dollars stay the same but the risk doubles — you have your biggest bet on at the worst possible time.
> - **Volatility targeting** flips that: leverage = target ÷ realized vol. As realized vol rises, leverage falls, and the dollar exposure shrinks on its own to keep the *risk* constant.
> - It **stabilizes delivered risk** (realized vol stops swinging and sits near the target), **cuts the worst drawdowns** of the same underlying edge, and lets **every position contribute equal risk** instead of equal money.
> - It is **not free**: it forces constant re-sizing, so it **trades a lot** (turnover, spreads, fees), and because realized vol is **backward-looking**, it can de-lever *after* a gap — selling low, right at the bottom.
> - The practical upshot: pick a vol target you can survive, estimate vol over a sensible window, and rebalance inside a **band** so you get the risk control without paying for every tiny wiggle. This is how you keep risk constant so you survive long enough to compound.

A trader holds the same position for a year. Same ticker, same number of shares, same dollar value on the statement every single morning — they never touch it. Ask them how much risk they are running and most people would say: the same risk, obviously, it's the same position. That answer is wrong, and the gap between "the same position" and "the same risk" is one of the most expensive misunderstandings in all of trading. Because the *market* changed underneath that unchanged position. For six calm months the asset barely moved — a percent here, a percent there. Then a crisis hit and the same asset started swinging five percent a day. The position on the statement looked identical. The risk it carried had quintupled. The trader did nothing, and yet they walked, unknowingly, from a small bet into an enormous one — and they did it at the exact moment the market was most dangerous.

This is the trap that volatility targeting exists to escape. The core confusion is that we instinctively measure the size of a bet in **dollars** — "I have \$100,000 in this" — when the thing that can actually hurt us is **risk**, and risk is dollars *times how much the thing moves*. A \$100,000 position in something that wiggles 8% a year and a \$100,000 position in something that lurches 40% a year are not the same bet. One can lose you \$8,000 in a typical year; the other can lose you \$40,000. They cost the same. They are not the same risk. And when the *same* asset's volatility changes over time — which it always does — a fixed dollar holding silently becomes a different-sized risk bet every day.

Look at Figure 1, because it is the whole argument in one picture. On the left is the naive rule: hold the same dollars no matter what. When volatility doubles, your risk doubles with it — the bet balloons exactly when the market turns hostile. On the right is the discipline this post is about: hold the same *risk*. When volatility doubles, you cut the dollars in half, so the bet you are actually exposed to stays constant whether the market is sleepy or on fire. Same edge, two sizing rules. One lets the storm decide how much you bet; the other lets *you* decide, and holds you to it.

![Same dollar size lets risk balloon when volatility spikes while same risk size shrinks the dollar exposure as volatility rises a before and after comparison](/imgs/blogs/volatility-targeting-sizing-by-risk-not-by-dollars-1.png)

This post sits in Track C of the series — *position sizing and leverage* — and it is the bridge between two ideas you may already know. [The Kelly criterion](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) answers *how big should my bet be given my edge*; volatility targeting answers the more humble, more universal question of *how do I keep my risk constant when the market won't hold still*. We are going to build the whole machine from nothing: what volatility is and why it is the natural unit of position size, the one formula at the heart of it (leverage = target ÷ realized), how it stabilizes your risk and shaves your worst drawdowns, why "equal risk" is a deeper idea than "equal money," and then — honestly — the bills it makes you pay: turnover, and the cruel lag where it sometimes sells you out at the bottom. Throughout, remember the spine of this entire series: the trader's first job is not to make money, it's to **not blow up**, and you can only compound if you are still in the game. Volatility targeting is, at heart, a survival tool: it stops the market from quietly handing you your biggest bet on the worst possible day.

## Foundations: dollars, volatility, and what "size" really means

Before we can size by risk, we have to be precise about every word in that sentence. If you have read the earlier posts on [volatility](/blog/trading/risk-management/volatility-and-why-it-is-not-risk) this will be familiar — skim it. If not, this section is the floor everything else stands on.

**A position and its dollar size.** A position is simply how much of something you own (or are short). Its **dollar size** — also called notional or exposure — is the market value of that holding: 1,000 shares at \$100 each is a \$100,000 position. This is the number on your statement, and it is the number our instincts reach for when we ask "how big is my bet?"

**Return and volatility.** A return is the percentage change in price over a period. A +1% day on a \$100,000 position is a \$1,000 gain; a −1% day is a \$1,000 loss. **Volatility** is the typical size of those returns — formally, their standard deviation. We quote it annualized: a "16% annual volatility" means the asset's yearly return typically lands within about 16% of its average, which corresponds to daily wiggles of roughly 1% (because volatility scales with the square root of time, and 1% × √252 ≈ 16%). Volatility is the single most important word in this post, because it is the **conversion factor between dollars and risk**.

**Realized volatility.** This is volatility measured from what *actually happened* — you take the asset's recent daily returns, compute their standard deviation, and annualize it. It is backward-looking by construction: you can only measure the wiggle that has already occurred. The phrase "realized vol" will appear constantly; it always means *the standard deviation of the recent return history*, the thing we can compute right now from data we already have. Hold onto the fact that it looks backward — it is the source of both the method's power and its single ugliest failure mode.

**Risk, defined as dollars at stake.** Here is the pivot. The risk of a position, measured the same way the whole industry measures it, is its **dollar volatility**: the dollar size multiplied by the percentage volatility. A \$100,000 position in a 10% vol asset carries \$10,000 of annual risk (\$100,000 × 0.10). A \$100,000 position in a 40% vol asset carries \$40,000 of annual risk. *Same dollars, four times the risk.* That product — dollars × vol — is the quantity we actually care about, the one that shows up as the swing in your account, and the one a dollar-only view of "size" completely hides.

**Leverage.** Leverage is how much exposure you hold per dollar of your own capital. With \$100,000 of capital and a \$100,000 position, your leverage is 1.0× — you are "fully invested," no borrowing. A \$50,000 position is 0.5× (you're holding cash too); a \$200,000 position is 2.0× (you borrowed \$100,000, or used a future/margin to get there). Leverage is the dial volatility targeting turns. It is also, as the [leverage post](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) in this track stresses, the thing that turns a survivable loss into a terminal one if you set it wrong — so we will treat every leverage number as a risk decision, never a free knob.

**A volatility target.** This is the whole idea in three words: you decide, *in advance*, how much risk you want your portfolio to run — say, **10% annualized volatility** — and then you size every position so the portfolio actually delivers that. The target is a choice about how bumpy a ride you can tolerate without panicking and quitting at the bottom. Ten percent is a calm, sleep-at-night number; 20% is aggressive; 40% is what blows people up. The number you pick is the most important risk decision in the method, because everything else is just arithmetic in service of it.

With those six terms, the entire method is one sentence: **pick a target volatility, measure the asset's realized volatility, and set your leverage to the ratio of the two so the position you actually hold delivers the risk you chose.**

**The one empirical fact that makes any of this work: volatility clusters.** A reasonable skeptic should immediately object — if realized volatility is backward-looking, why would sizing off *yesterday's* vol tell me anything about *today's* risk? The answer is the single most reliable statistical regularity in all of financial markets: **volatility is persistent**. Big moves are followed by big moves and quiet is followed by quiet. A 3%-down day is far more likely to be followed by another large move (up or down) than a 0.2% day is. Calm regimes last weeks or months; storms also last weeks or months. This is not a coincidence or a quirk of one market — it shows up in stocks, bonds, currencies, commodities, and crypto, across every era anyone has measured. It is so robust that an entire branch of statistics (the GARCH family of models) exists just to describe it. Volatility targeting is, at bottom, a bet on *exactly this fact*: that the vol you measured over the last month is a useful estimate of the vol you'll experience over the next week. Because vol clusters, reacting to the recent past is a genuinely good — though never perfect — way to size for the near future. Returns themselves are nearly unpredictable; their *volatility* is one of the few things that is. The method exploits the one thing markets do reliably forecast about themselves.

Hold that fact firmly, because it cuts both ways and explains everything that follows. It is *why* the method works at all: persistence means yesterday's storm warns you about today's. It is also *why* the method fails on gaps: persistence is a property of *gradual* regime change, and a bolt-from-the-blue shock — a surprise default, a sudden devaluation — has no warning in the prior data, so the backward-looking estimate is blind to it until after it lands. The clustering that powers the method on slow weather is exactly what abandons it on sudden lightning. Every strength and every weakness in the rest of this post traces back to this one empirical fact about how volatility behaves over time.

#### Worked example: the same dollars, two different risks

Take our recurring **\$100,000 retail account**, fully invested in a single asset. Suppose the asset's annual volatility is **10%**.

- Dollar size: \$100,000.
- Risk (dollar volatility): \$100,000 × 0.10 = **\$10,000 per year**. In a typical year your account swings about ±\$10,000 around its drift.

Now leave the position untouched — same \$100,000, same shares — but the market turns and the asset's volatility climbs to **20%**.

- Dollar size: still \$100,000. Your statement looks identical.
- Risk: \$100,000 × 0.20 = **\$20,000 per year**. Your account now swings about ±\$20,000.

You did nothing, and your risk **doubled**. You are now running a \$20,000-a-year bet while believing you are running a \$10,000-a-year bet, and you are running it in the middle of a storm. To get back to your intended \$10,000 of risk you would have to *cut the position in half*, to \$50,000: \$50,000 × 0.20 = \$10,000. That halving is not a market call. It is not a forecast that prices will fall. It is just the arithmetic of keeping your risk where you put it.

*A fixed dollar position is a moving risk bet — the market quietly doubles your stake every time it doubles its volatility, unless you actively shrink the dollars to hold the risk steady.*

## The one formula: leverage equals target over realized

Everything in volatility targeting follows from a single equation. You want your position to deliver a target volatility. The position's volatility is its leverage times the asset's realized volatility. Set the first equal to the second and solve:

$$\text{leverage} = \frac{\text{target volatility}}{\text{realized volatility}}$$

That's it. That is the entire engine. If your target is 10% and the asset is currently realizing 10% vol, leverage is 10%/10% = **1.0×** — hold exactly your capital. If the asset calms down to 5% vol, leverage is 10%/5% = **2.0×** — you can hold *twice* your capital and still only run 10% risk, because the asset is half as wild. If the asset blows out to 20% vol, leverage is 10%/20% = **0.5×** — you must cut to half your capital to keep the same 10% risk.

Notice what this rule does, almost magically: it makes your position size the **inverse of volatility**. When the asset is calm, you lever up. When it is stormy, you lever down. The dial moves automatically and in exactly the direction a survivor wants — *small when it's dangerous, large when it's safe.* This is the precise opposite of what a fixed-dollar holding does (which keeps the bet the same regardless) and the brutal opposite of what panicking humans do (who tend to hold biggest right before a crash and smallest right after one, having been scared out at the lows).

Figure 2 shows the rule running over time. The amber line is the asset's rolling realized volatility — calm for a while, then a stress patch, then a full storm. The blue line is the leverage that falls straight out of the formula. Watch them move opposite each other: as realized vol climbs from below the 10% target up to 35%, leverage gets cut from roughly 2× down toward 0.3×. The position is being de-risked into the storm and re-risked into the calm, with no human judgment involved — just target ÷ realized, recomputed each day.

![Position leverage plotted against rolling realized volatility showing leverage rising to about two times in calm regimes and cut to about a third in stormy ones as the inverse of realized volatility](/imgs/blogs/volatility-targeting-sizing-by-risk-not-by-dollars-2.png)

A few honest notes on the arithmetic before we go further. First, "realized volatility" in that denominator is a *measured* number, so it depends on **how you measure it** — over how many days, weighting recent days more or less, whether you use closing prices or intraday ranges. We will come back to the estimation window, because that choice changes everything about how the strategy behaves. Second, leverage can in principle go very high when realized vol is tiny — 10%/2% = 5× — which is why every real implementation **caps** leverage at some maximum (commonly 1.5× to 3×) so a freakishly quiet patch doesn't load you into an enormous position right before quiet ends. Third, this is the *single-asset* version; in a portfolio you scale by the *portfolio's* volatility, which accounts for how positions diversify each other — a deeper idea we get to when we discuss equal risk and its cousin, [risk parity](/blog/trading/risk-management/risk-parity-sizing-equal-risk-not-equal-money).

#### Worked example: sizing to a 10% target, then re-sizing when vol doubles

Let's run the formula end-to-end on the **\$100,000 account**, target volatility **10%**.

**Step 1 — calm market.** The asset is realizing **10%** annual vol.
$$\text{leverage} = \frac{10\%}{10\%} = 1.0\times.$$
Hold a **\$100,000** position. Risk = \$100,000 × 0.10 = **\$10,000 per year**. Exactly on target.

**Step 2 — the market calms further** to **5%** vol.
$$\text{leverage} = \frac{10\%}{5\%} = 2.0\times.$$
Now hold a **\$200,000** position (borrow \$100,000, or use a future). Risk = \$200,000 × 0.05 = **\$10,000 per year**. Still on target — you doubled the dollars *because* the asset got half as wild, and the two changes exactly cancel.

**Step 3 — a storm hits** and vol doubles from your base case to **20%**.
$$\text{leverage} = \frac{10\%}{20\%} = 0.5\times.$$
Cut to a **\$50,000** position. Risk = \$50,000 × 0.20 = **\$10,000 per year**. On target again — you halved the dollars because the asset got twice as wild.

Across all three regimes — vol of 5%, 10%, 20%, a four-fold range — your **risk never left \$10,000**. The dollar size ranged from \$50,000 to \$200,000, a four-fold range in the *opposite* direction. That is the trade volatility targeting makes: it lets the dollars swing wildly so the risk can stay perfectly still.

*Volatility targeting holds your risk constant by letting your dollar exposure move inversely to the market's volatility — calm doubles your position, a storm halves it, and your actual stake never changes.*

## What it buys you, part one: your delivered risk stops swinging

The first and most direct payoff is the one the method is named for: your *delivered* risk — the volatility you actually experience — stops swinging around and sits near the target you chose. This sounds tautological, but it is genuinely valuable, and it is worth seeing in a picture how much wilder the alternative is.

Figure 3 compares two books running the same asset over three years through a couple of vol storms. The red line is a **fixed-size** book — it holds the same exposure forever and never re-sizes. The green line is a **vol-targeted** book aiming for 10%. Both lines plot *realized* volatility: the risk each book actually delivered, measured rolling through time. The fixed book's realized vol swings from about **10% all the way up to 44%** as the market's regime changes — more than a four-fold range in the risk it was running, none of it chosen. The vol-targeted book holds in a tight band, roughly **6% to 19%**, clustered around its 10% target. It is not perfect (we'll see why — the estimate lags), but the contrast is stark: one book lets the market set its risk; the other sets its own.

![Realized volatility of a vol targeted book staying near its ten percent target while a fixed size book's realized volatility swings from ten to over forty percent with the market regime](/imgs/blogs/volatility-targeting-sizing-by-risk-not-by-dollars-3.png)

Why does steady risk matter so much that we'd build a whole machine for it? Three reasons, all of them about survival.

**First, you can actually plan around a known risk.** If you know your book runs 10% vol, you know a bad year is roughly a 10%–20% drawdown and you can size your *life* around that — how much capital you commit, how much you keep in reserve, whether you can stomach it. If your risk secretly quadruples in a storm, all of that planning is fiction exactly when it matters.

**Second, steady risk is steadier psychology.** The thing that actually ends most traders is not a model failure; it is a *human* failure — panicking and quitting at the bottom of a drawdown they didn't expect and can't tolerate. A book whose risk balloons in a crisis hands you the worst losses precisely when your nerve is weakest. A vol-targeted book has already cut its exposure going into the storm, so the loss it delivers is smaller and more survivable — which is the difference between holding on and capitulating. We dig into that human layer in the [drawdown psychology](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) discussions later in the series, but the seed is here: constant risk is constant nerves.

**Third, and most subtly, steady risk compounds better.** This is the connection to growth-optimal sizing. Geometric (compound) growth is hurt by volatility — the famous *volatility drag*, where a +50% then −50% leaves you down 25%, not flat. A rough but useful formula for the long-run compound growth of a strategy with arithmetic edge *μ*, volatility *σ*, run at leverage *L*, is:

$$g \approx \mu L - \tfrac{1}{2}(\sigma L)^2.$$

That second term is the drag, and it grows with the *square* of your effective volatility. A book that lets its vol spike to 40% in storms pays a vicious drag penalty in exactly those periods. A book that holds vol near 10% pays a small, steady, predictable drag. Keeping realized volatility flat and near its optimal level is, quietly, a way of harvesting more compound growth from the same edge — the leverage/vol relationship the method is built to exploit. (The full Kelly story behind that formula lives in the [Kelly post](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews); here we just need the upshot: smooth, on-target vol compounds better than the same average vol that swings.)

## What it buys you, part two: shallower drawdowns

Steady volatility is nice on a chart, but the payoff that actually keeps you alive is what it does to your **drawdowns** — the peak-to-trough losses that, as the series keeps hammering, are the risk you actually feel and the thing that breaks people. Because volatility clusters — big moves bunch together, calm follows calm and storms follow storms — the periods of high realized vol are *also* the periods where the deep drawdowns happen. A vol-targeted book, by cutting exposure when realized vol rises, is automatically cutting exposure going *into* the very stretch where the worst losses occur. It de-risks before the cliff, not after.

Figure 4 shows this on a single seeded path with a real positive edge built in. The top panel is the equity of two books running that same edge — fixed-size in red, vol-targeted in green — through a long volatility storm. The bottom panel is the underwater drawdown curve for each. The fixed book takes its full position into the storm and suffers a **−25.9%** maximum drawdown. The vol-targeted book, having cut its exposure as realized vol climbed, suffers a much shallower **−14.5%** maximum drawdown from the *same underlying edge*. It gave up a little upside in the calm (the green equity line ends slightly lower, around \$107,400 versus \$110,500, because it ran less exposure on average), but it cut the worst loss nearly in half.

![Equity curves and underwater drawdown of a fixed size book versus a vol targeted book on the same edge showing the targeted book cuts the worst drawdown from about twenty six percent to about fifteen percent](/imgs/blogs/volatility-targeting-sizing-by-risk-not-by-dollars-4.png)

That trade — a touch less return for a much shallower drawdown — is exactly the trade the survival spine of this series tells you to take, and the recovery math says why. A −25.9% drawdown needs a +35% gain to recover (because the gain needed to climb out of a drawdown *d* is *d*/(1−*d*), and 0.259/0.741 ≈ 0.35). A −14.5% drawdown needs only a +17% gain to recover (0.145/0.855 ≈ 0.17). The vol-targeted book doesn't just lose less in the storm; it has a *dramatically* easier climb back to even afterward, because the recovery math is convex — every extra percent of drawdown costs you more than the last in gain-to-recover. Cutting your worst drawdown is worth more than the average it costs you, and that asymmetry is the deepest reason vol targeting is a survival tool and not just a smoothing tool.

#### Worked example: the drawdown an unchanged book walks into

Take the **\$10,000,000 book**, fully invested, in an asset that has been calmly realizing **10%** vol. The manager is comfortable: a 10% vol book, in a typical bad year, draws down perhaps 15%–20%. Reserves and investor expectations are set around that.

A crisis arrives and the asset's vol jumps to **30%**. The manager, sizing by dollars, does nothing — same \$10,000,000.

- Risk before: \$10,000,000 × 0.10 = **\$1,000,000** of annual vol.
- Risk now: \$10,000,000 × 0.30 = **\$3,000,000** of annual vol.

The book's risk has *tripled* without a single trade. A "bad year" for a 30%-vol book is a 40%–50% drawdown — on \$10,000,000 that is a **\$4,000,000 to \$5,000,000** loss, against a book that was built and sold as one that loses 15%–20%. That gap is exactly where funds blow up: the realized loss arrives at three times the size everyone planned for, the investors redeem, and the forced selling makes it worse.

Now run the vol-targeting rule instead. When vol hit 30%, leverage = 10%/30% = **0.33×**, so the manager cuts to a **\$3,333,333** position. Risk = \$3,333,333 × 0.30 = **\$1,000,000** — back on target. The book is small going into the storm, the drawdown lands near the 15%–20% that was planned for, the investors don't panic, and the fund survives to compound on the other side.

*The fixed-dollar book walks into a tripled drawdown it never chose; the vol-targeted book has already shrunk itself to the loss it can survive.*

## Equal risk: the deeper idea underneath the formula

So far we have sized one asset against one target. The real elegance of volatility targeting shows up when you have *several* positions, because it gives you a principled answer to a question dollar-thinking can't even ask properly: **how do I make my positions comparable?**

If you put \$100,000 into a quiet government bond and \$100,000 into a wild tech stock, you have equal *money* in each — but you do not have anything like equal *risk*. The bond might carry \$5,000 of annual vol and the stock \$40,000. Your "balanced" portfolio is, in risk terms, almost entirely a bet on the stock; the bond is a rounding error on your P&L. Equal money is a mirage of diversification. You *feel* diversified because the dollar amounts match, but eight times more of your risk lives in one position.

Volatility targeting fixes this by sizing each position to contribute the *same risk*. You give every position the same dollar-volatility budget — say \$10,000 of annual vol each — and back out the dollar size from there: dollar size = risk budget ÷ asset vol. The quiet bond gets a *large* dollar allocation (because it takes a lot of bonds to make \$10,000 of vol) and the wild stock gets a *small* one (because a little stock already makes \$10,000 of vol). Now each position genuinely pulls equal weight in your risk, and your diversification is real rather than cosmetic. This principle — **size by risk contribution, not by capital** — is the seed of [risk parity](/blog/trading/risk-management/risk-parity-sizing-equal-risk-not-equal-money), which extends it to whole asset-class portfolios and even levers up the safe assets so they contribute as much risk as the dangerous ones. It is also the engine behind the [all-weather and risk-parity portfolios](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) that try to own every economic regime by balancing risk across them. The common thread: dollars are the wrong unit; risk is the right one.

#### Worked example: equal money versus equal risk on the \$100,000 account

Split the **\$100,000 account** across two assets. Asset A (a calm bond fund) realizes **5%** vol. Asset B (a volatile growth stock) realizes **40%** vol. Target each position to **\$5,000** of annual risk so they contribute equally.

**The naive equal-money split:** \$50,000 in each.
- A's risk: \$50,000 × 0.05 = **\$2,500**.
- B's risk: \$50,000 × 0.40 = **\$20,000**.
- B carries *eight times* A's risk. This is a stock bet with a bond-shaped decoration — 89% of the portfolio's single-name risk sits in B.

**The equal-risk split:** size each to \$5,000 of vol.
- A: dollar size = \$5,000 ÷ 0.05 = **\$100,000**.
- B: dollar size = \$5,000 ÷ 0.40 = **\$12,500**.

Now each position contributes \$5,000 of risk. Notice that the equal-risk portfolio wants \$100,000 in the bond and \$12,500 in the stock — a total of \$112,500, which is *more* than your \$100,000 of capital. That is leverage showing up naturally: to give a calm asset a meaningful risk budget, you often have to hold *more than your capital* in it. (Whether to actually take that leverage, and what it costs and risks, is the whole subject of the leverage posts in this track — equal risk *invites* leverage on safe assets, and that invitation is not free.)

*Equal money quietly concentrates almost all your risk in the wildest asset; equal risk spreads it evenly and, in doing so, often asks you to lever the calm assets up to pull their weight.*

There is one more layer the single-asset formula glosses over, and it matters for a real book: when you combine several positions, the *portfolio's* volatility is **not** just the sum of the pieces' volatilities. Because the positions partly offset each other — when one zigs, another sometimes zags — the combined book is usually *less* volatile than its parts added up. So the leverage you apply to hit a portfolio target depends on how *correlated* your positions are. Two equal-risk positions that move independently combine to a portfolio vol of only about 71% of either one alone (the diversification benefit); two that move in lockstep combine to 100% — no benefit, because they're really one bet. This is why a vol-targeted book of *uncorrelated* bets can run *higher* gross leverage than a book of correlated bets and still hit the same risk target: diversification literally buys you room to size up. It is also the precise place the method is most fragile, because correlations are not constant — they tend to **spike toward 1 in a crisis**, exactly when you need the offset most. The moment everything starts moving together, your "diversified" portfolio quietly becomes one big position, its true volatility jumps, and a book sized for the calm-regime correlations is suddenly over-risked. That correlation-failure mode is a whole topic in its own right — it is treated as an allocation question in the [risk-parity and all-weather post](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) and as a *risk* failure mode elsewhere in this series — but flag it here: portfolio-level vol targeting inherits all the risk of the correlation assumptions baked into "how much do these positions offset each other."

#### Worked example: scaling the whole \$10,000,000 book to a target

Now do it at the book level. The **\$10,000,000 book** holds a diversified basket of positions whose combined, un-levered realized volatility is **8%** per year, and the target is **12%**.

$$\text{leverage} = \frac{12\%}{8\%} = 1.5\times.$$

So you scale the entire book to **\$15,000,000** of gross exposure (\$10,000,000 of capital levered 1.5×). The book's risk is then \$15,000,000 × 0.08 = **\$1,200,000** per year — which is 12% of the \$10,000,000 capital, exactly on target.

Now suppose a stress regime arrives and two things happen at once: each position's vol rises *and* the positions start moving together, so the basket's combined realized vol jumps from 8% to **18%**. The formula responds:

$$\text{leverage} = \frac{12\%}{18\%} = 0.67\times,$$

and you cut gross exposure from \$15,000,000 to **\$6,700,000**. Risk = \$6,700,000 × 0.18 = **\$1,206,000** — back on the 12% target. The book de-levered by more than half, automatically, because *both* drivers of its risk — higher single-asset vol and higher correlation — fed into that one combined realized-vol number in the denominator. That is the quiet power of targeting *portfolio* vol rather than each position separately: the correlation spike that would silently double a naive book's risk shows up in the estimate and forces the cut.

*Targeting portfolio volatility, not position volatility, is what lets the method catch a correlation spike — because the offsetting between positions, and its failure in a crisis, both land in the single combined number you're sizing against.*

## The mechanism, step by step

Pulling the pieces together, here is the loop a vol-targeting book actually runs, every rebalance period. It is a genuine feedback loop: the output (your position) doesn't feed back into the input, but the *next* period's vol estimate does, so the whole thing chases a moving target forever. Figure 5 lays it out.

![The vol targeting feedback loop estimate realized volatility then compute leverage as target divided by realized then scale the position then repeat at the next rebalance](/imgs/blogs/volatility-targeting-sizing-by-risk-not-by-dollars-5.png)

Walking the loop:

1. **Returns stream.** You collect the asset's recent daily returns — the raw material.
2. **Estimate realized vol.** Compute the standard deviation of those returns over your chosen window (more on the window below), and annualize it by multiplying by √252. This is the number in the denominator.
3. **Compute leverage.** Divide your target vol by that realized vol. Cap it at your maximum (so a freak-calm patch can't over-load you).
4. **Scale the position.** Set your dollar exposure to leverage × capital. This usually means *trading* — buying a bit more if leverage rose, selling a bit if it fell.
5. **Re-sized book.** You now hold a position whose risk is pinned near the target, given the latest information.
6. **Next rebalance.** Time passes, new returns arrive, realized vol updates, and you go back to step 2. Forever.

Two design choices inside that loop govern almost everything about how the strategy feels, and they are the two places practitioners spend all their time: **the estimation window** (step 2) and **the rebalance rule** (step 4). Get those right and you have a robust risk machine. Get them wrong and you have a turnover monster that whipsaws itself to death.

## The estimation window: the dial that decides everything

Look again at the formula — leverage = target ÷ *realized vol* — and notice that "realized vol" is not a fact handed down from heaven. It is a *measurement*, and every measurement has a choice of window baked into it. How many days of returns do you average over? Do you weight recent days more heavily than old ones? That choice is the most consequential knob in the whole machine, because it sets the trade-off between two things you can't have at once: **responsiveness** and **stability**.

A **short window** — say 20 trading days — is *responsive*. When the market regime shifts, a short window registers it fast: a week of big moves quickly drags the estimate up, and your leverage gets cut quickly. But the same responsiveness makes it *jumpy* — the estimate jitters around on small samples, so a couple of noisy days can spook it into trading when nothing fundamental changed. Short windows give you fast protection and high turnover.

A **long window** — say 90 trading days — is the opposite. It is *smooth and stable*: the estimate barely budges on any single day, so you trade less and your leverage is calm. But it is *slow*: a regime change takes weeks to fully register, so you stay over-levered well into a developing storm before the long average finally catches up. Long windows give you low turnover and slow protection.

There is no free choice here — you are picking a point on a spectrum between "fast but jumpy" and "smooth but slow," and the right point depends on your horizon and your tolerance for turnover. Most serious implementations split the difference with an **exponentially-weighted** estimate, which weights recent days more than old ones on a decaying scale. This is the best of both worlds in a real sense: it reacts quickly to fresh moves (like a short window) because recent days dominate, but it doesn't lurch when one old day drops out of the sample (like a long window) because the weights fade smoothly rather than cutting off at a hard edge. An exponentially-weighted vol with an effective span of roughly 30–60 days is the workhorse most practitioners reach for.

A subtler refinement matters too: you can make the estimator **asymmetric on purpose** — quick to recognize that vol has *risen* (so you de-lever fast when danger appears) but slow to accept that vol has *fallen* (so you don't rush back into a position right after a shock, when the calm is often a false dawn before a second leg down). This deliberate asymmetry directly attacks the sell-the-bottom failure mode we're about to meet: it cuts fast and re-enters slowly, which is the right bias for survival even though it costs a little upside. The estimation window is not a technical footnote — it *is* the strategy's personality, and tuning it is most of the real work of running vol targeting well.

## What it costs you, part one: turnover

Volatility targeting is not free, and the first bill is **turnover** — the amount of trading the strategy forces on you. Because realized vol drifts around continuously, the leverage the formula demands drifts too, which means you are *constantly* nudging your position bigger and smaller to track it. Every one of those nudges crosses the bid-ask spread, pays commission, and (in size) moves the market against you. A buy-and-hold book trades once and then sits there for years. A naively-implemented vol-targeting book trades a little every single day, forever.

Figure 6 puts numbers on it. It plots cumulative turnover — the total dollar volume traded, as a percent of your capital — over two years, for the same vol-targeting strategy under three rules. The static buy-and-hold book trades **100%** of capital once (the initial purchase) and then flatlines. The vol-targeting book that rebalances *every day* churns through about **837%** of capital over the two years — it effectively turns its entire book over more than eight times, paying the spread on every turn. The third line is the fix we'll get to: a rebalance *band* that only trades when leverage drifts far enough to matter cuts that to about **570%** — still a lot more than buy-and-hold, but a third less churn for almost identical risk control.

![Cumulative turnover of a vol targeted book rebalanced daily reaching over eight hundred percent of capital versus a banded rebalance at about five hundred seventy percent versus a static buy and hold at one hundred percent](/imgs/blogs/volatility-targeting-sizing-by-risk-not-by-dollars-6.png)

Why does this matter beyond the obvious "trading costs money"? Because the costs **eat your edge directly**, and they do it whether or not the strategy is working. If your edge is, say, 3% a year and your turnover-driven costs run to 1% a year, you have just handed a third of your alpha to the brokers and the spread — for risk control you could have gotten 90% of with a tenth of the trading. The art of implementing vol targeting is getting the risk-stabilization benefit without paying full freight on turnover, and that art is mostly about *not* reacting to every tiny vol wiggle.

#### Worked example: the turnover bill on the \$10,000,000 book

Take the **\$10,000,000 book** running daily vol targeting, and suppose realistic all-in trading costs (spread plus commission plus a little market impact) of **5 basis points** — 0.05% — per dollar traded. From Figure 6, the daily-rebalance version trades about **837%** of capital over two years, which is roughly **419% per year**.

- Annual dollar volume traded: \$10,000,000 × 4.19 = **\$41,900,000**.
- Annual cost at 5 bps: \$41,900,000 × 0.0005 = **\$20,950 per year**.

Now the **20% rebalance band** version trades about 570% over two years, ≈ **285% per year**:

- Annual dollar volume: \$10,000,000 × 2.85 = **\$28,500,000**.
- Annual cost: \$28,500,000 × 0.0005 = **\$14,250 per year**.

The band saves about **\$6,700 a year** in pure frictions on this book — roughly a third of the trading bill — for essentially the same risk profile. On a book that might earn a few hundred thousand dollars of edge a year, \$6,700 of avoidable cost is real money, and it is *guaranteed* money, paid rain or shine, against an edge that is only a probability.

*Vol targeting's risk control comes with a turnover tax; the difference between a daily rebalance and a sensible band can be a third of your trading bill, paid every year whether the strategy wins or loses.*

## What it costs you, part two: the realized-vol lag and selling the bottom

The turnover bill is annoying but manageable. The second cost is genuinely dangerous, and it is the one that turns volatility targeting from a clean idea into something you have to handle with care. The problem is buried in that word **realized**: your vol estimate is *backward-looking*. It can only see the wiggle that has already happened. So when the market does something sudden — a gap, an overnight crash, a shock that prints a huge move in a single session — your realized-vol estimate **hasn't updated yet at the moment of the shock**. You are still fully levered *into* the very move you'd most want to have been small for. The de-levering happens *after*, once the big return is in your estimation window — which means you cut your position *after* the price has already fallen. You sell low. You de-risk at the bottom.

Figure 7 shows this cruel sequence. The top panel is an asset price drifting calmly along, then **gapping down 12% overnight** on day 200. The bottom panel is the vol-targeting book's leverage. Right up to and *through* the gap, leverage is high — about **1.13×** — because realized vol, measured over the calm 60 days before, was low. The book takes the full 12% hit at full size; it was maximally exposed to the worst day. Only *afterward*, once that −12% day and the elevated days following it work their way into the 60-day window, does realized vol spike and leverage collapse — bottoming around **0.31×** roughly 39 trading days later. By then the damage is done and, often, the price is near its lows. The strategy sells you out at the bottom, locks in the loss, and then sits in cash through the recovery it just funded.

![A failure mode where a vol targeted book stays fully levered into an overnight gap crash and only cuts leverage afterward selling low because realized volatility is backward looking and lags the shock](/imgs/blogs/volatility-targeting-sizing-by-risk-not-by-dollars-7.png)

This is not a flaw you can fully engineer away — it is the unavoidable price of using a backward-looking estimate to size a forward-looking risk. But there are real mitigations, and serious practitioners use all of them:

- **Faster, asymmetric vol estimates.** Use an exponentially-weighted vol (which reacts faster to recent moves) or an estimator built from intraday ranges, so the spike registers sooner. Some books de-lever fast on the way up and re-lever slowly on the way down, deliberately asymmetric to avoid re-buying into a falling market.
- **Floors and caps on leverage**, so even in dead-calm you're never so levered that a gap is catastrophic, and even in a storm you don't go fully flat and miss the bounce.
- **Pairing it with genuine tail protection.** Vol targeting handles the *grind* — the gradual regime shift from calm to stormy. It does *not* protect against the instantaneous gap, because nothing backward-looking can. For the gap you need something forward-looking and convex: options, a put overlay, a tail hedge. This is exactly why the [variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) and explicit tail hedges are complements to vol targeting, not substitutes — one smooths the slope, the other catches the cliff.

There is also a systemic, second-order version of this failure that every vol-targeter should know about: when *everyone* runs the same rule, a vol spike makes *everyone* de-lever at once, and all that synchronized selling pushes prices down further, which raises realized vol further, which forces more de-leveraging — a reflexive feedback loop. The 2018 "Volmageddon" and the 2024 yen-carry unwind both had this flavor: a crowded, vol-sensitive trade where the de-risking itself became the shock. Vol targeting is a sound *individual* discipline that can become a *collective* hazard when it's crowded, a tension explored from the strategic angle in the [crowded-trades exit game](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge). The lesson is not to abandon the rule but to know that in a crisis your model is doing the same thing as a thousand others, and to size for that.

## Common misconceptions

**"Holding the same position means holding the same risk."** This is the foundational error the whole post exists to correct. Risk is dollars *times* volatility, and volatility changes constantly. An untouched \$100,000 position in an asset whose vol goes from 10% to 20% has gone from \$10,000 of risk to \$20,000 of risk — your bet *doubled* while you did nothing. Same position, double the risk. The statement lies; the dollar volatility tells the truth.

**"Vol targeting predicts the market — it cuts before crashes."** It predicts nothing. It is purely *reactive*: it responds to volatility that has already shown up in the data. It cuts exposure in stormy regimes only because volatility *clusters* — today's high vol tends to persist into tomorrow — so reacting to yesterday's storm usefully sizes you down for today's. But it has zero ability to see a bolt-from-the-blue gap coming; as Figure 7 shows, it stays fully levered straight into the shock and de-levers only afterward. It is a regime-following tool, not a crystal ball.

**"More rebalancing means better risk control."** Past a point, no. Rebalancing daily versus weekly versus inside a band barely changes how well your risk tracks the target — the risk control is nearly identical — but it changes your turnover, and therefore your costs, enormously. From Figure 6, going from a band to a daily rebalance roughly *doubled* the turnover (from ≈570% to ≈837% of capital over two years) for essentially the same risk profile. The extra trading is pure cost, no benefit. Rebalance only when the drift is big enough to matter.

**"A 10% vol target means I can never lose more than 10%."** No. A volatility *target* describes the typical, year-in-year-out size of your swings — not a loss limit and not a worst case. A 10% vol strategy will, in a bad-but-normal year, draw down 15%–20%, and in a genuine tail event can lose far more, because (a) returns are fat-tailed so big moves happen more than the normal distribution implies, and (b) the realized-vol lag means a gap hits you before you've de-levered. The target is a *risk dial*, not a *stop-loss*. Confusing the two is how people get blindsided by a 30% loss on a "10% vol" book.

**"Vol targeting always beats buy-and-hold."** Not in return, often, and not always. Because it cuts exposure in storms, it usually *gives up some upside* during the violent recoveries that follow storms (which are themselves high-vol) — you're small right when the market snaps back hardest. In Figure 4 the targeted book ended a bit *behind* the fixed book in raw dollars. What it reliably improves is the *shape* of the ride — shallower drawdowns, steadier vol, better risk-adjusted return (Sharpe) — not necessarily the headline number. If you only judge by total return in a calm decade, you'll conclude it "underperformed." That misses the entire point, which is survival across the decade that *isn't* calm.

## How it shows up in real markets

Volatility targeting isn't an academic toy — it is, in some form, how a huge share of the systematic world sizes positions, and its strengths and failure modes have played out in public.

**The CTAs and risk-parity funds (2010s).** Managed-futures funds and risk-parity portfolios run volatility targeting at industrial scale, sizing dozens of futures markets to equal risk and scaling the whole book to a target vol. For most of the 2010s this delivered exactly what the theory promises: smooth, on-target risk and shallow drawdowns relative to a static book — the equal-risk and steady-vol payoffs from Figures 3 and 4, lived out across a real fund's track record. It is the operating system underneath the [all-weather and risk-parity approach](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime).

**Volmageddon, February 5, 2018.** The short-volatility complex was, in effect, a giant crowded vol-sensitive trade. When the VIX jumped about 20 points — from 17.3 to 37.3, roughly a +116% single-day move, the largest one-day VIX percentage rise on record — the products that had to *buy* volatility to rebalance did so all at once, and one of them, the XIV note, lost about **96%** of its value after the close and was terminated the next day. The mechanism was precisely the reflexive de-leveraging loop: a vol spike forced synchronized rebalancing that *amplified* the vol spike. Vol-sensitive sizing is sound individually and combustible when crowded.

**COVID, February–March 2020.** The fastest bear market on record sent the VIX to a record **82.69** close on March 16 and drove the S&P 500 down about **34%** peak-to-trough in roughly a month. Here both faces of vol targeting showed at once. Books that had been running it were already somewhat de-risked as vol climbed through late February — a real benefit. But the speed of the move meant the realized-vol lag bit hard: the worst single days arrived before the estimates fully caught up, so even vol-targeters took real losses on the way down, then sat de-levered through part of the violent April–May recovery. The smooth-the-slope benefit and the sell-the-bottom cost, in the same event.

**The 2024 yen-carry unwind, August 5, 2024.** A crowded funding-carry trade unwound in days; the Nikkei fell about **12.4%** in a single session — its worst day since 1987 — and the VIX spiked intraday to about **65.7**. Vol-sensitive and leverage-sensitive strategies de-levered into the air pocket, and the de-leveraging itself was part of the shock. Like 2018, it is a reminder that a vol-target rule shared by thousands of books becomes a *collective* behavior, and the collective can move the very prices each book is reacting to.

The pattern across all four: volatility targeting reliably delivers steady risk and shallower drawdowns in the *slow* regime changes it was built for, and reliably gets caught flat-footed by *sudden* gaps and by its own crowdedness. It is a discipline that handles the weather and struggles with the lightning — which is exactly why it belongs in a kit alongside tail hedges, not as a standalone shield.

## The volatility-targeting playbook

Concrete rules, so this is a practice and not just a theory. The whole point is to keep your risk constant so you survive long enough to compound — here is how to actually run it.

- **Pick a volatility target you can *survive*, not the one that maximizes backtested return.** A target you can hold through a bad year without panicking is worth more than a higher one you'll abandon at the bottom. For most people running real money, **10%–15% annualized** is a sane, sleep-at-night range. Treat the target as a survival decision: ask "what drawdown does this imply in a bad year, and can I hold through it?" (Roughly, expect a bad-year drawdown of 1.5×–2× your vol target.) If the answer is no, lower the target.

- **Estimate realized vol over a window that matches your horizon — and bias it to react.** A short window (20 days) is responsive but jumpy and trades a lot; a long window (60–90 days) is smooth but slow to register a regime change. A common, robust choice is an **exponentially-weighted estimate with a ~30–60 day effective span**, which reacts faster to recent moves than a flat window without overreacting to a single day. Whatever you choose, annualize it the same way every time (×√252) and never let the denominator go to zero — floor it.

- **Cap leverage, and floor it.** Set a hard **maximum leverage** (commonly 1.5×–3×) so a freakishly calm patch can't load you into a giant position right before calm ends. Set a **minimum** (don't go fully flat) so you don't completely miss the recovery after a storm. The caps are where you make the method robust to its own arithmetic.

- **Rebalance inside a band, not on a clock.** Don't re-size for every tiny vol wiggle. Trade only when your leverage has drifted more than, say, **20% away** from target (or when your dollar size is off by more than some threshold). The band gets you ~90% of the risk control for a fraction of the turnover — from the examples above, roughly a third less trading and a third less cost, for essentially the same risk profile.

- **Know what it does *not* protect you from, and cover that separately.** Vol targeting smooths the *slope* — the gradual shift from calm to stormy. It does nothing for the *cliff* — the overnight gap, the limit-down open, the shock that prints before your estimate can move. For the cliff you need forward-looking, convex protection: options, a put overlay, an explicit tail hedge. Pair the two; don't expect one to do the other's job.

- **Remember you're in a crowd.** In a real crisis, your vol-target rule is being run by thousands of other books, all de-levering at the same time. Size for the possibility that the de-leveraging *itself* moves the market against you, and don't assume you can exit at the price on the screen. The rule is sound; the crowd running it is a risk in its own right.

- **The bottom line.** Size by the risk a position carries, not the dollars it costs. Set a vol target you can survive, estimate vol sensibly, cap your leverage, rebalance in a band, and bolt on tail protection for the gaps. Do that, and the market stops being able to quietly hand you your biggest bet on its worst day — which is, in the end, just one more way of making sure you're still in the game when your edge finally pays.

### Further reading

- [The Kelly criterion: how much to bet when you have an edge](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) — the growth-optimal answer to *how big*, which vol targeting operationalizes as a leverage dial.
- [Volatility and why it is not risk](/blog/trading/risk-management/volatility-and-why-it-is-not-risk) — the foundations of the number we're targeting, and the ways it can still mislead.
- [Risk parity: sizing for equal risk, not equal money](/blog/trading/risk-management/risk-parity-sizing-equal-risk-not-equal-money) — extends "equal risk" from one position to a whole asset-class portfolio.
- [All-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) — the allocation view of sizing every bet to equal risk across economic environments.
- [The variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the forward-looking, convex protection that covers the gap risk vol targeting can't.
