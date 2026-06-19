---
title: "Fractional Kelly and Optimal-f: Betting Less to Sleep at Night"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why every real trader bets a fraction of full Kelly: the growth curve is flat near the top, full Kelly is wildly volatile, and you never actually know your edge."
tags: ["risk-management", "kelly-criterion", "fractional-kelly", "optimal-f", "position-sizing", "bet-sizing", "drawdown", "estimation-error", "volatility"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **One-sentence thesis:** Full Kelly maximises long-run growth only if you *know* your edge exactly, and the price of that maximum is a path so violent that no human survives it, so real traders bet a *fraction* of Kelly.
> - The growth curve is **flat near the top**: cutting your bet from full to half Kelly costs you only about a quarter of the growth.
> - But the **risk falls fast**: half Kelly roughly halves the volatility of your equity path and cuts the variance by three-quarters.
> - That trade — keep ≈75% of the growth for ≈50% of the swing — is why "half Kelly" is the practitioner's rule of thumb.
> - The deeper reason is **estimation error**: you never know your true edge, you estimate it, and an overestimated edge turns "full Kelly" into silent over-betting.
> - **Default to a quarter-to-half Kelly**, and scale further down the less certain you are of the edge — survival first, growth second.

A professional blackjack team in the 1990s had a real, measurable edge: counting cards gave them roughly a 1% advantage per hand. They had done the math. They knew the Kelly criterion. And they sized their bets at *one-fifth* of what full Kelly told them to bet.

Why throw away four-fifths of your theoretical growth rate when you have a genuine, statistically verified edge? Because the people who ran those teams had watched what full Kelly does to a bankroll in practice. They had seen players with a real edge go broke not because the edge failed, but because a normal, expected run of bad luck — entirely consistent with their 1% advantage — took a full-Kelly bankroll down 60%, 70%, sometimes to zero, before the edge had time to assert itself. The edge was real. The *path* killed them.

This is the single most important correction to the Kelly criterion, and it is the gap between the formula and the practitioner. In [the previous post](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) we derived full Kelly from first principles: the bet fraction that maximises the long-run growth rate of your capital. That post is correct. This post is about why almost nobody who knows what they are doing actually bets it. The answer comes down to two facts that the textbook formula hides, and that Figure 1 puts side by side: the growth curve barely changes as you back off from full Kelly, while the risk of the path falls off a cliff.

![Two-axis chart showing long-run growth as a parabola that is nearly flat near full Kelly while the equity-path volatility rises in a straight line, with full Kelly and half Kelly marked](/imgs/blogs/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night-1.png)

Look at the green growth curve and the amber volatility line. As you scale your bet from zero up toward full Kelly, growth climbs steeply at first and then *flattens* — near the peak, growth is almost insensitive to how much you bet. Volatility, meanwhile, climbs in a straight line: every extra unit of bet size adds the same extra wobble to your equity. At half Kelly you sit at the green dot: you have given up only a quarter of the growth (you keep 75%) but you have halved the volatility of the ride. That asymmetry is the whole argument. By the end of this post you will be able to derive every number on that chart, and you will understand why the prudent default is not full Kelly, not even half, but somewhere in the quarter-to-half range.

## Foundations: the building blocks you need first

Before we cut the bet, let us make sure every term is solid. This post leans on four ideas. If you have read the Kelly post, two of them are review; read them anyway, because the precise definitions are what make the fractional-Kelly argument airtight.

**Bet fraction.** Throughout, $f$ is the fraction of your *current* bankroll you put at risk on each opportunity. If $f = 0.10$ and you have a \$100,000 account, you risk \$10,000 this round. Crucially, the fraction is of the *current* balance, not the starting balance — this is "fixed-fractional" sizing, and it is what lets a Kelly bettor compound. When the account grows to \$120,000, the same $f = 0.10$ now risks \$12,000; when it shrinks to \$80,000, it risks \$8,000. Betting a fixed fraction is what makes ruin (hitting exactly zero) impossible in a frictionless model and what makes growth *geometric* rather than arithmetic.

**The edge.** An edge is a positive expected value. For a clean binary bet, we describe it with two numbers: $p$, the probability you win, and $b$, the payoff ratio (you win $b$ times your stake on a win, lose your stake on a loss). The running example in this post is a bet with $p = 0.55$ and $b = 1$ — a coin that lands your way 55% of the time, paying even money. That is a real, exploitable 10% edge per dollar wagered: your expected profit per \$1 staked is $0.55 \times \$1 - 0.45 \times \$1 = \$0.10$. It is deliberately modest, because modest edges are what real strategies have.

**Full Kelly.** The Kelly fraction $f^{*}$ is the bet size that maximises the expected *logarithm* of your wealth — equivalently, the long-run geometric growth rate. For a binary bet the formula is

$$f^{*} = \frac{p(b+1) - 1}{b} = \frac{pb - (1-p)}{b}.$$

For our $p = 0.55$, $b = 1$ bet, $f^{*} = (0.55 \times 1 - 0.45)/1 = 0.10$. Full Kelly says: bet 10% of your bankroll every time. We *maximise log wealth* rather than expected wealth because wealth compounds multiplicatively — a sequence of gains and losses multiplies together — and the quantity that adds up across a multiplicative process is the log. The post on [the asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) is the deep version of why: a −50% move followed by a +50% move does not return you to even, because $0.5 \times 1.5 = 0.75$. Growth lives in the geometric mean, and the geometric mean is the average of the logs.

**Growth rate.** The long-run growth rate at bet fraction $f$ is the expected log-return per bet:

$$g(f) = p \ln(1 + bf) + (1-p)\ln(1 - f).$$

On a win your bankroll multiplies by $(1 + bf)$; on a loss by $(1 - f)$. We take logs and average. This function is the hero of the whole post. It is a smooth hump: zero at $f = 0$ (you bet nothing, you grow nothing), rising to a maximum at $f = f^{*}$, then falling back through zero and going negative when you over-bet. *Everything* about fractional Kelly is a statement about the shape of this hump near its top.

**Volatility of the path versus risk of ruin.** Two different things, and keeping them separate is the key to this post. The *volatility* of your equity path is how much it wobbles bet-to-bet — the standard deviation of your log-returns. The *risk of ruin* is the chance the path falls so far it never recovers. As we will see, volatility scales *linearly* with bet size (double the bet, double the wobble), while the deep-drawdown and ruin risks scale much *faster* than linearly, because drawdowns compound. That difference in scaling — risk growing faster than growth near the top — is the mathematical engine of fractional Kelly. For the full treatment of why a wobble and a wound are different things, see [volatility and why it is not risk](/blog/trading/risk-management/volatility-and-why-it-is-not-risk).

With those five terms nailed down, we can ask the real question: if full Kelly maximises growth, why on earth would anyone bet less?

## The growth curve is flat near the top

Start with the shape of $g(f)$. Near a smooth maximum, every function looks like a downward parabola — that is just calculus, the second-order Taylor expansion. So near $f^{*}$, the growth rate behaves like

$$g(c \cdot f^{*}) \approx g(f^{*}) \cdot (2c - c^{2}),$$

where $c$ is the fraction of full Kelly you bet ($c = 1$ is full Kelly, $c = 0.5$ is half Kelly, and so on). The factor $2c - c^{2}$ is the share of the maximum growth you capture. Let us tabulate it, because these four numbers are the spine of the post:

- $c = 0.25$ (quarter Kelly): $2(0.25) - 0.25^{2} = 0.5 - 0.0625 = 0.44$. You keep **44%** of the growth.
- $c = 0.50$ (half Kelly): $2(0.5) - 0.5^{2} = 1.0 - 0.25 = 0.75$. You keep **75%** of the growth.
- $c = 1.00$ (full Kelly): $2(1) - 1 = 1.0$. You keep **100%** — by definition, this is the peak.
- $c = 2.00$ (double Kelly): $2(2) - 4 = 0$. You keep **0%**. Betting twice Kelly grows your capital at exactly zero in the long run.

Run the exact $g(f)$ formula for our $p = 0.55$, $b = 1$ edge and you get growth-fraction values of 0.437, 0.749, 1.000, and a small *negative* number at double Kelly — the parabola is an excellent approximation. The headline is the half-Kelly row: **back your bet down by half and you keep three-quarters of the growth.** You are throwing away 25% of your compounding to halve your bet. The reason that is a bargain is everything that happens to risk, which we turn to next — but first absorb the symmetry of the parabola. The growth rate is the *same* at half Kelly ($c = 0.5$) and at one-and-a-half Kelly ($c = 1.5$): both give $2c - c^{2} = 0.75$. One of those two — under-betting — has half the volatility; the other — over-betting — has three times it. Same growth, wildly different risk. Only a fool picks the volatile one, and yet over-betting is the single most common sizing error in trading.

#### Worked example: half Kelly on the \$100,000 account

You have a \$100,000 account and the $p = 0.55$, $b = 1$ edge, so $f^{*} = 0.10$.

- **Full Kelly:** stake $0.10 \times \$100{,}000 = \$10{,}000$ per bet. Long-run growth rate $g(0.10) = 0.55\ln(1.10) + 0.45\ln(0.90) = 0.55(0.0953) + 0.45(-0.1054) = 0.00501$ per bet. Over 500 bets that compounds to a factor of $e^{0.00501 \times 500} = e^{2.50} \approx 12.2\times$, so \$100,000 becomes roughly \$1,220,000 *on the median path*.
- **Half Kelly:** stake $0.05 \times \$100{,}000 = \$5{,}000$ per bet. Growth rate $g(0.05) = 0.55\ln(1.05) + 0.45\ln(0.95) = 0.55(0.0488) + 0.45(-0.0513) = 0.00375$ per bet. Over 500 bets, $e^{0.00375 \times 500} = e^{1.88} \approx 6.5\times$, so \$100,000 becomes roughly \$650,000 on the median path.

So full Kelly's median end-wealth is about \$1.22M and half Kelly's is about \$0.65M — full Kelly nearly doubles it. That looks like a big sacrifice for half Kelly. But notice the growth *rates*: 0.00375 versus 0.00501 is exactly the 75% ratio. The end-wealth gap looks dramatic only because it is the small per-bet difference compounded over a long horizon, and — as the next sections show — the full-Kelly path that *gets* to \$1.22M spends much of the journey down 50% or more, which most accounts (and most people) cannot endure.

*The growth you give up by halving your bet is real but small; the volatility you shed is large — that is the entire case for betting less.*

## Why the risk falls fast: full versus half versus quarter Kelly

Now the other half of the trade. We have seen growth barely moves; here is what happens to the thing you actually live through — the drawdown. A drawdown is the percentage drop from a previous high-water mark in your account; it is the pain you feel in real time, the number that makes you doubt your strategy and the number a fund's investors see on their statements. (The dedicated treatment is [drawdown and the underwater curve](/blog/trading/risk-management/drawdown-and-the-underwater-curve-the-risk-you-actually-feel).) The cleanest way to see how bet size drives drawdown is to simulate: take the *exact same* 10% edge, size it at full, half, and quarter Kelly, run thousands of equity paths, and record the worst peak-to-trough drawdown each path suffers. Figure 2 stacks those three distributions.

![Three overlaid histograms of worst drawdown across six thousand simulated runs for full, half, and quarter Kelly, with full Kelly shifted far to the right toward deep drawdowns](/imgs/blogs/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night-2.png)

The three distributions barely overlap. Quarter Kelly clusters around a 30% worst drawdown — uncomfortable but survivable. Half Kelly centres near 53%. And full Kelly — the *growth-optimal* bet, the one the textbook tells you to make — has its mass piled up around an 83% worst drawdown, with a fat tail running all the way to wipeout. Read that again: the bet that maximises your long-run growth rate will, on a *typical* run, take you down more than 80% from a high before you recover. The full-Kelly bettor is not unlucky when this happens. They are *exactly on plan*. Full Kelly bakes in routine 80% drawdowns the way a long road trip bakes in gas stops.

This is the practical heart of the matter. The asymmetry-of-losses post showed that an 80% drawdown requires a +400% gain to recover; a 90% drawdown requires +900%. Full Kelly drives you into precisely the region of the recovery curve where the math becomes monstrous. Half Kelly's median 53% drawdown needs +113% to recover — hard, but a doubling, not a five-bagger. Quarter Kelly's 30% needs +43%, which a real edge produces in a reasonable time.

#### Worked example: the drawdown each sizing puts you through

Take the \$100,000 account through a representative bad stretch — the kind of losing run a 55% edge throws off all the time (a 55% win rate means 45% losers; strings of losses are normal).

- **Full Kelly (10% bets), worst drawdown ≈ 80%:** the account falls from a peak of, say, \$300,000 down to about \$60,000. To get back to \$300,000 you need to gain $\$300{,}000 / \$60{,}000 - 1 = +400\%$. At a growth rate of 0.005 per bet that is roughly $\ln(5)/0.005 \approx 320$ bets of pure climbing just to undo the hole.
- **Half Kelly (5% bets), worst drawdown ≈ 53%:** the account falls from \$300,000 to about \$141,000. Recovery needs $+113\%$ — about $\ln(2.13)/0.00375 \approx 200$ bets.
- **Quarter Kelly (2.5% bets), worst drawdown ≈ 30%:** the account falls from \$300,000 to \$210,000. Recovery needs $+43\%$ — about $\ln(1.43)/0.0022 \approx 165$ bets.

The full-Kelly bettor has the highest growth rate yet spends the *most* bets digging out of holes, because the holes are so much deeper. The compounding advantage of full Kelly is constantly being squandered on recovery.

*Full Kelly's growth edge is real, but it is paid for with drawdowns so deep that most of it is spent climbing back out of craters you did not have to fall into.*

## The growth-versus-risk frontier bends backward

Plot growth against the volatility of the equity path as you scale the Kelly fraction from zero up to twice Kelly, and you get the single most clarifying picture in bet sizing. Figure 3 is that frontier.

![Curve of long-run growth against equity-path volatility as the Kelly fraction scales from zero to two times Kelly, rising along a green arc to full Kelly then bending backward along a red dashed arc](/imgs/blogs/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night-3.png)

Read it from the origin. As you increase your bet (moving right, adding volatility), growth climbs steeply at first — the early units of risk are richly rewarded. Quarter Kelly already captures 44% of the maximum growth for only 25% of the volatility. Half Kelly captures 75% for 50%. Then the curve *flattens* as it approaches full Kelly: the last stretch of growth, from 75% up to 100%, costs you the entire second half of your volatility budget. And here is the part that should change how you think: **past full Kelly the curve bends backward.** Beyond the peak you are paying more volatility *and* getting less growth. At 1.5× Kelly you have the same growth as half Kelly (the 75% we computed) but *three times* the volatility. Every point on the red dashed arc is strictly dominated by a point on the green arc with less risk and equal or more growth.

There is a precise analogy here to the [mean-variance efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) from portfolio theory: only the upward-sloping part of a risk-return curve is "efficient", and the backward-bending part is a region no rational allocator would ever choose. The Kelly frontier has exactly this shape. Full Kelly sits at the very tip — the maximum-growth point — but the tip is also where the curve is *flattest*, meaning you pay the most risk per unit of marginal growth. Practitioners live on the steep lower-left of the green arc, in the quarter-to-half region, where every unit of volatility you accept still buys you a generous slice of growth. The flat top is for people who care about the last basis point of growth and nothing about the path. No such people survive long enough to collect it.

#### Worked example: the marginal trade near the top

On the \$10,000,000 book, compare two moves along the frontier.

- **Moving from quarter to half Kelly:** growth rises from 44% to 75% of maximum (a gain of 31 percentage points of growth) while volatility rises from 25% to 50% (a 25-point increase). You are buying 31 points of growth for 25 points of volatility — a *great* trade. On the book, growth per year might go from roughly 1.9% to 3.3% while the annualised swing roughly doubles. Worth it, if you can stomach the swing.
- **Moving from half to full Kelly:** growth rises from 75% to 100% (a gain of only 25 points) while volatility rises from 50% to 100% (a 50-point increase). Now you are paying 50 points of volatility for 25 points of growth — *half the growth per unit of risk* of the previous move. The marginal trade has gotten twice as bad.

The diminishing return is the whole story. Each extra step toward full Kelly costs more volatility and buys less growth. Somewhere on the way up — for most people around a quarter to a half — the marginal growth stops being worth the marginal risk.

*The frontier is steep where you should live and flat where the textbook tells you to stand; full Kelly is the worst growth-per-risk trade on the whole curve.*

## The same story in continuous time: Kelly is a leverage choice

The binary-bet version is the cleanest way to *see* fractional Kelly, but most traders do not place discrete even-money bets — they hold a continuously-varying position in an asset or strategy. The good news is that the entire argument carries over essentially unchanged, and in the continuous version it becomes even more transparent, because "Kelly fraction" turns into "leverage" and the parabola turns into a simple, exact formula.

Describe a strategy by two numbers: its arithmetic expected return per year $\mu$ (the edge) and its volatility per year $\sigma$ (the risk). If you run the strategy at leverage $L$ — meaning you hold $L$ dollars of the strategy for every dollar of capital — your geometric growth rate is approximately

$$g(L) \approx \mu L - \tfrac{1}{2}\sigma^{2}L^{2}.$$

The first term is the edge scaled up by leverage; the second is the *volatility drag* — the cost that the asymmetry of losses imposes on any volatile, compounding asset, growing with the *square* of leverage. This is the continuous twin of the binary growth formula, and it has the same downward-parabola shape. Maximise it (take the derivative, set it to zero) and you get the continuous Kelly leverage:

$$L^{*} = \frac{\mu}{\sigma^{2}}.$$

This is one of the most useful formulas in all of risk management. It says: bet leverage equal to your edge divided by your variance. A high edge justifies more leverage; high volatility — which appears *squared* — punishes it hard. And just like the binary case, the growth function is flat near its peak and symmetric, so half-Kelly leverage ($L^{*}/2$) captures exactly 75% of the maximum growth, while double-Kelly leverage ($2L^{*}$) drives growth back to zero. The volatility-drag term, growing as $L^{2}$, is what bends the curve back down: past the optimum, each extra unit of leverage adds more drag than edge.

#### Worked example: Kelly leverage on a real strategy

A systematic strategy has an expected return of 8% per year and a volatility of 16% per year — a respectable but unremarkable edge, with a Sharpe ratio of $0.08 / 0.16 = 0.5$.

- **Full Kelly leverage:** $L^{*} = \mu / \sigma^{2} = 0.08 / 0.16^{2} = 0.08 / 0.0256 = 3.125$. Full Kelly says hold \$3.125 of the strategy per \$1 of capital — a 3.1× levered position. The resulting growth rate is $g(3.125) = 0.08(3.125) - 0.5(0.16^{2})(3.125^{2}) = 0.25 - 0.125 = 0.125$, or 12.5% per year. But the *volatility* of the levered equity is $\sigma L = 0.16 \times 3.125 = 0.50$ — a 50%-vol portfolio. On the \$10,000,000 book that is a position with annual swings of \$5,000,000 and routine 50%+ drawdowns. Almost no one can hold that.
- **Half Kelly leverage:** run at $L = 1.5625$ instead. Growth is $g(1.5625) = 0.08(1.5625) - 0.5(0.0256)(1.5625^{2}) = 0.125 - 0.03125 = 0.09375$, or 9.375% per year — exactly **75%** of the full-Kelly growth, as promised. The portfolio volatility is now $0.16 \times 1.5625 = 0.25$, a 25%-vol portfolio: half the swing, a quarter of the variance. On the book, the annual swing drops from \$5,000,000 to \$2,500,000 while the growth only falls from 12.5% to 9.375%.
- **Over-levered (double Kelly, $L = 6.25$):** $g(6.25) = 0.08(6.25) - 0.5(0.0256)(6.25^{2}) = 0.50 - 0.50 = 0$. A 100%-vol portfolio that compounds at *zero* in the long run despite holding six times the edge. This is the continuous twin of betting double Kelly on the coin, and it is exactly how leverage blows up funds with genuinely good strategies.

*In continuous time Kelly is just a leverage number — edge over variance — and the same flat-topped parabola means half the leverage keeps three-quarters of the growth, so the prudent trader runs well below the leverage the formula names.*

This continuous view is also where the most common professional blow-up lives. A strategy that looks safe at low leverage — a market-neutral book, a carry trade, a relative-value spread — has a Kelly leverage that can be very high precisely *because* its volatility looks low in calm times. Run it at full Kelly leverage and a single regime change, where the true $\sigma$ jumps and correlations move against you, drives the $\sigma^{2}L^{2}$ drag term through the roof and the position implodes. Fractional Kelly on the leverage — running at a third to a half of $L^{*}$ — is the buffer that survives the moment your estimate of $\sigma$ turns out to have been measured in the calm before the storm. We will see this exact failure in the case studies below.

## The deeper reason: you do not know your edge

Everything so far assumed you *know* your edge — that $p$ is exactly 0.55. If that were true, the only argument for fractional Kelly would be the volatility-of-the-path argument above, and a sufficiently patient, deep-pocketed, emotionless bettor could rationally choose full Kelly and ride out the 80% drawdowns. But here is the fact that turns "fractional Kelly is prudent" into "fractional Kelly is *correct*": **you never know your edge.** You *estimate* it, from a finite, noisy sample of past results, and your estimate has error bars. Figure 5 lays out the two worlds side by side — the textbook world where the edge is known, and the real world where it is a guess.

![Before and after comparison contrasting a textbook world where the edge is known exactly and full Kelly is optimal against a real world where the edge is a noisy estimate and estimation error forces a scaled-down bet](/imgs/blogs/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night-5.png)

In the left column, you are *told* the edge: $p = 0.55$, exactly, with certainty. Full Kelly's 10% bet is genuinely optimal, and there is nothing to hedge against, because the formula is exact. That is a fantasy. In the right column — reality — you ran a backtest, or traded a strategy for a few hundred trades, and *measured* a 55% win rate. But a measured 55% from a sample of, say, 400 trades has a standard error of roughly $\sqrt{0.55 \times 0.45 / 400} \approx 2.5$ percentage points. Your true edge could comfortably be 52.5% or 57.5%. And here is the asymmetry that bites: Kelly is *very* sensitive to the edge. If you flatter yourself and assume $p = 0.60$ when the truth is 0.55, the Kelly formula tells you to bet $f^{*} = (0.60 - 0.40)/1 = 0.20$ — which is *twice* the true Kelly fraction. You think you are betting full Kelly; you are actually betting double Kelly, sitting at the zero-growth point of the parabola.

This is the subtle, lethal part. Estimation error does not average out in your favour. Because the growth curve is *asymmetric* around the peak — it falls off much faster on the over-betting side than it rises on the under-betting side — overestimating your edge by some amount hurts you more than underestimating it by the same amount helps you. When your edge is uncertain, the expected growth is maximised by betting *less* than your point estimate suggests. The size of the haircut grows with your uncertainty. A rough and widely used result: if your edge estimate has meaningful error, the growth-optimal bet is roughly *half* of the naive Kelly fraction — which is exactly the half-Kelly rule of thumb, now justified on completely different grounds. Half Kelly is not just "the volatility is nicer"; it is the *actual* growth-optimal bet once you admit you do not know your edge.

It is worth being precise about *why* the error does not cancel, because the asymmetry is the whole point and it is easy to wave past. Suppose your true edge is fixed but your estimate of it is a random variable scattered around the truth — sometimes too high, sometimes too low, symmetrically. You then bet Kelly *as if your estimate were correct*. On the runs where you overestimated, you over-bet, and you land on the steep right side of the parabola where growth falls fast. On the runs where you underestimated, you under-bet, and you land on the gentle left side where growth falls slowly. Average those two cases and the over-betting runs drag the average down more than the under-betting runs prop it up. The expected growth, taken over your estimation error, is therefore *strictly lower* than the growth you would get if you knew the edge — and the gap is recovered by deliberately shading your bet downward, which moves you off the dangerous right side. The mathematics is a direct consequence of the curve's concavity; the more uncertain your edge, the wider the scatter, the more the right side dominates, and the deeper the optimal haircut. A strategy whose edge you have measured over five years of out-of-sample trading deserves a small haircut; one you fit on a hundred backtest trades deserves a brutal one.

There is a second, compounding reason the haircut should be deep: **your edge is not even constant.** The binary model treats $p$ as a fixed number, but real edges decay. Other traders find the same inefficiency and arbitrage it away; the market regime that made your signal work shifts; your own execution degrades as your size grows. So the edge you measured in the past is not just a noisy estimate of a fixed true edge — it is a noisy estimate of an edge that is probably *already smaller* than it was during your sample. Both effects push the same direction: bet less than the formula says, because the formula's input is optimistic in two distinct ways at once.

#### Worked example: the cost of a flattering 5-point error

You believe your edge is $p = 0.60$ (so you bet $f = 0.20$), but the truth is $p = 0.55$. On the \$100,000 account you stake \$20,000 per bet.

- Your *realised* growth rate, scored under the true edge, is $g(0.20) = 0.55\ln(1.20) + 0.45\ln(0.80) = 0.55(0.1823) + 0.45(-0.2231) = 0.1003 - 0.1004 = -0.0001$ per bet. **Negative.** You believe you are compounding at the maximum rate; you are actually shrinking, slowly, forever.
- The true full-Kelly bet was \$10,000 (a 0.005 growth rate). By overestimating your win probability by 5 points, you doubled your bet to \$20,000 and converted a healthy positive growth rate into a *negative* one.
- Now suppose you had instead bet *half* of your (wrong) Kelly estimate: $f = 0.10$, staking \$10,000. By dumb luck that equals the true full Kelly, and your realised growth is the full 0.005. Betting fractionally protected you from your own estimation error.

*The fractional haircut is not timidity; it is the rational response to the fact that your edge is a guess, and an overestimated edge silently turns full Kelly into ruinous over-betting.*

Figure 4 makes the whole estimation-error story into one curve: how much real growth you keep as a function of how badly you overestimate your edge.

![Curve of realised long-run growth against the number of percentage points by which the win probability is overestimated, starting at one hundred percent and crossing zero at a five-point overestimate](/imgs/blogs/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night-4.png)

At zero overestimate you get the full 100% of growth. Inflate your win probability by 2.5 points and your realised growth drops to about 75% — you have effectively talked yourself into betting 1.5× Kelly. Inflate it by 5 points and realised growth hits *zero*: your "full Kelly" is really 2× Kelly. Beyond that, growth goes negative and ruin risk climbs. The curve is steep and unforgiving on the right. Now overlay the reality that your 55% estimate has 2.5-point error bars: a single standard deviation of estimation error is enough to drag you from 100% growth down to 75%, and two standard deviations to zero. *This is why you halve the bet before you do anything else* — the halving is a buffer sized to your uncertainty about the edge itself.

## The half-Kelly bargain, stated cleanly

Pull the two threads together — flat growth, fast-falling risk, plus estimation error — and the half-Kelly rule of thumb pops out as a remarkably clean bargain. Figure 6 is the two-bar summary.

![Grouped bar chart comparing full Kelly at one hundred percent against half Kelly across long-run growth, path volatility, and path variance, showing half Kelly keeps seventy-five percent of growth at twenty-five percent of variance](/imgs/blogs/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night-6.png)

Three bars, each expressed as half Kelly's value relative to full Kelly:

- **Long-run growth:** half Kelly keeps **75%** (the $2c - c^{2}$ result at $c = 0.5$).
- **Path volatility:** half Kelly is **50%** of full Kelly's, because volatility scales *linearly* with bet size — half the bet, half the standard deviation of your returns.
- **Path variance:** half Kelly is **25%** of full Kelly's, because variance is the *square* of volatility — half the volatility means a quarter of the variance.

State it as a sentence and it sells itself: **three-quarters of the growth, half the volatility, a quarter of the variance.** You give up 25% of your compounding to remove 75% of the variance of the ride. If you measure risk by variance — and much of finance does, because variance is what drives drawdowns and what investors feel — half Kelly is removing three units of risk for every one unit of growth it sacrifices. There is no other knob in trading with a payoff ratio that lopsided.

#### Worked example: half Kelly on the \$10,000,000 book

A fund runs a \$10,000,000 book with the $p = 0.55$, $b = 1$ edge.

- **Full Kelly:** risks \$1,000,000 per bet (10% of the book). Annualised, suppose this delivers a 5.0% expected growth rate with a 20% annualised volatility of the equity curve.
- **Half Kelly:** risks \$500,000 per bet (5%). Growth falls to $0.75 \times 5.0\% = 3.75\%$. Volatility falls to $0.50 \times 20\% = 10\%$. Variance falls from $0.20^{2} = 0.04$ to $0.10^{2} = 0.01$ — a quarter.
- The Sharpe-style ratio of growth to volatility actually *improves* slightly as you move down from full Kelly, because near the top growth is flat while volatility keeps falling linearly. The book is taking less risk *and* earning more growth per unit of risk. Investors who see a 10%-vol fund with a 3.75% edge and an 53% worst drawdown will stay; investors who see a 20%-vol fund with an 80% drawdown will redeem at the bottom and lock in the loss — which is its own path to ruin.

*Half Kelly is the rare trade where cutting your risk by three-quarters costs you only a quarter of your growth and even improves your growth-per-unit-of-risk — it is close to a free lunch in a discipline that has almost none.*

## What optimal-f adds, and where it goes wrong

There is a closely related framework you will meet in trading circles called **optimal-f**, popularised by Ralph Vince. It is worth understanding both for what it gets right and for the trap it sets. Optimal-f asks the same question as Kelly — what fixed fraction of capital maximises long-run geometric growth? — but instead of plugging $p$ and $b$ into a formula, it searches over your *actual historical trade results* for the fraction $f$ that would have produced the highest terminal wealth on that exact sequence of trades.

In principle, optimal-f generalises Kelly to arbitrary payoff distributions: real trades are not clean binary bets with a single payoff ratio $b$; they have a whole distribution of outcomes (small wins, big wins, small losses, the occasional disaster). Optimal-f handles that by working directly with the empirical sequence. For a simple binary edge, optimal-f and full Kelly give the same answer. For a realistic, fat-tailed return distribution, optimal-f finds the growth-maximising fraction for *that distribution*. So far, so good.

Here is the trap, and it is the estimation-error problem on steroids. Optimal-f is computed by *maximising over your historical sample* — and your historical sample is finite, noisy, and includes whatever the *worst* loss you happened to experience was. Optimal-f is exquisitely sensitive to that single worst trade: the fraction it recommends is essentially set by the largest historical loss, because betting any larger would have busted you on that one trade. This means optimal-f is **overfit to your sample's worst loss**. Your true worst loss — the one lurking in the tail you have not seen yet — is almost certainly bigger than your historical worst (this is the lesson of [fat tails and the normal distribution trap](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap)). When the as-yet-unseen worse loss arrives, your optimal-f bet, calibrated to the smaller historical worst, is over-betting, and you take a catastrophic hit.

#### Worked example: optimal-f and the worst trade you have not seen

A strategy's backtest over 500 trades shows a worst single-trade loss of 8% of capital. Optimal-f, searching the sample, recommends a fraction that risks about 12% of capital per trade — sized so that even the 8% worst-loss trade, levered up, costs about $0.12 / 0.08 \approx 1.5\times$... and the search pushes right up to the edge of what that worst loss could survive.

- Then a market gap delivers a 16% single-trade loss — twice the historical worst, entirely plausible in a fat-tailed world. At the optimal-f bet size, that 16% raw loss is amplified to roughly a 24% hit to capital on one trade.
- On the \$10,000,000 book, that is a \$2,400,000 loss in a single day, from a strategy whose backtest "never lost more than 8%". Optimal-f did exactly what it was designed to do — maximise growth on the *observed* sample — and that is precisely why it over-bet the unobserved tail.

*Optimal-f is Kelly with the volume turned up: it squeezes the last drop of growth out of your historical sample, which is exactly why it is the most overfit, most fragile sizing rule when the future's worst loss exceeds the past's.*

The practical takeaway is not to abandon optimal-f but to treat its output the way you treat raw full Kelly: as an *upper bound* to be heavily discounted, never a target. Whatever optimal-f or full Kelly tells you, the same answer applies — bet a fraction of it, and the more your number depends on the tail of a finite sample, the deeper the fraction.

There is a further subtlety that catches even sophisticated practitioners. Optimal-f, and Kelly more generally, is a *single-bet* sizing rule — it tells you the fraction for one independent opportunity at a time. Real portfolios hold many positions *simultaneously*, and those positions are correlated. If you size each of ten positions at its own half-Kelly fraction but they all secretly move together — say they are all long-volatility, or all short-duration, or all the same crowded factor — then your *aggregate* exposure is far larger than any single half-Kelly bet, and in a crisis when correlations rush to one, the whole book behaves like a single massively over-Kelly position. This is why portfolio Kelly is computed on the *joint* distribution, using the covariance of the strategies, not by sizing each leg in isolation. The continuous formula generalises cleanly: the vector of Kelly weights is the inverse covariance matrix times the vector of edges, $\mathbf{w}^{*} = \Sigma^{-1}\boldsymbol{\mu}$ — which is, not coincidentally, the same shape as the mean-variance optimal portfolio. The practical defence is the same fractional discipline applied at the book level: compute the joint Kelly exposure, then run the whole portfolio at a fraction of it, and treat your correlation estimates with even more suspicion than your edge estimates, because correlation is the parameter that betrays you exactly when it matters most. Correlation going to one in a crisis is a [risk failure mode covered in its own post](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis); the sizing implication is simply that diversification you cannot trust in a crisis is leverage you did not know you had.

## The \$100,000 account, two paths, one edge

It helps to see it all in a single picture: one account, one edge, two bet sizes, the *exact same* sequence of wins and losses driving both. Figure 7 runs the \$100,000 account through 500 bets at full Kelly and half Kelly, with the only difference being the bet size.

![Two equity curves on a log scale for a hundred thousand dollar account over five hundred bets, the full Kelly path jagged with deep drawdowns and the half Kelly path smoother, both driven by identical coin flips](/imgs/blogs/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night-7.png)

The full-Kelly path (red) is a roller coaster: it lurches upward, then plunges 60% in a losing streak, claws back, plunges again. Its terminal wealth on this particular sequence is higher than half Kelly's — full Kelly *is* the faster grower in the long run — but look at what it costs in white-knuckle drawdowns along the way. The half-Kelly path (green) tracks the same ups and downs, but muted: every drawdown is roughly half as deep, the curve is a smoother ramp, and it still ends comfortably up. The faint paths around each show the spread across other luck draws: the full-Kelly cloud is enormously wider, with paths that drop toward the floor and never recover, while the half-Kelly cloud is tight.

#### Worked example: surviving the same losing streak at two bet sizes

Suppose both accounts hit the same run of four losses in a row partway through.

- **Full Kelly (10% bets):** each loss multiplies the account by 0.90. Four in a row: $0.90^{4} = 0.656$. The account is down **34%** from where it started the streak. On a \$300,000 peak that is a fall to \$197,000 — and that is just *four* losses, which a 45%-loss-rate bet throws off constantly. A run of eight ($0.90^{8} = 0.43$) puts you down 57%.
- **Half Kelly (5% bets):** each loss multiplies by 0.95. Four in a row: $0.95^{4} = 0.815$. Down **18.5%**, to \$244,000. A run of eight ($0.95^{8} = 0.66$) is down 34% — the same damage eight half-Kelly losses do that *four* full-Kelly losses did.
- The kicker: because half Kelly's drawdowns are shallower, it spends far less time underwater, and time underwater is when traders abandon strategies, investors redeem, and risk managers cut limits at the worst possible moment. Survival is not just a mathematical property of the path; it is a *behavioural* one, and shallow drawdowns are what let humans actually stay in the trade.

*Same edge, same luck, half the bet: the half-Kelly account suffers every drawdown the full-Kelly account does, at half the depth, which is the difference between a survivable scare and a career-ending hole.*

## Common misconceptions

**"Full Kelly is optimal, so I should bet it."** Full Kelly is optimal for *one specific objective* — maximising long-run growth rate — *if* you know your edge exactly and have an infinite horizon and zero behavioural constraints. Drop any of those assumptions and full Kelly is over-betting. With a typical edge it produces median worst-drawdowns around 80%, and with realistic estimation error your "full Kelly" is frequently 1.5× or 2× the true Kelly. The growth curve is flat at the top precisely so that backing off costs almost nothing: half Kelly keeps 75% of the growth. Betting full Kelly is choosing an 80% drawdown to buy a 25% growth bump you will mostly squander on recovery.

**"Betting less just means I make less money."** Less than you would on the *median* path, yes — but the median is not what you get. The full-Kelly path that actually plays out spends so much time in deep drawdowns that its realised growth, after the recoveries, is far below the theoretical maximum, and a meaningful fraction of full-Kelly paths simply never recover. Half Kelly keeps three-quarters of the growth for half the volatility and a quarter of the variance, and it improves your growth-*per-unit-of-risk*. You are not making less money for no reason; you are buying a vastly smoother path at a small price, and the smoother path is more likely to deliver its growth.

**"If I know my win rate, there is no estimation error."** You do not know your win rate; you have an *estimate* of it from a finite sample. A 55% win rate measured over 400 trades has roughly a 2.5-point standard error. Because Kelly is so sensitive to the edge — a 5-point overestimate doubles the bet — that sampling error alone is enough to turn full Kelly into double Kelly. The half-Kelly haircut is, in part, exactly the buffer that absorbs this error. The fewer trades your estimate rests on, the larger the haircut should be.

**"Optimal-f is better than Kelly because it uses my real trade data."** Optimal-f uses your real data to *maximise growth on that exact sample*, which means it is overfit to your sample's worst loss. Your future worst loss will, in a fat-tailed world, exceed your historical worst — and at the optimal-f bet size that unseen loss is catastrophic. Optimal-f is the most aggressive, most fragile sizing rule precisely because it extracts the last drop of in-sample growth. Treat its output as an upper bound to discount heavily, never a target.

**"Quarter Kelly is for cowards; serious traders bet bigger."** Quarter Kelly keeps 44% of the maximum growth — nearly half — for only 25% of the volatility, with a median worst-drawdown around 30% instead of 80%. The 1990s blackjack teams with a *verified* edge bet around a fifth of Kelly, not because they were timid but because they understood that the edge only pays if you survive long enough to realise it, and a 30% drawdown is survivable where an 80% one is not. Smaller bets are how professionals stay professionals.

**"The faster grower always wins, so over time full Kelly dominates."** Over a *long enough* time and on the *median* path, the faster grower does pull ahead — that is true and worth respecting. But "long enough" can be longer than your career, longer than your investors' patience, and longer than the time it takes a fat-tailed tail event to take a full-Kelly book to zero. Ruin is absorbing: a full-Kelly path that touches catastrophe is out of the game forever and never collects its superior growth rate. Survival is the precondition for compounding, and full Kelly trades too much survival for too little growth.

## How it shows up in real markets

The fractional-Kelly lesson is written across the history of blow-ups: in almost every case, someone with a real edge sized it as if the edge were certain and the path were smooth, and the path was neither.

**Long-Term Capital Management (1998).** LTCM had genuine edges — convergence trades with positive expected value, run by Nobel laureates. But they were levered roughly 25-to-1 on their balance sheet, with around \$1.25 trillion in gross derivative notional against about \$4.7 billion of equity — a position size deep into over-Kelly territory for the true risk of their trades. When their estimated correlations and spreads turned out to be wrong (the edge was *less certain* than their models assumed), the over-sized book lost about \$4.6 billion in roughly four months and required a \$3.6 billion Fed-organised rescue. The trades were not even badly chosen; they were *badly sized*, betting full-or-more Kelly on an edge whose true uncertainty they had underestimated. See [the LTCM case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade) for the strategic dimension.

**Amaranth Advisors (2006).** Amaranth lost about \$6.6 billion in roughly a week on concentrated, levered natural-gas calendar spreads. A single trader's position was sized as if the edge were near-certain and the book infinitely liquid; it was neither. The bet size was so far past any prudent fraction of Kelly that a normal adverse move in an illiquid market was fatal. The estimation error — assuming the spread relationship was more reliable than it was — combined with over-Kelly sizing to vaporise the fund.

**Archegos (2021).** Archegos held concentrated single-stock exposure levered roughly 5× through total-return swaps, with the size hidden from each of its prime brokers. When a few positions moved against it, the over-sized, over-levered book unwound violently, costing its banks more than \$10 billion (Credit Suisse alone took about \$5.5 billion). Even granting that the underlying stock picks had an edge, the *sizing* assumed a certainty and a liquidity that did not exist — the textbook over-Kelly failure, dressed up in swaps.

**Volmageddon (2018) and the yen-carry unwind (2024).** Both are crowded-carry blow-ups where the participants were, in effect, betting far past Kelly on a strategy (short volatility; funding-carry) whose true tail risk they had underestimated from a calm-period sample. On 5 February 2018 the VIX jumped about 20 points in a day and the XIV note lost about 96% of its value overnight; on 5 August 2024 the Nikkei fell 12.4% — its worst day since 1987 — as a crowded carry trade unwound and the VIX spiked intraday toward 65. In each case the historical sample (a few quiet years) made the edge look more certain and the tail look smaller than it was, the classic optimal-f overfitting trap, and the over-sized bet did the rest.

The thread through all of them: a real or plausible edge, *sized* as if the edge were known with certainty and the path would be smooth, with no fractional haircut for the uncertainty that always lurks. Every one of these would have *survived* a half-Kelly or quarter-Kelly version of the same trade. Fractional Kelly is not a tax on your returns; it is the premium on the insurance policy that keeps you in business when the edge turns out to be fuzzier than you thought.

## The fractional-Kelly playbook

Concrete rules for sizing a real strategy, in priority order:

1. **Default to a quarter-to-half Kelly.** Compute full Kelly (or optimal-f) as an *upper bound*, then bet a quarter to a half of it. Half Kelly keeps ~75% of the growth at 50% of the volatility; quarter Kelly keeps ~44% at 25%. Start at the conservative end and earn your way up, never the reverse.

2. **Scale the fraction down with your uncertainty about the edge.** The fewer trades your edge estimate rests on, the deeper the haircut. A backtest of 100 trades deserves quarter Kelly or less; a strategy with thousands of out-of-sample fills and a stable edge can justify half Kelly. If you cannot put error bars on your edge, you cannot justify more than quarter Kelly.

3. **Never bet above full Kelly, ever.** The growth curve is symmetric: 1.5× Kelly has the same growth as half Kelly but three times the volatility. Anything past full Kelly is the strictly-dominated, backward-bending part of the frontier. Treat full Kelly as a hard ceiling that you stay well below, not a target.

4. **Discount optimal-f the hardest.** Because optimal-f is fit to your sample's worst loss, and your true worst loss is bigger, treat its output as the most inflated of all the upper bounds. A common practice is to bet half of optimal-f or less, and to stress-test the bet against a worst loss double your historical maximum.

5. **Cap the per-bet risk independently.** Layer a hard rule on top of the fraction — e.g. never risk more than 1–2% of capital on a single position regardless of what Kelly says. This catches the case where a high estimated edge produces an uncomfortably large Kelly bet; the cap keeps a single bad fill or gap from doing outsized damage. See [leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) for why leverage magnifies the volatility-drag term that fractional Kelly is built to contain.

6. **Re-estimate the edge continuously and re-size.** Kelly sizing is fixed-fractional, so it already shrinks the bet as the account falls. But also lower your *fraction* if live results suggest the edge is weaker than the backtest — a deteriorating edge is the most dangerous, because it makes your "full Kelly" silently into over-Kelly. When in a deep drawdown, the right move is usually to cut the fraction further, not to "bet bigger to recover".

7. **Judge yourself on the path, not the median.** The median full-Kelly path looks great in a backtest; the path you actually live through is the jagged one with the 80% drawdown. Size for the drawdown distribution (Figure 2), not the expected terminal wealth. The goal is to keep the worst plausible drawdown inside what you, your investors, and your risk limits can survive — because [survival is the compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine), and you cannot compound an edge you have been knocked out of the game to collect.

The single sentence to carry out of this post: **full Kelly is the answer to a question — "how do I maximise growth assuming I know my edge and have an iron stomach?" — that no real trader is actually asking.** The real question is "how do I capture most of my edge's growth while surviving the path and my own uncertainty about the edge?", and the answer to *that* question is a fraction of Kelly. Bet less. Sleep at night. Be there tomorrow to bet again.

### Further reading

- [The Kelly criterion: how much to bet when you have an edge](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) — the full-Kelly derivation this post corrects.
- [Drawdown and the underwater curve: the risk you actually feel](/blog/trading/risk-management/drawdown-and-the-underwater-curve-the-risk-you-actually-feel) — the path-pain that fractional Kelly is buying down.
- [Risk management, the only free lunch: survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — why staying in the game is the precondition for any edge.
- [Kelly criterion and sequential betting (quant interviews)](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — the heavier math and the interview-style derivations.
- [The mean-variance efficient frontier (math for quants)](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) — the portfolio-theory analogue of the growth-versus-risk frontier and its backward-bending arc.
