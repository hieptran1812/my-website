---
title: "Value at Risk, and Exactly How VaR Lies"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "VaR claims that with X% confidence you will not lose more than a stated amount in a day; here is how it is computed three ways, the precise ways it misleads, and how to use it without being fooled."
tags: ["risk-management", "value-at-risk", "var", "expected-shortfall", "tail-risk", "backtesting", "monte-carlo", "fat-tails", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **The one idea:** Value at Risk answers exactly one narrow question — "with X% confidence, how much could I lose in a day?" — and the way it answers is precise enough to be useful and narrow enough to be dangerous if you forget what it leaves out.
> - VaR is the **edge of the cliff, not the size of the drop**: a 99% VaR of \$273,196 tells you losses cross that line on about 1 day in 100, and says *nothing* about whether that bad day costs \$300k or \$9M.
> - It is computed **three different ways** — historical, parametric (variance-covariance), and Monte Carlo — and on the *same* data they disagree, here by about \$33,000 on a \$10M book.
> - It is **backward-looking**: a VaR fitted on a calm window of \$184,646 gets blown through by crisis days losing over \$1,000,000 each, because the future is not a resample of the recent past.
> - It is **not subadditive and therefore gameable**: a desk can lower its headline VaR while *raising* real risk by selling deep out-of-the-money tail options or shopping for the kindest method.
> - The fix is not to throw VaR away — it is a genuinely useful common language — but to **pair it with Expected Shortfall, stress tests, and exceptions backtesting**, and to treat the number as a starting point for a conversation, never the end of one.

In January 2008, the trading desks of the largest banks in the world were carrying Value at Risk numbers that looked, by any reasonable reading, comfortable. A typical large dealer might report a one-day 99% VaR in the low hundreds of millions against a balance sheet measured in the trillions — a number that said, in effect, "on 99 days out of 100, we will not lose more than this." The models were sophisticated, the data was clean, the regulators had blessed the methodology. And then, over the following year, several of those same institutions posted *single-day* losses that their VaR models said should happen roughly once every several thousand years. Not once. Repeatedly. Day after day, the impossible kept happening, and the number that was supposed to bound the loss sat there, serenely, like a speed limit sign on a road that had just turned to ice.

This is the central paradox of Value at Risk, the single most widely used risk number on Earth. It is not that VaR is wrong, exactly. VaR is, in a narrow technical sense, almost always *right*: it makes a probabilistic claim about a threshold, and the threshold is usually crossed about as often as advertised — until it isn't. The problem is what VaR refuses to tell you, and how confidently it refuses. It hands you a clean, round, authoritative dollar figure and lets you believe that figure is "your risk," when in truth it is one carefully chosen point on a distribution whose most important features — the depth of the tail, the regime you are actually in, the way the number can be quietly gamed — live precisely in the region VaR declines to describe.

The mental model to carry through this entire post is in Figure 1, and it is the whole argument in one picture. Picture the full distribution of your daily profit and loss as a landscape. VaR is a single vertical line drawn near the left edge — the loss level you will exceed only 1% of the time. Everything to the *left* of that line, the shaded red region, is the 1% of days that are worse. VaR tells you precisely where the line sits. It tells you absolutely nothing about how far the ground drops on the other side. **VaR is the edge of the cliff; it is not the height of the fall.** A trader who confuses the two — who reads "99% VaR = \$273,196" as "the most I can lose is about \$273,196" — has misunderstood the number so completely that the number has become a liability rather than an asset.

![Daily profit and loss histogram on a ten million dollar book with the ninety-nine percent VaR cutoff line marked and the silent tail of losses beyond it shaded red](/imgs/blogs/value-at-risk-and-exactly-how-var-lies-1.png)

This post is the honest user's manual for VaR. We are going to build the number from absolutely nothing — no statistics background assumed — define every term, and then compute a real one-day 99% VaR three completely different ways on the same book so you can watch the methods disagree. Then we turn to the heart of it: the precise, nameable ways VaR lies. It is silent about the size of the tail beyond the cutoff. It is backward-looking and assumes tomorrow resembles its training window. It is non-subadditive, which makes it gameable and occasionally penalizes diversification. And it is *falsely precise* — it reports four significant figures of confidence it has not earned. We will be fair throughout: VaR survived for a reason, and the conclusion is not "burn it down" but "here is how to use it without letting it fool you." Because the entire point of this series is survival, and you cannot survive a risk you have mistaken for a smaller one. You can only compound if you are still in the game, and the fastest way out of the game is to trust a risk number that was quietly lying to you.

## Foundations: the building blocks of Value at Risk

Before we touch VaR itself, let us nail down every prerequisite term from zero. If you trade, skim this; if you do not, this section is the floor everything else stands on.

**Profit and loss (P&L).** Your P&L for a day is simply how much money you made or lost that day, in dollars. If your book is worth \$10,000,000 at the open and \$9,950,000 at the close, your P&L for the day is −\$50,000. Across many days, your P&L forms a *distribution* — a collection of outcomes, some positive, some negative, with some values common and some rare. Almost everything in risk management is a statement about the shape of that distribution, and VaR is a statement about one specific part of it: the bad left edge.

**A return and its volatility.** A return is the percentage change in your money over a period: a \$50,000 loss on a \$10,000,000 book is a return of −0.5%. The *volatility* (or "vol") of returns is how spread out they are — formally, the standard deviation. If a book's daily returns have a standard deviation of about 0.97%, then on a typical day it moves roughly ±0.97% (±\$97,000 on \$10M), and larger moves happen less often. Volatility is the single most important input to most VaR models, and — as we will see — its biggest hidden assumption.

**A distribution, and its percentiles.** A distribution describes how likely each outcome is. The *percentile* of a distribution is the cutoff below which a given fraction of outcomes fall. The 1st percentile of your daily P&L is the loss level that only 1% of days are worse than. That single concept *is* VaR: a 99% VaR is just the 1st-percentile loss, reported as a positive number. The 5th percentile is the 95% VaR, the 0.1th percentile is the 99.9% VaR, and so on. When someone says "the confidence level," they mean which percentile you are reading.

**The normal distribution (the bell curve).** The famous symmetric bell curve is the default assumption baked into the simplest VaR method. It has a convenient property: once you know its mean and its standard deviation, you know *everything* about it, including every percentile. For a normal distribution, the 1st-percentile outcome sits exactly 2.326 standard deviations below the mean (that magic number, 2.326, is the "99% one-sided z-score"). This is enormously convenient and, as the whole back half of this post will argue, enormously misleading — because real market returns are *not* normal. They have **fat tails**: extreme moves happen far more often than a bell curve predicts.

**Confidence level and horizon.** A VaR is incomplete until you state two things: the *confidence level* (95%, 99%, 99.9% — how deep into the tail you are reading) and the *horizon* (1 day, 10 days — over what period the loss could occur). "1-day 99% VaR" means "the loss level we expect to exceed on about 1 day in 100." Change either knob and the number changes, sometimes dramatically. A bank's regulatory VaR is often 10-day 99%; a desk's internal VaR is often 1-day. Quoting a VaR without both numbers is like quoting a speed without saying miles or kilometers per hour.

**The two accounts we will use throughout.** To keep this concrete, every worked example in this post uses one of two books. The first is a **\$100,000 retail account** — a serious individual trader. The second is a **\$10,000,000 book** — a small professional desk. We will compute VaR for both, in dollars, so the number never floats free of money. Hold those two numbers in mind: \$100,000 and \$10,000,000.

With those terms — P&L, return, volatility, distribution, percentile, the normal curve, confidence level, horizon — you have everything you need. Now let us state precisely what VaR claims, and then compute one.

## What VaR actually claims (and the trap inside the claim)

Here is the formal definition, stated carefully because the precision is exactly where the trap hides. The **one-day, X%-confidence Value at Risk** is the smallest dollar loss *L* such that the probability of losing more than *L* in one day is at most (1 − X%). In plainer words: *with X% confidence, you will not lose more than \$L in a day.*

Read that sentence again, slowly, because almost everyone misreads it. It does **not** say "the most you can lose is \$L." It does **not** say "your worst-case loss is \$L." It says only that losses *worse* than \$L happen with probability at most (1 − X%) — for a 99% VaR, on at most about 1 day in 100. On the other 1 day in 100, you lose *more* than \$L, and the definition is studiously, deliberately silent about *how much* more. That silence is not a bug in the definition; it is the definition. VaR is a statement about the *frequency* of a threshold being crossed, with zero content about the *severity* once it is.

This is the difference between two questions a risk manager can ask, and VaR answers only the first:

1. *How often do I have a bad day?* — VaR answers this. (About 1% of days, for a 99% VaR.)
2. *How bad is a bad day when I have one?* — VaR is silent. (This is what Expected Shortfall answers, and why it exists.)

Figure 3 lays the two halves side by side: what VaR says, and what it pointedly does not. On the left is the genuine content — a threshold losses rarely cross, a single comparable number, a common language for a trading floor. On the right is everything the number omits: how deep the bad day goes, whether tomorrow resembles the past the model was fitted on, and whether the number itself was quietly gamed lower. Both columns are true at once. VaR is *useful* and *dangerously incomplete*, and a professional holds both facts in mind simultaneously.

![Side by side comparison of what Value at Risk says about a loss threshold versus what it does not say about tail depth regime and gaming](/imgs/blogs/value-at-risk-and-exactly-how-var-lies-3.png)

The trap, then, is linguistic as much as mathematical. The phrase "Value at Risk" *sounds* like it names the total value you have put at risk — your maximum exposure. It does not. It names a percentile of a loss distribution. The name is one of the great branding coups and one of the great misnomers in finance: it borrowed the gravity of "your value, at risk" and attached it to a number that is merely "the loss you exceed 1% of the time." Every misuse of VaR — every executive who looked at a VaR report and felt safe, every board that capitalized to a VaR number and got wiped out anyway — traces back to taking the name at face value.

#### Worked example: reading a VaR statement correctly

Suppose your \$10,000,000 desk reports a 1-day 99% VaR of \$273,196 (this is the number we will actually compute in the next section). Here is what that statement licenses you to believe, and what it does not.

- **Licensed:** On a typical day, losses worse than \$273,196 should occur about 1% of the time — roughly 2 to 3 trading days per year (since a year has about 252 trading days, and 1% of 252 ≈ 2.5).
- **Licensed:** As a percentage of the book, that VaR is \$273,196 / \$10,000,000 = 2.73% — a useful, comparable figure you can line up against another desk's 1.8% or 4.1%.
- **NOT licensed:** "The most I can lose tomorrow is \$273,196." False. On the ~2.5 bad days a year, the loss is *worse* than \$273,196 by an unstated amount.
- **NOT licensed:** "I have \$273,196 at risk." False. Your entire \$10,000,000 is at risk; VaR is a percentile of how the day might go, not a cap on the damage.

The correct mental translation of "99% VaR = \$273,196" is the mouthful: *"On about 99 of every 100 days, I expect to lose less than \$273,196; on the other day or so, I will lose more, and this number does not tell me how much more."*

*VaR is a frequency claim wearing a severity claim's clothing; read it as "how often," never as "how bad."*

## Computing VaR three ways: same book, three answers

There is no single "the VaR." There are *methods*, and they hand back different numbers. The three standard methods are **historical simulation**, the **parametric (variance-covariance)** method, and **Monte Carlo simulation**. Let us run all three on the same \$10,000,000 book, using the same underlying return sample — a realistic, mildly fat-tailed series of about ten years of daily returns with a measured daily volatility of 0.97%. Figure 2 shows the three answers as bars. They disagree by about \$33,000, on the same data, for the same day. That disagreement is not a mistake; it is what VaR *is*.

![Bar chart of one day ninety-nine percent VaR computed by historical parametric normal and Monte Carlo methods on the same return sample showing three different dollar figures](/imgs/blogs/value-at-risk-and-exactly-how-var-lies-2.png)

### Method 1: Historical simulation

The historical method is the most honest and the most stubborn: it makes *no assumption about the shape of the distribution at all*. You take a window of past returns — say the last 2,500 trading days — apply each of those returns to your *current* book to get a set of hypothetical P&Ls, sort them from worst to best, and read off the loss at the chosen percentile. For a 99% VaR over 2,500 days, you find the 25th-worst day (since 1% of 2,500 is 25) and that loss is your VaR.

The appeal is obvious: it lets the data speak. If the last decade had fat tails, the historical VaR inherits those fat tails automatically, with no bell-curve assumption to flatten them. The weakness is equally obvious, and it is the seed of the "backward-looking" lie we will dissect later: **your VaR is only as fat-tailed as your window.** If your 2,500-day window happens to be a calm decade with no crash in it, the historical method will confidently report a small VaR, because it has literally never seen a bad day. History is a guide to the future only to the extent that the future rhymes with it.

The window length itself is a quiet trap. A *short* window (say 250 days) reacts quickly to changing volatility — good — but it is dominated by recent calm and forgets old crises entirely, so it tends to under-state risk going into a storm. A *long* window (say 2,500 days) remembers more disasters but reacts sluggishly, so it keeps a fat VaR long after a crisis has passed and a thin one long after calm has ended. Worse, the historical method has a notorious artifact called **ghosting**: the day a single large loss rolls off the back of the window, your VaR can *drop sharply* overnight, even though nothing about your book or the market changed — the number fell purely because the sample forgot a bad day. A risk number that lurches because of a calendar boundary, not a change in risk, is a vivid reminder that historical VaR is an estimate, not a measurement. Practitioners patch this with *exponentially weighted* schemes that fade old observations smoothly rather than dropping them off a cliff, but no weighting scheme fixes the underlying problem: the method can only ever know about disasters that already happened to be in its memory.

### Method 2: Parametric (variance-covariance)

The parametric method makes the bell-curve assumption explicit and exploits it for speed. You assume returns are normally distributed, estimate just two numbers from your data — the mean *μ* and the standard deviation *σ* — and then read the percentile straight off the normal formula. For a 99% VaR:

$$\text{VaR}_{99\%} = -(\mu - 2.326 \cdot \sigma) \times \text{Book}$$

That 2.326 is the 99% z-score; for 95% it is 1.645, for 99.9% it is 3.090. The method is blazingly fast — for a whole portfolio you need only the covariance matrix of the holdings, which is why it earned the name "variance-covariance" — and it scales to thousands of positions. Its fatal flaw is the assumption that makes it fast: **real returns are not normal.** The bell curve has thin tails; markets have fat ones. So the parametric method systematically *under-states* the chance of extreme losses, precisely in the region a risk number is supposed to protect you. It is the method most likely to report the lowest, most comforting number — which, as we will see, is exactly why a trader hoping to look safe might *prefer* it.

### Method 3: Monte Carlo simulation

The Monte Carlo method is the most flexible and the most computationally hungry. Instead of using only the past returns that actually happened (historical) or assuming a bell curve (parametric), you build an explicit *model* of how returns behave — which can be fat-tailed, can include jumps, can model the dependence between assets however you like — and then you *simulate* hundreds of thousands of possible days from that model. You sort the simulated P&Ls and read off the percentile, exactly as in the historical method, but now your "history" is a huge synthetic sample drawn from a model you chose.

Monte Carlo's strength is that it can capture fat tails *and* complex, non-linear positions like options, where a small move in the underlying produces a wildly non-linear move in P&L. Its weakness is that it is only as good as the model you feed it: garbage assumptions in, garbage VaR out. If you simulate from a normal distribution, you have just reinvented the parametric method with extra steps. If you simulate from a genuinely fat-tailed model — here, a Student-t distribution with 4 degrees of freedom — you get a number that respects the tail.

#### Worked example: the same 99% VaR, three ways, on the \$10M book

Let us put real numbers on all three. We start from one return sample with measured daily volatility *σ* = 0.97% and a near-zero mean, on a \$10,000,000 book.

- **Historical.** Sort 2,500 days of P&L, take the 25th-worst (the 1st percentile). The loss there is **\$265,962**. This number already "knows" about the fat tail because the bad days are physically in the sample.
- **Parametric-normal.** Plug into the formula: VaR = −(μ − 2.326 × σ) × \$10,000,000. With μ ≈ 0 and σ = 0.97%, this is roughly 2.326 × 0.0097 × \$10,000,000 ≈ **\$233,150**. Notice it is the *smallest* of the three — the normal assumption shaved the fat tail off and reported a friendlier number.
- **Monte Carlo (fat-tailed).** Simulate 400,000 days from a Student-t model matched to the same vol, sort, take the 1st percentile: **\$266,008**. Because the t-distribution has genuinely fat tails, this lands close to the honest historical number, and well above the parametric one.

Line them up: \$265,962 vs \$233,150 vs \$266,008. The spread between the lowest and highest is **\$32,858** — about 0.3% of the entire book, or 12% of the VaR number itself — *on identical data, for the same day, at the same confidence level.* The parametric method, by assuming normality, under-states the risk by roughly \$33,000 relative to the fat-tailed methods.

*There is no "the VaR" — only a VaR-given-a-method, and the method you pick can move the headline number by more than 10% without changing a single trade.*

This is the first concrete way VaR misleads: it is reported as a single authoritative figure, but it is the output of a *choice* of method, and that choice is rarely visible to whoever reads the report. An executive who sees "VaR: \$233,150" has no way of knowing that the same book under the historical method is \$265,962, or that the difference is the bell-curve assumption quietly deleting the tail. We will return to this when we discuss gaming, because "pick the method that reports the lowest number" is the oldest trick in the book.

## The first lie: VaR is silent about the size of the tail

Now we reach the deepest and most important way VaR misleads, the one Figure 1 was built to show. VaR draws a line at the cutoff and tells you nothing about the region beyond it. Two books can have *identical* 99% VaR and *wildly* different actual risk, because VaR cannot see the shape of the tail past its own line.

Concretely: take two desks, both reporting a 1-day 99% VaR of exactly \$273,196 on a \$10M book. Desk A's bad days, when they come, cluster just past the line — a typical breach loses \$300,000, occasionally \$400,000. Desk B's bad days are catastrophic — a typical breach loses \$2,000,000, and the worst can lose \$9,000,000, because Desk B is short deep out-of-the-money options that blow up in a crash. **VaR reports the same number for both.** It physically cannot distinguish them, because the only thing it measures is *where the 1st-percentile line sits*, and both desks cross that line at the same place. Everything that makes Desk B a death trap lives in the shaded red region of Figure 1 — the region VaR refuses to describe.

This is why the very next post in this series is about **Expected Shortfall** (also called CVaR or Conditional VaR), the number that answers the question VaR ducks: *given that you have a bad day, how bad is it on average?* Expected Shortfall is the average loss *in the tail beyond the VaR cutoff*. On our cover book, the 99% VaR is \$273,196, but the *average* loss on the days worse than that is \$353,352 — about 29% deeper than the cutoff suggests — and the single worst day in the sample loses \$777,841, nearly three times the VaR. The VaR line at \$273,196 is the *least bad* of the bad days, dressed up as if it summarized them. For the full treatment of the number that fixes this, see [CVaR, Expected Shortfall, and asking how bad is bad](/blog/trading/risk-management/cvar-expected-shortfall-and-asking-how-bad-is-bad).

#### Worked example: two books, identical VaR, opposite fates

Take the \$100,000 retail account this time. Suppose you compute a 1-day 99% VaR of \$2,500 (2.5% of the account) two different ways of trading.

- **Account A — a diversified stock portfolio.** Your bad days, beyond the \$2,500 line, lose maybe \$3,000–\$4,000. The tail is *thin*: a breach hurts, but the days beyond the cutoff are only modestly worse than the cutoff. Your average tail loss might be \$3,200.
- **Account B — selling weekly out-of-the-money put options for steady premium.** On 99% of days you collect premium and your VaR also computes to \$2,500. But on the rare day the market gaps down, your short puts explode: a breach loses \$15,000, \$30,000, in a true crash potentially the *entire* \$100,000. Your average tail loss is \$22,000 — seven times the VaR.

Both accounts report "99% VaR = \$2,500." One is a sane portfolio; the other is a [short-vol blow-up](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) waiting to happen. VaR sees them as twins.

*VaR measures the height of the fence; it has no opinion about whether there is a six-foot drop or a thousand-foot cliff on the other side — and selling tail risk is the business of building the cliff while keeping the fence the same height.*

This single property — silence about the tail — is why VaR, used alone, gives short-volatility and tail-selling strategies a flattering report card right up until the moment they detonate. The strategy's entire risk lives past the cutoff, exactly where VaR has nothing to say.

## The second lie: VaR is backward-looking

Every VaR method, without exception, is fitted on the *past* — a historical window, an estimated volatility, a model calibrated to old data. And every VaR method therefore carries the same buried assumption: **tomorrow will look statistically like the window I was trained on.** When that assumption holds, VaR works. When the regime breaks — when a calm market turns violent — VaR is the last to know, because it is looking in the rear-view mirror while the crash unfolds through the windshield.

Figure 4 makes this visceral. We fit a 99% VaR on a 250-day *calm* window — a year of gentle, low-volatility trading. The fitted VaR is \$184,646, and it sits as a flat amber line, comfortable and authoritative. Then the regime breaks: volatility triples, the drift turns sharply negative, and a 60-day crisis unfolds. The crisis days blow clean through the calm-period VaR line again and again — 30 separate days in the series breach it, and the worst single day loses **\$1,156,237**, more than six times the VaR that was supposed to bound the loss. Nothing about the VaR number was "wrong" when it was computed; it correctly summarized the calm window. It was simply answering a question about a world that had ceased to exist.

![Daily profit and loss bars over a calm period then a crisis with a flat calm period VaR line that the crisis losses repeatedly blow through](/imgs/blogs/value-at-risk-and-exactly-how-var-lies-4.png)

This is not a hypothetical. It is the precise mechanism behind the 2008 failure that opened this post, and behind every "we had never seen anything like it" post-mortem. The banks' VaR models in 2007 were fitted on the preceding years of unusually low volatility — the "Great Moderation." Those models reported small, comfortable numbers because the recent past had been small and comfortable. The models were not lying about the past; they were faithfully reporting it. They simply could not know that the correlation structure of mortgage-linked assets was about to invert, that liquidity was about to vanish, that the calm window they had memorized was about to be revealed as the eye of a storm. **A backward-looking risk measure is, by construction, blindest exactly when risk is changing fastest** — which is to say, exactly when you need it most.

There is a subtler version of this problem that bites even sophisticated desks: VaR estimates are *themselves* noisy, and the noise is worst in the tail. The 99.9% VaR depends on the very rarest events in your sample, of which you have, by definition, almost none. Estimating a 1-in-1000 day loss from ten years of data (about 2,500 days) means you are extrapolating from maybe two or three observations. The number you report has wide error bars you never see, and those error bars are widest precisely at the high-confidence levels that sound the most reassuring. This connects directly to the mathematics of [tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants), which exists precisely because estimating the far tail from limited data requires its own specialized machinery — you cannot just read it off a histogram.

#### Worked example: the calm-window VaR that lied

Your \$10,000,000 desk has traded quietly for a year. You compute a 1-day 99% VaR from those 250 calm days: **\$184,646**. You set your risk limits around it. You feel safe. Then the regime breaks.

- Day 251: a −2.5% day. Loss = \$250,000. **Breach** — already past the \$184,646 line.
- Day 268: a −6.1% day. Loss = \$610,000. **Breach** — 3.3× the VaR.
- The worst day of the crisis: −11.6%. Loss = \$1,156,237. **Breach** — 6.3× the VaR.

Over the crisis, 30 separate days breach a VaR that was supposed to be exceeded about 1% of the time. The VaR was not "wrong" — it was a faithful summary of a calm year. It was *obsolete*, and obsolescence in a risk number is indistinguishable from a lie when you are the one trusting it.

*A VaR is a photograph of yesterday's weather; it is a useless guide to a storm precisely because the storm is the thing that makes today different from yesterday.*

The practical defense, which we will formalize in the playbook, is to *never trust a single static VaR*. You watch how VaR is *moving* (a rising VaR is an early warning), you re-estimate it on windows of different lengths, and above all you supplement it with **stress tests** — deliberate "what if 2008 happened tomorrow" scenarios that do not depend on the recent past at all. Stress testing is the antidote to backward-lookingness: it asks "what would this book lose in a *named* historical or hypothetical disaster," sidestepping the question of whether the disaster is in your sample.

## The third lie: VaR is not subadditive, and therefore gameable

Here is a property you would *expect* any sane risk measure to have, and that VaR does not reliably have. **Subadditivity** means that the risk of a combined portfolio should never exceed the sum of the risks of its parts: diversification should reduce risk, or at worst leave it unchanged, never increase it. Formally, a risk measure ρ is subadditive if ρ(A + B) ≤ ρ(A) + ρ(B) for any two positions A and B. Expected Shortfall has this property always. **VaR does not.** There exist portfolios where combining two positions *increases* the VaR — where VaR claims diversification made you riskier, which is nonsense.

This is not merely a theoretical wart. Non-subadditivity is the mathematical reason VaR is *gameable*, and gaming VaR is a real, documented practice. Because VaR only "sees" the loss at the cutoff and is blind to the tail beyond it, you can rearrange a portfolio to push risk *past* the cutoff — into the region VaR cannot see — and the reported number *falls* even as your true exposure *rises*. Figure 6 maps the mechanism: a desk that wants to report a low VaR number has several levers, and pulling them lowers the headline while raising the real risk.

![Graph showing how a desk lowers its headline VaR by selling out of the money tail options shopping for the kindest method and using a calm lookback window while real tail risk rises toward a blow up](/imgs/blogs/value-at-risk-and-exactly-how-var-lies-6.png)

The clearest lever is **selling deep out-of-the-money tail risk.** Sell options so far out of the money that they only pay out in a genuine crash — say, a put that only costs you money if the market falls 15% in a day. On 99% of days, including the 1% "bad" days that define your VaR cutoff, those options expire worthless and you simply pocket the premium. So your P&L on every day that VaR can see is *improved* by the premium, and your 99% VaR *falls*. But your exposure to a true tail event — the 0.1% day, the crash that lives past the VaR cutoff — has *exploded*. You have moved all your risk into the blind spot. The headline VaR says "less risky"; the reality is "a hidden bomb." This is, concretely, the [Volmageddon 2018 trade](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup) and a hundred others: a strategy that looks superb on every risk-managed day and dies on the one day the risk measure was never watching.

The second lever is **method shopping.** As we computed above, the same book has a parametric VaR of \$233,150 and a historical VaR of \$265,962. A trader incentivized to report a low number simply argues for the parametric method "because it's the industry standard / more stable / what the model vendor recommends" — and books \$33,000 less risk on paper without changing a single position. The third lever is **window selection**: choose a lookback window that excludes the last crisis, and your historical VaR shrinks because the bad days are no longer in the sample.

#### Worked example: lowering VaR while raising risk

Your \$10,000,000 desk currently runs a plain long book with a 99% VaR of \$265,962 (historical method). Your boss wants the desk's reported VaR under \$220,000. You have two dishonest ways to get there, neither of which reduces your actual risk:

- **Sell tail options.** Write deep-OTM index puts that only pay out on a >12% one-day crash, collecting \$15,000/month in premium. The premium lifts your everyday P&L, so the 1st-percentile loss improves and your VaR drops to, say, \$210,000. But you have just sold insurance against a catastrophe — on a true crash you could lose \$3,000,000+. **VaR fell \$56,000; real tail risk rose by millions.**
- **Switch to parametric.** Re-report the *same* book under the normal assumption: VaR = \$233,150 instead of \$265,962. Still over budget. Combine with a shorter, calmer lookback window and you slide under \$220,000. **Zero trades changed; \$45,000+ of reported risk vanished by assumption.**

In both cases the risk *report* improved and the risk itself got worse or stayed the same. That is only possible because VaR is blind past its own cutoff and sensitive to method and window — the exact properties that make it non-subadditive.

*Any number a trader can lower without lowering their actual risk is a number a trader will eventually lower without lowering their actual risk — and VaR is full of those levers.*

The defense here is structural, not statistical: you do not let the trader choose the method or the window, you pair VaR with Expected Shortfall (which is subadditive and *cannot* be gamed by pushing risk into the tail, because it measures the tail), and you stress-test the book against named tail scenarios that ignore the cutoff entirely. The firm-level view of why these controls have to be independent of the trader — and how risk management functions as a business discipline rather than a spreadsheet — is the subject of [risk management as a business function](/blog/trading/hedge-funds/risk-management-as-a-business-function).

## The fourth lie: VaR is falsely precise

The final way VaR misleads is the most insidious because it is purely psychological: VaR reports more confidence than it possesses. When a model hands you "99% VaR = \$273,196," the six-digit precision and the crisp 99% *feel* authoritative — they feel like the output of a measurement, like reading a thermometer. But that number is the product of a chain of choices and estimates, each carrying its own uncertainty: which method, which window, which volatility estimate, which distributional assumption. The true uncertainty around that \$273,196 is not ±\$50; it is plausibly ±\$50,000 or more, as the method-disagreement of \$32,858 already hinted. **VaR's precision is a presentation artifact, not a property of the underlying risk.**

Nowhere is this clearer than in the choice of confidence level, which Figure 7 dissects. On the *same* distribution, the 95% VaR is \$165,152, the 99% VaR is \$277,533, and the 99.9% VaR is \$474,070. The number nearly *tripled* — a factor of 2.9 — purely by turning the confidence dial from 95% to 99.9%, with no change to the book whatsoever. Each of those numbers is a "correct" VaR. Each tells a different story: the 95% number makes the desk look modestly risky, the 99.9% number makes it look frightening, and both describe the identical portfolio. A reader who is handed just one of them, with no sense of how sensitive it is to the dial, has been given false precision wearing the costume of fact.

![Histogram of daily profit and loss with three VaR cutoff lines at ninety-five ninety-nine and ninety-nine point nine percent confidence showing the dollar figure nearly tripling across the levels](/imgs/blogs/value-at-risk-and-exactly-how-var-lies-7.png)

The false precision compounds with the backward-looking problem in a particularly nasty way at high confidence levels. The 99.9% VaR depends on the rarest 0.1% of days. In a ten-year (2,500-day) sample, that is the worst 2 or 3 days *that happened to occur* — a sample so small that the estimate is dominated by luck. Re-run the same model on a slightly different window and the 99.9% VaR can swing by 20% or more, because you have added or dropped a single extreme day. So the *most reassuring-sounding* VaR — the high-confidence one that sounds like it captures the worst case — is precisely the *least reliable* one, with the widest hidden error bars. The number that sounds most like a guarantee is the number you should trust least.

#### Worked example: the same book at three confidence levels

Your \$10,000,000 desk, one distribution, three honest VaRs:

- **95% VaR = \$165,152** (1.65% of the book). Story: "We have a rough day about once a month, costing under \$170k. Manageable."
- **99% VaR = \$277,533** (2.78% of the book). Story: "A couple of times a year we lose around \$280k. Notable."
- **99.9% VaR = \$474,070** (4.74% of the book). Story: "Once every few years a day costs us nearly half a million. Serious."

All three are the *same book on the same day*. The 99.9% number is **2.9× the 95% number** — \$308,918 of "risk" appears purely by moving a dial. Anyone who quotes you a single VaR without the confidence level has, intentionally or not, chosen which story to tell you.

*A VaR with three significant figures and no error bar is a guess wearing a lab coat; the honest version is always a range, and the range is widest exactly where the number sounds most precise.*

## Common misconceptions

**"VaR tells me the most I can lose."** No — this is the single most dangerous misreading. A 99% VaR of \$273,196 tells you losses *exceed* that level about 1% of the time; it says nothing about how much more you lose on those days. On our cover book the average loss beyond the cutoff is \$353,352 and the worst is \$777,841 — both far past the VaR. VaR is the *floor* of the bad region, not the *ceiling* of possible loss.

**"A lower VaR always means a safer book."** No — VaR is gameable. A desk can *lower* its reported VaR to \$210,000 by selling deep out-of-the-money tail options while *raising* its true crash exposure into the millions, because the sold premium improves every day VaR can see and the catastrophe lives in the blind spot past the cutoff. A falling VaR can be the signature of *more* hidden risk, not less.

**"VaR is objective — there's one right number."** No — on identical data, the historical method gave \$265,962, parametric gave \$233,150, and Monte Carlo gave \$266,008, a \$32,858 spread driven entirely by method choice. And on one distribution, switching from 95% to 99.9% confidence moved the number from \$165,152 to \$474,070. VaR is a *choice* dressed as a measurement.

**"If my VaR was accurate in backtests, it's reliable going forward."** No — VaR is backward-looking. A model that backtested perfectly on a calm 250-day window reported \$184,646 and was then breached 30 times in a crisis, with a worst day of \$1,156,237 — 6.3× the supposed limit. Backtests confirm the model described the *past*; they cannot confirm the future will resemble it. That is exactly why backtesting must be ongoing (see the next misconception), not a one-time blessing.

**"99.9% VaR is more reliable than 99% because it's more conservative."** No — it is *less* reliable. The 99.9% VaR is estimated from the rarest handful of days in your sample (2–3 days in a decade), so it has enormous hidden uncertainty and can swing 20%+ if you add or drop one extreme observation. Higher confidence sounds safer but rests on thinner data; the number that sounds most like a guarantee has the widest error bars.

**"VaR failed in 2008, so VaR is useless."** No — this overcorrects. VaR is a genuinely useful common language: it lets a trading floor compare risk across desks in one number, sets a baseline for limits, and flags trends (a rising VaR is an early warning). It failed in 2008 because it was *used alone*, trusted as a severity cap, fitted on a calm window, and read as objective. Used correctly — paired with Expected Shortfall, stress tests, and exceptions backtesting — it remains one of the most useful tools on the desk. The lesson is "don't trust VaR alone," not "don't use VaR."

## How it shows up in real markets

**2008 and the failure of VaR.** The global financial crisis is the canonical case study in every way VaR lies, all firing at once. Through 2006–2007, major banks' VaR models were fitted on the "Great Moderation" — years of unusually low volatility — and so reported small, comfortable one-day VaRs. The models were faithfully *backward-looking*: they described the calm past accurately. Then the regime broke. Mortgage-linked correlations inverted, [correlations went to one in the crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis), and liquidity vanished. Banks posted single-day trading losses their models said should occur once in thousands of years — and posted them repeatedly, in clusters, exactly as Figure 4's crisis window shows. Worse, the *false precision* of the pre-crisis numbers had lulled boards and regulators into capitalizing to a VaR that was both obsolete and silent about the tail. The crisis was not a failure of VaR's arithmetic; it was a failure of treating a narrow, backward-looking, tail-blind, method-dependent number as if it were "the risk." It is the reason regulators (Basel) subsequently pushed banks from VaR toward **Expected Shortfall** for the trading book — a direct admission that VaR's silence about tail severity was the fatal flaw.

**LTCM, 1998.** Long-Term Capital Management ran around 25-to-1 balance-sheet leverage on convergence trades, with risk models that — like everyone's — were calibrated on historical relationships. When Russia defaulted in August 1998 and the world fled to quality, the historical correlations the models relied on broke completely: positions that were supposed to be diversified all moved against the fund at once. LTCM lost roughly \$4.6 billion of capital in about four months. The deeper lesson for this post is that LTCM's risk numbers were *backward-looking and tail-blind* in exactly the ways we have catalogued — they described a world of stable convergence that the crisis erased overnight. The full strategic anatomy is in [the LTCM 1998 case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade).

**Volmageddon, February 5, 2018.** This is the gaming lie made flesh. A crowd of traders and products (most infamously the XIV exchange-traded note) had sold short volatility for years — collecting steady premium, reporting tame VaR numbers day after day, because their entire risk lived *past* the VaR cutoff, in the rare event of a volatility spike. On February 5, 2018, the VIX jumped about 20 points (from 17.3 to 37.3, a ~116% one-day rise, the largest on record), and XIV's net asset value fell roughly 96% after the close, terminating the product. Every day before that, a VaR model would have given the short-vol book a clean bill of health. The risk was real the entire time; it simply lived in the blind spot. This is the [short-vol blow-up](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup) that no VaR number could have warned about, because the danger was structurally invisible to it.

**COVID, March 2020.** The fastest bear market on record is a pure regime-break case. The S&P 500 fell about 34% from its February 19 peak to its March 23 trough, and the VIX hit a record closing high of 82.69 on March 16. Any VaR fitted on the placid years before 2020 was instantly obsolete — daily moves arrived that were many times the pre-crisis VaR, clustered exactly as Figure 4 depicts. Books that had looked safe by VaR for years took losses far past their cutoffs in a matter of days, because the recent past the models had memorized had nothing to say about a global dash for cash.

**The yen-carry unwind, August 5, 2024.** A more recent reminder that the pattern repeats. A crowded funding-carry trade unwound in days; the Nikkei fell 12.4% in a single session (its worst day since 1987) and the VIX spiked to an intraday 65.7. Once again, VaR models fitted on a calm carry regime were silent right up to the unwind, because the danger of a crowded trade lives in the reflexive deleveraging that no calm-period sample contains. The mechanism — risk concentrated in a regime the model has never seen — is identical to 2008, 2018, and 2020, separated only by the asset and the date.

## The risk playbook: using VaR without being fooled

VaR is a useful instrument played badly by almost everyone. Here is how to use it as a professional — as one input among several, never as "the risk."

**1. Always quote VaR with its three parameters, and a method.** Never say "VaR is \$273k." Say "1-day 99% historical VaR is \$273k." The horizon, the confidence level, and the method are part of the number; a VaR without them is a story with the genre hidden. If someone reports a single VaR with no method, ask which one — and ask what the *other* two methods say.

**2. Never read VaR as a maximum loss.** Translate every VaR in your head as "the loss I exceed (1−X)% of the time, with no statement about how much more." On a \$10M book, "99% VaR = \$273k" means "about 2–3 days a year I lose more than \$273k, and I don't yet know how much more." The "how much more" is a separate question requiring a separate tool.

**3. Pair VaR with Expected Shortfall, always.** ES (CVaR) is the average loss *in the tail beyond VaR* — it answers the severity question VaR ducks, it is subadditive so diversification can never make it spuriously worse, and it cannot be gamed by pushing risk past the cutoff because it *is* the cutoff's far side. On our book, VaR is \$273,196 but ES is \$353,352 — the ES is the number that knows about the tail. Report both; act on the larger. The full treatment is in [CVaR, Expected Shortfall, and asking how bad is bad](/blog/trading/risk-management/cvar-expected-shortfall-and-asking-how-bad-is-bad).

**4. Stress-test against named disasters that ignore the recent past.** Because VaR is backward-looking, supplement it with scenarios that do not depend on your sample: "what does this book lose if 2008 / March 2020 / August 2024 happens tomorrow?" Stress tests are the antidote to obsolescence — they ask about specific catastrophes regardless of whether those catastrophes are in your window. A book is only as safe as its worst plausible stress, not its calm-period VaR.

**5. Backtest exceptions continuously, and treat excess breaches as an alarm.** Count how often realized losses breach the VaR line. At 99%, you expect about 1% of days — roughly 5 breaches in 500 days. In Figure 5, a normal-VaR model on fat-tailed data produced **16 breaches in 500 days, 3.2× the expected 5** — a clear signal the model under-states risk (this is the logic behind the regulatory "traffic light" / Kupiec tests). Too many breaches means your VaR is too small; clustered breaches mean a regime has broken. A VaR you never backtest is a smoke detector you never test.

![Five hundred days of daily profit and loss against a constant ninety-nine percent VaR line with breaches highlighted and a verdict that the model under-states risk](/imgs/blogs/value-at-risk-and-exactly-how-var-lies-5.png)

**6. Watch VaR's *trend*, not just its level.** A single static VaR is a photograph; the *change* in VaR is the early-warning system. A VaR that is creeping up is telling you that recent volatility is rising — often the first quantitative hint that a regime is shifting, before the crisis is obvious. Use the level for limits and the trend for vigilance.

**7. Never let the measured party choose the method or window.** Method-shopping and window-selection are how VaR gets gamed (Figure 6). The choice of method, confidence level, and lookback must be set by an independent risk function and held fixed, so that the only way to lower VaR is to actually lower risk. This is why risk control must be structurally independent of the trader — the [business-function view](/blog/trading/hedge-funds/risk-management-as-a-business-function) of why the risk seat reports separately from the trading seat.

**8. Size positions to the *tail*, not the cutoff.** This is where VaR rejoins the spine of this whole series. The reason all of this matters is the [asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain): a deep enough drawdown is mathematically near-impossible to recover from, so the loss that ends your career is never the average bad day at the VaR cutoff — it is the catastrophe in the tail VaR cannot see. Size your positions so that the *Expected Shortfall*, the *stress-test loss*, and the *worst plausible day* — not the comfortable VaR number — leave you with enough capital to trade tomorrow. VaR tells you where the cliff edge is. Survival depends on what happens at the bottom of the fall, which is exactly the part VaR was never built to describe. Why the deep mathematics of the tail needs its own machinery is the subject of [tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants).

The honest summary is this: VaR is not a liar so much as a *very literal witness* who answers exactly the question asked and not one word more. Ask it "how often do I have a bad day," and it answers well. Ask it "how bad can it get," "is this still true tomorrow," or "is this number being gamed," and it stays silent — and that silence, mistaken for reassurance, is how VaR lies. Use it for what it knows, pair it with tools that answer what it doesn't, and never, ever capitalize your survival to a number that was only ever describing the edge of the cliff.

### Further reading

- [CVaR, Expected Shortfall, and asking how bad is bad](/blog/trading/risk-management/cvar-expected-shortfall-and-asking-how-bad-is-bad) — the number that answers the severity question VaR ducks.
- [Fat tails and the normal distribution trap](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap) — why the bell-curve assumption inside parametric VaR under-states extreme losses.
- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — why the tail VaR can't see is the part that ends careers.
- [Tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) — the specialized mathematics of estimating the far tail from limited data.
- [Risk management as a business function](/blog/trading/hedge-funds/risk-management-as-a-business-function) — why the controls that keep VaR honest must be structurally independent of the trader.
