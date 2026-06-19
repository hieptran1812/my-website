---
title: "The Asymmetry of Losses: Why a 50% Loss Needs a 100% Gain"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Losses and gains are not mirror images: a drawdown of d needs a gain of d/(1-d) to recover, big losses are nearly impossible to climb back from, and volatility itself quietly drains your compounding."
tags: ["risk-management", "drawdown", "recovery-math", "compounding", "volatility-drag", "sequence-risk", "position-sizing", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **The one idea:** losses and gains are not symmetric — a drawdown of size *d* needs a gain of *d/(1−d)* to recover, so a −50% loss requires a +100% gain just to break even, and a −90% loss requires +900%.
> - The recovery cost is non-linear: it crawls for small losses and then **explodes** past about −50%, which is why deep drawdowns are nearly impossible to climb out of.
> - Returns **multiply**, they don't add: −50% then +50% lands at −25%, not zero — your gut does arithmetic, but your account does geometry.
> - The same average return delivers **less wealth when it is more volatile** — geometric growth ≈ arithmetic mean − ½·variance. Volatility is a literal drag on compounding.
> - The *order* of returns matters once money moves in or out: an early big loss is far more destructive than a late one (**sequence risk**).
> - The practical upshot is the whole series in one sentence: **cap the drawdown before it caps you.** This single piece of arithmetic dictates your max-loss limit and your position size.

A trader I know once described the worst year of his career in a single sentence: "I made it all back, and it still wasn't enough." He had run a book down 38% in a brutal stretch, clawed his way to a +38% year afterward, and was genuinely confused — *furious*, even — that he was still underwater. He had done the obvious arithmetic in his head, the arithmetic all of us do: I lost 38, I gained 38 back, that's a wash. It is not a wash. A −38% followed by a +38% leaves you at 0.62 × 1.38 = 0.856 of where you started. He was still down more than 14%, and no amount of staring at the screen was going to change it.

This is not a quirk of one trader's bad luck. It is the single most important — and most consistently underestimated — fact in all of risk management. **Losses and gains are not symmetric.** A loss does more damage than the same-sized gain undoes, and the bigger the loss, the more wildly out of proportion the gain you need to recover it becomes. This asymmetry is the mathematical bedrock that every other idea in this series sits on: position sizing, drawdown limits, the case against leverage, the value of diversification, even why "risk management is the only free lunch." All of it traces back to one curve.

What makes it so dangerous is that it hides in plain sight. The asymmetry barely shows up at the small scales where most people build their instincts — a −5% loss really does only need +5.3% back, near enough to fair that nobody notices the discrepancy. So we generalize from small losses, conclude that losing and winning are roughly even, and carry that comfortable belief straight into the one situation where it is catastrophically false: the big loss. By the time the asymmetry is large enough to feel, you are already deep in the hole it dug, doing the panicked arithmetic of a trader who has just discovered that the rules he assumed were never the rules at all. The entire purpose of this post is to move that discovery *forward* in time — to make the curve visceral before you ever need it, so that you size, limit, and behave as though the asymmetry is real, because it is.

That curve is Figure 1, and it is worth burning into memory. On the horizontal axis is the drawdown you have suffered. On the vertical axis is the gain you then need to get back to even. If the world were fair — if a 50% loss only needed a 50% gain to undo it — the relationship would follow the dashed 45° line. It does not. The real relationship is the red curve, which hugs the fair line for small losses and then peels violently away, curving up toward infinity. By the time you have lost half your money, you need to *double* what's left. By the time you've lost 90%, you need a **ten-bagger** just to break even.

![Drawdown versus gain needed to recover line chart with the minus fifty to plus one hundred and minus ninety to plus nine hundred points marked against a forty-five degree symmetric reference line](/imgs/blogs/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain-1.png)

This post owns that curve. We are going to build it from absolutely nothing — no finance background assumed — derive the exact formula, prove why losses and gains can never be symmetric, and then follow the consequences all the way out: why deep drawdowns are nearly a death sentence, why volatility itself drains your wealth even when your average return is good, and why the *order* in which your gains and losses arrive can change your final outcome by millions. By the end you will understand, in your gut and on paper, why the trader's first job is not to make money. It is to **not lose so much that the math of recovery turns against you.** Because you can only compound if you are still in the game — and the whole point of [why risk management is the real edge](/blog/trading/risk-management/why-risk-management-is-the-real-edge-surviving-to-trade-tomorrow) is that staying in the game is harder, and more valuable, than almost anyone admits.

## Foundations: the building blocks of the recovery math

Before we touch the asymmetry, let's nail down every term, from zero. If you already trade, skim this; if you don't, this section is the floor everything else stands on.

**Return.** A return is a percentage change in your money over some period. If you start a period with \$100,000 and end with \$110,000, your return was +10%. If you end with \$90,000, your return was −10%. We will write returns as fractions where it's cleaner: +10% is +0.10, −10% is −0.10. The single most important property of a return is what it does to your money: it **multiplies**. A return of *r* turns wealth *W* into *W × (1 + r)*. A +10% return multiplies your money by 1.10; a −10% return multiplies it by 0.90.

**Drawdown.** A drawdown is the drop from a peak. Specifically, if your account hit a high-water mark of \$120,000 and is now sitting at \$90,000, your drawdown is the percentage fall from that peak: (90,000 − 120,000) / 120,000 = −25%. Drawdown is the loss you actually *feel*, because it's measured from the best you ever had, not from where you started the year. It's also the number that gets people fired, gets funds redeemed, and gets traders to do something stupid out of desperation. We'll write a drawdown as a positive fraction *d* (so *d* = 0.25 means a 25% drawdown) and the wealth left after it as *W × (1 − d)*.

**Recovery.** Recovery is the gain *from the bottom of the drawdown* that gets you back to the old peak. Crucially, this gain is measured against the *smaller* post-loss balance, not the original peak. That single fact — that you must grow a shrunken base back up to a fixed target — is the entire source of the asymmetry. Hold onto it.

**Compounding.** Compounding is what happens when returns multiply across multiple periods. Two periods of returns *r₁* and *r₂* turn *W* into *W × (1 + r₁) × (1 + r₂)*. Because these factors multiply rather than add, the order in which you experience them doesn't matter *if no money moves in or out* (multiplication commutes), but the *combination* of them does — and a single zero in the chain (a −100% return, multiplying by 0) wipes everything to nothing, permanently. This is the seed of why ruin is **absorbing**: once you multiply by zero, no later factor, however large, brings you back. That absorbing property is the heart of [risk of ruin and why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough).

**Arithmetic mean vs. geometric mean.** The arithmetic mean of a set of returns is the plain average you learned in school: add them up, divide by how many. The geometric mean is the *compounded* average — the single constant return that, repeated, gives the same final wealth as the actual sequence. For anything that compounds, the geometric mean is the number that matters, and it is **always less than or equal to** the arithmetic mean. The gap between them is the whole story of volatility drag, and we'll quantify it precisely later.

One more idea, because it's the reason this whole topic is *the* spine and not just one chapter among forty. There's a special, terrible point on the loss axis: **−100%.** A −100% return multiplies your wealth by zero, and zero is **absorbing** — once you reach it, every future return, no matter how spectacular, multiplies by that zero and leaves you exactly where you are: at nothing. A +10,000% year on \$0 is still \$0. This is why ruin is categorically different from a drawdown: a drawdown is a state you can climb out of (at the punishing rates we're about to derive), but ruin is a state you can *never* leave. The recovery curve we're building doesn't just get steep as losses deepen — it runs to *infinity* as the loss approaches 100%, which is the mathematical way of saying "there is no gain large enough to recover from being wiped out." Keep that asymptote in mind: everything in risk management is ultimately about staying far enough from it that the recovery math still has an answer.

With those terms — return, drawdown, recovery, compounding, the two means, and the absorbing barrier at zero — you have everything you need. Now let's build the curve.

## The recovery formula, derived from one line of algebra

Here is the entire derivation. Suppose you suffer a drawdown of *d* (a fraction between 0 and 1). Your wealth, starting at *W*, becomes:

$$W_{\text{after}} = W \times (1 - d)$$

To get back to *W*, you need a gain *g* applied to the *reduced* balance that returns it to the full *W*:

$$W_{\text{after}} \times (1 + g) = W$$

Substitute and the *W* cancels from both sides — the result doesn't depend on how big your account is, only on the *fraction* you lost:

$$(1 - d)(1 + g) = 1 \quad\Longrightarrow\quad 1 + g = \frac{1}{1 - d} \quad\Longrightarrow\quad \boxed{\,g = \dfrac{d}{1 - d}\,}$$

That box is the spine of this entire series. The gain needed to recover a drawdown *d* is *d* divided by *(1 − d)*. Let's read it. When *d* is small, *(1 − d)* is close to 1, so *g ≈ d* — small losses are *almost* symmetric, which is exactly why small losses lull people into thinking all losses behave that way. But as *d* grows, the denominator *(1 − d)* shrinks toward zero, and dividing by a number approaching zero sends *g* toward infinity. The function doesn't just grow — it **accelerates**, and then it explodes.

Plug in the canonical values and the asymmetry becomes undeniable:

| Drawdown *d* | Gain needed *g = d/(1−d)* |
|---|---|
| −10% | +11.1% |
| −20% | +25.0% |
| −25% | +33.3% |
| −33% | +50.0% |
| −50% | **+100.0%** |
| −60% | +150.0% |
| −70% | +233.3% |
| −80% | +400.0% |
| −90% | **+900.0%** |
| −95% | +1,900.0% |

Look at how it accelerates. Going from a 10% to a 20% drawdown only adds about 14 percentage points to the recovery gain (11.1% → 25.0%). But going from an 80% to a 90% drawdown adds **five hundred** percentage points (400% → 900%). The recovery cost of the *last* ten points of a deep drawdown is enormous compared to the first ten. The curve doesn't punish you linearly for losing more — it punishes you *convexly*, with the penalty itself growing faster the deeper you go. Figure 2 plots the same table as a set of bars, so you can see exactly where the cost stops being a nuisance and becomes a wall.

![Horizontal lollipop chart of the gain needed to recover drawdowns from ten to ninety percent showing the bars exploding past the fifty percent level](/imgs/blogs/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain-2.png)

#### Worked example: the \$100,000 account at each drawdown

Take a real \$100,000 retail account and walk it down the table, step by step, so the formula becomes dollars.

- Lose **10%**: you have \$90,000. To get back to \$100,000 you need +\$10,000 on a \$90,000 base, which is +\$10,000 / \$90,000 = **+11.1%**. Mildly unfair, easy to shrug off.
- Lose **25%**: you have \$75,000. You need +\$25,000 on \$75,000 = **+33.3%**. Now the gap is opening — a quarter lost, a third needed back.
- Lose **50%**: you have \$50,000. You need +\$50,000 on \$50,000 = **+100%**. You must *double your remaining money* to undo a single bad year. A trader earning a respectable 10% a year would need roughly **seven straight years** of good returns (1.10⁷ ≈ 1.95) to recover one −50% drawdown.
- Lose **80%**: you have \$20,000. You need +\$80,000 on \$20,000 = **+400%**. You must turn \$20,000 into \$100,000 — a 5× — just to get back to where you started.
- Lose **90%**: you have \$10,000. You need +\$90,000 on \$10,000 = **+900%**. A ten-bagger to break even. At a 10%-a-year pace this is roughly **24 years** of compounding (1.10²⁴ ≈ 9.85).

*The lesson in dollars: the first half of your money is cheap to lose and expensive to replace, and every dollar of loss after that is exponentially more expensive than the last.*

The reason is the one you were told to hold onto. The gain is computed against your *shrunken* balance, but the target is your *original* one. Lose half and your engine of recovery — the capital that actually generates returns — has been cut in half too. You are trying to climb back to the same height with half the rope. Lose 90% and you're trying to scale a cliff with a tenth of the rope. The deeper the hole, the smaller the shovel you're left to dig out with. This is why, throughout this series, we keep coming back to the same blunt rule: **the cheapest dollar of risk to manage is the one you never lose.** Cutting a 50% drawdown down to a 20% one isn't a 30-point improvement in your loss — it's the difference between needing +100% and needing +25% to recover, a *four-fold* reduction in the climb back.

There's a precise way to talk about how the penalty accelerates, and it's worth doing because it explains the shape of the curve, not just its endpoints. Calculus gives the *marginal* recovery cost — how much extra gain each additional point of drawdown forces on you — as the rate of change of *g = d/(1−d)*, which works out to *1/(1−d)²*. Read that denominator: it's *(1−d)* **squared**. At a shallow 10% drawdown the marginal cost is *1/0.81 ≈ 1.23* — each extra point of loss costs about 1.23 extra points of recovery, barely more than fair. At a 50% drawdown it's *1/0.25 = 4* — each extra point of loss now costs *four* extra points of recovery. At an 80% drawdown it's *1/0.04 = 25* — each additional point of loss costs **twenty-five** points of recovery gain. The marginal price of risk doesn't creep up; it screams up, because it scales as one over the *square* of what you have left. This is the mathematical signature of every blow-up: the last few percent of a deep drawdown are astronomically more expensive than the first few, which is exactly when a desperate trader, staring at an unrecoverable hole, reaches for leverage and finishes the job.

#### Worked example: a string of small losses adds up the wrong way

People guard against the one giant loss but wave off a run of small ones — "it's only down 5% again." Stack them and the multiplication does its quiet work. Start with the \$100,000 account and suffer six consecutive −5% months (a bad but utterly ordinary half-year):

$$\$100{,}000 \times (0.95)^{6} = \$100{,}000 \times 0.7351 = \$73{,}509$$

Six "small" 5% losses didn't cost 6 × 5% = 30%; they compounded to a **−26.5%** drawdown, leaving \$73,509. And to recover *that* you don't need +26.5% — you need *d/(1−d)* = 0.2649 / 0.7351 = **+36.0%**, or +\$26,491 on your reduced base. A string of small, individually-forgivable losses quietly walks you up the steep part of the recovery curve while you're still telling yourself nothing serious has happened. *The lesson: drawdowns don't need one dramatic event to become dangerous — a series of small losses compounds into the same trap, and the recovery math doesn't care whether the hole was dug fast or slow.*

## Why losses and gains can never be symmetric

It's worth pausing to be precise about *why* this asymmetry is baked in, because the mental model is the part people get wrong, not the formula.

Your brain runs on **arithmetic**. When you hear "−50% then +50%," it adds the percentages: −50 + 50 = 0, back to even. That mental model is wrong in a specific, mechanical way: percentages are not quantities you add, they are *factors you multiply*, and they multiply against different bases. The −50% multiplies your starting balance by 0.50. The +50% then multiplies the *already-halved* balance by 1.50. The result is 0.50 × 1.50 = 0.75 — you end at 75 cents on the dollar, down 25%, not even. Figure 5 lays the wrong mental model and the real mechanism side by side.

![Before and after comparison contrasting additive percentage intuition against the multiplicative reality where minus fifty percent then plus fifty percent equals seventy five thousand dollars](/imgs/blogs/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain-5.png)

The asymmetry follows directly. A gain and a loss of the same *percentage* are not the same *multiplier* working in reverse. The reverse of multiplying by 0.50 is multiplying by 2.00 (a +100% gain), not by 1.50. To undo a multiplication by *(1 − d)* you must multiply by *1/(1 − d)*, and *1/(1 − d)* is always larger than *(1 + d)* whenever *d > 0*. That inequality — *1/(1 − d) > 1 + d* — *is* the asymmetry, written in one line. The "fair" symmetric world would need the undo-gain to equal *d*; the real world needs *d/(1 − d)*, which is strictly bigger, and the gap between them widens without bound as *d* grows.

There's a deeper way to see why this is unavoidable rather than a coincidence. A loss removes capital, and **capital is what produces future returns.** A percentage gain is a fixed *rate*, but the dollars it produces depend on the base it acts on. Shrink the base and you've handicapped every future gain. A loss therefore does double damage: it costs you the dollars directly, and it shrinks the engine that would have earned them back. A gain does only single good: it adds dollars, but it doesn't retroactively un-shrink anything. Losses compound against you with a built-in head start. *That* is why no amount of symmetric thinking can make them symmetric.

#### Worked example: the \$10,000,000 book and the "I made it all back" trap

Scale up to an institutional \$10,000,000 book and replay the trader from the introduction. He ran the book down 38% in a bad stretch:

$$\$10{,}000{,}000 \times (1 - 0.38) = \$6{,}200{,}000$$

Then he had a great year, +38%:

$$\$6{,}200{,}000 \times (1 + 0.38) = \$8{,}556{,}000$$

He is at \$8,556,000 — still **down \$1,444,000, or −14.4%** — despite "making 38% back." To actually recover from −38% he needed *d/(1−d)* = 0.38 / 0.62 = **+61.3%**, not +38%. The 23-point gap between what he needed (+61.3%) and what he earned (+38%) is the asymmetry, denominated in real money: roughly \$1.44 million of a \$10 million book, gone, hidden inside a year that *felt* like a full recovery.

*The intuition: "I made the percentage back" is one of the most expensive sentences in trading, because the percentage you lost and the percentage you need are two different numbers.*

## One big loss versus a smooth path: same average, different fate

Here is where the asymmetry stops being a curiosity and starts deciding who survives. Consider two traders who, over ten years, earn the *exact same arithmetic average return* — 10% a year. One earns it smoothly: +10% every single year. The other earns a slightly higher 16.7% in nine of the ten years but suffers a single catastrophic −50% year in the middle. Their average yearly return is identical: in both cases the ten numbers sum to 100% and average to 10%. If returns added, they'd end up in the same place. They don't.

![Two compounding paths starting at one hundred thousand dollars with identical ten percent average returns ending far apart because one path takes a single fifty percent loss](/imgs/blogs/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain-4.png)

#### Worked example: \$100,000 down two very different roads

Both traders start at \$100,000. We compound, year by year.

**The smooth trader** (+10% every year): each year multiplies by 1.10, so after ten years the account is

$$\$100{,}000 \times 1.10^{10} = \$100{,}000 \times 2.5937 = \$259{,}374$$

**The one-big-loss trader** (+16.7% in nine years, −50% in one): the nine good years multiply by 1.167 each, and the bad year multiplies by 0.50. Order doesn't change the product, so:

$$\$100{,}000 \times (1.1667)^{9} \times (0.50) = \$100{,}000 \times 4.004 \times 0.50 = \$200{,}212$$

Same average return — *identical* 10% arithmetic mean — and yet the smooth trader ends with **\$259,374** while the one-big-loss trader ends with **\$200,212**. That single −50% year cost roughly **\$59,000**, about 23% of the smooth trader's final wealth, even though it was fully "paid back" on average by the nine fat 16.7% years around it. The crash punched a hole that the surrounding good years, despite averaging out perfectly on paper, could never quite fill — because they were earning on a base the crash had already cut in half.

*The intuition: the arithmetic average of your returns lies to you about your wealth. What you keep is set by the multiplication, and one big loss is a multiplication you cannot out-average.*

This is the precise mechanism behind a phrase you'll hear from every survivor in this business: **avoid the big loss.** Not because big losses feel bad — though they do — but because the geometry of compounding gives a single deep drawdown a permanent, un-averageable cost. It is the same principle that makes [risk management the only free lunch and survival a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine): cutting the depth of your worst loss raises your long-run *compound* growth more reliably than improving your average return, because you're attacking the term in the product that does the most damage.

It's worth being blunt about an implication that traders resist: **your final wealth is set by your geometric mean, and the geometric mean is held hostage by your worst outcomes.** A useful and slightly unsettling way to see this is that the geometric return behaves a little like a *minimum* — it's dragged down toward your lowest results far more than it's pulled up by your best ones. Consider three years of returns: +50%, +50%, and −90%. The arithmetic average is a healthy (50 + 50 − 90) / 3 = +3.3% a year. But the actual compounding is 1.50 × 1.50 × 0.10 = 0.225 — you ended with **22.5 cents on the dollar**, a −77.5% disaster, despite a positive average. The single −90% year didn't get *averaged* against the two great years; it *multiplied* them down to almost nothing. The two +50% years were earning the right to be obliterated by one bad one. This is why, when you read a track record, the question that matters most is not "what was the average return?" but "**what was the worst year, and how deep was the worst drawdown?**" — because that worst number, through the multiplication, quietly sets the ceiling on everything else. A strategy with a lower average return but a shallower worst drawdown will frequently out-compound a flashier one, and the recovery asymmetry is the entire reason.

## The long climb back: drawdown and time underwater

The recovery formula tells you *how much* gain you need. It is silent on something just as brutal: *how long it takes.* A −50% drawdown needs +100% to recover, and at any realistic rate of return, +100% takes *years*. During those years you are **underwater** — below your old high-water mark — and being underwater is where the psychological and business risks of trading live. Investors redeem. Risk managers cut your size. You start to doubt the strategy, abandon it at the worst moment, and lock in the loss permanently. Figure 3 shows the shape of a recovery: the fast plunge, the long grind, and the shaded region of time spent below the peak.

![Account equity curve dropping fifty percent from a peak then slowly compounding back to even over many months with the underwater region shaded red](/imgs/blogs/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain-3.png)

#### Worked example: how many years to climb out of a 50% hole

Suppose your strategy earns a genuinely good 12% a year on average. You suffer a −50% drawdown, leaving \$50,000 of your original \$100,000. How long until you're back to even? You need to double your money, so you need *1.12ⁿ ≥ 2*. Solving:

$$n = \frac{\ln 2}{\ln 1.12} = \frac{0.6931}{0.1133} \approx 6.1 \text{ years}$$

**Six years underwater** to recover one −50% drawdown — *if* nothing goes wrong in those six years, *if* you don't lose conviction, *if* your investors don't pull capital, and *if* your edge survives unchanged. Now contrast a −20% drawdown at the same 12% rate: you need *1.12ⁿ ≥ 1.25* (since recovering −20% requires +25%), which solves to *n ≈ 1.97* years. Cutting the drawdown from 50% to 20% — a bit more than halving its depth — cuts the recovery time from **six years to two.** The time-underwater penalty, like the gain-needed penalty, is convex: deeper holes don't just need more gain, they need disproportionately more *time*, and time is the one resource a trader can't manufacture.

*The intuition: depth of drawdown and time underwater are two faces of the same coin, and both punish you faster than linearly — which is why a hard cap on drawdown depth is worth more than it looks.*

This is the asymmetry's quiet second act. The first act is "you need more gain than you lost." The second is "and you'll spend years earning it, exposed the whole time to quitting, redemptions, and a regime change that kills your edge before you finish." The depth of a drawdown isn't just a number on a chart — it's a clock, and the clock runs against you. It's also why the [high-water-mark trap in a drawdown](/blog/trading/hedge-funds/the-high-water-mark-trap-in-a-drawdown) is so corrosive for a fund: a manager underwater earns no performance fee until the old peak is reclaimed, which can take those same six years, during which the economics of the business quietly collapse even if the strategy is sound.

There's a cruel psychological trap hidden in the *shape* of a recovery, and it's worth naming because it's where so many traders abandon a sound strategy at the worst possible moment. Recovery from a deep drawdown looks discouraging for a long time, because most of the *gain* you earn early goes toward a base that's still far below the peak. Start at \$50,000 after a −50% drawdown and earn a strong +20% in your first recovery year: you're at \$60,000 — a genuinely good year, yet still **−40%** from your \$100,000 peak, because +20% on a halved base only closes a fraction of the gap. You've done well and you *feel* like you're failing, because the high-water mark hasn't moved at all. Most of the visible progress comes near the *end* of the climb, when each percentage gain is finally acting on a base close to the old peak. The danger window is the long, demoralizing middle, where a trader who doesn't understand the math concludes "this isn't working" and quits — converting a temporary drawdown into a permanent loss by their own hand. Understanding the recovery curve is partly a defense against your own despair: it tells you that slow visible progress is *expected*, not evidence of failure, and that the only fatal mistake is to stop climbing.

## Volatility drag: how variance quietly steals your compounding

We've established that *one big loss* hurts disproportionately. The deeper truth is that **volatility itself** — even without any single catastrophe — drags down your compound growth. Two strategies with the *same average return* will end up with different wealth, and the more volatile one ends up poorer. This is variously called **volatility drag**, **variance drain**, or the **arithmetic-geometric gap**, and it is one of the most underappreciated facts in investing.

Here's the result, and then the why. For a series of returns with arithmetic mean *μ* and variance *σ²* (where *σ* is the standard deviation, the usual "volatility" number), the geometric — that is, the *compounded* — growth rate is approximately:

$$g \approx \mu - \tfrac{1}{2}\sigma^{2}$$

The compounded growth you actually *keep* is your average return *minus about half your variance.* Variance is a direct, quantifiable tax on compounding. Two books can both average 8% a year, but if one is placid (low *σ*) and the other wild (high *σ*), the wild one compounds slower and ends with less money, purely because of the drag term.

Why does this happen? It's the recovery asymmetry again, in disguise. Volatility means your returns swing both above and below the average. The down-swings, by the multiplicative logic we've already proved, do more damage than the equal-sized up-swings repair. Average a +30% and a −30% and you get 0% arithmetically — but multiply 1.30 × 0.70 = 0.91, a **−9% compounded result.** That 9% gap is exactly the ½·σ² drag: with returns of ±30%, the variance is 0.30² = 0.09, and half of it is 0.045 per swing, which over the pair of moves accumulates to the loss you see. The bigger the swings, the bigger the gap, and it grows with the *square* of volatility — double your volatility and you quadruple the drag.

Where does the "half the variance" come from, exactly? It falls out of compounding directly. Compound growth lives in the *logarithm* of your returns — the geometric mean is the average of *log(1 + r)*, not of *r* itself. And the logarithm is a *concave* function: it bends downward, so it rewards a gain less than it penalizes the equal-sized loss. A short Taylor expansion of *log(1 + r)* around zero gives *r − ½r² + …*, and averaging that across your returns turns the *−½r²* term into *−½σ²* once you account for the spread of returns around the mean. The concavity of the logarithm — the same mathematical fact that makes −50% then +50% land below zero — *is* the variance drain. They are not two phenomena; they are one, viewed at two zoom levels. This is also precisely why growth-optimal sizing maximizes *expected log wealth* rather than expected wealth: log is the function that correctly prices the asymmetry, and betting to maximize it is the same as betting to keep the variance tax from eating your compounding. The full derivation lives in [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews); here, the takeaway is that the log isn't a mathematical convenience — it's the honest accounting of what compounding does to a volatile return stream.

![Compounded wealth after twenty years falling as volatility rises while the average return stays fixed at eight percent showing the gap lost to volatility drag](/imgs/blogs/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain-6.png)

#### Worked example: two \$100,000 accounts, same 8% average, different vol

Both accounts target the same 8% arithmetic average return over 20 years, but one runs at 15% volatility and the other at 30%. Using *g ≈ μ − ½σ²*:

- **Low-vol account (σ = 15%):** geometric growth *g ≈ 0.08 − ½(0.15)² = 0.08 − 0.01125 = 0.0688*, or 6.9%. After 20 years: \$100,000 × 1.0688²⁰ ≈ **\$378,000**.
- **High-vol account (σ = 30%):** geometric growth *g ≈ 0.08 − ½(0.30)² = 0.08 − 0.045 = 0.035*, or 3.5%. After 20 years: \$100,000 × 1.035²⁰ ≈ **\$199,000**.
- **The drag, in dollars:** both accounts had the *same 8% average return*, yet the high-vol account ends with **\$179,000 less** — it finishes with barely half the wealth of its calmer twin. The only difference between them was volatility, and the gap is the variance tax, compounded over 20 years. (For reference, a hypothetical zero-volatility 8% would reach about \$466,000 — the smooth ceiling that volatility drags you below.)

*The intuition: your average return is the headline, but your geometric return — average minus half your variance — is the paycheck. Cutting volatility is a way to earn more money without earning a higher average return.*

This is the rigorous version of the survival thesis. It explains why professionals obsess over *risk-adjusted* return and not raw return, why volatility targeting and Kelly sizing exist, and why "smooth and steady" genuinely beats "high and wild" even at the same average. It is also the bridge from this post to position sizing: the reason you *can't* simply lever up a good strategy to the moon is that leverage multiplies both *μ* and *σ*, but the drag term grows with *σ²* — so past a point, adding leverage adds more drag than return, and your compound growth turns *negative* even though your average return is still positive. That turning point is the Kelly limit, and the full treatment lives in [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews); here, just hold the mechanism: **variance is not free, and leverage buys you variance.**

Watch the leverage math do exactly this. Take a strategy with a 6% arithmetic edge (*μ* = 0.06) and 20% volatility (*σ* = 0.20). At leverage *L*, the levered edge is *0.06 × L* and the levered volatility is *0.20 × L*, so the compound growth is *g(L) = 0.06L − ½(0.20L)² = 0.06L − 0.02L²*. Unlevered (*L* = 1) it compounds at 0.06 − 0.02 = 4.0%. Double the leverage (*L* = 2) and it's 0.12 − 0.08 = 4.0% — *exactly the same compound growth*, because the extra return is perfectly eaten by the extra drag. Push to 3× and it's 0.18 − 0.18 = **0.0%** — you've doubled and tripled your bets to earn *nothing*, the variance drain having swallowed the entire levered edge. Past 3× the growth goes negative: at 4× it's 0.24 − 0.32 = **−8%** a year, a strategy with a positive expected return now reliably destroying capital. The optimal leverage here is *L\* = μ/σ² = 0.06/0.04 = 1.5×*, and the asymmetry is what makes "more leverage = more growth" false the instant you cross it. Every blow-up that ran 5×, 10×, or 25× leverage was operating far out on the part of this curve where leverage *subtracts* from long-run growth while it *adds* to the depth of the eventual drawdown — the worst of both, dressed up as ambition.

## Sequence-of-returns risk: when the order suddenly matters

We said earlier that, with no money flowing in or out, the *order* of your returns doesn't change your final wealth — multiplication commutes, so 1.10 × 0.90 = 0.90 × 1.10. That's true, and it's a useful sanity check. But it comes with a giant asterisk: **the moment money moves in or out of the account, order matters enormously.** This is called **sequence-of-returns risk**, or sequence risk, and it's the asymmetry wearing yet another mask.

The mechanism is simple once you see it. If you are withdrawing a fixed dollar amount each year — a fund paying fees and redemptions, a retiree drawing income, a trader pulling a salary — then a big loss *early*, when your balance is large and your withdrawal eats a huge fraction of a depleted account, does catastrophic damage you can never undo. The same big loss *late*, after years of growth, lands on a much larger base and barely matters. Same returns, same arithmetic average, wildly different outcomes — purely because of *when* the loss arrived relative to the cash flows.

![Two book value paths with identical returns and identical average but different ending wealth because one takes a forty percent loss in year one and the other in year ten under a fixed annual withdrawal](/imgs/blogs/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain-7.png)

#### Worked example: the \$10,000,000 book, a \$700,000 withdrawal, and the order of a crash

Take the \$10,000,000 book. Each year it earns a return *and then* \$700,000 is withdrawn (think management fees plus an owner's draw). Over ten years it experiences nine good years (returns of 12%, 10%, 15%, 8%, 20%, 6%, 14%, 9%, 11%) and one terrible −40% year. The *only* difference between our two scenarios is **when** the −40% year hits. The arithmetic average of the ten returns is identical either way: 6.5%.

**Crash in year 1 (loss first):**
- Year 1: \$10,000,000 × (1 − 0.40) = \$6,000,000, then withdraw \$700,000 → **\$5,300,000**. The withdrawal just bit 11.7% out of a crashed account.
- The nine good years then compound on this badly depleted base, withdrawals dragging each year. After all ten years: the book ends at roughly **\$4,158,000**.

**Crash in year 10 (loss last):**
- The nine good years run first on the full base, with withdrawals that are a small fraction of a growing account. By year 9 the book has grown to roughly \$16.8 million.
- Year 10: ×(1 − 0.40) then −\$700,000 → the book ends at roughly **\$9,365,000**.

Same ten returns. Same 6.5% arithmetic average. Same \$700,000 annual withdrawal. And yet the loss-first path ends with **\$4.16 million** while the loss-last path ends with **\$9.37 million** — a gap of **\$5.2 million**, more than half the starting capital, created by nothing but the *order* of a single bad year. The early crash forced the fixed withdrawal to consume a much larger slice of a much smaller base, year after year, locking in damage that the later good years couldn't repair because they were compounding on a permanently shrunken foundation.

*The intuition: with cash flowing out, an early loss is far deadlier than a late one — the same crash is survivable at the end of the journey and ruinous at the start.*

It's worth being precise about *why* order is harmless in a vacuum and lethal with cash flows, because the distinction trips up even experienced investors. With no money moving, your final wealth is just the product of all your *(1 + r)* factors, and multiplication commutes — 1.2 × 0.8 × 1.1 gives the same answer in any order, so the sequence genuinely doesn't matter. But a withdrawal is a *subtraction* wedged between the multiplications, and subtraction does not commute with multiplication. When you withdraw \$700,000 from a depleted \$5.3 million account, you've removed 13.2% of it; when you withdraw the same \$700,000 from a fat \$16 million account, you've removed 4.4%. The *dollar* amount is identical but its *proportional* bite depends entirely on the balance it's taken from — and the balance depends on the sequence of returns that came before. An early loss makes every subsequent fixed withdrawal a larger proportional drain, permanently. That's the whole mechanism: cash flows convert the harmless order of multiplications into a sequence that decides your fate.

Sequence risk is why the *path* of your equity curve, not just its destination, is a risk you must manage — and why it bites hardest exactly when you can least afford it: early in a fund's life, right after raising capital, when a drawdown both shrinks the base and triggers redemptions. It applies to a saver in reverse, too — putting money *in* over time means an early crash is a *gift* (you buy the recovery cheap) while a late crash, on your largest balance, is a disaster. Whichever direction the cash flows, the rule is the same: the loss that lands when your balance and its associated cash flow are largest does the most permanent damage. It is the asymmetry compounded by cash flows, and it is one more reason the prudent move is always the same: **protect the downside first, especially early, because an early loss is a tax you pay for the rest of the journey.**

## Common misconceptions

**"−50% then +50% gets me back to even."** No. Returns multiply against shifting bases: 0.50 × 1.50 = 0.75, so you're still down 25%. To undo −50% you need +100%, full stop. The symmetric intuition is wrong for *every* loss size; it just *feels* almost-right for small ones (−5% really does only need +5.3%) and then fails spectacularly as losses grow.

**"If my average return is positive, I'll make money over time."** Not necessarily. Your *compound* growth is roughly your average return minus half your variance. A strategy averaging +5% a year with 40% volatility has a drag of ½ × 0.40² = 8%, so its geometric growth is about −3% — it loses money over time *despite a positive average return.* Positive expectancy is necessary but not sufficient, which is the entire argument of [risk of ruin and why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough).

**"A drawdown is just a temporary dip — it'll come back."** Sometimes. But a −50% drawdown needs +100% to recover, which at a real 12% return takes about six years, during which you're exposed to quitting, redemptions, and regime change. And a deep enough drawdown may *never* come back: if you lose 90%, you need +900%, a ten-bagger, which most strategies will never produce. Depth determines whether "temporary" is measured in months, decades, or never.

**"Two strategies with the same average return are equally good."** Only if they have the same volatility. The more volatile one delivers less compound wealth because of variance drain. At an 8% average over 20 years, dropping volatility from 30% to 15% nearly *doubles* your terminal wealth (≈\$199k vs ≈\$378k on \$100k) — without changing your average return by a single basis point.

**"The order of my returns doesn't matter — it all averages out."** True only when no money moves in or out. Add a withdrawal (fees, income, redemptions) and an early loss becomes dramatically more destructive than a late one. The same −40% year cost \$5.2 million more when it struck in year 1 versus year 10 of the same \$10 million book. Order is harmless in a vacuum and lethal in the real world of cash flows.

**"I can just use more leverage to recover faster."** Leverage multiplies your average return *and* your volatility, but the drag grows with the *square* of volatility. Past the Kelly-optimal point, additional leverage adds more variance drain than return, and your compound growth *falls* — eventually going negative even while your average return stays positive. Trying to lever your way out of a drawdown is how a recoverable loss becomes a terminal one, which is the subject of the leverage chapters later in this series.

## How it shows up in real markets

The recovery asymmetry isn't a textbook curiosity — it is the autopsy report on nearly every famous blow-up. Here are the numbers, with real dates.

**Long-Term Capital Management, August–September 1998.** LTCM lost about **\$4.6 billion** of its capital in roughly four months, running balance-sheet leverage near **25:1** on gross notional derivative positions around **\$1.25 trillion**. The leverage is the key: a relatively modest move in the underlying convergence trades, amplified 25×, produced a drawdown so deep that recovery was mathematically hopeless within the fund's funding horizon. The Fed had to organize a **\$3.6 billion** recapitalization. The asymmetry plus leverage is the deadly combination — leverage manufactures the deep drawdown, and the *d/(1−d)* curve guarantees you can't climb out of it before your lenders pull the plug.

**Amaranth Advisors, September 2006.** Amaranth lost about **\$6.6 billion** — most of it in a single week — on concentrated, levered natural-gas calendar spreads. A book that had been up sharply for the year gave it all back and then some in days. Once a concentrated position is down 65%+, you need a +185%+ move *in the same illiquid market that just crushed you* to recover; there is no such move, so the fund simply ended.

**Archegos Capital Management, March 2021.** Archegos ran roughly **5×+ leverage** through total-return swaps on a handful of concentrated single stocks, with each prime broker blind to the total size. When the stocks fell, the levered drawdown was deep and instant; the family office was wiped out and the banks absorbed over **\$10 billion** in losses (Credit Suisse alone about **\$5.5 billion**). The leveraged depth put recovery out of reach before the margin calls finished.

**Volmageddon, February 5, 2018.** The VIX jumped about **20 points** (from 17.3 to 37.3) in a single day — the largest one-day VIX percentage spike on record — and the XIV short-volatility note fell roughly **96%** after the close and was terminated. A −96% loss needs a +2,400% gain to recover; there is no recovering from that, which is why the product didn't recover, it *ceased to exist.* The asymmetry's endgame is the absorbing barrier — a loss deep enough that the recovery gain is effectively infinite, and the position is simply gone.

**COVID crash, February–March 2020.** The S&P 500 fell about **34%** peak-to-trough (Feb 19 to Mar 23) in the fastest bear market on record, with the VIX closing at a record **82.69** on March 16. A −34% index drawdown needs +52% to recover; broad equities did claw back over the following year, which is the *survivable* end of the spectrum — painful but recoverable, precisely because 34% sits on the gentler part of the recovery curve. The lesson cuts both ways: the asymmetry is forgiving of moderate drawdowns and merciless about deep ones, and the entire job of risk management is keeping yourself on the forgiving side.

**Yen-carry unwind, August 5, 2024.** The Nikkei fell **−12.4%** in a day (its worst since 1987) as a crowded funding-carry trade unwound, with the VIX spiking to an intraday **65.7**. Leveraged carry traders saw deep, fast drawdowns; the levered ones who were forced to delever at the bottom converted a recoverable −12% market move into a permanent loss by being stopped or margined out before the snap-back. The market recovered; many of the *traders* did not, because leverage and forced selling moved them down the recovery curve to a point of no return. It is correlation and crowding as a *risk failure mode*: when everyone is in the same levered trade, the unwind makes the drawdown both deeper and faster than any single position would suggest.

The through-line is relentless: every one of these is a story of a drawdown made *deep* — usually by leverage, concentration, or a crowded trade — to the point where the *d/(1−d)* recovery gain became unachievable before the clock (margin calls, redemptions, terminated products) ran out. None of them failed because the recovery math is exotic. They failed because the recovery math is **inexorable**, and they let the drawdown get deep enough that it took over.

## The risk playbook: letting the asymmetry size your bets

The recovery curve isn't just something to understand — it's something to *obey*. Here is how this one piece of arithmetic translates directly into the rules that keep you alive.

**1. Set a hard max-drawdown limit, and set it on the steep part of the curve, not the cliff.** Decide in advance the deepest drawdown you will tolerate before you cut risk, and place it where recovery is still realistic. A −20% limit needs +25% to recover (about two years at a good clip); a −33% limit needs +50% (about four years). Once you're contemplating limits past −50%, you're committing to needing +100% and 6+ years underwater — too far. **Most professional traders cap individual-strategy drawdowns somewhere in the −15% to −25% range** precisely because that's where the recovery gain (+18% to +33%) is still achievable inside a year or two.

**2. Size every position so its worst realistic loss can't breach that limit.** Work backward from the drawdown cap. If your hard limit is −20% and a single position could plausibly lose 30% of its value in a bad event, that position can be at most (20% ÷ 30%) ≈ **two-thirds of your capital** before it alone could blow the limit — and you'd never concentrate that hard, so the real number is far smaller once you account for several positions moving together. The asymmetry sets the limit; the limit sets the size. This is the concrete link to [the Kelly criterion's growth-optimal sizing](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — Kelly is, at heart, the bet size that maximizes geometric growth *after* paying the variance-drain tax, which is why full Kelly is already aggressive and most practitioners bet a fraction of it.

#### Worked example: turning a drawdown cap into a per-trade risk budget

Make it concrete on the \$100,000 account. Suppose you set a hard −20% drawdown limit (you'll cut risk hard before \$80,000) and you risk a fixed fraction of capital per trade, exiting a position if it loses a set amount. How much can you risk per trade? Even a good strategy has losing streaks; with a 50% win rate, a run of **10 consecutive losers** is rare but entirely realistic over a career. If you want a 10-loss streak to leave you *inside* your −20% limit, you need *(1 − r)¹⁰ ≥ 0.80*, where *r* is the fraction of capital you risk per trade. Solving:

$$1 - r \ge 0.80^{1/10} = 0.9780 \quad\Longrightarrow\quad r \le 0.0220$$

So you can risk at most about **2.2% of capital per trade** — roughly \$2,200 on the \$100,000 account — to survive a 10-trade losing streak inside a −20% drawdown cap. This is the origin of the famous "**1–2% per trade**" rule of thumb: it isn't arbitrary folk wisdom, it's the drawdown cap divided through the asymmetry and a realistic losing streak. Risk 5% per trade instead, and that same 10-loss streak gives *(0.95)¹⁰ = 0.599* — a **−40% drawdown**, which needs +67% to recover and puts you deep on the steep part of the curve. The position size you can survive is *dictated*, not chosen. *The intuition: your per-trade risk isn't a matter of confidence or conviction — it's the largest number that keeps a normal losing streak from pushing you up the recovery curve into territory you can't climb back from.*

**3. Treat volatility as a cost, not just a risk.** Because compound growth is *μ − ½σ²*, cutting volatility raises your real (geometric) return even if your average return is unchanged. Prefer the smoother path. Target a volatility level and size down when realized volatility rises — you're not being timid, you're refusing to pay the variance tax. A strategy you can run at 12% volatility instead of 24% keeps roughly an extra ½(0.24² − 0.12²) = 2.16% of compound growth a year, for free.

**4. Protect the downside hardest when you're early or large.** Sequence risk means an early loss, or a loss while the base (and the withdrawal it must support) is largest, does the most permanent damage. New strategy, freshly raised capital, peak account size — these are the moments to carry the *least* risk, not the most. Build your risk budget to be most conservative exactly when a drawdown would be most destructive to the path.

**5. Refuse to lever your way out of a hole.** When you're down, the temptation is to double up to recover faster. The math says the opposite: leverage deepens drawdowns and adds variance drain that grows with *σ²*, so the very move that promises a fast recovery is the one most likely to push you from a recoverable drawdown to a terminal one. Down is the time to *reduce* risk, restore the engine, and let the gentler part of the recovery curve do the work. For the firm-level version of this discipline — how funds institutionalize it and how it shows up in investor terms — see [the high-water-mark trap in a drawdown](/blog/trading/hedge-funds/the-high-water-mark-trap-in-a-drawdown).

The whole playbook reduces to one sentence, which is also the thesis of this entire series: **the deepest drawdown you ever take is the most important number in your career, so cap it before it caps you.** Everything else — measuring risk, sizing positions, diversifying, hedging tails — is machinery built to keep that one number small. The trader who internalizes the *d/(1−d)* curve doesn't need to be reminded to cut losses; the curve does the reminding, in the only language markets respect: arithmetic.

### Further reading

- [Why risk management is the real edge: surviving to trade tomorrow](/blog/trading/risk-management/why-risk-management-is-the-real-edge-surviving-to-trade-tomorrow) — the series intro and the survival thesis this post makes precise.
- [Risk of ruin: why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) — what happens when a positive edge meets the absorbing barrier at zero.
- [Risk management, the only free lunch: survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — why cutting tail losses raises long-run growth more than any return forecast.
- [The high-water-mark trap in a drawdown](/blog/trading/hedge-funds/the-high-water-mark-trap-in-a-drawdown) — the asymmetry from the GP seat, where time underwater quietly breaks the business.
- [The Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — the formal math of sizing to maximize geometric growth after the variance tax.
