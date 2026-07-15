---
title: "Loss Aversion and the Disposition Effect: Why You Cut Winners and Ride Losers"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A practitioner's deep dive into the single most expensive bias in trading: why the shape of how gains and losses feel makes you snatch small profits and cling to big losses, what the data says, and the drills that fix it."
tags: ["loss-aversion", "disposition-effect", "prospect-theory", "behavioral-finance", "trading-psychology", "risk-management", "cutting-losses", "let-winners-run"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The most expensive habit in trading is baked into the shape of how gains and losses feel, not into any single bad decision.
>
> - **Loss aversion**: a loss hurts roughly *twice* as much as an equal gain feels good. The best estimate of the "pain multiplier" is $\lambda \approx 2.25$ (Tversky & Kahneman, 1992).
> - **The value curve bends two ways**: you become *risk-averse when you're ahead* (so you snatch small winners) and *risk-seeking when you're behind* (so you gamble on losers to get back to breakeven). This is the reflection effect.
> - **The disposition effect** is the market fingerprint of that shape: in 10,000 real brokerage accounts, investors sold winners at **14.8%** and losers at just **9.8%** — realizing gains about 1.5x as often as losses (Odean, 1998).
> - **It's not even tax-smart** — the winners people sold went on to *outperform* the losers they kept by about **3.4 percentage points** over the next year. The one honest fix is to stop letting your entry price vote.
> - **The drill**: ask "would I buy this, right here, right now, at this price?" If no, sell — winner or loser. Then pre-commit your exit with a mechanical stop before you ever feel anything.

You have almost certainly done this, and you probably did it this week.

A stock you own is up 8%. You feel a little jolt of satisfaction and a louder whisper of anxiety: *what if it gives it all back?* You sell, book the small win, and feel clever. Meanwhile, another position is down 22%. Selling it would make the loss *real*, so you tell yourself the story hasn't changed, you'll "give it room," and you hold. Maybe you even add more to "average down."

Zoom out a year and the pattern is brutal in its symmetry: a graveyard of tiny, satisfying winners and a handful of enormous, unhealed losers. You were right often enough. You still lost money. Not because you picked badly — because of *when you chose to sell*.

That asymmetry has a name, a Nobel Prize behind it, and a shape you can literally draw. The diagram below is the mental model the whole article tours: a curve that is gently sloped above zero and steeply sloped below it. Every mistake in this post falls out of that single kink.

![The prospect-theory value function: a curve that is shallow and concave over gains but steep and convex over losses, so a $100 loss drops about 2.25x as far as a $100 gain lifts it.](/imgs/blogs/loss-aversion-and-the-disposition-effect-1.webp)

Look at the right side (gains): the curve rises, but it *flattens* — each extra dollar of profit thrills you a little less. Now the left side (losses): the curve drops *steeply* right away, then also flattens. The drop is far deeper than the rise is high. That is the entire engine of self-sabotage, and the rest of this piece is just following its gears into your P&L — and then out of it.

This is educational, not financial advice. The numbers in the *worked examples* are round and hypothetical so you can do them in your head; every number attributed to a *study or market* is real and sourced at the end.

## Foundations: the building blocks of loss aversion

Before the trading, the psychology. You don't need any finance background for this section — just a willingness to notice how your own mind prices a win against a loss.

### What "loss aversion" actually means

**Loss aversion** is the finding that losses feel more intense than equivalent gains. Not *slightly* more — roughly *twice* as much. Losing \$100 delivers about as much pain as winning \$225 delivers pleasure. The two are not mirror images; the loss side is amplified.

This came out of **prospect theory**, the framework Daniel Kahneman and Amos Tversky published in *Econometrica* in 1979 — the work that won Kahneman the 2002 Nobel Prize in Economics (Tversky had died in 1996 and Nobels are not awarded posthumously). Prospect theory replaced the tidy economic assumption that people value outcomes by their final wealth with something messier and truer: people value **changes** — gains and losses relative to a **reference point** — and they weigh those changes on a bent, asymmetric scale.

Three terms, defined once and used forever after:

- **Reference point** — the anchor you measure gains and losses *from*. In markets it is almost always your **entry price** (what you paid). This is the villain of the whole story: it is arbitrary, it is in the past, and your brain treats it as the center of the universe.
- **Value function** — the curve in the figure above. It maps an *objective* outcome (you're up \$100, you're down \$100) to a *subjective* value (how good or bad that feels). Its shape is the mechanism.
- **Loss-aversion coefficient (**$\lambda$**)** — the number that says how much steeper the loss side is than the gain side. Tversky & Kahneman's 1992 follow-up ("Advances in Prospect Theory") estimated it at $\lambda \approx 2.25$ from experimental data. That is where "losses hurt about twice as much" comes from — it's the median human, measured.

Written out, the value function they fit looks like this:

$$v(x) = \begin{cases} x^{0.88} & \text{if } x \ge 0 \ \text{(a gain)} \\ -2.25\,(-x)^{0.88} & \text{if } x < 0 \ \text{(a loss)} \end{cases}$$

Here $x$ is the objective outcome (positive for a gain, negative for a loss) and $v(x)$ is what it feels worth. Two features do all the work. The exponent **0.88** (less than one) makes each side **curve toward flat** — the *diminishing sensitivity* that bends the line. And the **2.25** multiplier on the loss branch makes the drop steeper than the rise — that's $\lambda$, loss aversion, in the algebra. You will never need to compute this. You only need to see that the two branches are *not* symmetric.

#### Worked example: the asymmetric arithmetic of a good day and a bad day

Suppose you make \$100 on Monday and lose \$100 on Tuesday. Objectively you are flat — \$0 net. But run it through the value function:

- Monday's gain: $v(+100) = 100^{0.88} \approx +57$ "units of feeling."
- Tuesday's loss: $v(-100) = -2.25 \times 100^{0.88} \approx -129$ units.

Net *felt* result: $+57 - 129 = -72$ units. A financially break-even two days leaves you feeling like you're **down**. Now flip it: to *feel* even after losing \$100, you'd need a gain big enough that $g^{0.88} = 129$, i.e. $g \approx \$250$. You need to make roughly **\$250** to emotionally erase a **\$100** loss.

**Intuition:** break-even in dollars is a loss in your gut — which is exactly why "getting back to even" feels so urgent that you'll take stupid risks to do it.

### Why the reference point is the trap

Nothing about your entry price predicts the future. The market has no memory of what you paid; the next tick is set by supply and demand *right now*, not by your cost basis. Yet loss aversion nails your reference point to that entry price and then measures everything — your fear, your greed, your urge to sell — against it.

A useful reframe you'll use for the rest of the article:

> The market doesn't know you own it, doesn't know what you paid, and doesn't owe you a round trip back to your entry. Your cost basis is a fact about your past, not a fact about the asset's future.

Everything that follows — cutting winners, riding losers, "averaging down," refusing to sell "until it comes back" — is what happens when a mind that feels losses at 2.25x is allowed to make decisions relative to a number that no longer matters.

### Mental accounting and the break-even effect

Two more pieces complete the machine, and both come from Richard Thaler's work on how we file money in our heads.

**Mental accounting** is the habit of sorting money into separate mental "accounts" and refusing to treat them as fungible, even though a dollar is a dollar. In trading, every position gets its own scoreboard: this trade is "winning," that one is "losing," and you feel compelled to close each account *in the black*. A rational investor cares only about the total portfolio and its future; the mental accountant cares about the win/loss status of each individual ticket. That per-ticket bookkeeping is what makes you unable to sell a loser — closing it means stamping "LOSS" on that specific mental account, which the value curve prices at 2.25x pain.

**The break-even effect** is the sharpest edge of it. In a 1990 study memorably titled "Gambling with the House Money and Trying to Break Even," Thaler and Eric Johnson showed that after a loss, people turn *specifically* risk-seeking in ways that could return them to their starting point. A gamble that gets you back to even is disproportionately attractive — you'll accept worse odds for it than you would in any other situation. That is the disposition effect's held loser in one sentence: the loser isn't kept because you think it's a good asset, it's kept because selling forecloses the chance of getting back to even, and the breakeven point exerts its own gravitational pull.

Its mirror image, from the same paper, is the **house-money effect**: after a *gain*, people get more willing to take risk, treating the profit as "the house's money" rather than their own. It's why a hot streak can make you reckless. In the disposition story, though, the break-even effect is the more expensive of the two.

#### Worked example: the break-even gamble

You're down \$500 on a position. A strange genie offers you a choice: accept the \$500 loss now, or flip a fair coin — heads you're back to even (the loss erased), tails you lose another \$500 (down \$1,000 total).

- Expected value of the coin flip: $0.50 \times \$0 + 0.50 \times (-\$500) = -\$250$ of *additional* loss on top of the \$500 you're already down. The flip is strictly worse than simply taking the \$500 loss.
- Most people take the flip anyway. The pull of "back to even" overrides the arithmetic. Now notice: "hold the −\$500 loser and hope it round-trips" *is* this coin flip — just played out in slow motion over days instead of one toss.

**Intuition:** riding a loser isn't investing, it's repeatedly accepting a negative-expected-value bet whose only appeal is the chance to make the loss disappear.

## 1. The reflection effect: risk-averse when ahead, risk-seeking when behind

Here is the subtle, dangerous part. Loss aversion is not just "losses hurt more." It's that the *curvature* flips as you cross zero — and that flip changes your appetite for risk depending on which side of your entry price you're on.

Look again at the value curve. On the **gain** side it is **concave** (bulging up, then flattening). On the **loss** side it is **convex** (bulging down, then flattening). Kahneman and Tversky called this the **reflection effect**: preferences for risk *reflect* — reverse — around the reference point.

Concavity over gains makes you **risk-averse** when you're winning. A bird in the hand (a sure \$50) feels better than a coin-flip for \$100, because the second \$50 adds *less* joy than the first. So when a trade is green, you reach to lock it in.

Convexity over losses makes you **risk-seeking** when you're losing. A sure loss of \$50 feels *worse* than a coin-flip that's either \$0 or \$100 lost, because the second \$50 of loss hurts *less* than the first. So when a trade is red, you'd rather gamble — hold, hope, average down — than accept the certain, smaller pain of selling now.

Put those two together in one picture and you have the machine that snatches winners and rides losers:

![A 2x4 matrix contrasting being ahead vs behind: ahead the curve is concave, the extra dollar feels less thrilling, you turn risk-averse and sell winners too early; behind the curve is convex, the extra dollar of loss hurts less, you turn risk-seeking and ride losers too long.](/imgs/blogs/loss-aversion-and-the-disposition-effect-2.webp)

The same person, obeying the same curve, makes two *opposite* mistakes depending only on whether the position is above or below the reference point. That's why this is so hard to fix by willpower: you're not fighting one bad habit, you're fighting a shape that inverts itself right at the line that matters most to you.

#### Worked example: the two coin flips that expose your reflection

This is a version of Kahneman and Tversky's original experiment. Answer both fast, with your gut.

**Choice A (you're in the money):** Take a *guaranteed* \$900, or take an 80% chance at \$1,200 (and 20% chance at \$0).

- Expected value of the gamble: $0.80 \times \$1{,}200 = \$960$. The gamble is worth **\$60 more** than the sure thing.
- Most people take the sure \$900 anyway. Concave-over-gains = risk-averse. You leave \$60 of expected value on the table to avoid the small chance of walking away with nothing.

**Choice B (you're under water):** Accept a *guaranteed* \$900 loss, or take an 80% chance of losing \$1,200 (and 20% chance of losing \$0).

- Expected value of the gamble: $0.80 \times (-\$1{,}200) = -\$960$. The gamble is **\$60 worse** than just accepting the loss.
- Most people take the *gamble* anyway. Convex-over-losses = risk-seeking. You accept \$60 of *worse* expected value for the chance to avoid a certain loss.

Same numbers, mirror-imaged. In gains you overpay for certainty; in losses you overpay for a lottery ticket. In a trading account, Choice A is "sell the winner at +9% before it can pull back," and Choice B is "hold the −20% loser because it *might* come back."

**Intuition:** you are not risk-averse or risk-seeking as a personality — you are *both*, flipping automatically at your entry price, and both flips cost you money.

## 2. From feeling to trade: how the curve becomes your P&L

It's one thing to describe a curve, another to trace how it becomes an actual sell button getting clicked. The path is short and mechanical:

![A branching flow: one entry splits into price-up (a gain) and price-down (a loss); the gain routes through the concave curve and fear of giving it back to selling the winner, the loss routes through the convex curve and the urge to reach breakeven to holding the loser, and both converge on a year-end P&L of many big losers and a few tiny winners.](/imgs/blogs/loss-aversion-and-the-disposition-effect-4.webp)

Follow the two paths. You open a position at your entry price — the reference point is set. Then the market does one of two things:

- **It goes up.** You're now in the concave region. Each additional tick of profit thrills you less, while the *fear of giving back* what you already have grows. The path of least emotional resistance is to **sell the winner** and lock in a small, certain, satisfying gain.
- **It goes down.** You're now in the convex region. Accepting the loss means crossing back through the steepest part of the curve — the most painful act available to you. Holding, by contrast, keeps the loss "on paper," unrealized, still deniable. The path of least resistance is to **hold the loser** and gamble on a return to breakeven.

Both paths converge on the same destination: a year-end P&L stuffed with a few tiny realized winners and a pile of large unrealized (or eventually, catastrophically realized) losers. You didn't decide to build that portfolio. The curve built it for you, one "sensible" sell and one "give it room" hold at a time.

This is worth sitting with, because it reframes the whole problem. **Cutting winners and riding losers is not a discipline failure that happens occasionally. It is the *default output* of an un-modified human mind interacting with a price that moves.** Doing the right thing requires actively overriding the machine — which is why every fix in this article is structural, not motivational.

The loss side gets extra reinforcement from a bias you already know from everyday life: the **sunk-cost fallacy**. Once you've committed money (and ego) to a position, walking away feels like admitting the commitment was wasted, so you throw good money after bad to "justify" the original decision — the same instinct that keeps people sitting through a bad movie because they paid for the ticket. In a trade, that shows up as *averaging down*: the price you already paid becomes a reason to buy *more* of a falling asset, which is exactly backwards. The entry price is a sunk cost. It is gone whether you sell or hold; the only question that matters is what the position is likely to do *from here*. Add regret aversion on top — selling a loser converts a deniable "paper" mistake into an undeniable, timestamped one you'll have to own — and the loser gets three separate psychological locks on the exit door: loss aversion, sunk cost, and regret. No wonder willpower loses.

## 3. The recovery math: why riding a loser is a mathematical trap, not just an emotional one

Even if the emotions didn't exist, riding losers would still be a terrible idea, because the *arithmetic* of drawdowns is savagely asymmetric. A loss and the gain required to erase it are not the same size — and the gap explodes as the loss deepens.

The rule is simple. If you lose a fraction $L$ of your capital, the gain $G$ you need to get back to breakeven is:

$$G = \frac{L}{1 - L}$$

Because the loss shrinks the base you're now growing from, the required gain outruns the loss. Draw it and the curve looks like a wall:

![A convex curve rising ever more steeply: the horizontal axis is the size of the loss you are sitting on and the vertical axis is the gain needed to get back to breakeven, showing a 10% loss needs only +11% but a 50% loss needs +100% and a 90% loss needs +900%.](/imgs/blogs/loss-aversion-and-the-disposition-effect-3.webp)

The left end is gentle. The right end is a cliff. That shape is why "let me give it room to come back" is such a devastating phrase: every extra percent you let a loser fall makes the comeback disproportionately less likely.

#### Worked example: a \$10,000 account and the two ways to lose

You have \$10,000. Watch what the recovery requirement does as a single loser deepens.

| Loss you took | Account left | Gain needed to get back to \$10,000 |
|---|---|---|
| −10% | \$9,000 | **+11.1%** |
| −20% | \$8,000 | **+25%** |
| −33% | \$6,700 | **+50%** |
| −50% | \$5,000 | **+100%** (a *double*) |
| −75% | \$2,500 | **+300%** |
| −90% | \$1,000 | **+900%** |

At −10%, the hole is trivial: a good week fills it. At −50%, you now need the position to *double* just to get you back to where you started — and doubling is a rare, hard, time-consuming event. At −90%, you need a **ten-bagger** to break even, which almost never arrives before the position (or your patience, or your capital) is gone.

Now the same math from the discipline side. Suppose you cut every loser at −8% (a common mechanical rule). To wipe out an 8% loss you need only about **+8.7%** — an ordinary move you'll get many chances at. The small stop keeps you permanently on the *flat*, forgiving part of the recovery curve, where mistakes are cheap to fix.

**Intuition:** small losses live in a world where recovery is easy; big losses live in a world where recovery is nearly impossible — and the disposition effect is a machine for marching you from the first world into the second.

There's a hidden second cost the table doesn't show: **time**. Even if a −50% loser eventually doubles to get you back to even, look at what you paid for that round trip — potentially *years* in which that capital produced nothing while it clawed back to the starting line. A market that compounds at, say, 8% a year would have roughly doubled your money over the same stretch had it been deployed anywhere else. So the true cost of riding a deep loser isn't just the drawdown risk; it's the *opportunity cost* of dead money trapped in a recovery mission while the rest of the world compounds. Cutting the loser small and redeploying keeps your capital working instead of serving a sentence.

## 4. The disposition effect: what 10,000 real accounts revealed

Everything so far is theory and arithmetic. Does it actually show up when real people trade real money? Yes — so reliably that it has its own name.

In 1985, Hersh Shefrin and Meir Statman published a paper in the *Journal of Finance* with a title that is also a perfect one-sentence summary of the bias: **"The Disposition to Sell Winners Too Early and Ride Losers Too Long."** They coined the term **disposition effect** and tied it directly to prospect theory, mental accounting, and regret aversion — the machinery from the sections above.

Then, in 1998, Terrance Odean put it to the test with data instead of theory. In another *Journal of Finance* paper — **"Are Investors Reluctant to Realize Their Losses?"** — he analyzed the trading records of **10,000 accounts** at a large U.S. discount brokerage, covering January 1987 through December 1993. He built two simple ratios:

- **PGR — Proportion of Gains Realized**: of all the winning positions an investor *could* have sold on a given day, what fraction did they actually sell?
- **PLR — Proportion of Losses Realized**: same question for losing positions.

If people were rational (or even neutral) about their entry price, PGR and PLR would be roughly equal. They were not.

![A bar chart of Odean's 1998 results: outside December, the proportion of gains realized is 14.8% versus 9.8% for losses; in December the pattern reverses to 10.8% and 12.8% as tax-loss selling kicks in.](/imgs/blogs/loss-aversion-and-the-disposition-effect-5.webp)

Across the sample, **PGR = 0.148 and PLR = 0.098**. Investors realized their winners at roughly **1.5 times** the rate they realized their losers — and the gap was statistically overwhelming (Odean reported a t-statistic above 35, meaning the odds this is noise are effectively zero). Put plainly: a stock that was up was about **50% more likely to be sold** than a stock that was down. The disposition effect isn't a tendency you can wave away as a few undisciplined people; it's the *typical* behavior of the *typical* account.

### The tax twist that proves it's irrational

Here is the detail that turns "interesting" into "damning." In most tax systems, the *smart* thing to do is the exact opposite of the disposition effect: you should **realize losses** (to harvest the tax deduction) and **defer gains** (to postpone the tax bill). Tax law literally pays you to cut losers and hold winners.

Odean saw the tax-rational behavior appear — but only in **December**, when the tax deadline concentrates the mind. In December the pattern flipped: PGR fell to **10.8%** and PLR rose to **12.8%**, as investors finally sold losers to book the deduction before year-end. For the other eleven months, the emotional pull of the reference point overwhelmed the financial incentive. People knew (or could have known) the tax-smart move and did the opposite anyway, for eleven-twelfths of the year.

**When it breaks:** the disposition effect is strongest exactly where it costs the most — in taxable accounts, in volatile names, and in the months when no external deadline forces your hand.

## 5. A year of P&L: the disposition tax versus the mirror discipline

Let's make the cost concrete with a full year of trading, because the disposition effect doesn't bill you on any single trade — it bills you in the *shape of your distribution*: the average size of your winners versus the average size of your losers.

![A two-column comparison of a year of trades: the disposition trader sells winners fast for +$350 total but holds losers to −$3,000, netting −$2,650; the mirror trader lets winners run to +$3,500 and stops losers out at −$450, netting +$3,050.](/imgs/blogs/loss-aversion-and-the-disposition-effect-6.webp)

#### Worked example: two traders, same picks, opposite habits

Two traders take the *same* six trades — three that go their way, three that don't. Same entries, same instruments, same market. The only difference is *when they sell*.

**The disposition trader** (obeys the curve):

- Winners, sold fast to lock the good feeling: +\$120, +\$80, +\$150 → **+\$350 total**.
- Losers, held and hoped: −\$900, −\$1,400, −\$700 → **−\$3,000 total**.
- **Net for the year: −\$2,650.** Three wins, three losses, deeply negative. A 50% hit rate that still bleeds, because the average loser (−\$1,000) is more than eight times the average winner (+\$117).

**The mirror trader** (inverts the habit — cut losers, let winners run):

- Winners, allowed to run to their targets: +\$1,200, +\$800, +\$1,500 → **+\$3,500 total**.
- Losers, stopped out mechanically and small: −\$150, −\$140, −\$160 → **−\$450 total**.
- **Net for the year: +\$3,050.** The same 50% hit rate, now profitable, because the average winner (+\$1,167) dwarfs the average loser (−\$150).

The swing between these two traders is **\$5,700** on identical picks. Nothing about their forecasting skill differs. Everything about their *exit rule* does. The disposition trader is "right" just as often — and goes broke.

**Intuition:** your win *rate* barely matters; your win-size-to-loss-size ratio is where the disposition effect quietly drains the account, one small snatched winner and one nursed loser at a time.

## What it looks like at the screen

Biases feel abstract in a blog post and completely invisible in the moment — which is the problem. So here is the disposition effect as it actually *feels*, tick by tick, so you can catch yourself in the act. These are the real-time tells:

- **The green-position itch.** A trade goes +5%, +6%, and you feel a rising, physical need to "just take it." You start refreshing the quote. You catch yourself mentally spending the profit. The thought is always some version of *"lock it in before it's gone."* That itch is concavity-over-gains talking. The winner has done nothing wrong; your curve has simply flattened.
- **The red-position squint.** A trade goes −10% and you *stop looking at it directly*. You check every other position first. When you do look, you find yourself reading news that confirms the thesis and skipping news that threatens it. You mentally re-file it from "trade" to "long-term hold." That squint is loss aversion protecting you from the steep part of the curve.
- **The breakeven bargain.** The loser rallies back toward your entry and you hear yourself think *"if it just gets back to what I paid, I'll get out."* Notice what that sentence assumes: that your entry price is a meaningful level for *this asset*. It isn't. It's a meaningful level for your *ego*. The market will not honor the bargain.
- **The averaging-down rationalization.** *"I liked it at \$50, I love it at \$40."* Sometimes true. But feel the motive underneath: are you adding because your thesis is stronger at \$40, or because a bigger position at a lower average makes the round-trip to breakeven *feel* closer? If it's the second, that's convexity-over-losses dressed up as conviction.
- **The relief spike on a small winner and the numbness on a nursed loser.** Selling a +3% winner gives a disproportionate hit of relief; the −25% loser you've held for months produces a dull, familiar ache you've learned to ignore. Your emotions are calibrated to the curve, not to the dollars. The dollars say the loser is the emergency.

If you can name the feeling while it's happening — *"that's the green-position itch"* — you've already created the half-second of distance you need to run a rule instead of a reflex. Half a second is enough. The whole battle is fought there.

## Why the bias is so hard to unlearn

If loss aversion were mere ignorance, one good article would cure it. It isn't. The asymmetry is old wiring, which is exactly why *knowing* about it barely dents your behavior in the moment.

For most of human history, an organism that treated a loss — a missed meal, an injury, a lost foothold — as equivalent to a gain of the same size would have been out-survived by one that treated losses as more urgent. Under scarcity, the downside of a loss could be existential (falling below the survival threshold), while the upside of an equal gain was merely nice. A brain tuned to *feel losses more* kept its owner alive. That tuning is still running when you stare at a red position; it just fires on account balances now instead of on calories.

Neurologically, the loss reaction is fast, automatic, and emotional — it lives in the quick, intuitive system that reacts *before* your deliberate, analytical system has finished forming a sentence. By the time the reasoning part of your brain says "the cost basis is irrelevant to the future," the loss-averse part has already flooded you with the urge to avoid realizing the loss. You are not refereeing a debate between two equal voices; you're trying to out-argue a reflex that got a head start. (For the underlying machinery, see [the neuroscience of risk and reward](/blog/trading/trading-psychology/the-neuroscience-of-risk-and-reward).)

This is why every durable fix here is *structural*. You cannot reliably win a real-time argument against a reflex that is faster than your reasoning and older than your species. What you *can* do is make the decision in advance — when the position doesn't exist yet and the reflex has nothing to grab — and then bind your future self to it with a mechanical order. Willpower is you fighting the curve live, and losing most rounds. Structure is you fighting the curve once, in advance, while calm, and then never having to fight it again on that trade.

## Common misconceptions

**"I just need more discipline / willpower."** The disposition effect is the *default* output of a normal brain, not a character flaw you can grit your way past. Willpower is a finite resource that fails exactly when you're stressed, tired, and staring at a red position — i.e., precisely when you need it. Every durable fix in this article replaces willpower with *structure* (pre-committed stops, mechanical rules) so the decision is already made before the emotion arrives.

**"Holding a loser isn't a loss until I sell — it's still 'on paper.'"** This is the single most expensive sentence in investing. Your net worth does not care about the label "realized." A \$5,000 position that's now worth \$3,000 is a \$3,000 position, full stop; the \$2,000 is *already gone*. Refusing to sell doesn't preserve the money — it only preserves the *story* that you haven't lost it, while exposing you to losing even more.

**"Taking profits is always smart — you can't go broke booking a gain."** You absolutely can. A strategy of tiny winners and large losers goes broke *reliably*, even with a good hit rate (see the year-of-P&L example: 50% wins, −\$2,650). "You never go broke taking a profit" is true only if you also never take a large loss — and the same instinct that makes you snatch profits makes you hold losses. The two habits are one bias wearing two masks.

**"If I sell my winner and it keeps rising, I'll feel like an idiot — so it's safer to sell."** You're optimizing to avoid *regret*, not to make money. Regret is real, but it's a feeling, not a P&L line. The trader who lets winners run will "feel like an idiot" on every winner that pulls back after they *didn't* sell — and will still crush the trader who caps every winner, because a handful of big runners pays for all the small give-backs.

**"Averaging down is just buying the dip / being greedy when others are fearful."** Buying more can be correct — *if* your reason is a fresh, independent judgment that the asset is a better buy at the lower price, made as though you had no position. It is almost always *wrong* when the real reason is to lower your average cost so breakeven feels nearer. The tell: would you open this position, at this size, at this price, if you were flat? If not, you're not buying the dip; you're feeding the loser.

**"The disposition effect is a retail-investor problem; professionals don't have it."** It's weaker in pros, but it's there. It's been documented in mutual fund managers, in professional futures traders, in home sellers refusing to list below purchase price, and in market-makers. The reference point is a feature of human cognition, not of inexperience. Pros beat it with *systems*, not with immunity.

**"I'll sell as soon as it gets back to breakeven."** This is the break-even effect wearing business-casual. It treats your entry price as a level the asset is *obligated* to revisit — but the market prices the future, not your history. Worse, if the position does rally back to your cost, you'll often refuse to sell there too (now it has "momentum"), and if it doesn't, you bag-hold indefinitely. "Back to even" is a feeling, not a plan, and the asset has never heard of it.

**"Stops just guarantee I sell at the bottom and get shaken out."** Sometimes a stop exits right before a bounce — that's the visible, memorable cost, and it stings. The invisible benefit is every −8% loss a stop stopped from becoming a −50% disaster. You *notice* the shake-outs because they're salient; you never see the counterfactual catastrophes the stop quietly prevented. Judge stops on the whole distribution of outcomes, not on the one time it hurt.

## How it shows up in real markets

### 1. The Odean study: the winners you sell beat the losers you keep

Odean's 1998 dataset is the canonical case, and it delivered a finding more damning than "investors are emotional." He tracked what happened to the stocks *after* investors sold or held them. If people had some hidden skill — selling winners that were about to peak, holding losers that were about to bounce — the disposition effect might be defensible.

The opposite was true. The winning stocks investors **sold** went on to **outperform** the losing stocks they **held** by roughly **3.4 percentage points over the following year**. Investors were systematically dumping their *best* horses and clinging to their *worst* ones — and then the best horses kept running and the worst horses kept lagging. The reference point wasn't just irrational; it was pointed exactly backwards.

### 2. Barber & Odean: trading is hazardous to your wealth

Two years later, Barber and Odean widened the lens in a *Journal of Finance* paper with a title that says it all: **"Trading Is Hazardous to Your Wealth"** (2000). Across **66,465 households** at a large discount broker from 1991–1996, they found the households that traded the *most* earned an annual return of about **11.4%**, while the market returned roughly **17.9%** over the same window — a gap of more than six percentage points a year, torched by overtrading and the costs that come with it. The disposition effect is one thread in that damage: when your winners get cut short and your losers get nursed, high activity just accelerates the bleed.

### 3. The pattern replicates everywhere researchers look

The disposition effect is one of the most robust findings in behavioral finance because it keeps showing up in fresh data. Grinblatt and Keloharju documented it across the *entire* population of Finnish investors. Andrea Frazzini showed that even professional mutual-fund managers exhibit it, and that it causes prices to *underreact* to news (because holders of losers refuse to sell into bad news, prices adjust slowly). It's been found in real-estate markets, in experimental trading games, and across retail brokerages worldwide. Different countries, different assets, different decades — same steep-sided curve underneath.

### 4. The crypto retail crucible

Modern retail crypto trading is the disposition effect under a magnifying glass: extreme volatility, 24/7 markets, no closing bell to force a decision, and a culture that turns "never sell at a loss" into an identity ("HODL," "diamond hands"). The same asymmetry that made an investor ride a −22% stock now makes them ride a −80% token, deep in the vertical part of the recovery curve where a comeback requires a +400% move. The bias didn't change; the volatility just raised the stakes on every un-cut loser. When you see someone "holding through" a 90% drawdown as a badge of honor, you're watching convexity-over-losses cosplay as conviction.

### 5. Barings Bank: riding a loser all the way to zero

The disposition effect's logical endpoint is not a bad year — it's a crater. In 1995, a single trader named Nick Leeson destroyed Barings, a British merchant bank founded in **1762** that had helped finance the Napoleonic Wars.

Leeson, running Barings' Singapore futures desk, took large bets that the Japanese Nikkei index would rise. When it fell instead, he did exactly what the value curve demands: rather than accept the loss, he buried it in a secret error account — the now-infamous account number **88888** — and *doubled down*, adding to the losing position to engineer a return to breakeven. Then, on 17 January 1995, the Kobe earthquake sent the Nikkei tumbling, and his losing bet became a catastrophe. True to the break-even effect, Leeson responded by increasing his bets *again*, wagering on a rapid Nikkei recovery that never came.

By the time he fled on 23 February 1995, the hidden losses had reached about **£827 million** (roughly US\$1.4 billion) — more than *twice* Barings' entire available trading capital. The bank was insolvent within days and was sold to the Dutch group ING for a symbolic **£1**. Leeson was sentenced to six and a half years in a Singapore prison.

Leeson committed fraud, and the concealment is its own story. But strip the fraud away and the *trading* mistake is the plain disposition effect at industrial scale: a loser he could not bring himself to close, held and averaged and prayed over in an ever-more-desperate reach for breakeven, until the loss grew larger than the institution housing it. Every retail trader "averaging down" on a −40% position is running Leeson's exact playbook, differing only in the count of zeros. The lesson isn't "don't commit fraud." It's that a loss you refuse to realize does not stop growing just because you've stopped looking at it.

## The drill: cut the cord from your cost basis

You cannot delete the value curve from your brain. You *can* build machinery that makes the decision before the curve gets a vote. Two tools, in order.

### Tool 1 — the flip test

The disposition effect lives entirely in the gap between your entry price and the current price. So close the gap: make every hold/sell decision as if you had *no* position and no cost basis. One question does it:

![A decision flow: start from your open position while ignoring your entry price, ask whether you would open this position right now at this exact price, and if yes hold or add, if no sell now because the cost basis is irrelevant.](/imgs/blogs/loss-aversion-and-the-disposition-effect-7.webp)

**"Would I open this position, at this size, right now, at this exact price?"**

- **If yes** — the thesis and the setup still hold at today's price — then hold, or even add. The trade is valid on its own merits, independent of what you paid.
- **If no** — you wouldn't buy it here today — then **sell it now**, regardless of whether it shows a profit or a loss. Your cost basis has nothing to say about tomorrow's move.

The beauty of the flip test is that it's symmetric: it fires on winners *and* losers. It tells you to sell a +30% winner whose thesis has played out (you wouldn't chase it here) and to sell a −30% loser whose thesis is broken (you wouldn't touch it here) — while telling you to *hold* a −15% loser whose thesis is intact and simply out of favor. It replaces the question "am I up or down?" (which your curve answers wrongly) with "is this a good position to own right now?" (which is the only question that pays).

For the hardest version of that judgment — separating a thesis that's genuinely broken from one that's just having a bad week — see [thesis broken or just noise, the hardest call you make](/blog/trading/analyst-edge/thesis-broken-or-just-noise-the-hardest-call-you-make).

### Tool 2 — the mechanical exit, pre-committed

The flip test still runs *in the moment*, and in the moment the curve is loud. The stronger fix is to decide the exit **before you have a position at all**, when you're calm and the reference point hasn't been set:

![A five-step pipeline: before entry write the stop and target, place a one-cancels-other bracket order on entry, on a stop hit exit mechanically at minus one R, on a winner trail the stop to let profit extend past plus three R, then review by rule not by outcome.](/imgs/blogs/loss-aversion-and-the-disposition-effect-8.webp)

The protocol, step by step:

1. **Before entry, write the exit.** Define the stop-loss price (where the thesis is wrong) and the target/plan for the winner. Express risk in **R** — where 1R is the dollar amount you lose if the stop hits. Everything downstream is measured in R.
2. **Place the bracket on entry.** Enter the stop-loss *and* the take-profit as a single **OCO** (one-cancels-the-other) order at the moment you open the trade. Now the exit exists in the market, not in your willpower. The most emotional decision has already been made by your calm, earlier self.
3. **Loser hits the stop — exit mechanically.** −1R, no renegotiating, no "give it room," no averaging down. The stop is not a suggestion; it's the price at which your thesis was proven wrong. This is the single hardest step, and it's the whole game.
4. **Winner runs — trail the stop.** Don't cap it at the first target out of fear. Ratchet the stop up behind the price so profit can extend past +3R while your downside stays protected. This is the antidote to the green-position itch: the rule, not your nerves, decides when to get out.
5. **Review by rule, not by outcome.** Judge each trade on *did I follow the plan?*, not *did it make money?* A losing trade taken and exited by the rules is a **good** trade; a winning trade where you held a blown stop and got bailed out by luck is a **bad** trade you must not repeat. Separating process from outcome is how you stop the market's random rewards from reinforcing your worst instincts — the deeper version of that idea is in [process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting).

#### Worked example: the expectancy math that lets you be wrong more than half the time

Say your mechanical rule cuts losers at −1R and lets winners run to an average of +3R. Even with a *losing* hit rate, the expectancy is positive. Take a 40% win rate:

$$E = (0.40 \times +3R) + (0.60 \times -1R) = 1.2R - 0.6R = +0.6R \text{ per trade}$$

You are **wrong 60% of the time** and still make **+0.6R on average per trade**. Over 100 trades at \$100 per R, that's about **+\$6,000**. Now feed the *same* 40% win rate into the disposition trader's rule — winners capped at +0.5R, losers allowed to run to −3R:

$$E = (0.40 \times +0.5R) + (0.60 \times -3R) = 0.2R - 1.8R = -1.6R \text{ per trade}$$

Same forecasting skill, same 40% accuracy — one rule makes \$6,000 over 100 trades, the other loses \$16,000. The entire difference is the *shape* of the exits: cut small, run large versus cut large, run small.

**Intuition:** a good exit rule turns being wrong most of the time into a profit, and the disposition effect turns being right half the time into a loss — which is why professionals obsess over exits, not entries.

### The two systems, side by side

Here is the same trader, in the same market, under the two exit rules. Every row is a choice between letting your entry price decide and letting a pre-committed rule decide:

| | Disposition trader | Mirror trader |
|---|---|---|
| Winners | Sold fast to snatch the gain | Let run / trail the stop |
| Losers | Held and averaged down | Cut mechanically at −1R |
| Avg winner : avg loser | small : large | large : small |
| Reference point | Entry price (the past) | Current price (the future) |
| Decision made | In the moment, by feeling | In advance, by rule |
| Expectancy at 40% win rate | −1.6R per trade | +0.6R per trade |
| Over 100 trades @ \$100/R | −\$16,000 | +\$6,000 |

The rows *are* the article. Each one is the same fork: your entry price, or your rule.

#### Worked example: sizing so you can actually honor the stop

The mechanical exit only works if the −1R loss is small enough that you'll actually accept it. Tie your size to a fixed fraction of capital — say, risk **1%** per trade.

- Account: \$50,000. Risk per trade: 1% = **\$500** (that's your 1R).
- You want to buy a \$100 stock with a stop at \$95 — a \$5 risk per share.
- Position size: \$500 ÷ \$5 = **100 shares** (\$10,000 of stock).

If the stop hits, you lose \$500 — a survivable 1% dent, easy to honor without the value curve screaming. Ten stops in a row (a brutal streak) costs about 10% — painful but recoverable, squarely on the flat part of the recovery curve. The disposition trap springs precisely when a loss is *too big to accept*; correct sizing keeps every loss small enough that accepting it is easy.

**Intuition:** position sizing isn't only risk management — it's what makes the exit rule psychologically *executable* in the first place.

### Wiring it into a routine

Rules only work if they survive contact with a live market. A few reinforcements:

- **Pre-commit in writing.** A stop you typed into the broker as an OCO order beats a stop you're "watching mentally," because the mental one is negotiable and the entered one is not.
- **Size so the stop is survivable.** If a −1R stop would hurt too much to honor, your position is too big. Risk a small, fixed fraction per trade so no single stop triggers the emotional override.
- **Keep a trade journal with the flip-test verdict.** For every open position, note weekly: *would I buy this here today?* The paper trail makes the pattern (and your excuses) visible.
- **Watch for the tells.** When you feel the green-position itch or the red-position squint, that's your cue to run the flip test *out loud*, not to trade on the feeling.

This is the same battle fought across every bias — regret, hope, fear, greed. If you want the wider map of how these forces interlock, see [fear, greed, hope, and regret — the four emotions](/blog/trading/trading-psychology/fear-greed-hope-and-regret-the-four-emotions) and [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders).

## When this matters to you

If you hold any positions with the option to sell them — stocks, funds, crypto, options — the disposition effect is already shaping your returns, whether or not you can feel it. It matters most in the accounts and moments where the pull is strongest:

- **In a taxable brokerage account**, where the tax-smart move (harvest losses, defer gains) is the exact opposite of what the curve wants, so obeying your instincts is doubly expensive.
- **In volatile assets** — small caps, single-name options, crypto — where an un-cut loser can slide from the flat part of the recovery curve to the vertical part fast, and a comeback becomes arithmetically hopeless.
- **In drawdowns and stressed markets**, when willpower is depleted and the red-position squint is most tempting — which is exactly when a *pre-committed* rule earns its keep.
- **In your long-term savings, not just active trading.** The same instinct shows up in a retirement account when you refuse to trim a single stock that has quietly become half your net worth (a winner you won't "disturb"), or when you hold a fund that has lagged its benchmark for years because selling would admit the pick was wrong. The disposition effect doesn't require a day-trading screen; it just requires an entry price and an ego.

It even shows up in reverse as a *warning sign about the fix*: a trader who has just internalized "cut losers fast" can over-correct into cutting *everything* at the first wiggle, churning out of good positions on noise. The flip test guards against that too — it asks whether the thesis still holds, not merely whether the position is red. Cutting a loser whose thesis is intact is just the green-position itch pointed at a red screen. The discipline is not "always sell losers"; it's "let a forward-looking rule, not your entry price, make every call."

You will never make the curve go away; $\lambda \approx 2.25$ is part of the standard human wiring. But you don't have to let it hold the sell button. Nail your exits down before the emotion arrives, judge every open position by "would I buy it here today?", and review yourself on process rather than the market's random applause. Do that, and the most expensive habit in trading becomes just another feeling you notice, name, and route around.

## Sources & further reading

Primary sources behind the headline numbers:

- Daniel Kahneman & Amos Tversky (1979), "Prospect Theory: An Analysis of Decision under Risk," *Econometrica* 47(2): 263–291 — the origin of the value function, reference point, and reflection effect.
- Amos Tversky & Daniel Kahneman (1992), "Advances in Prospect Theory: Cumulative Representation of Uncertainty," *Journal of Risk and Uncertainty* 5: 297–323 — the empirical estimate of the loss-aversion coefficient $\lambda \approx 2.25$ and the 0.88 value-function exponent.
- Hersh Shefrin & Meir Statman (1985), "The Disposition to Sell Winners Too Early and Ride Losers Too Long: Theory and Evidence," *Journal of Finance* 40(3): 777–790 — coined the disposition effect.
- Terrance Odean (1998), "Are Investors Reluctant to Realize Their Losses?" *Journal of Finance* 53(5): 1775–1798 — the 10,000-account study; PGR = 0.148 vs PLR = 0.098, the December tax reversal (PGR 10.8% / PLR 12.8%), and the ~3.4-percentage-point underperformance of held losers versus sold winners.
- Brad Barber & Terrance Odean (2000), "Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors," *Journal of Finance* 55(2): 773–806 — 66,465 households; most-active traders ~11.4% annual vs ~17.9% market, 1991–1996.
- Richard Thaler & Eric Johnson (1990), "Gambling with the House Money and Trying to Break Even: The Effects of Prior Outcomes on Risky Choice," *Management Science* 36(6): 643–660 — the break-even effect and the house-money effect.
- Bankruptcy of Barings Bank (1995): losses of about £827 million (~US\$1.4 billion) in account 88888, the bank's sale to ING for £1, and Nick Leeson's role — see *Encyclopædia Britannica*, "Bankruptcy of Barings Bank," and the contemporaneous reporting.

Further reading and replications:

- Mark Grinblatt & Matti Keloharju (2001), "What Makes Investors Trade?" *Journal of Finance* — the disposition effect across all Finnish investors.
- Andrea Frazzini (2006), "The Disposition Effect and Underreaction to News," *Journal of Finance* — the bias in mutual-fund managers and its effect on prices.
- Daniel Kahneman (2011), *Thinking, Fast and Slow* — the accessible book-length treatment of prospect theory and loss aversion.
- On this blog: [fear, greed, hope, and regret — the four emotions](/blog/trading/trading-psychology/fear-greed-hope-and-regret-the-four-emotions), [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders), [process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting), and [thesis broken or just noise, the hardest call you make](/blog/trading/analyst-edge/thesis-broken-or-just-noise-the-hardest-call-you-make).
