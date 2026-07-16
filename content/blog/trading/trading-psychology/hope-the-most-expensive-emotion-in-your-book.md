---
title: "Hope: The Most Expensive Emotion in Your Book"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A practitioner's deep dive into hope as failed position management: the science of why 'it'll come back' turns a planned -1R loss into a -10R disaster, a clean test that separates conviction from hope, and the drill that kills it before it costs you."
tags: ["hope", "trading-psychology", "position-management", "loss-aversion", "sunk-cost", "disposition-effect", "prospect-theory", "risk-management", "cutting-losses", "stop-loss"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — Hope is not an emotion you feel *about* a trade. It is failed position management wearing an optimistic face, and it is the single most expensive habit in most trading accounts.
>
> - **What hope actually is**: when a trade goes against you, hope whispers "it'll come back" — so you move the stop, remove it, average down, or quietly convert a losing trade into a "long-term investment." Each move turns a planned **-1R** loss into **-8R** or **-10R**.
> - **The mechanism**: hope is the *risk-seeking-in-losses* branch of prospect theory dressed up as optimism. It is [loss aversion](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect), [sunk cost](/blog/trading/trading-psychology/sunk-cost-and-averaging-down-into-a-loser), and the disposition effect all pointing the same way: hold.
> - **The clean test**: *conviction* means your thesis is intact and your pre-set invalidation has **not** been hit. *Hope* means the thesis is broken (or you can't state it) and you're holding on a feeling. If you can't name what would make you wrong, you're hoping.
> - **The number to remember**: Bill Ackman's fund rode Valeant from a ~\$262 peak down for two years — a loss reported between about \$2.8bn and \$4bn, on the order of \$7.7 million for every trading day he held it — while hoping the story would turn.
> - **The drill**: write a hard invalidation *before* entry, set a mechanical stop you never touch, and run one question on every red position — "is my thesis still valid, or am I just hoping?"
>
> This is educational, not financial advice. Round numbers in the *worked examples* are hypothetical so you can do them in your head; every study, statistic, and case number is real and sourced at the end.

You have felt this one. Everyone who has ever held a position has felt this one.

A trade goes against you. It ticks past the level where you told yourself you'd get out, and something in your chest tightens and then, strangely, relaxes. A quiet voice offers you a deal: *don't sell yet. It'll come back. It always comes back.* You don't move your finger to the sell button. Instead you find yourself doing a very specific kind of arithmetic — *if it just gets back to where I bought it, I'll close it and never do this again* — and you slide the stop-loss down a little to give it "room to breathe." The position is now bigger in your mind than it is in your account, and it owns you.

That voice has a name. Traders call it **hope**, and it is the most expensive word in the language of markets. Not because hope is a character flaw — it is one of the finer things about being human — but because in a trading book, hope is almost never an emotion. It is a *decision*, badly made, about a position you should have already closed. It is the moment you stop managing risk and start managing your feelings, and you use the position to do it.

The diagram below is the mental model the rest of this article tours. Read it once now: a red position hits your planned stop, and the path forks. One branch is boring and cheap — you cut, book a small loss, and the trade is closed. The other branch is the hope cascade, and every box in it is a position-management decision that makes the loss larger.

![A layered diagram: a trade that hits its planned -1R stop forks into a disciplined branch (cut now, book -1R, closed) and a hope cascade (move or remove the stop, average down, relabel as a long-term investment) that balloons to -8R to -10R.](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-1.webp)

Look at the shape of it. The disciplined branch is one box and then it stops — the cost of cutting is fixed and known the instant you take it. The hope branch is a *chain*: move the stop, then average down, then relabel the trade, then watch the loss compound. Hope isn't a single bad feeling. It is a sequence of small, reasonable-seeming choices, each one designed to avoid the tiny pain of booking a loss, and each one buying a larger loss later. This post is the tour of that chain — the science underneath it, the exact way it eats your P&L, the clean test that tells hope apart from real conviction, and the mechanical drill that shuts it off.

## Foundations: the building blocks of hope

You don't need any trading background for this section. You need one honest admission: that your mind treats the price you paid for something as sacred, and treats the act of selling it for less as a kind of injury. Everything else is built on that.

### Hope is not a feeling — it's a position-management failure

Start with a definition, because the whole article depends on getting it right.

**Position management** is the set of decisions you make *after* you enter a trade: how much to risk, where to get out if you're wrong, when to add, when to trim, when to walk away. It is, arguably, most of the job. Anyone can click "buy." The money is made and lost in what you do next.

**Hope**, in this precise sense, is what happens when your position management is driven by *how you want the outcome to be* rather than *what the position is actually doing*. A trade is red. Your plan said to be out by now. But instead of executing the plan, you consult your wish — *I want this to come back* — and you let the wish overrule the plan. That's it. That's the whole thing. Hope is the substitution of a feeling for a rule.

Here is the distinction that trips everyone up, so let's nail it early. **Optimism about the future is not hope in this sense.** A trader who buys a stock because they genuinely believe the business will grow is being optimistic, and that's fine — that's a thesis. Hope is different and more specific: it is holding a position *whose reason for existing has already failed*, and holding it anyway, because selling would make the loss real. Optimism is a reason to be *in* a trade. Hope is a reason you invent to avoid *leaving* one.

> Optimism buys the ticket. Hope refuses to get off the train after it's derailed.

### The reference point: why "it'll come back" whispers

Why does hope whisper "it'll come back" specifically? Why not "it'll fall further"? Because your brain has anchored on a single number — the price you paid — and it measures everything from there.

That anchor is called a **reference point**, and it's the villain of this whole story. In 1979, psychologists Daniel Kahneman and Amos Tversky published **prospect theory** in the journal *Econometrica* — the work that later won Kahneman the 2002 Nobel Prize in Economics (Tversky had died in 1996, and Nobels aren't awarded posthumously). Their central insight was that people don't evaluate outcomes by their final wealth, the way tidy economic theory assumed. People evaluate **changes** — gains and losses measured *from a reference point* — and they weigh those changes on a bent, asymmetric scale.

In a trading account, the reference point is almost always your **entry price**. It is arbitrary (the market has no idea what you paid), it is in the past (it tells you nothing about the future), and your brain treats it as the center of the universe. "It'll come back" is your mind pleading to return to the reference point, to erase the change, to get back to zero. The market does not share this goal.

### Loss aversion: the engine that makes red hurt

The second building block is **loss aversion** — the finding, also from prospect theory, that losses feel more intense than equivalent gains. Not slightly more. Roughly *twice* as much. In their 1992 follow-up ("Advances in Prospect Theory," *Journal of Risk and Uncertainty*), Tversky and Kahneman put a number on it: a **loss-aversion coefficient** of $\lambda \approx 2.25$. Losing \$100 hurts about as much as winning \$225 feels good.

You can feel this asymmetry directly. Booking a \$1,000 loss doesn't feel like the mild administrative act it is; it feels like a small amputation. Hope is what your mind offers you to avoid that amputation. As long as you don't sell, the loss is only on paper — it isn't *real* yet, or so the feeling insists — and the amputation is deferred. Every day you hold, you're paying a small premium to postpone a pain that loss aversion has inflated to roughly double its true size.

We cover loss aversion and its market fingerprint — the disposition effect — in full depth in a [dedicated companion post](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect); here we only need the one fact, that red hurts about twice as much as green feels good, because it is the fuel hope runs on.

### R: the unit hope inflates

The last building block is a unit, and it's the most practically important idea in this entire post: **R**.

**R** is your initial risk on a trade — the amount you decided, *before entering*, that you were willing to lose if you were wrong. If you buy a stock at \$50 and place your stop at \$49, then \$1 per share is your R. If you bought 1,000 shares, your R is \$1,000. Everything else gets measured in multiples of R. A trade that makes twice what you risked is a +2R winner. A trade that loses exactly what you planned is a -1R loss. A trade that loses eight times what you planned is a -8R disaster.

Thinking in R is what turns fuzzy dollar feelings into a clean accounting system. And it's precisely the accounting system hope destroys. When you enter, -1R is the whole cost of being wrong — bounded, known, survivable. Hope's entire function is to detach the loss from that number and let it float. The rest of this article is, in a sense, just the story of how a -1R that you refused to take becomes a -10R that takes you.

#### Worked example: one clean R

Let's make R concrete before we watch hope destroy it. Suppose you have a \$100,000 account and a rule that you'll risk 1% per trade. That's \$1,000 of risk — your R.

You buy a stock at \$50. Your analysis says that if it trades below \$47, your reason for being in it is gone. So your stop is \$47 — that's \$3 of risk per share. To risk exactly \$1,000, you buy about 333 shares (\$1,000 ÷ \$3 ≈ 333). Now the arithmetic is clean:

- If the stock hits \$47, you lose 333 × \$3 = \$999 ≈ \$1,000 = **-1R**. You planned this. It's survivable — 1% of your account. You take 99 more identical trades and this one barely registers.
- If the stock runs to \$59, you make 333 × \$9 = \$2,997 ≈ **+3R**. One good trade pays for three bad ones.

**Intuition:** R turns "I'm down some money" into "I'm down exactly one unit of the risk I chose" — and the only reason that unit ever grows past 1 is that you let it.

## 1. The mechanism: hope is prospect theory's risk-seeking branch

Now the deep part. Hope isn't random weakness; it is the predictable output of a specific, measured feature of human decision-making. Once you see the machinery, hope stops feeling like a moral failing and starts looking like what it is: a bias with a known shape, which means it has a known counter.

The bent scale that loss aversion lives on has a second feature, and this is the one that produces hope. The scale isn't just steeper on the loss side — it's a different *shape* on each side. Over gains, it's **concave** (it curves toward flat, so each extra dollar of profit thrills you a little less). Over losses, it's **convex** (it also curves toward flat, so each extra dollar of loss hurts a little less than the last). That convex loss branch has a devastating consequence: it makes people **risk-seeking when they're behind.**

This is the **reflection effect**, and it's the beating heart of hope. Faced with a sure loss versus a gamble that might avoid it, people reliably take the gamble — even when the gamble is mathematically worse. They'll accept a *higher expected loss* for a *chance* at getting back to the reference point. Kahneman and Tversky demonstrated it in the lab in 1979; Richard Thaler and Eric Johnson sharpened it in 1990 ("Gambling with the House Money and Trying to Break Even," *Management Science*), documenting a distinct **break-even effect** — the powerful pull to take long-shot risks specifically when they offer a path back to even.

The figure below is the picture of it. The loss branch of the value curve is convex, so the straight line connecting two loss outcomes — the *chord*, which represents the felt value of a coin-flip between them — sits *above* the curve. In plain English: the gamble feels better than the certain loss, even when the gamble's actual math is worse.

![A value-function chart: the loss branch curves convex through the lower-left quadrant. A dashed chord connecting the origin and a deep loss sits above the curve, showing the gamble's felt value exceeds the certain loss. Callouts note the -$1,000 sure loss versus a 40/60 gamble with -$1,500 expected value.](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-5.webp)

#### Worked example: the break-even gamble

Put numbers on the chart so the trap is undeniable.

You're down \$1,000 on a position — exactly -1R. The thesis has quietly broken; a competitor launched, the guidance was cut, whatever it was that made you buy is gone. You have two choices:

- **Choice A — cut now.** You book a *certain* -\$1,000. Done. The math of this choice: expected outcome = **-\$1,000**.
- **Choice B — hold and hope.** You tell yourself it might bounce. Be honest about the odds on a broken position: say 40% chance it recovers to break-even (\$0 loss) and 60% chance it keeps sliding to -\$2,500. The math: (0.40 × \$0) + (0.60 × -\$2,500) = **-\$1,500** expected.

Choice B has a *worse* expected outcome — you expect to lose \$1,500 instead of \$1,000 — and yet it *feels* like the safer choice, because it's the only one that offers a path back to zero. That feeling is the convex curve doing your thinking for you. The chord (the gamble) sits above the point (the sure loss), so hope votes for B every time.

**Intuition:** hope makes a negative-expected-value gamble feel like the prudent move, because to a mind anchored on its entry price, a *shot* at breaking even is worth more than the extra downside is scary.

### Why it feels good to hold (the neuroscience aside)

There's a physiological layer under this, and it's worth one paragraph because it explains why hope is *pleasant* even as it's ruinous. Hope is a form of **anticipation**, and anticipation of a possible reward is exactly what the brain's dopamine system is built to produce. Work by Wolfram Schultz on reward-prediction signals, and by Brian Knutson and colleagues (2001, *Journal of Neuroscience*) showing that the anticipation of monetary reward recruits the **nucleus accumbens** — a core node of the brain's reward circuitry — established that the dopamine hit comes largely on the *expectation* of reward, not only its delivery. When you hold a losing position and daydream about it roaring back, you are getting a small, real neurochemical reward for the fantasy. The position pays you in dopamine while it costs you in dollars. That is a genuinely bad exchange rate, and it's why hope is so sticky: cutting the trade means cutting off the drip.

### What this costs / when it breaks

The reflection effect is not always wrong. If a position is genuinely mispriced and your thesis is intact, "holding through weakness" is conviction, and it can be right. The mechanism breaks — turns from conviction into hope — the moment the *reason* for the gamble becomes "to get back to my entry price" rather than "because the trade still has positive expected value from here." The curve can't tell the difference. You have to.

## 2. The three engines behind hope

Hope rarely arrives as a single feeling. It's a syndicate — three well-documented biases that each, independently, whisper the same instruction: *hold.* Naming them individually matters, because each one has its own tell and its own counter, and because seeing three separate errors converging on one bad action is what finally makes the action feel as wrong as it is.

![A 3x3 matrix. Rows: loss aversion, sunk cost, disposition effect. Columns: what it whispers, the move it makes, the P&L cost. Loss aversion whispers a loss hurts about twice a gain and refuses to realize the loss; sunk cost says I've lost too much to quit and averages down; disposition says sell winners, hold losers.](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-3.webp)

**Loss aversion** you already met: red hurts about twice as much as green feels good ($\lambda \approx 2.25$), so your mind will pay a real premium to *not realize* the loss. This is the raw fuel — the reason booking -1R feels like an injury rather than a routine cost of business.

**Sunk cost** is the accountant of the group. The **sunk-cost fallacy**, documented by Hal Arkes and Catherine Blumer in their 1985 study "The Psychology of Sunk Cost" (*Organizational Behavior and Human Decision Processes*), is the tendency to continue an endeavor *because of what you've already put into it*, even when quitting is the better forward choice. In their famous experiment, people who had paid full price for a theater season ticket attended more plays — including ones they no longer wanted to see — than people who got the same tickets at a discount, purely to justify the money already spent. In trading, sunk cost sounds like *"I've already lost too much to sell now"* — which is precisely backwards, because the money is already gone whether you hold or sell. Sunk cost is what turns holding into *averaging down*: you add more to a loser specifically to lower your break-even price, throwing good money after bad to validate the first mistake. We dig into that move in full in the [companion post on sunk cost and averaging down](/blog/trading/trading-psychology/sunk-cost-and-averaging-down-into-a-loser).

**The disposition effect** is the market fingerprint of the whole syndicate. Coined by Hersh Shefrin and Meir Statman in 1985 and measured decisively by Terrance Odean in 1998 ("Are Investors Reluctant to Realize Their Losses?", *Journal of Finance*), it is the documented tendency of investors to **sell winners too early and ride losers too long**. Odean studied 10,000 real brokerage accounts and found investors realized their gains at a rate of 14.8% but their losses at just 9.8% — cashing winners about 1.5 times as readily as losers. The kicker: it wasn't even tax-smart or informed. The winners they sold went on to *outperform* the losers they held by about 3.4 percentage points over the following year. Hope is the "ride losers too long" half of that pattern, in action.

Three biases, one instruction. That convergence is why hope is so hard to fight with willpower — you're not resisting one pull, you're resisting three that all point the same way. Willpower loses that fight. A rule wins it, which is what the drill later is for.

## 3. How -1R becomes -8R: hope in your P&L

Theory is nice. Now watch it eat your account, dollar by dollar. This is the section to internalize, because it converts an abstract bias into a specific, visible number on your statement.

![A P&L chart with time on the x-axis and R-multiples on the y-axis. Two paths leave the entry at 0R and both reach the planned -1R stop. The disciplined path goes flat at -1R; the hope path keeps sliding through -3R and -5R to a terminal -8R, annotated with 'move the stop, average down' and 'no line left to defend'.](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-2.webp)

Both paths in the chart start identically. You enter, the trade goes against you, and it reaches your planned stop at -1R. This is the fork. The disciplined path takes the -1R and flattens — that horizontal line is the entire rest of the story, a small fixed cost. The hope path refuses the -1R and keeps falling. Let's walk the exact moves that produce that descent.

#### Worked example: moving the stop, one leg at a time

Back to the clean setup. \$100,000 account, R = \$1,000. You buy 1,000 shares at \$50 with a stop at \$49 (\$1 of risk per share, so R = \$1,000).

**Leg 1 — the stop is hit.** The stock trades to \$49. Your plan says sell: take the -\$1,000, the clean -1R. But it's been a good week, you don't want to end it red, and the voice offers its deal. You cancel the stop and tell yourself you'll "watch it closely." *Loss so far if you'd cut: -1R. Loss now: still -1R on paper, but the line is gone.*

**Leg 2 — averaging down.** The stock slides to \$47. Now you're down \$3,000 (-3R), and sunk cost speaks up: *if you buy more here, your average cost drops and it only needs a small bounce to get you out.* You buy another 1,000 shares at \$47. You now own 2,000 shares at an average of \$48.50, and you've doubled your size at the worst possible time. *Total risk in the trade is now roughly 2R per dollar of further decline — you've made the position twice as sensitive to being wrong.*

**Leg 3 — no line left.** The stock drifts to \$44. On your 2,000 shares, that's a loss of 2,000 × \$4.50 = **\$9,000**. Against your original \$1,000 R, you are now down **-9R**. There is no stop, no plan, and no reason left — just a wish. The position that was supposed to cost you 1% of your account has cost you 9%. To simply get back to break-even on the doubled position, the stock now has to climb 10% from \$44 to \$48.50 — and to undo the whole loss and reach your original \$50 entry, it needs +13.6%, which a stock that just fell 12% has given you every reason to doubt.

**Intuition:** every leg down was a *choice* to avoid booking a slightly smaller loss, and each choice bought a much larger one — a -1R you could shrug off became a -9R that reshapes your month.

### The five moves of hope

That worked example contained the whole vocabulary. Hope only ever makes five moves, and naming them is the first step to catching yourself mid-move:

1. **Move the stop.** You slide the exit lower "to give it room." A stop that moves against you is not a stop; it's a suggestion you're ignoring.
2. **Remove the stop.** You cancel it entirely, converting a bounded risk into an unbounded one. This is the single most expensive click in trading.
3. **Average down.** You add size to a loser to lower your break-even. This is sunk cost in its purest form — buying more of your mistake to justify the first purchase.
4. **Convert to an "investment."** You silently change the trade's timeframe. What you bought as a two-week swing becomes, the moment it's underwater, a "long-term hold." The timeframe didn't change because your analysis deepened; it changed to hide the loss.
5. **Avert your eyes.** You stop looking at the position, hide it from your P&L glance, and check the message boards for reasons instead of the chart for facts. The loss you don't look at feels smaller. It isn't.

Every one of these is a position-management action, and every one is driven by the wish to get back to the reference point rather than any assessment of forward expected value. That's the signature of hope, and it's what makes it detectable in real time.

### What this costs / when it breaks

The cost is convex — it accelerates. -1R to -2R feels almost free (you were already red). -2R to -4R feels like "I've come this far." -4R to -8R is where accounts get maimed, because by then the position is so large relative to plan that a small further move is a big further loss. The move that breaks you is almost never the entry. It's the fourth or fifth leg of hope, taken calmly, by a version of you that has stopped keeping score.

## 4. The recovery math: why deep losses are a cliff

Here is the piece of arithmetic that should be tattooed inside every trader's eyelids, because it is the reason "it'll come back" is usually a lie. Losses and the gains required to undo them are **not symmetric.** A loss of X% does not need a gain of X% to recover. It needs *more* — and the gap explodes as the loss deepens.

![A convex curve chart with loss taken (%) on the x-axis and gain needed to break even (%) on the y-axis. The curve rises far faster than a dashed linear reference line: -25% needs +33%, -50% needs +100%, -80% needs +400%. An amber callout notes people wrongly assume a -50% loss needs only +50% back.](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-7.webp)

The math is simple and merciless. If you lose a fraction $L$ of your capital, the gain $G$ you need to get back to even is:

$$G = \frac{L}{1 - L}$$

The denominator is the killer. As $L$ climbs toward 1 (a total loss), the required gain runs off to infinity.

#### Worked example: the asymmetry of drawdown recovery

Run the numbers you'd actually face, starting from \$10,000:

- **Down 10%** → \$9,000. You need +11.1% to get back to \$10,000. Annoying, not dangerous.
- **Down 25%** → \$7,500. You need +33.3%. Now you're working noticeably harder to undo it than it took to cause it.
- **Down 50%** → \$5,000. You need **+100%** — the position has to *double* just to get you back to where you started.
- **Down 75%** → \$2,500. You need **+300%**.
- **Down 80%** → \$2,000. You need **+400%**. The stock has to *quintuple*. On a broken company, that's not a recovery plan; it's a prayer.

Notice the shape. The gain-needed line doesn't rise in step with the loss — it curves upward, away from the naive "a 50% loss needs 50% back" intuition (the dashed line in the figure), and the gap between what you think you need and what you actually need widens with every extra leg of hope.

**Intuition:** letting a loss deepen doesn't just cost you the extra dollars — it moves the finish line for recovery exponentially farther away, which is exactly why the small loss taken now beats the "small bounce" you're waiting for.

### What this costs / when it breaks

This is why professionals obsess over *not* letting losses compound past a threshold. A book of -1R and -2R losses is fully recoverable with one good +3R or +4R winner. A single -50% position needs to double before you've made a cent, and while it's trying, it's dead weight — which brings us to the cost hope never puts on the invoice.

## 5. The opportunity cost of dead capital

Even if the "it'll come back" position eventually *does* come back — and most fallen ones don't — hope charges you a second bill that never appears on any statement: the return that money could have earned somewhere else while it sat frozen in your broken trade. Economists call it **opportunity cost**, and in trading it is enormous and invisible.

![A timeline of a baghold: buy $20,000 with thesis intact, down 30% to $14,000 as the thesis breaks on bad earnings, down 50% to $10,000 while holding for the recovery, down 60% to $8,000 with capital frozen and three setups missed, and after 24 months still around $8,000 needing +150% just to break even.](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-6.webp)

Capital is the raw material of trading. A dollar tied up in a position you refuse to close is a dollar that cannot take the next good setup. The loss you're avoiding on paper is real; the gains you're forgoing are just as real, only they're silent because they never show up as a line item. You don't see the trade you didn't take.

#### Worked example: the cost of a two-year baghold

Suppose you put \$20,000 into a position. Three months in, the thesis breaks on a bad earnings report and it's down 30% to \$14,000. You should cut — the reason you bought is gone. Instead you "hold for the recovery." Follow the money over two years:

- It keeps sliding — down 50% to \$10,000, then down 60% to \$8,000 — and settles there, a classic "baghold."
- **The direct loss** is \$12,000. Painful, but that's only half the bill.
- **The opportunity cost.** For 24 months, \$8,000 of live capital sat frozen. Suppose your trading actually compounds at a realistic 12% a year. Redeployed, that \$8,000 becomes \$8,000 × 1.12² ≈ \$10,035 — you forgo about **\$2,035** in gains you'd otherwise have made. Worse, the *full* \$20,000 was impaired for the first stretch, and the three good setups you couldn't fund because your capital was trapped are gone forever.
- **And the position still owes you a miracle:** from \$8,000 back to your original \$20,000 is +150% — the recovery math from the last section, in the wild.

**Intuition:** dead capital costs you twice — the loss you're refusing to book *and* the compounding you'll never get on money you left buried in a broken trade.

### What this costs / when it breaks

The opportunity cost is why "I'll just hold it, it's not like I'm losing more if I don't sell" is one of the most expensive sentences in trading. You *are* losing more. You're losing it in a currency — foregone return — that your P&L statement is not designed to show you, which is exactly why hope gets away with it.

## 6. The clean test: conviction versus hope

Everything so far has been diagnosis. Here is the tool. If hope and conviction both *feel* like "holding through weakness," you need a test that doesn't rely on feeling — because the feeling is compromised by design. The test is four questions, and it fits on an index card.

![A two-column comparison. Left, red, HOPE (cut it): thesis broken or you can't state it, invalidation already blown through, holding only to avoid booking the loss, wouldn't buy it here. Right, green, CONVICTION (keep): thesis intact and stateable in one sentence, pre-set invalidation not hit, holding on the plan you wrote before entry, would add here from scratch.](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-4.webp)

Run every red position through these four questions. It is *conviction* only if all four land on the green side. If even one lands red, you're hoping, and the answer is to cut.

1. **Can you state the thesis, out loud, in one sentence — and is it still true?** Conviction has a live, specific thesis ("this trades cheap to its cash flows and the catalyst is the Q3 print"). Hope has either no thesis you can articulate anymore, or one that has already been falsified by events. If the reason you bought is gone, the position is just a lottery ticket you're too attached to throw away.

2. **Has your pre-set invalidation been hit?** Conviction defined, *before entry*, the specific fact or price that would prove it wrong — and that line has not yet been crossed. Hope either never set a line, or set one and blew straight through it while inventing reasons the line "doesn't count this time."

3. **Are you holding on the plan you wrote, or on a feeling you're having?** Conviction is executing a decision made in a calm moment before you had money on the line. Hope is improvising, in real time, under the influence of loss aversion, to avoid a pain.

4. **Would you open this exact position — this size, this price — right now, from scratch, with fresh capital?** This is the sharpest of the four. If a flat trader would buy it here, you have conviction. If the *only* reason you hold is that you already own it and selling would book a loss, that is the pure signature of hope. Ownership is not a thesis.

The beauty of the four questions is that they route entirely around the compromised feeling. They don't ask "do you believe in it?" — belief is exactly what hope counterfeits. They ask about your thesis, your pre-set line, your written plan, and what a neutral trader would do. Those are checkable. Feelings are not.

> Conviction lives in the plan you wrote before you had money on the line. Hope lives in the story you tell after.

## 7. The drill: a protocol for killing hope before it costs you

A bias with a known shape has a known counter, and hope's counter is *mechanization*. You cannot out-willpower three convergent biases pumping dopamine into your decision. You *can* pre-commit to a rule while you're calm, and then execute the rule when you're not. That's the entire game.

![A decision flow: before entry, write the hard invalidation and set a mechanical stop; when the position goes red, run the check on every red line; ask 'would I buy this, this size, right now, and is the thesis intact?'; yes routes to conviction (hold to the stop, never widen), no routes to hope (cut now, book -1R, redeploy the capital).](/imgs/blogs/hope-the-most-expensive-emotion-in-your-book-8.webp)

Here is the protocol, in the order you actually run it.

**Before entry — write the invalidation and set the stop.** *Before* you have a dollar at risk, while your judgment is uncompromised, write down the single fact or price that would prove this trade wrong. Not "if it goes down a lot" — a specific, mechanical line: "stop at \$47" or "out if the Q3 revenue misses." Then place that stop as a *mechanical order with your broker*, not a "mental stop." A mental stop is just hope with extra steps; it exists precisely so you can renegotiate it. A resting order in the market can't be talked out of by the voice.

**The one rule about the stop: it only ever moves in your favor.** A stop can trail *up* to lock in profit on a winner. It may *never* widen on a loser. The instant you find yourself moving a stop lower "to give it room," stop your hands — that specific action is the physical form of hope, and catching it is 80% of the battle.

**On every red position — run the one question.** For anything showing red, ask: *"Is my thesis still valid and my invalidation not yet hit — or am I just hoping?"* If you can't crisply state the still-true thesis, or the invalidation is already hit, you have your answer. Cut. Then run the re-buy test as a tiebreaker: *would I buy this, right here, this size, right now?* If no, the only thing keeping you in is the reference point, and that's not a reason.

**Ban the relabel.** A trade's timeframe is set at entry and is not allowed to change *because it went against you*. You may not convert a losing swing trade into a "long-term investment." If you genuinely want a long-term position, that's a brand-new decision, made from flat, sized from scratch, with its own invalidation — not a costume you put on a loser to avoid the sell button.

**Write the conviction-vs-hope rule down, once, and keep it visible.** A single index card taped where you trade: *Conviction = thesis intact + invalidation unhit + would re-buy here. Hope = any of those is false. Hope gets cut.* You are outsourcing the decision to a calmer version of yourself. That's not weakness; it's the whole professional edge.

### What it looks like at the screen

The protocol only works if you can catch hope *while it's happening*, and hope has a very consistent set of tells. Here is what it actually looks like at the screen, in your own behavior, in real time — learn to feel these the way you'd feel a hand on your shoulder:

- **You stop reading the chart and start reading the story.** You minimize the price window and open the message boards, the news, the bull-case thread — hunting for a reason it'll bounce instead of a fact about what it's doing. Facts have become the enemy.
- **You start doing recovery math instead of forward math.** Your inner monologue shifts from "what's the expected value from here?" to "if it just gets back to \$50, I'm out." You've begun computing the price that would erase the change, which means the reference point is now driving.
- **You feel a jolt of relief when it ticks up 2% and dread when it's merely flat.** Your emotional state is now pegged to the position's every wiggle, which is what happens when a trade has stopped being a position and become a hostage situation with you as the hostage.
- **You reach for the stop-loss order to move it — "just this once."** The physical act. Your cursor drifts to the stop and nudges it lower. This is the single clearest tell. There is no legitimate reason to widen a stop on a loser, ever.
- **You've stopped checking the stop and started checking the story.** Early in a healthy trade you know exactly where your exit is. Deep in a hope trade you can't remember if you even have one, but you can recite the bull case verbatim.
- **You hide the position from your own P&L glance.** You scroll past it, you don't add it up, you "don't want to look today." The loss you don't look at feels smaller — which is your mind managing your feelings with your capital, the exact definition of hope.

When you notice two or three of these at once, you are not analyzing a position. You are hoping. Run the four questions, and if any lands red, cut — before the next leg down makes the decision for you at a worse price.

## Common misconceptions

**"Hope is just optimism, and optimism is good."** Optimism is a *reason to enter or stay in* a trade whose thesis is intact — that's conviction, and it's fine. Hope is holding a position *whose thesis has already failed*, on a wish to return to your entry price. The difference isn't the emotion's flavor; it's whether the reason for the trade still exists. One is a thesis; the other is the risk-seeking branch of prospect theory wearing the thesis's clothes.

**"Averaging down is what smart value investors do."** Sometimes — but the smart version and the hope version look identical from the outside and are opposite on the inside. Conviction-averaging is planned *before entry* ("I'll scale in at \$50, \$45, \$40 because the value only improves as it falls, and my invalidation is \$35"), and the thesis is intact the whole way. Hope-averaging is unplanned, triggered by the loss itself, aimed at lowering your break-even, on a thesis that's already broken. The test is question four: would you buy this size, here, from scratch? If yes, it's an add. If you're only buying more because you already own it, it's hope.

**"It's not a real loss until I sell."** This is the most expensive myth in the list, and it's simply false. The moment the price drops, your capital is impaired — the account is worth less whether or not you click sell. "Unrealized" is an accounting word, not a reprieve. Worse, refusing to realize it keeps the capital dead (the opportunity cost) and lets the recovery math run away from you. The loss is real the instant it happens; selling just stops it from getting realer.

**"It always comes back eventually."** This is survivorship bias in one sentence. You remember the positions that came back because you're still holding them to tell the story; you conveniently forget the ones that didn't. A broad *index* has historically recovered given enough time, because it's constantly replacing its failing members. An *individual* stock has no such guarantee — plenty go to zero, get delisted, or spend a decade underwater. Hope quietly swaps "the market comes back" for "*my* stock comes back," and those are not the same claim.

**"Cutting the loss means I have no conviction — real traders hold."** Backwards. Cutting at a *pre-set invalidation* is the highest expression of conviction — in your process. Conviction doesn't mean marrying a position; it means trusting the plan you wrote when you were thinking clearly enough to define what "wrong" looks like. The trader who can't ever cut doesn't have more conviction; they have no invalidation, which means they never had a real thesis, only a hope.

**"Hope only bites amateurs."** As the next section shows, hope scales *with* ego and capital. The larger and more public your position — the more your identity is tied to being right — the harder it is to book the loss, and the bigger the loss you'll ride. Some of the most expensive hope trades in history were run by professionals with every resource and every reason to know better.

## How it shows up in real markets

Biases are easy to nod along with in the abstract. Here they are with names, dates, and numbers — four episodes where hope did exactly what this article describes, at scale.

### 1. Bill Ackman and Valeant: conviction that curdled into hope

Bill Ackman is one of the most successful activist investors alive, which is precisely why his Valeant Pharmaceuticals trade is the definitive modern hope story: it shows that skill and capital don't immunize you — they raise the stakes.

Valeant's stock had rocketed to a peak near **\$262 in August 2015** on a strategy of buying drugs and hiking their prices. Then the model came apart: allegations around its relationship with the specialty pharmacy Philidor, and intense scrutiny of its pricing, sent the stock into a collapse that erased more than 60% of its value in months (as reported by *Fortune* and *CBC*). Ackman's Pershing Square had started buying around **\$161 a share in early 2015**, ultimately committing more than **\$4.6 billion** to the stock and options (per *Forbes*). As the thesis broke, Ackman didn't cut — he *doubled down* on his conviction, defended the company publicly, and even took a board seat, tying his name and reputation ever tighter to the position.

He finally sold the last of it in **March 2017 near \$11 a share.** The loss was variously reported between about **\$2.8 billion** (*Bloomberg*) and roughly **\$4 billion** (*Forbes*, *Fortune*) — on the order of **\$7.7 million for every trading day** he held it (*Forbes*). Ackman himself called it a "huge mistake." The mechanism is the one from this post exactly: a real thesis that broke, a refusal to book the loss that hardened into public commitment, and a position ridden from \$262-adjacent all the way to \$11 in the hope it would turn. The higher your conviction and profile, the more it costs when conviction becomes hope.

### 2. Nick Leeson and Barings: hope that broke a 233-year-old bank

If Valeant shows hope at the level of a star investor, Barings shows it at the level of catastrophe. In 1995, a 28-year-old derivatives trader named Nick Leeson single-handedly destroyed **Barings Bank — a 233-year-old institution** that had helped finance the Louisiana Purchase and banked the Queen.

The engine was pure hope, mechanized. Leeson took losing positions on Nikkei 225 futures and, rather than book them, hid them in a secret error account numbered **88888** and **doubled his bets**, wagering that the market would recover. Then the Kobe earthquake struck in January 1995, the Nikkei crashed, and his hope-driven averaging-down turned a large loss into a fatal one. By the time it was discovered, the losses totaled roughly **£827 million** — more than twice the bank's available capital (per *Britannica* and the CNBC retrospective). Barings collapsed in late February 1995 and was sold to ING for one pound; Leeson was sentenced to six and a half years in Singapore's Changi Prison.

Every move from Section 3 is here at industrial scale: refuse to book the loss, hide it, average down to lower the break-even, and hold on the wish that the market bounces. The only differences from a retail hope trade were the number of zeros and the fact that the "position" was large enough to take a centuries-old bank down with it.

### 3. The hold-to-zero bagholder: Enron

The most common hope trade of all isn't a billionaire's or a rogue trader's — it's the ordinary investor who holds a fallen former high-flyer all the way to zero. Enron is the archetype.

Enron stock hit an all-time high of **\$90.75 on August 23, 2000**, with a market cap north of \$70 billion — a Wall Street darling. As the accounting fraud unraveled through 2001, the stock fell to **\$0.26 by November 30, 2001**, and the company filed for Chapter 11 on **December 2, 2001** (per *History* and *KHOU*). The tragedy of hope is sharpest in what happened to employees: roughly **62% of the 15,000-employee 401(k) plan** was invested in Enron stock, and as it fell, many held on — some because the plan's assets were frozen, but many out of the exact conviction-curdled-into-hope this post describes: *this is a great company, it'll come back.* It didn't. Billions in retirement savings went to essentially zero.

The lesson isn't "Enron was a fraud, how could they know." Most bagholds aren't frauds — they're good stories that stopped being true. The lesson is that "it'll come back" applied to a *single company* is a bet with no floor, and hope is what keeps a hand on a falling knife all the way to the ground.

### 4. Jesse Livermore's warning

The oldest and most quoted articulation of this whole post comes from a 1923 book, *Reminiscences of a Stock Operator* by Edwin Lefèvre — a thinly fictionalized biography of the legendary speculator Jesse Livermore. Nearly a century before the neuroscience, Livermore had the mechanism dead to rights:

> "The speculator's chief enemies are always boring from within. It is inseparable from human nature to hope and to fear... Instead of hoping he must fear; instead of fearing he must hope. He must fear that his loss may develop into a much bigger loss, and hope that his profit may become a big profit."

Read it twice. The amateur *hopes* when he should *fear* — hopes a loss will reverse instead of fearing it will grow — and *fears* when he should *hope*, snatching a small profit out of fear it'll vanish instead of letting a winner run. That is the disposition effect, stated as a moral law in 1923 and confirmed with 10,000 brokerage accounts in 1998. Hope pointed at a loss is the single most reliable way traders have destroyed themselves for as long as there have been markets.

## When this matters to you

You do not need to be running \$4 billion or a rogue futures book for this to touch your money. Hope shows up the first time you own anything that can fall — a stock, a fund, a coin, a house you're "sure" will bounce back. The tightening in your chest when a position goes past the line you set, and the quiet voice offering to renegotiate the exit, are the same in a \$500 account as in a \$500 million one. The only question is whether you've decided, in advance and in writing, what you'll do when it speaks.

The honest risk disclosure is this: every technique here reduces the size of your losses; none of them make you right more often. Cutting at your invalidation means you *will* sometimes cut a position that later recovers — that's the cost of the insurance, and it's real. What the discipline buys you is the thing that actually determines whether you survive: a book of small, bounded, survivable losses instead of the occasional account-ending one. You cannot control whether any single trade works. You can control, completely, whether a losing one is allowed to become a catastrophic one. Hope is the mechanism that strips you of that control by dressing a management failure up as optimism. The drill is how you take it back.

If you internalize one sentence, make it this: **the small loss taken now, on plan, is not the painful outcome — it is the thing that prevents the painful outcome.** Everything hope tells you is an argument against that sentence. Everything the data says is for it.

For your next step, read the three companion pieces that this post leans on — the ones that dissect the individual biases hope is built from — and then go find the reddest position in your own book and run the four questions on it, honestly, right now.

## Sources & further reading

**The science (primary sources behind the headline claims):**

- Kahneman, D. & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk." *Econometrica*, 47(2). The origin of the reference point, loss aversion, and the reflection effect (risk-seeking in losses).
- Tversky, A. & Kahneman, A. (1992). "Advances in Prospect Theory: Cumulative Representation of Uncertainty." *Journal of Risk and Uncertainty*, 5. Source of the loss-aversion coefficient $\lambda \approx 2.25$.
- Shefrin, H. & Statman, M. (1985). "The Disposition to Sell Winners Too Early and Ride Losers Too Long." *Journal of Finance*, 40(3). Coined the disposition effect.
- Odean, T. (1998). "Are Investors Reluctant to Realize Their Losses?" *Journal of Finance*, 53(5). 10,000 accounts; gains realized at 14.8% vs losses at 9.8%; sold winners outperformed held losers by ~3.4 percentage points over the next year.
- Arkes, H. & Blumer, C. (1985). "The Psychology of Sunk Cost." *Organizational Behavior and Human Decision Processes*, 35(1). The theater-ticket experiment.
- Thaler, R. & Johnson, E. (1990). "Gambling with the House Money and Trying to Break Even." *Management Science*, 36(6). The break-even effect.
- Knutson, B. et al. (2001). "Anticipation of Increasing Monetary Reward Selectively Recruits Nucleus Accumbens." *Journal of Neuroscience*, 21(16); and Schultz, W. (1998) on reward-prediction signals — the neuroscience of anticipation ("hope" as dopamine on expectation).

**The cases (as-of dates and publishers):**

- Bill Ackman / Valeant: peak near \$262 (August 2015); Pershing Square entry ~\$161 (early 2015), >\$4.6bn committed, sold near \$11 (March 2017); loss reported ~\$2.8bn (*Bloomberg*, 2017-03-13) to ~\$4bn (*Forbes*, *Fortune*, March 2017); ~\$7.7m/day (*Forbes*).
- Nick Leeson / Barings Bank: ~£827m loss, account 88888, Nikkei futures, collapsed late February 1995, 233-year-old bank (*Britannica*; *CNBC*, 2020-02-26 retrospective).
- Enron: peak \$90.75 (2000-08-23), \$0.26 (2001-11-30), Chapter 11 (2001-12-02), ~62% of the 15,000-employee 401(k) in Enron stock (*History.com*; *KHOU*, 20-year retrospective).
- Lefèvre, E. (1923). *Reminiscences of a Stock Operator*. The Livermore fear/hope passage.

**Companion posts on this blog:**

- [Loss Aversion and the Disposition Effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect) — the full value-function math and the "would I buy it here?" fix.
- [Sunk Cost and Averaging Down Into a Loser](/blog/trading/trading-psychology/sunk-cost-and-averaging-down-into-a-loser) — the averaging-down move dissected.
- [Thesis Broken or Just Noise: The Hardest Call You Make](/blog/trading/analyst-edge/thesis-broken-or-just-noise-the-hardest-call-you-make) — how to tell a broken thesis from ordinary volatility, which is the input to the clean test above.
