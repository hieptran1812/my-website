---
title: "Mental Accounting and the House Money Effect: Why Not All Your Dollars Are Equal"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A trading-desk deep dive into mental accounting and the house money effect: why your brain sorts one pool of money into buckets, why 'the house's money' gets gambled, and the one-portfolio drill that makes every dollar equal again."
tags: ["mental-accounting", "house-money-effect", "break-even-effect", "behavioral-finance", "trading-psychology", "risk-management", "position-sizing", "fungibility"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Money is fungible: a dollar is a dollar no matter where it came from. Your brain refuses to believe that, and the refusal is expensive.
>
> - **Mental accounting** is the habit of sorting money into separate mental buckets — "capital", "this year's profit", "the trading account", "house money" — and handing each bucket a different risk appetite (Thaler, 1985).
> - **The house money effect**: after a gain you take bigger risks with "the house's money" as if it were not really yours. In one experiment, 70% took a coin-flip after a \$30 win versus 40% after a \$30 loss — the *same* bet (Thaler & Johnson, 1990).
> - **The break-even effect** is its mirror: after a loss you chase the bet that gets you back to zero, because a return to your reference point feels disproportionately good. It is how a small hole becomes a career-ending one.
> - Segregated buckets do not just mis-frame money — they **mis-size the book**: three accounts each "risking only 2–3%" can put 8%+ of your net worth at risk at once, past any cap you thought you had.
> - **The fix is one number**: size every position off your *total net liquidity* and *current open risk*, never off which bucket the money came from. Re-label all "house money" as *your money* at the start of every period.

You had a green morning. Three trades, all winners, and by lunch you are up \$4,000 on the day. On the next setup you hesitate over size — and then a quiet voice says: *I'm playing with the market's money now. If this one loses, I'm still green.* So you click for double your normal size. No stop, because a stop would be admitting the trade could take back "their" money.

That voice has a name, a Nobel Prize behind it, and a body count. It is **mental accounting**, and the specific trap it just sprang on you is the **house money effect**. The dollars you won this morning are not the casino's chips. They are your net worth — every bit as real, as spendable, as loseable as the capital you deposited. But your brain filed them in a different drawer, and it lets you gamble the contents of that drawer in a way you would never gamble your "real" money.

The diagram below is the mental model the whole article tours. One pool of money on the left — your total net liquidity, every dollar identical. Then a wall your mind builds. Then a set of buckets on the right, each with its own risk dial cranked to a different setting. The entire post is just following those dials into your P&L, and then prying them off.

![One pool of fungible money on the left, a mental partition in the middle, and four buckets on the right, each handed a different per-trade risk level.](/imgs/blogs/mental-accounting-and-the-house-money-effect-1.webp)

Look at the buckets. The dollar labeled "capital" gets risked at 2% per trade, carefully. The identical dollar labeled "this year's profit" gets 10%. Relabel it "house money" after a win and the dial jumps to 25%. Same money, same account, same you — but the *story* about where the dollar came from set the risk. That is the whole disease. The rest of this piece is the diagnosis and the cure.

This is educational, not financial advice. Every number in a *worked example* is round and hypothetical so you can do it in your head; every number attributed to a *study, market, or case* is real and sourced at the end.

## Foundations: the building blocks of mental accounting

Before the trading, the psychology. You do not need any finance background for this section — just a willingness to watch how your own mind prices one dollar against another.

### What "mental accounting" actually means

**Mental accounting** is the set of mental operations people use to organize, evaluate, and keep track of their money. The economist **Richard Thaler** named and formalized it in a 1985 paper, *Mental Accounting and Consumer Choice*, and expanded it in a 1999 review, *Mental Accounting Matters* — a body of work the 2017 Nobel citation singled out. His one-line definition: mental accounting is "the set of cognitive operations used by individuals and households to organize, evaluate, and keep track of financial activities."

In plain English: your brain runs a set of invisible envelopes. Rent money in one envelope, fun money in another, "found" money in a third. You would never raid the rent envelope to hit the blackjack table, but the \$100 you found in an old coat? That is practically begging to be spent. Same \$100. Different envelope. Different rules.

That envelope system is genuinely useful for a household budget — it is how people avoid blowing the mortgage payment on a weekend. But in a trading account it is poison, because markets do not care which envelope a dollar sits in. Risk is measured on the *whole* account, and a mind that sizes bets by envelope will size them wrong.

### Fungibility: the principle mental accounting violates

Here is the economic law your brain is breaking. Money is **fungible** — one unit is perfectly interchangeable with any other unit of the same kind. A dollar you earned, a dollar you inherited, a dollar you won on a lucky call: all identical, all worth exactly one dollar, all spendable and loseable on identical terms. There is no such thing as a "profit dollar" that is worth less than a "capital dollar." A dollar is a dollar.

> Fungibility is not a suggestion. It is arithmetic. The market will take your "house money" and your "real money" at exactly the same exchange rate: one for one.

Mental accounting is, at its core, the systematic violation of fungibility. It treats interchangeable dollars as if they were *not* interchangeable — as if the label ("profit", "capital", "the fun account") changed what the dollar is worth or how much you should be willing to risk it. It never does. This is the single sentence to carry out of the Foundations section: **every distortion in this post is a fungibility violation wearing a different costume.**

### Three terms you need, defined once

- **Reference point** — the level you measure gains and losses *from*. For a trader it is almost always your entry price, or your account balance at the start of some period (the day, the month, the year). It is arbitrary — the market has never heard of it — but your brain treats it as the center of the universe. Everything above it is a "gain"; everything below it is a "loss." (This is the villain of the sibling post on [loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect), and it is the villain here too.)
- **House money** — gains you have made but mentally file as belonging to "the house" (the market) rather than to you. The phrase comes straight from the casino: a gambler up early feels he is now betting with the *casino's* money, not his own, and so bets bigger.
- **Realized vs unrealized** — a gain is *realized* when you close the position and the profit is cash; it is *unrealized* ("paper") while the position is still open. The dollars are identical either way — an unrealized \$5,000 is \$5,000 of your net worth right now — but the mind files them in different envelopes and treats them very differently.

### The core demonstration: one sum, two labels

The fastest way to feel the bias is to watch the same money produce opposite decisions depending only on its label. The figure below is a real decision most traders make several times a year.

![The same \$10,000 labeled 'house money' drives one speculative all-in bet with no stop; labeled 'my capital' it is sized to 2% of the book with a hard stop and a \$1,200 max loss.](/imgs/blogs/mental-accounting-and-the-house-money-effect-2.webp)

On the left, the \$10,000 is labeled "house money" — this year's profit, the market's chips. So it goes into one speculative name, no stop, on the logic that losing it is "just giving back profit." One hundred percent of it is at risk. On the right, the identical \$10,000 is labeled "my capital." Now it is sized to risk 2% of the whole book, with a hard stop set before entry, as one of many positions. Max loss: \$1,200. Same money. The risk taken on it differs by roughly eightfold.

#### Worked example: the same \$10,000, two labels

Suppose you finish the first half of the year up \$10,000 on a \$50,000 account. A new setup appears. Watch how the label decides the size.

- **Framed as "house money."** You think: *this is profit, not my original stake. If the trade goes to zero, I'm just back to where I started — no real harm.* So you put the full \$10,000 into one volatile small-cap, no stop. If it drops 40%, you lose \$4,000. If the company gaps down on bad news, you can lose \$8,000 overnight. Your loss on the *idea* is uncapped because you never defined it.
- **Framed as "my capital."** You think: *this is \$10,000 of my money; my rule is 2% of the account at risk per trade.* Two percent of your \$60,000 account (the original \$50,000 plus the \$10,000 you made — it is all one account now) is \$1,200. You buy a position and place a stop such that if it hits, you lose \$1,200 and not a cent more. Same setup, same conviction, same money.

The gap between a capped \$1,200 loss and an uncapped \$4,000–\$8,000 loss is not a difference in the *trade*. It is a difference in the *story you told about the dollars* before you sized it.

**Intuition:** the money never changed — only its label did, and the label is what pulled the trigger on your position size.

## The science: how prior outcomes rewire your risk appetite

Everything above is intuition. Now the evidence, because this is not folk wisdom — it is one of the most replicated findings in behavioral economics, and the numbers are precise.

In 1990, Richard Thaler and Eric Johnson ran a set of real-money experiments and published them in *Management Science* under a title that says it all: *Gambling with the House Money and Trying to Break Even: The Effects of Prior Outcomes on Risky Choice*. The finding: a rational decision-maker is supposed to weigh only the incremental outcomes of the choice in front of them. Real people do not. They are powerfully swayed by what *just happened* — the prior gain or loss — even though that outcome is already in the bag and cannot be changed.

![A horizontal bar chart: 70% took the risky coin-flip after a \$30 gain versus 40% after an identical \$30 loss, with the gamble's expected value of zero shown in a side panel.](/imgs/blogs/mental-accounting-and-the-house-money-effect-3.webp)

The cleanest version works like the bars above. Tell people they have just won \$30, then offer them a 50/50 coin-flip to win or lose \$9 — a bet whose expected value is exactly zero. Roughly **70%** take the gamble. But when the coin-flip is offered without the prior win — you simply choose between a sure \$30 and the gamble — the majority keep the sure thing and refuse the bet. The prior gain, the "house money," is what turns cautious people into gamblers. And after a prior *loss* of \$30, only about **40%** take that same symmetric coin-flip. The identical bet, moved 30 percentage points, by nothing but the outcome that came before it.

#### Worked example: the coin-flip that pays the same either way

Let us do the arithmetic the subjects would not, so you can see exactly how irrational the swing is.

- The gamble: 50% chance to win \$9, 50% chance to lose \$9. Expected value = 0.5 × (+\$9) + 0.5 × (−\$9) = **\$0**. It is a perfectly fair bet — over many repetitions you neither gain nor lose.
- After winning \$30, your final outcome is either \$39 or \$21. Expected final wealth = 0.5 × \$39 + 0.5 × \$21 = **\$30** — identical to just keeping the \$30 and not gambling.
- So the gamble adds *nothing* to your expected wealth in either condition. A purely rational agent would be indifferent, and their choice would not budge based on whether the \$30 was "won" a moment ago.

Yet the choice swings from 70% to 40% depending only on the prior outcome. That 30-point swing is the house money effect and the break-even effect, measured in a lab, on real money.

**Intuition:** when the same fair bet gets taken far more often after a win than after a loss, your risk appetite is being set by your recent history, not by the odds in front of you.

### The shape underneath it: your value curve

Why does a prior gain make you bolder and a prior loss make you desperate? Because of the shape of how gains and losses *feel* — the value function from prospect theory, the same curve that drives loss aversion. Here it is, drawn for our purpose.

![An S-shaped value curve through a reference point: convex and steep over losses (risk-seeking, chase to break even) and concave over gains (risk-averse, until relabeled 'house money').](/imgs/blogs/mental-accounting-and-the-house-money-effect-4.webp)

Read the curve from the center out. To the right of your reference point — in the black, sitting on a gain — the curve is concave: it rises but flattens, so each additional dollar of profit thrills you a little less. That normally makes you *risk-averse* over gains (you snatch small winners). To the left — in the red, sitting on a loss — the curve is convex and steep: the pain drops fast, then flattens, which makes you *risk-seeking* over losses. Down there, a gamble that offers a shot at climbing back to zero is disproportionately attractive, because getting back to the reference point is where the steepest relief lives. That is the **break-even effect**.

The house money effect is the twist Thaler and Johnson added on top. When you are sitting on a gain and you *segregate* that gain into its own "house money" account, your reference point shifts and you stop feeling the normal risk-aversion over gains — you flip to risk-seeking, gambling the winnings as if losing them would not really hurt. Two opposite-looking behaviors, one curve, one reference point. The whole rest of the post is these two effects loose on a trading desk.

### The editing rule: why the mind segregates gains and integrates losses

There is one more layer that makes the two effects click together, and it is the actual mechanism Thaler and Johnson proposed: people do not evaluate a new bet in isolation. They first *edit* it together with the prior outcome, and they choose the framing that feels best — a process later researchers call **hedonic editing**. The mind has two moves available, and it picks whichever one hurts less:

- **Segregate** — keep the prior outcome and the new bet in separate accounts. After a gain, this is comfortable: the win is banked as a pleasant, standalone fact, and a subsequent small loss is charged against the "house money" bucket where it barely registers. Segregating a gain is what *licenses* the extra risk — the potential loss gets pre-absorbed by a cushion that feels like it is not really yours.
- **Integrate** — combine the prior outcome and the new bet into a single running total. After a loss, this is where the break-even bet gets its pull: a gamble that could return you to zero is integrated with the existing loss to produce the one outcome the mind craves, a clean slate. A bet that merely reduces the loss a little is far less attractive than one that erases it entirely — even at worse odds — because only the erasure reaches the reference point.

So the same editing machinery produces both effects. Segregating a gain makes fresh risk feel free; integrating a loss makes break-even bets feel urgent. In both cases the tell is that your willingness to bet is being set by the *frame* you put around the prior outcome, not by the expected value of the bet in front of you. A trader who always integrates — one book, one running total, no separate "house money" ledger — never gets the segregation discount that funds the giveback.

## 1. The house money effect: profits feel like the market's chips

The house money effect on a desk is simple to state and brutal to live through: **after a win, you size up, drop your risk discipline, and give the win back — often plus interest.** The gains never felt like yours, so protecting them never felt urgent, so you wagered them on trades you would never have taken with "real" money.

The mechanism is a reference-point shift. Before the win, your reference point was your starting balance, and every dollar below it was a painful loss you worked to avoid. After the win, your brain quietly moves the reference point *up* to the new, higher balance — but only for the purpose of feeling good about the gain. When it comes time to *risk* that gain, the brain reclassifies it as "house money", outside the account it actually cares about protecting. You get the emotional upside of the win (it counts as yours) and the emotional permission to gamble it (it is the market's). Heads you win, tails the house loses. Except the house is you.

![A six-point timeline: start \$50,000, run up +\$30,000 to an \$80,000 peak framed as 'house money', size up 6x with stop discipline gone, give back -\$28,000, end at \$52,000 keeping only \$2,000.](/imgs/blogs/mental-accounting-and-the-house-money-effect-5.webp)

The timeline above is the canonical giveback, and it is worth walking through as a full worked example because the pattern is so common it is almost a cliché on trading desks.

#### Worked example: the house-money giveback

You start the year with a \$50,000 account and a rule you actually follow: 2% risk per trade, so about \$1,000 at risk on any position, hard stops always.

- **January to March.** A genuinely good run. You catch a trend, press it correctly, and by April your account is \$80,000 — up \$30,000, a 60% gain. You earned it with discipline.
- **The reframe.** Sitting on \$30,000 of profit, the voice arrives: *This is house money. I built a huge cushion. I can afford to swing for the fences now — even if I give some back, I'm still way up.* The \$30,000 stops feeling like your money.
- **The size-up.** You start risking not \$1,000 but \$6,000 per trade — six times your rule — on concentrated, lower-quality setups. You widen or skip stops, because a stop would "cap the upside" and anyway it is only house money.
- **The giveback.** A normal losing streak arrives, as they always do — say four losers in a row. At \$6,000 a pop, that is \$24,000 gone. You chase, take one more oversized shot to "get the cushion back," and lose another \$4,000. You are at \$52,000.

You ran \$50,000 up to \$80,000 and kept \$2,000 of a \$30,000 run. The trades in April that lost \$28,000 were not worse *ideas* than the trades in February that made \$30,000. They were the same quality — but sized six times larger, because the money funding them had been relabeled as not-quite-yours.

**Intuition:** a profit you refuse to call "mine" is a profit you will not protect — and the market is delighted to take back anything you are not defending.

The tell that you are in this trap is the phrase itself. If you ever catch yourself thinking *it's the market's money* or *I'm playing with profit now*, stop. That sentence is the bias announcing itself. There is no such thing as the market's money in your account. It is all yours, and it can all leave.

## 2. The break-even effect: the desperate mirror image

If the house money effect is what a gain does to you, the **break-even effect** is what a loss does. Sitting in the red, a bet that offers a chance to claw back to your entry price becomes irresistible — not because the odds are good, but because *returning to the reference point* is where the steepest emotional relief lives on that value curve. You will accept a bad bet, or a huge one, for the shot at "getting back to even."

This is the engine of the classic blow-up: down a manageable amount, a trader doubles the position to "average down" and recover faster, the market keeps going, they double again, and a small, survivable loss metastasizes into an account-ending one. Each doubling feels rational in the moment because each one offers the thing the brain craves most down there — a path back to zero. The math is running the opposite direction: bigger size into a losing thesis is exactly how you convert a bruise into a bankruptcy.

Notice the cruel asymmetry with the house money effect. After a *win*, you take more risk because the money is not really yours. After a *loss*, you take more risk because getting back to even is worth almost anything. Gains and losses pull in opposite emotional directions, but they arrive at the same destination: **too much size at exactly the wrong time.** Your position sizing should be governed by your rules and your account equity. Instead it is being jerked around by whether you happen to be above or below an arbitrary line you drew when you entered.

The professional counter is unglamorous: your loss on any trade is defined *before* you enter it, by your stop, and you do not get to renegotiate it because you are now underwater and want to be whole. The moment you find yourself adding to a loser to "get back to even faster," you are not managing a position — you are feeding the break-even effect. (This is closely related to the overconfidence trap covered in [overconfidence and the illusion of control](/blog/trading/trading-psychology/overconfidence-and-the-illusion-of-control): the belief that you can *make* the trade come back if you just commit more to it.)

The two effects are worth putting side by side, because they are the same reference-point machinery pointed in opposite directions — and recognizing which one has you is the first step to overriding it:

| | House money effect | Break-even effect |
|---|---|---|
| **Trigger** | A prior *gain* | A prior *loss* |
| **The feeling** | "It's the market's money now" | "I just need to get back to even" |
| **What you do** | Size up, drop stops, gamble the winnings | Double down, average into the loser |
| **Reference point** | Shifts up; gains reclassified as not-yours | Anchored at entry; below it feels unbearable |
| **On the value curve** | Should be risk-averse over gains — you flip to risk-seeking | Risk-seeking over losses |
| **Classic outcome** | The giveback: a great run handed back | The blow-up: a small loss turns fatal |
| **P&L signature** | Win, then surrender the win plus interest | Loss, then a much larger loss |

Both columns end in the same place: too much size at the worst possible moment. One arrives dressed as celebration, the other as rescue. Neither is a decision about the trade in front of you — both are reactions to a number in the past.

## 3. Segregated accounts: how buckets mis-size the whole book

So far the buckets have distorted how you *frame* money. This section is where they do something more concretely dangerous: they distort how you *size* the book. Because if you keep money in separate mental (or literal) accounts and size each one off its own balance, your *total* risk can drift far past any limit you believe you have — and you will not see it, because you never look at the whole thing at once.

![Bucketed view: a core account bar at 6% and an aggressive account bar at 12%, each 'feeling fine'; one-portfolio view: a single bar at 8.4%, breaking through a dashed 6% cap by \$2,400.](/imgs/blogs/mental-accounting-and-the-house-money-effect-6.webp)

The figure shows the trap in one picture. On the left, two accounts, each sized to "feel disciplined." On the right, the same positions viewed as one book — and the aggregate risk has blown straight through the cap you thought you were honoring. The reason is a denominator trick: a percentage is only as meaningful as what it is a percentage *of*, and bucketing lets you quietly change the denominator.

#### Worked example: three "disciplined" accounts, one over-leveraged book

You have \$100,000 in total, split across accounts, and a firm rule you are proud of: never more than 6% of your capital at risk across all open positions at once.

- **Core account, \$60,000.** You hold three positions, each risking "just 2% of the core account." That is 2% × \$60,000 = \$1,200 each, so \$3,600 at risk. Feels tame — only 2% per name.
- **Aggressive account, \$40,000.** You hold four positions, each risking "3% of the aggressive account." That is 3% × \$40,000 = \$1,200 each, so \$4,800 at risk. Feels fine — it is the account you *expect* to be spicy.
- **The whole book.** Add it up: \$3,600 + \$4,800 = **\$8,400** at risk. As a fraction of your true \$100,000 net worth, that is **8.4%** — well past the 6% cap you swore you would never exceed. And if those seven positions are correlated (all long tech, say), the real risk is worse still, because they will lose together.

Every individual bet looked disciplined against its bucket. The book was over-limit by \$2,400 the whole time. You never saw it because you never once computed risk against the one denominator that matters — the total.

**Intuition:** risk lives at the portfolio level, so sizing off buckets lets your true exposure sail past any cap you set — the only honest denominator is every dollar you have.

This is the most underrated cost of mental accounting for traders. The house money and break-even effects at least *feel* like something — a rush, a desperation. This one is silent. You can be a calm, rules-following trader in each account and still be running an aggregate book that would horrify you if you ever added it up. The habit of thinking in buckets is precisely what stops you from adding it up.

## 4. Realized vs unrealized, and the year-to-date "cushion"

The buckets multiply. Beyond "capital vs profit" and "core vs aggressive," traders build a whole taxonomy of envelopes, each with its own distortion. The matrix below catalogs the common ones — the label the mind slaps on the money, the way that label bends your risk, and the fungible truth that dissolves it.

![A four-row matrix mapping each mental bucket to the mind's label, its risk distortion, and the fungible truth: year-to-date profit, unrealized gains, the 'trading account', and a big recent win.](/imgs/blogs/mental-accounting-and-the-house-money-effect-7.webp)

Two rows deserve their own treatment because they catch even experienced traders.

**Realized vs unrealized gains.** A gain that is still open ("paper profit") gets filed as "not real yet", and that filing does real damage. It is why traders refuse to book a loss on a position that was once deeply in profit — *"I'm not selling until at least I'm back to where I was up"* — treating the unrealized peak as the reference point and the current, lower level as a loss to be avoided. But an unrealized \$5,000 is \$5,000 of your net worth *right now*; if you would not buy the position here, at this price, with cash, then holding it is an active decision to keep \$5,000 at risk in something you no longer believe in. Realized and unrealized dollars are the same dollars. The market marks them identically.

**The year-to-date "cushion".** This is the house money effect scaled up to the whole year, and it is the most seductive bucket of all, because being up on the year is a genuine achievement that *feels* like it has earned you a right to gamble.

#### Worked example: the year-to-date cushion that wasn't free

It is November. You started the year with \$100,000 and you are up 40% — the account is at \$140,000. That \$40,000 is "the cushion", "the year's profit", the buffer that lets you sleep. A big, ambitious trade appears.

- **The reasoning.** *I'm up 40% on the year. Even if I lose \$30,000 on this, I'm still up 10% — a great year. I can afford to swing.* So you put on a position sized to risk \$30,000 — a 21% swing on your current \$140,000 equity, a size you would have called insane back in January.
- **The trade hits its stop.** You are down \$30,000. Account: \$110,000, up 10% on the year. Still "green", technically.
- **What actually happened.** You wagered \$30,000 — three-quarters of your entire year's profit — on a single trade, on the theory that the \$40,000 was somehow less yours than the original \$100,000. It was not. It was \$40,000 of your net worth, and you put most of it on the table in one shot.

The cushion was never a cushion. "Up on the year" is a fact about the *past* — dollars that are already in your account and fully yours. The dollars you put at risk are always in the *present*, and they are always 100% your own.

**Intuition:** a year-to-date gain is not a free stake the market handed you to gamble — it is your money, and the calendar resetting on January 1st does not make last year's profit any less real today.

## 5. Narrow framing: how often you look decides how much you gamble

Here is the deeper structure under every example so far, and it is the third pillar of Thaler's mental accounting (the first two being how outcomes are coded, and how money is budgeted into categories). Accounts do not just have *labels* — they have *boundaries in time*. How often you close the books and start a fresh tally is called your **evaluation frequency**, or **choice bracketing**, and it silently decides how much house money you manufacture for yourself.

A **narrow bracket** evaluates each little chunk of trading on its own: this trade, this morning, this day. A **broad bracket** evaluates the whole book over a long horizon: one continuous account, one running total, forever. Mental accounting is fundamentally a bracketing choice, and the house money effect is a symptom of brackets drawn too narrow. If you bracket the day, then this morning's gains become "today's house money" you feel entitled to gamble this afternoon — and tomorrow the meter resets and you do it again. If you bracket the whole book, there is no such thing as house money: there is only your capital, marked to market, sized off the same rule every time. Widen the bracket and the bucket disappears.

#### Worked example: two traders, same week, different brackets

Two traders, A and B, take the identical set of trades over one volatile week on a \$100,000 account.

- **Trader A brackets by the day.** Each morning she starts a fresh mental tally. On a green morning (+\$3,000 by noon), the \$3,000 becomes "today's house money," so she doubles her size in the afternoon to "play with the profit." On a red morning, she trades cautiously to "protect the day." Her afternoon size is a function of her *morning*, which is noise. Across the week her position sizing swings between 2% and 8% of the account depending on random morning outcomes — she has amplified the market's variance with her own.
- **Trader B brackets by the book.** He never resets. Every position, every afternoon, every day, is sized off the same number — total net liquidity right now — at the same 2% rule. A green morning changes his equity by \$3,000 and therefore his next bet by 2% of \$3,000 = \$60. That is the *only* honest effect a morning gain should have on your sizing: it changes the total by a little, so it changes the next bet by a little.

Same trades, same week. Trader A's returns are dominated by the interaction between random mornings and her house-money sizing; Trader B's are dominated by the trades themselves. Over a career, A blows up in a bad week that happens to follow a good morning; B compounds.

**Intuition:** the house-money bucket needs a boundary to hide behind, so the narrower your brackets, the more "free" money you invent to gamble — widen the bracket to the whole book and there is nothing left to gamble but your own capital, sized by rule.

The table makes the ladder explicit: every bracket you might draw, the "house money" it invents, and the sizing damage it does.

| Bracket you draw | What becomes "house money" | The sizing damage |
|---|---|---|
| The last trade | last trade's profit | revenge / victory-lap size on the next |
| The day | this morning's gains | gamble the afternoon, reset tomorrow |
| The month | this month's profit | size up after any green month |
| The year | year-to-date "cushion" | the November swing-for-the-fences |
| **The whole book** *(correct)* | **nothing — it is all capital** | **one rule, sized off total equity** |

Read the table top to bottom and the fix is obvious: the wider the bracket, the less house money exists, until at the bottom — one continuous book — there is none. That bottom row is the entire goal of the drill later in this post.

## What it looks like at the screen

Biases are easier to catch when you know their tells. Here is what mental accounting and the house money effect actually feel like in real time, at the screen, so you can spot the moment it grabs the wheel:

- You have a green morning and your hand reaches for **double size** on the next setup, with a thought like *I'm up on the day, so this one is on the house.*
- You **widen or delete a stop** on a winner because "it's only giving back profit if it turns" — you are protecting the feeling of the win, not the dollars.
- You mentally (or literally) keep a **"trading account" walled off from your "real" savings**, and you take risks in the trading account you would never take if the number on the screen were labeled "retirement."
- You are down on a position and you **add to it to get back to even faster**, and the add feels like relief, not risk.
- You refuse to sell a loser because you are *"waiting to get back to at least where I was up"* — anchoring to an unrealized peak that exists only in your memory.
- You size a big, out-of-character trade and justify it with **"I'm up X% on the year"** — the calendar as permission slip.
- After booking a profit, you feel a strange itch to *put it back to work immediately and aggressively*, as if realized profit is burning a hole in your pocket — the "found money" effect from your household budget, leaking onto your desk.

Every one of those is the same bug: a dollar being treated as less-than-fully-yours because of where it came from or which envelope it sits in. When you feel one, name it — *"that's house money talking"* — and re-run the sizing on your total account, not the bucket.

## The one-portfolio drill

The cure for mental accounting is not willpower. It is a *denominator*. You defeat the bias by refusing to let any bucket be the basis for a decision — every position is sized off one number, your total net liquidity, and one constraint, your current open risk. Here is the drill.

![A five-step pipeline: mark-to-market total net liquidity, re-label 'house money' as MY money, sum current open risk in dollars, size the next trade off total plus open risk, arrive at one book with one 2% rule.](/imgs/blogs/mental-accounting-and-the-house-money-effect-8.webp)

Run these five steps, in order, before every position and at the start of every session:

1. **Mark-to-market your total net liquidity.** One number: every account, every position at its current price, all cash, added together. This is your *real* size, right now. Not your capital, not your capital-plus-profit, not this account or that account. Everything. If you keep multiple brokerage accounts, sum them — the market does.
2. **Re-label all "house money" as MY money.** Explicitly, in words. At the start of each period (day, week, month), say it: *every dollar in this number is mine, and every dollar can leave.* This is the step that resets the reference point on purpose, before the bias resets it for you. There is no profit bucket and no capital bucket. There is one bucket, and it is all yours.
3. **Sum your current open risk in dollars.** Add up, across all positions, how much you lose if every stop hits. This is the number that tells you whether you have room for another bet — not your P&L, not your win rate, your open risk in dollars.
4. **Size the next trade off total net liquidity and the open-risk budget.** Your per-trade risk is a fixed fraction (say 2%) of the *total* from step 1, and the new position only goes on if it keeps your *total* open risk under your cap (say 6%). If you are already at the cap, the answer is no — regardless of how good the setup looks or how "up" you are.
5. **One book, one rule.** The output is a single portfolio with a single risk rule, sized off a single denominator. No envelope gets a special dial.

The discipline the drill enforces is exactly the discipline mental accounting erodes: it makes **every dollar equal again.** It is deliberately mechanical, because the bias operates on feeling and the only reliable counter to a feeling is a number you compute the same way every time. A trader who sizes off total net liquidity cannot have a house-money blow-up, because there is no house money in the drill — there is only the book, and the book is all theirs.

One more habit worth adopting: **re-baseline your reference point on a fixed schedule, out loud.** On the first of each month, whatever your account is worth *is your new starting capital.* Last month's profit is not a cushion to gamble; it is simply your capital now, to be protected exactly as fiercely as your original deposit. Doing this on a schedule beats letting the market do it to you in the middle of a hot streak.

#### Worked example: the drill catching a bad size in real time

You are up \$8,000 on the day, on a \$100,000 book, and a setup appears that you want to size at \$5,000 of risk — because "I'm up \$8,000, worst case I give back most of the day." Run the drill instead of the feeling.

1. **Total net liquidity:** \$108,000 (the \$8,000 is already yours, already in the number).
2. **Re-label:** every dollar of that \$108,000 is mine, including the \$8,000.
3. **Open risk:** you already have two positions on, risking \$1,400 and \$1,000 — \$2,400 of open risk live.
4. **Size the new trade:** your rule is 2% per trade and a 6% total-open-risk cap. Two percent of \$108,000 is \$2,160 — so the *most* this trade can risk is \$2,160, not \$5,000. And your cap check: \$2,400 already open + \$2,160 new = \$4,560, which is 4.2% of the book, under the 6% cap. The trade is allowed — at \$2,160, less than half the size the house-money feeling wanted.

The drill did not forbid the trade; it right-sized it. The \$5,000 figure came from the day's P&L (house money); the \$2,160 figure came from your total equity and your rule. One of those numbers is a feeling. The other is arithmetic.

**Intuition:** the drill does not require you to be less confident or less aggressive — it just replaces the denominator "today's profit" with the denominator "everything I own," and the right size falls out on its own.

## Common misconceptions

**"But it really is profit — I can afford to take more risk with it."** No. Once the profit is in your account, it is capital, indistinguishable from every other dollar. "Affording" more risk is a real concept — it is a function of your total net liquidity and your risk tolerance — but it has nothing to do with whether a given dollar arrived as a deposit or a gain. If you can genuinely afford 3% risk per trade, you can afford it on your whole book; if you can only afford 2%, then 2% applies to the profit too. The source of the dollar is irrelevant to how much risk it can bear.

**"Separating accounts is just good organization."** Organizing money into accounts for *tax, legal, or goal* reasons is fine — a retirement account and a trading account genuinely have different rules imposed from outside. The bias is not the accounts; it is sizing each one's risk off its own balance instead of your total, and letting the walls stop you from ever seeing aggregate exposure. Keep the accounts if you must; compute risk against the sum.

**"The house money effect only hits amateurs."** Thaler and Johnson found it in ordinary experimental subjects, but the same reference-point machinery operates in professionals — and history's most expensive trading blow-ups are studded with it (see the case studies below). The professional edge is not immunity to the feeling; it is a sizing process that ignores the feeling. Everyone feels the pull to gamble house money. Pros have a drill that overrides it.

**"Booking the profit makes it real and safe."** Realizing a gain converts it from paper to cash, which is psychologically satisfying, but it does not change your risk one bit if you immediately redeploy the cash aggressively because it now feels like "found money." Realized profit sitting in your account is exactly as at-risk as the next trade you put it into. The safety comes from sizing, not from the realize button.

**"I need the cushion to give me room to trade freely."** The cushion feeling is precisely the mechanism that makes you trade *worse* — bigger and looser — right after a good run, which is statistically when overconfidence is highest. What actually gives you room to trade is a fixed, small per-trade risk that keeps you in the game across inevitable losing streaks. Freedom in trading comes from small bets, not from a mental buffer you feel entitled to burn.

## How it shows up in real markets

The lab result is clean; the market version is bloody. Here are named episodes where mental accounting, the house money effect, or the break-even effect did the damage. The numbers are real and sourced; the psychological read is the interpretation this post argues for.

### 1. Jesse Livermore: the serial house-money giveback

Jesse Livermore was the most famous speculator of the early twentieth century, and his career is the house money effect written across decades. In the Wall Street crash of 1929 he was short the market and, by contemporary accounts, netted on the order of **\$100 million** — an almost unimaginable fortune at the time. Yet by 1934 he was **bankrupt**, his liabilities exceeding his assets, and he died in 1940 with his best years long behind him (Livermore lived 1877–1940). This was not a one-time reversal; across his life he made and surrendered several fortunes in the same pattern — run the stake up spectacularly, then risk it as if the winnings were playing money rather than his own hard-won capital.

Livermore himself understood the discipline he lacked; his writing is full of rules about sizing and patience that his behavior repeatedly violated after a big win. That gap — knowing the rule, feeling the house-money pull, and sizing off the feeling anyway — is the whole lesson. Being right about the market (and he was, historically so, in 1929) does not protect you if you then gamble the proceeds as though they were not really yours.

### 2. Nick Leeson and Barings: the break-even effect at institutional scale

Barings was Britain's oldest merchant bank, founded in **1762** — 233 years old and blue-blooded enough to have been called "the sixth great power" of Europe in the nineteenth century. In February **1995** it was destroyed by a single 28-year-old trader in its Singapore office. Nick Leeson took unauthorized, escalating derivatives positions on the Nikkei and hid the losses in a secret error account numbered **88888**. When the market moved against him, he did not cut the position — he *doubled down* to trade back to break-even, the exact behavior the break-even effect predicts. After the Kobe earthquake sent the Nikkei crashing in January 1995, he doubled down harder still, chasing a return to zero that never came.

By the time it was discovered, the loss was about **£827 million** (roughly US\$1.4 billion at the time) — more than twice the bank's available capital. Barings collapsed and was sold for £1. Leeson was sentenced to six and a half years in Singapore's Changi prison. The mechanism was not exotic fraud so much as the oldest emotional trap on the desk, industrialized: down a survivable amount, he chased break-even with ever-larger size, and turned a hole into an extinction event. Every doubling was the break-even effect offering the thing the losing brain wants most — a path back to the reference point.

### 3. Amaranth Advisors: house-money size at a multi-billion-dollar fund

Amaranth Advisors was a hedge fund that, at its peak, managed roughly **\$9.5 billion**. Its star energy trader, Brian Hunter, had made enormous profits on natural gas in prior years — reportedly over a billion dollars for the fund in 2005 — and that track record bought him the latitude to run ever-larger positions in 2006. That latitude is the institutional face of house money: after a huge win, the size that would have been unthinkable becomes "affordable" because the fund is sitting on a cushion the winner earned. Hunter put on giant natural-gas calendar-spread bets (long winter, short summer contracts). When the spreads moved against him in September **2006**, the positions were so large that the fund could not exit without crushing the very prices it needed, and margin calls did the rest.

In a matter of weeks the fund lost on the order of **\$6 billion** — roughly two-thirds of its assets — and collapsed, the largest hedge-fund blow-up of its era. The precise trigger was a spread move, but the precondition was size: a book grown to a scale that only a big cushion of prior profit could have psychologically justified. Being up enormous did not make Amaranth safe; it made it big, and big into a losing move is how the giveback becomes a collapse.

### 4. The record-month blow-up (the recurring desk pattern)

You do not need a nine-figure disaster to see the pattern; it recurs on ordinary desks constantly, and it has a shape reliable enough to name. A trader has a record month — call it up 30–40% — and the very next month gives most of it back. The post-mortem is almost always the same: after the record run, the trader sized up (house money), loosened stops (the profit "could afford it"), and concentrated into fewer, bigger bets (the cushion gave "permission"). The gains funded the recklessness that erased the gains. The reason this is worth naming without attaching a specific name is that it is *your* likely failure mode too: the single most dangerous moment in a trading career is not a drawdown — it is the peak right after a great run, when the house-money feeling is strongest and the discipline is loosest.

### 5. The lottery-and-tax refund tell (why you already know this bias)

Outside of trading, the same wiring is easy to catch in yourself, which is useful because it proves the bias is *yours*, not just other people's. A tax refund — money that was always yours, merely returned — gets spent more freely than an identical amount from your paycheck, because it is filed as "found money." A modest lottery or gambling win gets wagered again far more readily than money earned at work. Thaler's original mental-accounting work is full of these everyday asymmetries (the classic example: people will drive across town to save \$5 on a \$15 calculator but not to save the same \$5 on a \$125 jacket — the \$5 is filed relative to the purchase, not treated as the fungible \$5 it is). The trading version is not a different bias. It is the same envelope system, running on a bigger, faster, more dangerous pool of money.

## When this matters to you

Mental accounting is not a rare, exotic failure. It is the default setting of a human brain handling money, and it fires hardest at the exact moment you are most vulnerable: right after you win. If you take one thing from this piece, make it the reflex to distrust the feeling of a cushion — because that feeling is the bias, and it arrives precisely when your discipline has been rewarded and you are most inclined to relax it.

The good news is that the fix is not a personality transplant. It is a denominator and a habit: size every position off your total net liquidity and your current open risk, and re-label all "house money" as your money on a fixed schedule, out loud, before the market does the relabeling for you. Every dollar you own is equally yours, equally spendable, and equally loseable, regardless of whether you deposited it or earned it. The moment you truly internalize that — the moment a "profit dollar" and a "capital dollar" feel identical in your hand — the house money effect has nowhere left to stand.

For the broader map of how this bias connects to the others that share its reference-point machinery, see [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders); for the closely related asymmetry in how gains and losses feel, [loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect); and for the overconfidence that makes house-money size feel justified, [overconfidence and the illusion of control](/blog/trading/trading-psychology/overconfidence-and-the-illusion-of-control).

## Sources & further reading

- Richard H. Thaler, *Mental Accounting and Consumer Choice*, Marketing Science 4(3), 199–214 (1985) — the paper that named and formalized mental accounting; source for the definition, fungibility framing, and the calculator/jacket asymmetry. [INFORMS / PubsOnline](https://pubsonline.informs.org/doi/10.1287/mksc.4.3.199)
- Richard H. Thaler & Eric J. Johnson, *Gambling with the House Money and Trying to Break Even: The Effects of Prior Outcomes on Risky Choice*, Management Science 36(6), 643–660 (1990) — the source of the house money effect, the break-even effect, and the 70%-vs-40% coin-flip results. [INFORMS / PubsOnline](https://pubsonline.informs.org/doi/10.1287/mnsc.36.6.643)
- Richard H. Thaler, *Mental Accounting Matters*, Journal of Behavioral Decision Making 12(3), 183–206 (1999) — the review that ties mental accounting, the house money effect, and choice bracketing together. [Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1099-0771(199909)12:3%3C183::AID-BDM318%3E3.0.CO;2-F)
- Bankruptcy of Barings Bank (1995): founding in 1762, Nick Leeson, account 88888, the "doubling" strategy, the ~£827 million (~US\$1.4bn) loss, and the collapse — [Encyclopædia Britannica](https://www.britannica.com/event/bankruptcy-of-Barings-Bank) and [Nick Leeson (Wikipedia)](https://en.wikipedia.org/wiki/Nick_Leeson).
- Jesse Livermore: the ~\$100 million netted shorting the 1929 crash, the 1934 bankruptcy, and the 1877–1940 life — [Jesse Livermore (Wikipedia)](https://en.wikipedia.org/wiki/Jesse_Livermore).
- Amaranth Advisors: the ~\$9.5 billion fund, Brian Hunter's natural-gas calendar spreads, and the ~\$6 billion (roughly two-thirds of assets) loss in September 2006 — [Amaranth Advisors (Wikipedia)](https://en.wikipedia.org/wiki/Amaranth_Advisors) and [Brian Hunter (Wikipedia)](https://en.wikipedia.org/wiki/Brian_Hunter_(trader)).
- Sibling posts on this blog: [Loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect), [Overconfidence and the illusion of control](/blog/trading/trading-psychology/overconfidence-and-the-illusion-of-control), and [The cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders).
