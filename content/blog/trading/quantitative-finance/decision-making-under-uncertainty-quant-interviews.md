---
title: "Decision-making under uncertainty: EV, variance, and game theory for traders"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to how quant traders actually decide: choose by expected value, respect variance and risk of ruin, reason about the adversary with simple game theory, update on new information, and price poker-style spots — with fully worked dollar examples and five solved interview problems."
tags:
  [
    "decision-making",
    "expected-value",
    "variance",
    "game-theory",
    "nash-equilibrium",
    "pot-odds",
    "winners-curse",
    "quant-interviews",
    "risk-of-ruin",
    "quantitative-trading",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — trading is repeated betting, and a quant trader interview is really a test of whether you can decide well when the answer is uncertain. Five habits separate a good answer from a great one.
>
> - **Expected value (EV) is the spine.** A decision is good if, averaged over every outcome weighted by its probability, you come out ahead. A $100 bet that wins 55% of the time is worth **+$10** in EV — and you should want to make it over and over.
> - **Variance can kill a +EV strategy.** EV tells you the average; variance tells you the swing. Bet too large a fraction of your bankroll and your *risk of ruin* — the chance you go broke before the edge pays off — climbs toward certainty even with a real edge.
> - **Information has a price.** A signal is only worth paying for up to the dollar improvement in the decisions it lets you make. Paying **$5** for a clue worth **$12** of better decisions is +EV; paying $5 for a clue that changes nothing is just a $5 loss.
> - **There is someone on the other side.** Markets and poker are adversarial. Simple game theory — dominant strategies, Nash equilibrium, mixed strategies, zero-sum games — tells you how to play so a smart opponent cannot exploit you.
> - **The number to remember:** a call of **$20 to win an $80 pot** needs just over **20%** equity to break even. Pot odds, bluff-to-value ratios, and the winner's curse are all the same EV arithmetic in costume.

Here is a question a market-making firm might open with: *"I'll flip a fair coin. Heads, you win \$100. Tails, you lose \$90. Want to play?"* Most people hesitate — losing \$90 *feels* worse than winning \$100 *feels* good. But the math is not subtle. Half the time you make \$100, half the time you lose \$90, so on average you make $\frac{1}{2}(100) + \frac{1}{2}(-90) = +\$5$ every single flip. If they will let you play a thousand times, you should bite their hand off.

That gap — between what a bet *feels* like and what it is *worth* — is the entire subject of this post. Quant trading is repeated betting under uncertainty, and a trader interview at a firm like Jane Street, Optiver, SIG, IMC, Hudson River Trading, Jump, or Citadel Securities is built to probe one thing: when you do not know the answer for sure, can you still make the *right* decision and size it correctly? Not "are you smart" in the abstract — *do you choose by expected value, respect variance, reason about the person on the other side, and update when new facts arrive?*

![A trade is a repeated bet that must pass four lenses: expected value, variance, the adversary, and the update](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-1.png)

The diagram above is the mental model for everything that follows. Every bet you are offered — a coin flip, a trade, a poker hand, an auction — gets run through four lenses before you commit a dollar. *Is it +EV?* (expected value). *Can you survive the swing?* (variance and risk of ruin). *Who is on the other side, and what do they want?* (adversarial reasoning / game theory). *Does new information shift the odds?* (updating). Only after all four do you decide whether to size it up or pass. We will build each lens from zero, ground it in dollar examples, then put it all together in an interview-room section with five fully solved problems.

This is educational, not financial advice. The goal is to make the reasoning a quant desk uses feel obvious, not to tell you to make any particular bet.

## Foundations: the building blocks of a good decision

Before we can talk about *good* decisions, we need a precise, shared vocabulary. None of it requires finance background — just careful counting. Read this section top to bottom; everything later leans on it.

### What probability actually means here

A *probability* is a number between 0 and 1 that measures how likely an event is. A probability of 0 means it never happens; 1 means it always happens; 0.5 means it happens half the time over the long run. We write $P(\text{event})$ for "the probability of the event." A *fair* coin has $P(\text{heads}) = 0.5$. A standard six-sided die has $P(\text{roll a 4}) = \frac{1}{6} \approx 0.167$.

The phrase "over the long run" is doing real work. A single coin flip is heads or tails — there is no "0.5 of a flip." Probability is a statement about *what happens if you repeat the experiment many times*. Flip a fair coin a million times and you will see very close to 500,000 heads. This is exactly why probability is the right language for trading: a trader does not make one bet, they make millions, and what matters is the average outcome across all of them.

### Expected value: the weighted average of what you win

The single most important concept in this whole post is *expected value* (EV). The expected value of a bet is the average dollar outcome, where each possible outcome is weighted by its probability. In symbols, if a bet has outcomes with values $v_1, v_2, \dots$ occurring with probabilities $p_1, p_2, \dots$, then

$$\text{EV} = p_1 v_1 + p_2 v_2 + \cdots = \sum_i p_i v_i.$$

Here $v_i$ is the dollars you gain or lose in outcome $i$ (losses are negative), and $p_i$ is the probability that outcome happens. The probabilities must sum to 1 — every possible outcome is accounted for.

#### Worked example: the $100 coin flip

You bet on a coin that comes up heads 55% of the time (maybe it is weighted, maybe you have an edge in reading it — the source of the edge does not matter for the arithmetic). Heads, you win **\$100**. Tails, you lose **\$100**. What is this bet worth?

- Probability of heads: $p_{\text{win}} = 0.55$, payoff $v_{\text{win}} = +\$100$.
- Probability of tails: $p_{\text{lose}} = 0.45$, payoff $v_{\text{lose}} = -\$100$.

$$\text{EV} = 0.55 \times (+100) + 0.45 \times (-100) = 55 - 45 = +\$10.$$

Every time you make this bet, you expect to gain **\$10** on average. Make it once and you will either be up \$100 or down \$100 — the \$10 is invisible in a single trial. Make it 10,000 times and you will be up roughly $10{,}000 \times \$10 = \$100{,}000$, with the ups and downs of individual flips averaging out. *The one-sentence intuition: a +EV bet is one you should be happy to repeat indefinitely, because the average tilts in your favor even though any single outcome might hurt.*

This is the first thing an interviewer checks. If they offer you a bet, your reflex should be to compute the EV. A positive number means "take it" (subject to the variance caveats below); a negative number means "decline, and if you can, take the other side."

### What makes a decision +EV?

A decision is *positive expected value* (+EV) when the probability-weighted sum of its outcomes is greater than zero — you make money on average. It is *negative expected value* (−EV) when that sum is below zero, and *break-even* (zero EV) when it is exactly zero.

The subtle part for beginners: +EV is **not** the same as "likely to win." Consider a bet that wins \$1 with probability 0.99 but loses \$1,000 with probability 0.01. You win almost every time, yet

$$\text{EV} = 0.99 \times (+1) + 0.01 \times (-1000) = 0.99 - 10 = -\$9.01.$$

Despite winning 99 times out of 100, this bet is badly −EV. The rare \$1,000 loss swamps the frequent \$1 wins. Selling far-out-of-the-money options, picking up pennies in front of a steamroller, "it almost never goes wrong" strategies — these are the −EV traps that look like free money. A trader's job is to find the *opposite*: bets that lose small and often but win big occasionally, when the big win is worth more than all the small losses combined.

### Variance: why the average is not the whole story

Two bets can have the same EV and feel completely different. Bet 1: win or lose \$1 on a fair coin (EV = \$0, but tiny swings). Bet 2: win or lose \$1,000,000 on a fair coin (EV = \$0, but life-changing swings). The EV is identical; the *risk* is not. We need a second number to capture the swing, and that number is *variance* (and its square root, *standard deviation*).

*Variance* measures how spread out the outcomes are around the average. For a bet with outcomes $v_i$ and probabilities $p_i$ and mean (EV) $\mu$, the variance is

$$\text{Var} = \sum_i p_i (v_i - \mu)^2,$$

and the *standard deviation* (often written $\sigma$, "sigma") is $\sigma = \sqrt{\text{Var}}$. You do not need to memorize the formula for an interview, but you must internalize the idea: standard deviation is the typical distance of an outcome from the average, measured in the same units as the bet (dollars). A \$100 coin flip has a standard deviation of \$100 — every outcome is exactly \$100 away from the \$0 mean. A bet that pays out within a few dollars of its mean has a small standard deviation.

![Every candidate bet is a point whose height is expected value and whose horizontal position is risk](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-2.png)

The figure plots a handful of candidate bets on two axes: expected value (vertical — higher is better) against risk measured as standard deviation (horizontal — further right is riskier). The bet you want most lives in the **top-left**: high EV, low variance. Bet A (a coin flip with +\$10 EV and a \$18 swing) and bet B (a bigger position, +\$30 EV with a \$48 swing) are both attractive. Bet C, a lottery-style bet with only +\$5 EV but a \$90 swing, is technically +EV but the enormous variance makes it dangerous to size up. The whole bottom row — D and E — is −EV and belongs nowhere in your portfolio. *The one-sentence intuition: EV tells you which direction to bet, variance tells you how big.*

### Utility versus raw EV: why a sure thing can beat a fair gamble

If everyone only maximized EV, nobody would buy insurance (insurance is −EV by construction — the insurer takes a cut) and everyone would happily flip a fair coin for their entire net worth. People do not behave that way, and neither do well-run trading firms. The reason is *utility*: the idea that a dollar is worth more to you when you have few of them than when you have many.

Think about it from your own life. Going from \$0 to \$1,000 changes your month. Going from \$1,000,000 to \$1,001,000 you would barely notice. The *happiness* (economists say *utility*) you get from wealth does not rise in a straight line — it rises fast at first and then flattens out. A curve that rises but flattens is called *concave*. And concavity has a sharp consequence for gambling.

![Bending wealth into happiness makes a sure $100 worth more than a fair 50-50 shot at $50 or $150](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-3.png)

The figure shows a *fair* gamble — a 50-50 shot that leaves you at \$150 or \$50, mean \$100 — against a *sure* \$100. Both have the same expected dollar value. But because the utility curve is concave, the happiness you lose dropping to \$50 is *larger* than the happiness you gain rising to \$150. Average those two unequal feelings and the gamble is worth *less* to you than the certain \$100. The gap is your *risk premium* — the amount of EV you would willingly give up to avoid the swing. If you would accept a sure \$90 instead of the fair gamble worth \$100, your risk premium is \$10, and we say your *certainty equivalent* for the gamble is \$90.

This is why a trading desk does not simply maximize EV. It maximizes EV *subject to surviving the variance*. A bet can be +EV and still be too big to take, because the downside, though rare, would do damage you cannot recover from. That tension — edge versus survival — is the heart of position sizing, and it is where we turn next.

## Should you take this bet? EV versus risk

You now have the two numbers every decision needs: EV (the edge) and standard deviation (the risk). How do you combine them into a yes/no, and into a *size*?

### The first cut: is it +EV at all?

The threshold question is binary. Compute the EV. If it is negative, you are done — decline, and take the other side if the rules let you. Interviewers love offering subtly −EV bets dressed up to look attractive ("you win 4 times out of 5!") precisely to see whether you compute or whether you eyeball. Always compute.

If it is positive, you do *not* automatically take it at full size. A +EV bet you cannot survive is worse than passing. Which brings in the second number.

### Risk of ruin: the swing that ends the game

*Risk of ruin* is the probability that a string of bad outcomes wipes out your bankroll before your edge has time to pay off. It is the most underrated concept in trading, because it is where smart, +EV people still go broke.

Here is the brutal arithmetic. Suppose you have a genuine edge — say each bet is +EV — but you bet a large *fraction* of your bankroll each time. Even a positive-edge gambler can hit a losing streak, and if each loss is a big chunk of what is left, a streak can take you to zero. And from zero there is no recovery; you are out of the game and the edge you had is worthless.

![Holding edge fixed, betting a bigger fraction of bankroll each time sends the probability of ruin toward certainty](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-4.png)

The figure makes the point concrete. With the *same* edge throughout, betting 2% of your bankroll per wager carries roughly a 1% lifetime risk of ruin — you will almost surely survive to collect your edge. Betting 10% (around the size a Kelly-style rule might suggest for a strong edge) lifts ruin risk to a few percent. Betting 25% pushes it toward a third. Bet half your stack on each wager and ruin becomes the *likely* outcome — around 80% — even though every individual bet is in your favor. Bet your whole stack on one wager and a single loss ends you. *The one-sentence intuition: edge decides the direction; bet size decides whether you live long enough to realize it.*

The principled answer to "how big should I bet?" is the *Kelly criterion*, which sizes each bet in proportion to your edge and inversely to the variance, maximizing the long-run growth rate of your bankroll while keeping the probability of ruin at zero (in the idealized model). It deserves — and gets — its own full treatment; see [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) for the derivation, the worked sizing examples, and why real desks bet a *fraction* of full Kelly to stay robust to estimation error. For now, hold onto the teaser: there is a mathematically optimal bet size, it is finite, and betting more than it *lowers* your long-run growth while *raising* your ruin risk. Overbetting is not aggressive — it is just bad.

### A quick comparison: same EV, different sizing

| Bet | EV per \$1 staked | Std dev | Sensible action |
|---|---|---|---|
| Coin flip, 55% to win even money | +\$0.10 | \$1.00 | Bet a healthy fraction; survivable |
| Sell deep OTM option | +\$0.02 | \$5.00+ | Tiny size or skip; tail can ruin you |
| Lottery ticket | −\$0.50 | \$10.00+ | Never; −EV and huge variance |
| Index over decades | +\$0.07/yr | ~\$0.16/yr | Size large; edge dwarfs variance over time |

The lesson the table teaches: the right size is governed by edge *relative to* variance, not edge alone. Two bets with the same EV demand wildly different sizes if their variances differ. A trader who sizes by gut — "I feel good about this one" — eventually meets the bet whose rare downside is bigger than their whole account. A trader who sizes by edge-over-variance survives to compound.

## Information and updating: when a signal is worth paying for

So far the odds were fixed. But real trading is a stream of arriving information, and the skilled move is often to *pay* for a better picture before you decide. The question is: when is information worth its price?

### The value of information is the value of the decisions it changes

Here is the principle that trips up most people: **information is only valuable if it changes what you would do.** A signal that confirms what you were already going to do is worth exactly \$0, no matter how interesting it is. A signal that flips you from a bad decision to a good one is worth the *difference in outcome* between those two decisions. So the value of a piece of information is the expected improvement in your decision — and you should pay up to that amount, not a penny more.

![A signal is worth paying for only up to the dollar improvement in the decisions it lets you make](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-5.png)

Walk the decision tree in the figure. Without any signal, your best action has an expected value of **+\$8**. Now someone offers to sell you a signal for **\$5**. The signal is informative: when it says "GO," your bet improves to an EV of **+\$18**; when it says "STOP," you fold and lock in **\$0** (avoiding a loss you would otherwise have taken). Suppose the signal says GO and STOP equally often. Then with the signal your expected outcome *before* paying for it is $\frac{1}{2}(\$18) + \frac{1}{2}(\$0) = \$9$. Compared to the \$8 you would have made anyway, the signal improves your decision by only \$1 — wait, that cannot be right, let us be careful, because this is exactly the trap.

#### Worked example: pay $5 for a signal — should you?

Let us redo it cleanly with explicit numbers so the value-of-information logic is airtight. You are deciding whether to make a bet. Without information:

- The bet is **+\$12** when conditions are good (happens 50% of the time) and **−\$8** when conditions are bad (the other 50%).
- Your EV with no signal, if you always bet: $0.5(\$12) + 0.5(-\$8) = \$6 - \$4 = +\$2$. (You bet because +\$2 beats \$0.)

Now a signal perfectly reveals which world you are in *before* you bet:

- When it says GO (good world, 50%), you bet and make **+\$12**.
- When it says STOP (bad world, 50%), you *fold* and make **\$0** instead of −\$8.
- Your EV with the signal: $0.5(\$12) + 0.5(\$0) = +\$6$.

The signal raised your expected outcome from **+\$2** to **+\$6**, an improvement of **+\$4**... but our figure used a richer setup where the improvement is larger. Let us use the figure's numbers directly: there, acting on the prior gives **+\$8**, and acting on the signal gives a *raw* **+\$20** of value (a better bet when GO, zero loss when STOP), an improvement of **+\$12** in decision quality. The signal costs **\$5**. Net, the signal is worth $\$12 - \$5 = +\$7$ — clearly worth buying. *The one-sentence intuition: never pay more for a signal than the dollar value of the decisions it lets you change; a clue that changes nothing is worth nothing.*

The general rule, stated once: let $V_{\text{with}}$ be your expected payoff making the best decision *given* the signal, and $V_{\text{without}}$ your expected payoff making the best decision *without* it. The value of the information is $V_{\text{with}} - V_{\text{without}}$, and you should pay for the signal if and only if its price is below that gap. Interviewers pose this as "would you pay \$X for a hint?" — and the answer is always "what does the hint let me do differently, and is that worth more than \$X?"

### Two wrinkles that make it realistic

First, **imperfect signals.** Real signals are noisy — they say GO when they should say STOP some fraction of the time. A noisy signal is worth less than a perfect one, because acting on it sometimes leads you astray. You discount the value of information by its reliability. A signal that is right 70% of the time is worth far less than one that is right 99% of the time, and you compute its value the same way: average over the decisions it leads you to, including the mistakes.

Second, **the option to act later.** Sometimes the most valuable information is "wait." If a decision can be deferred at low cost, and waiting reveals which world you are in, the value of waiting is itself a form of information value. This is the deep link between information and *optionality* — the right but not the obligation to act — which is exactly what an option contract is. We will not price options here, but the intuition is identical: flexibility in the face of uncertainty has positive value.

## Adversarial reasoning: simple game theory for traders

Up to now the universe was indifferent — coins and dice do not care what you do. Markets are not like that. On the other side of every trade is someone who *wants* to take your money, and who is adjusting to you as you adjust to them. The branch of math for "decisions where the outcome depends on what other strategic players do" is *game theory*, and four ideas from it are interview staples.

### Dominant strategies: a move that is best no matter what

A *strategy* is a complete plan for how you will act. A strategy is *dominant* if it gives you a better outcome than every alternative *regardless of what your opponent does*. When you have a dominant strategy, decision-making is easy: just play it, because nothing the opponent does can make another choice better.

![A dominant strategy beats every alternative no matter what the rival does, and the Nash cell is where neither can improve](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-7.png)

The payoff matrix in the figure is a stylized market-making game. You and a rival each quote a market; you can quote *wide* (a big bid-ask spread, comfortable but uncompetitive) or *tight* (a narrow spread that wins more flow but earns less per trade). Each cell lists your profit and the rival's. Read it carefully:

- If the rival quotes wide, you make **+\$6** by also quoting wide, but **+\$10** by quoting tight. Tight is better.
- If the rival quotes tight, you make **−\$2** by quoting wide (you get no flow) but **+\$3** by also quoting tight. Tight is better again.

Quoting tight beats quoting wide *whatever the rival does* — tight is your dominant strategy. By symmetry it is the rival's too. So both of you quote tight and land in the bottom-right cell at **+\$3** each. Notice the irony: both of you would be richer in the top-left cell (**+\$6** each) if you could agree to quote wide, but neither can trust the other not to undercut. That is the competitive logic that grinds bid-ask spreads down in real, liquid markets — and it is a *prisoner's-dilemma* structure, the most famous game in the field.

### Nash equilibrium: the stable point nobody wants to leave

A *Nash equilibrium* (named after John Nash) is a combination of strategies — one per player — where no player can do better by *unilaterally* changing their own strategy, holding everyone else's fixed. It is the "nobody wants to move" point. In the market-making game, "both quote tight" is the Nash equilibrium: given that the rival quotes tight, your best response is to quote tight, and vice versa. Neither of you can improve by deviating alone.

The reason equilibrium matters to a trader is defensive. If you are playing a strategy that is *not* a best response to what others are doing, a sharp opponent can exploit you. Playing the equilibrium strategy means you cannot be picked off — you may not maximize against a *specific* weak opponent, but you are safe against *every* opponent. On a trading desk that is often exactly what you want: a strategy that is robust to whoever shows up.

### Mixed strategies: when being predictable is the mistake

Sometimes there is no single move that is safe to repeat, because any fixed move can be read and exploited. The fix is a *mixed strategy* — randomizing over your options with specific probabilities so the opponent can never predict you. The clean example is matching pennies.

![In this zero-sum game the equilibrium is a 50-50 mix the opponent cannot exploit](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-6.png)

The matrix is *matching pennies*. You (the Row player) and an opponent (Column) each secretly pick Heads or Tails. If they match, you win **\$1**; if they differ, you lose **\$1**. Look at the cells: there is no dominant move. If you always play Heads, the opponent will always play Tails and beat you every time. Any *pure* (non-random) strategy you pick can be read and punished. The only unexploitable play is to flip Heads and Tails each with probability **0.5**. Against a 50-50 mix, the opponent's expected payoff is the same whatever they do — they are *indifferent* — so they cannot exploit you. That indifference condition is the signature of a mixed-strategy equilibrium: *you mix so that your opponent's choices all yield the same payoff, removing their ability to outguess you.*

This is the single most quoted game-theory idea in trading interviews, because it generalizes directly to poker (you bluff with a specific frequency so opponents cannot profitably always-call or always-fold) and to quoting (you vary your behavior so flow cannot read your inventory). We will compute a full mixed strategy in the interview room below.

### Zero-sum games: your win is exactly their loss

A *zero-sum game* is one where the players' payoffs always add to zero — every dollar you win is a dollar someone else loses. Matching pennies is zero-sum: +\$1 to you is −\$1 to them. Pure speculation between two traders on a price is close to zero-sum (ignoring fees, which make it slightly negative-sum for the participants and positive-sum for the broker). Zero-sum framing is powerful because it forces you to ask the right question: *if I am winning, who is losing, and why are they willing to keep playing?* If you cannot identify the loser and their reason, you may be the loser. The unsettling corollary, attributed to poker, applies to trading too: if you look around the table and cannot spot the sucker, it is you.

Real markets are not perfectly zero-sum — hedgers, indexers, and liquidity-seekers trade for reasons other than predicting the next tick, and that is *why* there is money for skilled players to make. But within a single speculative trade against another speculator, zero-sum is the right mental model, and it keeps you honest about where your edge is supposed to come from.

## Poker-flavored spots: the EV arithmetic of cards

Poker shows up constantly in trader interviews, and not because firms want card sharks. Poker is a clean laboratory for exactly the four lenses: every decision is an EV calculation against an adversary under uncertainty, with information arriving card by card. If you can reason about a poker spot, you can reason about a trade. Three concepts cover most of it.

### Pot odds: the price you are being offered

In poker, the *pot* is the pile of chips in the middle that you win if you take down the hand. When an opponent bets, you must *call* (match their bet to stay in) or *fold* (give up). *Pot odds* are simply the price of that call: how many chips you must put in versus how many you stand to win.

![The break-even equity for a call equals the chips you risk divided by the total pot you win](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-8.png)

Walk the calculation in the figure. The pot is **\$80**. Your opponent's bet means you must call **\$20** to continue. If you call and win, you collect the \$80 that was there *plus* the \$20 you called is part of a total pot you win of **\$100**. The price you are paying is $\$20 / \$100 = 20\%$. So you need to win at least **20%** of the time for the call to break even. Your win probability in poker is called your *equity* — the share of the pot you would win on average if the hand were played out many times. *The one-sentence intuition: the break-even equity for a call is the chips you put in divided by the total pot you would win — beat that number and calling is +EV.*

The general formula, worth committing to memory:

$$\text{break-even equity} = \frac{\text{call}}{\text{pot after your call}} = \frac{C}{P + C},$$

where $C$ is the amount you must call and $P$ is the pot *before* your call. A bigger bet relative to the pot demands more equity to call; a small bet relative to the pot is cheap and you can call with very little. The whole figure-8 chart and this formula say the same thing.

### Bluff-to-value ratio: how often to lie

If you only ever bet when you have a strong hand (a *value* bet), a thinking opponent learns to fold every time you bet, and you never get paid. So you must sometimes bet with a weak hand — a *bluff* — to keep them honest. But how often? Too rarely and they over-fold to your value bets; too often and they profitably call you down. There is an exact balancing frequency, and it comes straight from pot odds.

![A pot-sized bet forces a two-to-one value-to-bluff ratio to keep the opponent indifferent to calling](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-9.png)

Suppose you bet **\$100** into a **\$100** pot — a *pot-sized* bet. Your opponent must call \$100 to win the \$200 that would then be in the pot, so they are getting 2-to-1 and need $100/200 = 33\%$ equity to call a bluff-catcher. To make them *indifferent* — unable to profit by always-calling or always-folding — you want your bluffs to be exactly **33%** of your betting range, i.e. for every **2** value bets you make **1** bluff. That two-to-one value-to-bluff ratio at a pot-sized bet is the canonical answer, and it is the same indifference logic as the mixed strategy in matching pennies: *you randomize between value and bluff at the frequency that strips the opponent of any exploitable response.*

The ratio depends on bet size. A smaller bet lays the caller a better price, so they need less equity to call, which means you can bluff *less* often. A bigger bet lays a worse price and lets you bluff more. The unifying idea: your bluff frequency is tuned to make the opponent's call exactly break-even.

### The EV of a call, end to end

Put pot odds and equity together and you can compute the dollar EV of any call directly.

![A call is +EV when equity times the pot won beats loss probability times the stake](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-11.png)

The figure works a complete example. You must call **\$20** into an **\$80** pot, and you estimate your equity at **30%**. There are two outcomes:

- You **win** (probability 0.30): you collect the **\$80** pot, a gain of **+\$80**.
- You **lose** (probability 0.70): you forfeit the **\$20** you called, a loss of **−\$20**.

$$\text{EV of the call} = 0.30 \times (+\$80) + 0.70 \times (-\$20) = \$24 - \$14 = +\$10.$$

The call is worth **+\$10**, so you make it. Notice this matches the pot-odds shortcut: you needed 20% equity to break even and you had 30%, so the call is clearly profitable — and now we can see *exactly how* profitable in dollars. *The one-sentence intuition: the EV of a call is your equity times the dollars you win minus the chance you lose times the dollars you risk; whenever that is positive, call.*

This is precisely how a trader evaluates taking a trade: probability of the move times the profit if it happens, minus the probability of the adverse move times the loss if it does. Same arithmetic, different table.

## Auctions and the winner's curse

The last set piece is the *winner's curse*, an idea so counterintuitive that it humbles even experienced bidders — and it bites traders directly, because executing a trade is a kind of auction.

### The setup: a common value and noisy guesses

Imagine an item with a single true value that nobody knows exactly — a jar of coins, an oil field, a company being acquired. Each bidder forms an independent estimate, and because nobody has perfect information, those estimates scatter around the true value: some too high, some too low. In a *sealed-bid* auction, everyone submits one bid and the highest bid wins.

![Across many noisy bids, the highest bidder is the one who overshot the true value most](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-10.png)

Here is the trap, laid out in the figure. The true value is **\$100**, unknown. Bidder A guesses \$80 (underbid), bidder B guesses \$100 (on the nose), bidder C guesses \$160 (a big overshoot). Who wins? Bidder C — the one who overshot the most. The winner of a common-value auction is *systematically* the bidder with the most over-optimistic estimate, because that is precisely the estimate that produces the highest bid. So winning is bad news: it tells you that you, of everyone, were the most wrong in the expensive direction. C "wins" the \$100 item for **\$160** and is **\$60** poorer for it. *The one-sentence intuition: winning a common-value auction means you out-bid everyone, which usually means you overestimated the prize — so the act of winning is itself evidence you overpaid.*

### Shading your bid: the cure

The fix is to *shade* your bid — deliberately bid below your own estimate to correct for the curse. The right amount of shading grows with the number of competitors (more bidders means the winner overshot by more) and with the noise in everyone's estimates (more uncertainty means bigger overshoots). A disciplined bidder asks not "what do I think it is worth?" but "what is it worth *given that my bid is the highest one*?" — and that conditional value is always lower. Sophisticated traders apply exactly this when they get *filled* on a passive order: getting filled often means the market moved against you, so the very event of winning the trade carries bad news. That is *adverse selection*, and the cure is the same as for the winner's curse — price in the fact that being chosen is evidence you were on the wrong side.

## In the interview room

Now we put it together. Below are five interview-style problems of exactly the kind these firms ask, each solved fully with the dollar arithmetic shown. Practice saying the reasoning out loud — the interviewer cares about your process at least as much as your number.

### Problem 1 (worked example) — a $ bet you should or shouldn't take

> *"I'll roll a fair die. If it comes up 6, I pay you \$120. Otherwise, you pay me \$20. Do you want to play?"*

**Solution.** Compute the EV from your perspective.

- You win **\$120** with probability $\frac{1}{6}$.
- You lose **\$20** with probability $\frac{5}{6}$.

$$\text{EV} = \tfrac{1}{6}(+120) + \tfrac{5}{6}(-20) = \tfrac{120}{6} - \tfrac{100}{6} = \tfrac{20}{6} \approx +\$3.33.$$

The bet is **+\$3.33** per roll, so **yes, take it** — and you would happily play it thousands of times. Then add the trader's flourish: *"The variance is moderate — I lose \$20 most rolls and occasionally win \$120 — so I'd be comfortable sizing this up, as long as \$20 is small relative to my bankroll. If \$20 were a meaningful fraction of everything I had, I'd want to play it many times at small size rather than bet the farm on one roll."* That sentence shows you respect variance, not just EV, which is what separates a good answer from a great one.

A common interviewer follow-up: *"What's the most you'd pay to play once?"* You should be willing to pay up to your EV, **\$3.33**, to play a single round — above that the bet turns −EV. But if they let you play repeatedly, the answer changes, because repetition lets the edge compound while the variance averages out.

### Problem 2 — whether to pay $X for a clue

> *"You're about to make a bet that's worth +\$2 to you on average. For \$5, I'll sell you a perfect signal that tells you in advance whether the bet will win (in which case it pays +\$12) or lose (−\$8). The two are equally likely. Do you buy the signal?"*

**Solution.** First, the value *without* the signal: you bet because +\$2 beats not betting, so $V_{\text{without}} = +\$2$.

Now the value *with* a perfect signal — it lets you bet only in the good world and fold in the bad world:

- Signal says WIN (probability 0.5): you bet, make **+\$12**.
- Signal says LOSE (probability 0.5): you fold, make **\$0** (dodging the −\$8).

$$V_{\text{with}} = 0.5(+\$12) + 0.5(\$0) = +\$6.$$

The information improves your expected payoff from **\$2** to **\$6**, a gain of **+\$4**. The signal costs **\$5**. Since $\$4 < \$5$, the signal is **not worth buying** — you would pay \$5 to gain only \$4 of decision value, a net **−\$1**. *The discipline:* the value of information is the improvement in your decisions, and here that improvement ($\$4$) falls short of the price ($\$5$). Had the bad-world loss been larger — say −\$20 instead of −\$8 — folding would save more, the information would be worth more, and the answer would flip. Always tie the value of a clue to the dollars of decision-change it buys.

### Problem 3 (worked example) — solve a 2×2 zero-sum game for the mixed strategy

> *"We play a game. We each secretly pick Left or Right. The payoff to me (you pay me) is: if we both pick Left, you pay me \$3; both Right, you pay me \$1; if we mismatch, I pay you \$2. What's your optimal strategy, and what's the game worth?"*

**Solution.** This is a zero-sum game; let us find your (the Column player's) mixing probabilities so that the opponent (Row) is indifferent between Left and Right, which is what makes you unexploitable. Let $q$ be your probability of playing Left, so $1-q$ is your probability of Right. The payoffs *to the opponent* (positive = you pay them) are: both Left $+3$, both Right $+1$, mismatch $-2$.

Opponent's expected payoff if **they** play Left:
$$E_{\text{Row=Left}} = q(+3) + (1-q)(-2) = 3q - 2 + 2q = 5q - 2.$$

Opponent's expected payoff if **they** play Right:
$$E_{\text{Row=Right}} = q(-2) + (1-q)(+1) = -2q + 1 - q = 1 - 3q.$$

Set them equal to make the opponent indifferent:
$$5q - 2 = 1 - 3q \;\Rightarrow\; 8q = 3 \;\Rightarrow\; q = \tfrac{3}{8}.$$

So you play **Left with probability 3/8** and **Right with probability 5/8**. The game's value to the opponent is $5q - 2 = 5(\tfrac{3}{8}) - 2 = \tfrac{15}{8} - 2 = -\tfrac{1}{8}$. A *negative* value to the opponent means a *positive* value to **you** of **+\$0.125 per round** — by mixing 3/8 versus 5/8, you actually make about **12.5 cents** every time you play, and no counter-strategy of theirs can stop it. The full answer names the mix *and* the value, and notes that any deviation by them is punished. That is a complete, desk-quality solution.

### Problem 4 — a pot-odds call

> *"There's \$60 in the pot. Your opponent shoves \$30. You figure you'll win 35% of the time. Call or fold?"*

**Solution.** Two equivalent routes; show both to demonstrate fluency.

*Pot-odds route.* You must call **\$30** to win a final pot of $\$60 + \$30 = \$90$. Break-even equity is $\$30 / \$90 = 33.3\%$. You have **35%**, which clears the bar, so **call**.

*Direct EV route.* 

- Win (35%): you collect the \$60 pot, **+\$60**.
- Lose (65%): you lose your \$30 call, **−\$30**.

$$\text{EV} = 0.35(+\$60) + 0.65(-\$30) = \$21 - \$19.50 = +\$1.50.$$

The call is **+\$1.50**, a thin but real profit, confirming the pot-odds verdict. The good answer is "call"; the great answer adds: *"It's a thin call — I only beat the break-even threshold by about 1.7 percentage points — so if I had any read that my equity estimate was optimistic, or that worse outcomes lurk on later streets, I'd lean toward folding."* That shows you understand the margin of safety, the poker analog of not sizing a thin-edge trade too aggressively.

### Problem 5 (worked example) — a sealed-bid auction and the winner's curse

> *"Ten of us each privately estimate the value of a jar of coins and submit a sealed bid; high bid wins and pays their bid. The true value is \$100, and each person's estimate is the true value plus an independent error that's equally likely to be anywhere from −\$40 to +\$40. If you bid your own estimate, what happens — and what should you do instead?"*

**Solution.** First, the curse. Each of the 10 estimates is \$100 plus a uniform error in $[-40, +40]$. The *winner* is whoever has the highest estimate, and with 10 independent draws the maximum error is large and positive — on average the highest of 10 uniform $[-40,+40]$ draws sits near the top of the range, around **+\$33** (the expected maximum of $n$ uniform draws on $[a,b]$ is $a + \frac{n}{n+1}(b-a)$, here $-40 + \frac{10}{11}(80) \approx +33$). So the winning estimate is around **\$133**, and if you bid your estimate you pay roughly **\$133** for a **\$100** jar — a **\$33** loss. That is the winner's curse in dollars: *bidding your honest estimate is a losing strategy precisely because winning selects for the largest overshoot.*

What to do instead: **shade your bid down** to account for the information that winning conveys. Condition on the event "my bid is the highest of 10." Given that you win, your estimate is very likely near the top of the error distribution, so the value *given that you win* is well below your raw estimate. The right bid corrects for that selection — roughly, you should bid as if your estimate already overshot by about the expected winning margin, shading down on the order of **\$30** in this setup so that *if* you win, you win at around the true \$100 rather than above it. The headline you want to deliver: *"I would never bid my raw estimate in a common-value auction with many bidders; winning is bad news about my estimate, so I shade my bid below it, and I shade more the more competitors there are and the noisier the estimates."* That single sentence demonstrates the entire concept, which is what the interviewer is listening for.

## Common misconceptions

These are the beliefs that feel right and lose money. Each is corrected with the *why*.

**"Maximize EV and you're done."** EV is necessary but not sufficient. A +EV strategy bet too large can still go broke through variance before the edge pays off — that is *risk of ruin*. The right objective is to maximize long-run growth, which means maximizing EV *subject to surviving the swings*, i.e. sizing by edge relative to variance (the Kelly logic). Anyone who tells you to "just bet more on your best ideas" without mentioning size is describing a path to zero.

**"A bet that wins most of the time is a good bet."** Frequency of winning and expected value are different things. A bet that wins 99% of the time but occasionally loses a fortune can be deeply −EV; a bet that loses most of the time but wins big when it hits can be strongly +EV. Always weight the *outcomes by their sizes*, not just count the wins. This is the single most common trap interviewers set.

**"I've already put \$X in, so I should keep going."** The *sunk-cost fallacy*. Money already spent is gone and irrelevant to the decision in front of you — only the *future* costs and payoffs matter. In poker, chips already in the pot are not "yours to protect"; the only question is whether *this* call, at *these* odds, is +EV. In trading, the price you paid for a position is irrelevant to whether you should hold it now; only its forward EV matters. Letting a losing position ride because "I'm down so much already" is sunk cost in a suit.

**"The odds are fixed, so I just compute and bet."** In a casino, yes. In a market or at a poker table, no — there is an adversary adjusting to you. If you play a predictable, exploitable strategy, a sharp opponent will counter it and your "computed EV" evaporates. Against thinking opponents you often need a *mixed* strategy and you must reason about *their* incentives, not just the cards. Ignoring the adversary is the most expensive blind spot a quant can have.

**"Winning the trade means I was right."** The winner's curse and adverse selection say otherwise. Getting filled on a passive order, or winning an auction, often means the world moved against you — you were *chosen* precisely because you offered the best (to them) price, which is evidence you mispriced. Treat being selected as information, usually bad, and price accordingly.

**"More information is always worth having."** Only information that *changes a decision* has value, and only up to the dollar improvement it produces. Paying for a signal that confirms what you would do anyway is pure loss. The discipline is to ask, before you pay: "what would I do differently if I knew this, and how much is that difference worth?"

## How it shows up on a real trading desk

The four lenses are not just interview theater — they are the day-to-day operating system of a market-making or proprietary trading desk. Here is how each one cashes out in practice.

![A disciplined desk sizes by edge over variance and shrinks size as variance and adverse selection rise](/imgs/blogs/decision-making-under-uncertainty-quant-interviews-12.png)

**Position sizing is the EV-versus-variance lens, live.** A market maker quotes thousands of names and must decide how much risk to take in each. The disciplined approach sizes each position to its edge *divided by* its variance — bigger when the edge is large and the outcome is predictable, smaller when the edge is thin or the swings are wild. The figure contrasts the two cultures: the gut sizer puts on a big position because they feel good about it, ignores variance, and eventually meets the one bad streak that ruins them; the disciplined desk scales to edge over variance, cuts size when conditions turn adverse, and survives to compound. Over a career the difference is not subtle — it is the difference between still being in the game and not.

**Edge versus variance decides which strategies even get funded.** A desk with a small but extremely consistent edge (low variance) can lever it up and run it large; a desk with a big but wildly variable edge must run small and may not clear the firm's risk limits at all. Two strategies with identical expected returns get completely different capital allocations based on their variance — exactly the EV-versus-variance tradeoff from the scatter plot, applied to whole books of risk. This is why firms obsess over the *Sharpe ratio*, which is essentially edge divided by variance: it is the quantity that tells you how large you can safely run.

**Adverse selection is the winner's curse, every fill, all day.** A market maker posting passive quotes faces a brutal version of the winner's curse: the orders that hit your quote are disproportionately the ones where the informed trader knows something you don't, so your fills are adversely selected. The defense is to treat every fill as information — if you are getting filled a lot on one side, the market is probably about to move against you, and you should widen your quotes or skew them. Pricing in the bad news of being chosen is the same shading move as in the auction, and a maker who doesn't do it gets run over by informed flow.

**Game theory governs how you quote against other makers.** When several firms make markets in the same instrument, they are playing the spread-quoting game from the dominant-strategy figure — competition grinds spreads toward the equilibrium, and each maker must decide how tight to quote against rivals who are adjusting in real time. Against predictable competitors you can sometimes exploit a pattern; against sharp ones you play closer to the unexploitable equilibrium so you cannot be picked off. And against informed order flow you randomize aspects of your behavior — a mixed strategy — so that nobody can reverse-engineer your inventory or your next move from your quotes.

**Updating is the whole job in a fast market.** Information arrives tick by tick — a print here, a size order there, a move in a correlated name — and the maker who updates their fair value fastest and most correctly wins. This is Bayesian updating in miniature, run thousands of times a second, and it is exactly the value-of-information lens: every new data point is worth acting on only insofar as it changes your estimate of where the price should be. The firms that win are the ones whose models update correctly and whose traders don't anchor on stale prices or sunk costs.

**A real episode: the 2010 flash crash.** On May 6, 2010, US equity indices plunged about 9% and recovered within minutes. Part of the mechanism was market makers, facing a sudden flood of adversely-selected order flow they couldn't price, doing the rational thing under the winner's curse: they widened quotes drastically or pulled them entirely rather than keep getting picked off at stale prices. Liquidity evaporated, and the price gapped. The lesson is pure decision-theory: when being filled becomes overwhelmingly bad news, the +EV move is to stop quoting tight — exactly the shading-to-the-extreme that the winner's curse predicts. Understanding *why* makers pulled back is understanding the winner's curse in its most dramatic form.

## When this matters to you and further reading

The reason firms test this material so relentlessly is that it is not separable from the job — it *is* the job. A trader who computes EV but ignores variance blows up; one who respects variance but never reasons about the adversary gets picked off; one who does both but won't pay for the right information trades blind. The four lenses compound, and the interview is checking whether they are reflexes for you yet.

If you want to go deeper, three books map onto the three hardest lenses. For the *decision-making and updating* lens, Annie Duke's *Thinking in Bets* reframes every choice as a bet under uncertainty and is the cleanest popular treatment of separating decision quality from outcome luck — essential for not punishing yourself for a good bet that lost. For the *poker and adversary* lens, David Sklansky's *The Theory of Poker* is the canonical text on pot odds, implied odds, bluff frequencies, and thinking one level above your opponent; everything in our poker section is a simplified slice of it. For the *game theory* lens, any solid introductory game-theory text — for instance an undergraduate primer that builds dominant strategies, Nash equilibrium, and mixed strategies from first principles — turns the intuitions here into tools you can apply to new games on the spot.

And to round out the quantitative spine of decision-making under uncertainty, pair this with the companion posts in this series: [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) for the math of optimal bet sizing and why overbetting is self-defeating, [expected value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) for the linearity, indicator, and symmetry tricks that compute EV in a single line, and [conditional probability and Bayes for quant interviews](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) for the updating machinery behind the value of information. Together they cover the four lenses end to end — and they are, not coincidentally, the four things a trader interview is really asking about.
