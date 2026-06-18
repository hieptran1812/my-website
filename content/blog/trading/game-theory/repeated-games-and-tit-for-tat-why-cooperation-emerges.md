---
title: "Repeated Games and Tit-for-Tat: Why Cooperation Emerges"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "The one-shot prisoner's dilemma says defect, but most market relationships repeat, and the shadow of the future makes cooperation the rational choice once you value tomorrow above a precise discount-factor threshold."
tags: ["game-theory", "repeated-games", "tit-for-tat", "trading", "cooperation", "reputation", "prisoners-dilemma", "discount-factor", "folk-theorem", "dealer-markets", "axelrod", "backward-induction"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The one-shot prisoner's dilemma says betray your counterparty; but almost every relationship in markets is *repeated*, and repetition changes the answer completely. When you will face the same dealer, broker, or trading partner again and again, the discounted value of all your future deals together becomes a bond you post against bad behavior. Defect once and you forfeit the whole stream. So cooperation stops being naive and becomes the rational move — *if* you value the future enough.
>
> - In a one-shot game, **defection is a dominant strategy** and the only Nash equilibrium is mutual betrayal. Repeat the game forever and **cooperation becomes a subgame-perfect equilibrium** — backed by the threat to punish any defection.
> - The exact condition is a discount factor: cooperation is sustainable if and only if $\delta \ge (T-R)/(T-P)$, where $T$ is the temptation to defect, $R$ the reward for mutual cooperation, and $P$ the punishment for mutual defection. For the canonical game ($T=5, R=3, P=1, S=0$) the threshold is $\delta^\* = 0.5$.
> - Robert Axelrod's tournament crowned **tit-for-tat** — start nice, then copy the opponent's last move — and it won not by being clever but by being *nice, retaliatory, forgiving, and clear*.
> - **The rule to remember:** the glue holding markets together is not goodwill, it is *continuation value*. A repeat counterparty behaves; an anonymous one defects. Know which game you are in, because you will be priced accordingly.

In the over-the-counter world of bond and swap trading, there is no exchange and no central order book. When a hedge fund wants to buy \$50 million of a corporate bond, it does not click "buy." It calls a dealer — a bank that holds inventory and quotes a price to buy or sell — and asks for a quote. The dealer could, on any single trade, quote a slightly worse price than the bond is really worth, pocket the difference, and the fund might never know. The fund, for its part, could lie about the size it really wants, or take the dealer's tight quote and then trade the rest of its order with a competitor, leaving the first dealer holding a bad position. Both sides have a knife at the other's throat on every single trade. And yet, day after day, decade after decade, these trades happen at fair prices, with a handshake's worth of trust, between people who could rob each other and mostly don't.

Why? Not because dealers and funds are nice. It is because they will talk again *tomorrow*, and the day after, and for years. The fund that gets cheated remembers and routes its next hundred trades elsewhere. The dealer who shades a price loses a client worth millions in future commissions. Each side is sitting on a pile of *future business* that good behavior protects and one act of betrayal would destroy. The relationship itself — the expectation of repeat dealing — is the collateral. This is the single most important idea in the strategic study of markets, and it is the reason the prisoner's dilemma, which by itself predicts universal betrayal, does not actually describe how most of the financial world behaves.

The diagram below is the mental model for the whole post: the same game, played once versus played forever, gives opposite answers. On the left, the one-shot logic that says defect. On the right, the shadow of the future that flips it to cooperate.

![One-shot prisoner's dilemma defection versus repeated prisoner's dilemma cooperation driven by the shadow of the future](/imgs/blogs/repeated-games-and-tit-for-tat-why-cooperation-emerges-1.png)

## Foundations: repeated games, the shadow of the future, and the discount factor

Let's build this from absolute zero. We will need four ideas, and we will define each one before we use it: the **stage game**, the **repeated game**, the **shadow of the future**, and the **discount factor**. If you have read the companion piece on [the prisoner's dilemma in markets](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once), you already have the first one; we will recap it quickly and then add the thing that changes everything.

### The stage game: one round of the prisoner's dilemma

A **game**, in the technical sense, is a situation with players, the choices (called *strategies*) each can make, and a *payoff* — a number measuring how good each outcome is for each player. The **prisoner's dilemma** is the most famous game ever written down. Two players each choose, simultaneously and without communicating, to either **cooperate** or **defect**. The payoffs follow one rule: defecting is better for you no matter what the other player does, yet if *both* of you defect you both end up worse than if both had cooperated.

To make this concrete we attach numbers. The standard symbols are $T$, $R$, $P$, and $S$:

- $R$ is the **reward** for mutual cooperation. Both behave; both do well.
- $T$ is the **temptation** to defect while the other cooperates. You betray a trusting partner and grab the most.
- $P$ is the **punishment** for mutual defection. Both betray; both do badly.
- $S$ is the **sucker's payoff** — you cooperate, they defect, and you get the worst outcome.

A prisoner's dilemma is *defined* by the ranking $T > R > P > S$. Temptation beats reward beats punishment beats sucker. Throughout this post I will use the standard numeric example: $T = 5$, $R = 3$, $P = 1$, $S = 0$, where the numbers are dollars of profit per round. You will see these four numbers again and again — memorize them and the rest is arithmetic.

This single round — one simultaneous choice each, one payoff each — is called the **stage game**. It is the building block. Played exactly once, its analysis is brutal and short: defecting earns you more than cooperating whether the other player cooperates ($5 > 3$) or defects ($1 > 0$). So defection is what game theorists call a **dominant strategy** — a move that is your best response to *everything* the other side could do. Both players reason this way, both defect, and they land in the $(P, P) = (\$1, \$1)$ box, having thrown away the $(R, R) = (\$3, \$3)$ they could have shared. That mutual-defection box is the unique **Nash equilibrium** of the stage game: a combination of choices where neither player can do better by changing only their own move. (For the full treatment of Nash equilibrium as the price at which a strategic standoff settles, see [Nash equilibrium, best response, and the price as a truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce).)

That is the whole one-shot story, and it is bleak: rational players betray each other and both lose. Now we change one thing.

### The repeated game: the same players meet again

A **repeated game** is the stage game played over and over by the same players, each able to see what the other did in every previous round. That second clause is the magic. In a one-shot game your move has no consequences beyond this round, because there is no next round. In a repeated game your move today shapes how your opponent treats you tomorrow. If you betray them now, they can betray you back forever. Suddenly betrayal has a price, and the price is paid in the currency of *future payoffs*.

The repeated game can be *finite* (exactly 100 rounds, then it stops) or *infinite/indefinite* (it could end any round, but you never know which is the last). That distinction turns out to matter enormously, and we will spend a whole section on why a *known* final round destroys cooperation while an *uncertain* horizon preserves it.

A **strategy** in a repeated game is no longer a single move — it is a rule that says what to do in each round *as a function of the history so far*. "Always cooperate" is a strategy. "Always defect" is a strategy. "Cooperate until they defect, then defect forever" is a strategy (it has a name — grim-trigger — and we will meet it). "Cooperate, then copy whatever they did last round" is a strategy too: that is **tit-for-tat**, the hero of this post.

### The shadow of the future

Here is the phrase that holds the whole theory together. The **shadow of the future** is the influence that the prospect of future interaction casts back onto your behavior today. When the future is long and matters to you, that shadow is dark and heavy: you behave today because misbehaving costs you a long stream of tomorrows. When the future is short or you barely care about it, the shadow is faint, and you grab what you can now.

Cooperation in markets — fair dealer quotes, honored verbal agreements, brokers who don't front-run your order — survives almost entirely because of this shadow. The dealer treats you fairly today not out of virtue but because cheating you forfeits years of your future order flow. Lengthen the relationship and you strengthen the shadow; make the counterparty anonymous and one-shot, and the shadow vanishes and so does the cooperation. This is why, as we will see, retail and anonymous flow is treated worse than a named repeat client: it has no shadow to discipline it.

Here is an everyday version that makes the shadow tangible. Compare the corner shop you walk past every day with a roadside stall at a tourist trap you will never return to. The corner shop will not overcharge you or sell you a bad apple, because it wants you back tomorrow and the day after for years — its shadow of the future is long, so it cooperates. The tourist stall, dealing with strangers who will never come back, has every incentive to overcharge and sell the bruised fruit, because there is no tomorrow with you — its shadow is gone, so it defects. Same product, same human nature, opposite behavior, and the only thing that changed is whether the seller expects to see you again. Markets are exactly this, scaled up to billions of dollars: the corner-shop counterparties deal fairly and the tourist-stall counterparties don't, and your job is to know which kind you are standing in front of.

### The discount factor: how much is the future worth?

To do the math we need to make "value the future" precise with a single number, $\delta$ (the Greek letter delta), called the **discount factor**. It is a number between 0 and 1 that says how much a dollar of payoff *one round from now* is worth to you *today*.

If $\delta = 0$, you value only this round; the future is worthless to you and you might as well be playing a one-shot game. If $\delta$ is close to 1, you value next round's dollar almost as much as this round's — you are patient, the relationship is long, and the future weighs heavily. A dollar two rounds out is worth $\delta^2$ today, three rounds out $\delta^3$, and so on.

Two real-world forces set your $\delta$. The first is genuine patience or interest rates: a dollar next year is worth less than a dollar today, and if the prevailing interest rate is $r$ then $\delta = 1/(1+r)$. The second, and usually the bigger one in markets, is the *probability the relationship continues*. If there is a chance $p$ that this is the last round you will ever play — the fund goes out of business, the desk gets shut down, you move to a competitor — then even a perfectly patient player effectively discounts the future by the survival probability. Combine them and $\delta$ is roughly the interest-rate discount times the continuation probability. A long, stable relationship between two firms that expect to keep trading has a high $\delta$. A one-night-stand trade with a stranger you'll never see again has $\delta \approx 0$.

The whole question of whether cooperation can survive comes down to one comparison: is the discounted stream of future cooperation worth more than the one-time temptation of defecting? We are about to compute exactly when the answer is yes.

#### Worked example: turning a continuation probability into a discount factor

Let me make the survival-probability idea concrete, because it is the version of $\delta$ that matters most in markets and the one people get wrong. Suppose you and a counterparty trade once a month, and you are perfectly patient about money — you do not discount future dollars for interest at all. But each month there is a 20% chance the relationship ends for reasons outside anyone's control: one of you changes jobs, the desk gets restructured, the mandate changes. So each month the relationship continues with probability $p = 0.80$.

Your effective discount factor is just that continuation probability: $\delta = 0.80$. A dollar of cooperative profit "next month" is worth 80 cents to you today, not because you are impatient but because there is only an 80% chance you will be around to collect it. Two months out it is worth $0.80^2 = 0.64$ of a dollar; three months out $0.80^3 = 0.51$. The relationship has a kind of half-life: after about three months, more likely than not it has ended.

Now run it through our threshold. With the canonical payoffs the cooperation threshold is $\delta^\* = 0.5$, and your $\delta = 0.80$ comfortably clears it — so cooperation is sustainable, and you should expect a fair deal. But drop the monthly continuation probability to 40% (a shaky, probably-ending relationship) and $\delta = 0.40$ falls *below* the 0.5 threshold: now even an honest counterparty's incentive to cooperate is gone, and you should price the deal as a near one-shot. The intuition: you do not need to be impatient for the future to stop disciplining behavior — you only need the relationship to be probably-ending, and a counterparty whose survival is in doubt is a counterparty whose $\delta$ has already collapsed.

## The folk theorem: when cooperation becomes rational

Now the payoff. We will derive the precise condition under which mutual cooperation is sustainable in the repeated prisoner's dilemma — a result so foundational that it has a slightly mythic name, the **folk theorem** (so called because it circulated as folklore among game theorists before anyone formally published it). The version we need is simple enough to derive in a few lines, and it gives us a sharp, usable number.

### Grim-trigger: the simplest threat that works

Suppose both players adopt a strategy called **grim-trigger**: cooperate every round, but the *instant* the other player defects even once, defect forever after, with no forgiveness. It is grim because the punishment is permanent and total. The question is: given that my opponent plays grim-trigger, is it in *my* interest to cooperate, or should I defect to grab the temptation $T$?

Let's compare my two options as discounted sums of all future payoffs.

**If I cooperate forever** (and so does my grim-trigger opponent), I earn the reward $R$ every round, forever. The discounted value is:

$$V_{\text{coop}} = R + \delta R + \delta^2 R + \dots = \frac{R}{1 - \delta}$$

That last step uses the formula for an infinite geometric series — a stream of $R$ per round, each round discounted one more power of $\delta$, sums to $R/(1-\delta)$. With $R = 3$ and, say, $\delta = 0.8$, that is $3/(1-0.8) = 3/0.2 = \$15$.

**If I defect once** to grab the temptation, my opponent's grim-trigger fires and we both defect forever after. So I get $T$ this round, then the punishment $P$ in every round thereafter:

$$V_{\text{defect}} = T + \delta P + \delta^2 P + \dots = T + \frac{\delta P}{1 - \delta}$$

With $T = 5$, $P = 1$, and $\delta = 0.8$, that is $5 + (0.8 \times 1)/(0.2) = 5 + 4 = \$9$.

Cooperation pays \$15, defection pays \$9. So at $\delta = 0.8$ I should cooperate. Defection's one big grab of $T$ is swamped by the long, sad tail of $P$s I condemn myself to.

### The threshold: solving for delta

Cooperation is rational exactly when $V_{\text{coop}} \ge V_{\text{defect}}$:

$$\frac{R}{1 - \delta} \ge T + \frac{\delta P}{1 - \delta}$$

Multiply both sides by $(1 - \delta)$ and solve for $\delta$. The algebra collapses to a clean, famous result:

$$\delta \ge \frac{T - R}{T - P}$$

This is the **cooperation threshold**, $\delta^\*$. As long as your discount factor clears this bar — as long as you value the future at least this much — cooperation backed by the grim-trigger threat is a stable equilibrium. Below it, you are too impatient, the temptation wins, and the relationship collapses into mutual defection. The threshold is computed in code by `data_gametheory.repeated_pd_delta_threshold(T, R, P, S)`, which returns exactly $(T-R)/(T-P)$.

Read the formula's intuition straight off the symbols. The numerator $T - R$ is the *extra* you get from defecting on a cooperator instead of cooperating — the size of the temptation. The denominator $T - P$ is the *whole* gap between the best one-shot grab ($T$) and the punishment you fall into ($P$). The bigger the temptation relative to that whole gap, the more patience you need to resist it. A huge temptation demands a very high $\delta$; a small one is easy to resist even if you are fairly impatient.

The chart below traces $\delta^\*$ as the temptation $T$ grows, holding $R=3, P=1, S=0$ fixed. The curve rises: the greater the prize for betraying a trusting partner, the more you must value the future for cooperation to hold. At our canonical $T = 5$ the threshold sits at exactly one-half.

![Discount-factor threshold for sustainable cooperation rising with the temptation to defect](/imgs/blogs/repeated-games-and-tit-for-tat-why-cooperation-emerges-2.png)

#### Worked example: the exact delta threshold for the canonical game

Let's compute the threshold for our running example and confirm it by hand. The payoffs are $T = \$5$, $R = \$3$, $P = \$1$, $S = \$0$.

The formula gives:

$$\delta^\* = \frac{T - R}{T - P} = \frac{5 - 3}{5 - 1} = \frac{2}{4} = 0.5$$

So cooperation is sustainable if and only if $\delta \ge 0.5$ — you must value next round's dollar at least half as much as this round's. Running `repeated_pd_delta_threshold(5, 3, 1, 0)` returns `0.5`, matching to the penny.

Now let's verify it sits exactly at the break-even point by computing both lifetime values *at* $\delta = 0.5$. Cooperate forever: $V_{\text{coop}} = R/(1-\delta) = 3/0.5 = \$6$. Defect once: $V_{\text{defect}} = T + \delta P/(1-\delta) = 5 + (0.5 \times 1)/0.5 = 5 + 1 = \$6$. They are equal — \$6 each — which is precisely what "threshold" means. Just above $\delta = 0.5$, cooperation pulls ahead; just below, defection does.

Now raise the temptation to $T = \$10$ (a juicier betrayal) and the threshold jumps: $(10 - 3)/(10 - 1) = 7/9 \approx 0.78$. You now need to value the future far more — a $\delta$ of at least 0.78 — to keep cooperation alive. The intuition: the bigger the one-time prize for cheating, the more future you have to be protecting before cheating stops looking worth it.

### Why this is the "folk theorem" in one sentence

The general folk theorem says something even stronger: in an infinitely repeated game with patient enough players, *almost any* outcome that gives each player more than they could guarantee themselves by defecting can be supported as an equilibrium. Cooperation is just the most useful special case. The deep message is that **repetition vastly expands what is possible**. A game that, played once, has exactly one grim outcome can, played forever, support a whole rainbow of cooperative outcomes — held in place not by goodwill but by the credible threat of punishment. The threat does the work; cooperation is what the threat protects.

One technical point is worth pinning down because it dissolves an objection people always raise: *"but no relationship literally lasts forever, so doesn't the whole thing collapse?"* The answer is that **you do not need an actually infinite game — you need an *uncertain* end.** Suppose after each round the relationship continues with probability $p$ and ends with probability $1 - p$. From inside the game, every round looks the same: there is always, probably, a next round, and you never know which round is the last. Mathematically, this *uncertain-horizon* game behaves exactly like an infinite game with discount factor $\delta = p$ (multiplied by any genuine interest-rate discounting). The continuation probability simply *becomes* the discount factor. So a relationship that will "probably keep going" round after round — which describes essentially every real trading relationship — gets the full cooperative force of the folk theorem, even though everyone knows it will end someday. What kills cooperation is not the existence of an end; it is a *commonly known, fixed, countable-down-to* end. Markets almost never have one, which is precisely why they cooperate.

## Tit-for-tat: the strategy that won the tournament

Grim-trigger proves cooperation is *possible*, but it is a terrible way to actually play, because it is unforgiving: a single mistake, a single misread, a fat-fingered defection, and you are both condemned to defect forever. Real relationships need a strategy that can recover from accidents. That strategy has a name and a famous origin story.

In 1980, the political scientist **Robert Axelrod** ran a computer tournament. He invited game theorists from around the world to submit strategies for the repeated prisoner's dilemma, and then had every strategy play every other strategy (and a copy of itself) over and over, tallying total scores. People submitted clever, complicated programs — strategies that tried to model the opponent, that defected probabilistically, that set traps. The winner, submitted by the mathematician Anatol Rapoport, was the *shortest* program in the tournament. It was tit-for-tat: **cooperate on the first move, then on every subsequent move simply do whatever your opponent did last round.**

That is the entire algorithm. Be nice to start. Then mirror. If they cooperated, you cooperate. If they defected, you defect — once — and then immediately go back to mirroring, so the moment they return to cooperation, so do you. Axelrod ran the tournament a second time, told everyone tit-for-tat had won and invited them to beat it, and it won *again*. Why does a strategy this simple beat everything thrown at it?

The chart below simulates a round-robin of our three archetypes — tit-for-tat, always-defect, and always-cooperate — against a field of reciprocating strategies, exactly the kind of field Axelrod's tournament contained. Tit-for-tat finishes on top.

![Tit-for-tat versus always-defect versus always-cooperate cumulative tournament scores rising over rounds](/imgs/blogs/repeated-games-and-tit-for-tat-why-cooperation-emerges-3.png)

### The four traits of a winning strategy

Axelrod distilled the success of tit-for-tat into four properties. Every strategy that did well in his tournament had them; every strategy that did badly lacked at least one. They are worth committing to memory because they are exactly the traits that make a market counterparty trustworthy.

![Four traits of tit-for-tat nice retaliatory forgiving and clear](/imgs/blogs/repeated-games-and-tit-for-tat-why-cooperation-emerges-4.png)

- **Nice.** Tit-for-tat never defects first. It opens with cooperation and offers peace. In Axelrod's tournament, the single best predictor of a strategy's score was whether it was nice — every one of the top-ranked strategies started by cooperating. Niceness lets two nice strategies meet and immediately lock into a mutually profitable cooperative stream, harvesting $R$ every round.
- **Retaliatory.** It punishes defection immediately, on the very next move. This is what makes it impossible to exploit: an opponent who tries to grab the temptation $T$ gets the punishment right back next round, so cheating never pays off over time. A *too*-nice strategy (pure always-cooperate) gets eaten alive by defectors; tit-for-tat does not, because it bites back.
- **Forgiving.** The instant the opponent returns to cooperation, tit-for-tat does too. It holds no grudge beyond the single retaliatory move. This is the property grim-trigger lacks, and it is what lets two tit-for-tat players recover from a single accidental defection instead of spiraling into permanent mutual destruction. Forgiveness keeps the relationship's cooperative value alive even after a stumble.
- **Clear.** It is so simple and predictable that an opponent can figure out the rule within a few rounds and learn that cooperation is rewarded and defection is punished. Clarity is underrated: a strategy that is too clever and unpredictable teaches the opponent nothing, so the opponent cannot learn that cooperating with you is safe. Tit-for-tat practically *trains* its opponents to cooperate.

#### Worked example: tit-for-tat versus a one-time defector

Let me walk the payoffs through a concrete six-round encounter so you can feel why being exploitable in the short run is what makes tit-for-tat unbeatable in the long run. Use the canonical payoffs ($T=5, R=3, P=1, S=0$).

Suppose your opponent tries the obvious exploit: cooperate for a while to lull you, then defect once to grab the temptation, then go back to cooperating. Here is the round-by-round play. You are tit-for-tat; they defect on round 3 only.

- Round 1: both cooperate. You +\$3, them +\$3.
- Round 2: both cooperate. You +\$3, them +\$3.
- Round 3: they defect, you (still mirroring round 2's cooperation) cooperate. You +\$0 (sucker), them +\$5 (temptation). They stole \$5 versus your \$3 — a \$2 gain for them this round.
- Round 4: you retaliate by copying their round-3 defection; they (having gone back to cooperating) cooperate. You +\$5 (temptation), them +\$0 (sucker). You take it right back.
- Round 5: both cooperate again. You +\$3, them +\$3.
- Round 6: both cooperate. You +\$3, them +\$3.

Tally it up. Your total: $3+3+0+5+3+3 = \$17$. Their total: $3+3+5+0+3+3 = \$17$. Dead even. Their one-time \$2 steal in round 3 was exactly clawed back by your \$2 surplus in round 4. The exploit gained them *nothing*, and meanwhile you both still captured the \$3-per-round cooperative stream in the other four rounds. The intuition: tit-for-tat lets you be robbed exactly once, then makes the robber give it all back — so nobody can profit from cheating you, and the cooperative relationship survives intact.

### Grim-trigger versus tit-for-tat: a tradeoff

So why ever use grim-trigger? Because the two strategies trade off *credibility of punishment* against *robustness to error*. Grim-trigger makes the most terrifying possible threat — eternal punishment — which means it can sustain cooperation at the lowest possible $\delta$ (it leaves the cheater nothing to hope for). Tit-for-tat's punishment is milder (one round, then forgiveness), so it sometimes needs a slightly higher $\delta$ to deter a determined cheater. But grim-trigger is catastrophically fragile: in a noisy world where moves are occasionally misread or mis-executed, one accidental defection triggers permanent war. Tit-for-tat shrugs off the accident after a single retaliatory round.

In real markets — which are noisy, where a "defection" might just be a quote that came in a hair wide because the dealer's risk system hiccuped — *forgiving* strategies dominate. A dealer who blacklisted you forever for one slightly-off trade would have no clients. The market runs on tit-for-tat's logic: punish proportionally, then give the relationship another chance.

### The noise problem: why even tit-for-tat needs slack

There is a subtle failure mode that only appears once you add realistic noise, and it matters because markets are extremely noisy. Suppose two players are both using tit-for-tat and both intend to cooperate forever — but with some small probability each round, a move gets *misimplemented*: a cooperative intention comes out looking like a defection (the quote was wide because of a system glitch, the fill was bad because of a routing error, the message was misread). What happens?

The first accidental "defection" gets mirrored by the other player as a retaliation next round. But now *that* retaliation looks like an unprovoked defection to the first player, who retaliates back. The two tit-for-tat players fall into an "echo" — an endless alternating sequence of defect-cooperate-defect-cooperate, each one's punishment provoking the other's, neither able to break out because each is faithfully copying the other's last (retaliatory) move. Two strategies that both wanted to cooperate forever end up earning far less than the cooperative reward, all because of one misread move and an inability to forgive a *string* of apparent defections.

This is why, under noise, slightly more generous strategies often beat strict tit-for-tat. **Tit-for-two-tats** — only retaliate after *two* consecutive defections — refuses to overreact to a single accident, breaking the echo before it starts. **Generous tit-for-tat** — retaliate, but occasionally forgive a defection at random — does the same thing probabilistically. The lesson generalizes far beyond the program: in any noisy repeated relationship, a little extra forgiveness is not weakness, it is insurance against a death spiral triggered by an honest mistake. Real trading desks build this in. A counterparty does not get cut off for one bad fill; they get cut off for a *pattern*. The slack is deliberate, and it is what keeps a noisy market from tearing its own relationships apart. The same "be tough but don't tilt on one bad beat" discipline shows up in the prop-trading and poker worlds, where the best players punish exploitation but refuse to let a single bad outcome wreck their strategy — see [the SIG and Susquehanna playbook on poker, game theory, and expected value](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev).

## Why finite and final-period games unravel

Everything above assumed the game goes on indefinitely — that there is always, probably, a next round. What happens if there is a *known last round*? The answer is devastating and worth understanding precisely, because it explains why cooperation breaks down at the end of relationships, near retirements, before bankruptcies, and on the last trade with a counterparty you'll never see again.

### Backward induction: reasoning from the end

Consider a prisoner's dilemma repeated exactly 100 times, and *both players know it stops at round 100*. Think about round 100 itself. It is the last round. There is no round 101, no future, no shadow. So round 100 is, for all strategic purposes, a one-shot prisoner's dilemma — and in a one-shot prisoner's dilemma the dominant strategy is to defect. Both rational players will defect in round 100, and both know the other will.

Now back up to round 99. You would normally cooperate in round 99 to protect the cooperative payoff in round 100. But you have just established that round 100 is a guaranteed mutual defection regardless of what anyone does in round 99 — there is nothing left to protect. So round 99 has no future worth defending either, which makes *it* effectively a one-shot game too. Both players defect in round 99.

Now back up to round 98. By the identical argument — round 99 is already a guaranteed defection, so cooperating in round 98 buys you nothing — both players defect in round 98. And the logic cascades all the way back. This chain of reasoning, working from the final round backward to the first, is called **backward induction**, and it is one of the sharpest and most unsettling tools in game theory. Its conclusion here: in a finitely repeated prisoner's dilemma with a commonly known endpoint, the *only* equilibrium is to defect in every single round, from the very first.

The unsettling part is how a single, distant, *known* endpoint reaches all the way back and poisons the present. The cooperation does not erode gently as you near the end; it is destroyed *now*, at round one, by the mere existence of a known round 100. This is the dark mirror of the shadow of the future: where an *uncertain* future casts a long shadow that disciplines cooperation, a *known, finite* future casts no shadow at all on the final round, and that single shadowless round, working backward, leaves every earlier round shadowless too. It is the strongest possible illustration of why markets work so hard to avoid commonly-known endpoints — why relationships are kept open-ended, why "this is our last deal" is a phrase that makes any seasoned trader nervous. The moment both sides agree the end is in sight, the cooperative equilibrium that took years to build can vanish in a single round of reasoning.

![Backward induction unraveling a finite repeated game from the last round to the first](/imgs/blogs/repeated-games-and-tit-for-tat-why-cooperation-emerges-6.png)

#### Worked example: the value of cooperating versus defecting once

The unraveling is easiest to feel by comparing the lifetime values directly at different patience levels, which is exactly what backward induction is implicitly weighing each round. Take the canonical payoffs and ask: across three discount factors — below, at, and above the threshold — is it worth cooperating, or should I grab the temptation and accept the punishment that follows?

At $\delta = 0.4$ (impatient, short relationship): cooperate-forever is worth $R/(1-\delta) = 3/0.6 = \$5.00$. Defect-once-then-punished is worth $T + \delta P/(1-\delta) = 5 + (0.4 \times 1)/0.6 = 5 + 0.67 = \$5.67$. Defection wins by \$0.67. Below the threshold, cheating pays.

At $\delta = 0.5$ (exactly the threshold): cooperate is worth $3/0.5 = \$6.00$; defect is worth $5 + (0.5 \times 1)/0.5 = \$6.00$. A dead tie, as we computed earlier.

At $\delta = 0.8$ (patient, long relationship): cooperate is worth $3/0.2 = \$15.00$; defect is worth $5 + (0.8 \times 1)/0.2 = 5 + 4 = \$9.00$. Cooperation wins by \$6.00. Above the threshold, loyalty pays handsomely.

The chart below shows all three side by side. The intuition: a finite game with a known end is like permanently sitting at $\delta = 0$ for the last round — no future to protect — and the moment that is true for *any* round, the no-future logic poisons every round before it.

![Discounted lifetime value of cooperating forever versus defecting once at three discount factors](/imgs/blogs/repeated-games-and-tit-for-tat-why-cooperation-emerges-5.png)

### Why real markets don't fully unravel

If backward induction is right, every relationship with a known end should collapse into defection — yet they obviously don't. Three things save cooperation in the real world, and each is important.

First, **the horizon is almost never known.** You rarely know which trade is your last with a counterparty. The relationship "probably continues" each round with some high probability, and as we saw, an uncertain horizon is mathematically equivalent to an infinite one with a discount factor — there is no fixed "last round" to anchor the backward induction. The unraveling argument needs a *commonly known* final round, and markets almost never provide one.

Second, **reputation spills across counterparties.** Even on a genuinely final trade with one client, a dealer who cheats may be cheating their *reputation* with every other client who hears about it. The game is not really two-player; it is one player against a whole market that talks. This is the subject of the sibling post on reputation and relationship trading.

Third, **people aren't perfectly rational backward-induction machines.** Experiments show that real players cooperate well into finitely repeated games and only start defecting near the very end — the "endgame." This bounded rationality, ironically, helps everyone: a little irrational niceness early on is contagious and profitable.

## Common misconceptions

A few beliefs about repeated games and cooperation are widespread and wrong. Each one has cost someone money.

**"Cooperation means being nice, and nice traders get exploited."** No — cooperation in a repeated game is *not* unconditional niceness. Tit-for-tat is nice *and retaliatory*; pure always-cooperate is the strategy that gets exploited, because it never punishes a defector. The whole point of the folk theorem is that cooperation is held in place by a *credible threat to punish*. A trader who is fair to fair counterparties and ruthless to cheats is not naive; they are playing the optimal repeated-game strategy. The naive ones are the always-cooperators who keep dealing with a counterparty that has already burned them.

**"If defection is the Nash equilibrium of the prisoner's dilemma, rational players always defect."** Only in the *one-shot* game. In the repeated game, mutual cooperation is *also* a Nash equilibrium — in fact a subgame-perfect one, the strongest kind — as long as $\delta \ge \delta^\*$. "Rational" does not mean "defect"; it means "do the best you can given the game you are actually in," and the repeated game rewards cooperation. People who quote the prisoner's dilemma to justify cynicism have usually forgotten that almost nothing in life is played exactly once.

**"A longer relationship is always more cooperative."** Length helps only because it usually raises $\delta$ (more future to protect) — but if the temptation $T$ is enormous, even a long relationship can break. The threshold $\delta^\* = (T-R)/(T-P)$ rises with $T$. A counterparty facing a one-time, career-making, relationship-ending temptation (think: a single trade big enough to retire on, or a firm that is about to go under anyway) may defect *despite* years of good behavior, because for them the future has suddenly collapsed to nearly zero. The end of a relationship — a retirement, a wind-down, a bankruptcy — is exactly when the shadow of the future shrinks and the gloves come off.

**"Tit-for-tat is the best strategy, full stop."** It won Axelrod's tournaments, but it is not universally optimal — no single strategy is. In a noisy world where moves are sometimes misread, two pure tit-for-tat players can fall into an "echo" of mutual recrimination (one accidental defection triggers an endless alternation of retaliations). Slightly more forgiving variants (tit-for-two-tats: only retaliate after *two* consecutive defections) sometimes do better under noise. The lesson is the *principles* — nice, retaliatory, forgiving, clear — not the literal one-line program.

**"The folk theorem says cooperation is guaranteed."** It says cooperation is *possible* — a sustainable equilibrium — when players are patient enough. It does *not* say cooperation is the only equilibrium or the one that will happen. Mutual defection remains an equilibrium too. Which one the players land in depends on expectations, history, and whether each believes the other will cooperate. The folk theorem expands the menu of possible outcomes; it does not pick the dish.

**"Repetition only matters between two specific people."** Wrong, and the error is expensive. The repeated game that disciplines a counterparty is often *not* the two of you alone — it is that counterparty against the whole market that talks. A dealer can be on a genuinely final trade with you and still behave, because cheating you damages a reputation worth millions in dealings with everyone *else*. Conversely, a counterparty with no community to answer to — no peers who would hear about a defection, no reputation to lose — can cheat you on a "repeat" trade because the broader shadow that should discipline them is missing. When you assess whether a counterparty will cooperate, do not just ask "will I deal with them again?" Ask "who *else* would hear if they cheated me, and would those people punish them for it?" The denser and more talkative the community around a counterparty, the safer you are, even on a one-off trade. This is the bridge from anonymous repeated play to named reputation, which the next post in the track takes up in full.

## How it shows up in real markets

The repeated game is not a toy. It is the hidden architecture of how almost every relationship-based market actually functions. Here are the concrete places it shows up, with the mechanism made explicit each time.

**Dealer and OTC markets.** In over-the-counter markets for bonds, swaps, and FX, a fund trades repeatedly with a small set of dealers. Each pair is playing an indefinitely repeated game with a high $\delta$. The fund could lie about its order size; the dealer could shade quotes. Both mostly don't, because the discounted value of the relationship — years of two-way flow worth millions in spread and commission — dwarfs any one-trade temptation. Crucially, dealers *track* this: they keep mental and literal scorecards of which clients are "good flow" (informative, fair, sticky) and which are "toxic" (they pick off the dealer's stale quotes and disappear). A client who behaves gets tighter quotes, larger size, and a callback in a crisis; a client who defects gets wider quotes and gets cut off. That is tit-for-tat, run by a trading desk, in dollars. The mechanics of how a market maker prices in the fear of being picked off are the subject of [the market maker's game](/blog/trading/game-theory/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed).

**Cartel discipline.** A cartel like OPEC is a repeated prisoner's dilemma among producers. Each member is tempted to defect — to pump above its quota and grab extra revenue at the high cartel price (the temptation $T$). But if everyone defects, the price collapses and everyone earns the competitive punishment payoff $P$. The cartel holds together only when each member values its future share of cartel profits ($\delta$ high) more than the one-time gain from over-pumping, and only when defection is *detectable* and *punishable* — when the others can see you cheating and credibly threaten a price war. Saudi Arabia has historically played the "swing producer," explicitly threatening to flood the market (the grim-trigger punishment) to discipline quota-breakers, most dramatically in the 2014–2016 and March 2020 price wars. The full Cournot-game treatment of how a cartel's optimal output sits between monopoly and competition is in [cartels, collusion, and the Cournot game](/blog/trading/game-theory/cartels-collusion-and-the-cournot-game-from-opec-to-algorithms).

**Why retail and anonymous flow gets worse prices.** When you trade as an anonymous retail client through an app, you are, from the venue's point of view, a one-shot counterparty with $\delta \approx 0$. There is no relationship to protect, no future flow to harvest, no reputation you carry. The venue and any market maker interacting with your order have no shadow of the future disciplining them to give you the keenest price. Repeat institutional counterparties — who route millions of shares a day and will remember a bad fill — get measurably tighter effective spreads and better treatment precisely because they *can* retaliate by taking their flow elsewhere. The lesson is uncomfortable but important: in a relationship market, anonymity is expensive, because anonymity removes the only thing that makes the other side behave.

**Interbank lending and the freeze of 2008.** Banks lend to each other overnight in a repeated game built entirely on trust and reputation. In normal times the shadow of the future — every bank needs to keep borrowing tomorrow — keeps the market fair and liquid. In September 2008, after Lehman Brothers failed, banks suddenly were not sure their counterparties would *survive to the next round*. The continuation probability collapsed, $\delta$ crashed toward zero, and the repeated game reverted to its one-shot Nash equilibrium: don't lend, hoard cash, defect. The interbank market froze not because bankers became dishonest overnight but because the shadow of the future vanished. Central banks had to step in and become the counterparty of last resort precisely to restore a future worth protecting.

**Trading-floor and prime-brokerage relationships.** Hedge funds and their prime brokers (the banks that lend them money and clear their trades) play a long repeated game. The fund relies on the broker to keep funding it; the broker relies on the fund's continued business and on the fund not blowing up. When a fund is perceived to be near death — when its future suddenly looks finite — brokers race to pull financing and seize collateral first, the run-for-the-exit defection we covered in the crowded-trades and prisoner's-dilemma posts. The repeated cooperation that held all year evaporates the moment the counterparty's horizon goes finite. This is the link between this post and [crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game): cooperation is robust right up until the future ends, and then it isn't.

**Market-maker reciprocity in payment for order flow.** Wholesalers who buy retail order flow compete in a repeated game with the brokers who sell it. A wholesaler who consistently provides good price improvement keeps the relationship and the flow; one who degrades execution quality loses the contract to a rival. The repeated nature of the broker–wholesaler relationship is part of what disciplines execution quality even where the underlying retail trader is anonymous — the *broker* is the repeat counterparty playing the long game on the retail client's behalf, which is one reason routing arrangements matter so much. The routing game itself is unpacked in [maker-taker, payment for order flow, and the routing game](/blog/trading/game-theory/maker-taker-payment-for-order-flow-and-the-routing-game).

**Syndicate and IPO allocation.** When an investment bank underwrites a new stock or bond issue, it allocates the new shares to a recurring set of institutional clients. This is a repeated game with a long horizon: the bank will be doing deals for decades, and the clients want to keep getting allocations of the hot ones. The bank rewards clients who behave — who hold allocations rather than immediately flipping them, who pay attention to the bank's research, who bring it other business — with bigger pieces of the good deals. Clients who defect (flip every hot IPO for a quick profit and disappear) get smaller allocations or get cut from future deals. The whole allocation relationship is a tit-for-tat played in shares: cooperate (be a stable, multi-product client) and you get the good allocations; defect (flip-and-run) and you are punished with worse ones. No contract specifies any of this; the shadow of the future enforces it.

**The handshake market in floor-traded commodities.** Before electronic trading, commodity and futures pits ran almost entirely on verbal agreements — a trader would shout a trade, and it was binding on a nod, with paperwork following hours later. A market that settles millions of dollars on someone's word *should* be a defector's paradise. It wasn't, because the pit was the densest possible repeated game: the same few hundred traders, face to face, every single day, for entire careers. A trader who reneged on a verbal trade — "I never said that" — was committing a visible defection in front of the whole community, and the community's punishment was swift and total: no one would trade with you again, which on a trading floor is a professional death sentence. The continuation value of being a trusted member of the pit dwarfed any single trade you could win by lying, so almost no one did. The pit was a living demonstration that cooperation needs no laws when the repeated game is tight enough and reputation is visible enough.

## The repeated PD payoff matrix and the grim-trigger threat

Before the playbook, let's put the whole structure in one picture. The matrix below is the stage game's payoffs — mutual cooperation at $(\$3, \$3)$, the temptation/sucker corners at $(\$5, \$0)$, the one-shot Nash at $(\$1, \$1)$ — with the third row showing the grim-trigger threat that converts the green cooperate-cooperate cell from a tempting target into a stable equilibrium. The pure Nash of the stage game, computed by `nash_2x2` on these payoffs, is the single mutual-defect cell. Repetition plus the threat is what promotes the green cell to an equilibrium of its own.

![Repeated prisoner's dilemma payoff matrix with the grim-trigger punishment threat and cooperation threshold](/imgs/blogs/repeated-games-and-tit-for-tat-why-cooperation-emerges-7.png)

#### Worked example: pricing the value of a reputation

Let me make the abstract "continuation value" into a number a trader would actually care about, because this is the quantity the whole post is really about. Suppose you are a dealer and a particular client sends you flow that nets you, on average, \$3,000 of profit per week, and you expect the relationship to continue with about 99% probability each week (a stable, happy client). Your effective weekly discount factor from the survival probability alone is $\delta = 0.99$.

The discounted lifetime value of that relationship, treated as a stream of \$3,000 per week, is $V = \$3{,}000 / (1 - 0.99) = \$3{,}000 / 0.01 = \$300{,}000$. That is what the relationship is *worth* to you, today, as a present value. Now suppose on one trade you could cheat the client — shade the quote — for a one-time extra \$5,000 (your temptation). Is it worth it? Cheating risks the \$300,000 relationship for a \$5,000 grab. You would be trading a \$300,000 asset for \$5,000 — a catastrophic deal — *as long as the client would detect the cheat and leave*. That single comparison, \$5,000 temptation versus \$300,000 continuation value, is the entire reason dealers behave. The intuition: a reputation is a financial asset with a present value, and you protect it for exactly the same reason you wouldn't burn down a building you own to grab the cash in the till.

Notice what changes the answer. If the client *wouldn't* detect the cheat, or is leaving anyway next week (their $\delta$ just dropped to near zero), the \$300,000 evaporates and the \$5,000 grab becomes tempting. That is why the dangerous moment in any relationship is the moment one side's future goes finite — and why you should watch your *own* counterparties for signs their horizon is shortening.

## The playbook: how to play it

Here is how to turn the repeated-game lens into decisions at your desk. The single question to ask before any negotiation, quote, or trade is: *what is the other side's discount factor, and do they know mine?*

**Identify which game you are in.** Before you trade, classify the counterparty. Is this a repeat relationship with a long, probable future (high $\delta$ — expect cooperation, and offer it) or a one-shot, anonymous interaction (low $\delta$ — expect defection, and protect yourself)? A repeat dealer will quote you fairly because your future flow disciplines them; an anonymous venue has no such discipline and you should assume the price reflects it. Misclassifying the game is the core error: extending one-shot cynicism to a repeat partner poisons a valuable relationship, while extending repeat-game trust to a one-shot stranger gets you picked off.

**Build and protect your continuation value.** Your reputation as a fair, predictable counterparty is a financial asset with a present value — we just priced one at \$300,000. Treat it that way. Being the client who is informative, fair, and sticky earns you tighter quotes, bigger size, and a callback in a crisis. Every act of sharp dealing spends down that asset. The math says: never grab a temptation $T$ smaller than the continuation value you would forfeit, which in any healthy long relationship means almost never.

**Be nice, retaliatory, forgiving, and clear.** This is tit-for-tat as a personal operating system. Start cooperative. If a counterparty deals with you unfairly, retaliate proportionally and promptly — route the next order elsewhere, widen what you show them — so cheating you never pays. But forgive once they correct: holding an eternal grudge (grim-trigger) is fragile and costly in a noisy market where some "defections" are just accidents. And be predictable, so good counterparties can learn that cooperating with you is safe and profitable.

**Watch for the horizon going finite — in them and in you.** The most dangerous moment in any repeated game is when one side's future collapses. A counterparty near bankruptcy, a desk being wound down, a trader about to retire, a fund facing redemptions — their $\delta$ is crashing toward zero, and the cooperation that held all year can snap. When you see a counterparty's horizon shortening, stop extending repeated-game trust and revert to one-shot defenses (collateral, prepayment, smaller size, tighter terms) *before* they defect. Symmetrically, recognize that your own incentives to cooperate weaken near *your* endgames.

**The invalidation.** This framework assumes the relationship is genuinely repeated and that defection is detectable and punishable. It breaks down when (a) the counterparty is truly anonymous and one-shot, (b) defection is invisible — you can't tell you were cheated — or (c) the temptation is a one-time, relationship-ending jackpot that exceeds the whole continuation value. In those cases, do not rely on the shadow of the future to protect you; rely on contracts, collateral, escrow, central clearing, and structural defenses instead. The repeated game disciplines behavior only when there is a future, the cheating shows, and the punishment bites.

**Sizing and the bottom line.** The deepest practical takeaway is that the glue holding markets together is *continuation value*, not goodwill. Where the future is long and visible, trust is rational and cheap, and you can deal on a handshake. Where it is short or hidden — anonymous venues, dying counterparties, final trades — assume the one-shot equilibrium and price the defection in. You are never really trading against a person; you are trading against their *discount factor*. Find out what it is, and you will know whether the hand you are shaking will still be there tomorrow.

This is educational material about market structure and strategic interaction, not individualized financial advice.

## Further reading & cross-links

- [The prisoner's dilemma in markets: why everyone sells at once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) — the one-shot game this post lifts the shadow of the future onto. Start here if the stage game isn't yet second nature.
- [Cartels, collusion, and the Cournot game: from OPEC to algorithms](/blog/trading/game-theory/cartels-collusion-and-the-cournot-game-from-opec-to-algorithms) — the repeated prisoner's dilemma among producers, and why cartel discipline is exactly the grim-trigger threat in oil.
- [Expected value, edge, and variance: thinking like the house](/blog/trading/game-theory/expected-value-edge-and-variance-thinking-like-the-house) — the discounted-value arithmetic in this post is expected value across time; this is the foundation.
- [Who is on the other side of your trade](/blog/trading/game-theory/who-is-on-the-other-side-of-your-trade) — the series' central question, which the repeated game answers: whether the other side cooperates depends on whether they expect to face you again.
- [Nash equilibrium, best response, and the price as a truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce) — the solution concept underneath both the one-shot defection and the repeated-game cooperation.
- [Crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) — what happens when a counterparty's horizon goes finite and the cooperation snaps.

This post opens the **Repeated Games & Reputation** track of the series. The next piece extends it from anonymous repeated play to named reputations, showing why dealers behave when their good name is the asset on the line.
