---
title: "The Keynesian Beauty Contest and Level-K Thinking"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Short-term price is not what an asset is worth to you but what you think others think others will pay, so the winning move is to reason exactly one level above where the crowd actually stops."
tags: ["game-theory", "trading", "keynesian-beauty-contest", "level-k", "behavioral-finance", "iterated-dominance", "nash-equilibrium", "momentum", "narratives", "crowd-psychology"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — In the short run the price of an asset is not what it is worth to *you*; it is what you think the average buyer will pay — which depends on what they think the average buyer will pay, and so on. Markets are John Maynard Keynes's *beauty contest*, and your edge is reasoning exactly one level above where the crowd actually stops, not at the idealized equilibrium.
>
> - Keynes's 1936 analogy: you win a newspaper beauty contest not by picking the face *you* find prettiest, but by picking the face you think the *average* entrant will pick — and everyone is doing the same, so you are guessing "what does everyone think everyone thinks."
> - The cleanest test of this is the **guess-2/3-of-the-average game**: everyone picks a number from 0 to 100, the winner is closest to 2/3 of the average. The unique Nash equilibrium — the only logically stable answer — is **0**, reached by *iterated elimination* of choices no rational player would make.
> - But real people stop at finite reasoning depth. **Level-k thinking** models this: level-0 picks randomly (~50), level-1 best-responds to level-0 (~33), level-2 to level-1 (~22), and so on. In real experiments the average winning guess lands around **22–33**, *not* 0 — people reason 1 to 2 levels deep, almost never to infinity.
> - The one rule to remember: **do not play the equilibrium, play one level above where the crowd actually reasons.** Being too-right-too-early (full Nash depth) loses to being one step ahead of the real crowd. Model your opponents' true reasoning level, not the textbook ideal.

In 1936, in the middle of writing the most influential economics book of the century, John Maynard Keynes paused to describe how the stock market really works using a newspaper game. Back then, British papers ran contests where they printed photographs of, say, a hundred faces, and readers mailed in their picks for the six prettiest. The prize went not to the reader with the best taste, but to the reader whose six picks most closely matched the *most popular* picks across all the entries. Keynes saw immediately what this did to a clever player. "It is not a case of choosing those which, to the best of one's judgment, are really the prettiest," he wrote, "nor even those which average opinion genuinely thinks the prettiest. We have reached the third degree where we devote our intelligences to anticipating what average opinion expects the average opinion to be." And, he added, "there are some, I believe, who practise the fourth, fifth and higher degrees."

That single paragraph is the most honest sentence ever written about short-term markets, and almost everyone who quotes it stops one step too early. The point is not merely that prices reflect popularity. The point is that the popularity is *itself* a guess about popularity, which is a guess about a guess, all the way down — and that the smart move is not to take this regress to its logical end, because almost nobody else does. The trader who reasons one degree deeper than the crowd wins. The trader who reasons *infinitely* deep — who plays the cold, correct, equilibrium answer — usually loses, because the crowd never gets there and the price agrees with the crowd long before it agrees with the logician.

This post builds the whole idea from zero: the beauty contest itself, the cleanest experiment that turns it into a number you can compute, the Nash equilibrium and the iterated logic that gets you there, and the *level-k* model of how deep real people actually reason. Then it turns all of that into a trading lens for momentum, narratives, and the question every position implicitly asks — *what will the crowd chase next, and am I one step ahead of them or one step behind?*

![Diagram of the beauty contest where you pick what you think the average person will pick which depends on what others think others will pick settling into the short-term price](/imgs/blogs/the-keynesian-beauty-contest-and-level-k-thinking-1.png)

The diagram above is the mental model for the entire post. Read it left to right: the naive question — "what is this worth to me?" — gets dropped almost immediately, because in a contest your private opinion does not pay. What pays is your guess about *what the average person will pay*, which is the same guess everyone else is making, which forces a regress — what they think others think others will pay — and the whole tangle settles at the level the crowd's reasoning actually reaches. By the end you will see that the green box on the right, the short-term price, is not a fact about the asset. It is a *fixed point of beliefs about beliefs*, and your job is to locate it one rung before everyone else does.

## Foundations: the beauty contest, the 2/3 game, Nash, and level-k from zero

Let me define every piece carefully, because this is one of those ideas that sounds like a slogan ("price is just psychology") until you see the machinery, at which point it becomes a precise, testable, *computable* model. We will build four things in order: the beauty contest, the experiment that makes it concrete, the equilibrium, and the model of real human reasoning.

### What a beauty contest is, as a game

A **game**, in the technical sense this whole series uses, is any situation where your best move depends on what other people do. (Contrast a *bet against nature* — a coin flip, the weather — where the world does not care what you choose.) A beauty contest is a particular kind of game called a **coordination game with a twist**: you are not trying to coordinate on the truth, you are trying to coordinate on *each other*.

Spell out the players, strategies, and payoffs — the three things that define any game:

- **Players:** every entrant (in a market, every trader).
- **Strategies:** which face to pick (in a market, what to buy or sell, and at what price).
- **Payoff:** you win if your pick matches the *average* pick — not the truth, not your taste, the average. The prize is *coordination with the crowd*, and crucially, the crowd is also trying to coordinate with you.

The twist that makes it deep is that the target you are aiming at — "average opinion" — is *made of* the very guesses everyone is making, including yours. There is no fixed bullseye sitting out in the world. The bullseye is wherever everyone collectively decides to aim, which is wherever everyone thinks everyone will aim. This is what Keynes meant by the third, fourth, and fifth degrees, and it is exactly the structure of a short-term price.

Why does this describe markets? Because **the short-term price of an asset is not what it is worth to you — it is what you can sell it to someone else for.** And what *they* will pay is not what it is worth to *them* either; it is what they think they can sell it to a third person for. The chain of "I buy it because I think I can sell it higher" is the beauty contest, dressed in tickers. We have circled this idea in the [reflexivity post](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) (the price feeds back into the fundamentals) and in [the greater-fool / musical-chairs post](/blog/trading/game-theory/the-greater-fool-and-rational-bubbles-the-musical-chairs-game) (you pay above value because a greater fool will pay still more). The beauty contest is the cleanest, most measurable version: it strips away the fundamentals entirely so you can *see* the belief-on-belief structure naked.

It helps to pin down the difference between this game and the ordinary kind of decision a beginner expects markets to be. In an ordinary decision — should I buy an umbrella? — you weigh the facts and pick what is best *for you*, and the world does not push back: the rain falls whether or not you bought the umbrella. That is a decision against nature. A beauty contest is the opposite. The thing you are trying to predict is *not a fact about the world but a fact about the other predictors*, and they are all trying to predict you right back. There is a hall-of-mirrors quality to it that simply does not exist when you are betting on the weather, and it is the source of every strange, self-referential thing markets do — the bubbles, the crashes, the trades that work until everyone notices them and then stop working precisely *because* everyone noticed.

Keynes drew one more conclusion from his newspaper game that is worth stating outright, because it is the part most quoters skip. He observed that this structure makes the market reward a *particular kind of intelligence* — not the intelligence that figures out what things are worth, but the intelligence that figures out what other people will think things are worth. "It is not sensible," he wrote, "to pay 25 for an investment of which you believe the prospective yield to justify a value of 30, if you also believe that the market will value it at 20 three months hence." In other words: being right about value and wrong about the crowd is a *losing* combination over a trading horizon. That single sentence is the seed of everything in this post, and it is why we spend the rest of it learning to measure the crowd's depth rather than the asset's value.

### The 2/3-of-the-average game: the contest as a number

The trouble with the original beauty contest is that "prettiest face" is subjective, so you cannot compute the right answer. Game theorists fixed this with a version that has a *unique* mathematically correct answer while keeping the exact same belief-on-belief structure. It is called the **guess-2/3-of-the-average game**, or the **beauty-contest game**, and it goes like this:

- Everyone secretly writes down a number from **0 to 100**.
- The organizer computes the **average** of all the numbers.
- The winner is whoever's number is **closest to 2/3 of that average**.

That is the whole game. The "2/3" is the key. It means the winning number is always *lower* than the average — so you should never pick the average, you should pick something below it. But everyone reasons that way, which drags the average down, which drags the target lower still, and the spiral is the entire point.

Walk it slowly. Suppose, naively, that everyone picks at random. The average of random numbers between 0 and 100 is about **50**. So if you expected an average of 50, you should pick 2/3 of 50, which is **33.3**. That is your best guess *if you think everyone else is random*.

But you are not the only clever one. If *everyone* figures out that they should pick 33.3, then the average will be 33.3, not 50 — so you should pick 2/3 of 33.3, which is **22.2**. And if everyone gets *that* far, the average is 22.2 and you should pick **14.8**. Each layer of reasoning multiplies the previous guess by 2/3, and the sequence marches downward: 50, 33.3, 22.2, 14.8, 9.9, 6.6, ... toward zero.

![Line chart showing the 2/3-of-the-average game converging from 50 to 33 to 22 toward the Nash equilibrium of zero](/imgs/blogs/the-keynesian-beauty-contest-and-level-k-thinking-2.png)

The chart above plots that convergence, computed directly from the series model in `data_gametheory.beauty_contest_path` (start 50, multiply by 2/3 each step). The blue dots are the guesses at each level of reasoning; the red dashed line at the bottom is the Nash equilibrium, which we will get to in a moment. The amber band marks where real human play actually clusters — levels 1 and 2, guesses around 22 to 33 — which is *nowhere near* the bottom. Hold onto that gap between the logic (zero) and the reality (the mid-20s); it is the whole trade.

Why 2/3 specifically, and not, say, 1/2 or 0.9? The fraction sets *how strongly the target pulls below the average*, and therefore how fast the spiral runs. The general game is "guess `p` times the average," where `p` is some number between 0 and 1. As long as `p` is less than 1, the only Nash equilibrium is still 0, because each round multiplies the guess by `p` and the product shrinks to nothing. But the *speed* changes everything about how the game feels. With `p` close to 1 (say 0.95), the target barely moves below the average each round, the spiral is gentle, and the depth at which people stop hardly matters — everyone's guess is near everyone else's. With `p` small (say 1/2), the target plunges fast, a single extra step of reasoning moves your guess a long way, and the *gap* between a level-1 player and a level-2 player is enormous. The standard 2/3 is chosen because it makes that gap large enough to see clearly: a level-1 player guesses 33, a level-2 player guesses 22 — an 11-point chasm that the experiment can measure. That measurable gap between adjacent levels is exactly the thing a trader is trying to monetize.

Formally, the level-`k` guess in a "guess `p` times the average" game starting from an anchor `start` is just the anchor multiplied by `p` raised to the `k`-th power:

$$g_k = \text{start} \times p^{\,k}$$

where `start` is the level-0 anchor (≈ 50 for a 0–100 range), `p` is the fraction (2/3 in the standard game), and `k` is the number of reasoning steps. Plug in `start = 50`, `p = 2/3`: `k = 0` gives 50, `k = 1` gives 33.3, `k = 2` gives 22.2, and the limit as `k` grows without bound is 0. This one line is the whole engine, and it is exactly what `data_gametheory.beauty_contest_level_k` computes.

#### Worked example: one round of beauty-contest reasoning

Suppose you are in a 2/3 game with 100 strangers and you have to pick a number. You reason: "Most of these people will not think hard. They will anchor on the middle of the range and pick something near 50." If the average is \$50 — read the numbers as if they were dollars on a number line — then 2/3 of \$50 is \$33.33. So a level-1 thinker picks \$33.

Now suppose instead you are playing against a room of *economists*, who you expect to reason one step further. They will all pick \$33, so the average will be \$33, and 2/3 of that is \$22. Against that crowd you should pick \$22.

The difference between \$33 and \$22 is not about who is *smarter* in the abstract — both numbers come from correct arithmetic. It is entirely about **how deep you think the other players will reason.** The intuition: in this game, the right answer is not a fact about numbers, it is a fact about *the people in the room.*

### The Nash equilibrium: where the logic actually ends

If each round of reasoning multiplies the guess by 2/3, where does it stop? The series 50, 33.3, 22.2, 14.8, ... never quite reaches zero in finitely many steps, but it converges to zero — and **zero is the unique Nash equilibrium** of the game.

A **Nash equilibrium** — the central solution concept of game theory, covered in depth in the [Nash equilibrium post](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce) — is a set of strategies where no player can do better by unilaterally changing their own choice, *given* what everyone else is doing. It is a self-consistent resting point: a guess that, if everyone believed it, would be confirmed.

Check that zero is one. If *everyone* picks 0, the average is 0, and 2/3 of 0 is 0. The winning number is 0, and everyone already picked 0, so no one can improve by deviating. It is self-confirming — a fixed point. Now check that *nothing else* is. Suppose everyone picked 30. Then 2/3 of the average (30) is 20, so the winner picked closest to 20, not 30 — anyone who picked 30 would have done better picking 20. So 30 is *not* stable; it unravels. The same unraveling happens at every positive number. Only at zero does the target finally catch up to the guesses. Zero is the unique fixed point, the unique Nash equilibrium.

### Iterated elimination: the formal road to zero

There is a cleaner, more rigorous way to reach zero than chasing the 2/3 spiral to its limit, and it is worth seeing because it is the same logic that prices use to grind a market toward a level. It is called **iterated elimination of dominated strategies** — a mouthful that means "repeatedly delete the choices no rational player would ever make, and see what survives."

A strategy is **dominated** if some other strategy is always at least as good, no matter what everyone else does. A rational player never plays a dominated strategy, and — here is the key — a rational player who knows the *others* are rational knows they won't either, which lets you delete a second layer, and so on. Walk the rounds:

![Pipeline of iterated elimination rounds shrinking the rational range from 0 to 100 down to only zero surviving as the Nash equilibrium](/imgs/blogs/the-keynesian-beauty-contest-and-level-k-thinking-5.png)

- **Round 0:** anyone may pick 0 to 100. The highest the average could possibly be is 100 (everyone picks 100).
- **Round 1:** if the average can be at most 100, then 2/3 of it can be at most ~67. So *any number above 67 is dominated* — it can never win, because the target can never exceed 67. A rational player deletes everything above 67.
- **Round 2:** but if every rational player has deleted everything above 67, then the average can be at most 67, so the target can be at most 2/3 of 67 ≈ 44. Now everything above 44 is dominated. Delete it.
- **Round 3:** the ceiling drops to ~30. Then ~20. Then ~13. Each round shaves the top off the surviving range.
- **The limit:** repeat forever and the only number that survives every round of deletion is **0**. That is the Nash equilibrium, reached not by psychology but by pure iterated logic.

The diagram above shows the range collapsing round by round, from the full 0–100 down to the single survivor at zero. And here is the crucial caveat, written right into the figure: *every round assumes everyone else also did the full chain of logic.* Round 2 only works if you are confident no one picks above 67 — which requires everyone to be rational *and* to know everyone is rational *and* to know that everyone knows, and so on. That tower of assumptions is called **common knowledge of rationality**, the subject of the [common-knowledge post](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know), and in the real world it simply does not hold. People stop after one or two rounds. Which is exactly what the level-k model captures.

### Level-k thinking: a model of finite reasoning depth

The Nash answer assumes everyone reasons to infinity. Real people don't. **Level-k thinking** is a behavioral model — developed by Rosemarie Nagel, Dale Stahl, Paul Wilson, and others in the 1990s — that replaces "everyone is infinitely rational" with "people reason a small, finite number of steps, and they differ in how many." The model is beautifully simple:

- **Level 0** is the anchor: a player who does not strategize at all. In the 2/3 game, the standard assumption is that level-0 picks roughly at random, so on average **~50** (the midpoint).
- **Level 1** best-responds to the belief that *everyone else is level 0*. If others pick ~50, the best response is 2/3 × 50 = **~33**.
- **Level 2** best-responds to the belief that everyone else is *level 1*. If others pick ~33, the best response is 2/3 × 33 = **~22**.
- **Level 3** best-responds to level 2: 2/3 × 22 = **~15**.
- ...and **level ∞** is the full Nash equilibrium: **0**.

Each level is one more step of "I think you think...". The number `k` is literally how many times you iterate. And the empirical finding — replicated across hundreds of experiments and tens of thousands of subjects — is that most people are level 1 or level 2. Very few are level 0 (pure noise), very few go past level 3, and almost nobody plays the Nash zero.

![Bar chart of level-k guesses showing level 0 at 50, level 1 at 33, level 2 at 22, level 3 at 15, level 4 at 10](/imgs/blogs/the-keynesian-beauty-contest-and-level-k-thinking-3.png)

The bars above are the level-k guesses computed straight from `data_gametheory.beauty_contest_level_k` — 50, 33, 22, 15, 10 for levels 0 through 4. The gray bar (level 0) is the naive anchor; the green bars (levels 1 and 2) are where real humans cluster; the amber bars (levels 3 and 4) are where you start *over*-thinking — reasoning deeper than the crowd actually does, which, as we will see, is its own way to lose. The model's whole value is that it tells you not "what is the rational answer" but "**what answer should you give, given how deep the actual people in the room reason?**" That question — not the equilibrium — is the one a trader has to answer.

#### Worked example: choosing your level against a known crowd

Suppose you are told the exact composition of a 100-person 2/3 game: 20 players are level 0 (random, average ~\$50), 50 are level 1 (pick ~\$33), and 30 are level 2 (pick ~\$22). What should *you* pick?

First compute the average the crowd will produce. Weight each group:

$$\text{avg} = \frac{20(50) + 50(33) + 30(22)}{100} = \frac{1000 + 1650 + 660}{100} = \frac{3310}{100} = \$33.1$$

The target is 2/3 of that average: 2/3 × \$33.1 = **\$22.1**. So your winning pick is about \$22 — which is the *level-2* answer, exactly one step deeper than the dominant level-1 bloc. Notice you did *not* pick the Nash zero, and you did not even pick the deepest level present (you matched it, you didn't exceed it). You picked one rung above the *center of mass* of the crowd. The intuition: your job is not to be the smartest person in the room, it is to be one step ahead of the *average* person in the room — and you find that step by modeling the actual distribution, not the ideal.

## Why the crowd never reaches zero (and why that is the whole opportunity)

Here is the single most important empirical fact in this entire topic, and it is the one that converts a cute puzzle into a trading edge. **When you run the 2/3 game with real people, the answer is almost never close to zero.** The winning number — 2/3 of the actual average — typically lands somewhere in the low-to-mid 20s, and the raw average of guesses is usually in the low-to-mid 30s.

The most famous data come from Rosemarie Nagel's 1995 laboratory experiments and from several large-scale *newspaper* contests run in the late 1990s — the German magazine *Spektrum der Wissenschaft*, the *Financial Times* (run by Richard Thaler in 1997), and a Danish newspaper *Politiken* — which together drew thousands of entrants. Across these contests the average guess clustered in the **30s**, with two pronounced spikes: one near **33** (the level-1 answer) and one near **22** (the level-2 answer). A thin tail of entrants picked **0 or 1** — the game-theory students who knew the Nash equilibrium — and a small protest spike picked **100**. The *Financial Times* contest, with about 1,400 entries, had a winning number of **13**; a very large pooled analysis by Bosch-Domènech and coauthors (2002), covering roughly 7,500 newspaper entrants, found the same level-1 and level-2 spikes dominating the distribution.

![Histogram of real beauty-contest entries clustering at level 1 near 33 and level 2 near 22 with a thin spike of Nash-zero entrants and a winning average around 23](/imgs/blogs/the-keynesian-beauty-contest-and-level-k-thinking-4.png)

The histogram above shows the stylized shape of those real distributions (sourced from Nagel 1995 and the Bosch-Domènech et al. 2002 newspaper pool). The green bars are the level-1 and level-2 clusters where the crowd actually lives. The red bar at the far left is the cluster of people who picked the Nash zero — and notice how *small* it is, and notice that those people *lost*. The amber dashed line marks the winning number, around 23: a level-2-ish answer. The people who played the equilibrium were the most *correct* about the game's logic and the most *wrong* about the prize.

That is the whole lesson in one picture. The Nash equilibrium tells you where the logic ends. It does not tell you where to bet. To win, you have to model the *people* — and the people stop reasoning after one or two steps. So you should stop one step *after they do*. Not at infinity. One step past the crowd's actual depth.

Why do people stop at one or two steps rather than going deeper? It is not that they are incapable of the arithmetic — most can compute 2/3 of 33 in their heads. There are three honest reasons, and each one carries directly into markets. The first is **uncertainty about others**: even if you can reason ten steps deep, you have no confidence the *others* will, so reasoning past their likely depth is wasted effort that only makes you wrong. The second is **the cost of thinking**: deeper reasoning takes time and attention, which are scarce, so people rationally stop once the marginal step stops obviously helping. The third, and most important for trading, is **the fear of being too clever**: experienced players have learned, often painfully, that the deepest answer loses, so they deliberately throttle their depth to stay near the crowd. A seasoned trader who guesses 22 instead of 0 is not failing to see the equilibrium — they are *refusing* to play it, because they have internalized that the prize goes to crowd-plus-one, not to the logician. That refusal is a skill, and it is the single hardest thing for a smart beginner to learn, because it feels like deliberately playing dumb.

### The depth is contestant-dependent, and it can shift

A subtle and important wrinkle: the right depth is not a constant of nature, it depends on *who is playing*. When the same 2/3 game is run with game-theory PhD students, the average drops — they reason deeper, so the winning number falls toward the teens or single digits. When it is run with a general newspaper audience, the average is higher. And when a population plays the game *repeatedly*, the average **falls over rounds** as people learn that everyone else is also shaving the target down — the crowd's effective depth increases with experience.

This is exactly the dynamic in markets. A brand-new, retail-heavy market (a meme stock, a fresh token launch) has a shallow crowd — level 0 and level 1 dominate, narratives run hot, and the "average opinion about average opinion" is naive and easy to front-run. A mature, professional market (on-the-run Treasuries, large-cap index futures) has a deep crowd — most participants are already two or three levels in, the easy front-runs are arbitraged away, and the winning depth is higher. The lesson is the same in both: find the crowd's current depth, then add one. But the *number* you add one to is completely different, and reading it wrong is how you become the sucker.

There is a second, subtler way the depth shifts that traders chronically underestimate: depth is not uniform *across the same crowd at the same time*. In any real market some participants are level 0 (price-insensitive flows: index funds rebalancing, retirement contributions, forced liquidations), some are level 1 (trend-followers and narrative-chasers), and some are level 2 or deeper (the desks explicitly modeling everyone else). The "crowd depth" you want to estimate is really the *center of mass* of this mixture, weighted by how much capital each layer moves. A market can look deep on average while a fat tail of level-0 flow keeps a front-run alive — which is exactly the structure that lets a sophisticated desk profitably anticipate, say, a large index reconstitution: the index funds buying the added names are pure level-0 flow, utterly predictable and price-insensitive, so the level-1 move is to buy ahead of them. The skill is not asking "is this market smart?" but "where is the capital-weighted average reasoning depth, and is there a predictable shallow layer I can get in front of?"

## The belief ladder: standing one rung above the crowd

Let me make the "one level above" idea visual and exact, because it is the operational core of everything that follows. Picture the reasoning as a ladder, each rung a deeper layer of "I think you think...".

![Layered ladder of higher-order beliefs from level 0 random to level 1 best-response to level 2 to level 3 over-thinking up to the Nash zero with the crowd at level 1-2 and the winning play one rung above](/imgs/blogs/the-keynesian-beauty-contest-and-level-k-thinking-6.png)

The ladder above runs from the ground (level 0: "I just pick something," ~50) up through level 1 ("they pick 50, so I pick 33"), level 2 ("they think *that*, so I pick 22"), level 3 (~15, already over-thinking), all the way to the Nash zero at infinity. The crowd, marked on the left, actually lives at level 1 to 2. And the winning move, marked on the right, is to stand **exactly one rung above the crowd** — not at the top of the ladder.

This is the part people get wrong, and it is worth dwelling on because it is so counterintuitive. Your instinct, once you understand the game, is to reason as deeply as possible — to climb to the top, to play the *correct* equilibrium. But in this game **the most correct answer is a losing answer.** If the crowd is at level 1–2 (guessing ~33 and ~22) and you triumphantly play the Nash zero, the winning number is around 22 and you are off by 22 — you lose to the level-2 player who was one step ahead of the crowd, not infinitely ahead of it.

The skill, then, is not depth for its own sake. It is **calibrated depth**: read where the crowd actually stops, and go *one* rung higher. Too shallow and you are the crowd (no edge, you are the exit liquidity). Too deep and you are early and alone (right about the destination, wrong about the timing, and you lose anyway). The edge lives in the narrow band of *the crowd's depth, plus one*.

#### Worked example: the cost of being too-right-too-early

Three traders play a 2/3 game where the true crowd is level 1 (everyone else guesses ~\$33, so the average is \$33 and the winning target is 2/3 × \$33 = \$22).

- **Trader A** is level 0, guesses \$50. Distance from the \$22 target: \$28. Loses badly.
- **Trader B** is level 2, guesses \$22. Distance from the target: \$0. **Wins.**
- **Trader C** is the genius who knows the Nash equilibrium and guesses \$0. Distance from the \$22 target: \$22. Loses — and loses by almost as much as the naive Trader A.

Trader C understood the game more deeply than anyone. Trader C still lost, by \$22, to Trader B who was merely *one step* ahead of the crowd. The takeaway is brutal and worth memorizing: **in a beauty contest, being right about the equilibrium is worth nothing; being right about the crowd's depth-plus-one is worth everything.** The market does not pay you for being correct. It pays you for being correct *at the same time the crowd is about to agree with you.*

## Common misconceptions

**"The rational answer is zero, so a smart player should pick zero."** This is the single most common error, and it is exactly backwards. Zero is the rational answer *only if everyone is infinitely rational and you know it* — common knowledge of rationality. Real players are not, and you know *that* too. The smart play is to model their actual finite depth and respond to it. A player who picks zero is being rational about the *game* and irrational about the *players*. In the data, zero-pickers lose. The correct meta-rational move is to be rational about the irrationality you are facing.

**"It is just about being smarter than everyone else."** No — it is about being *exactly one level deeper*, no more. Reasoning ten levels deep when the crowd reasons two levels deep makes you *more* wrong, not less. Depth past the crowd's depth-plus-one is wasted at best and actively losing at worst. This is why brilliant macro analysts who are "right about the fundamentals" can bleed for years: they are reasoning at level 5 in a market trading at level 2. Their answer is correct and untradeable.

**"If I just figure out the fundamental value, I'll be fine in the short run."** The beauty contest is precisely the structure where fundamentals do *not* determine the short-run price. The 2/3 game has no fundamentals at all, and it still has a sharp, predictable price (~22) set entirely by belief-on-belief. In real markets fundamentals anchor the *long* run, but over the horizon of a trade, the price is the crowd's coordination point. As we noted in the [reflexivity post](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves), price and value feed each other; the beauty contest is what the price half of that loop is *doing*.

**"Everyone reasons to the same depth, so I can't get an edge."** Empirically false and it is the whole reason an edge exists. The level-k distribution is *spread out* — some level 0, mostly level 1–2, a few deeper. Heterogeneity in depth is what creates winners and losers. If literally everyone were exactly level 2, the game would be a tie and no one would have an edge; the dispersion is the opportunity. Your job is to estimate the *center of mass* of that distribution and sit just above it.

**"Once I know the trick, I'll always win."** No, because the crowd learns too. Run the game repeatedly with the same people and the winning number falls each round as everyone shaves the target down — the crowd's depth rises with experience. The edge is not a fixed trick; it is a *continuous* act of re-estimating the crowd's current depth. A front-run that worked last cycle (when the crowd was shallow) fails this cycle (when the crowd has wised up). This is why "buy the obvious narrative" works until it suddenly doesn't.

## How it shows up in real markets

The beauty contest is not a metaphor for markets; it is a literal description of how the short-run price is set. Here are concrete, dated episodes where the belief-on-belief structure — and the level-k depth of the crowd — was the whole story.

### Momentum and trend-following: betting the crowd hasn't finished arriving

The most direct market translation of the beauty contest is **momentum** — buying what has been going up because you expect others to keep buying it. A momentum trader is not claiming the asset is undervalued; they are claiming the crowd's arrival is *incomplete*, that there are still level-1 and level-2 buyers who haven't bought yet. Decades of data (the academic literature from Jegadeesh and Titman's 1993 paper onward) show momentum is one of the most robust return factors across markets and centuries — precisely because crowds arrive in waves, not instantly, so the price keeps coordinating upward as deeper layers of buyers join.

The level-k lens explains *when* momentum breaks: it works while the crowd's effective depth is still increasing (new buyers arriving), and it reverses violently when the crowd reaches its maximum depth and the only people left are the front-runners trying to sell to each other. The crash is the moment the average opinion realizes the average opinion has already arrived. Momentum is the beauty contest played in continuous time.

There is a precise way to see why early momentum pays and late momentum is a trap, and it maps directly onto the level-k spiral. Read a price rally as the crowd climbing the ladder one rung at a time, in public, over days or weeks. At the start, the level-0 anchor is "this is a forgotten, boring asset" — the price is low. The level-1 players arrive first ("this is starting to move, others will notice"), then the level-2 players ("the level-1 momentum crowd is here, I'll ride the trend they create"), and so on. Each new layer of buyers pushes the price up and *raises the anchor* for the next layer. The rally is literally the crowd's effective `k` increasing in real time. The front-runner's edge is being one rung ahead of the layer currently arriving — buying before the level-1 wave when you are level 1, before the level-2 wave when you are level 2. The trap is buying when you are the *last* rung: when the crowd's depth has maxed out, the next layer of buyers does not exist, and the only liquidity left is the early players selling to you.

#### Worked example: pricing a momentum trade as a beauty contest

Suppose a small stock trades at \$10. You estimate the crowd will climb three rungs of a beauty-contest rally before it tops, and that each rung adds buyers who push the price up by 40% over the prior rung's level. You are deciding whether to buy now (you would be the level-1 entrant, one rung ahead of the level-0 anchor crowd) or to wait and buy after the first wave.

- **If you buy now at \$10** (level-1 entry, ahead of the arriving crowd): the level-2 wave lifts it to \$10 × 1.40 = \$14, and the level-3 wave to \$14 × 1.40 = \$19.60. If you sell into the level-3 buyers near the top, you make \$19.60 − \$10 = **\$9.60 per share**, a 96% gain, by being one rung ahead the whole way up.
- **If you instead wait and buy at \$14** (you are now the level-2 entrant, but the crowd is *also* at level 2 — you matched it, you did not beat it): only one rung of buyers remains, the level-3 wave to \$19.60. You make \$19.60 − \$14 = \$5.60, a 40% gain — *if* you sell perfectly at the top.
- **If you buy at \$19.60** (the top, you are level 0 again — "number go up" — while the early players are at level 3 and selling): there is no rung above you. The next move is the unwind back toward \$10, and you lose roughly \$9.60 per share.

The arithmetic makes the lesson concrete: the *same* asset and the *same* rally produce a 96% gain, a 40% gain, or a near-total loss, depending only on which rung you bought at relative to the crowd. The intuition: in a momentum trade you are not buying the company, you are buying *your position in the queue of buyers* — and the only profitable position is ahead of the people still arriving.

### The 2021 meme-stock episode: a shallow crowd, easy to front-run

GameStop (GME) in January 2021 is a near-perfect natural experiment in crowd depth. A wave of retail traders — coordinating openly on Reddit's r/wallstreetbets — pushed the stock from under \$20 at the start of January to an intraday high of \$483 on January 28, 2021. Almost nobody buying at \$300 thought GameStop was *worth* \$300 as a business. They were playing the beauty contest in its purest form: buy because the crowd is still arriving, and sell to a greater fool before it stops.

The crowd was *shallow* — mostly level 0 and level 1, driven by narrative and momentum, easy to front-run early. The traders who made the most got in when the crowd was at level 0 (the stock was a forgotten value play) and reasoned to level 1 ("this is about to go viral"). The ones who lost the most bought at the top, reasoning at level 0 themselves ("number go up") while the early players — now at level 2 — were already selling to them. The stock fell back below \$60 (split-adjusted, far lower) within weeks. The lesson is textbook level-k: *the same trade is a fortune or a wipeout depending entirely on whether you were one rung above the crowd or one rung below it.*

### Initial coin offerings and token launches: brand-new, maximally shallow crowds

Crypto token launches are the most extreme shallow-crowd environments in finance, because the asset often has *no* fundamentals to anchor on — it is belief-on-belief with the fundamentals leg amputated entirely, exactly like the 2/3 game. During the 2017 ICO mania and again in the 2021 and 2024 meme-coin cycles, tokens with literally zero cash flows reached billion-dollar valuations purely on the crowd's coordination. The winning players were level-1 thinkers who bought what they expected to *trend* and sold into the level-0 latecomers chasing the chart. The on-chain mechanics of who front-runs whom are covered in the [MEV post](/blog/trading/game-theory/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks); the *strategic* layer is pure beauty contest. The recurring carnage — most tokens round-trip to zero — is the crowd reaching maximum depth and discovering there is no greater fool left.

### The Fed and the "what will the market do?" reflex

Even in deep, professional markets the beauty contest runs — just at greater depth. When the Federal Reserve makes a policy decision, professional traders are rarely trading their own view of what the Fed *should* do. They are trading their view of *how other traders will react*, which is itself a view about how others think others will react. The classic "buy the rumor, sell the news" pattern — explored in the [public-signals / Fed post](/blog/trading/game-theory/buy-the-rumor-sell-the-news-public-signals-and-the-fed) — is a beauty-contest reflex: the price moves to where the crowd *expects* the crowd to be *before* the announcement, so by the time the news is public, the coordination is already complete and the price often reverses. The depth here is high (level 2–3 is table stakes among professionals), but the structure is identical: you are still guessing what average opinion expects average opinion to be.

This is why a Fed decision that comes in *exactly as expected* can still move the market hard, and in either direction — a fact that baffles people who think the price should reflect the *facts*. The facts were already known; what moves is the *coordination*. If the crowd had pre-positioned for a dovish surprise that did not come, the unwind of that positioning sells the news even though the news itself was neutral. The price was never tracking the Fed's actual decision; it was tracking the crowd's belief about the crowd's belief about the decision, and the announcement simply collapses that tower of beliefs into a single realized number. Traders who do well around central-bank events are not better at forecasting policy — most cannot forecast it at all — they are better at reading where the crowd's coordination has *already* gone and fading it when it has overshot. That is a beauty-contest skill, not a macro-forecasting one, and confusing the two is a classic way for a brilliant economist to lose money.

### Macro consensus trades and the crowded-exit problem

The most dangerous level-k failure in professional markets is the **crowded trade**: a position that *everyone* who reasons to depth 2 has put on, so the entire crowd is at the same rung. When everyone is level 2, there is no one left to sell to at a higher level — the trade is "obvious," which means the coordination has already happened and the only remaining move is the unwind. This is the subject of the [crowded-trades / exit-game post](/blog/trading/game-theory/crowded-trades-and-the-exit-game): the violent reversals in these trades (the 2007 quant quake, the 2018 short-vol blowup, countless "consensus" macro trades) happen because the beauty contest has reached its fixed point and the next move is everyone trying to exit the same door at once. The level-k warning sign is *unanimity*: when the depth-2 trade is universally agreed upon, the edge is gone and the exit risk is maximal.

### IPO and Treasury-auction pricing: guessing the clearing crowd

When you bid in an IPO or a Treasury auction, you are not bidding what *you* think the asset is worth — you are bidding what you think the *marginal accepted bidder* will think it is worth, because that is what sets the clearing price you'll be marked against. Bid too high (level 0, your own valuation) and you suffer the **winner's curse** — overpaying because you were the most optimistic, the subject of the [adverse-selection / winner's-curse post](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news). Bid at the crowd's clearing level (depth-matched) and you participate fairly. The whole auction is a beauty contest where the "prettiest face" is the clearing price, and the skill is estimating where the crowd of bidders will coordinate.

## The playbook: how to play the beauty contest

This is the part where the model becomes a position. The beauty contest tells you that your edge is never "I know what this is worth" in the short run — it is "I know where the crowd's coordination point is, and I am one rung above it." Here is how to operationalize that.

![Three-by-three grid of your reasoning depth versus the crowd's actual depth showing you win only at plus-one above the crowd and lose when too shallow or too deep](/imgs/blogs/the-keynesian-beauty-contest-and-level-k-thinking-7.png)

The grid above is the strategic map. Columns are the crowd's *actual* reasoning depth; rows are *yours*. The green winning cells run down the diagonal at **one level above** the crowd: when the crowd is at level 0 you win at level 1; when the crowd is at level 1 you win at level 2. Match the crowd exactly (the amber diagonal) and you are *the crowd* — no edge, you net zero before costs. Go too shallow (red, below the line) and you are the exit liquidity. Go too deep (red, above the line, like the Nash-zero player) and you are right too early and lose anyway. The entire game is finding the green cell, which means *first estimating the crowd's depth* and *then* adding exactly one.

**Who is on the other side.** In a beauty-contest trade, your counterparty is the crowd one rung *below* you — the level-0 latecomer you sell your momentum to, or the level-1 narrative-buyer you front-ran. Your *risk* is the player one rung *above* you, who is selling to you. Before you put on a "the crowd will chase this" trade, name both: who is the fool you expect to sell to, and who is the sharper player who might be selling to *you*? If you cannot identify a plausible greater fool below you, you are probably the fool.

#### Worked example: estimating the crowd's depth and sizing one rung above

Suppose you are looking at a sector rotation everyone is starting to talk about. You want to estimate the crowd's depth and decide whether there is room to be one rung ahead. You score the market on three observable signals, each worth one "rung" of depth:

- *Coverage:* the idea is in roughly half of the research notes you read but not yet in the mainstream financial press. Call that depth ≈ 1.5 — it has moved past level 0 (nobody) but is not yet universal (level 2+).
- *Positioning:* surveys show real money is *underweight* the sector, meaning most professionals have not yet acted. That holds the effective depth *down* — the crowd has thought about it but not committed, so the coordination is incomplete.
- *Price:* the sector is up modestly, not parabolically — consistent with a level-1 crowd arriving, not a maxed-out level-3 blow-off.

Net read: the crowd is around **level 1**, and still arriving. Your edge is to be **level 2** — one rung above. Concretely, that means buying the sector *now*, before the coverage goes mainstream and the positioning flips from underweight to crowded, and planning to *sell into* the level-1 wave once every note and every survey shows the trade has become consensus. On sizing: because you are only one rung ahead — not five — the trade can work on a *short* horizon (the crowd is close behind you), so you can size it normally and set your invalidation at the moment of unanimity. Contrast a case where the same three signals scored the crowd at level 2 already (idea everywhere, positioning crowded, price extended): there your "level-3" edge is thin and fragile, the greater fool below you may not exist, and you would size *down* hard or pass. The intuition: the depth estimate is not academic — it directly sets both your entry timing and your position size, because being one rung ahead of a near crowd is a fast, sizeable trade, while being one rung ahead of a deep crowd is a slow, dangerous one.

**Estimate the crowd's depth before you size.** This is the actual skill, and it is concrete:
- *Who is in this market?* Retail-heavy and new (meme stock, fresh token) → shallow, level 0–1 dominates → narratives and momentum front-run well, and the obvious story is tradeable. Professional and mature (rates, index futures) → deep, level 2–3 → the obvious trade is already in the price.
- *How long has the trade been "obvious"?* A narrative on day one (crowd shallow) is a front-run. The same narrative on every screen and in every headline (crowd deep, depth-2 unanimous) is an exit. Unanimity is the sell signal.
- *Is the crowd's depth rising?* In a repeated game the crowd wises up. A pattern that front-ran cleanly last cycle may be the consensus this cycle. Re-estimate every time; never reuse last cycle's depth.

**The trade.** Buy when you have identified a *shallow* crowd that is still arriving and you are exactly one rung ahead of it — early in a narrative, before it is universal. The position is "the crowd's coordination point is higher (or lower) than the current price, and they haven't all gotten there yet."

**The invalidation — and it is sharp.** You are wrong, and you exit, when *the crowd reaches your level.* The moment the depth-2 trade becomes unanimous — every analyst note, every headline, every retail screen showing the same idea — the beauty contest has hit its fixed point and there is no one left to sell to at a higher rung. That unanimity is not confirmation; it is the exit bell. The crowded-trade unwind (see the [crowded-trades post](/blog/trading/game-theory/crowded-trades-and-the-exit-game)) is what happens to everyone who mistook unanimity for safety.

**Sizing and the too-deep trap.** Size *down*, not up, as your conviction in the *fundamental* outruns the crowd's depth. The bigger the gap between "what it's worth" (your level-5 view) and "where the crowd is" (level 2), the *longer* you may be wrong and the *more* a large position can liquidate you before you are vindicated. The graveyard of macro funds is full of correct level-5 theses sized as if the market traded at level 5. If you are reasoning much deeper than the crowd, either wait for the crowd to catch up before sizing, or accept that you are running a *long-horizon value* trade, not a beauty-contest trade, and size it for years of being early. The two are different games; do not confuse the sizing.

**The behavioral-honesty rule.** The whole edge comes from one discipline: **model your opponents' real reasoning level, not the idealized equilibrium.** It is emotionally satisfying to play the "correct" Nash answer and feel smart. It is profitable to play the *crowd-plus-one* answer and feel slightly uncomfortable — because plus-one always feels too shallow to the person who can see all the way to zero. Resist the pull toward the equilibrium. The market does not pay for correctness; it pays for being one step ahead of the people actually trading, and those people stop thinking long before the logic does. As we saw with mixed strategies in the [unpredictability post](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable), the optimal play is rarely the one that feels cleverest — it is the one calibrated to the actual opponent in front of you.

*This is educational, not financial advice. Beauty-contest trades — momentum, narratives, "the crowd will chase this" — are among the most reflexive and fastest-reversing in markets; the same structure that pays you for being one rung ahead punishes you brutally for being one rung behind, and position sizing for that asymmetry matters more than the call itself.*

## Further reading & cross-links

- [Nash equilibrium, best response, and the price as a truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce) — the solution concept the beauty contest's zero is *an example of*, and why the equilibrium is a resting point of beliefs, not a forecast.
- [Common knowledge and "I know that you know that I know"](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know) — the tower of beliefs-about-beliefs that iterated elimination needs, and why it fails in the real world (which is *why* level-k beats Nash).
- [Reflexivity: markets that watch themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) — the feedback loop where the beauty-contest price feeds back into the fundamentals it was supposed to reflect.
- [Mixed strategies and the value of being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable) — the companion behavioral lesson: the optimal play is calibrated to the actual opponent, not to what feels cleverest.
- [The greater fool and rational bubbles: the musical-chairs game](/blog/trading/game-theory/the-greater-fool-and-rational-bubbles-the-musical-chairs-game) — the beauty contest with the music turned on: buying above value because a greater fool will pay more.
- [Crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) — what happens when the depth-2 trade becomes unanimous and the beauty contest hits its fixed point.
