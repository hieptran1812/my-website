---
title: "The Prisoner's Dilemma in Markets: Why Everyone Sells at Once"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "The prisoner's dilemma explains why individually rational traders dump at the same moment and crash a market that mutual patience would have kept calm."
tags: ["game-theory", "prisoners-dilemma", "trading", "nash-equilibrium", "market-crashes", "liquidity", "behavioral-finance", "risk-management", "coordination", "dominant-strategy"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A market crash is rarely a failure of reasoning; it is everyone reasoning correctly, in private, at the same time. That is the prisoner's dilemma: when selling first is your best move no matter what anyone else does, the only stable outcome is that everyone sells first — and the price falls through the floor that mutual patience would have held.
>
> - In a prisoner's dilemma, **defection is a *dominant strategy*** — it beats cooperation whatever the other side does — so the unique Nash equilibrium is *mutual defection*, even though *mutual cooperation* would have paid both players more.
> - Markets are stuffed with prisoner's dilemmas: the run for the exit, the broker fee race to zero, the analyst who can't go bearish, the forced-liquidation cascade, the de-risking stampede.
> - The gap between what is *individually rational* and what is *collectively optimal* is the whole story of a panic. In the run-for-the-exit game below, the Nash outcome leaves each holder with \$60 of a \$100 position when mutual patience would have left them \$90 — a \$30 cooperation loss each.
> - **The rule to remember:** against anonymous, one-shot counterparties, *expect defection* — and either be early, be hedged, or don't be in the crowded trade at all.

In the autumn of 1998, a hedge fund called Long-Term Capital Management held positions so large that its prime brokers — the big banks that lent it money and cleared its trades — each knew something terrifying. If LTCM blew up, every bank holding the same kind of position would have to sell into the same thin market at the same moment. Each bank, sitting alone in its own risk meeting, reached the same conclusion: *get out first*. Everyone trying to get out first is precisely what turns a stressed market into a crashed one. The Federal Reserve had to organize a bailout, not because LTCM's trades were wrong, but because the *game* the banks were playing had only one stable ending, and it was ugly.

This pattern — smart people each making the locally correct choice, and the sum of those correct choices being a disaster — is the most important idea in game theory for anyone who wants to understand markets. It has a name. It is the **prisoner's dilemma**, and once you can see it, you will see it everywhere: in every flash crash, every bank run, every fire sale, every "the model said sell so we sold" cascade. The point of this post is to build the prisoner's dilemma from absolute zero — two suspects in two rooms — and then show you that the run for the exit is the *same game*, played by thousands of strangers who will never meet, and who therefore have no reason to trust each other.

The matrix above is the mental model we are about to unpack. Two players, two choices each, four boxes — and in the one box both players land in, both are worse off than they could have been.

![Prisoner's dilemma payoff matrix with the mutual confess Nash highlighted](/imgs/blogs/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once-1.png)

## Foundations: the prisoner's dilemma from zero

Before we touch a single market, let's build the game the way it was invented — with two suspects, a prosecutor, and no markets at all. Everything that follows is just this game wearing different costumes.

### The story

Two people — call them you and a partner — are arrested for a crime you committed together. The police separate you into two rooms; you cannot talk to each other. A prosecutor walks into each room and makes the same offer:

> "We have enough to convict you both of a minor charge — one year each — even if you both stay silent. But here's the deal. **Confess and rat out your partner.** If you confess and your partner stays silent, you walk free (zero years) and your partner gets ten years. If you both confess, you each get five years. If you both stay silent, you each get the one year on the minor charge. Your partner is hearing the same offer right now, in the next room. Decide."

That is the whole game. Two players. Each has two *strategies*: **cooperate** (stay silent — cooperate *with your partner*, not with the police) or **defect** (confess, betray your partner). Each combination of choices produces a *payoff* for each of you, measured here in years of prison (which we treat as negative — fewer years is better).

Let me define the three terms that do all the work, because they recur in every market example:

- A **strategy** is a choice a player can make. Here: cooperate or defect.
- A **payoff** is what a player gets from a given combination of everyone's choices — here, a sentence in years. A *payoff matrix* is just the table of all four outcomes.
- An **equilibrium** is a combination of strategies where no player can improve their own payoff by changing only their own choice, holding everyone else fixed. We'll make this precise in a moment; it's the single most important concept in the post.

### Reading the matrix

The cover figure shows the payoff matrix. Rows are *your* choice (silent or confess), columns are your *partner's* choice. Each box lists both sentences. The four outcomes:

- **Both silent (both cooperate):** one year each. The best *joint* outcome — two years of prison total.
- **Both confess (both defect):** five years each. Ten years of prison total.
- **You confess, partner silent (you defect):** you walk (zero), partner gets ten.
- **You silent, partner confesses (you cooperate):** you get ten, partner walks.

Stare at that for a second and a knot forms in your stomach, because the math is about to tell you to do something that *feels* like betrayal — and is.

### Dominant strategies: the move that's best no matter what

Here is the engine of the dilemma. Put yourself in your room. You don't know what your partner will do, so reason through both cases:

- **Suppose your partner stays silent.** If you also stay silent, you get one year. If you confess, you walk — zero years. Zero is better than one. **You should confess.**
- **Suppose your partner confesses.** If you stay silent, you get ten years (the "sucker" outcome). If you confess, you get five. Five is better than ten. **You should confess.**

Whatever your partner does, confessing leaves you better off. That is the definition of a **dominant strategy**: a choice that beats every alternative in *every* situation the other player could create. When you have a dominant strategy, you don't need to predict the other player at all — you just play it.

And here is the gut punch: the game is *symmetric*. Your partner is in the next room running the identical logic and reaching the identical conclusion — confess. So you both confess. You both get five years. You land in the box that, looking at the matrix from above, you can both plainly see is *worse for both of you* than the one-year-each box you could have reached by both staying silent.

#### Worked example: the dominant-strategy calculation in dollars

Prison years are abstract; let's redo the exact same logic with money, because that's the version that matters for trading. Two firms, A and B, each running a profitable but capacity-limited strategy. They can **hold** their positions calmly (cooperate) or **dump** aggressively to grab the last of the liquidity (defect). Payoffs are the profit each firm books, in millions.

- Both hold: each books a steady \$3M.
- Both dump: they crash their own market and each books \$1M.
- You dump, they hold: you grab the liquidity first and book \$5M; they're stuck and book \$0.
- You hold, they dump: you're stuck at \$0; they book \$5M.

Now run the dominant-strategy test for firm A:

- If B holds: A holds for \$3M or A dumps for \$5M. **Dump wins** (\$5M beats \$3M).
- If B dumps: A holds for \$0 or A dumps for \$1M. **Dump wins** (\$1M beats \$0).

Dumping is dominant — it beats holding by \$2M in the first case and \$1M in the second. By symmetry, B dumps too. They land on \$1M each. Mutual holding would have paid \$3M each, so each firm walks away \$2M poorer than the patient outcome that was *right there on the table*.

The intuition: when grabbing first beats waiting no matter what the other side does, both grab first, and the prize they were fighting over shrinks for everyone.

### Nash equilibrium: why "both defect" is the only stable resting point

The 1950 idea that earned John Nash a Nobel Prize is deceptively simple. A **Nash equilibrium** is a combination of strategies — one for each player — where *no single player can do better by changing only their own choice, given what everyone else is doing.* It's a "no regrets, given the others" point: each player is playing their *best response* to the others, so nobody has a private incentive to move.

Let's test all four boxes of the prison game to find which are Nash equilibria:

- **Both silent:** could you do better by deviating? Yes — switch to confess and you go from one year to zero. You *regret* staying silent. **Not** an equilibrium.
- **Both confess:** could you do better by deviating? Switch to silent and you go from five years to ten. Worse. Your partner faces the same. Nobody can improve alone. **This is a Nash equilibrium.**
- **You confess, partner silent:** your partner could switch from silent (ten years) to confess (five years) and improve. So your partner deviates. **Not** an equilibrium.
- **You silent, partner confesses:** symmetric — you'd switch. **Not** an equilibrium.

There is exactly one Nash equilibrium: **(defect, defect)** — both confess, five years each. That's the highlighted red box in the cover figure. The tragedy is baked into the structure: the *only* stable outcome is the one that's bad for both. We didn't compute this by hand and hope; the series' helper library does it for us, and we'll lean on that for every payoff matrix in this series.

```python
import data_gametheory as gt

A = [[-1, -10], [0, -5]]   # your payoff (negative prison years; higher = better)
B = [[-1, 0], [-10, -5]]   # partner's payoff (symmetric)
                           # row/col: 0 = silent (cooperate), 1 = confess (defect)
print(gt.nash_2x2(A, B))
    # -> {'pure': [(1, 1)], 'mixed': None}
    #    (1, 1) = (confess, confess): the unique pure Nash, no mixed NE
```

`nash_2x2` checks every cell for the no-regret condition and returns the equilibria. The answer — `(1, 1)`, both defect — confirms the logic above. *Pure* means each player picks one strategy with certainty (as opposed to a *mixed* equilibrium where you randomize; the PD has none, because defection is strictly best every time, so there's nothing to randomize over).

The deep lesson of the foundations: **what's individually rational and what's collectively optimal can be two different boxes.** Each player optimizing their own payoff drives the pair to an outcome both would have escaped if only they could trust each other and coordinate. The market versions all turn on this same wedge.

### What makes a game a prisoner's dilemma: the four payoffs in order

Not every two-by-two game is a prisoner's dilemma. The PD is a *specific* arrangement of the four payoffs, and naming them lets us spot the structure in the wild. Game theorists give the four payoffs that face a single player four letters:

- **T — the *Temptation* payoff:** what you get for defecting while the other player cooperates. (You walk free; you sell first and get the good bid.) This is the highest of the four.
- **R — the *Reward* payoff:** what you get for mutual cooperation. (One year each; both hold and recover \$90.) Second highest.
- **P — the *Punishment* payoff:** what you get for mutual defection. (Five years each; both sell and recover \$60.) Second lowest.
- **S — the *Sucker* payoff:** what you get for cooperating while the other defects. (Ten years; you're the lone bagholder at \$50.) The worst.

A game is a prisoner's dilemma precisely when these four obey one ordering:

$$T > R > P > S$$

That ordering is doing all the work, and it's worth reading slowly. `T > R` means defecting when the other cooperates beats mutual cooperation — temptation is real. `P > S` means defecting when the other defects beats being the lone cooperator — defection protects you from the sucker outcome. Put those two together and defection is dominant: it's better in *both* columns. And `R > P` means the cooperative outcome is better than the defective one for both — which is why the equilibrium is a tragedy rather than just an outcome. Drop any of these inequalities and the dilemma dissolves; keep them and the trap is mathematically inevitable.

In the run-for-the-exit numbers, T = \$95 (sell first while they hold), R = \$90 (both hold), P = \$60 (both sell), S = \$50 (you hold, they sell). \$95 > \$90 > \$60 > \$50 — the ordering holds exactly, which is why `nash_2x2` returned mutual selling. When you're trying to decide whether a market situation *is* a prisoner's dilemma, this is the test: write down the four payoffs and check whether they line up `T > R > P > S`.

## The wedge: individually rational versus collectively optimal

Notice what the prisoner's dilemma is *not*. It is not a story about stupidity, panic, or irrationality. Both prisoners are perfectly rational. Both firms in the worked example are correctly maximizing profit. The bad outcome is not a *mistake* — it is the *equilibrium*. That distinction is the reason the prisoner's dilemma is so much more useful for understanding markets than "investors panicked."

If a crash were just panic — people acting against their own interest in a fit of fear — the fix would be obvious: stay calm, be the adult in the room, hold while everyone else loses their head. And sometimes that works. But the genuinely dangerous version of a crash is the one where holding is *the wrong move for you personally*, where the calm adult is the one who gets run over. In a true prisoner's dilemma, "be the patient one" is the sucker's strategy — it's the box where you get ten years while your partner walks. The market punishes the cooperator who cooperates alone.

This is why "I'll just get out before the music stops" is both the most common and the most self-defeating plan in finance. Everyone has that plan. Everyone planning to exit just before everyone else *is* the rush. We'll come back to this in the misconceptions section, but hold the thought: the structure of the game, not the temperament of the players, is what produces the crash.

## The run for the exit is a prisoner's dilemma

Now the payoff. Take a crowded, illiquid position — a small-cap stock, an emerging-market bond, a leveraged ETF, a thinly traded token. A lot of people own it. Bad news arrives. Everyone simultaneously faces the same two choices: **hold** (stay calm, sell slowly or not at all — cooperate) or **sell now** (dump into whatever bids exist — defect).

The figure below lays it out as a many-player version of the prison game. The shock hits, each holder independently faces the same decision, each runs the same dominant-strategy logic, and they converge on the same answer — sell first — which means the price gaps down for everyone. The green branch at the bottom shows the road not taken: if every holder had stayed calm, the same bad news would have produced an orderly, shallow decline instead of a crash.

![Run for the exit modeled as an N-player prisoners dilemma](/imgs/blogs/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once-2.png)

Why is this a prisoner's dilemma and not just nerves? Because of *liquidity*. Liquidity is the ability to trade size without moving the price — the depth of standing buy and sell orders. In a calm market it's deep; the moment everyone wants the same side, it evaporates. There is only so much *bid* (standing demand to buy) at any price. The first sellers hit the high bids and get a decent price. The later sellers find those bids already taken and have to accept lower ones. The last sellers find no bids at all and crater the price.

So selling *early* gets you a better price than selling *late*. That single fact turns holding into the sucker's strategy and selling into the dominant one — exactly the prison game's structure, now with a market clearing price doing the role of the prosecutor.

### The payoffs, computed

Let's put numbers on it so it's not hand-waving. Model two representative holders of a \$100 position each. Strategies: hold (cooperate) or sell now (defect). The payoff is the value each recovers:

- **Both hold:** the market absorbs the news in an orderly way; each recovers \$90 (a real but modest drop — bad news *is* bad news).
- **Both sell:** they swamp the bids, the price gaps down, and each recovers only \$60 in the fire sale.
- **You sell, they hold:** you hit the good bids first and recover \$95; they're left holding into the air pocket and recover \$50.
- **You hold, they sell:** you're the bagholder at \$50; they got out at \$95.

This is a genuine prisoner's dilemma — and the helper confirms the equilibrium is mutual selling:

```python
import data_gametheory as gt

A = [[90, 50], [95, 60]]   # you: recovered value of a $100 position
B = [[90, 95], [50, 60]]   # other holder (symmetric)
                           # 0 = hold (cooperate), 1 = sell now (defect)
print(gt.nash_2x2(A, B))
    # -> {'pure': [(1, 1)], 'mixed': None}   (sell, sell) is the unique Nash
```

The next figure plots your four payoffs. The two green/amber bars on the left are when the *other* holder stays calm — and notice that selling (\$95) beats holding (\$90) even then. The two red bars on the right are when the other holder sells — and selling (\$60) beats holding (\$50, the bagholder outcome) there too. Selling first wins in *both* columns. It's dominant. So both sell, and both land on \$60 — the dashed line shows the \$90 they each gave up.

![Bar chart of your payoff from selling versus holding in four cases](/imgs/blogs/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once-3.png)

#### Worked example: the cooperation loss of a fire sale

Let's quantify exactly how much the run destroys. Take the run-for-the-exit payoffs above and ask: what does the group get under the collectively optimal outcome versus the Nash outcome?

- **Collectively optimal (both hold):** each recovers \$90, so the pair holds \$90 + \$90 = \$180 of combined value.
- **Nash (both sell):** each recovers \$60, so the pair holds \$60 + \$60 = \$120 combined.

The difference — \$180 − \$120 = \$60 of combined value, or \$30 per holder — is the **cooperation loss**: real wealth that simply vanished because each holder did the individually rational thing. Nobody stole it. No outside shock destroyed it. It evaporated in the *gap between the bid the first seller got and the bid the last seller got.* Scale this from two holders to ten thousand and from \$100 to a \$50 billion market and you have the anatomy of a crash.

The grouped-bar figure makes the wedge visual: the tall blue bars are combined value, the green bars are one holder's share, and the arrow marks the \$60 of combined value that rational selling destroys.

![Grouped bar chart of combined value versus one holders value at the cooperative and Nash outcomes](/imgs/blogs/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once-5.png)

The intuition: a fire sale doesn't transfer wealth from sellers to buyers so much as it *shreds* wealth that orderly behavior would have preserved — and every participant individually chose to shred it.

### Timing is the whole game

Why is selling *first* so much better than selling *late*? Because the price you receive decays second by second as the bids get eaten. The timeline figure traces a single panic: at the calm open you could sell at \$100; two seconds after the shock the fast sellers are filling at \$95; eight seconds in the bids have been pulled and you're getting \$88; thirty seconds in you're selling into an air pocket at \$72; ninety seconds in the last seller dumps at the \$60 low.

![Timeline of the falling price each successive seller receives in a panic](/imgs/blogs/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once-7.png)

That decaying reward is the prosecutor's offer made continuous. Each instant you wait, the "confess now" payoff is still better than the "stay silent" payoff — so the rational move is always to sell *this* instant, which is why the whole crowd compresses into the first few seconds and the price detonates. The decay *is* the dilemma.

#### Worked example: should you be the patient holder?

Suppose you're tempted to be the calm adult — to hold a \$10,000 position through the news while everyone else dumps. Using the timeline above, work out your expected outcome two ways.

If you sell in the first wave (defect), you get roughly \$95 on the dollar: \$10,000 × 0.95 = \$9,500. If you hold while the crowd sells (cooperate alone), you're the bagholder selling into the \$60 low: \$10,000 × 0.60 = \$6,000. The cost of being the lone patient holder in a true run is \$9,500 − \$6,000 = \$3,500, or 35% of your position.

Now the crucial caveat that separates a real PD from a fake panic: holding only pays if *enough others also hold* so the price recovers toward the \$90 orderly mark. If you could somehow guarantee the others stay calm, holding to \$90 (\$9,000) beats panic-selling to \$95-then-it-recovers — but you can't guarantee that, because you can't see into the other rooms. The intuition: patience is the right move only when it's *coordinated*, and in an anonymous market it almost never is.

## More markets that are secretly prisoner's dilemmas

The run for the exit is the headline example, but the same structure runs through finance. Here are five more, each a prisoner's dilemma in disguise.

### The broker fee race to zero

For decades, US stock brokers charged fat fixed commissions — about \$70 a trade in the 1970s. Then commissions were deregulated, and the game changed shape. Each broker faced a choice every quarter: **hold prices** (cooperate, keep margins fat for everyone) or **undercut** (defect, win market share by charging less). Undercutting is dominant: if rivals hold, you steal their customers; if rivals cut, you must cut too or lose everyone. So commissions ratcheted down, round after round, until in October 2019 Charles Schwab cut online stock commissions to **\$0** and every major rival matched within days. The figure traces that descent.

![Pipeline of broker commissions falling round by round toward zero](/imgs/blogs/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once-4.png)

This is a *repeated* prisoner's dilemma — the same game played quarter after quarter — and it's worth a moment on why it still collapsed to zero. In a repeated game, players *can* sometimes sustain cooperation through the threat of future punishment ("if you cut prices, I'll cut mine and we both bleed forever"). But that only holds if the players value the future enough and can detect cheating — and a fragmented industry of dozens of brokers, where any one can quietly undercut, can't enforce it. We compute the exact discount-factor threshold for sustaining cooperation in a later post; for now, note that the fee war ended where the prisoner's dilemma predicts: at the competitive floor. (Brokers didn't go bankrupt — the \$0 commissions are paid for indirectly, often by selling customer order flow, which is its own game.)

#### Worked example: the broker's quarterly defection

Put one broker's quarter into numbers. Say the industry sits at a comfortable \$5 commission and a broker handles 10 million trades a quarter, booking \$5 × 10M = \$50M.

- If they **hold** at \$5 while a rival cuts to \$3, customers flee; suppose volume drops to 4 million trades: \$5 × 4M = \$20M.
- If they **cut** to \$3 first, they hold their 10M and maybe grab a rival's share to 13M: \$3 × 13M = \$39M.

\$39M (cut) beats \$20M (hold and get undercut). And if everyone has already cut to \$3, cutting to \$2.50 still beats sitting at \$3 and losing volume. The dominant move is always to cut, so the price spirals toward the cost floor. The intuition: in a fragmented industry of anonymous competitors, "let's all keep prices high" is the silent-prisoner strategy that someone always betrays.

### The analyst who can't go bearish

A sell-side equity analyst — someone at a bank who publishes buy/sell ratings on stocks — faces a quiet prisoner's dilemma. Honesty (a "sell" rating on an overvalued darling) is socially optimal: it informs investors. But individually, a "sell" rating gets you frozen out of management meetings, dropped from IPO syndicates, and yelled at by the bank's investment-banking side that wants the company's underwriting business. So the dominant move for each analyst is to stay bullish — to issue "buy" or the cowardly "hold." When every analyst defects toward bullishness, the *aggregate* signal becomes nearly useless. This is why, famously, "sell" ratings have historically made up only a single-digit percentage of all analyst ratings even at market tops. Each analyst is rational; the collective output is a permanently rose-tinted research industry.

The payoff structure is worth spelling out because it's a PD with the cooperation being *truth-telling*. If all analysts were honest (cooperate), the body of research would be trustworthy and valuable — the reward `R`. But any single analyst who stays bullish while others turn honest (defect) keeps their access and banking fees — the temptation `T` — at the cost of the readers who trust them. And an honest analyst surrounded by bullish ones is the sucker `S`: ignored, cut off, and eventually pushed out. So everyone drifts bullish, and the punishment `P` is a research industry nobody fully believes. The lesson for a reader of research is not "analysts are corrupt" — they're rational — but "read the *structure* of incentives behind a rating, not just the rating," and treat a rare genuine "sell" as the costly, and therefore credible, signal it is.

### The de-risking stampede

Big institutions run risk models that tell them to cut exposure when volatility spikes — strategies with names like "volatility targeting" and "risk parity" do this mechanically. Each fund's rule is individually sensible: when the world gets scary, hold less. But because thousands of funds run *similar* rules, a volatility spike triggers all of them to sell at once — which spikes volatility further, which triggers more selling. Each fund de-risking is cooperating with its *own* risk committee and defecting against the *market*. The collectively optimal move (everyone holds, volatility subsides) is unreachable because no fund can trust the others to hold, and "I de-risk while you don't" leaves you the only one exposed when it really blows up. February 2018's "Volmageddon" — when a spike in the VIX volatility index detonated a crowd of short-volatility products in a single afternoon — was exactly this stampede.

What makes the de-risking version especially insidious is that the defection is *automated* and *invisible until it fires*. No human chooses to panic; a risk model simply crosses a threshold and submits sell orders, and because the popular models share similar thresholds and similar inputs, they cross at nearly the same moment. From the outside it looks like a sudden, coordinated wave of selling out of nowhere — but it's just a few hundred independent machines each playing the dominant strategy their owners programmed in calmer times. The same dynamic shows up in trend-following funds (which sell as prices fall by design) and in the delta-hedging of options dealers (who are mechanically forced to sell into a falling market to stay hedged), all of them adding to the same one-directional flow. The honest lesson is that "everyone follows good risk management" does not add up to a well-behaved market; it can add up to a synchronized stampede, because good *individual* risk management and good *collective* outcomes are, once again, two different boxes.

### Short-term versus long-term (and ESG)

A subtle, slow-motion prisoner's dilemma runs through corporate behavior. A company that invests for the long term — research, worker training, not polluting — produces a collectively better world. But each company, and each fund manager judged on quarterly numbers, faces pressure to defect: cut the long-term spending, juice this quarter's earnings, buy back stock. If your rivals are all maximizing the short term and you're the lone long-term cooperator, you under-perform the index and lose your investors. So everyone defects toward short-termism. The same structure underlies a lot of the difficulty with environmental, social, and governance (ESG) goals: a single firm cutting emissions at a cost is the sucker if competitors don't, so "everyone pollutes a bit less" is collectively optimal but not a Nash equilibrium. This is why these problems usually require an *external referee* — regulation, an enforced standard, a binding contract — to change the payoffs. The prisoner's dilemma can only be escaped by changing the game.

This is also why moral exhortation alone — "be a long-term company," "be a responsible investor" — so reliably fails to move the needle. Telling players in a prisoner's dilemma to cooperate is asking them to volunteer for the sucker payoff while their competitors collect the temptation. The interventions that actually work are the ones that change the *numbers*: a carbon tax that makes polluting expensive enough that not polluting becomes the dominant move; a binding industry standard that punishes the defector; a compensation contract that pays a CEO on five-year results instead of this quarter's. Each of these is the same trick we'll see in every rescue — an external force rewriting the payoff matrix so that the cooperative box becomes the Nash equilibrium. Once you internalize that the only durable fix for a prisoner's dilemma is to change the game, you stop being surprised that markets full of decent, intelligent people keep producing collectively bad outcomes, and you start asking the more useful question: *who can change these payoffs, and will they?*

### The liquidation cascade

The most violent market prisoner's dilemma adds *leverage* — borrowed money — to the run for the exit, and turns it into a self-feeding loop. The figure traces the spiral: a price drop triggers margin calls (demands that leveraged holders post more cash or sell); those forced sales push the price down further; the lower price breaches *more* accounts' margin requirements; their brokers liquidate them at market; and the wave of forced selling craters the price into an air pocket with no buyers.

![Pipeline of a forced liquidation cascade feeding on itself](/imgs/blogs/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once-6.png)

What makes this worse than a plain run is that the selling isn't even voluntary at the bottom — it's *forced*. Each leveraged holder would love to cooperate (hold and wait for recovery), but the margin clerk takes the choice away. And each *broker* faces its own prisoner's dilemma: liquidate the client fast (defect) and you protect yourself but accelerate the crash; wait (cooperate) and you might eat the loss yourself if the client goes underwater. So brokers liquidate fast, and the cascade accelerates. Crypto markets, where leverage is extreme and liquidations are automated and public, show this in its purest form — a single large liquidation can chain into hundreds of millions of dollars of forced selling in minutes.

## The escape hatch: when the game repeats

So far every example has been a *one-shot* game — the prisoners play once, the holders dump once, and defection is inescapable. But many market interactions repeat: the same brokers compete quarter after quarter, the same dealers quote each other every day, the same handful of large funds face each other for years. When a prisoner's dilemma is played *repeatedly* against the *same* opponents, something remarkable happens — cooperation can become an equilibrium after all. Understanding when it can and can't is the difference between a fee war that ends at zero and a cartel that holds.

### Why repetition changes everything

In a one-shot game, there's no tomorrow to lose, so defecting costs you nothing in the future. In a *repeated* game, defecting today invites retaliation tomorrow. The classic cooperative strategy is **tit-for-tat**: cooperate on the first move, then simply copy whatever your opponent did last round. If they cooperate, you keep cooperating; the moment they defect, you defect right back. An even harsher version, **grim trigger**, cooperates until the first defection and then defects forever. Both strategies make defection expensive: you grab the one-time temptation payoff `T`, but then you're punished down to `P` for the rest of the game.

Whether that threat is enough to keep everyone cooperating depends on how much the players value the future versus the present. Game theorists capture this with a **discount factor**, written $\delta$ (delta), a number between 0 and 1: a payoff one round in the future is worth $\delta$ times a payoff today. A patient player who cares deeply about the future has $\delta$ near 1; a short-sighted player who only cares about right now has $\delta$ near 0. Cooperation survives only when players are patient enough that the future stream of mutual reward outweighs the one-time temptation to cheat.

### The threshold, computed

There's a clean formula for exactly how patient the players must be. Under grim-trigger punishment, mutual cooperation is a stable equilibrium of the repeated prisoner's dilemma if and only if:

$$\delta \geq \delta^{*} = \frac{T - R}{T - P}$$

The threshold $\delta^{*}$ is the gain from cheating once (`T − R`, how much temptation beats reward) divided by the total swing between temptation and punishment (`T − P`). If players discount the future *less* than this threshold — i.e. they're patient enough — cooperation holds; if they discount *more*, defection wins even in the repeated game. The series helper computes it directly:

```python
import data_gametheory as gt

    # Run-for-the-exit PD: T=95, R=90, P=60, S=50 (T > R > P > S)
print(gt.repeated_pd_delta_threshold(T=95, R=90, P=60, S=50))
    # -> 0.142857...
    # cooperation is sustainable iff the players' delta >= ~0.14
```

#### Worked example: how patient must the holders be?

Take the run-for-the-exit payoffs (T = \$95, R = \$90, P = \$60, S = \$50) and ask: if these two holders faced each other repeatedly, how patient would they need to be to *both choose to hold* every time?

Plug into the formula: $\delta^{*} = (95 - 90) / (95 - 60) = 5 / 35 \approx 0.143$. So as long as each holder values next round at least 14.3% as much as this round, the threat of mutual selling forever is enough to keep them both holding. That's a very low bar — these two would cooperate easily if they kept facing each other.

Now contrast a fee war where the temptation is huge and the punishment mild: say T = \$100 (grab all the flow), R = \$60 (split fat margins), P = \$58 (compete a little). Then $\delta^{*} = (100 - 60)/(100 - 58) = 40/42 \approx 0.952$ — the players would have to be *extraordinarily* patient to resist undercutting, which is why fragmented, impatient fee markets collapse to zero while a tight, repeated, observable cartel of a few patient players can hold. The intuition: cooperation in markets is fragile exactly where the one-time temptation is large, the players are many and anonymous, and cheating is hard to detect — which is precisely the run for the exit.

### Why the run for the exit stays a one-shot game

Here's the crucial point for traders: a crash is the *one place* repetition can't save you. A run for the exit is effectively a one-shot game even if the same people trade every day, because (a) the counterparties are anonymous — you can't punish "the person who sold ahead of me" because you'll never know who they were; (b) the stakes are concentrated into a single moment — there is no "next round" if the position is wiped out today; and (c) the future is heavily discounted in a panic — survival now dwarfs reputation later. All three forces push $\delta$ toward zero exactly when you need it high. That is why the cozy cooperation of a repeated game evaporates the instant the market turns into a stampede, and why "we've all played nice for years" is no protection at all in the one moment it matters.

## Common misconceptions

**"I'll just get out before everyone else."** This is the single most dangerous belief in markets, and it's wrong for a precise game-theoretic reason: *everyone has the same plan.* The plan to exit "just before the crowd" is identical for every member of the crowd, so the attempt to execute it simultaneously *is* the crowd. You cannot be reliably faster than a market full of people who are all trying to be fastest, many of them with co-located servers measuring their speed in microseconds. The honest version of the plan is "I'll sell on a pre-defined trigger before the news, accepting I might be early" — which is a different, and survivable, strategy.

**"The crash was irrational — people panicked."** Sometimes, sure. But the prisoner's dilemma shows that a crash needs *zero* irrationality to happen. If selling first is your dominant strategy, then selling first is the *rational* move, and a market of rational players all selling first produces the crash. Calling it "panic" hides the real mechanism and leaves you unprepared, because you'll wait for the "rational" buyers to step in — and in a true run, the rational move *is* to sell, so they don't.

**"If everyone would just stay calm, there'd be no crash."** True, and useless — that's the cooperative outcome, and the whole point of a prisoner's dilemma is that it is *not stable*. Telling a market to stay calm is like telling the two prisoners to both stay silent: collectively optimal, individually betrayed. "Everyone stay calm" only works when an outside force changes the payoffs — a circuit breaker that halts trading, a central bank that promises to buy, a lock-up that legally prevents selling. Hope is not a coordination mechanism.

**"More information would prevent these crashes."** Often the opposite. The run for the exit isn't caused by ignorance — everyone has the *same* information (the bad news) and reasons correctly from it to the same conclusion (sell). Giving everyone faster, clearer information can make the stampede *faster and more synchronized*, because everyone reaches the "sell first" conclusion at the same instant. The problem is the payoff structure, not the data.

**"This is just the tragedy of the commons / herd behavior with a fancy name."** They're cousins, but the distinction matters. Herd behavior implies people are *copying* each other mindlessly. The prisoner's dilemma is scarier: people aren't copying anyone — they're each independently computing the same dominant strategy *in isolation*. You don't need to see what others are doing to join a run; you only need to know that selling first beats selling late.

## How it shows up in real markets

These aren't toy examples. The prisoner's dilemma has a body count.

**Long-Term Capital Management, 1998.** LTCM held enormous, leveraged convergence trades, and its prime brokers all knew that if the fund was forced to unwind, every bank holding similar positions would dump into the same illiquid market. The dominant move for each bank was to sell its own version of the trade *first*, ahead of the LTCM unwind — which is exactly what would have crashed the market they were trying to escape. The Federal Reserve Bank of New York organized a roughly \$3.6 billion recapitalization by a consortium of banks in September 1998, precisely to *change the game* — to force the cooperative outcome (an orderly wind-down) that the banks could not reach on their own. It worked because the Fed acted as the external referee the prisoner's dilemma always needs.

**The 2008 money-market run.** When Lehman Brothers failed in September 2008, a money-market fund called the Reserve Primary Fund "broke the buck" — its share price fell below the sacred \$1.00. Every investor in every similar fund instantly faced the run-for-the-exit game: redeem now and get your dollar, or wait and risk being the last one out at less than a dollar. Redeeming first was dominant, so investors yanked hundreds of billions of dollars out of money-market funds in days, freezing the short-term funding that the entire economy runs on. The US Treasury had to guarantee money-market funds — again, an external referee changing the payoffs so that "wait" became safe. The deeper plumbing of that liquidity crisis, and how the policy response reversed it, is its own story; we link it below.

**Silicon Valley Bank, March 2023.** SVB is the textbook modern version, because it ran at the speed of mobile banking. The bank had taken losses on its bond portfolio, and once that became visible, every depositor faced the classic coordination game: leave your money and risk the bank failing, or pull it now. Pulling out first was the dominant move, and — crucially — venture capitalists could text each other and pull funds with a tap, so the run compressed into hours. Roughly \$42 billion was withdrawn in a single day. No depositor was irrational; each was correctly playing "get out first," and the sum of their correct choices killed a bank that, given a calm week, might have raised capital and survived. (A bank run is a close cousin of the PD with its own twist — there are *two* equilibria, calm and run — which we treat in full in a later post.)

**GameStop and the short squeeze, January 2021.** The famous squeeze had a prisoner's dilemma running on the *short* side. Hedge funds that had sold GameStop short faced a buy-to-cover stampede: as the price rose, each short-seller's dominant move was to buy back (cover) *first*, before rivals, because covering pushes the price up and the later coverers pay more. Their collective rush to cover first drove the price higher still — a run for the exit in reverse. (Squeezes get their own full treatment later in the series; here they're just another costume on the same game.)

**The "Volmageddon" volatility spike, February 2018.** On February 5, 2018, the VIX volatility index more than doubled in a day, and a crowd of products that were short volatility — betting calm would continue — got detonated. The structure was a de-risking stampede: each fund's rule said "buy back volatility exposure when it spikes," and every fund doing that at once spiked volatility further, forcing more buying. One large exchange-traded note, the XIV, lost about 96% of its value overnight and was shut down. Each fund followed a sensible risk rule; the collective result was a self-amplifying blow-up.

**Crypto liquidation cascades.** Because crypto offers extreme leverage and runs automated, transparent liquidation engines, it shows the cascade in its rawest form. A sharp move triggers a band of leveraged longs to be liquidated; the forced selling pushes the price into the next band of stop-losses and margin thresholds; those get liquidated too. Multi-hundred-million-dollar liquidation waves can chain through in minutes — the forced-selling spiral from the cascade figure, sped up and laid bare on a public ledger. We link a detailed on-chain treatment of lending and liquidations below.

**The "dash for cash," March 2020.** When the pandemic hit, even the safest assets — US Treasury bonds — sold off violently, which is bizarre, because Treasuries usually *rally* when investors are scared. The game-theoretic reading: every institution that needed cash faced the same run-for-the-exit dilemma, and the most liquid thing to sell was Treasuries, so everyone sold the safe asset first to raise dollars before everyone else did. The dealers who normally make markets in Treasuries hit their own balance-sheet limits and pulled back (their dominant move), so the deepest market on earth briefly seized. The Federal Reserve had to buy roughly \$1 trillion of Treasuries in a matter of weeks to break the run — the external referee again, this time forcing the cooperative outcome onto the world's most important market.

**OPEC and the cartel that keeps cracking.** Oil producers face a perpetual repeated prisoner's dilemma: collectively, restraining output keeps prices high and everyone rich (cooperate); individually, each country is tempted to quietly pump above its quota to grab extra revenue at the high price (defect). When enough members cheat, the agreement collapses, output floods the market, and the price crashes — as it did spectacularly in early 2020 when a Saudi-Russia production dispute briefly sent prices negative. OPEC survives at all only because it's a *repeated* game among a *small, identifiable* set of players who can detect and punish cheating — exactly the conditions, from the discount-factor math above, under which cooperation can hold. Even then it constantly frays, which tells you how strong the temptation to defect really is.

**The 1907 panic, before there was a Fed.** Long before electronic trading, the structure was identical. In October 1907, depositors at trust companies in New York faced the run-for-the-exit game and lined up to withdraw before the institutions failed. There was no central bank to act as referee, so the financier J.P. Morgan personally organized a pool of capital to backstop the solvent institutions — a private actor manufacturing the external force the prisoner's dilemma requires. The panic is one reason the Federal Reserve was created in 1913: lawmakers recognized that a market with no referee will, again and again, reach the only equilibrium it has, and that equilibrium is everyone running for the door at once.

## The playbook: how to play it

Seeing the prisoner's dilemma in a market isn't an academic exercise — it changes how you size, hedge, and exit. Here's the operating manual.

**Know which game you're in.** First, ask: *is this a one-shot, anonymous interaction, or a repeated one with people I'll face again?* The whole tragedy of the PD is concentrated in the *one-shot, anonymous* version — and that is exactly what a public, electronic market is. The counterparty who takes the other side of your panic sell is a stranger you'll never identify and never meet again. So your baseline expectation against anonymous one-shot counterparties should be: *they will defect.* They will sell first, undercut, and step away from the bid when you most need it there. Don't model the crowd as if it will cooperate with you; it won't, because it can't.

**Who's on the other side, and what's their payoff.** In a run, the other side splits into people who got out before you (they win), the bagholders behind you (they lose more), and the market makers who *widen their spread or pull their bids* the instant flow turns one-directional, because their dominant move is to stop catching falling knives. If you're planning to sell into a stressed market, the question is never "what's it worth" — it's "who is still bidding, and what is *their* incentive to keep bidding." Usually the answer is: nobody, and none. This is the dealer's side of every trade, and it's worth understanding in its own right — see the cross-link below.

**Your edge: be early, be hedged, or don't be in the crowd.** There are only three game-theoretically sound ways to survive a prisoner's-dilemma market:

1. *Be genuinely early* — exit on a pre-defined trigger (a level, a piece of news, a position-size limit) *before* the crowd's logic activates, accepting that you'll sometimes leave money on the table by being early. Being early on purpose beats trying to be fastest in the rush.
2. *Be hedged* — own the protection (a put option, a short against the position, a stop that's actually a hard order) so the crowd's stampede doesn't depend on your reaction speed. A hedge changes your payoff matrix so that "everyone sells" no longer ruins you.
3. *Don't be in the crowded trade at all* — the surest defense against a stampede is not standing where the stampede will run. Crowdedness itself (everyone leveraged, everyone long the same thing, everyone running the same risk model) is the precondition; size down or stay out when you can see the crowd forming.

**Invalidation: when this lens is wrong.** Not every sell-off is a prisoner's dilemma. If liquidity is deep and the holders are diverse, long-horizon, and not leveraged, the "sell first" payoff doesn't dominate — patience can genuinely pay, and the calm holder is rewarded, not run over. The PD lens applies specifically when (a) liquidity is thin or one-directional, (b) the holders are homogeneous and leveraged, and (c) selling first measurably beats selling late. When those conditions are absent, treat a dip as a dip, not a dilemma. The mistake in both directions is expensive: panicking out of a non-PD sell-off, or calmly holding into a real one.

**Spotting a prisoner's dilemma before it fires.** You can see the conditions assembling before the stampede. Watch for the three ingredients together: *crowding* (positioning surveys, fund flows, or on-chain data all pointing one way), *leverage* (margin debt high, funding rates stretched, open interest bloated), and *thin or one-directional liquidity* (a market where the bids vanish the moment selling starts). When all three line up, you are looking at a loaded prisoner's dilemma — the payoffs already obey `T > R > P > S`, and all that's missing is the spark. The spark itself is almost impossible to predict; the *loaded structure* is not, and the structure is what you can act on. A trade that's "obvious" and "everyone's in" is, by that very fact, the most dangerous kind, because the obviousness is what makes the exit a stampede.

**Sizing and exit.** Size positions in crowded, illiquid, or leveraged trades as if the exit will be a stampede, because if it goes wrong it will be. The right question at entry is "how much can I lose if I'm the *last* one out, not the first?" Concretely: if a calm exit costs you 5% but a stampede exit costs you 35% (the bagholder number from the timeline figure), size the position so that the 35% loss — not the 5% one — is survivable. Set your exit triggers in advance and make them mechanical, because in the heat of a run your in-the-moment judgment will be fighting the same dominant-strategy pressure as everyone else's — and that's a fight you don't want to be having in real time. The whole value of a pre-committed rule is that it lets your calm self bind your panicked self.

**Be the referee for yourself.** Every escape from a prisoner's dilemma in this post came from an *external force* changing the payoffs — the Fed, the Treasury, J.P. Morgan, a circuit breaker, a regulator. You can't summon the Fed, but you can install your own private referees: a hard stop-loss that takes the decision out of your hands, a hedge that pre-pays for the bad outcome, a position limit your past self imposed on your future self. Each of these is a commitment device that changes *your* payoff matrix so that the crowd's dominant strategy is no longer fatal to you. That is the practical translation of the entire idea: you cannot make the market cooperate, but you can change the game *you* are playing inside it.

The one sentence to carry out of here: *against anonymous, one-shot counterparties in a thin market, the equilibrium is everyone defecting at once — so don't plan to be the calm one; plan to be early, hedged, or absent.*

## Further reading & cross-links

- [The trade is a game: why markets are strategic, not random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the series opener on treating every trade as a strategic interaction rather than a bet against nature.
- [Nash equilibrium, best response, and the price as a truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce) — the deeper treatment of the equilibrium concept we used to find the (defect, defect) box.
- [Zero-sum, positive-sum, and the house: where trading profits come from](/blog/trading/game-theory/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from) — why the cooperation loss in a fire sale isn't a transfer but destroyed value.
- [The SIG / Susquehanna playbook: poker, game theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev) — how a real prop firm builds an entire culture around thinking one level deeper than the other player.
- [How an options market maker thinks: the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the dealer who widens the spread and pulls bids the instant your flow turns one-directional.
- [The 2008 financial crisis: the liquidity crisis and policy response](/blog/trading/macro-trading/2008-financial-crisis-the-liquidity-crisis-and-policy-response) — the money-market run as an external-referee rescue of a collective-action failure.
- [Analyzing lending and liquidations](/blog/trading/onchain/analyzing-lending-and-liquidations) — the forced-liquidation cascade laid bare on a public ledger, the rawest version of the run.

*This is educational, not investment advice. It explains how a market mechanism works and how to think about risk — it is not a recommendation to buy or sell anything.*
