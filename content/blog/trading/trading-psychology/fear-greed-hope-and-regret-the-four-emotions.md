---
title: "Fear, Greed, Hope, and Regret: The Four Emotions That Move Every Trade"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A practitioner's map of the four emotions that drive every trading decision — what each one is, where it comes from in the brain, exactly how it distorts entries, holds and exits, and the drill that turns the reflex into a checklist step."
tags: ["trading-psychology", "behavioral-finance", "loss-aversion", "prospect-theory", "disposition-effect", "regret-theory", "fear", "greed", "risk-management", "discipline", "emotional-control", "decision-making"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Every trading mistake you have ever made was, underneath, one of four feelings taking the wheel: fear, greed, hope, or regret. Each is an ancient survival reflex that kept your ancestors alive and quietly wrecks your P&L on a price chart.
>
> - **Fear** panic-sells at the low, freezes you on planned entries, and makes you over-hedge — because the amygdala treats a red number like a predator.
> - **Greed** oversizes, chases extended moves, and refuses to take profit — because a winning streak floods the brain's reward circuit and inflates your bet.
> - **Hope** rides losers, moves or cancels stops, and quietly relabels a broken trade as an "investment" — this is the *disposition effect*, and the data on it is brutal.
> - **Regret** either fuels a revenge trade after a miss or freezes you into paralysis — the same painful counterfactual pointing in two opposite directions.
> - The one number to remember: a loss feels about **2.25 times** as intense as an equal-sized gain (Tversky & Kahneman, 1992). That single asymmetry explains most of the list above.
> - You cannot delete the feeling. You *can* name it and pre-commit the opposite action — that is the entire drill, and it is the difference between a professional and a donor.

The market does not take your money. Your amygdala does.

That sounds like a slogan, but it is close to literally true. When a position moves against you and you sell at the exact bottom, no external force compelled the click. A two-gram lump of tissue behind your eyes — the same one that flinches at a snake — fired first, and your finger followed. When you double your size after three wins and give it all back on the fourth, the extra risk did not come from your strategy. It came from a squirt of dopamine that made "more" feel like "safe."

Trading is the rare human activity where your instincts are not just unhelpful but *systematically inverted*: the feeling that would have saved your ancestor's life is precisely the one that empties your account. To trade well you do not need to become a robot. You need to know your four opponents by name, understand the biology that makes them so persuasive, see exactly where each one attacks the trade, and rehearse a specific counter-move for each. That map is the whole article.

![The four emotions that move every trade: fear, greed, hope, and regret, each with its adaptive origin, how it distorts a trade, and its screen tell](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-1.webp)

The diagram above is the mental model, and the rest of this piece is a tour of it. Four feelings, four ancient jobs, four ways of destroying a good plan, and four tells that give each one away in real time. We will build the science from zero first, then take each emotion apart, then walk a single trade through all four in sequence, and finish with a drill you can run tomorrow morning.

## Foundations: four ancient reflexes and the brain that runs them

Before we name the four emotions, we need three building blocks: how the brain makes a fast decision, why losses hurt more than gains, and which specific pieces of neural hardware are doing the trading. None of this requires a finance background — if you have ever felt your stomach drop, you already have the intuition. We are just going to make it precise and attach the real research to it.

### Two systems: the fast feeler and the slow thinker

The psychologist Daniel Kahneman — who won the 2002 Nobel Memorial Prize in Economics for this work — popularized a two-system model of the mind. *System 1* is fast, automatic, and emotional: it recognizes a face, flinches at a loud noise, and forms a gut feeling about a chart in milliseconds. *System 2* is slow, effortful, and deliberate: it multiplies 17 by 24, checks a thesis, and computes a position size. System 1 runs the show almost all the time because it is cheap and usually right about the world your ancestors lived in.

The trouble is that markets are not that world. A falling price is not a charging animal. But System 1 cannot tell the difference — it processes a plummeting position with the same circuitry it uses for physical threat. Kahneman and the psychologist Paul Slovic called the shortcut where a feeling stands in for a judgment the *affect heuristic*: instead of asking "what is the expected value of holding this?", System 1 asks "how does this make me feel right now?" and answers that instead. Every one of our four emotions is System 1 hijacking a decision that belonged to System 2.

> A trader's edge is not superior feelings. It is a reliable procedure for overriding them.

### Loss aversion: why a loss weighs more than a gain

Here is the single most important fact in all of trading psychology. In 1979, Kahneman and Amos Tversky published [*Prospect Theory: An Analysis of Decision under Risk*](https://www.jstor.org/stable/1914185) in *Econometrica*, and it rewired how economists think about risk. Their central finding: people do not evaluate outcomes by their final wealth, as classical economics assumed. They evaluate *changes* — gains and losses relative to a reference point (usually your entry price) — and **losses loom larger than equivalent gains**.

How much larger? In their 1992 follow-up, [*Advances in Prospect Theory*](https://link.springer.com/article/10.1007/BF00122574) in the *Journal of Risk and Uncertainty*, Tversky and Kahneman estimated the coefficient of loss aversion at about **2.25**. In plain terms: the pain of losing \$100 is roughly two and a quarter times the pleasure of gaining \$100. Their value function looks like this:

$$
v(x) = \begin{cases} x^{\alpha} & x \geq 0 \\ -\lambda\,(-x)^{\beta} & x \lt 0 \end{cases}
$$

where `x` is the gain (positive) or loss (negative) relative to your reference point, the exponents `α` and `β` (both ≈ 0.88) bend the curve so each extra dollar matters a little less than the last, and `λ` ≈ 2.25 is the loss-aversion multiplier that makes the curve *steeper below zero than above it*. You do not need to memorize the formula. You need to memorize the picture.

![Prospect theory value function: the curve is steeper below zero than above it, so a $100 loss feels like -2.25 units while a $100 gain feels like +1.0 unit](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-2.webp)

That kink at the origin — gentle on the right, a cliff on the left — is the source code for fear and hope both. It is why a paper loss screams at you and an equal paper gain merely murmurs.

#### Worked example: the +EV bet you will refuse

Suppose I offer you a single flip of a fair coin. Heads, you win \$150. Tails, you lose \$100. Should you take it?

The math is not close. The *expected value* — the average outcome if you could play it many times — is:

$$
EV = 0.5 \times (+\$150) + 0.5 \times (-\$100) = +\$25
$$

A positive \$25 per flip. Over a career of such bets you get rich. Yet in experiments, most people decline this exact wager. Why? Run it through the value function instead of the wallet. The \$150 gain feels like about +150 "joy units." The \$100 loss, multiplied by 2.25, feels like about −225 "pain units." So the *felt* value of the bet is:

$$
\text{felt} = 0.5 \times (+150) + 0.5 \times (-225) = 75 - 112.5 = -37.5
$$

Negative. It *feels* like a bad bet even though it is a mathematically great one. **Loss aversion makes you refuse good bets and — as we will see — cling to bad ones; it is the tax your feelings levy on every decision, and the first job of a trading process is to stop paying it.**

### The brain that does the trading

Loss aversion is not a metaphor; it has an address. In 2010, the neuroscientists Benedetto De Martino, Colin Camerer, and Ralph Adolphs studied two rare patients with focal damage to the *amygdala* — the brain's threat-detector — and published [*Amygdala damage eliminates monetary loss aversion*](https://www.pnas.org/doi/10.1073/pnas.0910230107) in *PNAS*. Those two patients, and only those, showed **no loss aversion at all**: they took the +EV gambles that everyone else's amygdala vetoes. Fear of loss lives, in large part, in that specific structure. When your position gaps down and you feel the cold flush, that is your amygdala doing exactly the job evolution gave it — it just cannot tell a drawdown from a predator.

Two more pieces of hardware complete the picture. In 2005, Camelia Kuhnen and Brian Knutson scanned people's brains while they made investment choices and published [*The Neural Basis of Financial Risk Taking*](https://web.stanford.edu/group/knutson/nfc/kuhnen05.pdf) in *Neuron*. They found that activation of the *nucleus accumbens* — a core node of the reward and dopamine system — spiked *before* people made **risk-seeking mistakes** (the neural signature of greed), while activation of the *anterior insula* — which processes disgust and anticipated pain — spiked before **risk-averse mistakes** (the signature of fear). The two errors have two different circuits, and both fire *in anticipation*, before you consciously decide.

And it is not only neural — it is hormonal. In 2008, John Coates and Joe Herbert sampled the saliva of 17 male traders on a real London trading floor over eight days and published [*Endogenous steroids and financial risk taking on a London trading floor*](https://www.pnas.org/doi/abs/10.1073/pnas.0704025105) in *PNAS*. A trader's morning **testosterone predicted his profit** that day; his **cortisol — the stress hormone — rose with market volatility** and with the variance of his own results. When markets get wild, cortisol climbs, and chronically elevated cortisol shifts you toward exactly the fearful, risk-averse choices that make you sell the bottom. Your risk appetite is not a fixed personality trait. It is partly a chemical state that the market itself is modulating in real time.

Hold those three facts: losses weigh ~2.25x (prospect theory), fear lives in the amygdala and greed in the reward system (both firing *before* you decide), and stress hormones swing your risk appetite with volatility. Now we can take the four emotions apart one at a time.

## 1. Fear: the trigger-puller

Fear is the oldest and fastest of the four. Its adaptive job is beautifully simple: detect a threat and get you away from it *now*, before the slow, deliberate part of your brain has finished forming a sentence. The neuroscientist Joseph LeDoux mapped a "low road" to the amygdala — a fast, crude sensory pathway that can trigger a fear response before the signal even reaches your conscious cortex. That is why you jump back from a garden hose that looks like a snake and only afterward feel silly. The system is designed to be trigger-happy, because a false alarm costs you a flinch and a real miss costs you your life.

On a price chart, that same machinery is a disaster, because the "threat" it is reacting to is a number that has already happened and that you voluntarily chose to expose yourself to. Fear in a trade shows up in three distinct distortions.

**Panic exits at the low.** The position gaps down, the loss crosses some private pain threshold, cortisol spikes, and System 1 issues a single command: *make it stop.* You sell — not at your planned stop, not at a level you reasoned about beforehand, but at whatever price ends the discomfort fastest, which is very often the exact low of the move. This is the most expensive single behavior in retail trading.

**Freezing on planned entries.** Fear does not only make you sell; it makes you fail to buy. The setup you spent all week waiting for finally triggers, and now — with real money about to be at risk — the amygdala floods you with reasons to wait "just one more candle." You freeze, the move goes without you, and then regret takes over (we will get there). The fear of being wrong beats the plan to be right.

**Over-hedging.** The subtler cousin. After a scare, fear makes you buy so much protection — puts, tight stops, tiny size — that you strangle the position's ability to work at all. You are now paying a permanent insurance premium against a threat that has mostly passed, and your winners are too small to matter.

![Fear versus the plan: the disciplined path exits at a pre-set stop for a known $500 loss, while fear cancels the plan and sells at the day's low for a $1,200 loss](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-3.webp)

The figure contrasts the two exits side by side. Same price action, same account — the only variable is who is driving. Let us put real numbers on it.

#### Worked example: what panic costs versus a plan

You buy 100 shares of a stock at \$50, so your position is worth \$5,000. Before you enter — while System 2 is calm and in charge — you decide the trade is wrong if it hits \$45, and you set a stop there. Your planned, pre-committed risk is:

$$
100 \text{ shares} \times (\$50 - \$45) = \$500
$$

Now the trade goes against you. The stock gaps straight through \$45 to \$42 on bad news. Your amygdala is now fully in command. The disciplined move was to be *already out* at \$45 (or as close as the gap allowed), taking the roughly \$500 loss you signed up for. Instead, fear whispers that \$42 could become \$35, the loss is now unbearable, and you sell everything at \$38 — which happens to be the low tick of the day.

$$
100 \text{ shares} \times (\$50 - \$38) = \$1,200
$$

By that afternoon the stock has closed back at \$44. Fear cost you the difference between a \$500 planned loss and a \$1,200 panic loss — an extra **\$700** — plus the rebound you will never participate in because you are flat and rattled. **The stop you set in calm is a message from your rational self to your terrified self; fear's entire trick is to convince you to ignore it at the exact moment it matters most.**

**When it breaks the other way:** fear is not always wrong. In a genuine regime change — a fraud revealed, a currency peg breaking, a 2008-style cascade — the fearful instinct to *get out now* is the correct one, and the disciplined trader's stop fires for the same reason. The skill is not "never feel fear." It is "obey a stop you set in advance, so the *same* action serves you whether the fear is signal or noise."

## 2. Greed: the reward chase

If fear is the flinch, greed is the reach. Its adaptive origin is just as old: when a scarce, high-value resource appears — ripe fruit, a fresh kill, a mate — the organism that seized *more of it, right now* out-reproduced the one that shrugged. The reward system evolved to make grabbing feel good and to make you want to do it again. On a trading screen, the "scarce resource" is a move that is already running, and the reward system does not care that chasing it is a terrible idea.

The biology here is the flip side of the fear circuit. Recall Kuhnen and Knutson: **nucleus accumbens** activation — the dopamine-driven reward anticipation — preceded *risk-seeking mistakes*. And recall Coates and Herbert: **testosterone rose on winning days**, and elevated testosterone is associated with increased risk appetite and confidence. Put those together and you get the single most dangerous pattern in trading: **the winning streak.** Each win releases reward chemistry and lifts testosterone, which raises confidence, which raises size — a positive-feedback loop that keeps escalating until one ordinary, expected loss lands on a wildly oversized position and erases the whole streak in a single trade.

![The greed loop: a win lifts dopamine and confidence, which lifts position size, until one normal loss on an oversized bet erases the entire streak](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-4.webp)

The loop in the figure is why so many blow-ups happen *right after* a trader's best month, not their worst. Greed expresses itself as three distortions.

**Oversizing.** The most direct and most lethal. After a run of wins, the position that used to feel scary now feels small, so you size up — not because the setup is better, but because the reward chemistry has recalibrated your sense of "normal." Position size is the one variable that turns a survivable loss into an account-ending one, and greed attacks it precisely.

**Chasing extended moves.** The stock is already up 20% on the day when you notice it. The disciplined entry — a pullback, a base, a defined-risk level — is long gone. But the move *feels* like free money walking away, so you buy at the top of the extension with your stop a mile away or nowhere at all. You have bought maximum risk at minimum reward.

**Refusing to take profit.** Greed's quietest form. The trade hits your target, System 2's plan says *sell*, but the position is green and "more" feels inevitable, so you hold. The winner round-trips back to your entry and the paper gain evaporates. You did not lose money on paper — which is exactly why it stings less and happens more.

#### Worked example: how oversizing turns a normal loss into a crisis

You run a \$50,000 account and your rule — set in calm — is to risk 1% per trade, or \$500. A stock trades at \$50; you decide the trade is wrong below \$45, a \$5 stop. Correct size:

$$
\frac{\$500 \text{ risk}}{\$5 \text{ per share}} = 100 \text{ shares} \quad (\$5{,}000 \text{ position, } 10\% \text{ of account})
$$

Now you have won five in a row. Greed has quietly redefined "normal," and you take the same setup at **400 shares** — a \$20,000 position, 40% of your account. The market does what markets do: the stock gaps *through* your \$45 stop and you actually get filled at \$40. Compare the two worlds:

| Position | Shares | Fill vs \$50 entry | Realized loss | % of \$50,000 account |
|---|---|---|---|---|
| Disciplined size | 100 | −\$10 | \$1,000 | 2% |
| Greed size | 400 | −\$10 | \$4,000 | 8% |

Same stock, same bad news, same slippage. The disciplined loss is an annoying 2% you recover next week. The greed loss is 8% in a single trade — and now *fear* takes over on the next one, and you are trading scared with a hole to dig out of. **Greed does its damage through size, and size is the one input the market cannot see and cannot punish for you — you have to govern it yourself, before the streak convinces you that you have it figured out.**

## 3. Hope: the loser's best friend

Hope is the gentlest-sounding of the four and, measured in dollars, quite possibly the most destructive. Its adaptive origin is the will to endure — the refusal to give up when the situation looks grim but rescue is still possible. Hope kept your ancestors walking through the desert one more mile to the water. It is a genuinely useful trait almost everywhere in life. Almost.

In a trade, hope has one job: to keep you in a losing position past the point where your own rules said to leave. Recall the prospect theory value function — that S-curve steeper below zero. There is a second feature of that curve we have not used yet: in the domain of losses, it is *convex*, which makes people **risk-seeking when underwater**. Faced with a certain \$500 loss (take the stop) versus a gamble that might break even or might lose \$2,000 (keep holding and hope), loss aversion pushes people toward the gamble — they will take on *more* risk to avoid *booking* a loss. That is hope, and it has a name in the finance literature: the **disposition effect**.

In 1985, Hersh Shefrin and Meir Statman named it in a paper whose title says everything: [*The Disposition to Sell Winners Too Early and Ride Losers Too Long*](https://www.jstor.org/stable/2327802) (*Journal of Finance*). Then in 1998, Terrance Odean put real brokerage data behind it in [*Are Investors Reluctant to Realize Their Losses?*](https://faculty.haas.berkeley.edu/odean/papers/gains%20and%20losses/AreInvestorsReluctant.pdf) (*Journal of Finance*). Studying 10,000 accounts at a discount broker from 1987 to 1993, he measured two ratios: the *proportion of gains realized* (PGR) and the *proportion of losses realized* (PLR). The result was stark and consistent:

- **PGR = 0.148** — investors sold about 14.8% of their winning positions.
- **PLR = 0.098** — they sold only about 9.8% of their losing positions.

Investors realized their gains at roughly **1.5 times** the rate they realized their losses — they cut winners and rode losers, the exact opposite of the trading-desk maxim *cut your losses and let your winners run*. Worse, Odean found the winners they sold went on to *outperform* the losers they kept over the following year, so the behavior was not even a lucky mistake. It was pure cost.

![The disposition effect: hope makes you snatch gains early and hold losers, quantified by Odean's finding of PGR 0.148 versus PLR 0.098](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-5.webp)

The matrix lays out the trap. In every cell, hope pushes you toward the behavior that feels better now and costs more later. It expresses itself three ways.

**Riding losers.** The core move. The trade is underwater, your stop is hit, but selling means *admitting* the loss — converting a "temporary paper setback" into a "realized mistake." Hope says it will come back. Sometimes it does, which is exactly what makes the habit so sticky; the intermittent reward trains you like a slot machine.

**Moving or removing the stop.** Hope rarely says "ignore your stop" outright — that would feel reckless. Instead it negotiates: *give it a little room; the stop was too tight; the real level is a bit lower.* So you slide the stop down. Then down again. A stop you are willing to move is not a stop; it is a suggestion, and hope will move it all the way to zero.

**Relabeling a trade as an "investment."** The most elegant self-deception in all of trading. You bought a stock as a two-day swing. It is now down 30% and you are telling yourself — and your spouse — that you are a *long-term investor* in a great company. You did not change your analysis. You changed the *story* to make the loss feel like patience instead of a mistake.

#### Worked example: the arithmetic of riding a loser

You buy 200 shares of a stock at \$30 — a \$6,000 position — and plan a stop at \$27, a \$600 risk. The stock falls to \$27. Now hope makes its pitch: *the market's just shaky today, give it room.* You cancel the stop. The stock keeps sliding to \$20.

$$
200 \text{ shares} \times (\$30 - \$20) = \$2,000 \text{ loss (paper)}
$$

You are now down \$2,000 instead of the \$600 you signed up for. But the real trap is what it now takes to get *back* to even, because percentage losses and the gains needed to erase them are not symmetric:

| Drawdown from \$30 | Price | Gain needed to recover |
|---|---|---|
| −10% | \$27 | +11% |
| −33% | \$20 | **+50%** |
| −50% | \$15 | **+100%** |

At \$20 you are down 33%, and you now need a **+50% rally** just to break even. At \$15 you would need the stock to *double*. Hope quietly moved you from a trade that needed a small bounce to survive into one that needs a heroic move to break even — and the deeper you ride it, the more heroic the required move becomes. **A loss you refuse to book does not go away; it compounds into a mathematically harder problem, which is why "cut it at the stop" is not cowardice — it is the only cheap exit you will ever get.**

## 4. Regret: the counterfactual engine

Regret is the most sophisticated of the four because it operates on trades that *did not happen* — or on the phantom of the version of events where you acted differently. Its raw material is the counterfactual: the vivid, painful mental image of the money you would have if only you had bought, sold, held, or waited. Its adaptive origin is genuinely valuable: regret is how we *learn* from decisions, updating our behavior by simulating the better path we missed. The problem is that in markets, the counterfactual is always available, always vivid, and almost always misleading.

In 1982, the economists Graham Loomes and Robert Sugden formalized this in [*Regret Theory: An Alternative Theory of Rational Choice Under Uncertainty*](https://academic.oup.com/ej/article-abstract/92/368/805/5220411) (*The Economic Journal*). Their insight: people do not evaluate an outcome in isolation. They compare it to what *would have happened* under the choice they rejected, and they feel real, decision-shaping pain from that comparison. A \$0 outcome feels fine if the alternative was −\$500 and agonizing if the alternative was +\$5,000 — even though your actual wealth is identical in both cases. Regret makes forgone gains *hurt*, and hurt drives behavior.

That pain points in two opposite directions, which is why regret is the trickiest emotion to spot.

![Two faces of regret: the same painful counterfactual drives either a revenge trade (action-regret) or paralysis (inaction-regret)](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-6.webp)

**The revenge trade (action-regret amplified).** You take a loss, and instead of accepting it as the cost of doing business, you experience it as an insult that must be *answered*. So you immediately put on another trade — bigger, faster, on a worse setup — specifically to "make it back." This is the gambler's tilt, transplanted onto a brokerage account. The revenge trade is defined by its motive: you are not trading the market in front of you, you are trading against the memory of the last loss. Nick Leeson's doubling-down (which we will get to) is the institutional-scale version of exactly this reflex.

**Paralysis (inaction-regret, or regret-avoidance).** The mirror image. Because acting and being wrong produces sharp regret, you protect yourself by *not acting* — you skip the valid setup, you sit out, you wait for a certainty that never comes. This feels safe because inaction has no visible loss. But the missed opportunities are just as real; you have simply chosen a form of loss that does not show up on a P&L statement and therefore does not sting. A trader frozen by regret-avoidance can go months without pulling the trigger on trades their own system flagged.

#### Worked example: the real cost of a revenge trade

A stock on your watchlist that you *did not buy* runs from \$40 to \$70 in a week — a move that would have made you \$3,000 on your usual size. You are sick about it. The counterfactual — *I'd be up \$3,000 right now* — plays on a loop. So you go looking for something, anything, to make it back, and you chase an unrelated stock at \$40 with no real setup, 300 shares, no plan.

$$
\text{The miss: } \$3,000 \text{ of imaginary money you never actually had.}
$$

$$
\text{The revenge trade: } 300 \text{ shares} \times (\$40 - \$35) = \$1,500 \text{ of real money, gone.}
$$

The missed trade cost you exactly **\$0** — you had no position, so your account never moved. The revenge trade, taken purely to soothe the regret of the miss, cost you a real **\$1,500**. **Regret converts an imaginary loss into a real one; the first job when you miss a move is to remember that a forgone gain is a story, not a debit, and it does not entitle you to take a worse trade.**

## What it looks like at the screen: catching the reflex in real time

Everything above is useful only if you can spot the emotion *while it is happening*, in the two or three seconds before your hand moves. The feelings do not announce themselves — they disguise themselves as analysis. Fear arrives dressed as prudence ("this could get much worse"), greed as opportunity ("I can't miss this"), hope as conviction ("I still believe in the thesis"), regret as justice ("I need to make that back"). Learning the *physical and behavioral tells* is how you catch them before the disguise works.

**Fear at the screen.** Your body tells you first: a jolt in the chest or gut when the number turns red, a held breath, a sudden urge to *do something now*. Behaviorally: you find yourself hovering over the sell button on a position that has not hit its stop, refreshing the quote every few seconds, and reaching for the phone to check news you would not have cared about if the trade were green. The giveaway sentence in your head is some version of *"just get me out"* — an urge to end a feeling rather than execute a plan.

**Greed at the screen.** The tell is a warm, expansive, slightly euphoric confidence — the feeling that you have it figured out. Behaviorally: you type a size larger than your rule and it feels *right*; you buy something not on your plan because it is "obviously going higher"; you cancel a take-profit order that is about to fill because "there is so much more in this." The giveaway phrase is *"just a bit more"* — and the single most reliable external tell is **sizing up right after a win.**

**Hope at the screen.** Quieter, and it lives in a losing position. The tell is that you have stopped looking at the chart and started reading news, forums, or the company's investor-relations page — anywhere that might supply a reason the trade will come back. Behaviorally: you drag your stop lower, you start using the word "investment" about a trade you entered as a scalp, and you calculate what the position needs "just to get back to even." The giveaway phrase is *"it'll come back."* If you catch yourself building a case *for* a position you already own that is *against* you, that is hope talking.

**Regret at the screen.** Two flavors, two tells. Revenge: a hot, urgent, almost angry need to put on a trade *immediately* after a loss or a miss, usually bigger than normal — the feeling is closer to indignation than to analysis. Paralysis: a valid setup triggers and you feel a leaden reluctance, a search for one more confirmation, a relief when the moment passes and you did not have to act. The giveaway phrases are *"I need to make that back"* (revenge) and *"I'll wait for the next one"* said about a setup that already qualified (paralysis).

The practical drill, which we will formalize at the end, is a single interrupt: the instant you notice a body sensation or one of those giveaway phrases, you *stop and name it out loud* — "that's fear," "that's the greed size," "that's hope talking." Naming a feeling engages System 2 and demonstrably reduces the feeling's grip; affective neuroscientists call it *affect labeling*, and it is the cheapest edge in trading.

## The lifecycle: which emotion attacks which stage

The four emotions are not randomly distributed across a trade. Each has a favorite stage of the trade lifecycle to attack, and knowing the schedule tells you which opponent to expect when.

| Trade stage | Dominant emotion | The distortion it produces |
|---|---|---|
| **Entry** | Greed (and fear) | Chasing an extended move, oversizing after a streak — or freezing on a valid entry out of fear of being wrong |
| **Hold** | Hope | Riding a loser past the stop, moving stops down, relabeling the trade as an "investment" |
| **Exit** | Fear | Panic-selling at the low, dumping the position to end the discomfort rather than at a planned level |
| **Re-entry** | Regret | Revenge-trading a loss or a miss, or paralysis that keeps you out of the next valid setup |

Read top to bottom, that table is the biography of a bad trade: you enter in greed, hold in hope, exit in fear, and re-enter in regret. That is not four separate mistakes — it is one continuous emotional cascade, each feeling handing off to the next. The best way to make it concrete is to watch it happen to a single position.

## How it shows up in real markets

### 1. One trade, four emotions: a −\$2,000 round trip

Let us walk one realistic trade through the entire cascade, tracking the running P&L at each handoff. The numbers are illustrative, but the sequence is one that plays out on thousands of screens every day.

![One trade, four emotions: entry in greed, hold in hope, exit in fear, re-entry in regret, with the running P&L falling from $0 to -$2,000](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-7.webp)

**Entry (greed).** A stock breaks out to new highs on heavy volume. It is already up 8% on the day when you see it — the disciplined entry is long gone — but the move feels like money walking away. You chase it, buying 100 shares at \$80, an \$8,000 position, with no written stop because "it's clearly going higher." Running P&L: **\$0.** It even ticks up to \$84, a paper +\$400, and greed tells you not to take the \$400 — there is *so much more* here.

**Hold (hope).** The breakout fails, as most chased breakouts do, and the stock reverses to \$72. You are down \$800 on paper. You *had* a mental stop at \$76, but crossing it hurt, so hope negotiated: *it's just a shakeout, give it room, the breakout thesis is still intact.* You cancel the mental stop. Running P&L: **−\$800.**

**Exit (fear).** Overnight, bad news. The stock gaps to \$66 and keeps sliding. Now the cold flush hits — cortisol, amygdala, the whole apparatus. The loss is unbearable and the only thought is *make it stop.* You panic-sell all 100 shares at \$64, which turns out to be within a few cents of the day's low. Running P&L: **−\$1,600 realized.**

**Re-entry (regret).** Over the next week the stock rebounds to \$74. Now the counterfactual is agony: *I sold the exact bottom, and I was right about the breakout in the first place.* Regret demands satisfaction. You revenge-buy 100 shares at \$73 with no new setup — you are trading against the memory of the loss, not the chart in front of you. It drifts to \$69. Running P&L: **−\$2,000.**

One position. Four feelings. Every single decision felt reasonable in the moment because each was dressed as analysis. The strategy was never tested, because the strategy never actually got to run — the emotions traded the account, and the account paid for all four of them.

### 2. Nick Leeson and Barings: hope and revenge, at institutional scale

The four emotions do not only bleed retail accounts; they have destroyed 233-year-old banks. In February 1995, a single 28-year-old derivatives trader named Nick Leeson bankrupted [Barings Bank](https://www.britannica.com/event/bankruptcy-of-Barings-Bank), Britain's oldest merchant bank and once banker to the Queen. The mechanism was the emotional cascade in this article, running unchecked with someone else's balance sheet.

Leeson took large, unauthorized positions on the Nikkei index and hid the losing ones in a secret error account numbered **88888**. This is hope and the disposition effect in its purest institutional form: rather than book losses, he concealed them and held on, certain they would come back. When the Kobe earthquake struck Japan on **17 January 1995** and sent the Nikkei tumbling, his hidden positions moved sharply against him — and instead of cutting, he did the revenge trade at scale, *doubling down*, buying enormous Nikkei futures positions to try to push the index back up and recover what he had lost. The index did not cooperate. By the time the positions were closed, the losses totaled roughly **£827 million** (about **\$1.4 billion**), more than the entire capital of the bank, and Barings collapsed. The [Reserve Bank of Australia's 1995 review](https://www.rba.gov.au/publications/bulletin/1995/nov/1.html) of the affair remains a sober account of how a single trader's concealed, escalating positions brought the whole institution down.

Strip away the scale and it is the exact same script: a loss that hope refused to book, then a revenge trade to "make it back," each step chosen to escape the pain of the last. The only differences between Leeson and the retail trader in the previous section were the number of zeros and the fact that Leeson was risking a bank's capital rather than his own.

### 3. Jesse Livermore's inversion: fear and hope, reversed

The single best piece of advice ever written about these emotions is nearly a century old. In *Reminiscences of a Stock Operator* (1923), Edwin Lefèvre's thinly fictionalized portrait of the legendary speculator Jesse Livermore, the narrator names the trader's enemies directly: *"The speculator's chief enemies are always boring from within. It is inseparable from human nature to hope and to fear."* And then he delivers the inversion that is the whole game:

> "The successful trader has to fight these two deep-seated instincts. He has to reverse what you might call his natural impulses. Instead of hoping he must fear; instead of fearing he must hope. He must fear that his loss may develop into a much bigger loss, and hope that his profit may become a big profit."

Read that twice, because it is the entire discipline in two sentences. The amateur *hopes* when the trade goes against them (and rides the loser) and *fears* when it goes their way (and cuts the winner). The professional does the opposite: they *fear* the losing trade (and cut it fast, before hope can talk them out of the stop) and *hope on* the winning trade (letting it run instead of grabbing the small gain). Prospect theory, discovered by Nobel laureates half a century later, is just the rigorous version of what Livermore's fear-and-hope inversion says: your instincts are backwards, and the job is to reverse them on purpose. Note the risk, too — Livermore himself made and lost several fortunes and died broke, a reminder that knowing the rule and *living* it under fire are very different things.

### 4. Barber and Odean: the measurable cost of trading on feeling

If you want the emotions priced in aggregate, the cleanest number comes from Brad Barber and Terrance Odean's [*Trading Is Hazardous to Your Wealth*](https://faculty.haas.berkeley.edu/odean/papers/returns/individual_investor_performance_final.pdf) (*Journal of Finance*, 2000). They studied **66,465 households** at a large discount broker from 1991 to 1996. The finding: the average household turned over about **75% of its portfolio each year**, and the most active traders earned an annual return of **11.4%** while the market returned **17.9%** over the same period — a gap of roughly six and a half percentage points a year, most of it self-inflicted through overtrading and the costs and timing errors it produces. The authors attributed the pattern to overconfidence — the cognitive cousin of greed. This is the four emotions with the receipts attached: when feelings trade the account, the account underperforms the index it could have simply bought and held. (This is educational, not a recommendation to buy or sell anything.)

## Common misconceptions

**"Discipline means feeling nothing."** No. Professionals feel fear, greed, hope, and regret as intensely as anyone — the De Martino study found that only people with *amygdala damage* stop feeling loss aversion, and you do not want to be one of them. Discipline is not the absence of the feeling; it is the presence of a procedure that runs *despite* the feeling. Aim to act right while afraid, not to stop being afraid.

**"I need to work on my psychology before I fix my process."** Backwards. Emotions get their leverage from ambiguity — from decisions you have not made in advance. A pre-written entry, a hard stop, and a fixed position size *remove* the moments where fear and greed can vote. The fastest way to improve your psychology is to build a process that gives your emotions fewer decisions to hijack. Rules are a form of emotional armor.

**"Cutting a loss means I was wrong."** A stop is not a verdict on your intelligence; it is the price of admission to a probabilistic game. In a strategy that wins 50% of the time with wins twice the size of losses, being stopped out is not a mistake — it is one of the losing half you were *paid* to accept. Hope's whole con is to reframe a normal, planned loss as a personal failure so that booking it feels like an admission rather than a routine.

**"A paper loss isn't a real loss until I sell."** This is the most expensive sentence in trading, and it is false. Your position is worth what it is worth *right now*, regardless of whether you have clicked sell. Refusing to book a loss does not preserve capital; it just hides the mark and, as the drawdown-recovery table showed, converts a small problem into a mathematically larger one while you wait.

**"Big gains come from big size."** Big *blow-ups* come from big size. Durable gains come from surviving long enough for a positive edge to compound, and survival is a function of *not* letting greed oversize you into a loss you cannot recover from. The trader who risks 1% and compounds beats the one who risks 20% and eventually meets the trade that ends them.

**"If I just had more conviction, I wouldn't get shaken out."** Conviction is often hope wearing a nicer suit. The market does not reward how strongly you believe; it rewards being positioned correctly and managing risk. "More conviction" is precisely the feeling that talks you into moving the stop. Trust the process you built in calm over the feeling you have in the storm.

## The drill: name it before you act

Knowing the four emotions changes nothing by itself. What changes your P&L is a rehearsed, physical routine that fires *between* the feeling and the click. Here is the whole protocol.

**Step 1 — Build the rules in calm.** Emotions attack ambiguity, so eliminate it before the session. For every trade, write down *before you enter*: the entry level, the hard stop, the position size (as a fixed fraction of your account — say 1% risk), and the take-profit or trailing rule. A decision made in advance cannot be hijacked in the moment, because there is nothing left to decide. This is the foundation the rest rests on.

**Step 2 — Keep the emotion-label journal.** This is the core drill. The instant you feel the urge to deviate from the written plan — to move a stop, to size up, to chase, to bail early, to revenge-trade — you *stop* and write one line, before you act:

> *Feeling: [fear / greed / hope / regret]. Trigger: [what just happened]. Urge: [what I want to do]. Plan says: [what I wrote in calm].*

The act of naming the feeling and its trigger is not busywork — it is *affect labeling*, and it demonstrably reduces the emotion's grip by pulling the deliberate System 2 back into a decision that System 1 was about to make alone. Over a few weeks the journal also becomes a personal map: you will find you have a *signature* emotion — most traders lean fear-and-hope or greed-and-regret — and knowing yours tells you which opponent to watch for.

**Step 3 — Run the counter-move.** For each emotion, you have *one* pre-planned opposite action, rehearsed until it is automatic. This is the table to tape to your monitor.

![The counter-move table: for each of the four emotions, the impulse it produces and the single pre-committed counter-move that neutralizes it](/imgs/blogs/fear-greed-hope-and-regret-the-four-emotions-8.webp)

- **Fear → obey the written stop, and never sell faster than your plan.** If the price has not hit your pre-set stop, your hand does not touch the sell button. The stop is the only exit decision fear is allowed to make, and you made it in calm.
- **Greed → the size is fixed *before* entry, and profit is taken at pre-set targets.** Position size is decided by the 1%-risk rule, not by how confident you feel. Confidence is not information.
- **Hope → the stop is a fact, not a feeling; it never moves down.** You may trail a stop *up* to protect a winner. You may never move it *down* to give a loser room. A stop that moves down is hope with a mouse.
- **Regret → the next trade needs its own setup; a miss is not a signal.** After a loss or a miss, the only valid next trade is one that independently qualifies on your rules. "Making it back" is not a setup.

**Step 4 — Add a circuit breaker.** Decide in advance the daily loss (say, 3% of the account) or the number of consecutive losers (say, three) at which you *stop trading for the day*, no exceptions. This is the institutional-grade defense against the revenge cascade — the rule that would have saved Nick Leeson from himself. When the amygdala and the dopamine system have both been triggered, the only reliable move is to remove yourself from the button entirely and let the chemistry subside.

None of this makes the feelings go away. Fear will still flush your chest when the number turns red; greed will still make a bigger size feel right after a win. The drill does not delete the emotion. It inserts a name and a pre-committed action into the half-second between the feeling and the trade — and that half-second is the entire difference between a trader and a donor.

## When this matters to you

Every technique in this article is really one skill wearing four masks: the ability to notice a feeling, name it, and run a pre-committed action instead of the reflex. That skill is not only for people with a Bloomberg terminal. It is the same muscle that stops you from panic-selling your retirement fund in a crash (fear at the exit), from piling into whatever asset just tripled (greed chasing an extended move), from holding a broken position for years because selling means admitting a mistake (hope riding a loser), and from making a rash financial decision to "make back" a loss you are still angry about (the revenge trade). Anyone who has money exposed to markets is already playing this game, whether or not they trade actively.

The honest closing note is the one Livermore's life illustrates: knowing the rules and living them under pressure are different skills, and the second one is never finished. You do not graduate from your emotions. You build a process that survives them, you keep the journal that catches them, and you accept that the four opponents will show up for every session for the rest of your career. The professional is not the person who stopped feeling fear, greed, hope, and regret. It is the person who learned to trade anyway.

*This article is educational and describes mechanisms and history; it is not individualized financial advice. Every strategy that can make money can lose it — size accordingly, and never risk money you cannot afford to lose.*

## Sources & further reading

- Daniel Kahneman & Amos Tversky, [*Prospect Theory: An Analysis of Decision under Risk*](https://www.jstor.org/stable/1914185), *Econometrica* 47(2), 1979 — the origin of loss aversion and the value function.
- Amos Tversky & Daniel Kahneman, [*Advances in Prospect Theory: Cumulative Representation of Uncertainty*](https://link.springer.com/article/10.1007/BF00122574), *Journal of Risk and Uncertainty* 5, 1992 — the source of the ~2.25 loss-aversion coefficient (α = β ≈ 0.88, λ ≈ 2.25).
- Hersh Shefrin & Meir Statman, [*The Disposition to Sell Winners Too Early and Ride Losers Too Long*](https://www.jstor.org/stable/2327802), *Journal of Finance* 40(3), 1985 — names the disposition effect.
- Terrance Odean, [*Are Investors Reluctant to Realize Their Losses?*](https://faculty.haas.berkeley.edu/odean/papers/gains%20and%20losses/AreInvestorsReluctant.pdf), *Journal of Finance* 53(5), 1998 — PGR = 0.148 vs PLR = 0.098 across 10,000 accounts.
- Brad Barber & Terrance Odean, [*Trading Is Hazardous to Your Wealth*](https://faculty.haas.berkeley.edu/odean/papers/returns/individual_investor_performance_final.pdf), *Journal of Finance* 55(2), 2000 — 66,465 households; most active traders 11.4% vs market 17.9%.
- Graham Loomes & Robert Sugden, [*Regret Theory: An Alternative Theory of Rational Choice Under Uncertainty*](https://academic.oup.com/ej/article-abstract/92/368/805/5220411), *The Economic Journal* 92(368), 1982.
- Camelia Kuhnen & Brian Knutson, [*The Neural Basis of Financial Risk Taking*](https://web.stanford.edu/group/knutson/nfc/kuhnen05.pdf), *Neuron* 47(5), 2005 — nucleus accumbens vs anterior insula and risk mistakes.
- Benedetto De Martino, Colin Camerer & Ralph Adolphs, [*Amygdala damage eliminates monetary loss aversion*](https://www.pnas.org/doi/10.1073/pnas.0910230107), *PNAS* 107(8), 2010.
- John Coates & Joe Herbert, [*Endogenous steroids and financial risk taking on a London trading floor*](https://www.pnas.org/doi/abs/10.1073/pnas.0704025105), *PNAS* 105(16), 2008 — testosterone predicts P&L; cortisol rises with volatility.
- Edwin Lefèvre, *Reminiscences of a Stock Operator* (1923) — the Livermore fear-and-hope inversion.
- [Bankruptcy of Barings Bank](https://www.britannica.com/event/bankruptcy-of-Barings-Bank), *Encyclopædia Britannica*, and the [Reserve Bank of Australia's 1995 review](https://www.rba.gov.au/publications/bulletin/1995/nov/1.html) of the Leeson affair.
- On this blog: [Why your brain is bad at markets](/blog/trading/trading-psychology/why-your-brain-is-bad-at-markets), [Narrative addiction: when a good story beats the data](/blog/trading/analyst-edge/narrative-addiction-when-a-good-story-beats-the-data), and [Thesis broken or just noise: the hardest call you make](/blog/trading/analyst-edge/thesis-broken-or-just-noise-the-hardest-call-you-make).
