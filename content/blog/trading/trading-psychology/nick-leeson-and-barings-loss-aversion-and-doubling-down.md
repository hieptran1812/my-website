---
title: "Nick Leeson and Barings: Loss Aversion and the Doubling-Down Spiral"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "How one trader's refusal to accept a tiny £20,000 loss — hidden in an error account and doubled again and again into the Nikkei — compounded into an £827 million catastrophe that destroyed a 233-year-old bank, and the loss-aversion psychology that drove every step."
tags: ["loss-aversion", "sunk-cost", "escalation-of-commitment", "rogue-trader", "nick-leeson", "barings", "risk-management", "trading-psychology", "disposition-effect", "doubling-down"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — Barings did not die from one giant bet. It died from a small loss its star trader could not bring himself to accept, hidden and then doubled over and over until it swallowed the whole bank.
>
> - The seed was tiny: an error of about **£20,000** in July 1992, buried in a secret "error account" numbered **88888** rather than booked as the loss it was.
> - Every time the position lost, Nick Leeson **doubled it** to win the money back — the martingale trap. Hidden losses reached roughly **£208 million** by the end of 1994.
> - On **17 January 1995** a magnitude-6.9 earthquake hit Kobe, Japan. Leeson was **short volatility** on the Nikkei — a bet the market would stay calm — and the gap blew the position apart. He doubled again, buying tens of thousands of Nikkei futures as it fell.
> - By **23 February 1995** the loss was about **£827 million** (roughly \$1.4 billion), more than **twice** the bank's capital. Barings — founded in **1762**, a **233-year-old** institution — was declared insolvent on 26 February and sold to ING for **£1**.
> - The one lesson that would have stopped all of it: **cut the first small loss.** A closed loss cannot compound; a hidden, doubled one can end you.

You have never lost \$1.4 billion. But you have almost certainly done the thing that got Nick Leeson there.

You bought something. It went down. Selling would have made the loss *real* — a number you would have to look at and own — so instead you told yourself the story had not changed, you would "give it room," and you held. Maybe you even bought more, lower, to bring your average price down. That move felt like conviction. It was actually the exact reflex that, scaled up and hidden inside a bank with no one checking, turned one junior trader's £20,000 mistake into the collapse of one of the oldest banks in the world.

This is the purest case study in trading psychology, because almost nothing about it is exotic. Leeson was not a criminal mastermind running a complex fraud. He was a young man who could not accept a small loss, and who happened to sit in a seat where no one could see him refuse to. The diagram below is the mental model the whole article follows: a small loss, refused, hidden, and doubled — a loop that pays off almost every day right up until the day it kills you.

![The doubling-down spiral: a small refused loss is hidden in account 88888, doubled to win it back, papered over on most days by a cooperative market, until one gap (the Kobe earthquake) makes it fatal and Barings collapses.](/imgs/blogs/nick-leeson-and-barings-loss-aversion-and-doubling-down-1.webp)

The picture above has one green box, and it is the most dangerous one. "Market cooperates — loss papered over — report a profit" is what happened on the overwhelming majority of days. Doubling down *usually works*. That is precisely why it is a trap: the strategy rewards you again and again, teaching you it is safe, until a single move you did not size for erases everything at once. The rest of this piece walks each box, and each one is a place the chain could have been broken.

This is educational, not financial advice. The round numbers in the *worked examples* are hypothetical so you can do the arithmetic in your head; every figure attributed to the real Barings collapse or to another market episode is sourced at the end.

## Foundations: how a small loss becomes a hidden one

Before any of the trading, you need five ideas. None of them require a finance background — they are facts about how a normal human mind prices a loss, plus a couple of pieces of market plumbing. A practitioner can skim this section; a beginner should not skip it, because every later mistake falls out of these definitions.

### Loss aversion: a loss hurts about twice as much as a gain feels good

**Loss aversion** is the finding that losses feel more intense than equivalent gains — not a little more, roughly *twice* as much. Losing \$100 delivers about as much pain as winning a bit over \$200 delivers pleasure. This is the single most important fact in behavioral finance, and it came out of *prospect theory*, the framework Daniel Kahneman and Amos Tversky published in 1979 (the work behind Kahneman's 2002 Nobel Prize in Economics).

The mechanism is a bent, asymmetric "value curve": you feel *changes* from a **reference point** — almost always your entry price, what you paid — and the curve that maps an objective outcome to a subjective feeling is shaped so that the loss side is steeper than the gain side.

![The prospect-theory value curve: shallow and concave over gains so you take small wins, but steep and convex over losses so a small loss feels huge and you turn risk-seeking, doubling the bet to claw back to the entry price.](/imgs/blogs/nick-leeson-and-barings-loss-aversion-and-doubling-down-2.webp)

Picture the curve above, because it is the engine of everything Leeson did. Two features do all the work. On the right (gains) the curve is **concave** — it flattens, so each extra pound of profit thrills you a little less, which nudges you to *lock in small wins*. On the left (losses) the curve drops **steeply** and is **convex** — and convexity below zero has a strange consequence: down here you become *risk-seeking*. Faced with a certain small loss versus a gamble that might erase it, the bent curve makes the gamble feel better. That is the psychological permission slip for "double down to get back to even." Hold that thought; it is the whole story.

### The disposition effect: cutting winners, riding losers

The **disposition effect** is what loss aversion looks like in an actual account: the documented tendency to *sell winners too early and hold losers too long*. You snatch the small, satisfying profit (concave gain side) and you cling to the loss, refusing to realize it (steep loss side), hoping it comes back. Leeson's entire scheme was the disposition effect with the brakes cut: he never, ever realized a loss. He just moved it somewhere no one would look. If you want the mechanism in isolation, see [loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect).

### Sunk cost and escalation of commitment

A **sunk cost** is money already spent that you cannot get back. The rational rule is brutal and simple: sunk costs should not influence the next decision — only future costs and benefits matter. Humans violate this constantly. The £20,000 was gone the instant the error happened; whether Leeson doubled the position afterward should have depended only on whether the *new* trade was good, not on the old loss. But a refused loss creates **escalation of commitment**: the more you have sunk into a losing course, the harder it is to abandon, because quitting means admitting the whole prior investment was a mistake. Each doubling was Leeson escalating his commitment to a decision he should have reversed on day one. More on the reflex itself in [sunk cost and averaging down into a loser](/blog/trading/trading-psychology/sunk-cost-and-averaging-down-into-a-loser).

### Short volatility: getting paid to bet nothing happens

Leeson's core market bet was **short volatility** — specifically, he *sold* options in a structure called a **short straddle**. Here is the plain-English version. An *option* is a contract that pays off if the market moves; the buyer pays a *premium* (a fee) for it. When you *sell* options, you collect that premium up front, and you keep it if the market stays quiet. So a short-volatility position is a bet that *nothing much happens*: you get paid a little for promising to cover someone else's loss if the market makes a big move.

The catch is the shape of the payoff. You collect a small, known premium in calm markets, but your loss if the market gaps is **unbounded** — the bigger the move, in either direction, the more you lose, with no ceiling. It is the financial equivalent of selling earthquake insurance: steady income right up until the earthquake. Which is not a metaphor here.

### The Nikkei, SIMEX, and margin: the plumbing

Three more pieces of plumbing make the rest legible. The **Nikkei 225** is the headline Japanese stock index — Japan's rough equivalent of the Dow, a single number that summarizes the market. A **future** on that index is a contract to buy or sell it at a set level on a set date; because you post only a fraction of the contract's value up front, futures give you *leverage* — a small amount of cash controls a large amount of exposure. Leeson traded these on **SIMEX**, the Singapore International Monetary Exchange, and on the Osaka exchange in Japan.

The word that will do the killing later is **margin**. When you hold a leveraged position, the exchange makes you post cash collateral, and it demands *more* cash — a **margin call** — every time the position moves against you. Margin is the exchange's way of making sure you can pay what you owe. It is also the mechanism that makes a hidden losing position impossible to hide forever: a position bleeding losses needs an ever-growing river of cash to meet its margin calls, and that cash has to come from somewhere visible. For a while, Leeson met his by wiring funds from London under the fiction that he was funding a low-risk arbitrage business. The margin calls were the loss trying to surface; the wired funds were the concealment holding it down. Eventually the two could not both keep growing.

### The error account and segregation of duties

An **error account** is a normal, boring piece of market plumbing: a holding pen where a trading desk parks trades booked by mistake (a "buy" keyed as a "sell", a wrong quantity) until they are corrected and zeroed out. Barings' Singapore error account was numbered **88888**. In healthy firms this account is reconciled daily by someone *other* than the trader.

That "someone other" is the load-bearing idea: **segregation of duties.** The person who places trades (the front office) must not be the same person who settles and confirms them and reports the numbers (the back office), because if one person does both, they can hide anything. In Singapore, Leeson ran *both*. He was the trader and the person who checked the trader. That single structural flaw is what turned a personal weakness into a bank-killing one — there was literally no independent set of eyes on account 88888.

#### Worked example: the asymmetric arithmetic of a £20,000 loss

Suppose you are Leeson in July 1992, and a junior on your desk has just cost the bank about \$30,000 (roughly £20,000 at the time) by selling futures that should have been bought. Run the two options through the value curve.

- **Book it.** You report a certain loss of \$30,000. On the steep loss side of the curve, a \$30,000 loss might *feel* like \$60,000-\$70,000 of pain — a bad afternoon, an awkward conversation, a dent in your reputation as the desk's rising star.
- **Hide it and try to trade out.** You move the \$30,000 into account 88888 and put on a position to win it back. Now the *certain* pain is replaced by a *gamble*: most likely you make the \$30,000 back and feel nothing, with a small chance the loss grows. Because the loss side of the curve is convex, that gamble feels *better* than the certain loss, even though its true expected value is worse once you account for the risk.

The bent curve does not just tolerate the second choice — it actively prefers it. A rational actor compares the two on expected value and picks "book it" because the hidden gamble adds risk for no edge. A loss-averse human compares them on *felt value* and picks "hide it."

**Intuition:** the reason a small loss is so hard to take is that your brain is not pricing the dollars, it is pricing the feeling — and the feeling is rigged to make the reckless choice the comfortable one.

## 1. The seed: why the first small loss was the fatal one

Every catastrophe in this story is downstream of one decision, and it was not a trading decision. It was the decision, in July 1992, to *not book* a roughly £20,000 error and instead park it in account 88888. Nick Leeson had joined Barings in 1989 on a £12,000 salary, been sent to run the Singapore futures operation in April 1992, and quickly become a celebrated earner. Admitting an early mistake threatened that image. So he hid it.

Notice what the hiding actually did. It did not make the £20,000 disappear — the money was still gone. What it changed was the *reference point*. Once the loss was hidden, Leeson was no longer a trader with a clean book who had suffered a small setback. He was a trader who was secretly *down*, and everything he did afterward was measured against getting back to zero. He had moved himself onto the steep, convex, risk-seeking part of the value curve and he never got off it.

This is the part beginners underestimate. The disaster was not the size of the first loss — £20,000 is a rounding error for a bank. The disaster was that the loss was *never realized*, so it never stopped being a live wound. A booked loss is a closed chapter. A hidden loss is an open one, and open losses recruit every bias you have — sunk cost, loss aversion, overconfidence, hope — into the project of healing them.

> A realized loss is a fact you have survived. An unrealized one is a threat you are still running from — and you will do almost anything to make a threat go away.

### What "just this once" costs

The hidden loss also flipped Leeson's incentives in a way that compounds. Once you are hiding one loss, a *second* small loss is even more unbearable, because booking it would risk exposing the first. So the second one gets hidden too. And now the third. Concealment is not a one-time act; it is a treadmill that speeds up. Each hidden loss raises the stakes of ever coming clean, which makes the next concealment feel more necessary, not less.

By late 1992 the account already held far more than the original error, and Leeson had crossed the line from "a trader who made a mistake" to "a trader running a hidden book." He would stay on that treadmill for more than two years.

## 2. The doubling engine: how £20,000 becomes £208 million

Here is the mechanical heart of the collapse, and it is a piece of gambling math older than markets: the **martingale**. The idea is seductive. If you lose, double your next bet. When you finally win, the doubled win covers all your prior losses plus a profit equal to your original stake. On paper, you cannot lose — as long as you have infinite money and the game lets you keep doubling.

You do not have infinite money. Neither did Barings. And that is the entire flaw.

![Doubling after every loss makes the position balloon 1x, 2x, 4x, 8x, 16x, 32x while the account looks calm; one loss on the last, largest bet wipes out every prior papered-over win and then the account itself.](/imgs/blogs/nick-leeson-and-barings-loss-aversion-and-doubling-down-4.webp)

Look at how the bet size explodes in the figure. Conceptually, doubling turns a string of small manageable bets into one enormous unmanageable one. The first few bars are almost invisible — 1x, 2x — which is exactly why it feels safe. But the position grows geometrically, and geometric growth is deceptively violent: after just ten doublings your bet is more than 1,000 times the original. The position that must go right to save you keeps getting bigger than every position that came before it, combined.

### Why "it works almost every time" is the whole problem

Run the probabilities and the trap becomes obvious. Suppose each round is a near coin-flip. The chance of losing one round is about one-half. The chance of losing five in a row is about one in thirty-two — small, comfortable, easy to dismiss. So on most sequences you double once or twice, the market wobbles back, you win, and you quietly zero out account 88888. You feel like a genius. The strategy *confirms itself*.

But you are not playing five rounds. You are playing hundreds, over years. And across hundreds of rounds, a five-in-a-row losing streak stops being a freak event and becomes a near-certainty *eventually*. When it lands, the bet that goes wrong is the 32x bet, and it takes down not just itself but every win you booked before it. The martingale does not lower your risk. It hides your risk, packing it into a rare, gigantic tail event — and then that tail event arrives with a specific date on it.

There is a formal name for the flaw the martingale ignores: **risk of ruin.** Ruin is an *absorbing state* — the instant your equity path touches zero, the game is over, and no later edge can revive you, because you have no stake left to bet. The martingale maximizes your probability of a small win on any given sequence while quietly maximizing your exposure to that absorbing barrier. A strategy that is "right" 99% of the time but touches zero on the other 1% is not a good strategy with bad luck; it is a bad strategy whose bill has not yet arrived. The mathematics does not care how many times you got away with it — the first time the path touches zero is the only time that counts, and doubling is a machine for eventually reaching it.

By the end of 1994, this engine had taken the original £20,000 and turned it into hidden losses of roughly £208 million, all buried in 88888 and reported upward as a *profitable* operation. Leeson was, on paper, one of the bank's best traders. The timeline below shows the compounding.

![From a £20,000 junior's error in July 1992 to £827 million by 23 February 1995: the hidden loss compounded across 32 months of doubling, invisible until the Kobe earthquake surfaced it.](/imgs/blogs/nick-leeson-and-barings-loss-aversion-and-doubling-down-3.webp)

#### Worked example: doubling to get back to even

Suppose you start a trade with a \$1,000 position and a simple rule: if it loses 10%, double the position rather than take the loss.

- Round 1: \$1,000 position, drops 10%, you are down \$100. Instead of booking the \$100, you double to a \$2,000 position.
- Round 2: the \$2,000 drops 10%, another \$200 lost. Cumulative loss \$300. You double to \$4,000.
- Round 3: the \$4,000 drops 10%, \$400 lost. Cumulative \$700. Double to \$8,000.
- Round 4: the \$8,000 drops 10%, \$800 lost. Cumulative \$1,500. Double to \$16,000.
- Round 5: the \$16,000 drops 10%, \$1,600 lost. Cumulative loss now \$3,100 — more than three times your original position — and to "get back to even" you now need a \$32,000 position, thirty-two times where you started.

Five ordinary 10% moves against you, and a \$1,000 idea has become a \$32,000 problem. Now imagine the sixth move is not 10% but a gap of 25% — an earthquake. On \$32,000 that is an \$8,000 loss in a single print, and there is no doubling your way out because you have run out of capital and the market has run out of patience (the margin call arrives).

**Intuition:** doubling does not fight the losing streak, it *feeds* it — every double makes the streak you are trying to escape more expensive to survive, until one ordinary bad move becomes an unrecoverable one.

### What this costs, and when it breaks

The cost of the doubling engine is invisible right up until it is total. That is its defining and most dangerous property. A strategy that loses a little every day announces its flaw and you fix it. A strategy that wins on 95% of days and catastrophically loses on the other 5% *feels* like a winner, banks real bonuses, earns real praise — and is secretly a time bomb. It breaks the first time the market makes a move larger than the capital you have left to double with. For Leeson, that move had a name.

## 3. The trade that broke it: short volatility meets the Kobe gap

Through late 1994 and into January 1995, Leeson's book was heavily **short volatility** — he had sold large quantities of Nikkei options in straddle structures, a bet that the Japanese market would stay in a calm range. In a quiet market this bleeds premium into your account every day; it is the "collect a little, promise to cover a big move" trade from the Foundations section, done at enormous scale. It is also, not coincidentally, a trade that *generates the appearance of steady profits* — perfect cover for a hidden loss you are trying to grow your way out of.

The payoff shape is a tent, and it is worth staring at.

![A short straddle's payoff is a tent: it earns the premium only if the market stays near the strike, and loses without bound as the market moves either way. On 17 January 1995 the Nikkei gapped down more than 1,000 points, falling off the left edge into the deep-loss tail.](/imgs/blogs/nick-leeson-and-barings-loss-aversion-and-doubling-down-5.webp)

Think of the tent as a promise: as long as the Nikkei stays near the strike (the figure marks it around 19,000, where the position made its maximum profit — the premium collected), you keep the money. But the moment the market makes a large move in *either* direction, you slide down one of the tent's edges into loss, and the edges do not stop — the loss grows without bound. Leeson had, in effect, sold a huge amount of "the Nikkei will stay calm" insurance.

At 5:46 in the morning on **17 January 1995**, a magnitude-6.9 earthquake struck the Japanese port city of Kobe, killing more than 6,000 people. It was the definition of an unpriced, unhedgeable shock. The Nikkei, which had been trading around 19,300-19,400, fell hard; over the following days it dropped more than **1,000 points**, sliding to around 17,800 by 23 January. Leeson's calm-market bet was now deep in the loss tail, and the losses on the short-volatility position were mounting fast.

A disciplined trader — or, more to the point, a trader with anyone watching him — takes the loss here. It is enormous but survivable if surfaced. Leeson did the opposite. He escalated. He began buying Nikkei 225 futures *outright*, in gigantic size, betting the index would rebound and rescue the whole book. This is the doubling engine again, now in its terminal phase: he was no longer just short volatility, he was massively long the market as it fell.

#### Worked example: a short straddle meets the gap

Suppose you sell a straddle on an index trading at 19,000, and for it you collect a premium worth \$500 per contract. Your best case is that the index sits at 19,000 at expiry: you keep the whole \$500, per contract, for doing nothing. Your breakevens — the points where the premium is exactly eaten up — sit at, say, 18,500 and 19,500.

- Index at 19,000 at expiry: you keep \$500 per contract. Lovely.
- Index at 18,500 (down 500): you break even — the loss on the position exactly cancels the premium.
- Index at 17,800 (down 1,200, the post-Kobe level): you are now 700 points past your breakeven. If each point past breakeven costs roughly \$50 per contract, that is a loss of about \$35,000 per contract, against the \$500 you collected — a loss seventy times the premium.

Now scale it. Leeson was not holding a handful of contracts; by late February his outright long Nikkei futures position alone reportedly reached around **61,000 contracts**, a notional exposure on the order of **\$7 billion**. A move of 1,000-plus points against a position that size is a loss measured in hundreds of millions, and every further point down deepened it.

**Intuition:** selling volatility means your worst case is not symmetrical with your best case — you can win a premium, but you can lose a fortune, and a single gap can hand you years of collected premium back in one morning.

### When this breaks: the margin call you cannot meet

Short-volatility and leveraged futures positions require **margin** — cash posted to the exchange as collateral, topped up as the position moves against you. As the Nikkei fell and Leeson's positions bled, the margin calls grew into figures that could not be hidden as "customer funds" much longer. The bank was wiring enormous sums to Singapore to meet them, believing it was funding a profitable arbitrage business. It was in fact funding the final, futile doublings of a book that was already lost. When the money and the story both ran out, on 23 February 1995, Leeson left a note reading "I'm sorry" and fled.

## 4. The missing control: why no one saw it for three years

Every trader has a bad instinct now and then. What made Leeson's fatal was that the one structural safeguard against a bad instinct — **segregation of duties** — did not exist in his corner of Barings. He ran the front office *and* the back office. He placed the trades and he settled them. He was the referee of his own game.

![The missing control: in Singapore, Leeson ran both the front office (placing trades) and the back office (settling and confirming them), so no independent set of eyes ever reconciled account 88888; with duties properly separated, the error account is flagged and zeroed in days, not hidden for years.](/imgs/blogs/nick-leeson-and-barings-loss-aversion-and-doubling-down-6.webp)

The contrast in the figure is the entire governance lesson of Barings. On the left, one person controls the whole chain, so account 88888 is never questioned and the loss stays invisible for nearly three years. On the right, the trader only trades; a separate back office settles and reports independently; a separate risk-and-audit function reconciles the books and checks limits daily — and in that world, an error account that never zeros out gets flagged within days. The individual psychology (loss aversion, sunk cost, the doubling reflex) is universal and unfixable; you cannot rewire a human brain. What you *can* fix is the structure, so that when a human brain does the human thing, someone independent catches it before it compounds.

This is why the collapse is studied in every risk-management course. Barings did not lack rules; it lacked *separation*. The internal-audit function had even flagged the concentration of duties in Singapore as a risk before the collapse. The warning was not acted on with enough urgency, and a 233-year-old bank paid for it.

#### Worked example: cut at £20,000 versus ride to £827 million

Put the two counterfactuals side by side with the real numbers.

- **The disciplined path:** book the £20,000 error in July 1992. The desk absorbs a trivial loss. Leeson has an uncomfortable week. Barings, at that point one of Britain's most storied banks, continues into its 234th year and beyond. Total cost: about **£20,000**.
- **The path taken:** hide the £20,000, double to recover it, hide the next loss, double again, for 32 months. By 23 February 1995 the loss is about **£827 million** — roughly \$1.4 billion — more than **twice** the bank's available capital. Barings is declared insolvent on 26 February and sold to the Dutch bank ING for **£1**. Total cost: the entire bank, plus thousands of jobs, plus a 233-year-old name.

The ratio between those two numbers is roughly **40,000 to 1**. The same reflex, the same account, the same market — the only variable was whether the first loss was taken or hidden.

**Intuition:** the price of accepting a loss is fixed and small; the price of refusing to accept it is uncapped, and it compounds — which means the cheapest risk decision you will ever make is the one to cut the first small loser.

## 5. Why the incentives all pointed the wrong way

There is a temptation to file Barings under "one bad apple" and move on. That misreads it. The deeper horror is that every incentive in the system pointed Leeson *toward* the spiral and away from confession, so that a normal person optimizing for the normal rewards would do roughly what he did.

Start with the fake profits. Because the real losses were hidden in 88888, the *reported* book looked spectacular — a young trader minting steady returns from a quiet arbitrage business. Those reported profits generated real bonuses, real praise, real promotions, and a growing reputation as the firm's star. Every one of those rewards raised the cost of ever coming clean, because confession would not just surface a loss, it would reveal that the celebrated success was fiction. Loss aversion operates on reputation and identity, not only on money: Leeson was, by 1994, loss-averse about his entire self-image as a winner. The steep side of the value curve had been extended to cover his career.

Now add the organizational blindness. London *liked* the profits and did not look hard at how a supposedly low-risk arbitrage desk was generating them, or why it kept needing enormous sums wired over to fund "client margin." When a business is printing money, the incentive to interrogate it weakens exactly when it should strengthen. The people who could have caught it were, in a quiet way, rewarded for not catching it. This is why "just hire honest traders" is not a control: the honest response to a hidden hole and a system that is happily paying you for pretending it is not there requires a kind of heroism most people do not have at 27 with a career on the line.

#### Worked example: how the hidden hole looked like a profit

Suppose a desk's *true* result for a period is a \$50 million loss, but \$60 million of losses are hidden in an error account while \$10 million of genuine gains are reported.

- **Reported to head office:** +\$10 million profit. The desk looks like a strong earner. A bonus pool is set — say 10% of reported profit, or \$1 million — and the trader is promoted.
- **Reality in the hidden account:** −\$60 million, growing, and now requiring cash to meet margin calls on the losing positions.
- **The wired "margin":** head office sends, say, \$40 million to Singapore believing it funds a profitable arbitrage book. It is in fact feeding the hole. On the books it looks like an asset (client balances); in truth it is a subsidy to a loss.

Every number the firm can *see* says "success — send more capital." Every number it cannot see says "catastrophe — pull the plug." The reporting system has inverted the signal, so the rational institutional response (fund your best desk) is the exact opposite of the correct one (shut it down).

**Intuition:** a hidden loss does not just grow in the dark — it actively disguises itself as a profit, so the organization pours fuel on the fire precisely because the fire is invisible.

## What it looks like at the screen

You will not get a memo telling you that you have entered the doubling-down spiral. It arrives as a set of feelings and small physical tells, and learning to recognize them in yourself is worth more than any rule. Here is what the spiral feels like from the inside, at the screen.

You notice you have stopped looking at the position's *current* merits and started staring at your *entry price* — the number in the "avg cost" column has become the center of your attention, and the live price is just measured as distance from it. You find yourself computing, almost involuntarily, "what would it take to get back to even?" rather than "is this a good trade right now?" That question is the tell. A trader evaluating the present asks about the future; a trader trapped by a loss asks about the past.

Your language, even in your own head, turns defensive and narrative. The position is no longer "a bet that isn't working"; it is "a great company the market is being stupid about," or "just noise," or "oversold." You are building a *story* to justify not selling, because selling would confirm the loss. You feel a flash of physical relief every time the price ticks back up a little — the same relief an addict feels, and just as fleeting. You feel a spike of something close to panic when it makes a new low, and that panic, crucially, makes you want to *add* rather than *exit* — because averaging down turns the new low into a lower average cost, which feels like fixing the problem instead of deepening it.

You start hiding the position from the people who would ask about it — closing the tab when a colleague walks by, not mentioning it to your partner, leaving it off the weekly review. Concealment is the loudest tell of all, because you only hide what you are ashamed of, and you are only ashamed of a decision you already know is wrong. Leeson's 88888 was that instinct given an account number. Yours might be a spreadsheet you have stopped updating, or a broker statement you no longer open. The urge is identical; only the scale differs. When you catch yourself doing any of this — fixating on the entry price, narrating instead of deciding, feeling relief-and-panic instead of thinking, hiding the position — you are on the convex side of the value curve, and the only move that helps is the one that feels worst: close it, book the loss, and get flat before the doubling reflex takes the wheel.

## Common misconceptions

**"Leeson was a criminal genius who ran an elaborate fraud."** The fraud was almost embarrassingly simple: a hidden account, faked documents to cover margin calls, and a lot of doubling. What made it lethal was not sophistication but *concealment plus scale plus no oversight*. The lesson is scarier than a criminal-mastermind story, because it means an ordinary person with an ordinary weakness could do it — if no one is watching.

**"The Kobe earthquake caused the collapse."** The earthquake was the trigger, not the cause. The cause was a hidden, doubled, unhedged short-volatility book that had already lost about £208 million *before* the quake. If Kobe had not happened, some other move — an interest-rate surprise, a political shock — would eventually have hit a position engineered to lose on any large move. Blaming the earthquake is like blaming the match instead of the gas leak.

**"Doubling down is always wrong."** Not quite — *averaging down on a loser* is the trap. Deliberately scaling into a position at better prices can be legitimate *if it was planned before you entered, sized in advance, and the thesis is intact.* The poison is specifically doubling *reactively, to recover a loss, in a size you did not plan, on a position whose thesis has broken.* The difference is whether the loss is driving the decision. When the loss is in the driver's seat, you are Leeson; when the plan is, you are an investor. See [sunk cost and averaging down into a loser](/blog/trading/trading-psychology/sunk-cost-and-averaging-down-into-a-loser) for where the line sits.

**"This can't happen anymore — controls are better now."** Controls are better, and yet Jérôme Kerviel lost Société Générale about €4.9 billion in 2008 and Kweku Adoboli lost UBS about \$2.3 billion in 2011, both using unauthorized positions hidden from risk oversight. The human reflex has not changed at all; only the specific control gaps move around. Any firm that lets one person both trade and check the trader is rebuilding the Barings flaw.

**"If I just have enough capital, the martingale works."** No amount of capital is infinite, and markets can move further than you can double. Worse, the more capital you have to double with, the larger the eventual blow-up. Capital does not defeat the martingale; it just raises the ceiling on how much it can cost you when it finally fails.

## How it shows up in real markets

The Barings mechanism — a refused loss, hidden and doubled, killed by a gap — is not a one-off. It is a template, and it recurs at every scale from a retail account to a global bank. Real instruments, real dates, real numbers.

### 1. Barings itself (1995): the archetype

The full sequence, one more time, because it is the reference case: a roughly £20,000 error in July 1992, hidden in account 88888; doubled and re-hidden until losses reached about £208 million by the end of 1994; a short-volatility Nikkei book blown open by the 17 January 1995 Kobe earthquake; a desperate escalation into roughly 61,000 long Nikkei futures (about \$7 billion notional); a final loss near £827 million by 23 February 1995 — more than twice the bank's capital. Barings, founded in 1762, was declared insolvent on 26 February and sold to ING for £1. Leeson was arrested in Frankfurt, extradited to Singapore in November 1995, sentenced to six and a half years in Changi prison, and released in 1999 after being diagnosed with colon cancer. Every element of the template is present in its purest form.

### 2. Société Générale and Jérôme Kerviel (2008): the same reflex, a bigger bank

In January 2008, Société Générale announced a loss of about **€4.9 billion** unwinding unauthorized positions taken by trader Jérôme Kerviel, who had built directional bets on European equity index futures with a notional value reported as high as **€49.9 billion** — larger than the bank's own market capitalization. Like Leeson, Kerviel had worked in the back office and understood exactly how to disguise positions from the controls. Like Leeson, the losses grew because they were hidden rather than booked. The bank had to liquidate the vast position into a falling market, crystallizing the loss. Thirteen years after Barings, a far larger and more modern institution reproduced the same essential failure: a trader who could hide from oversight and a loss that was doubled rather than cut.

### 3. UBS and Kweku Adoboli (2011): "temporary" losses that were not

In September 2011, UBS revealed roughly **\$2.3 billion** in losses from unauthorized trading by Kweku Adoboli on its equity-derivatives desk in London. Adoboli's account, in the aftermath, was strikingly Leeson-like: positions that moved against him were not booked but hidden and enlarged, in the belief that they were *temporary* and would be recovered. That belief — "this loss isn't real yet, I can trade out of it" — is the convex loss-side of the value curve talking. He, too, had spent time in the back office. He, too, was doubling to get back to even. The bank survived; the head of the investment bank and, ultimately, the CEO did not keep their jobs.

### 4. Sumitomo and Yasuo Hamanaka (1996): a decade of hidden doubling

Yasuo Hamanaka, chief copper trader at Japan's Sumitomo Corporation — nicknamed "Mr. Copper" — hid trading losses for roughly a decade, culminating in a disclosed loss of about **\$2.6 billion** (around 285 billion yen) in 1996. As with Barings, the losses were concealed in unauthorized accounts and grown through ever-larger positions, in Hamanaka's case an attempt to corner the physical copper market to prop up his own book. When the manipulation failed and copper prices fell, the hidden losses surfaced all at once. Hamanaka was sentenced to eight years in prison. Different commodity, different decade, identical psychology: a loss that was never allowed to be real.

### 5. Amaranth Advisors and Brian Hunter (2006): doubling in the open

Not every version hides from a firm's controls; some hide from *risk itself*. In September 2006, the hedge fund Amaranth Advisors lost about **\$6.6 billion** — roughly two-thirds of its \$9-plus billion in assets — in less than two weeks, on concentrated natural-gas futures bets by trader Brian Hunter. Amaranth's losses were not concealed from its own back office the way Leeson's were; the failure was one of *sizing and escalation*. Hunter kept scaling into a spread trade that kept moving against him, and the positions grew so large relative to the market that exiting them itself drove prices further against the fund. The doubling instinct does not require a rogue trader in the shadows. It only requires a losing position, a refusal to cut it, and enough size to make the exit impossible.

### 6. The retail version: the account that "isn't down until you sell"

The template runs at the smallest scale too, in ordinary brokerage accounts, every single day. A trader buys a stock at \$50; it falls to \$40; instead of booking the 20% loss, they buy more at \$40 to "lower the average" to \$45; it falls to \$30, and they buy again. The self-talk is "it isn't a loss until I sell," which is loss aversion dressed up as patience. The position swells exactly when the thesis is weakest, capital gets concentrated in the worst idea in the book, and one more leg down does outsized damage. There is no earthquake and no £827 million — but the shape is identical to Barings, and the driver is the same refusal to let a loss be real. If hope is doing the work of a plan, read [hope: the most expensive emotion in your book](/blog/trading/trading-psychology/hope-the-most-expensive-emotion-in-your-book).

The retail spiral even has its own modern accelerants. Leverage — margin accounts, options, contracts-for-difference, perpetual futures on crypto exchanges — lets a small trader replicate Leeson's core condition (a leveraged position that generates margin calls) in an app, on a phone, with a few taps. The zero-commission, gamified interfaces that reward frequent trading make cutting a loser feel like a defeat and adding to it feel like decisive action. And the same "it isn't real until I sell" story that kept Leeson in the market keeps a retail trader holding a position through a margin call, meeting it with fresh deposits, until the exchange liquidates the account automatically — the small-scale, fully-automated version of Barings being declared insolvent. The mechanism has been miniaturized and handed to everyone; only the number of zeros on the final loss has changed.

## The drill: four hard stops for the doubling-down spiral

Psychology you cannot fix; structure you can. The point of a drill is to put a mechanical stop in front of the reflex, so that when loss aversion reaches for the wheel there is already a barrier in place. Four rules, and each one maps to a specific place the Barings spiral could have been broken.

![The drill as a matrix: for each rule — cut the first small loss, never hide a loser, never double a loser, segregate duties — what Leeson did wrong, the disciplined rule that replaces it, and why that rule is the hard stop.](/imgs/blogs/nick-leeson-and-barings-loss-aversion-and-doubling-down-7.webp)

The matrix above is the whole protocol on one page. Walk it rule by rule.

**1. Cut the first small loss.** The single most valuable habit in trading is the willingness to take a small, certain loss now to foreclose a large, uncertain one later. Pre-commit the exit *before* you enter — a mechanical stop, sized so the loss is trivial — so that when the price hits it, the decision has already been made by a calmer version of you. Leeson's whole catastrophe is the counterfactual: had the first £20,000 been booked, there is no story. A closed loss cannot compound.

**2. Never hide a loser — surface every loss the same day.** The loss you conceal is the loss that grows, because concealment removes the only force that shrinks a bad position: another human being asking about it. Mark your book to market honestly, every day, in front of someone. If you are a retail trader with no boss, invent one: a trading journal you actually fill in, a weekly review with a peer, a partner who sees the statements. The [trading journal that actually changes behavior](/blog/trading/trading-psychology/the-trading-journal-that-actually-changes-behavior) exists precisely to be the independent eyes you otherwise lack.

**3. Never double a loser to get back to even.** Add to *winners*, on plan, never to losers, in a panic. Before every entry, ask the disposition-effect question: "Would I put this trade on, at this price, right now, if I did not already own it?" If the answer is no, you are not investing, you are averaging down to defend a sunk cost — and that is exactly the reflex that turned 1x into 32x. When the loss is driving the size, stop.

**4. Segregate duties — never be your own referee.** In an institution this is the front-office / back-office / risk separation Barings lacked. As an individual, it means never letting the part of your brain that *wants the position to work* be the same part that *decides whether it is working*. Externalize the judgment: a written plan set when you were calm, hard stops in the market, position-size limits you cannot override in the moment. The whole point is that when your loss-averse self reaches for the wheel, an independent system — a rule, a limit, a person — is already holding it. For how to build that external structure, see [rules, checklists and pre-commitment](/blog/trading/trading-psychology/rules-checklists-and-pre-commitment).

Run these as a literal pre-trade checklist. Before you size a position: is my stop set and mechanical? Is this loss, if it hits, trivial? Am I adding to a winner or defending a loser? Is anyone but me able to see this position? Four questions, thirty seconds, and they stand between you and the only mistake in trading that can end you rather than merely hurt you.

## When this matters to you

You will never run a bank's Singapore futures desk. But you will, guaranteed, hold a losing position someday and feel the exact pull Leeson felt — the certainty that this loss is not real yet, that one more trade will fix it, that booking it now would be admitting something you are not ready to admit. That feeling is not a character flaw unique to rogue traders; it is standard-issue human wiring, the same convex value curve in every skull. The difference between a bad week and a blown-up account is entirely in what you do in the first ten seconds after the loss appears — whether you cut it while it is small and certain, or hide it and let it become large and open.

The honest version of the risk: everything that can make money can lose it, and the positions that feel *safest* — the ones quietly collecting premium, winning on most days, papering over an old wound — are often the ones carrying the fattest hidden tail. A strategy that wins 95% of the time and ruins you the other 5% will feel like skill for a long time before it feels like catastrophe. Barings felt like a rising star's success story right up until the morning it did not exist.

So the practical takeaway is small and unglamorous, which is the point: take your small losses. Book them the day they happen. Never hide one, never double one to get even, and never be the only person who can see your own book. Those four habits are boring, they will cost you a stream of trivial losses you will resent at the time — and they are the entire difference between a career and a cautionary tale. This is educational, not advice; but of all the things in trading psychology, the willingness to let a small loss be real is the one most worth building into a habit before you need it.

## Sources & further reading

Primary and reference sources behind the figures in this post:

- [Bankruptcy of Barings Bank (1995)](https://www.britannica.com/event/bankruptcy-of-Barings-Bank) — Encyclopaedia Britannica. Collapse on 27 February 1995, roughly £830 million (about \$1.4 billion) in losses, account 88888, the Kobe earthquake trigger, and Leeson's six-and-a-half-year sentence.
- [Nick Leeson](https://en.wikipedia.org/wiki/Nick_Leeson) — Wikipedia. Timeline: joined Barings 1989 (£12,000 salary), Singapore posting April 1992, the ~£20,000 subordinate's error, the short straddle of 16 January 1995, total losses of £827 million (\$1.4 billion), extradition on 20 November 1995, release July 1999 after a colon-cancer diagnosis.
- [Barings Bank](https://en.wikipedia.org/wiki/Barings_Bank) — Wikipedia. Founded 1762; the UK's oldest merchant bank; declared insolvent 26 February 1995; acquired by ING for £1 with assumed liabilities, forming ING Barings.
- [The Barings collapse 25 years on](https://www.cnbc.com/2020/02/26/barings-collapse-25-years-on-what-the-industry-learned-after-one-man-broke-a-bank.html) — CNBC, 2020. Retrospective on the control failures and the hidden ~£208 million by end-1994.
- ["Doubling: Nick Leeson's trading strategy"](https://pages.stern.nyu.edu/~sbrown/leeson.PDF) — Stephen J. Brown & Onno W. Steenbeek (NYU Stern). Academic treatment of the martingale/doubling dynamic and the ~61,000-contract, ~\$7 billion Nikkei futures exposure.
- [2008 Société Générale trading loss](https://en.wikipedia.org/wiki/2008_Soci%C3%A9t%C3%A9_G%C3%A9n%C3%A9rale_trading_loss) — Wikipedia. Jérôme Kerviel, ~€4.9 billion loss, ~€49.9 billion in unauthorized notional positions, January 2008.
- [2011 UBS rogue trader scandal](https://en.wikipedia.org/wiki/2011_UBS_rogue_trader_scandal) — Wikipedia. Kweku Adoboli, ~\$2.3 billion loss, September 2011.
- [Sumitomo copper affair](https://en.wikipedia.org/wiki/Sumitomo_copper_affair) — Wikipedia, and [Washington Post (20 Sept 1996)](https://www.washingtonpost.com/archive/business/1996/09/20/sumitomo-loss-in-scandal-now-pegged-at-26-billion/42c6b92c-a751-4d92-83f2-b887e7fe812b/). Yasuo Hamanaka, ~\$2.6 billion (≈285 billion yen), disclosed 1996.
- [Brian Hunter (trader)](https://en.wikipedia.org/wiki/Brian_Hunter_(trader)) — Wikipedia. Amaranth Advisors, ~\$6.6 billion natural-gas loss, September 2006.

Sibling posts on this blog that dig into the specific biases at work here:

- [Loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect) — the value curve that made the first loss unbearable.
- [Sunk cost and averaging down into a loser](/blog/trading/trading-psychology/sunk-cost-and-averaging-down-into-a-loser) — where legitimate scaling ends and the doubling trap begins.
- [Hope: the most expensive emotion in your book](/blog/trading/trading-psychology/hope-the-most-expensive-emotion-in-your-book) — the emotion that keeps a losing position alive.
- [Rules, checklists and pre-commitment](/blog/trading/trading-psychology/rules-checklists-and-pre-commitment) — how to build the external hard stops the drill relies on.
