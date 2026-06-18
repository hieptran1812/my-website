---
title: "The Greater Fool and Rational Bubbles: The Musical Chairs Game"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Why it can be individually rational to knowingly buy something overvalued, why the music-stops timing game makes everyone plan to exit early and almost nobody manage it, and how to tell when you have become the greater fool."
tags: ["game-theory", "rational-bubbles", "greater-fool-theory", "musical-chairs", "trading", "market-bubbles", "coordination-game", "prisoners-dilemma", "behavioral-finance", "risk-management"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A bubble is not a crowd of fools; it is a crowd of people each correctly betting they can sell to the next person before the bid disappears. That is a *game*, not a mistake — and the question that decides whether you win is never "is this worth the price?" but "who is on the other side, and will there still be a buyer above me when I want out?"
>
> - **Greater-fool theory:** you can rationally pay above fair value if you expect to resell higher to someone else — the price you pay is a bet on the *next buyer*, not on the asset.
> - **Rational-bubble models** show a bubble can persist as long as each holder believes there is a high enough probability of selling higher to cover the crash risk — the math below pins the exact crash probability where riding stops being worth it.
> - The exit is a **coordination-and-timing game** with a prisoner's-dilemma core: selling before the others is the best move whatever they do, so the calm exit everyone wants collapses into a simultaneous rush — cross-linked to the [prisoner's dilemma post](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once).
> - **"I'll get out in time" is the universal losing belief:** everyone plans to, the exits are narrow, and by the time the top is obvious it is already behind you.
> - **The one number to remember:** in the model below, with a +40% up-move if the bubble survives and a crash back to 30 if it stops, riding one more period turns negative once the crash probability passes **q\* ≈ 0.36** — past that, the only winning move is to already be out.

In late January 2021, a near-bankrupt video-game retailer's stock went from about \$20 to an intraday high near \$483 in two weeks. Message boards filled with people who said, openly, that they knew GameStop was not "worth" \$400 a share by any normal measure of a struggling brick-and-mortar business. They were not confused about the fundamentals. They were playing a different game: buy now, and sell to the next wave of buyers who would push it higher. For a while, that bet paid spectacularly — the people who bought at \$40 and sold at \$300 made real money, taken from the people who bought at \$300. Then, in the first days of February, the new buyers stopped arriving, and the stock fell from near \$500 to about \$53.50 within a week — roughly an 89% collapse. Everyone who had said "I'll sell before it drops" discovered that almost no one actually did.

This is the most uncomfortable idea in markets: it can be *individually rational* to knowingly buy something you believe is overvalued. Not because you are wrong about its value, but because value is not the only thing that determines whether you make money. If you can resell it higher to someone else, you profit — and you can keep profiting right up until the moment there is no "someone else" left. The name for the person you are counting on is the **greater fool**, and the name for the whole structure is the **musical-chairs game**: the music plays, everyone dances around the chairs, and the only thing you have to do to win is not be standing when it stops. The catch — the entire catch — is that everyone is trying to do exactly that, the chairs are far fewer than the players, and the music stops without warning.

The graph above is the mental model we are going to build from zero. Fair value sits on the left as the anchor. You buy above it, not because it is worth the price, but because you expect to hand it to a greater fool. The chain extends as long as new buyers keep arriving at higher prices — and it ends, abruptly, when one of them does not, leaving the last holder marked all the way back to value.

![Flow from fair value through chained buyers to the music stopping and the last holder taking the loss](/imgs/blogs/the-greater-fool-and-rational-bubbles-the-musical-chairs-game-1.png)

## Foundations: greater-fool theory, rational bubbles, and the timing game from zero

Before we touch a single chart, we need three ideas built carefully, each from nothing. They are the load-bearing pieces of every bubble that has ever happened, and most explanations skip straight past them to moralize. We will not. We will define them precisely, because the whole point of a game-theory lens is that a bubble is a *structure*, not a moral failing.

### What "fair value" even means, and why price can leave it

Start with the anchor. The **fair value** (also called *intrinsic value* or *fundamental value*) of an asset is what it is worth on its own merits — the cash it will produce for whoever owns it, discounted back to today. For a stock, that is the stream of future dividends and buybacks the business can fund. For a bond, it is the coupons and the return of principal. For a rental property, it is the rent minus costs. For a gold bar or a tulip bulb or a token that pays nothing, fair value is murkier — it is whatever the next owner will give you, which is exactly the problem we are about to study.

The key claim of standard finance is that *price should equal fair value*, because if price drifts above value, sellers will appear (the asset is expensive, so owning it is a bad deal) and push it back down; if price drifts below, buyers will appear. This is the gravity that is supposed to hold price near value. A **bubble** is what happens when that gravity fails — when price floats far above any defensible fair value and *keeps rising anyway*, not despite the overvaluation but because of it. The rise itself becomes the reason to buy.

Why would gravity fail? Because there is a second source of return that has nothing to do with the asset's cash flows: the price you can resell it for. And that price depends on other people, not on fundamentals.

### Greater-fool theory: the price is a bet on the next buyer

Here is the core of it, stated as plainly as it can be: **you do not need an asset to be worth the price to make money buying it — you only need someone willing to pay you more later.** That someone is the *greater fool*. You are a fool to pay above value; you make money anyway if a greater fool pays an even-more-above-value price to take it off your hands.

This is not a trick or a slur. It is a precise decomposition of your return. Your total return from buying at price $P$ and selling at price $P'$ is:

$$\text{return} = \underbrace{(\text{value while you held it})}_{\text{fundamentals}} + \underbrace{(P' - P)}_{\text{resale gain}}$$

For a normal investment, the first term dominates: you own a business, it pays you, the resale price is a sideshow. In a bubble, the *first term is tiny or zero* (the tulip pays no dividend; the meme stock's business is shrinking) and the *second term is everything*. When the resale gain is your whole thesis, you are no longer investing in the asset — you are betting on the existence and the price of the next buyer. You have, knowingly or not, joined the greater-fool game.

The phrase "the bigger fool theory" gets used as an insult, but the strategy is real and sometimes rational. A trader who buys an obviously overpriced asset, fully aware it is overpriced, planning to resell it higher within days, is not necessarily irrational. They are making a *conditional* bet: *conditional on more buyers arriving, I win.* The question is never "is this a fool's price?" (it plainly is) but "am I early enough in the chain that a greater fool will still appear above me?"

### The musical-chairs game: the timing layer on top

Greater-fool theory tells you *why* you might buy. The **musical-chairs game** tells you the shape of the danger. Recall the rules of the children's party game: there are more players than chairs, music plays while everyone circles, and when the music stops, everyone scrambles for a chair. Whoever is left standing is out.

Map it onto the bubble. The **players** are everyone holding the asset, hoping to sell higher. The **chairs** are the exits — buyers who will still pay a good price when you decide to sell. The **music** is the flow of new money and rising prices that keeps the dance going. And the **music stopping** is the moment new buyers stop arriving at higher prices, so the bid — the price someone will actually pay you right now — vanishes.

The structural fact that makes this lethal: **there are always far more players than chairs.** Everyone in a bubble is, by definition, a buyer who plans to become a seller. But a seller needs a buyer on the other side, and at the top there is no fresh wave of buyers — they have all already bought. So the people trying to exit vastly outnumber the people willing to take the other side. The chairs are scarce precisely when everyone needs one at once.

This is the timing game: not "what is it worth?" but "*when* do I leave, given that everyone else is solving the same problem?" And as we will see, the equilibrium answer to that question — the answer everyone independently reaches — is brutal.

### Why the music keeps playing: the self-fulfilling rise

There is one more piece of the foundation, and it is the piece that turns a strange one-off mispricing into a roaring, self-feeding bubble. It is the *feedback loop* between the price and the players. In a normal market, a rising price makes the asset less attractive (it is now more expensive, so the deal is worse), which brings in sellers and cools the rise. In a bubble, a rising price makes the asset *more* attractive, because the rise itself is the evidence that greater fools exist. The price going up is the proof of the thesis "I can resell higher," so each up-tick recruits more players, whose buying pushes the price up further, which recruits still more players. The cause and the effect chase each other.

This is sometimes called *reflexivity* — the players' beliefs change the very thing they are betting on. The price is not just a measurement of demand; it is an *input* to demand. A higher price literally creates more buyers in a bubble, the opposite of how a market is supposed to behave. That is why bubbles can run so far past any sane valuation: there is no fundamental anchor pulling the price back, only the question of whether the next wave of fools is bigger than the last. As long as each wave is bigger, the music plays and the price climbs. The moment a wave is *smaller* than the one before — the moment the new-buyer inflow merely slows — the loop runs in reverse, and a falling price becomes evidence that the greater fools are gone, which scares the next would-be buyer away, which makes the price fall further.

The asymmetry is the killer: on the way up, the loop is gentle and slow because new fools have to be *recruited* one wave at a time. On the way down, the loop is violent and fast because the players are *already there* and all reach the exit at once. The same feedback that took months to inflate the bubble deflates it in days. This is the engine behind every drawdown chart later in this post — the rise is a recruitment problem (slow), the fall is a coordination problem (instant).

### Players, strategies, payoffs, and information — the game stated formally

Let's state the whole thing in the language of game theory, so the rest of the post has a precise object to reason about. A musical-chairs bubble is a *dynamic game of incomplete information* with these pieces:

- **Players:** every current and potential holder of the asset. The pool grows as the price rises (reflexivity) and the relevant players at any moment are the ones currently holding, deciding whether to keep holding or sell.
- **Strategies:** at each period, every holder chooses **ride** (hold, hoping for a higher resale) or **sell now** (take the current bid). New entrants choose **buy** or **stay out**.
- **Payoffs:** if you sell into a standing bid, you bank the resale gain. If you are still holding when the bid vanishes, you take the crash loss back toward fair value. The payoffs are *interdependent* — your sale needs someone else's purchase, so one player's exit is another's entry.
- **Information:** incomplete and asymmetric. No one knows the true crash probability $q$ this period; no one knows how many fresh buyers remain; no one can see whether the holder next to them is about to sell. Each player only sees the price tape and has to *infer* how close the music is to stopping from a noisy signal everyone else is also reading.

That last point — that everyone is reading the *same* noisy signal and reacting to it simultaneously — is what makes the exit a coordination problem with no good coordinated solution. We will return to it; for now, hold the four-part structure (players, strategies, payoffs, information), because every claim in this post is a statement about one of those four pieces.

#### Worked example: the resale return, and the moment it inverts

Let's make greater-fool theory concrete with numbers. Suppose a token's defensible fair value is \$10 (it pays nothing, but say \$10 is what a sober buyer might risk on its long-shot future). You buy it at \$40 — four times fair value. You are not confused; you know \$40 is a fool's price. Your plan: sell at \$60 within a month.

If a greater fool shows up and pays \$60, your return is `(\$60 − \$40) / \$40 = +50%` in a month. The \$10 of fundamental value never entered the calculation — your entire \$20 gain came from the resale, paid for by the next buyer. You won the round of musical chairs because you found a chair (a buyer) at \$60.

Now run the same math from the greater fool's seat. They bought at \$60, four-and-a-half times above value, planning to flip to a *still*-greater fool at \$80. If the new buyers dry up and the price falls back toward \$10, their loss is `(\$10 − \$60) / \$60 ≈ −83%`. Same asset, same strategy, opposite outcome — the only difference is *where in the chain* they sat. The intuition: in a greater-fool game, your profit is not produced by the asset; it is transferred to you from the person who buys after you, and to the person before you from you — the last person in the chain pays for everyone's gains.

## Can it be rational? Rational-bubble models and the +EV of riding

The unsettling claim deserves a real model, not a hand-wave. Economists call the formal version a **rational bubble**: a situation where prices rise above fundamentals and *every participant is behaving rationally given their beliefs.* The classic results (Blanchard and Watson in 1982; the broader rational-expectations literature) show that a bubble can persist indefinitely in a model where everyone is a cold-eyed expected-value maximizer — *as long as the expected gain from holding compensates for the expected loss from a crash.*

The logic is a balance scale. Each period you hold a bubble asset, two things can happen:

- With some probability $1 - q$, the bubble **survives** and the price rises by some growth rate $g$. You make money.
- With probability $q$, the music **stops** this period and the price crashes back toward fundamentals. You lose a lot.

For holding to be rational — for it to be *positive expected value* — the probable gain has to outweigh the probable crash. And here is the eerie part the model reveals: a bubble can grow *faster and faster* and still be a fair bet, because as the crash becomes more likely, the survivors have to be paid more to keep holding. The price has to accelerate to compensate for the rising risk of the stop. That is why bubbles so often go parabolic right before they pop — the acceleration is not a bug, it is what *keeps the holders rational* as the danger climbs.

### Computing the expected value of riding one more period

Let's pin it down with the kit's `expected_value` helper. Take a position currently marked at \$100. Model one more period of holding as a two-outcome gamble:

- If the bubble survives (probability $1 - q$), it rises 40% to \$140.
- If it crashes this period (probability $q$), you recover only \$30 of your \$100 — a 70% loss, the kind of gap-down you get when the bid vanishes.

The expected value of holding is then $\text{EV} = (1-q)\times 140 + q \times 30$. Compare that to your alternative: *sell now* and lock in \$100. Riding is worth it only while $\text{EV} > 100$. Solving $(1-q)\times 140 + q\times 30 = 100$ gives the break-even crash probability:

$$q^* = \frac{140 - 100}{140 - 30} = \frac{40}{110} \approx 0.36.$$

The chart traces this exactly. Below a 36% crash probability, riding one more period is positive expected value — staying in the game is the rational move, and a cold EV-maximizer *should* hold. Above 36%, the expected value of holding drops below the \$100 you could lock in by selling, and riding becomes a losing bet. The "music about to stop" point is not a vibe; it is the crossing of two lines.

![Expected value of holding a bubble one more period falling below selling now as crash probability rises](/imgs/blogs/the-greater-fool-and-rational-bubbles-the-musical-chairs-game-2.png)

The honest, queasy implication: for crash probabilities under about a third, *a perfectly rational person rides the bubble.* They are not a fool — they are doing the EV arithmetic and the arithmetic says hold. This is why "the people in bubbles are idiots" is the wrong model. Plenty of them are doing exactly the calculation above, correctly, and the calculation says stay in — right up until $q$ creeps past $q^*$ and they are supposed to leave at the same instant as everyone else doing the same math.

#### Worked example: why the price has to accelerate to stay fair

Make the rational-bubble engine explicit. Suppose every holder demands an expected return of 10% per period just to keep holding (their opportunity cost). In period one, the crash probability is a mild $q = 0.05$. For the *expected* return to be 10%, the price has to rise enough on the surviving 95% of paths to make up for the 5% of paths where it crashes to near zero.

Roughly: $\text{E[return]} = (1 - q)\times g - q \times 1$, where $g$ is the up-move and a crash loses ~100%. Setting this to 10%: $(0.95)\,g - 0.05 = 0.10$, so $g = 0.15/0.95 \approx 15.8\%$. The price must rise about 16% on the surviving paths.

Now jump to period ten, with the crash probability up to $q = 0.30$ as the bubble has gotten obviously stretched. The same equation: $(0.70)\,g - 0.30 = 0.10$, so $g = 0.40/0.70 \approx 57\%$. To stay a fair 10% bet, the price now has to rise *57% per period* on the surviving paths. The bubble has to go vertical just to keep paying its holders for the mounting crash risk. The intuition: a parabolic late-stage blow-off is not a sign the market has gone insane — it is exactly the price path a *rational* bubble must follow to keep its holders compensated as the music gets closer to stopping.

### Where the rational-bubble model breaks — and why that helps you

The rational-bubble model is beautiful and it is also too generous, in ways worth naming, because each crack in the model is a place where the real game gets *more* dangerous than the math, not less. Three breaks matter.

First, the model assumes everyone agrees on the crash probability $q$. In reality, players wildly disagree about $q$, and the ones who buy at the top are precisely the ones who estimate $q$ too low — the most optimistic, least informed wave. The "marginal buyer" who sets the price at the peak is, almost by selection, the person with the worst estimate of how close the music is to stopping. So the late price reflects the beliefs of the *most* mistaken participants, not the average. This is a cousin of the *winner's curse* from auction theory: the one who ends up holding is the one who most overestimated the value, just as the auction winner is the one who most overbid. The greater fool is, structurally, the most over-optimistic forecaster in the room.

Second, the model treats the crash as a clean, known-size event. Real crashes are *path-dependent* and reflexive: the size of the fall depends on how many players try to exit at once, which depends on how scared everyone is, which depends on the fall — the loop again. So the "30 you recover" in our EV model is itself a function of the panic, and in a real stampede it can be far less than you modeled. The model's tidy two-outcome gamble understates the left tail.

Third, and most importantly, the model assumes you can act on your EV calculation *independently*. You cannot. The moment your honest estimate of $q$ crosses $q^*$ and tells you to sell is the same moment everyone else's identical estimate tells *them* to sell. Your "sell" decision is correlated with theirs through the shared signal of the price tape, so you all try to exit through the same door at the same instant. The EV math is computed as if you face a passive coin flip; the real game has you facing a stampede triggered by the very signal you are acting on. This is why the next section moves from the lone-holder EV to the *strategic* exit game — the lone-holder calculation, however correct, omits the most dangerous player at the table: the rest of the crowd.

The practical upshot of all three breaks: treat the EV math as an *upper bound* on how good riding is. If even the generous model says riding turns negative past $q^* \approx 0.36$, the real game — with over-optimistic marginal buyers, reflexive crashes, and a correlated exit — turns negative sooner. When the model says "marginal," the reality is usually "already too late."

## The timing game: selling first is dominant, so everyone sells at once

So far we have a holder weighing EV in isolation. But you are not alone — every other holder is solving the same exit problem, and your best move depends on theirs. That makes it a game, and the game has a vicious structure.

### The 2×2 of the top: ride versus sell now

Strip it to two holders, both sitting near the top, each choosing **ride** (hold for more) or **sell now**. The payoffs, in stylized units:

- **Both sell now:** an orderly exit. The selling is spread out, the price holds reasonably, each gets a fair exit. Call it **+10 each.**
- **You sell now, they ride:** you cash out into a still-standing bid (**+10**); they are now the marginal holder when the music stops (**−30**).
- **You ride, they sell now:** the mirror — you hold the bag (**−30**), they cash out (**+10**).
- **Both ride:** tempting — if it keeps rising you both gain — but now the exit is shared and narrow and the crash risk is sitting right there, so the *expected* payoff of staying is dragged down to a fragile **+4 each.**

Feed these to the kit's `nash_2x2` solver and the result is unambiguous: the unique Nash equilibrium is **(sell now, sell now)**. Selling now is a *dominant strategy* — it beats riding whatever the other person does. If they sell, you'd rather sell (+10 vs −30). If they ride, you'd still rather sell (+10 vs +4). There is no column in which riding wins. So both sell, immediately, at the first sign the music might stop.

![Payoff matrix of ride versus sell now with selling now as the dominant strategy](/imgs/blogs/the-greater-fool-and-rational-bubbles-the-musical-chairs-game-4.png)

This is the same skeleton as the [prisoner's dilemma](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once): the cooperative outcome (everyone rides calmly, or exits slowly and gently) would leave the holders better off than the panicked rush, but it is not stable, because each individual is better off defecting — selling first. The rational, self-interested choice of every player produces the outcome none of them wanted: everyone heading for the exit at the same moment, which is precisely what collapses the bid.

### Why "I'll get out in time" is the universal losing belief

Now we can say exactly why the most common bubble plan fails. "I'll get out in time" assumes you can sell *before* the rush. But the timing game says the rush *is* the equilibrium — everyone who reasons clearly reaches "sell first," so the rush happens the moment the music's stop becomes likely. You are not racing a slow-moving crowd; you are racing a crowd that is solving the identical optimization and will move at the same instant you do.

Three forces make the plan fail together:

1. **Everyone plans to.** "Get out before the drop" is not a clever edge — it is the *default* plan of every single holder. If it were easy, no one would ever be caught, and bubbles would deflate gently. They do not.
2. **The exits are narrow.** As we saw, players vastly outnumber chairs at the top. When the bid you were counting on is the one everyone else was also counting on, it cannot absorb all of you. The price gaps down through the level where you planned to sell before your order even fills.
3. **Selling is a prisoner's-dilemma rush.** Because selling first dominates, the calm orderly exit is unstable. The first hint of trouble triggers the dominant strategy in everyone at once, and a gentle decline becomes a cliff.

The exit, drawn as a process, looks like this: the calm exit everyone wants, then each holder's private reasoning that selling first is safest, then the collapse of that into a simultaneous rush, then everyone worse off than if they had managed to coordinate.

![Process flow from a wanted calm exit through dominant sell-first reasoning to a simultaneous rush](/imgs/blogs/the-greater-fool-and-rational-bubbles-the-musical-chairs-game-6.png)

#### Worked example: the cooperation loss of the rush

Quantify what the rush costs versus the orderly exit, using the matrix payoffs. If both holders could somehow commit to **sell now in a coordinated, orderly way**, each banks **+10** — the (sell, sell) Nash, which here is also the good outcome because it spreads the selling out. The genuinely *bad* realized outcome is when the music stops while one of you is still riding: that holder takes **−30**.

Suppose, more realistically, that at the top there are 100 holders and only enough exit liquidity for 20 of them to get the good +10 exit; the other 80 are caught riding when it stops and take −30 each. Aggregate realized payoff: `20 × (+10) + 80 × (−30) = +200 − 2,400 = −2,200`. Had all 100 somehow exited in the orderly +10 outcome, the aggregate would have been `100 × (+10) = +1,000`. The rush destroyed `1,000 − (−2,200) = 3,200` units of value relative to the coordinated exit — and crucially, *no individual could have prevented it*, because for each of them, selling first was still the best response. The intuition: the loss in a crash is not bad luck, it is the predictable price of a game whose only equilibrium is the stampede.

### The narrowing exit: chairs disappear as players pile in

The reason the rush is so destructive is that the exit gets *narrower* over the life of the bubble even as more people need it. Early on, when the asset is rising off a low base, there is a chair for roughly everyone who wants one — new buyers arrive faster than sellers want to leave, so anyone who wants to sell finds a willing buyer near the going price. Late in the bubble, the new-money inflow that was the music slows, then reverses: the people who would have been your buyers have already bought, and now they want to sell too. The chairs vanish exactly when the most players are crowded into the trade.

![The crowd of players growing while exit liquidity shrinks, leaving a trapped crowd](/imgs/blogs/the-greater-fool-and-rational-bubbles-the-musical-chairs-game-3.png)

The amber wedge between the two lines is the trapped crowd — the holders for whom there is no chair. That gap is not a forecast of a specific bubble; it is the structural shape of *every* musical-chairs game, because the defining feature of a bubble is that it recruits more players than the exits can ever clear. When the gap opens, the timing game's dominant strategy fires, and the gap converts instantly into a crash.

#### Worked example: how far past the top you actually sell

Put numbers on "I'll get out in time." Say the bubble peaks at \$300 (fair value is \$120). You planned to sell "near the top," call it \$290. But the top is only visible in hindsight — at \$300 the tape still looks strong, and by the time the rollover is obvious, the price is already falling fast in the rush.

Suppose the price falls 8% per "tick" of the panic once it starts, and it takes you two ticks to recognize the top is in, decide, and get an order filled into a thinning bid (a generous assumption — many are slower). From \$300, two ticks down is `300 × 0.92 × 0.92 ≈ \$254`. But your market order in a vanishing bid suffers slippage — say another 10% — so you actually fill around `254 × 0.90 ≈ \$229`. You planned to sell at \$290; you realized \$229. That \$61 gap, about 21% of the peak, is the distance between the plan and the reality — and it only grows the more crowded the exit. The intuition: the price where you *plan* to sell and the price where you *actually* sell are separated by the width of the door and the speed of the crowd, and both work against you.

![Stylized bubble price path showing where holders plan to sell versus where they actually exit](/imgs/blogs/the-greater-fool-and-rational-bubbles-the-musical-chairs-game-5.png)

The blue path is the bubble; the green dot is the exact top, where everyone plans to sell; the red dot, lower and to the right, is where you actually get out — after the bid has thinned, in the rush, with slippage. The gap between the two dots is the "I'll get out in time" delusion, drawn to scale.

## The information game: why bubbles persist when everyone privately knows

Here is a puzzle that the timing game alone does not answer. If everyone holding a bubble asset privately believes it is overvalued — and in the meme-stock era, many said so out loud — why doesn't the bubble pop the instant that belief is widespread? Why can a bubble keep inflating for *months* after "this is a bubble" is the common opinion at every dinner table?

The answer is one of the most elegant results in game theory, and it turns on the difference between *private knowledge* and *common knowledge*. You can know something, and everyone around you can know it, and yet it can still fail to be *common knowledge* — the recursive state where everyone knows that everyone knows that everyone knows. A bubble can persist precisely in the gap between "everyone privately thinks it's overvalued" and "everyone knows that everyone is about to sell." For the deeper machinery of this, the [common-knowledge post](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know) builds the recursion from zero; here we just need the consequence.

The consequence is this: knowing the asset is overvalued does *not* tell you to sell. What tells you to sell is the belief that *the other holders are about to sell* — because your loss comes not from the overvaluation but from the rush. So each holder sits there thinking "yes, it's a bubble, but I don't see anyone running yet, and I can resell to the next wave, so I'll hold one more period." Everyone reasons this way at once. The shared private knowledge that "it's a bubble" coexists, stably, with the shared private plan to "ride one more period." The bubble survives not because the players are deceived about value, but because none of them has yet received the signal that the *rush* has begun. The pop happens when something — a piece of news, a big holder selling, a failed new high — makes the imminent rush common knowledge. Then everyone's "one more period" flips to "now," simultaneously, and the door slams.

#### Worked example: the trigger that converts private doubt into a rush

Put rough numbers on it. Imagine 1,000 holders, every one of whom privately puts the crash probability this week at $q = 0.40$ — comfortably above our $q^* \approx 0.36$, so each privately "should" sell. Yet nobody sells, because each one also believes the *others* haven't figured it out yet, so a greater fool is still available this week. Their individual plan: "I'll ride one more week and sell into the suckers."

Now a single visible event occurs — a large, well-known holder publicly dumps, or the asset fails to make a new high on obviously good news. Suddenly each holder updates not their view of *value* (they already thought it was a bubble) but their view of *what the others will do*. Each now believes the other 999 are about to sell. The dominant strategy — sell before the others — fires in all 1,000 at the same instant. Trading volume that averaged, say, 10 million shares a day spikes to 100 million in an hour, all on the sell side, into a bid that can absorb a fraction of it. The price gaps down 30% before most orders fill. Nothing changed about the asset's value between yesterday and today; what changed was that private doubt became common knowledge of an imminent rush. The intuition: bubbles don't pop when people learn the truth about value — they pop when people learn the truth about each other.

## Common misconceptions

**"People in bubbles are just stupid."** This is the comforting story and it is mostly wrong. As the EV chart showed, for crash probabilities under roughly a third, *riding the bubble is positive expected value* — a cold, rational maximizer holds. Many bubble participants are not confused about value at all; they are explicitly playing the greater-fool game and doing the resale arithmetic correctly. Calling them stupid blinds you to the real danger, which is that *you* might be doing the same correct arithmetic right up until $q$ crosses $q^*$ and you are supposed to exit in the same instant as everyone else. The trap is built for smart people, not dumb ones.

**"As long as I get out before the top, I'm fine."** Nobody reliably gets out before the top, because the top is only knowable afterward. The top is, by definition, the last price at which a greater fool appeared — you cannot see it coming, you can only see it gone. And even if you somehow timed it, the *timing game* says the exit you were counting on is the same one everyone else was counting on, so it gaps away from you in the rush. "Before the top" is not a strategy; it is a wish.

**"There's no greater-fool risk because the asset has real value."** Real value caps your downside; it does not protect your bubble-era entry price. A stock can have a perfectly good business worth \$50 and still be a bubble at \$300 — the \$250 of air above fair value is pure greater-fool premium, and it can vanish even if the business is fine. The 2000 dot-com crash wiped out plenty of companies that *did* go on to be enormous (the survivors), because their share prices had run far past even their eventual, real, large fundamental value. Good asset, fool's price — the two coexist.

**"The bubble can't pop while it's still going up."** It pops *because* it was going up. The rise is what recruited the players and narrowed the exit; the higher and faster it climbs, the further it has to fall and the fewer fresh buyers remain to catch it. As the rational-bubble math showed, the price has to accelerate to keep its holders compensated — and a parabola is the most fragile shape there is, because it requires an ever-larger inflow of new fools just to stand still. "Still going up" is not safety; in the late stages it is the opposite.

**"I'll know when to sell because I'll see everyone else getting nervous."** By the time the nervousness is visible to you, it is visible to all the other holders solving the identical timing game — and selling-first is their dominant strategy too. Shared nervousness does not give you a head start; it is the starting gun for the rush, fired for everyone at once. The signal you are waiting for is the same signal that triggers the stampede you are trying to beat.

**"If it's a rational bubble, then riding it is the smart, sophisticated play."** The rational-bubble model says riding *can* be positive expected value — it does not say it is a good idea for *you*. Three things separate the model from your reality. The model assumes you can estimate the crash probability $q$ accurately; you cannot, and the marginal buyer at the top is, by selection, the worst estimator in the room. The model assumes a known crash size; real crashes are reflexive and the loss is usually deeper than modeled. And the model is computed for a single risk-neutral player facing a coin flip, whereas a real bubble can *ruin* you if the loss is large relative to your capital — and ruin is not something expected value prices in. "Rational" in the model means "fair bet for an infinitely-deep-pocketed gambler," not "prudent for a person with one portfolio and one life." Sophistication is knowing the difference.

**"Short the bubble — if it's overvalued, betting against it is free money."** This is the mirror-image trap, and it is often *more* dangerous than holding. A bubble can stay irrational longer than you can stay solvent: the reflexive loop means an overvalued asset can keep climbing, and a short position loses money the whole way up, with theoretically unlimited loss as the price rises. Many sophisticated investors who correctly identified bubbles were carried out feet-first for being early on the short side — being right about value and wrong about timing is, for a short, indistinguishable from being wrong. The timing game cuts both ways: just as you can't reliably sell at the top, you can't reliably pick the moment the music stops to short into. Knowing it's a bubble is not, by itself, a tradeable edge in either direction.

## How it shows up in real markets

Five episodes, each a clean run of the same game. The peak-to-trough drawdowns below are the cited record of what "the music stopped" actually cost the last holders.

![Bar chart of peak-to-trough drawdowns for housing, dot-com, crypto, GME, and tulip mania](/imgs/blogs/the-greater-fool-and-rational-bubbles-the-musical-chairs-game-7.png)

**Tulip mania, the Dutch Republic, 1636–1637.** The archetype. Contracts for rare tulip bulbs changed hands at extraordinary prices through the winter of 1636–37 — by some accounts a single bulb traded for the price of a fine house. Crucially, almost no one buying at the peak intended to plant a flower; they intended to resell the contract higher. It was a pure greater-fool game played in a futures market for a product that paid no income at all. In February 1637 the new buyers simply did not show up at an auction, the bid collapsed, and bulbs that had traded for a fortune sold for a small fraction of their former prices — estimates put the collapse at roughly 90% or more, effectively the market for them ceased to exist. The mechanism is identical to everything since: the chairs were always far fewer than the players, and when the music stopped, the floor was the actual flower value, which was almost nothing. *(Source: Wikipedia, "Tulip mania"; NY Fed Liberty Street Economics, "Crisis Chronicles: Tulip Mania, 1633–37".)*

**The dot-com bubble, Nasdaq, 2000–2002.** The Nasdaq Composite peaked at 5,048.62 on March 10, 2000, having roughly tripled in eighteen months on a wave of internet stocks, many with no profits and some with no revenue. The thesis driving most buyers late in the run was explicitly resale: "the internet changes everything, get in before it goes higher." When the new money slowed in spring 2000, the timing game fired, and the index fell about **77%** to a low near 1,114 in October 2002. The instructive part: several of the era's biggest losers by share price were *real* companies whose businesses went on to be massive — but their stock prices had run so far past even their eventual fundamental value that the greater-fool premium evaporated regardless. Good companies, fool's prices. *(Source: Wikipedia, "Dot-com bubble".)*

**The 2021 meme-stock and crypto run.** A textbook modern musical-chairs game, accelerated by zero-commission apps and social coordination. Participants on forums said openly they knew the prices were detached from fundamentals — the entire stated plan was to buy and sell higher to the next wave, and many were explicit that the point was the *game*, not the value. Crypto rode the same dynamic on a larger canvas: Bitcoin peaked near \$69,000 in November 2021 and fell about **78%** to roughly \$15,476 by November 2022 as the inflow of new buyers reversed. The story repeated across hundreds of tokens, most of which fell far further and never recovered. The lesson is the timing game's lesson: a strategy that is *openly* "sell to a greater fool" is one where, by construction, the last wave of buyers has no one left to sell to. *(Source: Wikipedia / public price records; BTC peak ~\$69,000 to ~\$15,476.)*

**GameStop, January–February 2021.** The forward-referenced case study (a fuller treatment comes in the final track) is the cleanest single instance of the timing game in modern markets. GME ran from around \$20 to an intraday high near \$483 in late January 2021, then collapsed to about **\$53.50** within the first days of February — roughly an **89%** drawdown from the peak. The squeeze mechanics (forced buying by short sellers) supplied extra fuel, but the exit was pure musical chairs: the holders who said "I'll sell before it drops" were, in aggregate, the ones who could not, because they were all trying to sell into the same vanishing bid at the same moment. The people who made money were overwhelmingly the *early* sellers, who handed the bag to later, higher buyers. *(Source: FXStreet, GME price coverage, Feb 2021.)*

**The U.S. housing bubble, 2006–2012.** Housing is the slow-motion version, and it shows the game scales beyond fast-moving speculative assets. Through the mid-2000s, a widespread belief that "house prices only go up" turned homes into a resale bet: buy with heavy leverage, refinance or flip higher. The S&P CoreLogic Case-Shiller National Home Price Index peaked at 184.61 in July 2006 and fell about **27%** to a trough near 133.99 in February 2012. The drawdown looks mild next to tulips or GME, but housing carries enormous leverage — a 27% price fall on a house bought with a 5% down payment wipes out the equity many times over — and the rush to exit (foreclosures, forced sales) was the same prisoner's-dilemma stampede, just played out over years instead of days. The greater fool here was often the next leveraged buyer, and when they stopped appearing, the chain broke. The reflexive loop from the foundations is visible in the wreckage: rising prices let people refinance and buy more, which pushed prices higher and recruited more buyers; when prices stalled, the refinancing stopped, defaults started, forced sales pushed prices down, which triggered more defaults — the loop running in reverse, exactly as the model predicts. *(Source: S&P CoreLogic Case-Shiller U.S. National Home Price Index.)*

What unites all five is the same three-part signature, and once you can name it you can spot the next one without being told it is a bubble. One: the dominant return is resale, not fundamentals — the holders are betting on the next buyer, not on the asset's cash flows. Two: the rise is self-fulfilling — higher prices recruit more buyers rather than fewer, the reflexive loop. Three: the exit is a coordinated rush — when the new-buyer inflow slows, the dominant strategy fires in everyone at once and the bid collapses far faster than it built. Tulips, internet stocks, meme stocks, crypto tokens, and leveraged houses look nothing alike on the surface and are the *identical game* underneath. The surface is where you get fooled; the game is where you get warned.

A note across all five: the [smart-money-concepts honesty post](/blog/trading/technical-analysis/smart-money-concepts-honestly) is a useful companion here, because the seductive bubble narrative is almost always that "smart money" is quietly accumulating and you are early — when in the late stages the smart money is the early seller handing *you* the bag. Knowing which side of that you are on is the whole game.

## The playbook / How to play it

This series is about playing the game one level deeper than the person on the other side. For bubbles, that means refusing both the moralizing ("it's all a fraud, stay away") and the greed ("I'll get out in time"), and instead reasoning explicitly about the structure. Here is the practitioner's version. *(This is educational, not individualized advice — it describes mechanisms and tradeoffs, not what you personally should buy or sell.)*

#### Worked example: sizing a resale bet you might lose entirely

The single most important number in a bubble position is not the entry — it is the *size*, because the size is what decides whether being caught is a bruise or a catastrophe. Suppose you have a \$100,000 portfolio and you want a small allocation to an asset whose entire thesis is resale (no fundamentals, pure greater-fool). The honest assumption, from the drawdowns we are about to survey, is that if the music stops you keep maybe 15% of what you put in — a roughly 85% loss, in line with GME's −89% or tulip's −90%-plus.

If you size it at 2% of the portfolio (\$2,000), the worst realistic case costs you `\$2,000 × 0.85 = \$1,700`, or 1.7% of the whole portfolio — survivable, a rounding error against a good year. If instead you let it become 25% of the portfolio (\$25,000) because it ran and you got greedy, the same 85% loss costs `\$25,000 × 0.85 = \$21,250` — over a fifth of everything, and the kind of hole that takes years to climb out of. Same asset, same crash, same strategy — the only difference is the size, and the size is the only part of the bet you fully control. The intuition: you cannot control whether you are the greater fool, but you can guarantee that being the greater fool is non-fatal, and that guarantee is worth more than any view on the price.

**Who is on the other side.** When you buy a bubble asset, the seller is someone earlier in the chain taking their resale gain — banking real cash, handing you the position. When you go to *sell*, the buyer you need is a greater fool: someone who will pay you more than you paid. Your entire P&L is the difference between what the previous fool charged you and what the next fool pays. If you cannot name, even roughly, who the next buyer is and why they would pay up, you are likely the last fool in the chain.

**The game you are in.** A musical-chairs / coordination-timing game wrapped around a prisoner's dilemma. The cooperative outcome (everyone exits gently) is not available because selling first dominates. The only stable equilibrium is the rush. You cannot fix the game; you can only choose your seat and your exit rule before the music stops, when you can still think.

**Your edge — if you have one.** It is *not* a better fundamental valuation (everyone knows it's overvalued; that is not information). The only real edges are: (1) being genuinely *early* — in the chain while greater fools are still arriving, which the EV math supports while $q < q^*$; (2) sizing so small that being caught is survivable, not ruinous; and (3) having a *mechanical, pre-committed exit* that fires on a rule, not on your read of the crowd — because your read of the crowd is the same read everyone else has, and it triggers the stampede. A rule you set in advance ("out if it closes below the 50-day," "trim half at +100%") beats the dominant-strategy panic precisely because it removes the in-the-moment timing game from your own head.

**The invalidation.** The bubble thesis ("a greater fool will pay more") is invalidated the moment the *inflow of new buyers* slows — not when the price first dips, but when the marginal new participant stops arriving. Practical proxies: new-account growth flattening, the narrative going mainstream (when your relatives are buying, the pool of greater fools is nearly exhausted), volume rising on down-moves, the asset failing to make new highs on good news. When the music's stop becomes likely — when your honest estimate of $q$ crosses toward $q^* \approx 0.36$ — the EV of riding has already gone negative, and the dominant strategy is to already be out.

**The sizing and the exit.** Treat any position whose thesis is resale, not fundamentals, as a *speculation* and size it as money you can lose entirely — because in a true musical-chairs game, the last holders do lose nearly everything (recall the −77% to −95% drawdowns above). Decide your exit before you enter, make it mechanical, and accept that you will leave money on the table by selling early. Selling early is the *only* reliable way to win musical chairs: the people who consistently profit are not the ones who sell at the top (no one does) but the ones who sell *before* they need to, while a chair is still free. The forward-referenced [crowded trades and the exit game post](/blog/trading/game-theory/crowded-trades-and-the-exit-game) extends this to the broader problem of being in a position that *everyone else also wants to exit* — the same narrowing-door dynamic, applied beyond outright bubbles.

**Where this touches your life.** You do not have to trade meme stocks to be in a musical-chairs game. The same structure shows up when a hot housing market makes everyone feel they must buy now "before it's too late"; when a coworker shows you the token that tripled and asks why you're not in; when a fund markets a strategy whose returns came entirely from an asset class that kept going up. In every case the right question is not "has it gone up?" (it has — that's why you're hearing about it) but "what produces the return from here: the asset, or the next buyer?" If the answer is the next buyer, you are being invited to a round of musical chairs, and the polite thing about the invitation is that it never mentions how few chairs there are. The defense is not cynicism — bubbles are real and people do make money in them — but *seat awareness*: knowing whether you are early or late, sizing so that being late is survivable, and pre-committing to an exit you will actually take.

The deepest point, and the one worth carrying out of here: a bubble is not a market that has gone irrational. It is a market that has become a *pure game* — where price is unmoored from value and reattached to the behavior of other players solving the same timing problem you are. Once you see that, you stop asking "what is it worth?" and start asking the only question that pays: *who is on the other side, and will there still be a buyer above me when I need one?* Most of the time, in a bubble, the honest answer is no — and the discipline to act on that answer, early, is the entire edge.

## Further reading & cross-links

- [The Prisoner's Dilemma in Markets: Why Everyone Sells at Once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) — the dominant-strategy stampede that powers the exit rush, built from the two-suspects game up.
- [The Trade Is a Game: Why Markets Are Strategic, Not Random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the series spine: every trade is an interaction with a counterparty, not a bet against nature.
- [Smart Money Concepts, Honestly](/blog/trading/technical-analysis/smart-money-concepts-honestly) — why the "I'm early, smart money is accumulating" narrative is usually the bag-holder's story in disguise.
- [Crowded Trades and the Exit Game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) *(forthcoming)* — the narrowing-door dynamic generalized beyond outright bubbles to any position everyone wants to exit at once.
