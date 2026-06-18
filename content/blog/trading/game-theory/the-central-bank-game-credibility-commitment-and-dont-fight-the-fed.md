---
title: "The Central Bank Game: Credibility, Commitment, and Don't Fight the Fed"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Monetary policy is a strategic game between the central bank and the public, where credibility is the only asset that matters and time-inconsistency is the trap that explains why discretion delivers higher inflation for no gain."
tags: ["game-theory", "trading", "central-bank", "federal-reserve", "credibility", "time-inconsistency", "forward-guidance", "inflation-expectations", "dont-fight-the-fed", "kydland-prescott", "monetary-policy"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Monetary policy is not a central bank acting on a passive economy; it is a game between the bank and a rational public that anticipates the bank's incentives, and the whole contest turns on one asset: credibility.
>
> - The **Kydland-Prescott time-inconsistency** result is the spine: a bank with discretion is always tempted to engineer a little surprise inflation to boost jobs, the public anticipates this, so it raises inflation expectations, and the economy lands at higher inflation with *no* extra employment. Discretion is a strictly worse equilibrium than commitment.
> - **Credibility is the central bank's entire balance sheet of trust.** A credible commitment to low inflation — an independent bank, an inflation target, a rule — pulls the game to the good equilibrium. Losing credibility is fast; rebuilding it is a slow, costly purchase paid in recessions.
> - **"Don't fight the Fed" is an equilibrium statement, not a slogan.** A committed central bank with an effectively unlimited balance sheet is the house; betting against its stated path is betting against a counterparty that cannot be made to fold by your capital.
> - The number to remember: the discretionary inflation bias buys you **zero** durable extra employment — the entire output gain is competed away by expectations, leaving only the higher inflation.

In October 1979, a new Federal Reserve chairman named Paul Volcker walked into a contest he was losing before he started. US inflation was running near 12% and climbing, and after fifteen years of stop-go policy nobody on the other side of the trade believed the Fed would actually stop it. Bond traders had priced in that the Fed would always blink — that the moment unemployment ticked up, the central bank would ease, let inflation run, and protect jobs. That belief was *rational*. It was also exactly what made inflation impossible to kill: as long as everyone expected the Fed to fold, prices and wages kept getting set as if high inflation were permanent, which made it permanent.

Volcker's move was to change the game. He let the federal funds rate rip to 20% in 1981, drove the economy into the deepest recession since the 1930s, pushed unemployment to 10.8%, and absorbed years of being burned in effigy by farmers and homebuilders — all to do one thing that no model can do for you: prove, with real pain, that this Fed would not fold. By 1983 inflation was under 4%. The recession was the *price he paid to buy back credibility*. That is the central-bank game in one episode, and the figure below is its skeleton.

![Time-inconsistency trap diagram showing discretion leading to anticipated surprise inflation and a worse equilibrium than commitment](/imgs/blogs/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed-1.png)

The diagram is the mental model for the whole post. Read it left to right: a central bank that keeps its options open — discretion — is tempted, every single period, to surprise the economy with a little extra inflation to nudge employment up. But the public is not a rock the bank pushes against; it is a strategic player that *knows* the bank is tempted. So the public raises its inflation expectations to match. The surprise the bank was counting on evaporates, because you cannot surprise someone who already expects the surprise. What is left is the worst of both worlds: higher inflation baked into wages and prices, and no extra jobs to show for it. Commitment — credibly tying your hands to low inflation — is the path that avoids the trap. Everything that follows is an unpacking of why this happens, how a central bank escapes it, and what it means for anyone with a position when the Fed speaks.

## Foundations: the central bank, the public, and the game between them

Before any game theory, we need the players, their moves, and what each one wants. A finance reader with no macro background can follow all of it from here.

**A central bank** is the institution that controls the supply of a currency and the short-term interest rate — in the US, the Federal Reserve; in the eurozone, the European Central Bank (ECB); in Vietnam, the State Bank of Vietnam (SBV). Its core lever is the *policy rate*: the interest rate at which banks lend to each other overnight, which the central bank sets by controlling the supply of reserves. When the Fed "raises rates," it makes money more expensive to borrow, which cools spending and (eventually) inflation. When it "cuts rates," it does the opposite. A *basis point* — you will see it constantly — is one hundredth of a percent, so a 0.25% rate move is "25 basis points" or "25 bps."

**Inflation** is the rate at which the general price level rises — if a basket of goods cost \$100 last year and \$103 this year, inflation is 3%. Central banks in most rich countries target around 2% inflation per year. **Unemployment** is the share of people who want a job and cannot find one. The reason these two appear in the same game is an empirical regularity called the **Phillips curve**: in the *short run*, surprise inflation tends to push unemployment down, because firms see higher prices, think demand is booming, and hire — before workers realize their wages buy less. The key word is *surprise*. Anticipated inflation does nothing for employment, because everyone adjusts wages and prices in advance.

**The public** — households, firms, wage-setters, and the bond market — is the other player. The public does not care about inflation for its own sake; it cares about *forecasting it correctly*, because everyone signs contracts, sets wages, and prices loans based on what inflation they expect. If you expect 2% inflation and lend money at 5%, you think you are earning 3% in real terms. If inflation surprises you at 6%, you actually *lost* 1% in real terms — the borrower won, you lost. So the public's payoff is highest when its inflation expectation matches what actually happens. The public is rational: it knows the central bank's incentives and forecasts accordingly.

Now the game. The central bank moves: it can **commit** to low inflation (and stick to it) or use **discretion** (keep its options open, reserve the right to surprise). The public moves: it can **expect low inflation** or **expect high inflation**. Crucially, the public forms its expectation *knowing the bank's incentives* — and in the real game it gets to revise that expectation continuously, so over time the bank cannot fool it. The payoffs:

- If the bank **commits** and the public **expects low** inflation, you get the good equilibrium: low inflation, no nasty surprises, stable real interest rates. Everyone wins as much as the game allows.
- If the bank uses **discretion** and the public still **expects low** inflation, the bank gets its one good shot: it surprises with a little inflation, employment ticks up, and — for one period only — the bank looks like a hero. This is the *temptation*.
- If the public **expects high** inflation, then whatever the bank does, high inflation is already baked into wages and prices. If the bank then commits to low inflation anyway, it gets a recession (the public over-expected, real rates spike). If the bank delivers the expected high inflation, you get the bad equilibrium: high inflation, no employment gain, but at least no recession shock.

There is one more concept that makes the trap airtight: the **natural rate of unemployment**, sometimes called the NAIRU (the non-accelerating-inflation rate of unemployment). This is the unemployment rate the economy gravitates to when inflation is fully anticipated — set by real things like how easily workers and jobs find each other, not by monetary policy. The central bank cannot push unemployment durably below the natural rate; it can only do so *temporarily*, and only by surprise. The instant the surprise is anticipated, unemployment snaps back to the natural rate. This is why the discretionary equilibrium delivers *zero* durable employment gain: the only lever the bank has on unemployment is surprise, and surprise cannot survive a rational public. Everything the bank does that the public anticipates affects only the price level, never real employment. That single fact — anticipated money is neutral for real outcomes — is the foundation the whole game stands on.

The solution concept is **Nash equilibrium**: a pair of strategies where neither player can do better by unilaterally changing their move, given what the other is doing. We will compute it. There is also a sharper, dynamic version of the solution concept that monetary economists actually use — **subgame-perfect equilibrium**, which additionally requires that the bank's plan be optimal *at every future moment*, not just at the start. Time inconsistency is precisely the statement that the bank's promised plan (commit to low inflation) is *not* subgame-perfect under discretion: when the future moment arrives, the bank's best move is to renege. A plan that is optimal to announce but not optimal to carry out is the technical definition of a time-inconsistent policy, and it is why a rational public discounts the announcement.

The headline result — proven by Finn Kydland and Edward Prescott in their 1977 paper, work that won them the 2004 Nobel Prize — is that the only equilibrium of the *discretionary* game is the bad one. That is **time inconsistency**, and it is the engine of everything below.

#### Worked example: why the surprise is worth nothing in equilibrium

Suppose the bank believes the short-run Phillips curve says: each 1 percentage point of *surprise* inflation buys 0.5 points of lower unemployment. Inflation expectations start at 2%. The bank reasons: "If I let inflation run to 4%, that is a 2-point surprise, so unemployment falls by 1 point. Worth it." So it does — once. Unemployment dips.

But the public is not fooled twice. Next period it expects 4%, because it has watched the bank choose 4% when it had the chance. Now for the bank to get *another* surprise, it must do 6%. The surprise is again 2 points (6% actual minus 4% expected), unemployment dips again — but inflation is now 6% and climbing. The public keeps revising upward. In the limit, expected inflation rises to wherever the bank actually delivers, the surprise term goes to **zero**, and unemployment sits right back at its natural rate. Final score: inflation ratcheted from 2% to, say, 6% or 8%, and unemployment is *exactly where it started*.

The intuition: you cannot run a permanent surprise against a player who learns. The entire output gain is a one-time loan from the public's naivety, repaid in full with interest the moment the public catches on — and the interest is permanently higher inflation.

There is a subtle point worth dwelling on, because it is where beginners go wrong. The Phillips-curve trade-off is *real* in the short run — surprise inflation genuinely does lower unemployment for a while, which is exactly why the temptation exists. The bank is not being stupid when it inflates; it is responding to a true incentive. The tragedy is that the incentive is a trap: the move that is individually rational each period leads to an outcome that is collectively terrible across periods. This is the same structure as a prisoner's dilemma, where each player's dominant move (defect) produces an equilibrium worse for both than mutual cooperation. The central bank is, in effect, playing a prisoner's dilemma against its own future self and against the public's expectations at the same time — and we explore the market version of that structure in [the prisoner's dilemma in markets](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once).

One more foundational distinction matters: the difference between **anchored** and **un-anchored** expectations. Expectations are *anchored* when, no matter what the latest inflation print is, the public still expects inflation to return to the bank's target over the medium term — because it believes the bank will act to bring it back. Anchored expectations are the single most valuable thing a central bank possesses, because they do the bank's work for it: if everyone *expects* 2%, wage and price setters behave as if inflation is 2%, and inflation tends to *be* 2% with very little policy effort. Expectations *un-anchor* when a high inflation print makes people revise up not just this year's forecast but their *long-run* forecast — at which point a wage-price spiral can feed on itself and the bank must act violently to re-anchor. The entire 2021-2023 episode was, at its core, a fight to keep long-run expectations anchored even as short-run inflation soared. Markets price this directly: the *5-year, 5-year forward breakeven inflation rate* — the market's expectation of average inflation over the five years starting five years from now — is the single number central bankers watch as the live read on whether their credibility is holding.

## The Kydland-Prescott result: discretion is a strictly worse equilibrium

Let us make the game concrete with numbers and solve it, the way we would solve any 2×2 game. We use `nash_2x2` from the series toolkit. The central bank is the **row** player: row 0 = commit, row 1 = discretion. The public is the **column** player: column 0 = expect low, column 1 = expect high.

We assign payoffs (higher is better; think of them as the negative of a loss the player is trying to minimize):

- Central bank's payoffs (row): commit + expect-low = **3** (the good equilibrium — low inflation, stable); discretion + expect-low = **4** (the *temptation* — a surprise that boosts jobs this period); commit + expect-high = **1** (the bank tightens into an over-expecting public, a self-inflicted recession); discretion + expect-high = **2** (the bad equilibrium — high inflation, but no recession shock).
- Public's payoffs (column): the public's payoff is highest when its forecast is *correct*. Commit + expect-low = **3** (forecast right, low inflation realized); discretion + expect-low = **0** (forecast wrong, surprised by inflation, lost real value on every contract); commit + expect-high = **1** (forecast wrong the other way); discretion + expect-high = **3** (forecast right, high inflation expected and delivered).

The matrix below is the game. The shaded cell is the Nash equilibrium the solver finds.

![Commit versus discretion payoff matrix for the central bank and the public with the bad Nash equilibrium highlighted](/imgs/blogs/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed-2.png)

Now solve it. Look at the central bank's choice for each public expectation:

- If the public **expects low**, the bank compares commit (3) vs discretion (4). Discretion wins. The bank defects.
- If the public **expects high**, the bank compares commit (1) vs discretion (2). Discretion wins again. The bank defects.

Discretion is a **dominant strategy** for the central bank — it is better *no matter what the public expects*. This is the time-inconsistency trap in one line: whatever the bank *promised* yesterday, today it is always tempted to inflate a little. The public, being rational, runs exactly this reasoning. It knows the bank will choose discretion, so it best-responds by expecting high inflation (its payoff in the discretion column is 0 for expect-low vs 3 for expect-high). The unique Nash equilibrium is **(discretion, expect-high)**, with bank payoff **2**.

Compare that to **(commit, expect-low)**, the cell where the bank's payoff is **3**. The commitment outcome is *strictly better for the central bank itself* — not just for the public — yet the game does not deliver it under discretion. That is the paradox Kydland and Prescott proved: **the ability to act with discretion makes the central bank worse off.** Tying your own hands is not a constraint to be regretted; it is a gift, because it changes the public's best response and lets you reach the cell discretion can never reach.

#### Worked example: solving the central-bank game by hand

Let us verify the equilibrium the way the toolkit does. The bank's payoff matrix is `A = [[3, 1], [4, 2]]` and the public's is `B = [[3, 0], [1, 3]]`. A cell `(r, c)` is a pure Nash equilibrium if neither player wants to deviate.

```
import data_gametheory as gt
A = [[3, 1], [4, 2]]   ;; bank: row 0 commit, row 1 discretion
B = [[3, 0], [1, 3]]   ;; public: col 0 expect-low, col 1 expect-high
print(gt.nash_2x2(A, B))
;; -> {'pure': [(1, 1)], 'mixed': None}
```

Check cell **(1,1) = (discretion, expect-high)**. Bank: can it gain by switching to commit (row 0) in the expect-high column? Commit pays 1, discretion pays 2 — no, it would lose 1. Public: can it gain by switching to expect-low (col 0) in the discretion row? Expect-high pays 3, expect-low pays 0 — no, it would lose 3. Neither deviates, so (1,1) is the equilibrium. Bank gets 2.

Check the cell we *wish* was the equilibrium, **(0,0) = (commit, expect-low)**, bank payoff 3. Bank: switch to discretion (row 1) in the expect-low column? Commit pays 3, discretion pays 4 — yes, it gains 1 by defecting. So (0,0) is *not* an equilibrium: it is not self-enforcing, because the bank's own incentive breaks it. The good outcome is unreachable precisely because everyone knows the bank would cheat if it could.

The intuition: the difference between bank payoff 3 (commitment) and 2 (discretion) — that lost unit — is the **inflation bias**, the cost the economy pays for the bank's *inability to credibly promise*. It is not paid because the bank is stupid; it is paid because the bank is rational and everyone knows it.

## Rules vs discretion, and the inflation bias you can see in the data

The inflation bias is not just a theorem; it is a number you can put on a chart. The discretionary equilibrium sits at a higher inflation rate than the committed one, and — this is the part that makes it a tragedy rather than a trade-off — at the *same* unemployment rate. You paid in inflation and got nothing in jobs.

![Bar chart comparing inflation under a committed rule versus discretion with unemployment unchanged](/imgs/blogs/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed-3.png)

The chart shows the stylized outcome the theory predicts and the 1970s confirmed: under a credible rule, inflation settles near the target (say 2-3%); under discretion, the bias pushes it to 6-8% or higher, while unemployment in both regimes sits at the same natural rate. The vertical distance between the two inflation bars is the inflation bias — pure deadweight loss. The US lived the right-hand bar through the 1970s and the left-hand bar after Volcker re-anchored expectations.

The policy answer Kydland and Prescott pointed to is **rules over discretion**: if the bank can commit to a *rule* it is institutionally bound to follow, the public believes the low-inflation promise, and the game moves to the good equilibrium. In practice "a rule" takes several forms, each a different commitment device:

- **Central bank independence.** If the politicians who would love a pre-election boom cannot order the bank to inflate, the bank's promise is more credible. New Zealand pioneered legal independence with inflation targets in 1989; most modern central banks followed.
- **An explicit inflation target.** A public, numerical target (the Fed's 2%, formally adopted in January 2012) gives the public a clear yardstick and makes a deviation visible and embarrassing. Visibility is the point: a promise you can be caught breaking is more credible than a vague intention.
- **A policy rule** like the **Taylor rule**, which prescribes the policy rate as a formula of inflation and the output gap. Even as a benchmark the bank does not mechanically follow, it disciplines discretion by making deviations legible.

None of these is magic; each is an attempt to take the bank's dominant-strategy temptation off the table so the public will believe the good equilibrium is the one being played.

#### Worked example: the real-rate cost of being surprised

Why does the public care so much about forecasting inflation right? Walk the cash flows. You lend \$10,000 for one year at a 5% nominal rate, expecting 2% inflation. You think your **real return** — purchasing power gained — is about 5% − 2% = 3%, or \$300 of real value.

Now the bank uses discretion and inflation comes in at 6% instead. You still get your 5% nominal, \$500 back in interest, but prices rose 6%, so your \$10,500 buys what \$10,500 / 1.06 ≈ \$9,906 bought a year ago. In real terms you *lost* about \$94 on a loan you thought would earn you \$300. The borrower captured exactly that swing. Surprise inflation is a wealth transfer from lenders to borrowers — and the biggest borrower in the economy is the government, which is precisely why the temptation to inflate is structural.

The intuition: the public's "expect high" move is not paranoia; it is self-defense. Once burned, every lender demands a higher nominal rate to protect against future surprises, which is *itself* the higher-inflation equilibrium showing up in market prices. The bond market enforces the bad equilibrium by demanding an inflation-risk premium.

## Forward guidance: moving the market before you move at all

Here is where the central-bank game gets beautiful, and where it connects directly to how you trade. If the bank's true power is over *expectations*, then the bank's sharpest tool is not the rate change itself — it is **telling the market what it will do before it does it.** This is **forward guidance**: communication about the future path of policy, designed to move long-term interest rates, asset prices, and the real economy *today*, through expectations, before any actual policy action.

Think about why this works. A 30-year mortgage rate, a corporate bond yield, a stock's valuation — none of these depends much on today's overnight rate. They depend on the *expected average* of the overnight rate over years. So if the Fed credibly says "we will keep rates near zero through at least 2023" (as it effectively did during the pandemic), long rates fall *immediately*, mortgages get cheaper *immediately*, and stocks re-rate *immediately* — without the Fed having moved the policy rate at all. The bank moved the market by moving the expectation.

![Forward guidance pipeline showing a Fed statement moving long rates and asset prices before any policy rate change](/imgs/blogs/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed-4.png)

The pipeline above is the transmission: a statement creates an expectation, the expectation reprices the curve, and the curve reprices everything that discounts off it — all before the gavel falls on any actual rate decision. This is the mechanism behind the market adage **"buy the rumor, sell the news"**: by the time the Fed actually cuts, the cut has already been priced through forward guidance, so the *announcement* moves nothing — the *guidance* did all the work weeks earlier. We unpack that adage as its own game in [buy the rumor, sell the news](/blog/trading/game-theory/buy-the-rumor-sell-the-news-public-signals-and-the-fed); here the point is that the Fed is the original rumor-mill, and it does it on purpose.

The deepest part of forward guidance is what it does to **common knowledge**. In game theory, something is *common knowledge* when everyone knows it, everyone knows that everyone knows it, and so on without end — a far stronger condition than everyone merely happening to know it. Coordination games (and a market is one giant coordination game) are won and lost on common knowledge, not private belief. The Fed's announcements are a **common-knowledge machine**: a public statement does not just inform each trader; it makes the new policy path common knowledge, so every trader knows that every *other* trader has also updated, and can position accordingly. That is why an offhand remark in a 2 a.m. interview moves nothing, but the same words in an FOMC statement move trillions — the venue is what manufactures the common knowledge. We build the formal machinery of this in [common knowledge and "I know that you know"](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know).

The **dot plot** — the Fed's quarterly chart where each of the 19 FOMC participants anonymously marks their projected appropriate policy rate for the coming years and the longer run — is forward guidance turned into a coordination device. It is not a promise; it is a distribution of intentions, published precisely so the market can converge on a shared expectation. When the dots shift up, the market re-prices the whole curve before a single meeting has changed a single rate. The dots *are* the move. The market reads the *median* dot as the central tendency and the *dispersion* of the dots as the committee's uncertainty: a tight cluster signals conviction (a strong commitment the market should trust), while a wide scatter signals the committee itself does not know, which weakens the guidance's coordinating power. In March 2022 the median dot leapt to signal roughly seven hikes that year — a dramatic upward revision that re-priced the front end of the curve violently and well ahead of the actual hikes.

There is a strategic tension baked into central-bank communication that is worth naming, because it is a genuine game between the bank and the market over *information*. The bank wants its guidance to be believed (so it moves expectations), but it also wants to retain the flexibility to change course if the economy surprises — and those two goals fight each other. Guidance that is too firm becomes a commitment the bank may regret; guidance that is too soft fails to move expectations. The Fed resolves this by making its guidance *state-contingent* — "we will hike *if* inflation does X" rather than "we will hike on date Y" — which preserves flexibility while still anchoring expectations to a rule the market can follow. This is the bank trying to commit (to escape the time-inconsistency trap) and stay flexible (to respond to shocks) at the same time, and the whole art of modern central-bank communication is managing that contradiction without losing credibility on either side.

There is also a deep game-theoretic subtlety about *how much* the bank should reveal. A central bank that publishes its full reaction function makes its policy perfectly predictable — which sounds good, but a perfectly predictable bank also surrenders the ability to surprise when a surprise would genuinely help (for instance, a shock-and-awe move to break a one-off panic). So the bank manages a portfolio of transparency: maximally clear about its *long-run commitment* to the inflation target (where credibility lives), and deliberately less precise about its *near-term tactical moves* (where flexibility lives). Reading a Fed statement is reading which dial the bank is turning.

#### Worked example: a rate cut that is already in the price

Suppose the market currently prices a 90% probability that the Fed cuts 25 bps at its next meeting, and a stock index is trading at a level consistent with that expectation. You are tempted to buy ahead of the "good news" of a cut. Walk the game.

If the Fed cuts 25 bps exactly as expected, the *surprise* is near zero — the cut was 90% priced, so the index barely moves, or even falls as a few "buy the rumor" longs take profit (sell the news). Your expected gain from the cut alone is roughly the 10% chance it *did not* happen times the move you avoided — small. The real money was made by whoever bought when the cut was 40% priced and rode it to 90%.

Now suppose the Fed cuts but its forward guidance signals *more* cuts ahead than the market expected — a dovish surprise in the *path*, not the level. *That* moves the index, because the new information is in the guidance, not the action. Numerically: if the market priced two cuts this year and the dots now show four, the front-end of the curve can reprice 30-50 bps in minutes, dragging equities and the dollar with it.

The intuition: you never trade the rate decision; you trade the *gap between the decision-plus-guidance and what was already priced*. The Fed's action is almost always the least informative part of an FOMC day. The surprise lives in the words.

## Don't fight the Fed: why it is an equilibrium, not a slogan

"Don't fight the Fed" is the oldest piece of market wisdom there is, usually attributed to investor Marty Zweig. Beginners hear it as folklore — *the Fed is powerful, respect it.* But it is a precise game-theoretic statement, and seeing why turns it from a vibe into a rule you can size positions on.

Frame it as a game. You are a trader; the Fed is your counterparty. You can **fight** (take a position betting the Fed will fail to do what it says — e.g., short bonds expecting it to lose control of inflation, or bet a currency peg breaks) or **not fight** (position with the stated policy). The Fed can be **committed** (it will defend its stance with whatever it takes) or **wavering** (it lacks the will or the means and will fold). The payoffs, and why the Fed's commitment is so often the dominant move, come down to one structural fact.

![Don't-fight-the-Fed before-and-after showing a trader's finite capital against the central bank's unlimited balance sheet](/imgs/blogs/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed-5.png)

The asymmetry on the left of the diagram is everything: **you have finite capital; a central bank that issues its own currency has an effectively unlimited balance sheet in that currency.** If the Fed wants to hold a bond yield down, it can print dollars and buy bonds without limit — it can never be forced to stop by running out of dollars, because it *makes* the dollars. This is what the phrase "betting against the house" means literally: in a casino the house has a bankroll vastly larger than yours and a positive edge, so even when you are right on a given hand, the size asymmetry grinds you out. Against a committed currency-issuing central bank operating in its own currency, you are the player and the Fed is the house.

So solve the fight-the-Fed game. If the Fed is committed (and a major central bank acting in its own currency almost always *can* be, even when politically it might prefer not to), then a trader who fights it loses: the Fed can move the price further and longer than the trader can stay solvent. Your best response to a committed Fed is **don't fight**. The Nash equilibrium is **(don't fight, committed)** — and the toolkit confirms the trader's "fight" move is dominated whenever the Fed's commitment is credible.

But — and this is the trade hiding inside the rule — the asymmetry only holds **in the currency the bank can print.** A central bank defending its *own* currency against *depreciation* is the opposite case: to prop up its currency it must *sell* foreign reserves (dollars), and those it cannot print. Its ammunition is finite. That is why "don't fight the Fed" does not transfer to "don't fight an emerging-market central bank defending a peg" — there, *you* may be the one with the deeper effective bankroll, because the bank can literally run out of dollars. The rule is conditional on which side of the balance-sheet asymmetry the bank is on.

#### Worked example: sizing a position against the house

You want to short long-dated Treasuries in 2021, betting the Fed loses control and yields spike, while the Fed is running quantitative easing — buying \$120 billion of bonds a month and stating it will keep yields low. Model the contest as a war of attrition. You have, say, \$10 million of capital and can withstand maybe a 5% adverse move (\$500,000) before margin calls force you out.

The Fed, by contrast, can buy bonds at \$120 billion a month *indefinitely* — it faces no margin call, no investor redemptions, no solvency constraint in dollars. To win, you need the move to happen *before* your \$500,000 buffer is gone. The Fed needs only to outlast you, which it can do for as long as it chooses. Your break-even is a question of timing against a counterparty with infinite patience and infinite dollars.

The expected value: even if you are *right* that yields "should" be higher, your edge is the probability the Fed wavers (folds) before your buffer runs out, times your payoff, minus the probability it outlasts you, times your loss. Against a *committed* Fed in its own currency, that first probability is near zero, so the EV is negative regardless of whether your fundamental thesis is correct. Being right about the destination loses money if the house controls the path.

The intuition: you can be analytically correct and still lose to a counterparty whose constraint you do not share. Don't fight the Fed is a statement about *constraints*, not *correctness* — and the moment you find a central bank whose constraint *is* binding (finite reserves), the trade flips and you fight.

### Why the unlimited balance sheet is the whole edge

It is worth being precise about *why* a currency-issuing central bank's balance sheet is effectively unlimited, because the claim sounds like hyperbole and is not. When the Fed buys a \$1 million bond, it does not dig into a vault of pre-existing dollars; it *creates* \$1 million of bank reserves by crediting the seller's bank — an entry it can make in any size, instantly, because reserves are simply liabilities of the Fed that the Fed defines into existence. There is no budget it can exhaust, no funding it can fail to raise, no margin call it can receive. This is categorically different from *you*: every dollar you deploy is a dollar you had to have, and a position that moves against you triggers margin calls that can force you out at the worst possible moment regardless of whether your thesis is right.

That asymmetry is exactly what makes the trade against a committed Fed a *negative-expected-value* trade even when your fundamental view is correct, because winning requires the bank to *choose* to stop, and a player who never has to stop will not stop until *it* decides the policy goal is met. The phrase "betting against the house" is not a metaphor here — it is the literal structure. A casino with a near-infinite bankroll and a positive edge grinds out any finite player given enough hands; a currency-issuing central bank with an infinite bankroll in its own currency grinds out any speculator betting it will run out of ammunition, because it cannot run out. The only way to beat the house is to find the table where the house's bankroll is *finite* — and for a central bank, that table is a foreign-currency obligation it cannot print.

This also reframes quantitative easing (QE) — the bank buying long-dated bonds with newly created reserves — as a *commitment technology*, not just a stimulus. By putting its unlimited balance sheet behind a yield target, the bank makes the "don't fight us" message credible: it is not promising to hold yields down, it is *demonstrating* that it can and will, with infinite ammunition. The mechanics of QE, quantitative tightening (QT), and the rate corridor are the bank's actual moves; we treat the plumbing in the [central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) and keep our focus here on why those moves are credible.

## Credibility: a slow-build, fast-loss asset

If commitment is the prize, **credibility** is the currency you buy it with — and credibility behaves like no asset on a balance sheet. It is built grindingly slowly, over years of doing what you said, and it is destroyed in an afternoon. A central bank's credibility is the market's belief that the bank's promise equals its future action: high credibility means the public's expectation tracks the bank's stated target, which is exactly the condition for the good equilibrium.

![Line chart of central bank credibility rising slowly over years then collapsing fast after a policy mistake](/imgs/blogs/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed-6.png)

The asymmetry in the chart — the slow climb, the cliff — is why central bankers are so cautious with their words and so allergic to being seen to flinch. Credibility is a *repeated-game* asset. In a one-shot game, the bank always defects (we proved it). But monetary policy is played every six weeks, forever, and in a **repeated game** cooperation (here: keeping the low-inflation promise) can be sustained if the bank values the future enough — because the cost of cheating once is that the public stops believing you, permanently, which is far worse than the one-period gain from the surprise.

The toolkit makes this exact: model the bank's choice as a repeated prisoner's dilemma where "cooperate" = keep the promise and "defect" = inflate by surprise. The discount factor δ measures how much the bank weighs future periods (δ near 1 = patient, far-sighted; δ near 0 = myopic). Cooperation is sustainable only if δ exceeds a threshold δ\* = (T − R)/(T − P), where T is the temptation payoff, R the reward for cooperating, P the punishment when trust collapses. For our numbers the threshold is **0.5**: a bank that weights the future at least half as much as the present can credibly commit; a myopic bank below that threshold cannot, and the public knows it.

#### Worked example: when the central bank can credibly commit

Take the reputation game with temptation T = 4 (the one-shot gain from a surprise), reward R = 3 (keeping the promise every period), punishment P = 2 (the bad discretionary equilibrium once trust is gone), sucker S = 1. The discount-factor threshold is:

```
import data_gametheory as gt
;; T=temptation, R=reward, P=punishment, S=sucker payoff
delta_star = gt.repeated_pd_delta_threshold(T=4, R=3, P=2, S=1)
print(delta_star)
;; -> 0.5
```

So δ\* = (4 − 3) / (4 − 2) = 1/2. If the bank's effective discount factor is δ = 0.8 (far-sighted, institutionally independent, run by people who care about the next decade), then 0.8 > 0.5 and commitment holds: the one-time temptation gain of (T − R) = 1 is not worth losing the per-period (R − P) = 1 advantage forever. If instead δ = 0.3 (a politically captured bank facing an election next quarter, weighting the future at almost nothing), then 0.3 < 0.5, the temptation dominates, and *no promise it makes is credible* — the public correctly expects high inflation.

The intuition: credibility is not a personality trait of the chairman; it is a *structural* property of how much the institution discounts the future. Independence, long terms, and a legal mandate are all devices to push δ above the threshold. This is the same repeated-game logic that sustains cooperation among rational opponents in any long-lived market relationship — the central bank is just the most consequential repeated player there is.

## Credibility crises: when the central bank loses the game

The whole edifice rests on the public *believing* the bank. When that belief cracks, the bank is thrown back into the one-shot game, where the only equilibrium is the bad one — and clawing back out is brutally expensive. Three patterns recur.

The first is the **costly disinflation** — Volcker's purchase. When a bank has lost credibility, it cannot simply announce a lower target and be believed; it has to *prove* it by absorbing a recession that demonstrates it will choose low inflation over jobs when forced. The size of the recession needed is measured by the **sacrifice ratio**: the cumulative percentage of a year's output you must give up to bring inflation down 1 point. For the Volcker disinflation, estimates put the sacrifice ratio around 1.5-2, and the total cost at multiple years of lost output and double-digit unemployment. That is the price tag on a credibility re-purchase.

#### Worked example: the price tag on a credibility re-purchase

Put numbers on Volcker's bill. Suppose the **sacrifice ratio** is 1.5 — meaning bringing inflation down by 1 percentage point costs 1.5% of a year's output (GDP). Volcker needed to take inflation from roughly 12% down to about 4%, a drop of 8 points. The total output cost is then 8 × 1.5 = **12% of a year's GDP**, spread over the disinflation. On a US economy of roughly \$3 trillion at the time, that is on the order of \$360 billion of cumulative lost output — paid in factory closures, foreclosures, and an unemployment rate that touched 10.8%.

Now ask the game-theory question: why pay that? Because the *alternative* — staying in the high-inflation discretionary equilibrium — was a permanent tax on the economy with no upper bound, and every year the Fed waited, expectations un-anchored further, making the eventual disinflation *more* expensive (a higher sacrifice ratio). Volcker was buying an asset — re-anchored expectations — whose payoff was the four-decade Great Moderation of low, stable inflation. The recession was the purchase price; the credibility was the asset.

The intuition: disinflation is expensive precisely *because* the bank lost credibility first. A credible bank can lower inflation almost for free, because the public simply revises its expectations down on the announcement. A bank that has to *prove* it will choose low inflation pays for that proof in real output. The sacrifice ratio is the market's quoted price for credibility you let lapse.

The second pattern is the **currency-defense failure** — an emerging-market central bank that runs out of the one thing it cannot print: foreign reserves. This is the don't-fight-the-Fed rule running in reverse, and it is the canonical place where fighting a central bank *wins*.

![Line chart of an emerging-market central bank's FX reserves draining to near zero while defending a currency peg before it breaks](/imgs/blogs/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed-7.png)

The chart traces the mechanics of a peg defense: to hold its currency from falling, the bank sells dollars to buy its own currency, reserves drain, and the market — watching the reserve number tick down — knows the exact moment the ammunition runs out. Speculators with finite-but-large capital are now the deeper-pocketed side, and the bank's "commitment" is a bluff with a visible expiry date. The equilibrium flips to (fight, breaks). The classic case is George Soros versus the Bank of England in September 1992, when the UK could not defend the pound's place in the European exchange-rate mechanism; Soros's fund reportedly made around \$1 billion as sterling was forced out. The bank had finite reserves and a finite stomach for the recession-inducing interest rates a defense required; the speculators had more of both.

The third is the **slow erosion** — a bank that loses the public's trust gradually by being seen to be late, political, or wishful. This is the credibility question that hung over the Fed and the ECB in 2021-2022, when both initially called inflation "transitory," were proven wrong as inflation hit 9.1% in the US (June 2022, the highest since 1981) and over 10% in the eurozone, and then had to hike aggressively partly to *re-establish* that they would not tolerate high inflation — to defend the 2% target's credibility, not merely to cool the economy. Christine Lagarde's ECB and Jerome Powell's Fed both spent 2022-2023 paying down a credibility debt incurred by being slow. The hikes were as much a credibility purchase as an economic one.

## Common misconceptions

**"The central bank controls inflation directly, like a thermostat."** No — it controls *expectations*, and expectations control inflation. If the public does not believe the target, the bank can set whatever rate it likes and inflation expectations will drift, because wages and prices are set on what people *expect*, not on the policy rate. The thermostat only works if everyone believes it is connected to the furnace. This is the entire reason credibility, not the rate lever, is the central asset.

**"More central-bank discretion is better — flexibility to respond to events."** The Kydland-Prescott result is precisely that *more discretion makes the bank worse off*, because it changes the public's best response from "expect low" to "expect high." A bank that *can* surprise will be *expected* to surprise, and that expectation is the inflation bias. Tying your hands is a feature. This is counterintuitive and it is the single most important idea in the post: the freedom to cheat is a liability when everyone can see you have it.

**"Forward guidance is just talk; only the actual rate moves matter."** Backwards. Because long rates and asset prices depend on the *expected path* of policy, the talk *is* the policy. A credible statement re-prices the whole curve before any rate changes; the actual change is usually the least informative event on the day, because it was already priced through guidance. The market trades the surprise in the words, not the deed.

**"Don't fight the Fed means the Fed always wins."** Only in its own currency. A central bank defending against *depreciation* must spend finite foreign reserves it cannot print, and there the rule reverses — fighting it can win, as 1992 sterling showed. The rule is conditional on which side of the balance-sheet asymmetry the bank stands on. Confusing the unlimited-print case with the finite-reserve case is how traders lose money on both sides.

**"A central bank can just announce a lower inflation target and be believed."** Only if it has credibility to spend. A bank that has lost it must *prove* the new target by absorbing a costly recession (the sacrifice ratio) — the announcement alone is cheap talk that a rational public discounts. Credibility is earned by action under stress, not granted by press release.

## How it shows up in real markets

**The Volcker disinflation (1979-1983).** The defining credibility purchase. With US inflation near 12% and a Fed nobody believed, Volcker drove the funds rate to roughly 20% in mid-1981, triggered a recession that pushed unemployment to 10.8% by late 1982, and held the line through enormous political pressure. Inflation fell from ~12% to under 4% by 1983. The lasting result was not the rate path but the *re-anchoring of expectations*: the market relearned that the Fed would choose low inflation over jobs, which is what kept inflation contained for the next four decades — the "Great Moderation." The recession was the down-payment on a generation of credibility.

**The pandemic forward guidance (2020-2021).** The Fed cut to zero in March 2020 and pledged to hold rates there, later adopting "average inflation targeting" and guiding that it would not hike until employment and inflation goals were met — explicitly stating it expected near-zero rates through 2023. Thirty-year mortgage rates fell to record lows near 2.65% (January 2021), equities re-rated to record highs, and credit spreads collapsed — all driven by the *expected path*, much of it before any further action. Then came the cost: that same guidance arguably kept policy too easy too long into the 2021 inflation surge, which set up the credibility problem of 2022.

**The "transitory" misjudgment and the 2022 re-anchoring.** Through 2021 both the Fed and ECB framed surging inflation as transitory. By June 2022 US CPI hit 9.1% (a 40-year high) and eurozone inflation topped 10%. Having been wrong, both banks then hiked at the fastest pace in decades — the Fed taking rates from ~0% to over 5% by mid-2023, including four consecutive 75-bp hikes — partly to *re-establish* that the 2% target was a hard commitment, not a wish. The aggressiveness was a credibility signal as much as an economic one: a slow start forced an overt demonstration of resolve.

**Soros vs. the Bank of England, Black Wednesday (September 16, 1992).** The textbook reverse-Fed trade. The UK was defending sterling's floor in the European exchange-rate mechanism, which required either burning finite foreign reserves or raising rates to recession-inducing levels. Speculators, led by Soros's Quantum Fund, judged the BoE's commitment was a finite-reserve bluff and shorted the pound at scale. The BoE hiked rates to 12% then 15% in a single day, spent billions in reserves, and still could not hold — it abandoned the defense and let sterling float, with Soros reportedly netting ~\$1 billion. The lesson: fight a central bank when its ammunition is *finite and visible*.

**EM defenses and the visible-reserve trap.** The same mechanism recurs across emerging markets — Thailand's baht in July 1997 (the spark of the Asian crisis), Argentina's repeated peg collapses, and various defenses where a published, draining reserve number told speculators precisely when the bank would be forced to fold. Whenever a central bank's commitment has a *measurable expiry date*, the equilibrium shifts toward the speculators. The trade is not "central banks are weak" — it is "this particular commitment is finite and the market can see the meter running."

**The reflexive feedback in all of it.** Notice that the bank watches the market that is watching the bank — long rates and breakeven inflation are the market's read on the bank's credibility, and the bank reacts to *those prices* in setting policy. That two-way mirror is its own deep idea, and it is why a credibility wobble can become self-fulfilling: expectations un-anchor, the bank is forced to react, the reaction confirms the fear. We trace this loop in [reflexivity, markets that watch themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves).

## The playbook / How to play it

You will never set monetary policy, but you trade against its shadow on every macro position. Here is the game seen from your seat.

**Who is on the other side.** When you trade rates, FX, or any duration-sensitive asset around a central bank, your counterparty's behavior is *the central bank's reaction function* — what the bank will do given the data — filtered through the *entire market's* expectation of that reaction function. You are not forecasting the economy; you are forecasting the gap between the bank's likely path and what the curve already prices.

**The game you are in.** It is a repeated game of expectations, and the bank is the dominant repeated player whose chief asset is credibility. Most days the equilibrium is "don't fight the committed Fed in its own currency." The edge is in the conditional cases: (1) the *surprise in the guidance* relative to what is priced — trade the dots-vs-market gap, not the rate decision; (2) the *finite-reserve* defenses where the bank's commitment has a visible expiry and the asymmetry flips in your favor; (3) the *credibility-erosion* moments when a bank is being seen to flinch and expectations are starting to un-anchor.

**Your edge and where it lives.** Read the central bank as a strategic player solving its own time-inconsistency problem. When the bank is *building* credibility (hiking to prove resolve, as in 2022), it will tend to over-deliver hawkishness relative to the pure economic optimum — fade the "they'll pivot dovish any minute" trade until the credibility debt is paid. When the bank is *spending* credibility (a finite-reserve peg), short the commitment. When guidance and market pricing diverge, the surprise lives in the smaller, less-watched number — the *path*, not the *level*.

**The invalidation.** Your "fight the central bank" trade is invalid the moment you cannot articulate *which constraint of the bank's is binding*. "Yields should be higher" is not a thesis against a bank that can print; you need a specific finite resource (reserves, political will with a deadline, a legal limit) that the bank is about to hit. If you can't name the binding constraint, you are betting against the house, and the EV is negative even when your fundamental view is right.

**Sizing and exit.** Against a committed currency-issuing bank, size *small or not at all* — the path risk is unbounded because the counterparty has no margin call. Against a finite-reserve defense, the trade is a war of attrition: size to survive the bank's maximum credible defense (the rate spike and reserve burn it can sustain), and the exit is the reserve number or the political deadline, not a price level. In both cases, the central bank's *words* — the guidance, the dots, the press-conference tone — are the highest-information events; build the position around the expectation gap, not the announcement itself. And remember this is educational analysis of a mechanism, not a recommendation to put on any specific trade.

The one rule that ties it together: **credibility is the only thing a central bank truly owns, and the market price of that credibility — in real yields, breakevens, and the currency — is the single most important number you are trading.** When it is intact, don't fight the Fed. When it is finite and you can see the meter running, that is the whole trade.

## Further reading & cross-links

- [The central bank toolkit: rates, QE, QT, and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — the plumbing of *how* the bank moves rates and the balance sheet, the mechanics this post treats as the moves of the game.
- [Buy the rumor, sell the news: public signals and the Fed](/blog/trading/game-theory/buy-the-rumor-sell-the-news-public-signals-and-the-fed) — why a fully-anticipated decision moves nothing, and how forward guidance prices the event in advance.
- [Common knowledge and "I know that you know that I know"](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know) — the formal machinery behind why a *public* central-bank statement coordinates the whole market, where a private one would not.
- [Reflexivity: markets that watch themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) — the two-way mirror between the bank and the market that prices its credibility, and how a credibility wobble becomes self-fulfilling.
