---
title: "The Anatomy of a Bank Run: From Whisper to Collapse"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a single rumor can drain a solvent bank in hours — the Diamond-Dybvig logic of a run, the first-mover advantage, and why the digital age made runs faster than ever."
tags: ["banking", "bank-run", "diamond-dybvig", "liquidity-risk", "maturity-transformation", "silicon-valley-bank", "financial-crisis", "deposit-insurance", "contagion", "fire-sale"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank run is the failure mode every banking disaster is a variation of: a confidence-funded, maturity-transformation machine can be perfectly solvent and still die in hours, because the depositors who run first get paid in full and everyone knows it.
>
> - A bank holds maybe 10 cents of cash for every dollar it owes you. The other 90 cents is lent out long and cannot be recalled on demand. That gap is not a flaw — it is the business — but it is also the trapdoor.
> - The Diamond-Dybvig model shows a healthy bank has *two* stable outcomes: everyone stays (the bank lives) and everyone runs (the bank dies). Which one happens depends only on what each depositor expects everyone *else* to do.
> - Because the bank pays withdrawals first-come-first-served until the cash is gone, running early is the individually rational move — so a rumor is enough to start a stampede, and a solvent bank can be killed by belief alone.
> - The number to remember: in March 2023, depositors tried to pull **\$42 billion from Silicon Valley Bank in a single day**, with roughly **\$100 billion more queued** — a **\$142 billion** run on a **\$175 billion** deposit base, executed from phones, in about 36 hours.

On the morning of Thursday, March 9, 2023, Silicon Valley Bank was, by most conventional measures, a functioning bank. It had \$209 billion in assets, a roster of marquee technology and venture-capital clients, and a thirty-year history. By the close of business the next day it had been seized by regulators. Not because a loan portfolio went bad overnight, not because of fraud, not because the economy collapsed — but because its depositors, talking to each other on Slack channels and group chats and Twitter, all decided at roughly the same moment to take their money out.

The mechanism was almost embarrassingly simple. Word spread that the bank had taken a loss selling some bonds and needed to raise capital. To a community of venture capitalists trained to spot trouble early, that was enough. The advice rippled out — *move your money, just to be safe* — and because moving money in 2023 means tapping a few buttons in an app, \$42 billion left the bank on Thursday alone. That is roughly a quarter of the entire deposit base, gone in a day. Another \$100 billion or so was queued to leave on Friday morning. No bank on earth holds that much ready cash. The Federal Deposit Insurance Corporation stepped in before Friday's withdrawals could clear.

That is a bank run, and it is the subject of this post — the opening case in our study of how banks fail. Here is the unsettling part, the part this entire series is built around: **SVB was not obviously insolvent the day before it failed.** Its assets, if it could have held them to maturity, were worth more than it owed. It died of a liquidity crisis, not an asset crisis — it ran out of *cash*, not out of *value*. Understanding exactly how that happens, and why it is baked into the very structure of banking, is the foundation for every failure we will study after this one. Lehman, Northern Rock, Continental Illinois, Washington Mutual — strip away the specifics and each is a variation on the same theme. They all, in the end, died the same death.

![A calm bank versus a bank run shown as two flow diagrams](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-1.png)

The diagram above is the mental model for the whole post. On a normal day, deposits flow in and out in a steady trickle, the net change is near zero, and the bank's thin cash buffer easily covers it. The bank is fine. On the day of a run, the *same* bank with the *same* balance sheet faces every depositor at the door at once. The cash empties in hours, the bank is forced to dump assets at fire-sale prices, and it collapses. Nothing about the bank changed between the two pictures. Only the behavior of the depositors changed — and that is the entire, terrifying point.

## Foundations: what a bank run actually is

Before we can dissect a run, we need a handful of plain-English definitions. If you already know what maturity transformation is, skim this; if you don't, you cannot follow the rest of the post without it. We will build every term from zero.

### What a bank really is

Start with what a bank does with your money. When you deposit \$1,000 in a checking or savings account, the bank does not put your \$1,000 in a vault with your name on it. It lends most of it out — to a homebuyer as a mortgage, to a business as a working-capital loan, to the government by buying a bond. It keeps only a small slice as cash to handle day-to-day withdrawals. In return it pays you a little interest (or none), and it charges the borrower more, pocketing the difference. That difference — the *spread* — is how a bank makes its living.

The whole arrangement rests on one trick with a formal name: **maturity transformation**. The bank *borrows short* (your deposit, which you can take back any time — its "maturity" is zero, it is due on demand) and *lends long* (a 30-year mortgage, a 5-year business loan — money it won't get back for years). It is, quite literally, in the business of promising to give everyone their money back instantly while simultaneously having lent that money out for years. This is covered in depth in [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread), the opening post of this series; here we only need the one-line version: **a bank's promises are short and its assets are long, and that mismatch is the source of both its profit and its fragility.**

Why does this work at all? Because of the *law of large numbers* applied to depositors. On any given day, some people deposit, some withdraw, and as long as those flows roughly cancel out, the bank only needs a small buffer of cash to bridge the gap. A useful comparison is a coat-check at a theater. The attendant takes a thousand coats and can comfortably hand back the few that get requested during intermission, because nobody asks for all thousand at once. The coat-check works only because demand is staggered. A bank works the same way — until it doesn't.

### Illiquidity versus insolvency — the single most important distinction

If you remember one pair of words from this entire post, make it these two. They sound similar and are constantly confused, and the confusion is exactly what makes runs so dangerous.

**Insolvency** means a bank's *assets are worth less than its liabilities*. If a bank owes depositors \$92 and its loans and bonds are truly worth only \$80, it is insolvent — there is not enough value to go around, and even if you sold everything calmly, someone gets stiffed. Insolvency is a problem of *value*.

**Illiquidity** means a bank *cannot turn its assets into cash fast enough to meet demands as they come due*, even though those assets are worth plenty. The mortgage is good, the bond will pay in full at maturity, the value is genuinely there — but you cannot sell a 30-year mortgage at full price in twenty minutes when a queue of depositors is at the window. Illiquidity is a problem of *timing*.

A healthy bank is, by design, always somewhat illiquid. That is not a defect; it is the maturity-transformation business itself. The danger is that **illiquidity and insolvency are not separate states — a run can convert one into the other.** A bank that is merely illiquid, forced to sell good assets at panic prices, will crystallize losses that push it into genuine insolvency. We will trace that exact conversion later, with numbers. For now, hold the distinction: a run does not need a bank to be insolvent. It only needs the bank to be illiquid — which every bank always is.

### The cash buffer and the reserve ratio

How much cash does a bank actually keep against your deposits? Far less than most people expect. A typical commercial bank holds something on the order of **10 cents of cash and central-bank reserves for every dollar of deposits** — the rest is lent out or invested. (Banks also hold a buffer of high-quality, sellable securities like government bonds as a second line of liquidity; we will come back to why those bonds can fail you exactly when you need them.) The precise number varies by bank and by regulation, but the order of magnitude is the point: **the bank cannot pay everyone at once, and it was never designed to.** If even a fifth of depositors show up demanding cash on the same day, the buffer is gone.

This is not a scandal or a sign of a badly run bank. It is the structure of *every* bank, everywhere, by construction. A bank that held 100% of deposits as cash would be a safe-deposit box, not a bank — it would earn nothing, pay nothing, and create no credit. The fragility is the price of the function.

### The sequential-service constraint

Here is the rule that turns a vulnerability into a catastrophe. When depositors line up to withdraw, the bank pays them **in the order they arrive, in full, until the cash runs out.** The economist's term is the **sequential-service constraint**: first come, first served. The bank does not — cannot, in real time — pause, total up everyone's claims, and pay each person a fair pro-rata share. It hands out full dollars to whoever shows up, one after another, and at some point the till is empty and everyone behind that point gets nothing until a slow, court-supervised wind-down years later.

This single operational fact is what gives a run its terrible logic. If the bank paid everyone an equal fraction regardless of arrival order, there would be no reason to rush. But because it pays in arrival order until the cash is gone, *being early is worth real money.* That is the **first-mover advantage**, and it is the engine of every run.

### The two equilibria

Put the pieces together and you arrive at the central insight, the one a man named Douglas Diamond and his co-author Philip Dybvig formalized in 1983 (work that won Diamond a share of the 2022 Nobel Prize in economics, awarded the same year SVB was quietly building its trap). A healthy bank — solvent, well-run, profitable — has not one but **two stable outcomes**, and which one occurs depends entirely on what depositors *believe* everyone else will do.

In the **good equilibrium**, every depositor expects everyone else to stay calm, so no one has a reason to rush, so the bank comfortably funds its loans and stays alive. Withdrawals trickle, the buffer holds, the bank earns its spread. In the **bad equilibrium**, every depositor expects everyone else to run, so each person rushes to be early, so the bank is drained and dies — even though nothing was actually wrong with it. Both outcomes are self-consistent: in each, doing what you expect everyone else to do is exactly the right move for you. The terrifying implication is that **a run can be entirely self-fulfilling.** The belief that the bank will fail is sufficient to make it fail. No bad loan required.

![Two self-fulfilling equilibria flow from one belief about other depositors](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-2.png)

The diagram traces it. Everything hinges on one question each depositor asks on a nervous morning: *what will everyone else do?* If you expect calm, you leave your money in, the bank lives, and your expectation was right. If you expect a run, you rush to withdraw, the bank dies, and your expectation was also right. The bank's actual health does not select between these branches. Belief does. This is what makes a bank run unlike almost any other financial accident — it is a coordination failure, not a fundamentals failure, and a coordination failure can strike a perfectly good institution out of a clear sky.

## The Diamond-Dybvig model in plain English

Let's make the two-equilibria idea concrete, because the formal model is far simpler than its reputation. You do not need any calculus to feel why it is true.

Take a tiny bank with just two depositors, each of whom put in \$100, and one long-term project the bank invested that \$200 into. The project is genuinely good: if left alone for two years, it pays back \$240, a 20% gain. But if the bank has to yank the money out early, the project is only worth \$180 — pulling it apart early destroys value, the way a half-built factory sells for less than a finished one. This early-withdrawal penalty is the model's stand-in for the real world's fire-sale loss, and it is the crux of everything. The whole apparatus reduces to that one wrinkle: the bank's assets are worth more if held than if liquidated in a hurry, so a hurried liquidation — which is exactly what a run forces — is value-destroying for everyone, and the only question left is who gets out before the destruction.

Now both depositors decide, independently, whether to withdraw early (after one year) or wait (until year two). Walk the four combinations:

- **Both wait.** The project matures, pays \$240, and each depositor gets \$120. Best outcome for everyone. This is the good equilibrium.
- **Both withdraw early.** The bank must liquidate for \$180, so each gets \$90 — *less than they put in.* This is the bad equilibrium, and notice it is a real, stable outcome: given that the other person is withdrawing, withdrawing yourself is correct, because the alternative is being second in line for a half-empty till.
- **One waits, one withdraws.** Now the sequential-service constraint bites. The early withdrawer demands \$100. The bank liquidates enough to pay it. What's left for the patient depositor is whatever the wrecked project yields — possibly far less than \$100. The patient one gets punished for patience.

Stare at that third case, because it is the whole model. **If there is any chance the other depositor runs, your safest move is to run too** — waiting exposes you to getting stiffed. And since the other depositor is reasoning identically about *you*, the fear of a run is self-justifying. Neither of you wants the bad outcome. Both of you, acting individually-rationally, can stampede straight into it.

#### Worked example: the first-mover payoff versus waiting

Let's put real numbers on why running is rational. Suppose you are a depositor at a bank with \$100 of your money in it. The bank holds enough cash to pay the first 80% of depositors in full; the last 20% will, after a forced fire-sale and a slow wind-down, eventually recover about 60 cents on the dollar. You don't know exactly where you'll fall in the line.

Compare your two strategies, given that you think there is, say, a 50% chance a run happens.

**If you wait:**
- If no run (50% chance): you keep your money and earn, say, \$1 of interest. Payoff = \$101.
- If a run happens (50% chance): you are likely *late* (the people who waited are by definition behind the people who rushed), so you land in the unlucky 20% and recover 60 cents. Payoff ≈ \$60.
- Expected payoff = 0.5 × \$101 + 0.5 × \$60 = **\$80.50.**

**If you run now:**
- If no run (50% chance): you withdrew unnecessarily, forfeiting the \$1 of interest and maybe a small early-closure fee. Payoff ≈ \$99.
- If a run happens (50% chance): you are near the front, in the lucky 80% paid in full. Payoff = \$100.
- Expected payoff = 0.5 × \$99 + 0.5 × \$100 = **\$99.50.**

Running gives you \$99.50 expected; waiting gives you \$80.50. **The gap is \$19 of pure self-protection.** And notice the asymmetry: by running, your *worst* case is \$99 — you can barely lose. By waiting, your worst case is \$60 — you can get hurt badly. A rational depositor who simply does not want to be left holding the bag will run, almost regardless of what they think the *true* odds of a run are. That is the engine. The takeaway: in a run, the question is never "is the bank sound?" — it is "will I get my money before the cash runs out?", and the safe answer is always *go now.*

## The depositor's game: a payoff matrix

We can compress the whole logic into a single 2×2 grid — the kind of payoff matrix game theorists use. You have two choices: wait or run. The world has two states: others stay calm, or others run. Four boxes.

![A two by two payoff matrix showing run versus wait against others staying or running](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-3.png)

Read the matrix one row at a time. If you wait and everyone else stays calm, you get the best outcome — full value plus your interest, call it 100 cents on the dollar. But if you wait and everyone else runs, you arrive late, the cash is gone, and you might recover only 60 cents after a long wait. If you run and others stay calm, you've over-reacted slightly — you got your money but forfeited a little interest, maybe 99 cents. And if you run and others run, you are near the front of the line and get your full dollar back, 100 cents.

Now do what a game theorist does and ask: is there a choice that is never the worst option, no matter what others do? Look down each column. *If others stay calm*, running (99) is almost as good as waiting (100) — you barely lose. *If others run*, running (100) is vastly better than waiting (60) — you avoid disaster. **Running is never much worse and is sometimes enormously better.** In the language of game theory, running *weakly dominates* waiting. When a strategy weakly dominates, rational players gravitate to it. This is why a run does not require most people to actually believe the bank is doomed. It only requires them to believe *others might run* — and since everyone is making the same calculation, the mere possibility becomes the reality.

#### Worked example: how many depositors must flee to break the bank

A run does not need *everyone* to participate. It needs just enough to exhaust the cash. Let's compute exactly how many.

Take a bank with \$100 billion of deposits. Suppose it holds:
- \$10 billion in actual cash and central-bank reserves, plus
- \$15 billion in high-quality government bonds it can sell quickly (its liquidity buffer).

That is \$25 billion of resources it can mobilize fast — call it the bank's *ready liquidity*. The remaining \$75 billion of assets are loans and longer-dated securities that cannot be turned into cash in a day without a brutal discount.

So the fraction of deposits that must flee to drain the ready liquidity is:

$$\text{critical fraction} = \frac{\text{ready liquidity}}{\text{total deposits}} = \frac{\$25\text{bn}}{\$100\text{bn}} = 25\%$$

**Just one depositor in four needs to withdraw to empty the bank's fast money.** And here is the cruelty: the moment depositors sense the buffer is thinning, the first-mover logic kicks in for *everyone*, and the 25% who would have triggered the crisis are quickly joined by the rest. The buffer is not a wall; it is a fuse. Once it starts burning, it accelerates. The intuition: a bank's cash buffer is sized for a bad *week*, not a coordinated stampede — and a stampede can blow through a quarter of the deposit base before lunch.

## The first-mover advantage and the sequential-service constraint

We have referenced the first-mover advantage repeatedly; now let's make it the explicit focus, because it is the mechanical heart of the run. The reason a rumor can topple a healthy bank is not psychology in the loose sense — it is a hard structural fact about *how the bank pays out.*

![A grid showing how queue position determines a depositor's payoff in a run](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-9.png)

The grid above walks the line. The first depositor in the door at 9:00 a.m. gets a full dollar for every dollar. So does everyone near the front, because the cash buffer is still there. Around the middle, the buffer is running low and it becomes a coin flip. Then comes the depositor who happens to request the dollar that empties the till — the music stops at that exact person. Everyone behind them faces shuttered doors and a partial recovery that may take months or years to arrive, if it arrives in full at all. The lesson box on the bottom right states the conclusion plainly: because your payoff depends on your *position in the line*, running early is rational — and because everyone knows that, everyone runs.

Contrast this with a system that *removed* the first-mover advantage. Imagine a bank that, the instant a run began, froze all accounts and announced: "We will sell our assets in an orderly way over the next month and pay every depositor the exact same fraction of their balance, regardless of when you asked." Under that rule, there is no reward for being first — your payoff is identical whether you rush or wait. So nobody rushes. The run never gets going. This is not a hypothetical: it is exactly why two of the real-world cures for runs work the way they do.

The first is the **suspension of convertibility** — historically, a bank in trouble could simply slam its doors and stop honoring withdrawals for a few days. Crude, but it breaks the sequential-service constraint by stopping the line entirely. The second, far more important, is **deposit insurance**, which we will turn to shortly: if a government guarantees you get your money no matter what, your payoff no longer depends on your place in the queue, so you have no reason to run. Both cures attack the same root: they neutralize the first-mover advantage. Everything else — discount-window lending, capital buffers, liquidity rules — buys time, but the first-mover advantage is the disease itself.

#### Worked example: how fast \$X drains

Let's quantify the speed, because the speed is what changed everything in the modern era. Take our \$100 billion bank with \$25 billion of ready liquidity. Suppose a run starts and withdrawals come in at a steady rate.

**In the old world**, depositors had to physically queue at a branch. A busy teller can process maybe one withdrawal every two minutes, and a large branch might run twenty tellers. That is roughly:

$$20 \text{ tellers} \times 30 \text{ withdrawals/hour} = 600 \text{ withdrawals/hour per branch}.$$

If the average withdrawal is \$5,000, a branch bleeds about \$3 million an hour, or \$24 million in an 8-hour day. Across a hundred branches, that is \$2.4 billion a day. To drain \$25 billion of ready liquidity at that pace takes **about ten business days.** Ten days is slow enough for a central bank to organize a rescue, for a buyer to be found, for cooler heads to intervene. The branch and the teller were, accidentally, a circuit breaker.

**In the digital world**, there is no teller and no queue. A depositor with an app can move \$10 million in the time it takes to tap "confirm," and ten thousand depositors can tap "confirm" in the same five minutes. SVB's \$42 billion in one day works out to roughly **\$5 billion an hour** during business hours — more than two thousand times the per-branch rate of the analog era. At that pace, \$25 billion of ready liquidity is gone in **five hours.** The intuition: the same panic that took ten days to play out in 1933 now plays out before the close of a single trading session — and no rescue can be assembled that fast.

## The tipping point: when a whisper becomes a stampede

A run is not a switch that flips; it is a chain reaction that crosses a threshold. Understanding the *sequence* tells you where it can — rarely — be stopped, and why it usually can't.

![A pipeline from a whisper through panic to a bank's collapse](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-5.png)

It starts with a **whisper** — a disclosed loss, a downgrade, a rumor, a single viral post. Then the **first withdrawals**: a few cautious depositors move money "just to be safe." Crucially, these first movers are not panicking; they are being prudent. But their action is observable, and it raises the perceived probability that *others* will move, which is exactly the variable that drives everyone's decision. **Panic spreads** as the observation feeds back: people see (or hear, or read screenshots of) others leaving, the first-mover math tips them, and now the withdrawals are self-reinforcing. This is the tipping point — the moment the run goes from "a few nervous people" to "everyone." Past it, you get the **mass run**, the cash buffer drains, the bank is forced into a **fire-sale** of its assets to raise cash, and the losses from selling good assets at bad prices finally render it insolvent. **Collapse**: out of cash and out of capital, the regulator seizes it over a weekend.

The feedback loop is the key feature. In a normal market, a falling price attracts buyers — it is self-correcting. In a run, a falling deposit base attracts *more withdrawals* — it is self-*amplifying*. Each person who leaves makes leaving more attractive for the next person, because the buffer is now thinner and the first-mover advantage is now sharper. There is no negative feedback to stop it short of an outside force. That is why runs, once past the tipping point, are almost impossible to halt with words alone. The CEO of SVB held a call on Thursday afternoon urging clients to "stay calm." It is now studied as a textbook example of how *telling people not to panic confirms there is something to panic about.* The whisper had already become a stampede.

#### Worked example: the cash buffer versus the run

Let's watch a buffer fail in real time. Take a bank with:
- \$200 billion in deposits,
- \$20 billion in cash and reserves (a 10% buffer),
- \$30 billion in sellable government bonds.

So \$50 billion of ready liquidity against \$200 billion of deposits — a 25% liquidity coverage that, on paper, looks robust. Now a run starts and depositors pull money at \$8 billion per hour (a plausible digital-era pace for a bank this size).

- **Hour 1:** \$8bn out. Cash down to \$12bn. Fine.
- **Hour 2:** \$8bn out. Cash exhausted; the bank starts selling bonds. \$4bn of bonds sold.
- **Hour 3:** \$8bn out. \$26bn of bonds left.
- **Hours 4–6:** another \$24bn out. Bonds nearly gone.
- **End of Hour 6:** roughly \$48–50bn withdrawn, the entire ready-liquidity stack is exhausted, and the run is still going.

**Six hours.** A 25%-liquidity bank — a number that would pass a regulatory snapshot — is out of fast cash in a single morning if the run runs at digital speed. The remaining \$150 billion of assets are loans and long bonds it cannot sell today without a fire-sale discount. The intuition: liquidity ratios are measured against *modeled* stress scenarios that assume a gradual, partial outflow; a coordinated, app-driven stampede outruns the model, and the buffer that looked like a fortress turns out to be a few hours of runway. This is precisely the territory of [liquidity management and the LCR / NSFR](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the rules built to size that buffer, and their limits when a run moves faster than the rules assumed.

## Why a solvent bank can still die: the fire-sale spiral

We now arrive at the most counterintuitive and most important mechanism in the whole subject: how a run *creates* the insolvency it feared. A run does not just reveal a bad bank. It can manufacture a bad bank out of a good one.

![Before and after a fire-sale converts an illiquid bank into an insolvent one](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-7.png)

Walk the before-and-after. **Before the run**, the bank's assets — loans and bonds — are worth \$100 at fair value. It owes depositors \$92. So it has an \$8 equity cushion: it is *solvent*, worth more than it owes, with a healthy margin. Its only problem is that just \$10 of those assets are in cash; the rest is tied up. It is illiquid, not insolvent. **Then the run forces a fire-sale.** To raise cash fast, the bank dumps its bonds into a falling market and gets only 85 cents on the dollar. Selling assets that were worth \$100 for \$86 in a fire-sale *destroys \$14 of real value* — value that existed on Wednesday and is gone by Friday. Now the assets are worth \$86, the bank still owes \$92, and the equity cushion of \$8 has become a hole of *negative \$6.* The bank is now genuinely insolvent. **The run didn't discover insolvency; it caused it.**

There is a deeper reason the discount is so brutal precisely when the bank can least afford it. The price you get in a fire-sale is not the asset's fundamental value — it is whatever a buyer will pay *right now*, and a forced seller is the weakest possible negotiating party. Everyone in the market knows the bank *must* sell to raise cash by the close, so they low-ball it; the bank has no power to wait for a better bid. Worse, the natural buyers of the bank's bonds are *other banks*, and during a panic they are hoarding their own liquidity and have no appetite to add risk. So the pool of buyers shrinks exactly when the seller is most desperate. The result is that the fire-sale discount is not a fixed haircut — it widens the more distressed the seller looks and the more frightened the market is, which is why losses in a run are so often far larger than any pre-run stress model predicted.

This is the fire-sale spiral, and it has a vicious second loop. When one bank dumps bonds, it pushes their price down — which means *every other bank holding those bonds* now has a lower-valued portfolio, which makes *them* look weaker, which can start a run on *them*. The fire sale is contagious through prices, not just through rumor. We will return to contagion shortly; for now, hold the central point: **a run is not a referendum on whether a bank is sound. It is a force that can render an otherwise-sound bank unsound, by forcing it to convert long-term value into immediate cash at a ruinous discount.** This is why "but the bank was solvent!" is no defense. Solvency at fair value is cold comfort when you are being made to sell at fire-sale value.

#### Worked example: the fire-sale that breaks the bank

Let's nail the arithmetic, because the asymmetry is the lesson. A bank holds \$50 billion of long-term bonds, bought when interest rates were low. Rates have since risen, so on a mark-to-market basis these bonds are worth \$43 billion — a \$7 billion unrealized loss. (This is exactly the situation SVB was in, and it is the focus of the next post, on [the SVB duration trap](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run).)

Here is the trick that lets banks live with this. If the bank holds the bonds to maturity, it gets back the full \$50 billion face value — the \$7 billion loss is only "on paper" and never becomes real, as long as the bank is never *forced to sell.* So the bank reports the bonds at \$50 billion (a category called "held-to-maturity") and carries on. On paper, fine.

Now a run forces a sale. The bank must liquidate \$30 billion of those bonds *today* to raise cash. Two things happen:
1. The paper loss becomes real: selling at market means realizing the loss. On \$30 billion of bonds, that's roughly $30/50 \times \$7\text{bn} = \$4.2$ billion of crystallized loss, straight out of equity.
2. Dumping \$30 billion of bonds into a nervous market pushes the price down *further* — a fire-sale discount on top of the rate-driven loss. Say that costs another \$1.5 billion.

So a forced sale converts a comfortable \$7 billion of *paper* loss into nearly **\$5.7 billion of realized capital destruction** in an afternoon. If the bank's total equity was \$12 billion, it just lost almost half its capital cushion in one transaction — and the run is not over. The intuition: the entire "held-to-maturity" accounting treatment is a bet that the bank will *never be forced to sell.* A run is precisely the event that breaks that bet, which is why the paper losses that everyone agreed to ignore become the losses that kill the bank.

## The digital-age run: faster than any in history

We touched on speed in the worked example above; now let's give it the section it deserves, because the change in *velocity* is arguably the most important development in the study of bank runs in ninety years.

For most of banking history, a run had a natural speed limit: human bodies in physical space. To withdraw your money you had to travel to a branch, stand in a line, and wait for a teller. That friction was, in retrospect, a feature. It rationed the rate of withdrawal, it gave the run a visible shape (the famous photographs of crowds outside Northern Rock in 2007), and it bought time — days, usually — during which a central bank could announce support, a buyer could be lined up, or the panic could simply burn out as people saw the doors were still open.

![A horizontal bar chart comparing how many hours bank runs took across eras](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-6.png)

The chart tells the story of the disappearing friction. The teller-window runs of the 1930s played out over many days — the acute phase often ran a working week or more. Northern Rock's queues in September 2007, the first run on a British bank in nearly a century and a half, lasted about three days before the government guaranteed deposits. SVB in 2023 was effectively over in about **36 hours**, with the fatal withdrawals concentrated in a single day. The trend line points one direction: faster, every time.

Two forces drove the acceleration. The first is **mobile and online banking.** Moving money no longer requires a body in a building; it requires a thumb on a screen. A depositor can empty a multi-million-dollar account in seconds, at 2 a.m., from another continent. The physical circuit breaker is gone. The second, and arguably more potent, is **social media.** A run is, at its core, a coordination problem — everyone is trying to guess what everyone else will do. Social media is a machine for solving coordination problems instantly. When a venture capitalist with fifty thousand followers tweets "get your money out of SVB," and a hundred group chats light up with screenshots, the *common knowledge* that drives a run — not just "I'm worried," but "I know that you're worried, and you know that I know" — forms in minutes instead of days. The whisper that used to spread person-to-person across a town now spreads to an entire customer base in a single afternoon.

There is a third, subtler factor specific to modern wholesale-funded and corporate-deposit banks: **concentration and sophistication.** SVB's depositors were overwhelmingly businesses, not households, and overwhelmingly *uninsured* — about 94% of its deposits exceeded the FDIC's \$250,000 guarantee. Uninsured business depositors are exactly the people with the most to lose and the most reason to move first, and they are connected to each other through tight professional networks. A bank funded by a thousand large, sophisticated, interconnected, uninsured depositors is far more run-prone than one funded by a million small, insured, dispersed households — even if the balance sheets look identical. The *quality* of the deposit base, not just its size, determines how fast it can flee. This is why deposit *stickiness* is treated as a core asset in banking, and why the cheap, sticky, insured retail deposit is considered the franchise's crown jewel.

## Contagion: how one run becomes many

A single bank failing is a tragedy for its depositors and shareholders. A single bank failing that *triggers runs on healthy banks* is a systemic crisis. The difference is **contagion** — the mechanism by which fear, and losses, jump from one institution to the next. There are three distinct channels, and they often fire together.

The first is **informational contagion.** When Bank A fails, depositors at Bank B ask: "Is my bank like that one?" If Bank B has any visible similarity — the same business model, the same type of assets, the same kind of depositor — the answer "maybe" is enough to start the first-mover calculation. This is what happened in March 2023: SVB's failure on Friday triggered an immediate run on Signature Bank (seized that Sunday) and a slower bleed at First Republic (which failed weeks later), because all three were perceived as the same *kind* of bank — heavy on uninsured deposits and underwater bonds. Depositors were not being irrational; they were updating on genuinely relevant information. The failure of one bank is real evidence about the others.

The second is **fire-sale contagion**, the price channel we met earlier. When a failing bank dumps its bond portfolio, it drives down the market price of those bonds. Every other bank holding the same bonds now shows a bigger paper loss, looks weaker, and becomes more vulnerable to its own run. Losses propagate through shared asset prices even between banks that have no direct relationship. The more banks crowd into the same assets, the more powerful this channel becomes.

The third is **interbank contagion** — the direct-exposure channel. Banks lend to each other constantly, overnight, in enormous size, through the interbank and [repo markets](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight). If Bank A fails owing Bank B, then Bank B takes a direct loss, which can weaken or topple it, which threatens Bank C that lent to *B*, and so on down the chain. This was the dominant channel in 2008, when the failure of Lehman Brothers froze the entire money market because no bank could be sure which of its counterparties was about to follow Lehman down. The interbank market, which normally distributes liquidity efficiently, becomes a network for transmitting collapse.

#### Worked example: the contagion multiplier

Let's see how a small initial loss can cascade. Suppose five banks each lent \$10 billion to a sixth bank that just failed, and the failed bank's assets will eventually recover only 50 cents on the dollar.

- **Direct losses:** each of the five lenders loses \$5 billion (half of \$10bn). If a lender's equity was \$15 billion, a \$5 billion hit is a third of its capital — survivable, but it now looks weaker and has less buffer against a run.
- **Second round:** suppose one of those five, weakened, now faces a run and is forced into a fire-sale that knocks 5% off a widely-held \$200 billion bond market. Every bank holding those bonds — say twenty institutions with \$100 billion each — takes a 5% mark-down: $0.05 \times \$100\text{bn} = \$5\text{bn}$ of paper loss *each*, \$100 billion across the system, from a single forced sale.
- **The multiplier:** a \$5 billion direct loss has, through the fire-sale channel, propagated into \$100 billion of system-wide losses — a 20× amplification — without any new bad loans being made.

The intuition: in a connected banking system, the loss that *matters* is not the initial loss but the *amplified* loss after it ricochets through shared exposures and forced sales. This is why regulators obsess over systemic risk, why "too big to fail" exists, and why the cure for a run is almost never aimed at the failing bank alone — it is aimed at stopping the contagion before it spreads. That cure — deposit insurance and the lender of last resort — is the subject of [its own post](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard), and it is the single most effective firebreak ever invented for runs.

## Common misconceptions

A handful of beliefs about bank runs are widespread and wrong. Correcting them is where the real understanding lives.

**"A bank run only happens to a bank that's actually broke."** This is the deepest and most dangerous misconception, and the entire Diamond-Dybvig framework exists to refute it. A run can kill a *solvent* bank — one whose assets exceed its liabilities at fair value — purely through the self-fulfilling coordination failure. SVB's assets, held to maturity, were arguably worth more than its deposits; it died of a liquidity run, not an insolvency. The run does not require the bank to be broke; it can *make* the bank broke via the fire-sale spiral. Solvency is necessary for a bank to deserve to survive, but it is not sufficient to guarantee it will.

**"My money is sitting in the bank's vault."** Almost none of it is. A typical bank holds roughly 10 cents of cash per dollar of deposits; the other 90 cents is lent out or invested in longer-term assets. The bank's promise to give you your money "on demand" is a promise it can only keep because not everyone demands at once. This is not a trick played on you — it is the definition of a bank, and it is what allows the bank to pay you interest and fund the economy. But it does mean the vault metaphor is fiction.

**"Deposit insurance makes bank runs impossible."** It makes runs *by insured depositors* almost impossible — and that is an enormous achievement; classic household runs nearly vanished in countries with credible insurance. But it does nothing for *uninsured* deposits, which is exactly why SVB, with 94% of deposits above the \$250,000 limit, was still acutely run-prone in 2023. Insurance removes the first-mover advantage only for the balances it covers. Above the limit, the old logic is fully intact. Modern runs are increasingly runs by the *uninsured*.

**"The bank can just borrow to cover the withdrawals."** Sometimes, but a run is precisely the moment when borrowing dries up. Other banks, watching a run unfold, are the *least* willing to lend into it — they fear they will not be repaid, and they want to hoard their own liquidity. The bank can pledge assets to the central bank's discount window, but doing so signals distress and can accelerate the run, and the collateral may be valued at a haircut. Private funding is pro-cyclical: it is abundant when you don't need it and absent when you do. That is exactly why the *lender of last resort* — a central bank willing to lend freely against good collateral when no one else will — had to be invented.

**"Stopping a run is about restoring confidence with reassuring words."** Words are nearly useless once the tipping point is passed; a CEO saying "stay calm" often confirms the danger. What actually stops a run is removing the *first-mover advantage* — through deposit insurance, a credible guarantee, suspension of withdrawals, or a central bank standing behind the bank with unlimited liquidity. Runs are coordination problems, and you don't solve a coordination problem with a pep talk. You solve it by changing the payoffs so that running is no longer the safe move.

## How it shows up in real banks

The abstract model earns its keep when it explains real collapses. Here are the canonical cases, each a variation on the single mechanism, and each a preview of a deeper post to come in this series.

![A bar chart of FDIC bank failures per year showing the 2008 to 2012 wave and the 2023 spike](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-8.png)

Before the case studies, look at the shape of failure itself. The chart of FDIC-insured bank failures by year shows the one feature you most need to internalize: **failures come in waves, not as a steady drizzle.** There were *zero* failures in 2005 and 2006, then **465 banks failed between 2008 and 2012** — peaking at 157 in 2010 — and then a long quiet broken by the sharp 2023 spike of just five failures that nonetheless included three of the largest in U.S. history. Runs cluster because their triggers are correlated (a common shock, like a rate cycle) and because contagion links them. A bank is not just exposed to its own mistakes; it is exposed to the panic that another bank's mistake can ignite.

### Silicon Valley Bank, March 2023: the digital run

The defining run of the smartphone era, and the one that re-taught the world this whole subject. SVB had loaded up on long-dated bonds during the low-rate years; when the Federal Reserve hiked rates aggressively through 2022, those bonds fell in value, leaving SVB with roughly \$17 billion of unrealized losses across its securities books. The trigger was a March 8 announcement that it had sold \$21 billion of assets at a \$1.8 billion loss and needed to raise \$2.25 billion of capital. To its concentrated base of uninsured, interconnected venture-capital and startup depositors, that was the whisper.

![A bar chart of SVB deposits, the Mar 9 withdrawal, the Mar 10 queue, and the two-day total](/imgs/blogs/the-anatomy-of-a-bank-run-from-whisper-to-collapse-4.png)

The chart shows the scale against the \$175 billion deposit base. On Thursday, March 9, depositors attempted to withdraw **\$42 billion** — about a quarter of all deposits in one day. By Friday morning roughly **\$100 billion more** was queued. That \$142 billion two-day total is more than four-fifths of the entire deposit base, attempted in 36 hours, from phones. No bank holds that liquidity. The FDIC seized SVB on Friday, March 10 — the second-largest bank failure in U.S. history at the time. It is the cleanest example we have of every mechanism in this post: a paper loss (illiquidity, not insolvency), a concentrated uninsured base (run-prone), a social-media trigger (instant coordination), and a digital execution speed (no circuit breaker). The full anatomy is the subject of [the next post on SVB](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run).

### Northern Rock, September 2007: the photographs

The first run on a major British bank since 1866, and the last great *analog* run — the one with the queues you've seen in photographs. Northern Rock was a mortgage lender funded not mainly by sticky retail deposits but by short-term wholesale borrowing from other banks. When the 2007 credit crunch froze that wholesale market, Northern Rock could not roll over its funding and had to ask the Bank of England for emergency support. The news of that request, on September 13–14, 2007, was the whisper. Depositors — *retail* depositors, ordinary savers — formed physical queues outside branches for three days. The run only stopped when the U.K. government announced, on September 17, a guarantee of all Northern Rock deposits. That guarantee is the textbook cure: it removed the first-mover advantage by promising everyone their money regardless of position in line. The bank was eventually nationalized. The lesson it taught — that a wholesale-funded bank is running a hidden, faster maturity mismatch than a deposit-funded one — went tragically unlearned before 2008.

### The 1930s: the runs that built the system

The runs we still have institutions to prevent. In the United States between 1930 and 1933, successive waves of bank runs destroyed roughly nine thousand banks. These were classic household runs: ordinary people, terrified by the Depression and by the failure of neighboring banks, queuing to pull cash out of perfectly solvent institutions and stuffing it under mattresses — which, by removing money from the banking system, deepened the Depression itself. The runs were self-fulfilling at national scale. The response reshaped banking permanently: the Banking Act of 1933 created the FDIC and federal deposit insurance, which by guaranteeing deposits up to a limit removed the first-mover advantage for households and *ended the era of mass retail runs.* For roughly seventy-five years, deposit insurance worked so well that bank runs nearly disappeared from the public imagination — until the uninsured, digital runs of the 21st century reminded everyone that the underlying fragility never went away. It was only ever covered up, deposit by insured deposit.

### Lehman Brothers, September 2008: the run on a bank with no depositors

The case that proves the run is about *funding structure*, not retail deposits. Lehman Brothers was an investment bank — it had almost no ordinary depositors at all. Yet it died of a textbook run, because it funded itself by borrowing tens of billions overnight in the [repo market](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight), pledging its assets as collateral, and rolling that borrowing over every single day. That overnight funding *is* a deposit in every way that matters: it is short-term money funding long-term assets, and it can flee. When confidence in Lehman cracked, its repo lenders refused to roll over — the wholesale-market equivalent of a run on the windows — and a firm with \$639 billion in assets and over 30× leverage was bankrupt within days. Lehman's collapse is the ultimate proof that **a run is a property of the funding model, not of the customer.** Any institution that borrows short and lends long, whatever it calls its funding, can be run.

## The takeaway: how to read every bank failure that follows

Here is the durable mental model to carry into the rest of this series. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and lends long, earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive. The bank run is the failure mode where the *confidence-funded* part breaks — and because the maturity mismatch is structural and the cash buffer is thin, confidence is the load-bearing wall. Knock it out and everything else, however sound, comes down.

What this means in practice, for reading any bank: **separate the two questions that the word "failure" blurs together.** Is the bank *insolvent* — are its assets genuinely worth less than its liabilities? Or is it merely *illiquid* — can it not raise cash fast enough, even though the value is there? The two require completely different cures (more capital for the first, more liquidity for the second), and confusing them is how both banks and regulators make their worst mistakes. The further, harder truth is that the two are not independent: a run can convert illiquidity into insolvency through the fire-sale spiral, so a bank that is "only" illiquid is one panicked morning away from being insolvent. There is no such thing as a liquidity problem that cannot become a solvency problem if it lasts long enough.

And watch the variable that the model says actually matters, which is almost never the one in the headlines. The headline asks "is the bank making money?" The model says: *how run-prone is its funding?* A bank with a sticky base of small, insured, dispersed retail deposits can survive a great deal of bad news. A bank with a concentrated base of large, uninsured, interconnected, sophisticated depositors — or one funded by overnight wholesale money — is one whisper away from a stampede, no matter how profitable it looks. The 21st-century innovations of mobile banking and social media did not create this fragility; they removed the friction that used to hide it, compressing a ten-day run into a five-hour one and leaving no time for rescue. So when you evaluate a bank, look past the profit line to the funding line, and ask the only question a depositor in a panic ever asks: *if everyone tried to leave at once, how long would the cash last?* Every case study that follows in this series — SVB, Credit Suisse, Lehman, Northern Rock, Continental Illinois, the savings-and-loans, Washington Mutual — is, underneath its particulars, the same story you now know: a confidence-funded machine that ran out of confidence, and then out of cash, in that order.

## Further reading & cross-links

- [What a bank actually does: maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) — the series opener, where the borrow-short / lend-long machine that makes runs possible is built from the ground up.
- [Liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the Basel rules that size the cash buffer a run drains, and exactly why a liquidity ratio that passes on paper can still be a few hours of runway.
- [Deposit insurance, the lender of last resort and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — the two great firebreaks against runs, how they neutralize the first-mover advantage, and the cost they impose in distorted incentives.
- [Silicon Valley Bank 2023: the duration trap and the 36-hour digital run](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run) — the next post in the failures track, taking the SVB run apart bond by bond and hour by hour.
- [SVB and Credit Suisse 2023: the bank runs that re-taught the world](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level view of the March 2023 episode, for how these runs reshaped the regulatory conversation.

*This is educational material about how banks function and fail, not financial advice. Figures for specific institutions are drawn from regulatory filings and contemporaneous reporting as cited; live banking data goes stale quickly, so treat dated numbers as of the dates given.*
