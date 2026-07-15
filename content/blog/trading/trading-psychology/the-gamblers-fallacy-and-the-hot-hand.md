---
title: "The Gambler's Fallacy and the Hot Hand: Misreading Randomness"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "Traders hold two opposite superstitions about randomness at once — a streak is 'due' to end and a streak is 'due' to continue — and both quietly leak money. Here is the science of the gambler's fallacy and the hot hand, the dollar cost of each, and a shuffle-test drill that separates real structure from noise."
tags: ["trading-psychology", "gamblers-fallacy", "hot-hand", "clustering-illusion", "behavioral-finance", "randomness", "base-rates", "mean-reversion", "position-sizing", "probability"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 36
---

> [!important]
> **TL;DR** — Your brain is a pattern-finding machine pointed at a mostly-random world, so it invents two opposite superstitions and often holds both at once: after a run, the next outcome is either "due to reverse" (the gambler's fallacy) or "due to continue" (the hot hand). Both are misreadings of the same noise.
>
> - **The gambler's fallacy** (Tversky & Kahneman, 1971) is the belief that independent events self-correct: after five reds, black is "due." It is false because a coin and a roulette wheel have no memory — the odds of the next spin never change.
> - **The hot hand** is the mirror image: streaks will keep going. Gilovich, Vallone & Tversky (1985) found the basketball hot hand was "largely a cognitive illusion" — until Miller & Sanjurjo (2018) uncovered a selection bias in that very analysis, which means a small, real hot hand was hiding under it the whole time. Present both fairly: streaks mislead *and* streaks can be partly real.
> - **The clustering illusion:** genuinely random sequences look streaky. In 100 coin flips you should *expect* a longest run of about 6 or 7 — roughly log₂(N) — so a six-long streak is nothing special, and the sequence people *invent* as "random" is far too tidy to be real.
> - **The number to remember:** on 18 August 1913 the Monte Carlo roulette wheel came up black 26 times in a row — a run with a probability around 1 in 68 million — and gamblers lost millions betting on red because it felt "due."
> - **The drill:** shuffle-test your own equity curve or a price series; structure that survives the scramble is real, structure that vanishes was noise. Before any "due" bet, ask two questions: *are these events actually independent, and what is the base rate?*

On the evening of 18 August 1913, at a roulette table in the Monte Carlo Casino, the little ivory ball did something that felt, to everyone watching, like it violated a law of nature. It landed on black. Then black again. And again. As the run stretched past ten, past fifteen, past twenty, the crowd around the table swelled and the betting turned frantic — almost all of it on red. Surely, after this many blacks, red was overwhelmingly due. The wheel did not agree. It came up black **26 times in succession**, and by the time it finally landed on red, the casino had taken millions of francs from players who were certain that the "law of averages" owed them a correction. The [Monte Carlo fallacy](https://en.wikipedia.org/wiki/Gambler%27s_fallacy) got its name that night.

Here is the uncomfortable part: the exact same machinery that emptied those pockets in 1913 is running in your head every time you look at a price chart. It is the machinery that makes you short a stock because it is "up too many days in a row," and — in the very same session — makes you buy a level because "it has bounced three times, it'll bounce again." Those are opposite bets justified by opposite theories, and you can hold both within the same hour without noticing the contradiction. That is the tell that neither belief is a signal. Both are your brain doing what it evolved to do: finding patterns in noise. The diagram below is the mental model for the whole piece — one streak, two stories, and a single question that separates them from luck.

![One near-random streak splits into two opposite fallacies — 'it's due to reverse' and 'it will continue' — and only the pre-bet question of independence and base rate tells either one apart from noise.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-1.webp)

Read it left to right. A near-random sequence throws off a streak — five red candles, three bounces off a level, four winning trades. That streak forks your mind two ways. Down the top path is the gambler's fallacy: *it's due to reverse.* Down the bottom path is the hot-hand belief: *it will continue.* Notice that the streak is identical in both cases; only the story you attach to it differs. The two stories then collapse into a single pre-bet question — *are these events independent, and what's the base rate?* — and that question is the fork that matters. Skip it and you bet the pattern, and your edge leaks away one superstition at a time. Ask it and you size by the base rate, and your edge survives. Everything that follows is a tour of that picture. We start from zero — no probability background assumed.

## Foundations: the building blocks of a brain that misreads randomness

You need no statistics and no trading experience for this section. We are going to define, from scratch, the four ideas that make both fallacies possible: what a *base rate* is, what makes an event *independent*, why "random" does not mean "evenly spread," and the precise difference between the two superstitions. Everything technical in the article is built on these.

### A "base rate" is the long-run frequency — and it is what you should be estimating

Start with the single most useful phrase in this article: *base rate.* A base rate is just the long-run frequency of something — how often it actually happens over a large number of trials. The base rate of a fair coin landing heads is 50%. The base rate of red on a single-zero roulette wheel is 18 out of 37 — there are 18 red pockets, 18 black, and one green zero — so the odds of red are ${18/37}$, or about 48.6%. The base rate of your favourite chart setup working is whatever your journal says across hundreds of instances, not across the last three.

Almost every question that matters in markets is secretly a base-rate question. "Will this bounce?" means "what fraction of the time, historically, has this kind of setup bounced?" "Is it due to pull back?" means "how often, after a run like this, does a pullback actually follow?" The correct, boring, profitable answer is nearly always a number you could look up. Both fallacies are what happens when your brain refuses to look it up and substitutes a feeling of "due-ness" instead.

### Independence: when the past has zero grip on the next outcome

The second building block is *independence.* Two events are independent when the outcome of one tells you nothing about the outcome of the other. Each coin flip is independent of the last: the coin has no memory, no ledger, no sense that it "owes" you a tails. Each roulette spin is independent: the wheel does not know it just produced five reds. Dice are independent. A slot machine is independent.

The opposite is a *dependent* event, where the past genuinely changes the odds of the future. Drawing cards from a deck *without* reshuffling is the classic example: every ace you remove makes the next card less likely to be an ace, because there are physically fewer aces left. The chart below sorts the everyday cases into the two columns, because this single distinction decides whether a "due" bet is ever legitimate.

![A matrix showing that a 'due' bet is legitimate only for dependent events — cards without a reshuffle — while coins, roulette, dice, and daily price moves are effectively independent and never 'due'.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-2.webp)

The Cards column is the only one where "due" means something. Remove four aces from a 52-card deck and the next card genuinely cannot be an ace — the deck *remembers*, so counting what is left is a real, exploitable edge (this is exactly what card counters do). Every other column is independent or so close to it that treating it as independent is the safe assumption. A coin is never due. A roulette wheel is never due. And a daily price move — despite what your gut screams after five green days — is *barely* dependent at all, weakly enough that "it's due to reverse" is almost always the wrong frame. Hold that word "barely"; we will return to the small, real exceptions later, because pretending price is perfectly independent is its own kind of lie.

#### Worked example: the deck that remembers versus the coin that doesn't

Suppose you are dealt cards from a single 52-card deck, and the first four cards are all aces. What is the probability the fifth card is an ace? There were four aces; all four are gone; so the answer is exactly **0**. The past changed the odds completely — this is a dependent event, and "an ace is not due, it is impossible" is a correct, tradeable inference.

Now suppose instead you flip a fair coin and it lands heads four times. What is the probability the fifth flip is heads? Still **one-half.** The coin has no aces to run out of. If you bet \$100 that the fifth flip "must" be tails because four heads is "too many," you are making a bet with the same expected value as betting on any single flip: zero edge, minus whatever the house charges. **The lesson:** ask whether the past physically removes possibilities from the future — if it does, "due" is real; if it does not, "due" is a feeling with no cash value.

### Randomness is clumpy, not smooth

The third building block is the one people find hardest to accept: *random does not mean evenly spread.* Ask someone to write down a "random" sequence of coin flips and they will produce something that alternates neatly — heads, tails, heads, heads, tails, tails, heads — carefully avoiding any long run because a long run "doesn't look random." But real randomness is lumpy. It produces streaks, clusters, and runs that feel designed. A truly random 18-flip sequence will very often contain a run of five or six identical results; the human-invented "random" sequence almost never does. We will make this precise in a moment with the mathematics of runs.

### The two fallacies, defined precisely

With those three pieces in hand, the two superstitions are easy to state exactly, and they are mirror images:

- The **gambler's fallacy** is the belief that a streak makes the *opposite* outcome more likely — that independent events "self-correct" so the long-run average is restored soon. After five reds, black is "due." Psychologists trace this to what Amos Tversky and Daniel Kahneman named the ["belief in the law of small numbers"](https://www2.psych.ubc.ca/~schaller/Psyc590Readings/TverskyKahneman1971.pdf) (1971): we wrongly expect *short* samples to look like the long-run average, so a short run of one outcome feels like it must be paid back immediately.
- The **hot hand** is the belief that a streak makes the *same* outcome more likely — that whoever or whatever is "hot" will stay hot. After three makes, the shooter is on fire; after three bounces, the level will hold again.

They cannot both be right about a random series, and for a truly independent series *both are wrong.* The trader's special genius is to run them simultaneously, applying whichever one flatters the position they already want to take. The rest of the article takes each apart, prices the damage, and hands you a drill.

| | Gambler's fallacy | Hot hand |
|---|---|---|
| **Belief** | The streak will reverse ("due to end") | The streak will continue ("due to keep going") |
| **Bet it drives** | Fade the move; buy the dip / sell the rip | Chase the move; add to the runner |
| **Underlying error** | Expecting short samples to self-correct | Reading noise as a persistent state ("heat") |
| **When it's actually right** | Dependent events (cards) or genuine mean-reversion | Genuine positive autocorrelation (weak momentum) |
| **The fix** | Check independence and the base rate | Check independence and the base rate |

## The gambler's fallacy: the "due" delusion

Let us dissect the gambler's fallacy first, because it is the cleaner error. Its whole engine is a single false step, and once you can name that step you can catch yourself in the act. The diagram below lays out the five-step chain your mind runs after a streak of one outcome.

![A five-step pipeline of the gambler's fallacy: five reds in a row, invoke the 'law of averages', black feels 'due', size up on black, and the reality that the odds are still 18/37 so the bet is negative expected value.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-3.webp)

You observe five reds. You recall a half-remembered idea called the "law of averages" — the true fact that over a *huge* number of spins, reds and blacks converge to roughly equal counts. Then comes the false step, the one that does all the damage: you read that long-run tendency as a *force acting on the very next spin*, as if the wheel were keeping a running tally and now owes you a black to balance the books. Black "feels due." You size up on black. And reality is indifferent: the odds of black on that next spin are exactly what they always were, ${18/37}$, so your bet carries the house's negative expected value and nothing more. The real law of averages works by *swamping* early results with later ones, not by *reversing* them. If you flip 5 heads and then flip 10,000 more times, you do not get a compensating run of tails — you get roughly 5,000 more of each, and the early surplus of heads becomes a rounding error against the huge denominator. Balance is restored by dilution, never by correction.

> The wheel has no memory, and neither does the market's next tick. The only thing keeping score is you — and that scorekeeping is the entire problem.

#### Worked example: the Monte Carlo wheel and the bet that never gets better

Return to that 1913 table. A single-zero wheel lands black with probability ${18/37}$ ≈ 0.486 on every spin, independently. The chance of black coming up 26 times in a row is that number multiplied by itself 26 times:

$$P(\text{26 blacks}) = \left(\tfrac{18}{37}\right)^{26} \approx 7.3 \times 10^{-9}$$

That is about **1 in 137 million** for that specific colour; counting either colour running 26 straight, it is 2 × that, roughly **1 in 68 million** — a genuinely freakish event, which is exactly why the story survives. But here is the point the crowd missed. Standing at spin number 25, having just watched 25 blacks, what was the probability the next spin would be black? Still ${18/37}$. Not lower. The 25 blacks that already happened were *in the past*; they had already "used up" their improbability. The wheel was not sitting on a 1-in-68-million debt that it had to repay with a red. Each new spin started the count fresh.

So the gamblers piling onto red were not getting a better and better bet as the streak grew; they were getting the *same* mildly-losing bet, over and over, while telling themselves it improved each time. If ten of them bet \$1,000 on red at spin 20, they collectively expected to lose the house edge on \$10,000 — no matter how many blacks preceded it. **The lesson:** a long streak of an independent event does not make the reversal "due"; it just makes a rare thing that already happened, and the next bet is exactly as good, or bad, as the first.

## Why random looks streaky: the clustering illusion

The gambler's fallacy feeds on a deeper misperception: we do not know what randomness is supposed to look like, so when we see its natural clumpiness we mistake it for a pattern that demands a bet. This is the *clustering illusion*, and it is worth seeing with your own eyes before we do the math.

![Two rows of coin flips: a genuinely random 18-flip run that contains a natural six-long streak of heads, versus the too-tidy alternating sequence people invent when asked to produce 'random', whose longest run is only two.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-4.webp)

The top row is a genuinely random run of 18 flips. It contains a streak of six heads in a row — the cells outlined in bold — and your eye immediately wants to call that a "hot streak," a signal, something. It is not. It is what randomness *does.* The bottom row is what people produce when you ask them to write down a "random-looking" sequence by hand: it alternates almost every flip, with a longest run of just two, because a longer run feels wrong. The irony is total — the tidy sequence is the *unnatural* one. A forensic accountant can spot faked data precisely because humans under-produce streaks when they fabricate randomness. Your pattern-hunting brain sees the six-long run at the top and screams "pattern"; it sees the neat alternation at the bottom and feels calm. It has the two exactly backwards.

### The mathematics of runs: how long a streak should you expect?

We can make "streaks are normal" precise with two facts from the study of *runs* — a run being a maximal stretch of the same outcome. First, the expected number of runs. In a fair sequence of `N` flips, the expected number of runs is about `N/2 + 1`; in 100 flips you should expect roughly **51 distinct runs**. Second, and more useful for a trader eyeballing a chart, the expected *length* of the longest run grows only with the logarithm of the sample size — on the order of $\log_2(N)$. That logarithm is a slow, punishing function, and the chart below shows what it implies.

![A bar chart showing the expected longest streak growing only with the logarithm of the number of flips: about 3.3 for 10 flips, 6.6 for 100 flips, and 10 for 1000 flips, so single-digit streaks are routine.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-5.webp)

Because the expected longest run scales like $\log_2(N)$, doubling your data barely lengthens the streaks you should expect. Ten flips: expect a longest run around 3. One hundred flips: about 6 or 7. A thousand flips: about 10. So if you have watched a stock for a hundred days and it strung together six up-days in a row, you have witnessed *exactly the streak length randomness predicts* — no momentum, no regime, no signal required. The trader who sizes up because "six green days must mean something" is betting against arithmetic. (A small honesty note: $\log_2(N)$ is the leading-order rule of thumb; the exact expected longest run sits a touch higher for small `N`, but "about six or seven for a hundred flips" is right.)

#### Worked example: the streak that looks like skill

Suppose you follow 20 traders on social media, each posting the results of a coin-flip-fair strategy — 50/50, no real edge — over 100 trades each. How many of them will show an impressive winning streak you would be tempted to copy? With 100 trades apiece, each account is highly likely to contain a run of six or seven consecutive wins somewhere in its history, purely by chance. Across 20 accounts, you are nearly guaranteed to see several with a shiny six- or seven-long streak, and at least one is likely to sport a run of eight or nine. That "hottest" account looks like a genius. It is a coin that landed heads a few extra times in a row in front of an audience.

If you allocate \$5,000 to the trader with the flashiest streak, you are selecting on noise — picking the account whose random run happened to be longest — and the streak carries zero information about the next 100 trades. **The lesson:** in a big enough crowd of random processes, spectacular streaks are not just possible, they are guaranteed, and the one that impresses you most is usually the luckiest, not the best.

## The hot hand: the mirror-image error — and a twist

Now the other superstition. The hot hand is the belief that success breeds success — that a shooter who has hit three in a row, or a level that has held three times, is in a special "hot" state that makes the next success more likely. For decades, the scientific consensus was that this belief is simply the clustering illusion wearing a jersey: you see the natural six-long run, you call it "heat," and you are fooled. But the hot hand has a genuinely surprising twist, and telling this story fairly is where most popular accounts go wrong.

![A before-and-after comparison: Gilovich-Vallone-Tversky 1985 measured the shot-after-a-streak, found it near the base rate, and declared the hot hand an illusion; Miller-Sanjurjo 2018 showed a finite-sample selection bias made the streak-shot read artificially low, so correcting it reveals a real hot hand.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-6.webp)

The left column is the classic result. In 1985, Thomas Gilovich, Robert Vallone, and Amos Tversky published ["The Hot Hand in Basketball: On the Misperception of Random Sequences"](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/7782/readings/gilovich%20vallone%20tversky.pdf) in *Cognitive Psychology*. They took field-goal data from the Philadelphia 76ers, free-throw data from the Boston Celtics, and a controlled 100-shots-per-player experiment on Cornell varsity players, and they measured the probability of a hit *after* a streak of hits. It came out no higher than the player's overall base rate. Their verdict: the hot hand is "a widely shared cognitive illusion." Celtics coach Red Auerbach's reply is immortal — "Who is this guy? So he makes a study. I couldn't care less." For thirty years, the professor won and the coach lost.

The right column is the twist, and it is beautiful. In 2018, Joshua Miller and Adam Sanjurjo published ["Surprised by the Hot Hand Fallacy? A Truth in the Law of Small Numbers"](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA14943) in *Econometrica*, and they found a subtle statistical bias hiding inside the 1985 method itself. When you go looking, in a finite sequence, for "the shots that came right after a streak of hits," the very act of selecting those shots biases their measured success rate *downward*. The measure GVT trusted was rigged against the hot hand before any data arrived. Correcting the bias flips the conclusion: in the controlled shooting data, there really is a hot hand, on the order of several percentage points of shooting boost after a streak. Not the mythical "unconscious, can't-miss" version the fans believe — but not zero either.

Both things are true, and a good trader holds both: **streaks fool us constantly, and yet streaks can carry a sliver of real signal.** The task is never "believe in the hot hand" or "disbelieve in it"; it is to measure, correcting for the traps, and to size to the small real effect rather than the large mythical one.

### The coin-flip bias that fooled the professors

The Miller-Sanjurjo bias is so counterintuitive that it is worth doing by hand, with a coin, because the same selection trap infects how you read your own trade log.

#### Worked example: why "the flip after a heads" is not a fair coin

Flip a fair coin four times and write down the sequence. Now compute one number: among the flips that came *immediately after a heads*, what fraction were also heads? For the sequence H-H-T-H, the flips-after-a-heads are position 2 (H, a hit) and position 4 — wait, position 4 follows a T, so it does not count; only positions following an H count. Do this carefully for the sequence and you get a proportion. Here is the shock: if you average that proportion over all 16 equally likely four-flip sequences (using the 14 in which the question is even defined), the average is **not 0.5**. It is about **0.40.**

$$\mathbb{E}\!\left[\,\widehat{P}(H \mid \text{prev }H)\,\right] \approx 0.40 \quad\text{a downward bias of about 10 percentage points}$$

The coin is perfectly fair — every flip is 50/50 — yet the *measured* "probability of heads after a heads" averages well below 50% in short sequences. Why? Because selecting "the flip after a heads" is subtly like removing one heads from the pool of what you can still observe, which tilts the remaining sample toward tails. It is the same finite-sample selection effect that biased the 1985 basketball study. **The lesson:** when you eyeball "how do I do right after a win?" in a short trade history, the naive number is biased *against* finding persistence — so both "I have no hot streak" and "I clearly do" can be artifacts of the measurement, and only bias-corrected counting settles it.

## The trader's contradiction: holding both fallacies at once

Here is the crux of the whole piece, the thing that makes this pair of biases so much more dangerous for traders than for casino gamblers. A roulette player at least commits to *one* superstition — they believe reds are due, and they bet red. A trader flips between the two fallacies fluidly, unconsciously, choosing whichever one licenses the trade they already wanted. Watch how it plays out across a single afternoon:

- A stock is up five days straight. You are flat and itching to short. Which fallacy do you reach for? The gambler's: *"It's up five days, it's overextended, it's due for a pullback."* You short it.
- The same stock, same session, is now testing a support level it has bounced off three times this month. You are long and nervous. Which fallacy now? The hot hand: *"It's held three times, this level is strong, it'll bounce again."* You hold.

Both trades are justified by the *shape of the recent past.* One says the past will reverse; the other says the past will repeat. The only thing selecting between them is your existing position and your emotional need in the moment. That is the fingerprint of a rationalization, not an analysis. If a streak can be spun to support both "it will continue" and "it will end" depending on which you need, then the streak is telling you nothing — you are reading the tea leaves and calling it a read. This is why the pair matters more than either bias alone: together they form a machine for justifying any trade, which is the same as having no discipline at all. (It pairs naturally with the trap of judging a decision by its result rather than its logic — see [process versus outcome](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting).)

The escape is not to pick the "correct" fallacy. It is to notice that both are stories about a streak, and to replace the story with the base-rate question every single time.

## Sizing on a fake streak: what the hot hand costs your account

The gambler's fallacy usually costs you through bad *entries* — fading things that keep running, buying things that keep falling. The hot hand costs you somewhere more insidious: through *position size.* When you feel hot, you bet bigger. And because the "heat" is mostly random, the bigger bets land at random moments — including, cruelly, right as the streak breaks. Let us price this precisely.

#### Worked example: flat sizing versus streak-chasing on the identical trades

Take a strategy that is a coin flip: 50% win rate, wins and losses the same size, no edge and no anti-edge. Run one fixed sequence of 20 trades that happens to contain 11 wins and 9 losses, with a couple of natural winning streaks in it. Now size it two ways from a \$10,000 start:

- **Flat sizing:** risk 5% of current equity on every trade, win or lose, streak or no streak.
- **Streak-chasing:** risk 5% normally, but after two wins in a row — when you feel hot — jump to 20% on the next trade.

The chart below plots both equity curves on the *exact same* trade outcomes.

![Two equity curves on the identical 20-trade sequence: flat 5% sizing drifts up to 10,779 dollars, while streak-chasing that jumps to 20% after two wins spikes to 15,923 dollars and then collapses to 9,609 dollars as the big bets land right when the streaks break.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-7.webp)

Same wins, same losses, same order. The flat-sized account ends at **\$10,779**, up about 7.8%. The streak-chasing account rockets to **\$15,923** at its peak — during a hot run it looks like a genius, and this is exactly when the streak-chaser feels vindicated and doubles down on the method — and then gives it all back and more, finishing at **\$9,609**, *down* about 3.9%. Nothing about the trades changed. The only difference is that the streak-chaser's biggest bets landed just as the winning runs ended, because runs end at random and a bet placed *because* of a run is a bet placed with terrible timing on average.

There is a deeper reason baked into the arithmetic: compounding punishes volatility. Two accounts with the same average return but different bet sizes do not finish in the same place — the one that swings harder compounds more slowly, because a 20% loss requires a 25% gain to recover. Chasing streaks manufactures exactly the extra volatility that this "volatility drag" then taxes. **The lesson:** sizing up on a hot streak does not raise your expected return in a fair game — it just widens your swings and lets the inevitable reversal do maximum damage, so a random streak becomes a real drawdown.

## The "due for mean reversion" trap — and the honest exception

The most respectable-sounding version of the gambler's fallacy in markets wears the costume of "mean reversion." A trader watches an index climb for eight sessions and declares it "extended, due to revert." Sometimes they are right, and that reinforces the habit. But most of the time "due to revert" is the gambler's fallacy in a suit — the belief that a run of ups mechanically owes you a down.

Here we must be scrupulously honest, because the whole article would be a lie if it claimed price is a perfect coin. It is not. Decades of research find *weak, unstable* structure in returns. Over medium horizons of three to twelve months, there is documented positive autocorrelation — the *momentum* effect that Narasimhan Jegadeesh and Sheridan Titman established in 1993, where past winners keep winning for a while. Over very short horizons of days to a couple of weeks, there is documented *negative* autocorrelation — short-term reversal, the tendency of sharp moves to partially retrace. So genuine mean-reversion and genuine momentum both exist. The problem is that they are weak, they flip sign depending on horizon, and they drown in noise at the timescale a retail trader is usually staring at. The person shorting the eighth green day is not measuring the documented reversal effect with a tested horizon and a risk model — they are pattern-matching a streak and calling it analysis.

#### Worked example: the cost of fading a run on a hunch

Suppose an index has risen eight days straight and you decide it is "due," so you short \$10,000 of it with a mental stop 3% away and a target 3% away — a symmetric bet. If daily direction is essentially a coin flip at your horizon (the honest baseline for a broad index over a day), your short has roughly a 50% chance of hitting either the stop or the target, so before costs your expected profit is about **zero.** Now subtract reality: the bid-ask spread, the borrow cost to short, and the slippage on your stop. Say those total 0.4% round-trip, or \$40 on the \$10,000. Your expected value on the trade is now roughly **−\$40** — small, but you will make this bet hundreds of times, and hundreds of −\$40 expectations compound into a real hole. Worse, if there is any residual *momentum* at your horizon, the ninth day is marginally more likely to be green than red, tilting the fade from "slightly negative" to "reliably negative." **The lesson:** "it's due to revert" is only tradeable when you have measured a real reversion effect at a specific horizon with an edge that clears costs — otherwise fading a streak is paying the house to bet against arithmetic.

This is the close cousin of letting the most recent, most vivid move dominate your read — the [recency and availability trap](/blog/trading/trading-psychology/recency-availability-and-the-tyranny-of-the-last-trade), where the last few candles crowd out the base rate entirely.

## What it looks like at the screen

Biases are easy to nod along with in an article and impossible to feel in the moment, so here is the real-time phenomenology — the actual thoughts and micro-behaviours that mean one of these two fallacies has its hands on your mouse. Learn to hear these sentences in your own inner monologue; they are the tells.

- **You catch yourself using the words "due," "overdue," "has to," or "can't keep going."** These are gambler's-fallacy words. "It can't go up another day." "A red candle is overdue." The moment you hear "due," you are asserting that an independent-ish process owes you something. Say the base-rate question out loud instead.
- **You feel the specific itch to fade a strong trend** — a rising urge to short something precisely *because* it has gone up a lot, with no level, no catalyst, just the length of the run as your thesis. That itch is the fallacy, not a signal.
- **After a few winning trades, the size box starts creeping up on its own.** You type 200 shares instead of your usual 100 and feel that it is justified because you are "in a groove" and "seeing it well today." That is the hot hand reaching into your position sizing. The setup did not get better; your recent results did.
- **A level "feels strong" only because it has held before,** and you find yourself leaning harder on it each time it holds — bigger size, tighter stop, more conviction — even though each successful hold statistically weakens the wall a little by drawing in more resting orders. Three bounces feel like proof; they are three data points.
- **You feel unusually calm looking at a tidy, alternating chart and unusually excited by a streaky one.** That emotional asymmetry is the clustering illusion operating live: your gut is miscalibrated about what randomness looks like, rewarding you for spotting "patterns" that are noise.
- **Your justification for the current trade contradicts your justification from an hour ago.** You were fading strength at 10 a.m. and chasing it at 11. If you can catch the contradiction, you can catch the rationalization — that is the single most useful real-time check in this whole piece.

The common thread: every one of these tells is a *feeling of certainty produced by a streak.* Treat that feeling itself as the alarm. Certainty that arrives *because* of a run is exactly the signal to stop and run the base-rate question.

## Common misconceptions

**"The law of averages guarantees a correction is coming."** The law of large numbers says the *proportion* converges to the base rate over a huge number of trials; it says nothing about any single upcoming trial, and it works by dilution, not reversal. There is no cosmic ledger forcing a red after a run of blacks. A confident "we're due" is the gambler's fallacy verbatim.

**"A long streak is strong evidence the process is rigged or trending."** Sometimes, but far less often than it feels. Because the expected longest run grows like $\log_2(N)$, streaks of six or seven in a hundred observations are the baseline expectation for a fair process. You need a streak dramatically longer than $\log_2(N)$, or a formal runs test, before "the process is biased" beats "this is normal clumpiness."

**"The hot hand was debunked, so believing in streaks is always dumb."** This is the overcorrection, and it is now wrong. Miller and Sanjurjo (2018) showed the famous debunking rested on a biased statistic; corrected, a small real hot hand appears in controlled data. The mature position is neither faith nor scorn — it is "measure it carefully, size to the small real effect, and never to the mythical one."

**"Mean reversion is a law, so fading extended moves is smart."** Mean reversion is a *weak, horizon-dependent tendency*, not a law, and it coexists with momentum that pushes the other way. Fading a run without a tested edge and a cost model is just the gambler's fallacy with better vocabulary. The move being "extended" is not, by itself, a trade.

**"If I've been winning, I'm hot and should press."** Your recent wins are the most likely thing in the world to be a random run — see the coin-streak example — and pressing size on them is precisely how a fair game turns into a drawdown via volatility drag. Feeling hot is a reason to *check* your process, not to *lever* it.

**"Independent and dependent are obvious — I'd never confuse them."** Yet almost everyone treats "this level has held three times" (weakly dependent at best) like "three aces are gone" (strictly dependent), and treats a five-day rally (near-independent) like a stretched rubber band (mechanically reverting). The confusion is the default, not the exception.

## How it shows up in real markets

### 1. Monte Carlo, 1913, and the martingale that eats accounts

The 1913 run is more than a colourful anecdote; it is the graveyard of the *martingale* — the "double your bet after every loss" system that the gambler's fallacy makes feel invincible. The logic: keep doubling on red, and the first red recovers everything plus a unit. Suppose you start at \$10 and double after each of 26 losing blacks: your 26th bet would need to be roughly \$10 × 2²⁵ ≈ **\$335 million**, and you would have staked over half a billion in total, to try to win back \$10. No bankroll and no table limit survives a run that randomness readily produces. Every "add to the loser, it has to turn" averaging-down blowup in markets — from rogue traders to retail accounts martingaling a downtrend — is the same 1913 bet in a different costume ([source: the naming episode](https://en.wikipedia.org/wiki/Gambler%27s_fallacy)).

### 2. The permabears who faded the recovery

After a crash, the gambler's-fallacy-in-a-suit sounds like caution: "the market has bounced too far, too fast off the low — it's due to roll over and retest." Traders who shorted every leg up of a post-crash recovery on the theory that the rally was "overdue to fail" have, in cycle after cycle, been ground down as the index kept climbing. The rally being long was never a reason for it to end; a streak of up-months is exactly what a recovering market produces, and "due to reverse" fought the base rate the whole way up. The precise dates and magnitudes vary by cycle, but the mechanism is invariant: fading a run because it is a run.

### 3. Misreading a short track record as skill

A fund or a trader posts three great years and attracts a wave of capital. But three years is a tiny sample. As the coin-streak example showed, in a large population of managers running strategies with little or no true edge, several will inevitably string together a few winning years by chance, and those are precisely the ones marketing selects and money chases. The academic literature bears this out: some short-term performance persistence exists (Hendricks, Patel and Zeckhauser documented it in 1993), but Mark Carhart's 1997 study *On Persistence in Mutual Fund Performance* showed that most of it is explained by momentum exposure and cost differences rather than durable skill. Chasing last year's hottest fund is the hot-hand fallacy with a prospectus.

### 4. The hot hand's redemption in the shooting data

The other side deserves its due. When Miller and Sanjurjo corrected the selection bias in the controlled Cornell-style shooting experiments, a real hot hand emerged — a meaningful boost in shooting percentage following a streak of makes, larger than the noise. The market analogue is that genuine, weak persistence does exist in some series (the momentum effect), and disciplined momentum strategies harvest it. The lesson is not "streaks are always fake"; it is that the real effect is small and buried, so you must measure it with bias-corrected tools and size to what you can actually prove — never to the vivid story your gut tells about "heat."

### 5. The V-2 rockets over London

The cleanest demonstration that random clusters fool experts comes from wartime London. As German flying bombs fell on the south of the city, residents were certain the hits were *clustered* — that some neighbourhoods were being targeted and others spared. In 1946 the actuary R. D. Clarke tested it. He divided the area into [576 squares of a quarter square kilometre each](https://www.jstor.org/stable/41139158), counted the 537 impacts, and compared the distribution to a Poisson process — the signature of pure randomness. The fit was almost perfect. The "clusters" everyone saw were the clumpiness that randomness always produces; the bombs were, statistically, falling at random. Every trader who sees "clusters" of green days or bounces and infers a targeting hand is making the exact error of those Londoners, and the exact error is expensive.

## The drill: the shuffle test and the pre-bet question

Diagnosis is worthless without a procedure. Here is a concrete, repeatable one — a lab test for "is this a pattern or is this noise?" plus a two-question gate to run before any bet that a streak has tempted you into.

### The shuffle test

The core trick exploits the one thing that separates real structure from clustering illusion: real structure lives in the *order* of events, and noise does not. So destroy the order and see what survives.

![A flow of the shuffle test: take your equity curve or a price series, shuffle the order to break the time link, re-plot the scrambled series, and check whether the pattern survives — if it does it is real structure worth trading, if it vanishes it was noise.](/imgs/blogs/the-gamblers-fallacy-and-the-hot-hand-8.webp)

Run it like this:

1. **Take the series you think has a pattern** — your own trade-by-trade P&L, or a stretch of daily returns you believe is "trending" or "mean-reverting."
2. **Shuffle the order randomly**, many times. Randomly reordering the returns keeps every value identical — same wins, same losses, same mean and variance — but destroys any real time-dependence, any genuine momentum or reversion.
3. **Re-plot and re-measure.** Compare the "pattern" in your real, ordered series to what shows up across the shuffled versions.
4. **Read the verdict.** If the streaks, the "momentum," the equity-curve smoothness look just as strong in the shuffled versions, then the order carried no information — the pattern was clustering illusion, and there is nothing to trade. If the real ordering stands out sharply from the shuffled crowd (e.g., far longer runs, far stronger autocorrelation than any scramble produces), you may have real structure worth testing further.

Traders formalize this with a *runs test* (comparing your observed number and length of runs to what a random sequence would give) and with *bootstrap* or *permutation* tests on strategy returns. But you do not need the formal machinery to get the discipline: the mental habit of asking "would this pattern survive a shuffle?" is enough to defuse most streak-driven urges on the spot.

### The two-question pre-bet gate

Before you place any trade that a streak talked you into — a fade because something is "due," an add because something is "hot" — stop and answer two questions out loud:

1. **Are these events actually independent?** If the past does not physically change the odds of the next outcome (a coin, a roulette wheel, a single day's index direction), then nothing is ever "due" and no streak is self-sustaining. Only if there is a real mechanism — order-flow, a fundamental catalyst, a documented and *measured* autocorrelation — does the past legitimately inform the next bet.
2. **What is the base rate?** Not "what does this streak make me feel," but "across hundreds of similar situations, what actually happened next?" If you do not know the base rate, you are not making a probabilistic bet; you are telling yourself a story about a streak.

If the honest answers are "independent" and "I don't have a base rate," the correct action is to stand down and size by your normal rules, unaffected by the run. This gate, run every time, is what keeps you on the green path of the very first diagram. It sits naturally alongside a broader inventory of these traps in [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders).

## When this matters to you

Both fallacies converge on the same practical failure: they let the *shape of the recent past* set the *size and direction of your next bet*, on a series where the past has little or no grip on the future. That is a description of nearly every impulsive trade a discretionary trader ever makes. The fix is not to become a robot who denies that any streak ever means anything — the hot-hand story teaches that some streaks carry a real sliver of signal — but to route every streak through the same two questions before it touches your risk. Independence and base rate. Every time.

The reason this pays is compounding, in both directions. On the cost side, streak-chasing manufactures volatility that the geometry of returns then taxes, turning a break-even process into a losing one purely through sizing. On the benefit side, the trader who refuses to let random runs move their size keeps their variance low, survives the inevitable cold stretches, and lets a genuine edge — if they have one — compound quietly. You do not have to predict randomness. You only have to stop paying it a premium for stories it never promised to honour.

A closing caution, because this is educational and not advice: nothing here says "fade this" or "chase that." It says the *length of a run* is, by itself, almost never a reason to do either — and that the discipline of proving otherwise, with a shuffle test and a base rate, is the difference between trading a process and betting on a superstition.

## Sources & further reading

- Amos Tversky and Daniel Kahneman, ["Belief in the Law of Small Numbers,"](https://www2.psych.ubc.ca/~schaller/Psyc590Readings/TverskyKahneman1971.pdf) *Psychological Bulletin* 76(2), 1971 — the cognitive root of the gambler's fallacy.
- Thomas Gilovich, Robert Vallone, and Amos Tversky, ["The Hot Hand in Basketball: On the Misperception of Random Sequences,"](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/7782/readings/gilovich%20vallone%20tversky.pdf) *Cognitive Psychology* 17, 1985, pp. 295–314 — the classic hot-hand study (Philadelphia 76ers, Boston Celtics, Cornell).
- Joshua B. Miller and Adam Sanjurjo, ["Surprised by the Hot Hand Fallacy? A Truth in the Law of Small Numbers,"](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA14943) *Econometrica* 86(6), 2018, pp. 2019–2047 — the selection-bias correction that partly revives the hot hand.
- ["Gambler's fallacy,"](https://en.wikipedia.org/wiki/Gambler%27s_fallacy) Wikipedia — the Monte Carlo 1913 episode and the ~1-in-68-million probability of the 26-black run.
- R. D. Clarke, ["An Application of the Poisson Distribution,"](https://www.jstor.org/stable/41139158) *Journal of the Institute of Actuaries* 72(3), 1946, p. 481 — the London flying-bomb clustering study (576 squares, 537 hits).
- Narasimhan Jegadeesh and Sheridan Titman, "Returns to Buying Winners and Selling Losers," *Journal of Finance*, 1993 — evidence of medium-horizon momentum (real, weak positive autocorrelation).
- Mark M. Carhart, "On Persistence in Mutual Fund Performance," *Journal of Finance* 52(1), 1997 — most apparent fund "hot hands" are momentum and costs, not durable skill.
- Sibling posts on this blog: [Process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting), [Recency, availability, and the tyranny of the last trade](/blog/trading/trading-psychology/recency-availability-and-the-tyranny-of-the-last-trade), and [The cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders).
