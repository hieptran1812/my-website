---
title: "The Pre-Mortem and the Blameless Post-Mortem: The Two Reviews That Bracket Every Trade"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "Before the trade, assume it already failed and ask why; after the trade, review the process without blame and separate luck from skill. Here is the science behind both reviews and the exact routines that turn a P&L into a learning machine."
tags: ["trading-psychology", "pre-mortem", "post-mortem", "prospective-hindsight", "blameless-review", "decision-journal", "process-over-outcome", "risk-management", "gary-klein", "behavioral-finance"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Two cheap reviews, one before the trade and one after, do more for your P&L than any indicator. The pre-mortem sets your risk while you are still calm; the blameless post-mortem teaches you the right lesson once the result is in.
>
> - **The pre-mortem** (Gary Klein): before you enter, assume the trade has *already failed* and ask why. This past-tense framing — *prospective hindsight* — surfaces roughly **30% more** real risks than asking "what could go wrong," and you convert them into a stop and a size.
> - **The blameless post-mortem** (from aviation and site-reliability engineering): after you exit, review the *process*, not the P&L, and strip out blame so you learn instead of defend.
> - The point of the post-mortem is to **separate luck from skill**: keep any sound process whatever it paid, fix any flawed one whatever it paid, and treat the *lucky win* as your single most dangerous outcome.
> - **The number to remember:** a reckless trade that *won* +\$4,000 can carry a −\$3,800 expected value (an F), while a sound trade that *lost* −\$600 can carry a +\$345 expected value (an A). Grade the edge, not the outcome.
> - **The drill:** a 5-minute pre-mortem before every sizeable trade, and a four-question blameless template after — *What was the plan? Was it followed? Luck or skill? What one thing changes?*

You are about to put on a trade you like. The thesis is clean, the chart looks right, and your hand is already moving toward the size that feels good rather than the size you planned. This is the exact moment almost every account is quietly damaged — not by the trade, but by the absence of two five-minute habits that professionals in far more dangerous fields treat as non-negotiable.

Surgeons run a checklist before they cut. Pilots brief the ways a landing can go wrong before they attempt it. Reliability engineers write up every outage without pointing a finger. Traders, by contrast, tend to enter on a feeling and review on a mood — sizing up when euphoric, tearing up the playbook when stung. This article is about importing two disciplines from those safer worlds and bolting them onto every trade you make: a **pre-mortem** before you enter, and a **blameless post-mortem** after you exit. The diagram below is the whole method at a glance; the rest of the piece builds each box from the ground up.

![A pipeline: a trade idea passes through a pre-mortem that sets invalidation and size, then the outcome (luck plus skill), then a blameless post-mortem that grades the process, feeding one change into the next trade.](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-1.webp)

Read it left to right. The pre-mortem sits *before* the money is at risk, where you are still calm and can think about failure without flinching, and its job is to hand you two numbers — where you are wrong (the invalidation) and how much you can lose (the size). The blameless post-mortem sits *after* the outcome, where the temptation is to grade yourself by the dollars; its job is to grade the *decision* instead, so that the thing you carry into the next trade is a lesson about your process and not a superstition about your luck. Neither review needs software, data, or talent. Both need you to override a very strong instinct at a very specific moment. Let us build the tools that make that override automatic.

## Foundations: the two building blocks of a self-correcting trader

You need no finance background for this section. We are going to define, from zero, the handful of ideas that make these two reviews work: what *prospective hindsight* is and why it beats ordinary worrying, what a *blameless* review is and where it came from, and — the idea both reviews depend on — why the result of any single trade is a terrible measure of the decision behind it. This is where the science lives, and it is more settled than most traders realize.

### The pre-mortem is prospective hindsight, and it has a citation

Start with the reframe, because it is the entire trick. Ordinary risk management asks a forward-looking, conditional question: *"What could go wrong with this trade?"* It sounds prudent. In practice it keeps you optimistic. You are still standing at the start of the story, the plan is still alive, and some part of you is quietly defending it — so the risks you list come out vague and hedged, and you find three or four of them before your attention drifts back to the upside.

A **pre-mortem** changes one word and, with it, the whole cognitive stance. Instead of *"what could go wrong,"* you assert that it *did* go wrong: *"It is three months from now. This trade was a disaster. Write the story of how it died."* You are no longer forecasting a possibility; you are *explaining* an event that your mind now treats as a fact. That shift — from an uncertain future to a certain past — is called **prospective hindsight**, a term coined in a 1989 study by Deborah Mitchell (then at Wharton), Jay Russo (Cornell), and Nancy Pennington (University of Colorado), titled "Back to the future: Temporal perspective in the explanation of events."

Their finding is the empirical backbone of this whole practice. Reporting the research in the *Harvard Business Review* in September 2007, the decision scientist **Gary Klein** — who developed the pre-mortem as a business technique — summarized it this way: imagining that an event *has already occurred* increases people's ability to correctly identify reasons for a future outcome by about **30%**. Change the tense, find a third more of the real risks. The figure below shows the two stances side by side.

![A before-and-after comparison: the ordinary risk check ('what could go wrong?') yields three to four vague hedged risks while you quietly defend the plan; the pre-mortem ('it DID fail, why?') yields about 30% more concrete reasons while you hunt for the flaw.](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-2.webp)

There is a second, social reason the pre-mortem works, and it matters even for a solo trader arguing with themselves. The Nobel laureate **Daniel Kahneman** devotes a passage of *Thinking, Fast and Slow* (2011) to Klein's technique, and his verdict is that the pre-mortem *"legitimizes doubts."* Once a group — or a single mind — has committed to a plan, expressing doubt starts to feel like disloyalty, and the doubts get suppressed. By making the disaster the *premise* of the exercise, the pre-mortem gives every objection permission to speak: you are not being negative, you are answering the assignment. Kahneman's suggested wording is worth stealing, lightly paraphrased: *set the scene a year from now — the plan was implemented exactly as it stands, and the outcome was a disaster; take five to ten minutes to write a brief history of that disaster.* That is stress-testing a thesis by assuming it broke, which is exactly the discipline covered in [stress-testing your thesis with a pre-mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem) — here we push it all the way through to sizing.

### A blameless review is a learning system, not a trial

Now the after-the-fact half. A **post-mortem** is any review of a completed trade. A **blameless** post-mortem is one built on a single, hard-won rule from industries where mistakes kill people: you review the *process and the system* that produced the error, and you never indict the person. The Google *Site Reliability Engineering* book — the standard text on running large systems — puts it plainly in its chapter "Postmortem Culture: Learning from Failure" (Chapter 15): a blameless review "assumes that everyone involved in an incident had good intentions and did the right thing with the information they had." The reviewers explicitly note that this culture "originated in the healthcare and avionics industries where mistakes can be fatal."

That lineage is not decoration; it is proof the idea works under the highest possible stakes. Modern aviation's safety culture traces to a specific catastrophe. On December 28, 1978, United Airlines Flight 173, a DC-8 approaching Portland, Oregon, ran out of fuel and crashed into a suburb, killing ten people, while the captain fixated on a landing-gear light and ignored his crew's increasingly nervous hints about the fuel state. The investigation, led by the NTSB, concluded the problem was not a broken part but broken *communication and decision-making* under authority. NASA convened a workshop in 1979 that produced **Crew Resource Management** (CRM) — the training that teaches crews to voice concerns, cross-check each other, and treat the junior officer's doubt as data. Aviation did not respond by firing the captain. It responded by changing the system so the next captain could be corrected in time.

Medicine built the same muscle. Its century-old **Morbidity and Mortality (M&M) conference** reviews bad outcomes in front of the whole department, and the modern framing of **"just culture"** (associated with safety researchers such as Sidney Dekker and David Marx) draws a bright line between honest error — which you learn from — and recklessness — which you address, but which is rare. The point in every case is the same: if the response to a mistake is punishment, people hide mistakes, and hidden mistakes recur. If the response is curiosity, people surface mistakes, and surfaced mistakes get engineered out.

For a trader, "blameless" mostly means blameless toward *yourself*. The instinct after a loss is either to flagellate ("I'm an idiot, this whole approach is garbage") or to deflect ("the market is rigged, the news was a lie"). Both are forms of blame, and both stop you learning — one by attacking the process indiscriminately, the other by attacking everything except the process. The blameless stance asks a colder, more useful question: *given what I knew at the time, what about my system let this happen, and what one thing do I change?*

### Luck, skill, and why one trade cannot be graded

Both reviews rest on one uncomfortable fact, and if you do not accept it, neither review makes sense: **the outcome of any single trade tells you almost nothing about the quality of the decision behind it.** Trading is a game where luck dominates the short run. A good decision routinely loses and a bad one routinely wins, because between your decision and your result sits a slot machine called *variance* — the random scatter of outcomes around their long-run average.

To make this precise we need one number: **expected value**, or **EV**. Expected value is the average result you would get if you could make the same bet thousands of times, computed by weighting each outcome by its probability:

$$\text{EV} = p \times W - (1 - p) \times L$$

where ${p}$ is your probability of winning, ${W}$ is what you win when you win, and ${L}$ is what you lose when you lose. A positive EV (**+EV**) bet makes money on average; a negative EV (**−EV**) bet loses on average. Crucially, EV is a property of the *decision* — fixed the instant you place the trade — and it does not depend on how any single trade turns out.

#### Worked example: a good bet that loses most of the time

Suppose you are offered a trade that risks \$100 on a setup that works only **40%** of the time, but pays **\$300** when it works and costs you your \$100 when it doesn't. Your gut says "40%? Skip it." Compute the EV instead:

$$\text{EV} = 0.40 \times \$300 - 0.60 \times \$100 = \$120 - \$60 = +\$60$$

Every time you take this bet you make \$60 on average — an excellent trade. And yet it *loses 60% of the time*. Six times out of ten you hand over \$100 and feel foolish. If you graded this bet by its results, you would quit it after two or three losses, right before the math paid you back. **The single most important idea underneath both reviews: a +EV decision that loses is still the right decision, and abandoning it is the actual mistake.**

That gap — between a decision's quality and its result — is why you cannot simply look at your last few trades and know whether you are any good. It is why you need a *pre-mortem* to fix your risk before variance can scare you, and a *blameless post-mortem* to grade the decision after variance has muddied the result. It is also the deep link to a companion idea on this blog: [process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting). The pre-mortem and the post-mortem are the two practical tools that make "process over outcome" more than a slogan.

## 1. The pre-mortem: assume the trade is already dead

Here is the reframe in the seat, with the clock running. You have your thesis. Before you touch the size, you say to yourself, out loud if that helps: *"This position is a disaster. I closed it for a loss last week. What killed it?"* Then you write down every cause you can generate, for as long as they keep coming.

The reason this beats a normal "what are the risks" scan is the one the lab found: your brain is far better at *explaining* a concrete event than at *forecasting* an abstract one. Ask a person to predict the future and they hedge; tell them an outcome is fixed and ask them to explain it, and they generate specific, causal, detailed stories. The pre-mortem hijacks that machinery and points it at your own trade. You are not asking your imagination to be pessimistic — a state it resists when you are excited about a setup. You are giving it a fact ("it failed") and a job ("explain it"), which it does eagerly.

The blank page is still the enemy, though. "It failed, why?" can stall if you have no structure. So carry a fixed checklist of the families a trade dies in, and walk each branch. Most losses fall into one of five buckets, shown below.

![A tree rooted at 'how is this trade already DEAD?' branching into five failure families: thesis wrong (breakout is a fakeout), sizing too big (one loss craters you), liquidity (slippage, can't exit), external shock (earnings or macro), and behavioral (you break your own rules).](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-3.webp)

Walk them one at a time and force at least one concrete answer from each:

- **Thesis wrong.** The reason you are in the trade turns out to be false or already priced. The breakout was a fakeout; the "cheap" valuation was cheap for a reason; the catalyst was a rumor. *How, specifically, would you know the thesis was wrong?*
- **Sizing too big.** The trade can be right and still ruin you if a normal adverse move is larger than your account can stomach. Concentration and leverage turn a survivable loss into a fatal one.
- **Execution and liquidity.** You are right about direction and still lose, because the spread is wide, the fill is bad, or — the dangerous one — you cannot get out at your stop because the market gapped through it or the name simply stopped trading at your size.
- **External shock.** An earnings report in six days, a central-bank meeting, a headline, a correlated macro move. Binary events you can look up in advance and that a pre-mortem will not let you ignore.
- **Behavioral.** The most common killer of all: *you* break your own rules — you move the stop, you average down, you hold past the exit hoping. A pre-mortem lets you predict your own predictable misbehavior and pre-commit against it.

Notice what you now have that you did not have five minutes ago: not a vague sense that "this could go against me," but a specific, ranked list of the ways it dies. That list is not the end of the exercise. It is the raw material for the two numbers that actually control your risk.

### What this costs, and when it breaks

The pre-mortem's cost is real: it is five minutes and a small dose of discomfort at the exact moment you would rather be clicking. It also has a failure mode. A pre-mortem is a tool for *sizing and stop-setting*, not a veto — if you let it, it will "find a reason" to skip every trade, which is just fear wearing an analyst's coat. The discipline is to run it, harvest the risks, convert them into a stop and a size, and then *take the trade at that size*. The pre-mortem does not decide whether you trade; it decides how much you can lose when you do.

## 2. Turning the pre-mortem into a stop and a size

A list of fears is useless until it becomes arithmetic. This is the step most traders skip, and it is where the pre-mortem earns its keep: each failure mode you surfaced routes into one of exactly two decisions. Price-structure risks — the ways the chart proves your thesis wrong — set your **invalidation**, the price at which you admit you are wrong and get out. Exposure risks — the ways a single event or a crowded book magnifies a loss — set your **size**, how much you are allowed to lose in the first place. The figure traces the routing.

![A graph: a LONG at 150 on a 100k book fans out to five failure modes; thesis-fails and stop-too-tight feed the invalidation (close below 138), while earnings-gap, correlation, and too-big feed the size (50 shares, 600 dollars max loss).](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-4.webp)

Let us do it with real numbers.

#### Worked example: a pre-mortem that sets a stop and a size

You run a \$100,000 book. Your normal rule caps risk at **1% per trade**, so **\$1,000**. You want to go long a stock at **\$150** because it just broke out. Before sizing, you run the pre-mortem and it hands you five concrete failure modes:

1. **Thesis fails** — the breakout reverses and the stock closes back below the breakout shelf at **\$138**.
2. **Stop too tight** — put your stop just under \$150 and ordinary noise taps you out before the move.
3. **Earnings in six days** — a binary event that could gap the stock straight through any stop.
4. **Correlated** — you already hold four other technology longs, so this is not a new bet, it is *more* of the bet you already have.
5. **Too big** — at your full 1% risk the position would be a meaningful slice of a book already leaning one way.

Before sizing, rank them by rough damage and likelihood — the *damage-times-odds* step — so you know which risk you are actually managing rather than treating all five as equal:

| Failure mode | Damage if it hits | Rough odds | What it sets |
| --- | --- | --- | --- |
| Thesis fails (close below \$138) | −\$600 (the planned stop) | ~35% | Invalidation |
| Stop placed too tight | many small deaths | high, if you misplace it | Invalidation (put it at \$138, not \$149) |
| Earnings gap in 6 days | gap *through* the stop, −\$1,500+ | ~15% over the hold | Size (cut it) |
| Correlated with 4 tech longs | one macro move hits all five at once | ever-present | Size (cut it) |
| You move the stop yourself | turns a −\$600 loss into −\$1,500 | your known weakness | Pre-commit in writing |

The two rows that can cost you *more than the planned stop* — the earnings gap and the correlated-book blow-up — are both in the "size" column, which is the tell that this trade's real danger is exposure, not the entry. That is what a bare "what could go wrong" scan misses and a ranked pre-mortem makes obvious.

Now convert. The first two are price-structure risks, so they set the **invalidation**: a *close below \$138* means the breakout failed. That is \$12 of risk per share (${150 - 138 = 12}$), placed below the noise rather than inside it.

The last three are exposure risks, so they cut the **size**. Because of the earnings event and the correlation with your existing book, you decide this trade does not deserve a full 1%; you halve it to **\$600** of risk. The share count falls straight out of the arithmetic:

$$\text{shares} = \frac{\text{risk budget}}{\text{risk per share}} = \frac{\$600}{\$12} = 50 \text{ shares}$$

So you buy **50 shares at \$150** — a \$7,500 position — with a hard invalidation on a close below \$138, where your loss is ${50 \times \$12 = \$600}$, exactly the risk you chose. **The pre-mortem did not tell you whether to take the trade; it turned five vague fears into a \$138 stop and a 50-share size, so that the worst case is a number you picked on purpose instead of one the market picks for you.**

Everything downstream now behaves. If the stock rips to \$180 you make ${50 \times \$30 = \$1{,}500}$; if the breakout fails you lose the \$600 you signed up for. You have pre-committed to the exit while calm, which is the only reliable defense against the behavioral failure mode — the one where you move the stop in the heat of the moment because you "still believe in it."

## 3. The blameless post-mortem: grade the process, not the P&L

The trade is closed. Here is where the second review does its work, and where the strongest instinct in trading has to be overridden. Your P&L is sitting right there, green or red, and every fiber wants to grade the trade by that number. That is the error called *resulting*, and the blameless post-mortem exists to prevent it.

The move is to review the *process* — the decision you made with the information you had — with blame removed so you can see it clearly. Blame, aimed inward or outward, is not neutral: it changes what you are willing to look at. Blame yourself and you will overhaul a system that was fine on the evidence of one unlucky trade. Blame the market and you will keep a broken process because "it wasn't your fault." The blameless stance refuses both and asks only: was the decision sound given what I knew, and what does the system need? The contrast is the difference between a desk that keeps bleeding from the same wound and one that closes it.

![A before-and-after comparison: in blame culture an error leads to hiding and defending and the same leak recurs; in blameless culture the same error is surfaced and the process studied, so the system changes and the leak is closed.](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-5.webp)

The aviation and medicine lineage is the proof of concept. Consider one more institution born from that lineage, because it is the purest example of blamelessness paying off at scale: the **Aviation Safety Reporting System (ASRS)**, run by NASA since April 1976. Any pilot, controller, or crew member can confidentially report a near-miss, an error, or an unsafe condition, and — this is the engine — doing so grants limited immunity from FAA enforcement for unintentional violations. Because reporting is safe, people report; the system has collected well over a million confidential reports across its life, turning millions of small, embarrassing, individually-forgettable mistakes into systemic fixes that no punitive system would ever have surfaced. A trader's journal is a personal ASRS: a place where you can write down the trade you fumbled *honestly*, because no one is going to punish you for it, which is the only condition under which you will write it down at all.

The link to the rest of your psychology is direct. The blameless post-mortem is the specific countermeasure to [hindsight bias and the story you tell yourself later](/blog/trading/trading-psychology/hindsight-bias-and-the-story-you-tell-yourself-later) — because reviewing against *what you knew at the time* is precisely what stops the "I knew it all along" rewrite from contaminating the lesson.

## 4. Separating luck from skill

The central task of the post-mortem is to pull apart two things the P&L glues together: how good the decision was, and how it happened to turn out. Cross them and you get four cells. Two are easy — a sound process that won, a flawed process that lost — and two are traps. The grid below is the verdict you are trying to reach on every trade, and the rule for acting on it is the whole discipline in one line: **judge the row, act on the row.**

![A two-by-two matrix of process (sound, flawed) against outcome (won, lost): sound-won is 'skill met luck, keep the process,' sound-lost is 'bad beat, keep it, tuition on your edge,' flawed-won is 'most dangerous, lucky, fix it anyway,' flawed-lost is 'deserved loss, fix the process.'](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-6.webp)

Work through the cells:

- **Sound process, won** — skill met luck. Keep doing exactly this. Enjoy it, but do not learn anything new from the win itself; the win confirms nothing that the process did not already justify.
- **Sound process, lost** — a *bad beat*. This is a good trade that got unlucky, and it is your *least* dangerous outcome because it costs money but teaches nothing false. It is tuition on your edge. Keep the process; do not "fix" what wasn't broken.
- **Flawed process, lost** — a deserved loss. The process was bad and the result agreed. Fix the process; at least the feedback pointed the right way.
- **Flawed process, won** — the *most dangerous cell on the board*. You did something reckless and got paid, so every instinct screams "do it again." This is the lucky win, and it is where careers quietly end, because it is the cell that tempts you to institutionalize a mistake. Grade it an F and fix it *anyway*, precisely because the money is lying to you.

The action rule collapses to two moves. **Good process, any outcome → keep it.** **Bad process, any outcome → fix it.** The outcome column is noise; the process row is signal. A trader who strings together a month of sound-process grades is doing the job perfectly, even in a losing month, because over a large enough sample good process and good dollars converge — and the grades show up first. This is the operational core of [process versus outcome](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting): the post-mortem is where you physically sort your trades into these cells and refuse to act on the columns.

> A losing trade you would take again is a success. A winning trade you should never repeat is a failure. The scoreboard just hasn't caught up yet.

## 5. Grading a winner as bad and a loser as good

The grid is abstract until you put money on it, so here are the two trades that live in the dangerous cells, with their expected values computed. The chart shows both: for each trade, the realized P&L (what you saw) and the expected value (the truth), and they disagree in sign.

![A bar chart contrasting two trades against a zero line: a reckless short that realized plus 4000 dollars but has an expected value of minus 3800 (graded F), and a sound long that realized minus 600 dollars but has an expected value of plus 345 (graded A).](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-7.webp)

#### Worked example: the reckless winner that deserves an F

You shorted a stock into its earnings report with no thesis beyond "it's gone up too much," no stop, and three times your normal size. The stock gapped down and you made **+\$4,000**. The P&L is green, the dopamine is real, and every instinct says you are a genius. Grade the *decision* instead. On the information you had, this was a coin flip at best — call it 40% that it drops for your +\$4,000, and 60% that it keeps ripping and, un-stopped and oversized, costs you **−\$9,000**:

$$\text{EV} = 0.40 \times (+\$4{,}000) + 0.60 \times (-\$9{,}000) = +\$1{,}600 - \$5{,}400 = -\$3{,}800$$

This trade *loses \$3,800 every time you take it*, on average. It is one of the worst trades in your book, and it happened to win. In the verdict grid it is flawed-process/won — the most dangerous cell — and it earns a hard **F**. **The win rate was the disguise: a trade can pay you four thousand dollars and still be a decision you must never repeat.**

#### Worked example: the sound loser that deserves an A

Now the trade from Section 2 — the \$150 long, stop at \$138, sized to \$600, with a \$1,500 target. Say the breakout had a 45% chance of following through for a **+\$1,500** win and a 55% chance of failing for your planned **−\$600** loss:

$$\text{EV} = 0.45 \times (+\$1{,}500) + 0.55 \times (-\$600) = +\$675 - \$330 = +\$345$$

That is a solidly positive-expectancy trade. Suppose it hit the stop and you lost **−\$600**. The P&L is red, and the temptation is to conclude the setup is broken and tighten everything. But the decision was sound: a thesis with an edge, a stop below the noise, a size cut for the earnings and correlation risks the pre-mortem surfaced. In the grid it is sound-process/lost — a bad beat — and it earns an **A**. **You lost money and did your job perfectly; the loss is tuition on an edge worth paying for, not evidence to overhaul.**

Put the two side by side and the lesson is unavoidable: the trade that *won* deserved an F and the trade that *lost* deserved an A. If you grade by the P&L, you will repeat the reckless short and abandon the sound long — the two most expensive mistakes available to you — and you will feel disciplined doing it. The blameless post-mortem is the only thing standing between you and that inversion.

## What it looks like at the screen

Theory is tidy. In the seat, skipping these reviews and getting resulted feels like specific bodily events, and if you learn the tells you get a half-second of warning, which is sometimes all you need.

**The tell that you are skipping the pre-mortem** is a warm impatience — the sense that analysis is a formality between you and a trade you have already decided to take. Your hand is on the mouse before the risk is defined. The position size in your head is round and large and arrived without arithmetic. There is a faint irritation at the idea of writing anything down, because writing it down might slow you or, worse, talk you out of it. That irritation is the signal. The trades that most need a pre-mortem are exactly the ones you least want to run one on, because the wanting *is* the euphoria that the pre-mortem is designed to interrupt.

**The tell that you are about to get resulted after a lucky win** is expansive certainty — a specific, glowing conviction that you are "reading the market perfectly right now." You broke a rule and got paid, and behind the pleasure is an urge to *repeat the specific thing you did*: size up again, hold past the target again, skip the stop again. You feel a small reluctance to journal the trade honestly, because some part of you knows the journal will call it what it was. Euphoria plus the impulse to repeat the rule-break is the lucky-win cell announcing itself in real time.

**The tell after an unlucky loss on a good trade** is hot contempt, usually aimed at your own system: *"this setup is garbage, this whole approach doesn't work."* Your hand moves to tighten the next stop by half, or to shrink the next size to a third, or — the quiet one — to simply *not take* the next perfectly valid signal because you "don't trust it right now." Often it curdles into a revenge trade, an oversized unplanned position to "make it back." Contempt plus the impulse to change a process you validated an hour ago is resulting inverted; it is the same bias wearing the mask of discipline. This is the machinery behind [tilt and revenge trading](/blog/trading/trading-psychology/tilt-and-revenge-trading), and the review is what defuses it.

The through-line in all three states is identical: **a feeling that arrived in the last few minutes is trying to overwrite a decision you made — or should have made — with a clear head.** The physical cue is your permission to freeze the keyboard, not to trade on the feeling and not against it either, but to defer to the review, where variance has been separated back out from the decision. The screen is where these biases happen; the pre-mortem and the journal are where they get undone.

## The drill: two routines you can run this week

Everything above is diagnosis. Here is the treatment — two concrete, repeatable routines. Neither takes more than five minutes, and their whole value is that they are *scripted*, so you run them even when your emotional state is screaming at you not to.

### The journal both routines write to

The pre-mortem and the post-mortem are not two separate exercises; they are the two ends of one record. A **decision journal** is just a place — a notebook, a spreadsheet, a note on your phone — where the pre-mortem writes its output *before* the trade and the post-mortem writes its grade *after*. The single rule that makes it work is that the "before" half must be written *before*: a thesis reconstructed after the result is contaminated by hindsight bias and worth nothing.

Four fields go in before you enter, one after you exit. For the worked-example long, the entry reads:

- **Thesis** (why it is +EV): *"Broke out of a three-week base on rising volume; trend intact; measured move to \$180."*
- **Invalidation** (the pre-committed exit): *"Daily close below \$138 — the breakout shelf."*
- **Size and why** (the risk you chose): *"50 shares, \$600 risk = 0.6% of the book; cut from the full 1% because of earnings in 6 days and four correlated tech longs."*
- **Top failure mode** (from the pre-mortem): *"Earnings gap through the stop — accept it, do not add."*
- **Process grade** (written after, before you tally the dollars): *A / B / C / D / F, judged only against what you knew at entry.*

That is ninety seconds of writing, and it is the entire apparatus. Everything below is just the schedule for filling it in.

### Routine 1: the 5-minute pre-mortem before every sizeable trade

Before any trade large enough to matter, run this clock. It is deliberately short so that "I don't have time" is never an excuse; a trade you do not have five minutes to pre-mortem is a trade you are too rushed to take.

![A timeline of the five-minute pre-mortem: at 0:00 assume the trade is dead, 1:00 list every way it died, 2:00 rank by damage times odds, 3:00 set the invalidation, 4:00 size for the worst case, 5:00 commit or pass.](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-9.webp)

Run it minute by minute:

- **Minute 0 — Assume it's dead.** State it as fact: "This trade lost. It was a mistake." Do not soften it into "could."
- **Minute 1 — List every way it died.** Walk the five families (thesis, sizing, liquidity, shock, behavioral) and write a concrete cause under each. Aim for quantity; do not filter yet.
- **Minute 2 — Rank by damage times odds.** For each cause, a rough sense of how bad and how likely. The top one or two are what you are actually managing.
- **Minute 3 — Set the invalidation.** From the price-structure risks, pick the specific price or fact that proves you wrong. This is your stop, decided while calm.
- **Minute 4 — Size for the worst case.** From the exposure risks, pick the risk budget and divide by the per-share risk to get the size. Cut it for events and correlation.
- **Minute 5 — Commit or pass.** Take the trade at that size with that stop, or decline it — but decline only for a *sizing* reason ("too correlated, no room"), never out of raw fear.

### Routine 2: the blameless post-mortem template

After the trade closes — and, if you can bear it, *before* you look at the P&L — answer four questions in order. The order matters: it forces you to reconstruct the decision before the result can color it.

![A serpentine pipeline of the blameless post-mortem: open the review with no blame, then ask what was the plan, was it followed, luck or skill, then change one thing and log it and repeat.](/imgs/blogs/the-pre-mortem-and-the-blameless-post-mortem-8.webp)

1. **What was the plan?** Pull up the pre-mortem you wrote. The thesis, the invalidation, the size, the edge. If you did not write one, that is already your finding: you had no plan, so there is nothing to grade but luck.
2. **Was it followed?** Did you take the size you chose, honor the stop, exit where you said? An unfollowed plan and a bad plan are different diseases with different cures — do not confuse "I broke my rules" with "my rules are wrong."
3. **Luck or skill?** Drop the trade into the verdict grid. Was the process sound on the information you had, independent of the dollars? Name the cell out loud: bad beat, deserved loss, lucky win, earned win.
4. **What one thing changes?** Exactly one. A review that produces a list of ten changes produces zero changes; a review that produces one produces one. If the process was sound, the honest answer is often "nothing — repeat it."

The rule that makes the whole thing work, and the hardest to hold: **an A trade that lost is still an A; an F trade that won is still an F.** You are grading the row of the grid, never the column.

#### Worked example: a blameless review that closes a recurring leak

Here is what the template catches that staring at your P&L never will. You review your last **40 trades** blamelessly, sorting by process rather than result, and a pattern jumps out that had been invisible: the **8 trades** you took in the first five minutes after the open lost a combined **\$3,200** — an average of **−\$400** each — while the other **32 trades** *netted +\$2,600*. Same trader, same strategy; the only difference is the clock.

$$\text{opening trades: } -\$3{,}200 \quad\text{vs.}\quad \text{rest of day: } +\$2{,}600$$

A blame-first review would never surface this, because each opening-trade loss looked like an individual failure of nerve or read, and you would have flogged yourself for each one and changed nothing. The blameless, process-first review sees the *system*: the opening five minutes are a low-liquidity, high-noise environment where your edge does not exist. The one change is not "try harder at the open"; it is a rule — **no new entries in the first five minutes.** **The leak was worth about \$3,200 across those trades, and it was invisible to every review except the one that looked at the process instead of the person.** That is the entire case for blamelessness, in one line item.

## Common misconceptions

**"A pre-mortem is just negative thinking that talks you out of good trades."** No — a pre-mortem that vetoes trades is being run wrong. Its output is a stop and a size, not a yes/no. The discipline is to surface the risks *and then take the trade at a size that survives them.* Used correctly it makes you trade more confidently, because your worst case is a number you chose, not one the market surprises you with.

**"If the trade made money, the process was fine."** This is the single most expensive belief in trading. A profit tells you the trade *won*; it says nothing about whether the decision had an edge. The reckless short that made \$4,000 carried a −\$3,800 expected value. Judging a decision by its profit is resulting, and it is how gamblers get "confirmed" right up until the tail catches them.

**"Blameless means no accountability — you just excuse every mistake."** The opposite. Blameless culture separates honest error (which you learn from without punishment) from genuine recklessness (which is rare and *is* addressed). Removing blame does not remove standards; it removes the *fear* that makes people hide the errors you most need to see. A desk that punishes mistakes gets fewer *reported* mistakes and exactly as many *actual* ones.

**"I don't have time to do this on every trade."** The pre-mortem is five minutes and the post-mortem is four questions. If a trade is too small to warrant five minutes of thought, it is too small to move your account and you can skip the ceremony. If it is large enough to matter and you still can't spare five minutes, you are not trading, you are gambling on a clock.

**"Keeping a journal is just paperwork."** The journal is the physical mechanism that makes both reviews possible. Without a pre-trade record of your thesis, stop, and edge, you have nothing to grade the decision against except the result — which means you have no defense against resulting at all. The paperwork *is* the intervention; skip it and the reviews collapse into vibes.

**"A pre-mortem and a post-mortem are basically the same review at different times."** They are opposites in stance. The pre-mortem is *imaginative and pessimistic* — it invents a failure that hasn't happened to set your risk. The post-mortem is *forensic and neutral* — it examines a decision that has happened to extract a lesson. One sets exposure; the other sets learning. You need both, and neither substitutes for the other.

## How it shows up in real markets

None of this is theoretical. The absence of a pre-mortem and the presence of blame have written some of the most famous disasters in markets and beyond, and the presence of both has quietly prevented thousands you never heard about.

### 1. Long-Term Capital Management: the pre-mortem no one ran

Long-Term Capital Management was, in the mid-1990s, the most intimidating hedge fund on earth — founded in 1994 by John Meriwether, its board studded with talent including the Nobel laureates Myron Scholes and Robert Merton. Its strategy was to find tiny pricing discrepancies and apply enormous leverage, running a balance sheet of roughly **\$125 billion** on an equity base of only about **\$4.7 billion** — leverage on the order of **25-to-1** — atop a derivatives book with a notional value estimated near **\$1.25 trillion**. The models assumed its many positions were not all the same bet. A genuine pre-mortem — *"it is 1998, the fund has failed, why?"* — would have forced the answer that actually killed it: in a crisis, correlations go to one, everything you own falls together, and leverage that looked like 25-to-1 becomes a margin spiral. In August and September 1998, after Russia defaulted, exactly that happened. LTCM lost roughly **\$4.6 billion** in under four months, and the Federal Reserve Bank of New York organized a **\$3.6 billion** recapitalization by a consortium of 14 banks to prevent a wider collapse. The fatal risk was not unknowable; it was un-asked, because the plan was too beloved to assume dead.

### 2. Amaranth Advisors: the sizing a pre-mortem would have vetoed

Amaranth Advisors managed about **\$9.2 billion** at its peak when a single trader, 32-year-old Brian Hunter, put on a massive calendar-spread bet in natural gas futures — long winter contracts, short the near months. When the spread moved against him in September 2006, the position was so large relative to the market that it could not be exited without moving prices further against itself. Amaranth lost in excess of **\$6 billion** — much of it in a single week — and collapsed, one of the largest hedge-fund blow-ups on record. This is the *sizing* branch of the pre-mortem, written in nine figures: the trade could have been directionally reasonable and still been fatal, because the size was larger than the liquidity. A pre-mortem that asked "how would this position kill us even if we're right about direction?" answers itself — *we can't get out* — and that answer sets a size cap that Amaranth never imposed.

### 3. United 173 and the birth of Crew Resource Management

We met this one in the foundations, and it is the cleanest case of a blameless review changing a system rather than blaming a person. When United Airlines Flight 173 ran out of fuel over Portland on December 28, 1978, killing ten, the easy response was to blame the captain who fixated on a gear light while his crew's fuel warnings went unheeded. Aviation did the harder, better thing: it treated the crash as a *systems* failure of communication under authority and built **Crew Resource Management** to fix it — training that makes it normal for a junior officer to challenge a captain and for a captain to be corrected in time. The modern flight deck is a blameless post-mortem culture instantiated as procedure, and it is a large part of why commercial aviation became astonishingly safe. A trading desk that lets its most junior risk voice speak — and that reviews its blow-ups as system failures rather than character flaws — is running the same playbook.

### 4. The Aviation Safety Reporting System: blamelessness at scale

The NASA-run ASRS, live since 1976, is the purest demonstration that removing blame *increases* the information you get. By granting confidentiality and limited immunity, it induced pilots and controllers to voluntarily report their own near-misses and errors — well over a million reports across its history — each one a small hazard that a punitive system would have buried. Aviation's safety record improved not despite people admitting mistakes but *because* a system made admitting them safe. Your decision journal is the personal version: it works only if it is a blameless space where you record the fumble honestly, because the trade you are tempted to hide is precisely the one with the lesson in it.

### 5. Challenger: what a suppressed pre-mortem costs

The Space Shuttle Challenger disaster of January 28, 1986 is, among other things, a story about a pre-mortem that happened and was overruled. The night before launch, engineers at contractor Morton Thiokol — Roger Boisjoly prominent among them — warned that the O-ring seals could fail in the unusually cold forecast temperatures, and initially recommended against launching. Under management pressure, that doubt was not *legitimized*; it was suppressed, and the recommendation was reversed. Seventy-three seconds after liftoff, an O-ring failed and the shuttle broke apart, killing all seven aboard. The lesson Kahneman draws from the pre-mortem — that its value is to *make dissent safe before the decision is locked* — is exactly what a healthy culture the night before might have preserved. In a trading room, the analog is the risk manager whose concern gets waved away because the trade is already "decided." The pre-mortem exists to give that voice a mandate.

### 6. The origin: Klein, Kahneman, and a technique now everywhere

Finally, the quiet success story. Gary Klein introduced the pre-mortem as a business practice, published it in *HBR* in 2007, and Kahneman amplified it in a bestseller — and it has since spread into project management, medicine, the military, and product design precisely because it is cheap and it works. There is no dramatic blow-up to point at here, which is the point: the pre-mortem's wins are the disasters that *did not happen* — the trade sized down before the earnings gap, the position skipped because a pre-mortem surfaced the liquidity trap. Prevention is invisible, and that invisibility is why the discipline is so easy to neglect and so valuable to keep.

## When this matters to you

If you trade at all — or run a business, or make any repeated decision under uncertainty — these two reviews are the highest-leverage habits available to you, because they attack the problem at both ends. The pre-mortem protects you from your *pre-trade* optimism, converting excitement into a stop and a size before variance can hurt you. The blameless post-mortem protects you from your *post-trade* emotion, converting a noisy result into a clean lesson so that you reinforce good process instead of good luck. Between them, they turn a stream of individually-meaningless outcomes into a system that actually compounds skill.

None of this is investment advice, and none of it promises profit — a sound process can and will lose over any short stretch; that is the entire premise. What these routines buy you is not the elimination of losses but the *preservation of good decisions*, so that whatever edge you have survives long enough in a large enough sample to pay you. In a game where luck runs the short term and skill only runs the long one, the trader still standing when the long term arrives is usually the one who spent five minutes assuming the worst before every trade, and four honest questions grading the process after. Start with your next sizeable trade: before you click, assume it already failed, and write down why.

## Sources & further reading

- **Gary Klein, "Performing a Project Premortem," *Harvard Business Review* (September 2007)** — the article that introduced the pre-mortem as a business technique and reported the ~30% figure. [hbr.org](https://hbr.org/2007/09/performing-a-project-premortem).
- **Deborah J. Mitchell, J. Edward Russo & Nancy Pennington, "Back to the future: Temporal perspective in the explanation of events," *Journal of Behavioral Decision Making* 2(1), 25–38 (1989)** — the primary research on prospective hindsight behind the 30% claim. [Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1002/bdm.3960020103).
- **Daniel Kahneman, *Thinking, Fast and Slow* (2011), Ch. 24** — the "premortem legitimizes doubts" discussion and the suggested "a year into the future… the outcome was a disaster" prompt.
- **Betsy Beyer et al. (eds.), *Site Reliability Engineering* (Google / O'Reilly, 2016), Ch. 15, "Postmortem Culture: Learning from Failure"** — the blameless-postmortem standard; notes the practice "originated in the healthcare and avionics industries." [sre.google](https://sre.google/sre-book/postmortem-culture/).
- **NTSB report on United Airlines Flight 173 (Dec 28, 1978)** and the history of **Crew Resource Management** — the crash and the systems response that followed. [Crew resource management (Wikipedia)](https://en.wikipedia.org/wiki/Crew_resource_management).
- **NASA Aviation Safety Reporting System (ASRS)** — confidential, immunity-backed reporting since April 1976. [asrs.arc.nasa.gov](https://asrs.arc.nasa.gov/overview/immunity.html).
- **Long-Term Capital Management** — leverage, the 1998 losses, and the Fed-organized \$3.6bn recapitalization. [Federal Reserve History](https://www.federalreservehistory.org/essays/ltcm-near-failure); [Wikipedia](https://en.wikipedia.org/wiki/Long-Term_Capital_Management).
- **Amaranth Advisors** — the ~\$6bn+ natural-gas collapse of September 2006. [Amaranth Advisors (Wikipedia)](https://en.wikipedia.org/wiki/Amaranth_Advisors).
- **Sidney Dekker, *Just Culture*** — the framework separating honest error from recklessness in incident review.
- Sibling posts on this blog: [Stress-testing your thesis with a pre-mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem), [Process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting), and [Hindsight bias and the story you tell yourself later](/blog/trading/trading-psychology/hindsight-bias-and-the-story-you-tell-yourself-later).
