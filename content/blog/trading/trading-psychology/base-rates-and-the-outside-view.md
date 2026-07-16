---
title: "Base Rates and the Outside View: Escaping Your Own Story"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "Why the vivid story about this one trade beats the boring average in your head, how base-rate neglect leaks money through position size, and a mechanical drill for anchoring every forecast to its reference class before you size."
tags: ["trading-psychology", "base-rates", "outside-view", "reference-class-forecasting", "base-rate-neglect", "planning-fallacy", "kahneman", "tversky", "behavioral-finance", "cognitive-bias", "position-sizing", "probabilistic-thinking"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Your mind estimates the odds of *this* trade by how good its story is, not by how often trades like it actually work. The fix is to look up the boring average first — the base rate of the reference class — and only then adjust for what makes this one special.
>
> - **Base-rate neglect** (Kahneman & Tversky, 1973) is the habit of ignoring how common an outcome is and over-weighting the vivid, specific case in front of you. In the classic taxicab problem, a witness who is 80\% reliable still gives you only a **41\%** chance of being right, because most "blue" calls come from the far larger green fleet.
> - The **inside view** builds a forecast from the details of your particular plan; the **outside view** builds it from the track record of similar plans (Kahneman & Lovallo, 1993). The inside view feels like insight and predicts worse. The outside view feels like a cold shower and predicts better.
> - Daniel Kahneman's own textbook team estimated **2 years** to finish; the reference class said **7 to 10** and that **40\%** of such teams never finish at all. It took them **8 years**, and the book was never used. Even the experts who *knew* the base rate ignored it.
> - Pros do not throw away the story. They **anchor to the base rate first**, then adjust — a small, evidence-led move off the average, not a leap to whatever the story wants.
> - The one number to remember: in this post's sizing example, believing an **80\%** story instead of the **45\%** base rate turns a **5\%** account risk into a **25\%** one — a 5× difference in downside, from the exact same trade.

You are looking at a chart. The setup is beautiful. The breakout is clean, the volume is there, the story writes itself: the company just crushed earnings, the sector is rotating in, the whole thing is coiled like a spring. You *know* this one is going to work. You can feel it. So you size up.

Here is the uncomfortable question this article is about: how did you get from "beautiful chart" to "going to work"? You did not count. You did not ask how many charts that looked exactly this beautiful actually worked out over the last two years. You looked at the specific, vivid, coherent picture in front of you and let its quality stand in for its probability. You confused *how good the story is* with *how likely it is to be true*.

That confusion has a name — **base-rate neglect** — and it is one of the most expensive habits a trader can have, because it does not just make you wrong. It makes you wrong *and* oversized at the same time, which is the only combination that actually blows up accounts.

The cure is a single mental move, and the whole article is about learning to make it automatically. It is the difference between the **inside view** (your detailed story about this one trade) and the **outside view** (the plain historical average of trades like it). The diagram below is the mental model we will spend the rest of the article touring: the same setup, run through two different lenses, producing two different probabilities and — the part that costs money — two very different position sizes.

![Inside view versus outside view: the same trading setup produces two probabilities and two position sizes depending on which lens you look through.](/imgs/blogs/base-rates-and-the-outside-view-1.webp)

On the left, the inside view: one unique setup, a vivid story, a felt probability of 80\%, and a size to match. On the right, the outside view: the same setup seen as one member of a large class of similar setups, its historical base rate, a sober 45\%, and a size scaled to *that*. Same trade. Same chart. The only thing that changed is which lens you looked through — and the lens quietly set the size. Let's build the whole idea from the ground up.

## Foundations: the building blocks

We are going to reason from first principles, so nothing here assumes you have seen any of these terms before. If you already know Bayes' rule cold, skim; if you have never heard the phrase "base rate", this section is built for you and you can proceed from it.

### What a "base rate" actually is

A **base rate** is just how common something is, before you know anything specific about the particular case in front of you. It is the background frequency.

If 1 in 100 charts that look like yours goes on to work, the base rate is 1\%. If 40 out of every 100 do, the base rate is 40\%. That is the whole concept: *how often does this kind of thing happen, on average, across many tries?* The word "base" is doing real work — it is the base, the floor, the starting number you should stand on before the specific details of today's chart start pushing you around.

Base rates are boring. That is precisely why we ignore them. A base rate is an average over a crowd of cases, and it has no story, no drama, no reason it applies to *you*. The vivid particular case, by contrast, is drenched in story. And the mind reaches for the story every single time.

### The reference class: the crowd your case belongs to

To find a base rate, you first have to decide *what crowd your case belongs to*. That crowd is called the **reference class** — the set of past situations similar enough to yours that their track record tells you something about your odds.

Choosing the reference class is the whole game, and it is a judgment call. Is today's trade a member of "all breakouts I've taken"? Or the narrower "breakouts in this sector, on earnings, in an uptrend"? Or the very narrow "this exact setup, which I've traded 40 times"? Narrower classes are more relevant but have fewer examples (noisier averages); broader classes have more data but blur important distinctions. We will come back to how to draw the line. For now, hold the definition: **the reference class is the group of comparable cases; the base rate is how often the outcome you care about happened inside that group.**

### The inside view and the outside view

These two terms, which we owe to the psychologist Daniel Kahneman and the economist Dan Lovallo (in their 1993 paper *Timid Choices and Bold Forecasts*), name the two roads to any forecast.

- The **inside view** builds the estimate from the specifics of *your* case: the details of this plan, this chart, this catalyst, this story. It asks, "given everything I know about this particular situation, how will it unfold?"
- The **outside view** ignores the specifics almost entirely and builds the estimate from the *track record of similar cases*. It asks, "when situations like this one have happened before, how did they turn out on average?"

Here is the pattern that repeats through the whole of this article, and through the whole of behavioural finance: **the inside view feels far more informed and predicts far worse.** It feels better because it uses all your rich, specific knowledge. It predicts worse because that rich, specific knowledge is exactly what makes every case feel unique, novel, and exempt from the averages — and it almost never is.

### Base-rate neglect and its engine, representativeness

**Base-rate neglect** is what happens when the inside view wins: you let the vivid specifics of the case override the boring background frequency. You judge "how likely" by "how well the details fit a picture in my head" instead of "how often this outcome actually occurs".

The engine underneath it is a mental shortcut Kahneman and Tversky called **representativeness**: we judge the probability that something belongs to a category by how much it *resembles* our stereotype of that category, not by how common the category is. A chart that *looks* exactly like your mental image of "a trade that works" gets judged as likely to work — even if trades that look like that only work 40\% of the time. Resemblance is standing in for probability. The two are not the same thing, and the gap between them is where the money leaks.

### A first, simplest worked example — Bayes without the algebra

The cleanest demonstration of base-rate neglect is a puzzle Amos Tversky and Daniel Kahneman published in 1982, and it needs no finance at all. We will count our way through it — no formula required. The figure below is the entire argument in one picture.

![The taxicab problem counted out across one thousand cabs: of every 290 the witness calls blue, only 120 are truly blue, about 41 percent.](/imgs/blogs/base-rates-and-the-outside-view-2.webp)

#### Worked example: the taxicab problem

A city's taxis come in two colors. **85\%** are green; **15\%** are blue — that is the base rate. One night a cab is in a hit-and-run. A witness says it was **blue**. We test the witness under the same lighting and find they correctly identify a cab's color **80\%** of the time (and are wrong 20\%).

The witness said blue. How likely is it that the cab really *was* blue?

Most people answer around 80\% — they anchor entirely on the witness's reliability and forget the base rate completely. Let's count instead. Imagine 1,000 cabs pass under that streetlight:

- **150** are actually blue (15\%). The witness calls 80\% of them "blue" correctly → **120** true "blue" calls.
- **850** are actually green (85\%). The witness wrongly calls 20\% of them "blue" → **170** false "blue" calls.

So across 1,000 cabs, the witness says "blue" a total of ${120 + 170 = 290}$ times. Of those 290 "blue" calls, only 120 are on genuinely blue cabs. The probability the cab was really blue is therefore:

$$P(\text{blue} \mid \text{witness says blue}) = \frac{120}{120 + 170} = \frac{120}{290} \approx 41\%$$

Not 80\%. **41\%.** A witness who is right 80\% of the time gives you a call that is right only 41\% of the time, because blue cabs are so rare that the sheer volume of green cabs mistaken for blue swamps the genuine sightings. The base rate (15\% blue) drags the answer far below the witness's raw reliability.

The one-sentence intuition: **when the thing you are looking for is rare, even a strong, specific signal leaves you closer to the base rate than to certainty — because most of your "hits" are really the common case in disguise.**

Hold that picture, because a trading setup is the taxi and your read of the chart is the witness. A read you are "80\% sure of" on a setup that only works 40\% of the time is not an 80\% trade. It is much closer to a coin flip than your confidence suggests, and for exactly the same reason.

## 1. Base-rate neglect: why the vivid case beats the frequency

We have seen *that* people neglect base rates. Now let's see *how badly*, and *why the neglect is so stubborn* — because understanding the mechanism is what lets you interrupt it at the screen.

The definitive experiment is another Kahneman and Tversky study, from their 1973 paper *On the Psychology of Prediction*. It is worth walking through, because it isolates the bias in its purest form.

#### Worked example: the lawyer-engineer problem

Participants were told a person had been drawn at random from a group of 100 professionals — some lawyers, some engineers. Then they read a short personality sketch: a man who is "conservative, careful, ambitious, shows no interest in political issues, and enjoys carpentry, sailing, and mathematical puzzles." Sounds like an engineer, right? Now, the probability that he *is* an engineer — what is it?

The trick is that the experimenters ran two versions of the group:

- **Pool A:** 30 engineers, 70 lawyers. Base rate of engineers = 30\%.
- **Pool B:** 70 engineers, 30 lawyers. Base rate of engineers = 70\%.

If you are reasoning correctly, the base rate should move your answer a lot. With only a vague, engineer-flavored description, a Bayesian would land somewhere near the base rate itself — roughly 30\% in Pool A and roughly 70\% in Pool B. The figure below shows what *should* happen versus what *does*.

![The lawyer-engineer problem as a matrix: flip the base rate from 30 to 70 percent and the correct answer flips with it, but people's answer stays put on the description.](/imgs/blogs/base-rates-and-the-outside-view-3.webp)

What actually happened: people gave almost the **same answer** in both pools — a high probability of "engineer" both times — because the *description* resembled their stereotype of an engineer, and they judged by resemblance. When the base rate flipped from 30\% to 70\%, the correct answer flipped with it, but people's answer barely moved. They had thrown the base rate away and kept only the story.

The one-sentence intuition: **a description that fits a stereotype hijacks your probability estimate so completely that you will give the same answer whether the underlying odds are 30\% or 70\% — the base rate stops registering at all.**

### Why the neglect is so hard to shake

Three features make base-rate neglect sticky, and each one has a direct trading analogue:

1. **The specific case is causal; the base rate is merely statistical.** A story about *why* this trade will work ("the shorts are trapped, they'll have to cover") feels like an explanation. The base rate ("setups like this work 40\% of the time") feels like a mere fact with no bearing on *your* case. The mind privileges causal stories over statistical facts, so the story wins even when the fact is more predictive.

2. **The base rate is invisible unless you deliberately fetch it.** The vivid chart is right in front of you, glowing on the screen. The base rate lives nowhere — it is an average over dozens of past trades you have to *reconstruct* from a journal or a backtest. What is present beats what is absent. If you never fetch the number, it never gets a vote.

3. **Every case feels like the exception.** This is the deepest one. Precisely *because* you know so much about today's trade — the exact catalyst, the exact chart, the exact tape — it feels too specific, too special, to be governed by a dumb average. But "this one is different" is what a trader thinks before *every* trade, and it is right roughly as often as the base rate says it is.

> The base rate is the reputation of a strategy; the story is the excuse for why today is different. Excuses do not have edges. Reputations do.

What this costs, and when it breaks: base-rate neglect is cheapest when your read genuinely adds a lot of information (a real, tested edge) and most expensive when it does not (a pretty chart and a good feeling). The problem is that the two feel *identical from the inside*. That is why you cannot fix this by trying harder to tell them apart in the moment. You fix it by fetching the number.

## 2. The inside view versus the outside view

Base-rate neglect is the bias. The **inside view** is the state of mind that produces it, and the **outside view** is the discipline that escapes it. To feel the difference in your gut, there is no better story than the one Kahneman tells on himself.

### A named case study: Kahneman's textbook that never was

In the 1970s, Daniel Kahneman — years before the Nobel Prize — assembled a team in Israel to write a high-school textbook on judgment and decision-making. The team met on Friday afternoons for about a year. They had built a syllabus outline, written a couple of chapters, run some sample lessons. Things were going well.

One Friday, Kahneman ran an exercise. He asked everyone to privately write down how many more years they thought the project would take. The estimates clustered tightly: the median was **2 years**, ranging from 1.5 to 2.5. Comfortable, reasonable, inside-view numbers, built from the team's sense of their own progress and pace.

Then Kahneman turned to one team member — Seymour Fox, a genuine expert in curriculum development who had seen many such projects. He asked Fox to forget their own team for a moment and think about the *reference class*: other teams, at roughly this same early stage, building a curriculum from scratch. How long did *those* take? And what fraction never finished at all?

Fox went quiet. Then he admitted two things that landed like a punch. First, a substantial fraction of such teams — about **40\%** — never finished at all; they simply gave up. Second, of the teams that *did* finish, he could not think of a single one that had done it in less than **7 years**, and some took closer to ten.

Sit with the gap. The inside view — built by the very same people, the very same afternoon, using everything they knew about their own project — said 2 years with near-certainty. The outside view, the plain base rate of comparable projects, said 7 to 10 years with a 40\% chance of total failure. These are not small disagreements. They are different universes.

![Kahneman's textbook project: the inside-view estimate of two years against the reference class of seven to ten, and the eight-year reality.](/imgs/blogs/base-rates-and-the-outside-view-4.webp)

And here is the part that should terrify anyone who thinks *awareness* is the cure. The team — a group that literally studied decision-making, that had just been handed the base rate by an expert in the room — talked it over and decided to carry on. They kept the 2-year estimate. They did not weight the outside view at all. The base rate was in the room, spoken aloud by an expert, and it still lost to the story of their own progress.

The project took **8 years** to complete. By the time the textbook was done, the enthusiasm at the Ministry of Education had evaporated and it was never used. The outside view had been almost exactly right — 8 years lands squarely inside the 7-to-10 band — and everyone in the room had chosen to ignore it. (Kahneman tells the whole story in *Thinking, Fast and Slow*, 2011, in the chapter titled, fittingly, "The Outside View.")

The one-sentence lesson: **knowing about a bias does not protect you from it; only a mechanical procedure that forces the base rate onto the page does.** We will build that procedure in the drill section.

### Choosing the reference class: the width tradeoff

Back in Foundations we said that choosing the reference class is the whole game, and promised to return to how you draw the line. Here it is, because it is where the outside view is easiest to sabotage.

Every reference class faces a tradeoff between **relevance** and **sample size**, and they pull in opposite directions:

- A **wide** class — "all breakouts I've ever taken" — gives you hundreds of examples, so the base rate is statistically stable. But it blurs real distinctions: a breakout on earnings in a raging bull market is genuinely different from a breakout on a quiet day in a chop, and averaging them together hides that.
- A **narrow** class — "breakouts in this exact sector, on an earnings gap, in a confirmed uptrend, with this volume pattern" — is far more relevant to today's trade. But you might only have six such trades in your journal, and a base rate computed from six trades is almost noise; two coin flips landing differently would swing it by a third.

The practical resolution is to **anchor wide and adjust narrow**. Start from the broadest class that clearly applies, because that base rate is the most reliable number you own. Then narrow *only* along dimensions where you have enough samples to trust the sub-rate, and treat each narrowing as a small, evidence-weighted adjustment — the same anchor-and-adjust move from Section 4, applied to the choice of class itself. If the narrow sub-rate is built on a handful of trades, lean back toward the wide anchor; the wide number is boring but honest, and the narrow one is exciting but unstable.

There is one failure mode to name explicitly, because it is how the inside view sneaks back in wearing the outside view's clothes: the **reference class of one**. If you slice the class so finely — "breakouts in *this* stock, on *this* catalyst, in *these* exact conditions" — that only today's trade qualifies, you have not found a base rate at all. You have re-described the inside view and called it research. A reference class with one member is just your story with a spreadsheet. The tell is that the narrowing was chosen *after* you liked the trade, and each cut happened to carve away the losers. Any reference class you can only populate with the trade you already want to take is not a reference class; it is a rationalization.

The one-sentence intuition: **a good reference class is wide enough that its average is stable and narrow enough that it still resembles today's trade — and the moment you narrow it down to a single member, you have quietly switched back to the inside view.**

### Reference-class forecasting: the outside view with a job

The outside view is not just a mindset; it has been turned into a formal method. The Oxford planning scholar Bent Flyvbjerg calls it **reference-class forecasting**, and it is the same move made rigorous: to forecast a project, don't estimate it bottom-up from its own plan (the inside view) — instead, assemble a class of completed similar projects, look at how their outcomes were actually distributed, and place your project in that distribution. Flyvbjerg's database of more than 16,000 large projects found that only about **0.5\%** came in on budget, on time, and with the promised benefits. The method he built from that data — anchoring each new forecast to the reference class of finished ones — has since been adopted by governments in the UK, the Netherlands, Denmark, Switzerland, and Hong Kong. We'll return to his cost-overrun numbers when we look at real markets.

The trading translation is exact. Your "project" is the trade. Its "budget and timeline" are your profit target and holding period. And the inside view will lie to you about both in precisely the optimistic direction Flyvbjerg documented — unless you fetch the reference class first.

## 3. Why the inside view feels better and predicts worse

We keep asserting that the inside view *feels* more informative. Let's make the mechanism concrete, because once you can name why it fools you, you can catch it fooling you.

### WYSIATI: what you see is all there is

Kahneman's compression of the whole problem is the acronym **WYSIATI** — "what you see is all there is." Your mind builds the most coherent possible story out of the information currently in front of it, and — this is the trap — it does not, and cannot, charge itself for the information that is *not* in front of it. The confidence you feel is a measure of how neatly the *available* facts hang together, not of how much you actually know.

The inside view is WYSIATI in action. The specific facts of today's trade are vivid and available; the base rate — the dozens of similar trades whose outcomes you can't see glowing on the screen — is absent. So the story built from the visible facts feels complete, and its coherence gets mistaken for correctness. A trade with a beautiful, tightly-fitting story feels like a high-probability trade *for the same reason a good movie feels true*: everything on screen is consistent. Consistency is not accuracy. The base rate is the pile of things off-screen, and off-screen does not vote.

### The uniqueness illusion

The inside view also whispers that your case is special: this catalyst is unusual, this chart is unlike the others, this time the setup is cleaner. Every one of those observations can be *true* and still be *irrelevant*, because the reference class already contains plenty of catalysts that felt unusual and charts that felt clean. Feeling like an exception is the base rate of trades. The exceptions that matter are the ones you can *measure*, not the ones you can *feel*.

### The planning fallacy: the outside view's most-studied cousin

The single best-measured consequence of the inside view is the **planning fallacy** — our systematic tendency to underestimate how long things will take and how much they will cost, even when we have lived through the same overruns before. Kahneman and Tversky named it; the cleanest measurement is a 1994 study by Roger Buehler, Dale Griffin, and Michael Ross.

They asked 37 psychology students to predict when they would finish their thesis. The figure below shows how that went.

![The planning fallacy measured: students predicted 34 days, took 55, and overran even their own pessimistic 49-day worst case.](/imgs/blogs/base-rates-and-the-outside-view-5.webp)

The students' average prediction was about **34 days** (33.9, to be exact). Their actual average was about **55 days** (55.5) — roughly 22 days and **63\%** longer than they thought. Worse: the researchers also asked for a *worst-case* estimate, "assuming everything went as badly as it possibly could." The average worst case was **49 days** — and reality still overran it. Only about **30\%** of students finished by the date they themselves had named. These were smart people, forecasting their own work, on a task they had every incentive to estimate honestly. The inside view still overran their own forecast by roughly two-thirds.

#### Worked example: the planning fallacy at the trade level

Suppose you take a swing-trade setup and your inside-view plan says: "this should hit my +8\% target within 5 trading days; my stop is 3\% below entry." That plan is built entirely from today's chart — the momentum, the catalyst, the clean structure.

Now fetch the reference class. You pull your journal on the last 40 trades of this exact setup:

- Median days-to-target among the winners: **12 days**, not 5.
- Fraction that hit the full +8\% target at all: **45\%**.
- Fraction stopped out first: **40\%**.
- Fraction still open, going nowhere, at day 5: **35\%**.

Watch what the inside view got wrong. It did not just misjudge the probability (you thought this was a near-lock; the base rate says 45\%). It misjudged the **timeline**: your "5 days" is less than half the 12-day reality, which matters enormously if you are, say, holding through an earnings date at day 8 or paying financing on a leveraged position. The inside view compresses time exactly the way Buehler's students did — and in a trade, a compressed timeline quietly changes your risk (you hold through events you meant to avoid) and your capital efficiency (the money is tied up twice as long as budgeted).

The one-sentence intuition: **the inside view underestimates not just whether a trade works but how long it takes, and the extra time is where unbudgeted risk sneaks in.**

What this costs, and when it breaks: the planning fallacy is worst on trades with a time dimension you're financing or gating on an event — options with theta decay, leveraged holds, anything with an earnings date or expiry in the window. On those, the reference-class *timeline* is as important as the reference-class *hit rate*, and the inside view lowballs both.

## 4. How pros use base rates: anchor first, then adjust

By now the outside view might sound like an instruction to throw your analysis away and just trade the average. It is not. The whole point of doing the work on a specific trade is that sometimes the specifics genuinely *do* shift the odds. The skill is not "ignore the story." It is **anchor to the base rate, then adjust** — start from the average and move off it only as far as real, tested evidence justifies. This is Bayesian reasoning in plain clothes.

![Anchor to the base rate, then adjust: professionals start from the reference-class rate and move off it only as far as evidence justifies.](/imgs/blogs/base-rates-and-the-outside-view-6.webp)

The figure lays out the sequence. Name the reference class. Write down its base rate — literally write the number. *Then* ask what, specifically, is different about today's trade, and how much your own track record says that difference is worth. Adjust by that much. Not by how excited the difference makes you feel — by how much it has historically moved the outcome. The order is everything: the base rate is the anchor, and the story is a small, evidence-weighted nudge, not a fresh start from zero.

Contrast this with the amateur's sequence, which runs the other way: start from the story ("this is going to work"), arrive at a felt probability (80\%), and only *maybe*, if you remember, sanity-check it against history. Starting from the story means the base rate can only ever be a small correction to a number the story already inflated. Starting from the base rate means the story has to *earn* every point it moves you.

#### Worked example: a base-rate-adjusted probability

Let's put real (illustrative) numbers on it. You are looking at a breakout setup.

**Step 1 — reference class and base rate.** Your journal says this exact setup, over 40 past trades, hit its target **40\%** of the time. Anchor: 40\%.

**Step 2 — what's genuinely different today.** You notice one real, testable feature: the breakout is happening on a day the whole sector is strongly green, and you have separately checked that, historically, your setup's hit rate on strong-sector days was a bit higher than on flat days — call it a lift from 40\% to 45\%. That is a *measured* adjustment, not a feeling.

**Step 3 — adjust, don't leap.** So the base-rate-adjusted probability is **45\%**, not 40\% — and emphatically not the 80\% the story wanted. You moved 5 points, because 5 points is what the evidence bought. The story does not get to move you 40 points on vibes.

Compare the two routes to a number:

| Step | Inside-view route | Outside-view (anchor-and-adjust) route |
|---|---|---|
| Starting point | the story ("this will work") | the base rate (40\% hit) |
| The specific detail | inflates confidence freely | adjusts the base rate by a measured amount |
| Ending probability | ~80\% (felt) | 45\% (anchored) |
| What the detail had to do | nothing — it was assumed | earn each point against history |

The one-sentence intuition: **the base rate is the anchor and the story is the adjustment — reverse that order and the story becomes the anchor, which is exactly how an 80\% conviction attaches itself to a 45\% trade.**

A light touch of the actual Bayesian machinery, for those who want it: Bayes' rule says your updated (posterior) odds equal your prior odds times the strength of the new evidence (the likelihood ratio). The **prior** is the base rate. If your prior is weak-ish (a 40\% base rate is prior odds of about 2-to-3 against) and your new evidence is genuinely only mildly informative, the posterior barely budges — you land near the base rate, not near certainty. The taxicab problem was this same arithmetic: a 15\% prior, an 80\%-reliable witness, and a posterior of only 41\%. Every time you feel an 80\% conviction on a trade whose base rate is 40\%, you are implicitly claiming your read is a *very* strong likelihood ratio — as strong as a near-perfect witness. Usually it is not. Usually it is a pretty chart.

## 5. From probability to position size

Here is why all of this is not merely an academic point about calibration. **Probability sets size.** Whatever number you believe — 80\% or 45\% — flows straight into how much you bet, and the bet is where the base rate turns into dollars. Two traders can take the identical trade, at the identical entry, with the identical stop, and one of them can lose five times as much as the other, purely because of which lens set the size.

![Same setup, two position sizes: believing the 80 percent story risks 25 percent of the account; anchoring to the 45 percent base rate risks 5 percent.](/imgs/blogs/base-rates-and-the-outside-view-7.webp)

#### Worked example: the same trade, sized two ways

You have a **\$10,000** account. You are looking at one setup, with one stop. The only question is size.

**The inside-view trader** believes the story: felt probability 80\%, this one's a winner. High conviction earns a big bet — they put **25\%** of the account on it, **\$2,500** at risk to the stop. If the trade fails (and remember, the true base rate is 45\%, so it fails more than half the time), that's a **\$2,500** loss — **−25\%** of the account, gone on one trade.

**The outside-view trader** anchors to the base rate: 45\%, barely better than a coin flip, so this gets a *small* bet. They risk **5\%** of the account, **\$500**, to the same stop. If it fails, that's **−\$500**, **−5\%** of the account. Annoying, survivable, forgotten by next week.

Same setup. Same entry. Same stop. Same market. The inside-view trader is risking **5×** what the outside-view trader is risking — **\$2,500 versus \$500** — entirely because the story inflated their probability from 45\% to 80\%, and the inflated probability licensed the oversized bet. Run that difference across a hundred trades and it is the entire distance between a career and a crater.

Notice the asymmetry that makes this lethal. The base rate is 45\%, so on this kind of trade you are wrong most of the time no matter how good the story sounds. When you are wrong most of the time *and* sized as if you were right most of the time, the losses compound against you exactly when they are largest. This is the deep reason base-rate neglect is not just an accuracy problem — it is a *survival* problem. (The connection between the number you believe and the size you take is the whole subject of [position sizing as emotional regulation](/blog/trading/trading-psychology/position-sizing-as-emotional-regulation); base rates are simply the honest input that sizing needs.)

The one-sentence intuition: **an inflated probability does not just make you wrong — it makes you wrong and oversized simultaneously, which is the only failure mode that actually ends accounts.**

## What it looks like at the screen

Biases are easier to name than to catch, because in the moment they do not feel like biases — they feel like insight. So here is the felt experience of base-rate neglect, the specific tells, so you can recognize the state you are in *while you are in it*. This is the moment to watch for:

- **You can tell the story fast, and it feels like proof.** The words come easily — "shorts are trapped, sector's rotating, earnings gap and go." Fluency feels like evidence. It is not; it is just fluency. The easier a story is to tell, the more suspicious you should be that you are trading the story and not the odds.
- **You have not looked up a single number about the past.** You know everything about *today's* chart and nothing about how the last 40 charts like it resolved. If someone asked "what's the base rate on this setup?" you would have to guess. That blank is the tell.
- **The size crept up without a decision.** You did not consciously choose to risk more; it just *felt* right to size up because conviction was high. Size that rises with feeling rather than with a written probability is base-rate neglect converting directly into risk.
- **"This one is different" is doing the heavy lifting.** Every reason you can name for why the average doesn't apply to today is a specific, vivid, causal detail — and none of it is a *measured* difference from your own records. The exception is felt, not counted.
- **You feel mild irritation at the idea of checking.** Fetching the base rate feels like a buzzkill, a bureaucratic delay between you and a trade you already *know* is good. That irritation is the bias defending itself. The stronger the resistance to looking up the number, the more you needed to look it up.
- **The number in your head is round and high.** "I'm 80\% sure." "This is a 90\% setup." Precise-sounding, high, and sourced from nowhere. Real base rates are boring and middling — 40\%, 55\%, 47\%. A confident 80\% with no journal behind it is almost always the story talking.

The meta-tell that ties them together: **certainty that arrived without counting.** If you feel sure and you did not fetch a single historical number to get there, you are looking at the inside view, and the base rate is somewhere off-screen, unconsulted, quietly true.

## Common misconceptions

**"The outside view means ignoring my analysis and just trading the average."** No — it means *starting* from the average and adjusting for genuine, measured edge. Your analysis is the adjustment, not the anchor. The error is letting analysis be the *starting* point, because then it can only inflate. Anchor first; then your work has somewhere honest to push from.

**"Base rates don't apply to me because my situation is genuinely unusual."** Your situation *feeling* unusual is the base rate of situations — everyone's feels unusual from the inside, which is exactly why the feeling carries no information. The unusual features that actually change your odds are the ones you can *measure* against your own history, not the ones that merely feel special today.

**"If I'm 80\% confident, the trade is 80\% likely."** Confidence is a feeling about how well the story hangs together (WYSIATI); probability is a fact about how often the outcome occurs. They are different quantities and they routinely disagree. The taxicab witness was "80\% reliable" and still only 41\% right. Your 80\% conviction on a 45\% setup is the same illusion wearing a chart.

**"Knowing about base-rate neglect protects me from it."** Kahneman's own team — professional decision scientists, handed the base rate out loud, in the room — ignored it and lost six years. Awareness is nearly useless on its own. Only a *procedure* that forces the number onto the page reliably beats the bias. This is the single most important misconception to kill, because believing you're immune is what keeps you exposed.

**"A bigger, more detailed model of the trade gives a better probability."** More detail usually makes the inside view *more* convincing and *no more* accurate — extra detail adds coherence (the conjunction of specifics feels more real) without adding predictive power. Past a point, more analysis is just a better story, not a better estimate. The base rate does not get more accurate the harder you stare at today's chart.

**"Base rates are only for beginners; pros trade on read."** The opposite. What distinguishes professionals is that they *know their setups' base rates cold* and treat their read as a small adjustment to a known number — the discipline this whole article is about. The amateur trades on pure read and calls it intuition. (For the broader frame of treating every decision as a bet with known-ish odds rather than a right-or-wrong call, see [thinking in bets](/blog/trading/trading-psychology/thinking-in-bets-probabilistic-decision-making).)

**"The regime has changed, so the old base rate no longer applies."** Sometimes true — a genuine structural break (a new market maker, a rule change, a volatility regime shift) really can make an old average obsolete. But "the regime has changed" is also the single most convenient excuse for discarding a base rate you find inconvenient, and it is invoked far more often than regimes actually change. The honest test: can you point to a *specific, dated, external* change, and do you have even a small sample of trades from the new regime showing a different rate? If yes, build a new reference class from the new regime. If all you have is a feeling that "things are different now," that is not a regime change — it is base-rate neglect with a macro vocabulary.

## How it shows up in real markets

The taxicab is a toy and the thesis a textbook. Here is base-rate neglect and the outside view operating with real money and documented numbers.

### 1. The startup base rate everyone quotes wrong

Founders and the investors who back them are the purest inside-view thinkers alive: every startup's pitch is a vivid, coherent story about why *this* one is the exception. The base rate is unforgiving and well-measured. According to U.S. Bureau of Labor Statistics business-survival data, about **20\%** of new businesses fail in their first year, roughly **50\%** are gone by year five, and about **65\%** by year ten — only around one in three survives a decade (rates vary widely by sector). The popular "90\% of startups fail" figure is inside-view folklore that overstates the near-term rate; the real, boring, sourced number is closer to "half don't see year five." Notice that the *correct* number is less dramatic and more useful — which is base rates all over. Vivid beats accurate, until you fetch the data.

### 2. Active managers versus the index

Here is a base rate that has been measured every year for two decades, and ignored every year for two decades. S&P Dow Jones Indices publishes the **SPIVA** scorecard, which tracks how professional active fund managers do against their benchmark. Over the **15 years** ending December 2024, roughly **89.5\%** of U.S. large-cap funds *underperformed* the S&P 500 — fewer than one in six beat it over even a decade, and over 15-plus years, essentially no category had a majority of managers ahead. Every one of those funds was run by a professional with a detailed, inside-view story about why *their* process would beat the average. The outside view — the base rate — said about 1 in 9 would, over 15 years. Investors kept paying for the story anyway. This is base-rate neglect at the scale of trillions of dollars.

### 3. Megaproject overruns and the birth of reference-class forecasting

Flyvbjerg's Oxford research, mentioned earlier, put hard numbers on the planning fallacy in the physical world. Across large infrastructure projects, average real cost overruns run about **20\%** for roads, **34\%** for bridges and tunnels, and **45\%** for rail — and a staggering **157\%** for the Olympic Games. Every one of those budgets was an inside-view forecast built lovingly from the specifics of that one project. The fix that actually worked was not "try harder to estimate" but "**anchor the forecast to the reference class** of finished projects" — the outside view, formalized into policy. The exact same move you make when you pull your journal before sizing a trade.

### 4. The IPO you buy at the top of its story

A newly listed company arrives wrapped in the most polished inside-view narrative money can produce — a roadshow, a story, a vision. And the base rate is sobering: decades of research by University of Florida finance professor Jay Ritter have documented that IPOs, on average, have *underperformed* the broader market in the years following their listing. The individual IPO you're excited about has a thrilling story; the reference class of IPOs has a mediocre track record. Buying the story at the listing is base-rate neglect with an underwriter attached. (This is educational, not a recommendation about any security.)

### 5. The falling knife and the trader's own setup

The most personal version is the "this time the bounce is real" trade — buying a stock in a confirmed downtrend because *this* dip looks like the bottom. The reference class ("dips I've bought in downtrends") usually has a hit rate the trader has never once calculated, and it is almost always lower than the story suggests. The trader who *has* calculated it — who knows that particular setup works, say, 35\% of the time — sizes it like a 35\% trade and survives the losing streak. The trader who trades the story sizes it like an 80\% trade and does not. Same knife. Different lens. (Keeping the score that tells you your real hit rate is its own discipline — see [calibration: keeping score on your own forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts).)

## The drill: find the reference class before you size

Everything above collapses into one habit, and the habit is mechanical on purpose — because we have seen that awareness alone (even Kahneman's) does not work. The rule is: **before you choose a position size, you must write the base rate on the ticket, right next to your thesis.** Not think about it. Write it. The figure below is the whole loop.

![The reference-class drill: every trade routes through its reference class and base rate before position size is chosen.](/imgs/blogs/base-rates-and-the-outside-view-8.webp)

Walk the loop:

1. **Name the reference class.** Before anything else, answer out loud: "What is this trade an example of? What happens to setups *like this one*, on average?" Force yourself to describe the crowd today's trade belongs to — the specific setup, in similar conditions. If you can name it, you can find its rate.
2. **Fetch the base rate and write it down.** Pull the number from your journal or a backtest: of the last N trades in this class, how many hit target? How long did the winners take? What fraction stopped out? Write the number physically next to the thesis — "setup base rate: 40\% hit, median 12 days." The writing is not bureaucracy; it is what drags the invisible average onto the visible page so it can compete with the glowing chart.
3. **If you can't find a reference class, that IS the answer.** A setup so novel you have no comparable trades is not a high-conviction opportunity — it is an *unmeasured* one. Trade it at minimum size, as an experiment that builds the reference class for next time. Novelty is a reason to size *down*, not up.
4. **Adjust from the base rate, and log the adjustment.** Ask what is genuinely, *measurably* different today, and move off the base rate by only that much. Write both numbers — "base 40\%, adjusted to 45\% for strong sector." If you can't point to a measured reason, you don't get to adjust. The base rate stands.
5. **Size to the adjusted number, not the story.** The probability on the ticket — the anchored, adjusted one — sets the size. Not the feeling. A 45\% trade gets a 45\% trade's size, no matter how good the story sounds while you're typing the order.

The forcing function that makes this stick is the physical artifact: a line on every trade ticket that reads *base rate: ___*. An empty blank is a trade you are not allowed to size yet. Over a few hundred trades, filling in that blank does something quietly transformative — it builds the very database of reference classes that makes the next base rate easier to fetch, until pulling the outside view becomes as automatic as pulling up the chart. This is the same machinery behind [the trading plan as external cognition](/blog/trading/trading-psychology/the-trading-plan-as-external-cognition): you don't trust your in-the-moment mind to remember the base rate; you make the page remember it for you.

A tell that you have internalized the drill: you start feeling *less* certain about your best-looking setups, not more. When a chart looks perfect and your first instinct is to reach for the journal rather than the buy button, the outside view has taken hold.

## When this matters to you

This is not only a trading skill; it is a life skill that happens to be extremely profitable in markets. Every time you estimate how long a project will take, whether a relationship will last, how a negotiation will go, or whether the exciting new job is as good as it sounds, you are running an inside view drenched in specifics — and there is a boring, more accurate base rate one honest question away. "How do things like this usually turn out?" is one of the highest-leverage questions a person can learn to ask, and most people go their whole lives without asking it once.

In markets specifically, base-rate neglect is the bias that sets your size, and size is what determines whether you survive to get good. You can be a mediocre chart-reader and thrive if you size every trade to its honest base rate; you can be a brilliant one and still blow up if you size to the story. The story is the fun part, and there is nothing wrong with loving it — but love it the way you love a good movie, knowing it is built to be coherent, not to be true. Then go fetch the number.

This article is educational, not financial advice. Base rates tell you the odds of a *class* of trades; they cannot tell you what any single trade will do, and no method removes the risk of loss. The point is not certainty — it is sizing your bets to the real odds instead of the invented ones, so that being wrong, which you will often be, stays survivable.

The next time a setup makes the story write itself and your hand drifts toward a bigger size, do the one thing your mind will fight you on: stop, and ask what happens to trades like this on average. Write that number down. Then size to *it*. That single mechanical move — base rate first, story second, size to the anchor — is the whole of the outside view, and it is the difference between trading your evidence and trading your imagination.

## Sources & further reading

- Amos Tversky & Daniel Kahneman, "Evidential Impact of Base Rates" (1982), and Kahneman & Tversky, "On the Psychology of Prediction," *Psychological Review* (1973) — the taxicab and lawyer-engineer problems and the original account of base-rate neglect and representativeness.
- Daniel Kahneman & Dan Lovallo, "Timid Choices and Bold Forecasts: A Cognitive Perspective on Risk Taking," *Management Science* (1993) — the inside view vs. the outside view.
- Daniel Kahneman, *Thinking, Fast and Slow* (2011), chapter 23, "The Outside View" — the textbook-project story (2-year estimate, 40\% never finish, 7-to-10-year reference class, 8-year reality) and WYSIATI.
- Roger Buehler, Dale Griffin & Michael Ross, "Exploring the 'Planning Fallacy'," *Journal of Personality and Social Psychology* 67(3), 1994 — the thesis-completion study (34-day prediction, 55.5-day reality, ~30\% finishing on time).
- Bent Flyvbjerg, "From Nobel Prize to Project Management: Getting Risks Right" and the Oxford megaproject database — reference-class forecasting and the road/rail/tunnel/Olympics cost-overrun figures.
- [S&P Dow Jones Indices, SPIVA U.S. Scorecard](https://www.spglobal.com/spdji/en/research-insights/spiva/) — active-manager underperformance versus benchmarks (year-end 2024: ~89.5\% of large-cap funds trailing the S&P 500 over 15 years).
- [U.S. Bureau of Labor Statistics, Business Employment Dynamics](https://www.bls.gov/bdm/) — new-business survival rates (~20\% fail in year one, ~50\% by year five).
- Jay R. Ritter, IPO long-run performance research, University of Florida — the documented post-listing underperformance of IPOs on average.
- Sibling posts on this blog: [The Narrative Fallacy](/blog/trading/trading-psychology/the-narrative-fallacy-when-a-good-story-beats-the-data), [Thinking in Bets](/blog/trading/trading-psychology/thinking-in-bets-probabilistic-decision-making), [Calibration: Keeping Score on Your Own Forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts), and [Position Sizing as Emotional Regulation](/blog/trading/trading-psychology/position-sizing-as-emotional-regulation).
