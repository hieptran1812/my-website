---
title: "Negotiation, Offers, and the Comp Ladder"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The offer is the highest-leverage ten minutes of your year. Learn what is actually negotiable at quant firms, how to use competing offers honestly, how comp steps with level, and how to price the fine print that quietly costs you money: deferred comp, clawbacks, and non-competes."
tags: ["quant-careers", "quant-finance", "careers", "compensation", "negotiation", "job-offer", "deferred-comp", "non-compete", "expected-value", "garden-leave"]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The offer call is the single highest-leverage ten minutes of your year, and most candidates blow it by saying yes too fast.
>
> - **Negotiate the negotiable.** The sign-on bonus and start date move easily, the base and level move a little, and the annual bonus and the fine print do not move at all — so spend your leverage where it can actually land.
> - **A live competing offer is the only real leverage you have.** Create one honestly by interviewing in a cluster; never invent one, because bluffs get called and cost you the relationship.
> - **The fine print is real dollars.** Deferred comp (golden handcuffs), clawbacks, and especially non-competes and garden leave can quietly cost you six figures — a six-month non-compete on an 800k USD seat is roughly 250k USD of forgone earnings.
> - **The one number to remember:** the expected value of a single polite counter is enormous and the downside is near zero. Asking once on a sign-on might be worth ~30,000 USD in expectation against a downside measured in hundreds. *Always ask once.*

Maya had been chasing this seat for fourteen months. The internship, the conversion interviews, the superday — all of it had pointed here, to a Tuesday afternoon and a recruiter's voice saying the words she had rehearsed hearing: *we'd like to extend you an offer.* Base of 275,000 USD. Sign-on of 75,000 USD. A target bonus that the recruiter described, carefully, as "discretionary and performance-linked." And then the question that decides everything: "Can we get a verbal yes today?"

Maya said yes.

It was a good offer. It was also, almost certainly, not the *best* offer she could have had from that same firm on that same afternoon — and the gap between the two was probably larger than her entire first paycheck. Because the firm had a sign-on band, and 75,000 USD was the bottom of it, and a single sentence — "I'm thrilled, and I'd love to get to yes; is there any flexibility on the sign-on?" — would have cost her nothing and very likely moved the number. She did not say that sentence. She left, by a conservative estimate, somewhere between 50,000 and 100,000 USD on the table, in the ten minutes it takes to drink a coffee.

This post is about not being Maya-on-that-Tuesday. The whole arc of this career series has framed the job as a probabilistic edge — and getting the job is itself a sequence of positive-expected-value decisions. The offer is the last of those decisions, and it is the one with the steepest payoff curve of all, because the downside of negotiating politely is tiny and the upside is measured in years of compounding. Figure 1 lays out the sequence we'll walk through: receive the offer, *do not* accept on the spot, buy yourself time, create real leverage, model the full package, counter once on the things that move, and close in writing. Everything else in this post is detail hung on that spine.

![The negotiation sequence shown as a six-step pipeline from receiving an offer through asking for time, creating leverage, modeling the package, countering once, and closing in writing, with a loop back to creating leverage if the counter is refused](/imgs/blogs/negotiation-offers-and-the-comp-ladder-1.png)

## Foundations: what an offer actually contains

Before you can negotiate an offer you have to know what is *in* one, because the headline number that the recruiter says out loud is the least informative part of it. A quant offer is a bundle of components, and they behave very differently. Let's define every piece from zero — assume you have never read an offer letter — and then sort each piece into one of three buckets: **negotiable**, **semi-fixed**, and **fine print you must price but cannot move**.

**Base salary.** This is the fixed annual cash you get paid regardless of performance, in equal installments through the year. At top quant firms in 2026 the base for a new-grad or junior quant runs roughly **250,000 to 375,000 USD** (levels.fyi / Glassdoor, reported 2025–2026); Jane Street, for example, quotes an "annualized equivalent of about 300,000 USD" base across its quant-trader, quant-research, ML, and FPGA roles. The base is the floor of your income — the part you can count on for rent and a mortgage — but at these firms it is *not* where the money is. It is also fairly **flat within a level band**: everyone hired at the same level gets a base inside a narrow range, so there is only a little room to move it.

**Sign-on bonus.** A one-time cash payment for joining, usually paid in your first one or two paychecks (sometimes with a small clawback if you leave within a year). For new grads this is commonly **50,000 to 200,000 USD** (DATA: reported 2025–2026 ranges). The sign-on is special because it sits *off* the level band — it is the firm's discretionary "sweetener" to close you — which makes it the **single most negotiable line in the whole offer.** Remember that.

**Annual (year-end) bonus.** The variable pay, and at a quant firm the variable pay is the *whole game*. For a junior the on-target bonus might roughly match or exceed the base; for a strong mid-level trader it can be many multiples of base. But — and this is the honesty mandate of this series — the bonus is **discretionary and tied to profit-and-loss (P&L) contribution**, which means two things. First, it is not contractually promised, so it is the *hardest* thing to negotiate (you cannot move a number the firm hasn't committed to). Second, and more dangerous, **it does not repeat automatically.** A 1.3M USD bonus year can be a 300k USD bonus year if your seat changes, your P&L turns, or the firm's year is flat. We'll come back to this when we talk about the comp ladder, because mistaking one strong bonus for your "salary" is the most common comp mistake quants make.

> "P&L" is *profit and loss* — the dollar result your trading book or your research signal produced. At a pod shop or prop firm your bonus is roughly a percentage of the P&L you are credited with; that is why the variable pay is so large and so volatile.

**Deferred compensation.** Part of your bonus (often a third to a half above some threshold) is not paid as cash now; it is *deferred* and vests over the next two to four years, sometimes as a notional fund investment, sometimes as a cash deferral. The firm's stated reason is to align you with long-term performance. The real effect, from your side, is a set of **golden handcuffs**: an unvested pile of your own money that you forfeit if you leave. We'll devote a whole section to this, because it is the part of comp that most quietly governs whether and when you can ever change jobs.

**Level / title.** Your level sets your base band, your bonus target, and your ceiling — it is the dial behind the other numbers. It is mostly fixed by the firm's read of your seniority, but it is occasionally negotiable, especially if you are coming in experienced and a competing offer pegs you a rung higher.

**Start date.** When you join. This is almost free for the firm to move and is worth asking about if the timing matters to you — and it can even be a *negotiating chip* (an earlier start in exchange for a slightly better sign-on, say).

**Non-compete and garden leave.** Here is the clause most candidates skim and later regret. A **non-compete** is a contractual promise that, after you leave, you will not join a competing firm for some window — at quant shops commonly **three to twelve months.** **Garden leave** is the version where you are still technically employed and *paid* (usually base only) but barred from working — you are paid to sit at home, in your garden, doing nothing, so your knowledge of the firm's positions and signals goes stale before you can take it to a rival. From your perspective both are the same economic object: **a period where you cannot earn your full market comp.** That is a real, large, and almost-always-overlooked cost. We'll price it.

Figure 2 sorts these seven components into the negotiable / semi-fixed / fine-print buckets, and tags how movable each one is. The whole strategic point of the figure — and of this post — is that you should spend your limited negotiating capital on the green rows (sign-on, start date), spend a little on the amber rows (base, level), and *not waste a single sentence* trying to move the red rows (the discretionary bonus). The red fine-print rows you don't move at all; you *price* them and let them decide which offer you take.

![A matrix sorting offer components into how each behaves and whether you can move it: sign-on bonus and start date are most negotiable, base salary and level are semi-fixed within a band, the annual bonus is rarely negotiable, and deferred comp and the non-compete are fine print you must price but cannot move](/imgs/blogs/negotiation-offers-and-the-comp-ladder-2.png)

## What's actually negotiable (and what isn't)

The reason most people negotiate badly is that they push on the wrong line. They spend their political capital arguing the base up by 5,000 USD — a line that barely moves and barely matters — and never touch the sign-on, which is the line that moves most and matters most. So let's be concrete about where the give is.

**Highly negotiable: the sign-on bonus.** This is your first and best ask. Because the sign-on sits off the level band, the recruiter usually has discretionary room to add to it without going to a committee, and adding to it doesn't reset the band for the next hire. A sign-on quoted at the bottom of a 50k–200k band can frequently be nudged 25,000 to 75,000 USD higher with a single polite ask, and dramatically higher if you have a competing offer that beats it. If you only do one thing in your whole negotiation, **counter on the sign-on.**

**Often negotiable: the start date, and sometimes the level.** The start date is nearly costless for the firm to move, so if you want more time off before you start, or you need to finish a degree, just ask. The level is harder — it is the firm's judgment of your seniority — but it is genuinely on the table if you are an experienced hire, especially if a competing offer at a rival firm has slotted you a rung higher. Moving up a level is the highest-value thing you can negotiate because it lifts base, bonus target, *and* ceiling all at once. It is also the hardest, so save it for when you have real evidence (a competing offer at that level).

**A little negotiable: the base.** There is usually a small band — maybe 10,000 to 25,000 USD of room at the junior end — but pushing hard on base is low-EV because the band is tight and because base is the *least* important part of your comp anyway. Ask once, lightly, and move on.

**Barely negotiable: the annual bonus target.** You can sometimes get a *guaranteed first-year bonus* written in (a floor on year one to de-risk the unknown), and that is worth asking for, because it converts a discretionary number into a contractual one. But the ongoing target is set by your seat and your level, and the firm will (rightly) refuse to promise variable pay it can't predict. Don't burn capital here beyond asking for a year-one guarantee.

**Not negotiable, but must be priced: the deferred comp schedule and the non-compete.** These are usually standard across all hires at your level, and the firm will not carve out an exception for you on the vesting schedule. The non-compete is occasionally *shortenable* or its scope *narrowable* (which competitors, what geography) if you push and you have leverage — and that is worth a try because it can be worth six figures — but mostly you treat these as fixed terms whose dollar cost decides which offer is actually better. More on that below.

#### Worked example: the expected value of asking once

Let's put a number on why you should always make at least one polite counter, using Maya's offer. Maya has a sign-on of 75,000 USD, the bottom of a 50k–200k band. Suppose — these are illustrative teaching probabilities, since no firm publishes a "counter-success rate" — that a single polite, in-band counter on the sign-on:

- **succeeds with probability 0.60**, and when it lands it lifts her comp by **50,000 USD** (a mid-of-band slice of the sign-on);
- carries a **0.02 probability** that something goes wrong (the offer is somehow soured or, in the worst imagined case, pulled), with a downside we'll generously price at **5,000 USD** of search friction — the cost of having to re-interview or accept a small delay.

In reality the rescind probability for a *polite, single, in-band* counter is widely described as near zero — firms expect strong candidates to negotiate, and a recruiter who pulls an offer over a courteous sign-on ask is unheard of at a reputable shop. But let's keep 0.02 to be conservative. The expected value of asking is:

```python
p_win, gain_if_win = 0.60, 50_000      # expected gain when the counter lands
p_rescind, cost_if_rescind = 0.02, 5_000  # near-zero downside, kept conservative

ev_gain = p_win * gain_if_win          # 0.60 * 50,000  = 30,000 USD
ev_loss = p_rescind * cost_if_rescind  # 0.02 *  5,000  =    100 USD
ev_net  = ev_gain - ev_loss            # 30,000 - 100   = 29,900 USD
```

The net expected value of opening your mouth once is about **+29,900 USD.** Thirty thousand dollars of expectation, for one sentence, against a downside of a hundred dollars. There is almost no decision in your career with a payoff-to-risk ratio this lopsided. Figure 3 draws it: a tower of expected gain and net EV beside a downside bar so thin you can barely see it.

![A bar chart comparing the expected gain of a single counter at thirty thousand dollars and the net expected value at about twenty-nine thousand nine hundred dollars against an almost invisible expected downside bar of one hundred dollars, showing the lopsided payoff of asking once](/imgs/blogs/negotiation-offers-and-the-comp-ladder-3.png)

*If a decision has a 300-to-1 expected payoff ratio, you take it every single time — and asking politely for a better sign-on is exactly that decision.*

This is the EV-under-uncertainty spine of the whole series applied to your own paycheck. You would never, as a trader, pass on a bet that paid 300-to-1 in expectation with a bounded tiny loss. Negotiating your offer is that bet. (For the formal machinery here, see [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) and [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews) — the same calibration you'd bring to a trading game is the calibration you bring to an offer.)

## Competing offers: the only real leverage, and how to create it honestly

Here is the uncomfortable truth about negotiation that the self-help books talk around: **a polite ask gets you a sliver, but the only thing that gets you a *lot* is a credible alternative.** A firm pays you to *not go somewhere else.* If there is no somewhere else, your leverage is mostly the firm's goodwill and the band's discretionary slack. If there is a real, live, better offer in your hand, your leverage is the actual market price of you — and that is a different conversation entirely.

This is just supply and demand, and it is the same logic that prices everything else in markets: a thing is worth what the next-best buyer will pay. Your comp is worth what the next-best firm will pay for you, and the way you discover that price is by having more than one firm at the table at the same time. Everything else — your résumé, your charm, your stated "excitement about the mission" — is noise compared to a competing number.

There's a deeper reason a competing offer is so much more powerful than any argument you could make about your own worth. When you tell a recruiter "I think I'm worth more," you are giving them an *opinion*, and opinions are cheap and contestable — the recruiter can simply disagree, and now you're stuck arguing about your own value, which is an unwinnable conversation. When you tell a recruiter "another firm has offered me X in writing," you are giving them a *fact about the market*, and facts about the market are exactly what a quant firm respects. You've converted a soft, ego-laden conversation ("am I good enough to deserve more?") into a hard, external one ("what does the market price me at?"). The firm doesn't have to believe you're better; it only has to decide whether you're worth matching the market for. That's a question with a clean answer, and it's a question the recruiter is *equipped* to take to the comp committee, because "we'll lose this hire to a named competitor at a named number" is the one sentence that reliably unlocks band discretion. A competing offer doesn't make you look greedy; it makes you look *priced*.

This is also why a single offer, however good, gives you almost no leverage on its own. With one offer, your only alternative to accepting is *no job*, and "no job" is a worse outcome for you than any reasonable offer — so the firm knows you'll likely take it, and the discretionary slack stays closed. The first offer sets your floor; the *second* offer is what raises it. The asymmetry is stark enough that it should reshape how you run your whole search: a candidate with one 600k offer and a candidate with two 600k offers are in completely different negotiating positions, even though the headline numbers are identical, because only the second candidate has a credible "or else."

**So the strategy is structural, not tactical: cluster your processes.** The reason the [recruiting calendar and funnel post](/blog/trading/quant-careers/the-quant-recruiting-calendar-and-the-funnel) tells you to apply to many firms at once is not just to raise your hit rate — it is to make several offers *land in the same window* so they can bid against each other. An offer that arrives two months after you've already signed elsewhere is worthless as leverage. Offers that arrive within the same two weeks are a competitive auction for you. Timing your funnel so the offers overlap is the single most valuable thing you can do for your comp, and it costs nothing but planning.

**How to use a competing offer — honestly.** The rules are simple and they are non-negotiable for your own reputation, because the quant world is small and recruiters talk:

1. **Only reference offers you actually have.** Never invent one. We'll get to why in the misconceptions section, but the short version: bluffs get called (recruiters routinely ask "can you share the written offer?" or "who is it with and at what level?") and a called bluff doesn't just lose you the negotiation, it can lose you the offer and burn a firm you may want to work at someday.
2. **Be specific and factual, not threatening.** "I have a written offer from [firm] at [number], and your team is genuinely my first choice — is there room to get closer?" This frames you as *wanting to say yes*, which is true and which gives the recruiter a reason to fight for you internally. "Match this or I walk" frames you as a mercenary and gives them a reason not to.
3. **Let the firm respond; don't negotiate against yourself.** State your alternative and your preference, then stop talking. Silence is leverage.
4. **Decide in advance what would make you switch your top choice.** If the firm you prefer can't match but gets close, you may still choose it — and that's fine. Leverage is not a vow to take the highest number; it's information that lets you choose with eyes open.

#### Worked example: Maya negotiates a sign-on with a competing offer as leverage

Run the tape on Maya's offer again, but this time she did her funnel right and has a *second* live offer when the recruiter calls. Firm A (her top choice) offers base 275,000 USD + sign-on 75,000 USD + target bonus. Firm B has put a written offer in her inbox at base 280,000 USD + sign-on 150,000 USD. Maya wants Firm A — better fit, better mentorship, better long-run seat — but Firm B's headline is 80,000 USD higher upfront.

She does not say "match B or I'm gone." She says: *"I'm genuinely most excited about your team — it's my first choice. I do have a written offer from another firm with a sign-on of 150,000 USD. The base difference doesn't matter much to me; the sign-on gap is real. Is there any room to close it?"*

The recruiter has a 50k–200k sign-on band and a candidate they've already invested fourteen months of pipeline in. The candidate is telling them, truthfully, that a defined gap is the only thing in the way and that they *want* to accept. This is the easiest internal case a recruiter ever makes. Suppose they come back with a sign-on of 140,000 USD. Maya's gain from one honest, well-structured sentence:

```python
sign_on_before = 75_000
sign_on_after  = 140_000
gain = sign_on_after - sign_on_before   # 140,000 - 75,000 = 65,000 USD upfront
```

Sixty-five thousand dollars, upfront, taxed once, hers — because she had a real alternative and used it the way you're supposed to. And she still gets to take her first-choice seat. Note what made this work: it wasn't aggression, it was *a true competing number plus a stated preference.* The leverage was real and the framing was honest.

*A competing offer is not a weapon you threaten with; it is a price you report — and the firm that wants you will move to meet a price it can see is real.*

## The comp ladder: how pay steps with level, and why

To negotiate well you need a mental map of where the offer sits on the ladder and where the ladder goes, because an offer is not just this year's number — it's an entry point onto a curve. Here's the shape of that curve at top quant firms, as of 2026, with all the conditionality the honesty mandate demands. These are reported ranges (levels.fyi, Glassdoor, efinancialcareers, the "Young & Calculated" 2026 quant-pay survey), and they are *survivorship-biased* — they describe the people who survived the up-or-out filter, not the median person who applied.

- **New grad / 0–2 years (top tier):** base 250k–375k USD, sign-on 50k–200k USD, first-year total typically **450k–650k USD** on-target. levels.fyi reports Jane Street QR around a 250k median (L1 ~307k, up to ~565k); Citadel QR around a 325k median (L1 ~336k → L3 ~642k, top reported ~721k); Jane Street QT base ~300k with the 90th percentile around 512k.
- **Mid-level / 2–5 years:** the range explodes because it now tracks P&L. A pod-shop QR around 4 years of experience might be ~575k (base 175k + bonus 400k); a strong big-prop QT at 2.5 years might hit ~1.5M (base 200k + bonus 1.3M) in a good seat. The *standard* mid-market band is more like 375k–500k; strong is 650k–820k.
- **5-year mark (survivors):** standard 800k–1.2M; poor 475k–625k; strong 2M–4M.
- **8-year mark:** standard 1.6M–3.8M; poor 0.9M–1.8M; strong 8M+. Top-tier QT outliers at 5–7 years can be 8M–12M; a 10-year top-prop QT example: base 250k + bonus 2.75M = ~3M.

Before the structural facts, it's worth pausing on *why the ladder is shaped this way*, because the shape is not arbitrary — it falls directly out of how quant firms make money. A market-making or prop firm earns its living from edge that its people generate: a trader's calibration, a researcher's signal, an engineer's latency win. Unlike a software company, where a great engineer might make the product 20% better, a great quant can make the firm's P&L *several times* what a mediocre one does — and a *negative* quant can lose real money. The dispersion in output is enormous, so the firm pushes the dispersion into pay: it keeps the base modest and flat (the part it pays you just to show up and not blow up) and makes the bonus the lever (the part that tracks the edge you actually produced). That's why the curve fans out so violently after year two: by then the firm has a read on your P&L, and the bonus — the variable, P&L-linked part — starts to dominate. The "standard / strong / poor" bands aren't three different people; they're the same person in three different *years*, because edge is noisy and a great year doesn't repeat on command. Holding this picture in your head is what keeps you from the cardinal comp error of treating one strong bonus as a salary.

Two structural facts about this ladder change how you should think about any single offer:

**Fact one: the base is flat-ish; the bonus is the lever — and the bonus tracks P&L, which does not repeat.** Notice that base barely moves from new-grad to mid-level (250k → maybe 200k–250k; it can even *fall* as a fraction). The entire growth in the ladder is the *bonus*, and the bonus is a function of the P&L you are credited with. This is the most important and most misunderstood thing about quant comp: **a 1.3M USD year is not a 1.3M USD salary.** If your seat changes, your strategy decays, or the firm has a flat year, next year's bonus can be a fraction of this year's. The ladder above shows *bands* — standard / strong / poor — precisely because the same person bounces between them year to year. (The [comp-demystified post](/blog/trading/quant-careers/quant-compensation-demystified) is the full treatment; the takeaway for negotiation is: don't anchor a job choice on one firm's best-case bonus story.)

**Fact two: the ladder is steep, so *level* is the highest-value thing to negotiate.** Because comp roughly multiplies as you climb, getting placed one level higher at the start compounds for years — a higher base band, a higher bonus target, and a higher ceiling, all at once. This is why, *if* you have the leverage (a competing offer at that level, or genuinely senior experience), pushing on level beats pushing on this year's cash by a wide margin. For most new grads level isn't movable; for experienced hires it's the prize.

The link between the comp ladder and the firm-hopping math deserves its own post — [staying versus switching](/blog/trading/quant-careers/staying-vs-switching-the-firm-hopping-math) works through when a jump pays — but the connective tissue is this: *the ladder is why people leave* (a competitor offers to slot you higher) and the *fine print is why leaving is expensive* (deferred comp and non-competes). Those two forces, the pull of the ladder and the drag of the handcuffs, are the entire economics of a quant career. Let's price the drag.

## Golden handcuffs: deferred comp and clawbacks

Recall that a chunk of your bonus — often the slice above some threshold — is not paid as cash now. It is **deferred** and vests over the next two to four years. Each strong year adds a new deferred slice on top of the slices still vesting from prior years, so the slices *overlap and stack.* At any given moment there is a pile of **unvested deferred comp** sitting in your name that you will only receive if you stay long enough to vest it. Leave before then, and you forfeit it. That forfeitable pile is the "golden handcuff," and Figure 6 shows it accumulating year over year.

![A timeline of deferred compensation over four years showing each year adding a deferred slice that vests over the following years, the unvested forfeitable pile growing from about one hundred fifty thousand dollars to about five hundred thousand dollars, then branching into forfeiting the pile if you leave versus the slices vesting in turn if you stay](/imgs/blogs/negotiation-offers-and-the-comp-ladder-6.png)

The mechanic is worth slowing down on because it's *designed* to compound your lock-in:

- In year 1 you earn a bonus; part is cash, part is deferred (say it vests evenly over the next three years).
- In year 2 you earn another bonus; part of *that* is deferred too. Now you have two overlapping deferral schedules running.
- By year 3 or 4 you have three or four slices stacked, and your **unvested pile is at its largest** — it's the sum of all the not-yet-vested portions of several strong years. This is the moment the handcuffs are tightest.

The firm built it this way on purpose. The deeper into a good run you are, the more you'd forfeit by leaving, so the more it would take for a competitor to pry you loose — they'd have to *buy out* your unvested deferral (some do, with a "make-whole" sign-on) or you'd have to walk from it. Deferred comp is the firm's retention tool, and it works precisely because your own past success becomes the chain.

**Clawbacks** are the nastier cousin. A clawback clause lets the firm *reclaim* comp already paid — or cancel unvested deferral — under defined conditions: you leave for a competitor, you breach the contract, your trades are later found to have violated risk limits or compliance, or in some structures the firm has a bad year and reverses prior accruals. The most common one you'll meet as a quant is the **leave-for-a-competitor clawback**, which is really the non-compete and the deferral working together: leave for a rival and you both forfeit the unvested pile *and* may trigger the non-compete sit-out. Read the clawback triggers in the actual contract. They are the difference between "I'll leave whenever a better offer comes" and "I literally cannot afford to leave for eighteen more months."

**What this means for your negotiation and your decision.** You can't usually *move* the deferral schedule — it's standard. But you can and must *price* it, and you can negotiate around it in two specific ways:

1. **Ask about a make-whole / buyout when you're the one being recruited away later** — that's a future negotiation, but knowing it exists shapes today's choice.
2. **When weighing two offers today, count only the comp you'll actually *keep* on your likely time horizon.** If you might leave in two years, deferred comp that vests over four is worth far less than its headline to you. We'll do exactly this comparison next.

## Non-competes and garden leave: the cost nobody models

If deferred comp is the chain, the non-compete is the wall. A **non-compete** says that for some window after you leave — three to twelve months is the common quant range — you may not join a competing firm. **Garden leave** is the paid version: you remain on payroll (typically at base only, with no bonus and no deferral accruing) but you're barred from doing your job, so your knowledge of the firm's positions and signals decays before a rival can use it. Either way, the economic reality is identical and brutal: **for that window, you cannot earn your full market comp.**

This is the single most under-priced term in quant offers, because it doesn't appear as a number on the offer letter — it appears as a *clause*, and clauses don't feel like dollars. But it is dollars, and a lot of them. The way to price it is to ask: *what is the gap between what I'd earn at my next seat and what I'm paid (if anything) to sit out, multiplied by the length of the sit-out?* Figure 5 stacks that forgone-earnings cost up by the month for an illustrative mid-level quant.

![A bar chart of cumulative forgone earnings during a non-compete or garden leave for a mid-level quant, rising about fifty thousand dollars per month so that a three-month sit-out costs about one hundred fifty thousand dollars and a six-month sit-out about three hundred thousand dollars in forgone earnings](/imgs/blogs/negotiation-offers-and-the-comp-ladder-5.png)

#### Worked example: the cost of a six-month garden leave

Take a mid-level quant — call her Maya, three years in — whose total comp at her *next* seat would be about 800,000 USD per year (the standard mid-level survivor band). Her current firm has a six-month garden leave that pays **base only**, and her base is 200,000 USD per year. The cost of the sit-out is the gap between what she'd be earning at the new seat and the base she's paid to sit:

```python
new_seat_annual    = 800_000        # target total comp at the new seat (USD/yr)
garden_base_annual = 200_000        # base only, paid during garden leave (USD/yr)
months_out         = 6

monthly_gap = (new_seat_annual - garden_base_annual) / 12   # (600,000)/12 = 50,000/mo
forgone     = monthly_gap * months_out                      # 50,000 * 6   = 300,000 USD
```

A six-month garden leave costs Maya roughly **300,000 USD** in forgone earnings — and that's the *generous* case where she's still paid her base. If the non-compete is *unpaid* (you've left, you're not on payroll, you just can't work), the cost is the full 800k-annualized run-rate for the window: a six-month unpaid non-compete would be ~400,000 USD of forgone earnings. A twelve-month garden leave roughly doubles the paid-case number to ~600,000 USD. These are not rounding errors. They are larger than most people's entire first-year comp, and they are invisible on the offer letter.

*A non-compete is a tax on leaving, denominated in months of your market rate — and a six-month one on a 800k seat is a quarter-million-dollar tax you signed up for without noticing.*

There's a subtlety worth flagging: a non-compete cuts *both ways* in your favor and against you. The firm you're *joining* may have to pay you a make-whole sign-on to cover the comp you forfeit during your *previous* firm's garden leave — so a long non-compete at your old firm can actually become leverage for a bigger sign-on at the new one. But that only helps if you knew to ask. The default, if you say nothing, is that you eat the cost.

It's also worth being clear-eyed about *why* the firm wants the non-compete in the first place, because understanding its motive tells you where the give is. At a market maker or HFT firm, the crown jewels are the strategies, the signals, and the latency tricks — and the fastest way for a competitor to steal them is to hire the person who knows them. The non-compete and garden leave exist to let that knowledge *decay* before you can carry it across the street: a six-month sit-out means that by the time you start at the rival, the specific edges you knew have moved, the parameters have been re-tuned, and your inside knowledge is half-stale. This is why the firms with the most perishable edge (the low-latency shops like Jump and HRT and Citadel Securities) tend to enforce the longest sit-outs, while firms whose edge is more durable or more diffuse may be lighter. It also tells you the *scope* argument that works: if you're moving to a firm or a desk that genuinely doesn't compete with your old book — a different asset class, a different strategy, a different geography — you have a legitimate case that the non-compete shouldn't bite, because there's no edge to protect. Recruiters and legal teams will sometimes narrow the scope on exactly that logic, especially for someone moving into an adjacent rather than a head-to-head role.

One more practical point: the enforceability of non-competes varies by jurisdiction, and the legal landscape has been in flux. Some regions and regulators have moved to limit or ban broad non-competes, while finance has often retained carve-outs for highly compensated employees and for genuine trade-secret protection. The lesson is *not* "assume it won't be enforced" — at a well-resourced quant firm, assume it will be, and negotiate it as if it's binding. The lesson is that the term is *legal*, which means it has more give than a comp band: it's the kind of clause that an employment lawyer reviewing your offer can sometimes get shortened, scoped down, or converted to paid garden leave. For an offer with a long sit-out and a large package, paying a lawyer a few hundred dollars to review the restrictive covenants is one of the highest-EV expenditures in this entire process.

**Can you negotiate the non-compete?** Sometimes, and it's worth trying because the stakes are six figures. You generally can't delete it, but you can occasionally:

- **shorten the window** (twelve months → six),
- **narrow the scope** (define "competitor" narrowly, or limit it to a specific desk/strategy/geography), or
- **convert unpaid to paid** (garden leave at full base, or even at a fraction of total comp, beats an unpaid sit-out).

Each of these is worth real money on the math above. The non-compete is technically "fixed" fine print, but unlike the deferral schedule it's a *legal term*, and legal terms are exactly the kind of thing that has more give than recruiters volunteer — especially for senior hires.

#### Worked example: the higher headline that loses after the fine print

Now combine deferred comp and the non-compete into the comparison that actually decides offers. Maya has two:

- **Offer A — 650,000 USD headline.** Base 250k + bonus 350k + a 50k sign-on, but **200k of the bonus is deferred** over three years, and there's a **six-month non-compete** (garden leave at base only).
- **Offer B — 560,000 USD headline.** Base 250k + bonus 260k + a 50k sign-on, only **40k of the bonus deferred** over one year, and **no non-compete.**

A's headline is 90,000 USD higher. But Maya is realistic — she's early, she may want the freedom to switch in two years if a better seat opens. Let's compute the **cash she actually keeps in year one** and the **freedom cost**:

```python
    # Offer A: 650k headline, but a third is deferred away from year 1
    A_cash_year1 = 250_000 + (350_000 - 200_000) + 50_000   # base + (bonus - deferred) + sign-on
    #            = 250,000 + 150,000 + 50,000 = 450,000 USD kept in year 1
    A_noncompete_cost = 6 * ((800_000 - 250_000) / 12)      # 6-mo garden leave, base-only
    #            = 6 * 45,833 = ~275,000 USD of optionality cost if she ever leaves

    # Offer B: 560k headline, almost all cash, no lock-in
    B_cash_year1 = 250_000 + (260_000 - 40_000) + 50_000    # base + (bonus - deferred) + sign-on
    #            = 250,000 + 220,000 + 50,000 = 520,000 USD kept in year 1
    B_noncompete_cost = 0                                   # free to switch any time
```

Offer B keeps Maya **520,000 USD** in year one against Offer A's **450,000 USD** — B is *higher* on spendable cash this year despite a headline that's 90,000 USD *lower*. And B carries no ~275,000 USD non-compete tax on her freedom to leave. The "bigger" offer is the worse offer on both axes that matter to someone who might move. Figure 4 shows the flip.

![A before-and-after comparison of two offers showing Offer A with a 650k headline but a third deferred and a six-month non-compete keeping only about 450k in year one, versus Offer B with a 560k headline mostly in cash and no non-compete keeping about 520k in year one and beating Offer A](/imgs/blogs/negotiation-offers-and-the-comp-ladder-4.png)

*The headline is a marketing number; the cash you keep on your real time horizon, after the deferral and the sit-out tax, is the only number that should decide which offer you sign.*

(Caveat for honesty: if Maya is *certain* she'll stay four-plus years, the deferred 200k in Offer A vests and the picture narrows — A's total over the full horizon can win. The point is not "always take B." The point is **model both on your actual time horizon**, because the deferred comp and the non-compete change the answer and the headline hides them.)

## How to run a negotiation

Here is the sequence — the same one in Figure 1 — written as a script you can actually follow. None of it is aggressive; all of it is positive-EV.

**1. On the offer call: thank, get excited, and do *not* accept.** "This is fantastic, thank you — I'm really excited. Can you send the details in writing, and can I have a little time to review?" You are allowed to be thrilled. You are not allowed to say yes on the call. Saying yes on the call is the single most expensive habit in this whole post — it's what Maya did on her Tuesday.

**2. Ask for time — and get it in a number.** "Could I have until [date, ~1–2 weeks out] to give you a decision?" One to two weeks is standard and reasonable; firms expect it. This window is not idle — it's where you let your other processes finish so a competing offer can land. Time *is* leverage because leverage usually arrives late.

**3. Read the whole letter, then model the package.** Pull the deferral schedule, the vesting period, the clawback triggers, the non-compete length and whether it's paid, and the start date. Compute the year-one cash kept and the fine-print cost on your real time horizon, exactly as in the worked examples. You can't negotiate what you haven't priced.

**4. Counter once, on the highest-EV negotiable item, with a specific number.** Lead with the sign-on (or, if you have a competing offer at a higher level, lead with level). Be specific: "I'd love to get to yes. Is there flexibility to bring the sign-on to [number]?" If you have a competing offer, report it factually and state your preference. Then **stop talking** and let them respond. One clean, specific ask beats five hedged ones.

**5. If they move, close — in writing.** Get the revised terms in an updated letter and *read it again*. Confirm the sign-on, the bonus target (and any year-one guarantee you won), the deferral, the non-compete, and the start date all match what you agreed. Verbal agreements drift; the letter is the contract.

**6. If they can't move, decide with your model, not your ego.** Maybe you take it anyway because it's your first choice and the gap is small — that's a fine, eyes-open decision. The loop in Figure 1 (counter → back to leverage) is for the case where the first counter is refused *and* you have a live alternative: you can go back to the alternative, see if *it* will move, and return. But you only loop with real cards. You never bluff.

A note on tone that runs under all six steps: **the recruiter is not your adversary.** A good recruiter *wants* to close you and is often your advocate to the comp committee — "the candidate is great and a 40k bump on the sign-on gets us to yes" is an easy internal sell. Frame every ask as helping them help you say yes. Negotiation at a reputable firm is collaborative, not combative; the combative frame is what produces the horror stories.

## Common misconceptions

**"Negotiating risks the offer — better to just take it."** This is the fear that costs the most money, and it's almost entirely false at reputable firms. Firms *expect* strong candidates to negotiate; a single polite, in-band counter does not get an offer pulled. As the EV worked example showed, the downside is measured in hundreds of dollars of probability-weighted friction and the upside is tens of thousands. The people who get offers rescinded are the rare few who negotiate in bad faith — bluffing fake offers, making aggressive ultimatums, or trying to renegotiate after accepting. Be polite, be honest, ask once. The risk of *asking* is tiny; the cost of *not asking* is large and certain.

**"Everything is negotiable — push hard on every line."** Also false, in the other direction. Pushing on the discretionary bonus (which the firm can't promise) or grinding the base by 3,000 USD wastes your limited capital and makes you look like you don't understand how comp works — which, at a quant firm, is a bad look. Negotiation is *targeted*: you have a few units of goodwill to spend, and you spend them on the sign-on, the level (if you have leverage), and the non-compete terms. You don't spend them everywhere.

**"A non-compete doesn't matter — I'm not planning to leave."** This is the trap of the present tense. You're not planning to leave *today*; careers are long and the ladder pulls. A six-month non-compete is a ~250,000–400,000 USD tax that sits dormant until the day a better seat appears — and on that day it can be the thing that keeps you stuck, or forces the new firm to buy you out (which they may decline to do). You signed it when it felt abstract; you pay it when it's concrete. Price it *now*, while you still have the leverage to shorten it.

**"Take the highest base — it's the guaranteed part."** Tempting, because base feels safe. But base is the *smallest* lever on the quant comp ladder and a poor tie-breaker between offers. A 15,000 USD higher base is a rounding error next to a sign-on gap, a deferral schedule, or a non-compete window. Optimize the *whole package on your time horizon* — cash kept this year, deferred comp you'll actually vest, and the freedom cost of the fine print — not the one line that's easiest to compare.

**"A bigger bonus this year means a bigger salary."** The single most dangerous comp misconception, repeated from the foundations because it's that important. The bonus is P&L-linked and discretionary; it does *not* repeat automatically. A 1.3M USD year can be a 300k USD year. Never anchor a job decision, a mortgage, or a lifestyle on a single strong bonus. Model the *band*, not the best case. (This is the survivorship bias the whole series keeps flagging: the people quoting their best year are the ones who survived the filter — and even they don't repeat that year on demand.)

## How it plays out in the real world

In practice, the variation across firms is real and worth knowing as you read your own letter. **Prop trading and market-making firms** (Jane Street, Optiver, SIG, IMC, Citadel Securities, Jump, HRT) tend to lead on cash and sign-on for juniors — Five Rings and Jane Street top the H1B base disclosures around 300k USD, Citadel Securities around 257k — and their bonus is the lever as you climb. Deferral is common above a threshold; non-competes and garden leave are standard, and at the HFT firms (Jump, HRT, Citadel Securities) where your knowledge of latency-sensitive strategies is the crown jewel, the garden leave can run long precisely because the strategies decay so fast. **Pod shops** (Citadel, and the broader multi-strat world) tie comp tightly to *your* book's P&L, so the variance is highest and the deferral and clawback machinery is most elaborate — the deferred pile is large and the leave-for-a-competitor clawback is real. **Systematic funds** (Two Sigma, D.E. Shaw) lean more research-scientist in culture and often more structured in comp, with deferral tied to fund performance.

The mechanics of the negotiation itself are remarkably consistent: a verbal offer, a written letter, an expectation that you'll take a week or two, and a recruiter who expects one round of counter and has band discretion to meet a reasonable, well-supported ask — especially on the sign-on, especially with a real competing offer. The candidates who do well are not the most aggressive; they're the most *prepared* — they've read the fine print, priced the deferral and the non-compete, and have a specific number ready and a true alternative if they're lucky enough to have clustered their processes.

And the honest failure mode is Maya's: a brilliant candidate who out-competed thousands of applicants through a fourteen-month funnel, then gave away a year's worth of leverage in the last ten minutes by saying yes too fast. The interview is the hard part; the offer is the *valuable* part. Don't run the marathon and then trip on the finish line by not asking one polite question.

Figure 7 distills the whole thing into do's and don'ts and *why each one moves your EV* — the green rows raise expected value through real leverage and honest framing; the red rows destroy it through bluffing or chasing a headline that hides the fine print.

![A grid of negotiation do's and don'ts with the reason each moves expected value: do ask for time, counter on the sign-on, and price the deferred comp and non-compete because they raise EV through real leverage, and don't invent a fake competing offer or take the highest headline base because they destroy EV through bluffing or ignoring the fine print](/imgs/blogs/negotiation-offers-and-the-comp-ladder-7.png)

## When this matters / Further reading

This matters at exactly two moments, and they bookend a quant career. The first is the day your first offer lands — when the temptation to say yes and end the suspense is strongest and the EV of one polite counter is highest. The second is every time the ladder tries to pull you to a new firm, when the deferred-comp pile you'd forfeit and the non-compete you'd trigger decide whether the move actually pays. Both moments reward the same discipline: **negotiate the negotiable, price the fine print as real dollars, and never sign without modeling the deferred-comp and non-compete cost on your actual time horizon.**

The thread back to the rest of this series:

- For the full anatomy of quant pay — base versus bonus, why the bonus doesn't repeat, and the survivorship bias baked into every "everyone makes 600k" claim — see [quant compensation, demystified](/blog/trading/quant-careers/quant-compensation-demystified).
- For when a firm switch actually pays after you account for the handcuffs and the sit-out, see [staying versus switching: the firm-hopping math](/blog/trading/quant-careers/staying-vs-switching-the-firm-hopping-math).
- For *why* your offers can land in the same window — the planning that creates competing-offer leverage in the first place — see [the quant recruiting calendar and the funnel](/blog/trading/quant-careers/the-quant-recruiting-calendar-and-the-funnel).
- For the decision-theory machinery under all of this — calibrating a probability, weighing an expected value against a tail risk, choosing under uncertainty — see [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews) and [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews).

The comp figures throughout are reported ranges (levels.fyi, Glassdoor, efinancialcareers, and the "Young & Calculated" 2026 quant-pay survey), and the negotiation probabilities are explicitly illustrative teaching numbers — no firm publishes a counter-success rate. What is *not* illustrative is the structure: the sign-on is the most movable line, a real competing offer is the only large lever, the comp ladder grows through the bonus and not the base, and the fine print is real money. Internalize that structure and you'll never again leave a year's leverage on the table in the ten minutes that decide it.
