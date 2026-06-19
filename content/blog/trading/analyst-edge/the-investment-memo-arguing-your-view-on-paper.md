---
title: "The Investment Memo: Arguing Your View on Paper"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How to turn a view into a written investment memo that survives a skeptic: the structure, the steelmanned bear case, the bull/base/bear targets, and the risk/reward and sizing math that decide it."
tags: ["analysis", "market-view", "investment-memo", "thesis", "risk-reward", "position-sizing", "investment-committee", "bear-case", "scenario-analysis", "decision-process"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An investment memo is a written argument engineered to survive a skeptic: it leads with the call, proves it with evidence and three scenarios, steelmans the bear case and then rebuts it, and ends on a dollar risk/reward and an invalidation level you can be held to.
>
> - Write the body first and the executive summary **last** — the summary must be honest to what the body actually proved, not an aspiration you then try to justify.
> - The bear case is mandatory and must be **steelmanned**: state the strongest version of why you are wrong, then rebut it with evidence. A strawman bear case is worse than none.
> - Every memo lands on numbers: three scenarios with \$ price targets, a probability-weighted value, a reward-to-risk ratio, and a position size derived from the distance to invalidation.
> - The one rule to remember: **if your own bear case does not change the size of the position, you did not write a real bear case.**

A junior analyst stands up in the Monday meeting and pitches the best idea she has had all year. The company is mispriced, she says; the market is missing the margin story; the stock should be worth far more. She talks for four minutes and it sounds great. Then the portfolio manager leans back and asks one question: *"What's the bear case, and what would make you sell?"* She does not have a crisp answer. She knows the bear case exists — she has half-thought it — but she never wrote it down, never put a number on it, never decided in advance what would prove her wrong. Thirty seconds later the idea is dead in the room, and the PM has moved on. She was probably right about the company. It did not matter.

Two desks over, a different analyst pitched a similar-sized idea the week before. He did it badly out loud too — most people are worse speakers than they are writers. But he had circulated a four-page memo the night before. By the time the meeting started, the skeptics had already read the bear case *he* had written, in its strongest form, and *his* rebuttal of it. They had seen his bull/base/bear targets, the probability he put on each, the reward-to-risk ratio, and the exact price at which he would admit he was wrong. There was nothing left to ambush him with. The conversation was not "is this a good idea?" — it was "how big should we go?" That is the difference a memo makes. Not eloquence. Pre-emption.

This post is about that document. When the stakes are small — a quick trade, a tracking position — a one-page note is enough, and we covered that in the [trade-idea note](/blog/trading/analyst-edge/the-trade-idea-note-a-one-page-template). But when the stakes rise — a large position, real career risk, a pitch to an investment committee or to partners — the note grows into a memo: a structured written argument built specifically to survive someone trying to poke holes in it. Writing to persuade a skeptic is the single best stress test of a thesis, because the skeptic is real and the page does not let you bluff. Here is how to build one.

![Nine ordered sections of an investment memo from recommendation down to monitoring](/imgs/blogs/the-investment-memo-arguing-your-view-on-paper-1.png)

## Foundations: what an investment memo actually is

An **investment memo** is a written argument for taking (or not taking) a specific position, structured so that a reader who is actively looking for the flaw can find it quickly — or fail to. It is not a research report. A research report describes a company or a market; a memo argues for a decision. The unit of a report is *information*; the unit of a memo is *a recommendation with a defense*. If a paragraph in your memo does not move the reader toward "yes, take this" or "no, don't" — or toward a specific size — it does not belong.

Three audiences read a memo, and you write for all three at once:

- **The investment committee (IC) or the partners.** These are the people with the authority to allocate capital — or to veto you. They are time-poor and professionally skeptical; their job is to say no to most things. They read the call first, the bear case second, and the detail only if the first two survive.
- **The skeptic in the room.** Often one specific person who will, by temperament or assignment, try to break your idea. The whole memo is, in a sense, addressed to this person. You write it as if they are reading over your shoulder, because they will be.
- **Your future self.** Six months from now, when the position is up 20% or down 15%, you will reread this memo to decide whether to add, hold, or cut. The version of you that is excited today is writing instructions for the version of you that will be frightened or greedy later. The memo is a contract between them.

A memo differs from a one-page note in three ways, and each is a deepening, not just a lengthening.

**Depth.** The note states the thesis; the memo *proves* it. The note says "margins are mispriced"; the memo shows the segment-level margin bridge, the comparable companies, and the math that gets from today's price to the target. A note is a flag planted on a hill. A memo is the survey that shows the hill is really there.

**A steelmanned bear case.** This is the heart of the upgrade. A note might mention risks in a bullet list. A memo states the strongest possible argument that you are *wrong* — the version a smart short-seller would make — and then answers it with evidence. The discipline is not to list risks; it is to *argue against yourself as hard as you can*, and then show why the argument still fails.

**The financial and valuation case.** A note can gesture at "cheap" or "expensive." A memo carries the numbers: the scenarios, the price targets, the probability weights, the reward-to-risk ratio, and the size. A view without a financial case is a feeling. The memo forces you to convert the feeling into arithmetic, and arithmetic is where most weak theses quietly die.

The thread running through all three is **writing for a skeptic**. This is the meta-skill the memo trains. When you write to persuade someone who agrees with you, you can be lazy: gesture at the conclusion and they nod. When you write to persuade someone whose entire purpose is to find the hole, every loose claim becomes a liability. You start to feel, while writing, the exact spots where your argument is thin — because you can all but see the skeptic's pen circling them. Those spots are gold. They are precisely where your real work is. The memo is valuable less because the committee reads it and more because *writing it forces you to find your own weak points before anyone else does.*

This connects directly to the spine of this whole series: a view is only analysis if it can be falsified, sized, and updated. The memo is where you write all three down. It states what you believe, what the market believes (so the reader can see the gap), what would prove you wrong, and how many dollars you are willing to risk to be right. Everything in this post is in service of making those four things concrete and defensible on a single piece of paper.

## The memo structure, section by section

A memo has a near-universal skeleton. The order matters: it is the order a skeptical reader wants to consume it, which is not the order you write it (more on that below). Here is each section and what it must contain.

### 1. Recommendation and summary

One paragraph, at the very top, that a busy person can read in ten seconds and know the call. Action (buy / sell / short / pass), instrument, size, the base-case return, the reward-to-risk ratio, and the time horizon. Something like: *"Recommend buying 900 shares of XYZ at \$100, risking \$10,000 to the \$89 invalidation, for a base-case target of \$135 (+35%) over twelve months. Reward-to-risk 3.2:1. The market is pricing margin compression that the switching-cost evidence says will not happen."* That is the whole call in five lines. Everything below it exists to defend those five lines.

### 2. The thesis

The thesis in one or two sentences, with the *why*, not just the *what*. "The stock will go up" is not a thesis; it is a hope. "The stock will rerate as gross margin recovers to 40% once the new factory absorbs fixed costs, which the market is not modeling until next year" is a thesis: it has a mechanism, a number, and a reason the market is wrong. We built the anatomy of this in [structuring a thesis](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst) — claim, evidence, catalyst. The memo's thesis section is that structure, tightened to its sharpest form.

### 3. What's priced in / variant perception

This is the section that separates analysis from a book report, and it is where edge actually lives. State explicitly what the market currently believes — what is *priced in* — and then state precisely where you differ and *why your version is more likely right*. If you cannot articulate the consensus you are betting against, you do not have a view; you have an opinion that happens to agree or disagree with the tape. We devoted a whole post to this: [variant perception, where real edge comes from](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from). The memo demands you write the variant down as a sentence: *"Consensus believes X; I believe Y; the gap is worth Z."* If X and Y are the same, there is no trade.

### 4. Evidence and analysis

The three to five facts that support the thesis, plus the financial case. Not everything you know — the three to five things that, if a skeptic accepted them, would force them to accept your conclusion. This is where the discipline of writing for a skeptic bites hardest: you are not trying to overwhelm with volume, you are trying to build a chain where each link bears weight. The financial case lives here too: the model, the margin bridge, the comparable multiples, whatever converts the qualitative story into a number.

There is a quiet test for whether a piece of evidence belongs in this section: *would a skeptic concede it, and does conceding it move them toward your conclusion?* A fact that the skeptic disputes is not evidence yet — it is a second argument you now have to win, and it weakens the chain rather than strengthening it. A fact the skeptic concedes but that does not bear on the conclusion ("the company has a strong brand") is filler. The evidence that earns its place is the conceded-and-load-bearing kind: the contractual switching cost a customer disclosed in its own filings, the fixed-cost absorption math that any analyst can reproduce, the competitor's published pricing. Each is hard to dispute and each forces a step toward your target. Three of those beat thirty soft ones.

The financial case deserves its own paragraph because it is where qualitative theses go to be tested. A story about margin recovery is just a story until you build the bridge: gross margin is 32% today, the new factory adds \$40 million of revenue at 55% incremental margin, fixed costs are flat, and the arithmetic gets you to a blended 40% within a year. Now the skeptic cannot argue with the *story* — they have to argue with a *number*, and numbers are falsifiable. Often the act of building the bridge is what reveals the thesis is weaker than it felt: you discover the incremental margin assumption was doing all the work, or that the comparable multiple you assumed already prices in the recovery. The financial case is not there to dress up the thesis. It is there to give the thesis something to fail against.

### 5. Bull / base / bear scenarios with targets

Three explicit scenarios, each with a dollar price target and a probability. We built the scenario discipline in [base, bull, and bear](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios); the memo's job is to attach numbers and probabilities so the three collapse into a single **probability-weighted value** you can compare to the current price. A memo whose bull case is exciting but whose weighted value barely beats the price is a memo arguing against itself, and writing it out is how you discover that.

### 6. Catalyst and timeline

What closes the gap between price and value, and by when. A view can be right and still lose money if the catalyst never arrives — cheap can stay cheap for years, as we covered in [catalysts and timing](/blog/trading/analyst-edge/catalysts-and-timing-why-cheap-can-stay-cheap-for-years). The memo names the specific events (an earnings print, a contract renewal, a regulatory decision, a refinancing) that should force the rerating, and the rough window. No catalyst, no timeline, and the memo should say so honestly — a value trap dressed as a thesis is the most expensive kind.

### 7. Risk / reward and sizing

The dollar arithmetic: how much you make in the base case, how much you lose at invalidation, the ratio between them, and therefore how big the position should be. This is the section that turns a story into a position. We will work it in full below.

### 8. The bear case, steelmanned then rebutted

The single most important section, and the one amateurs skip or fake. State the strongest argument that you are wrong — the version the smartest short-seller on the other side would make — and then rebut it with evidence. If you cannot rebut it, you have just discovered your size is too big, or that there is no trade. The bear case is not a risk disclaimer; it is the adversarial proceeding you run against your own thesis.

### 9. Invalidation and monitoring

The price level or fact that would prove you wrong, written down *before* you enter, and the short list of things you will watch to know whether the thesis is on or off track. We covered the discipline in [what would change my mind](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront). In the memo it is a one-line commitment: *"If the next print shows margins below 35%, or the stock breaks \$89, the thesis is dead and I am out."*

#### Worked example: the recommendation and the size

Here is the recommendation section made concrete, with the full sizing arithmetic that backs it. The account is \$1,000,000. House rule: risk at most 1% of capital — \$10,000 — on any single name.

The stock trades at \$100. The base-case target is \$135. The invalidation (the stop) is \$89 — below that, the thesis is broken. So the **risk per share** is \$100 − \$89 = \$11, and the **reward per share** to the base case is \$135 − \$100 = \$35.

With a \$10,000 risk budget and \$11 of risk per share, the position is \$10,000 ÷ \$11 ≈ **909 shares**. That is roughly \$90,900 of notional — about 9% of the account — sized so that being wrong costs the predetermined \$10,000, not a penny more. The recommendation paragraph writes itself from those numbers: *"Buy 900 shares at \$100, \$10,000 at risk to \$89, base target \$135, reward-to-risk 3.2:1."*

The intuition: the size is not a feeling about how much you like the idea — it is the risk budget divided by the distance to the level that proves you wrong.

## Writing the executive summary last

Here is the single most useful piece of craft in this whole post, and it is counterintuitive: **write the summary last, even though it sits first.** The order in which you write a memo is the reverse of the order in which it is read.

![Memo written bottom-up from evidence to summary but read top-down from summary to detail](/imgs/blogs/the-investment-memo-arguing-your-view-on-paper-2.png)

Why does this matter? Because if you write the summary first, you write down the conclusion you *want* — the exciting version, the one that made you open the document — and then you spend the rest of the memo unconsciously cherry-picking evidence to justify it. The summary becomes a target you are working backward from, and the bear case becomes a formality you are obligated to dismiss. You have turned the memo into advocacy for a predetermined verdict. That is not analysis; it is a press release.

Write the body first and the summary becomes a *report* of what you actually found, not a pitch for what you hoped to find. You build the evidence. You construct the three scenarios and discover that the probability-weighted value is +30%, not the +80% your gut promised. You steelman the bear case and find that one leg of your thesis is weaker than you thought, so you trim the size. *Then* you write the summary — and it tells the truth, because by the time you write it, you already know the truth. The summary's job is to compress a conclusion you have honestly reached, not to manufacture one.

There is a second, subtler benefit. Writing the body first surfaces the spots where you are bluffing. When you sit down to write the evidence section and find you can only muster two real facts where you thought you had five, that is information. When you write the bear case and cannot rebut one of its legs, that is information. If you had written the summary first, you would have sailed past both, because the conclusion was already locked. Writing in the read-order lets the conclusion drive the evidence; writing in the build-order lets the evidence drive the conclusion. Only one of those is analysis.

A practical workflow: draft sections 4 through 9 first (evidence, scenarios, catalyst, risk/reward, bear case, invalidation), in roughly that order. Then go back to the top and write sections 1 through 3 (recommendation, thesis, variant perception) as a compression of what the body established. The body is where you do the thinking; the top is where you report it.

### Tailoring the memo to who reads it

The skeleton is fixed, but the emphasis shifts with the audience, and a memo that ignores its reader wastes its best ammunition. Write the same memo three ways for three readers and you will see what each section is really for.

For an **investment committee at a fund**, the reader is a peer who knows the market cold. You do not explain what a gross margin is; you assume it. The weight goes to the variant perception (they want to know precisely where you differ from the consensus they also follow) and to the bear case (they will attack it regardless, so you had better state it first). The financial case can be terse — a bridge and two comparables — because they can fill in the rest. The whole memo can be three pages, because the reader is fast.

For **partners or an allocation committee one level up**, the reader controls capital but may be less close to this specific name. Here the thesis and the catalyst carry more weight, because the reader needs the *why* and the *when* spelled out, not assumed. The risk/reward and sizing become central, because the question in their mind is portfolio-level: how does this fit, and how much can we lose? You write the same call, but you lead harder on the dollar risk and the fit, and you make the variant perception legible to someone who does not live in this sector.

For **your future self**, the reader is the most forgetful and the most emotional of the three. This version of the memo over-invests in the invalidation and monitoring sections, because future-you will be tempted to renegotiate them. It writes down not just *what* the level is but *why* you chose it, so that when the stock is sitting at \$90 and you are itching to move the stop to \$85, the memo can remind you in your own words why \$89 was the line. The future-self memo is the one that most resembles a contract, because it is one.

The mistake is to write one memo and forget who opens it. A committee memo handed to a generalist partner reads as jargon; a partner memo handed to the committee reads as condescending. The fastest way to lose a skeptical reader is to misjudge how much they already know — too little detail and they distrust your depth, too much and they distrust your judgment about their time. Decide who the primary reader is before you write the first word, and let that decision set the depth of every section.

#### Worked example: the bull / base / bear that decides the memo

The scenario section is the financial heart of the memo, and the worked example shows why you cannot pitch on the bull case alone. Same stock at \$100. Three scenarios, each with a dollar target and a probability that sums to 100%:

- **Bear (25% probability): \$70.** A new entrant prices aggressively, gross margin compresses to 18%, and the multiple derates. You lose \$30 per share.
- **Base (50% probability): \$135.** Margins recover to 40% as planned, the company holds share, the multiple is stable. You make \$35 per share.
- **Bull (25% probability): \$180.** Margins overshoot to 45%, the company takes share from the new entrant, and the multiple rerates higher. You make \$80 per share.

The number that decides the memo is the **probability-weighted value**:

```
PW = 0.25 × $70  +  0.50 × $135  +  0.25 × $180
   = $17.50      +  $67.50       +  $45.00
   = $130
```

The probability-weighted value is \$130 against a \$100 price — about **+30% expected upside**. That is the honest expected outcome, and it is far less thrilling than the \$180 bull case the excited version of you wanted to lead with. But +30% with a defined downside is a perfectly good trade. The memo is decided by the weighted average, not by the most exciting scenario in it.

![Bull base and bear price targets with the probability-weighted value above the current price](/imgs/blogs/the-investment-memo-arguing-your-view-on-paper-3.png)

The intuition: the bull case is the story you tell at the bar; the probability-weighted value is the number you actually bet on, and it is almost always lower than the bull case — which is exactly why you compute it.

## Pre-empting objections: writing for the skeptic

The defining mental posture of a memo writer is this: *every sentence is read by someone trying to break it.* Hold that posture and the memo writes itself differently. You stop making claims you cannot support, because you can feel the skeptic's pen. You add the number where you would have hand-waved. You pre-empt the obvious objection in the same paragraph that raises the point, so the reader never gets to feel clever for spotting a hole you already filled.

There is a useful technique here borrowed from debate, called **prebuttal**: raise the strongest objection to your own point *before* the reader does, and answer it on the spot. "You might worry that the margin recovery depends on the new factory ramping on schedule — and that the last two factory ramps slipped two quarters. Here is why this one is different: it is a copy of an existing line, not a new process, and the equipment is already installed and tested." Now the skeptic who was about to raise the factory-ramp risk finds you got there first, with a better version of their objection and an answer. You have spent your credibility well: a reader who sees you raise and answer the hard objections trusts you on the easy ones.

The opposite posture — writing to *impress* rather than to *survive* — produces the memos that die in committee. They are long, they are full of charts, they marshal an overwhelming volume of supporting detail, and they have one fatal property: they never state the strongest argument against themselves. A skeptic reading such a memo does exactly one thing — finds the unaddressed objection and asks it out loud. The author, who never wrote it down, has to improvise a rebuttal in real time in front of the people who control the capital. That is the worst place to think for the first time about why you might be wrong.

So the test for every memo, before you circulate it, is: *what is the one question that would most embarrass me if asked in the room, and have I already answered it on the page?* If the answer is no, the memo is not finished. The skeptic is not your enemy; the skeptic is the quality-control process you are running on yourself, externalized into a person. Write for them and the memo gets better. Write around them and it gets exposed.

## The financial case and the risk/reward section

The risk/reward section is where the memo stops being prose and becomes arithmetic. It answers a single question: *for every dollar I can lose, how many can I make, and therefore how big should this be?*

The two inputs are **reward** and **risk**, and the most common error is to measure them inconsistently. Reward is the distance from the entry to the base-case target — *not* the bull case. Risk is the distance from the entry to the **invalidation level**, the point at which you admit the thesis is broken and exit — *not* the bear-case scenario target. These are different numbers, and conflating them is how analysts talk themselves into bets that look better on paper than they are.

The reason risk is measured to invalidation, not to the bear target, is subtle and important. The bear *scenario* (\$70) is where the stock goes if the bear thesis fully plays out. The *invalidation level* (\$89) is where you get out. You do not ride the position down to \$70 — you exit at \$89, because by then the evidence says you are wrong. So your actual realized loss, if you are disciplined, is the distance to invalidation, not the distance to the bear target. The bear scenario informs the *probability* arithmetic; the invalidation level informs the *risk* arithmetic. Keep them separate.

![Risk reward ladder showing reward to the base target and risk to the invalidation level](/imgs/blogs/the-investment-memo-arguing-your-view-on-paper-5.png)

The reward-to-risk ratio is then simply reward ÷ risk. A ratio above roughly 2:1 is the rough floor most disciplined investors want before committing real size, because it lets you be wrong more often than right and still make money — the same expectancy logic we lean on in [position sizing](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) and that the technical-analysis series develops in [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies). The memo's risk/reward section makes this explicit so the committee can check it in one line.

#### Worked example: the full risk/reward section with the R:R ratio

Here is the section as it would appear in a real memo, with every number shown. Entry \$100, \$1,000,000 account, 1% max risk per name.

- **Reward** (entry → base target): \$135 − \$100 = **+\$35 per share.**
- **Risk** (entry → invalidation): \$100 − \$89 = **−\$11 per share.**
- **Reward-to-risk ratio:** \$35 ÷ \$11 = **3.2 : 1.**
- **Risk budget:** 1% of \$1,000,000 = **\$10,000.**
- **Position size:** \$10,000 ÷ \$11 ≈ **909 shares ≈ \$90,900 notional (≈ 9% of capital).**

Now stress it against the probabilities. The probability-weighted *gain* per share, using invalidation as the realized downside, is roughly: 0.50 × (+\$35) + 0.25 × (+\$80) + 0.25 × (−\$11) ≈ \$17.50 + \$20.00 − \$2.75 = **+\$34.75 per share** in expectation. Multiply by 909 shares and the expected dollar value of the position is about **+\$31,600** — for \$10,000 of defined risk. That is the memo's bottom line in a single number: a positive-expectancy bet at better than 3:1 reward-to-risk, sized so a full loss costs exactly the budgeted \$10,000.

The intuition: the risk/reward section is not decoration — it is the arithmetic that turns "I like this idea" into "I will buy 900 shares and risk \$10,000," which is the only form of a view you can actually be held to.

## The bear case, steelmanned then rebutted

If you take one technique from this post into your next memo, take this one. The bear case is where memos are won and lost, and it is the section that most cleanly separates a professional from an amateur.

A **steelman** is the opposite of a strawman. A strawman is a weak version of the opposing argument that is easy to knock down — and a skeptic spots it instantly and concludes you are either dishonest or have not thought hard enough. A steelman is the *strongest* version of the argument against you, stated so well that a short-seller reading it would nod and say "yes, that is exactly why I am short." Only after you have built the steelman do you rebut it.

![Strawman bear case rejected by the committee versus steelmanned bear case that strengthens the thesis](/imgs/blogs/the-investment-memo-arguing-your-view-on-paper-4.png)

Why steelman? Three reasons, in increasing order of importance.

**First, credibility.** A reader who sees you state the bear case better than they could have trusts your judgment. You have demonstrated that you understand the risk at least as well as they do, which means your rebuttal carries weight. A strawman does the reverse: it signals that you either do not understand the real risk or are hiding it, and either way the reader stops trusting the bullish parts of the memo.

**Second, pre-emption.** The strongest objection a skeptic can raise is one you did not address. By stating the steelman yourself, you take that weapon away. There is nothing left to ambush you with, because you already wrote down the worst thing anyone could say and answered it.

**Third — and this is the real reason — the steelman is how you find out whether you actually have a trade.** When you force yourself to write the strongest possible bear case, one of two things happens. Either you can rebut every leg of it with evidence, in which case your conviction goes *up*, because you have now survived your own best attack. Or you find a leg you cannot rebut — and that is the most valuable sentence in the whole memo, because it tells you to cut the size, tighten the invalidation, or pass entirely. The steelman is not a hoop to jump through. It is the adversarial proceeding that determines the size of the bet.

The mechanics: write the bear case as a numbered list of the two or three strongest reasons you are wrong, each stated at full strength. Then, under each, write the rebuttal — with evidence, not assertion. "Bear: a new entrant will compress margins to 18%. Rebuttal: the entrant has no distribution and the switching cost for existing customers is 14 months of integration work; in the last two industry entries, incumbents lost less than 3 points of share over three years — here is the data." If the rebuttal is "I don't think that will happen," it is not a rebuttal; it is a wish, and a skeptic will say so.

A useful source for the bear-case legs is to ask, for each pillar of your thesis, *what would have to be true for this pillar to be false, and how would I know?* If your thesis rests on margin recovery, the bear leg is "margins do not recover," and the evidence that would show it is the next two margin prints. If it rests on the catalyst arriving, the bear leg is "the catalyst slips or never comes," and the evidence is the contract calendar or the regulatory docket. Building the bear case this way — one leg per thesis pillar — guarantees you are attacking the load-bearing parts of your own argument rather than the cosmetic ones. The strongest bear case is not a different argument from your thesis; it is your thesis with each pillar negated.

One more discipline: rank the bear legs by how badly they would hurt and how likely they are. The leg that is both probable and devastating is the one that should drive your sizing, even if you have a partial rebuttal. A bear leg you cannot fully rebut and that is both likely and severe is not a footnote — it is the reason the position should be smaller than your enthusiasm wants. The bear case is, in the end, a sizing instrument: it tells you not just whether to take the trade but how much of it to take.

#### Worked example: the bear case that cut the size in half

This is the worked example that matters most, because it shows the steelman doing real work — changing the decision, not decorating it. Start with the same \$100 stock and the \$10,000 base position (909 shares, \$10k risk at the \$89 stop).

You write the steelmanned bear case and reach the third leg: *"The company's largest customer (22% of revenue) is up for contract renewal in nine months, and there are press reports it is evaluating the new entrant."* You try to rebut it. You can argue the switching cost is high. But you cannot find hard evidence the contract will renew — the renewal is genuinely uncertain, and it sits right inside your twelve-month horizon. This is not a leg you can knock down; it is a real, un-rebutted risk.

What does an honest memo do with that? It does not hide it (that would be dishonest and the skeptic would find it anyway). It *re-prices the probabilities.* You move the bear-case probability from 25% to 40% and the base from 50% to 40%, leaving the bull at 20%. Rerun the weighted value:

```
PW = 0.40 × $70  +  0.40 × $135  +  0.20 × $180
   = $28.00      +  $54.00       +  $36.00
   = $118
```

The expected value drops from \$130 to \$118 — from +30% to +18%. The reward-to-risk on the *probability-weighted* basis has fallen, and there is a binary contract event inside your horizon that you cannot handicap. The disciplined response is to **cut the size**: instead of risking the full \$10,000, you risk \$5,000 now (about 450 shares) and keep \$5,000 in reserve to add *after* the renewal resolves in your favor. You have used the steelman to convert an un-rebuttable risk into a smaller, stageable position.

The intuition: the bear case earned its place in the memo by changing the size of the bet — which is the test of whether a bear case is real. If writing it down does not move a single number, you wrote a disclaimer, not a bear case.

## Being explicit about what would make you wrong

A memo without an invalidation level is a memo that can never be wrong, which means it can never be scored, which means writing it taught you nothing. The invalidation section is short — often one or two sentences — but it is load-bearing, because it is the only part of the memo that commits you to a future action under conditions you have not yet faced.

There are two flavors of invalidation, and a good memo states both. **Price invalidation** is a level: "if the stock breaks \$89, I am out." It is mechanical and unambiguous, which is its virtue — it does not let you negotiate with yourself in the moment. **Thesis invalidation** is a fact: "if the next earnings print shows gross margin below 35%, the recovery thesis is dead regardless of the stock price." Thesis invalidation is closer to the truth of why you would exit, but it is slower and more subjective, so you pair it with the price level as a backstop.

The reason to write both *before* you enter is that you will be the worst possible judge of them *after* you enter. Once you own the position, every piece of news gets filtered through the lens of someone who wants to be right. A margin print of 34% becomes "a one-time inventory issue"; a break of \$89 becomes "an overreaction, a great chance to add." The whole point of writing invalidation in the memo, in advance, is to bind the calm, un-invested version of you who can still think clearly. This is the same execution-gap problem the technical-analysis series treats in [trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap): the plan is easy to make and hard to follow, so you make the plan binding before the emotion arrives.

The monitoring plan is the lighter cousin of invalidation. It is the short list of indicators you will check between now and the catalyst to know whether the thesis is tracking: the monthly margin data, the competitor's pricing, the customer-concentration disclosures, the level itself. The monitoring plan is what keeps the memo a living document rather than a thing you wrote once and filed. A view you do not track is not a view; it is a souvenir.

The best monitoring plans are written as *if-then* commitments, not vague intentions. "Watch the margins" is a wish; "if the Q2 print shows gross margin below 35%, re-read the bear case and consider cutting" is a plan. The difference is that the if-then version tells future-you exactly what observation triggers exactly what review, so you cannot drift past a warning sign by telling yourself you will "keep an eye on it." Tie each monitored indicator to the thesis pillar it tests and to the action it would trigger, and the monitoring plan becomes a thin, living extension of the invalidation section rather than a list of charts you glance at and ignore.

The interaction between the catalyst and the monitoring plan is where memos most often go quietly wrong. A thesis can be entirely correct on the destination and entirely wrong on the timing — the rerating happens, but eighteen months after your twelve-month horizon, by which point you have been stopped out or have given up. The monitoring plan is your early-warning system for a *slipping* catalyst: if the contract renewal that was supposed to land in Q3 gets pushed to Q1 next year, that is not necessarily a thesis-killer, but it is a horizon-killer, and the memo should have told you in advance what to do when the timeline stretches — hold, roll, or cut. A memo that names a catalyst but never says what to do if it is late has left out the most common way a good thesis loses money.

## Common misconceptions

**"The memo is a formality — the decision is really made in the meeting."** This is exactly backward, and believing it is how analysts get shredded. The memo *is* the thinking; the meeting is the audit of the thinking. If you treat the memo as paperwork to be completed after you have decided, you skip the one process — writing for a skeptic — that would have caught your errors. The analysts who get ambushed in committee are precisely the ones who wrote the memo as a formality. The good ones know the memo is where they argue with themselves, and the meeting is just where they find out if they argued well enough.

**"Lead with all the detail so they see how much work I did."** No. The reader does not award points for volume; they award capital for clarity. A committee member reads twenty memos a week. The one that leads with three pages of industry background before getting to the call gets skimmed, and the call gets missed. Lead with the recommendation. Put the detail where a skeptic who wants it can find it — in the evidence section, in an appendix — but never make the reader excavate the call. The work is supposed to be *behind* the memo, compressed into it, not poured out across it. A memo that needs to be long to be convincing is usually a thesis that is not actually convincing.

**"Hide or soften the weaknesses so the idea looks stronger."** This is the most tempting and the most fatal. Every memo has weaknesses; the only question is whether you surface them or the skeptic does. If you surface them — steelmanned, then rebutted — you control the framing and you build credibility. If you hide them, the skeptic finds them, and now you have two problems: the weakness itself, and the fact that you tried to hide it, which poisons trust in everything else you wrote. A hidden weakness is not a smaller weakness; it is a larger one, because it costs you the room's confidence on top of the actual risk.

**"Longer equals more convincing."** Length is correlated with weakness, not strength, past a certain point. The most convincing memos are dense, not long: every sentence does work, and the writer has done the hard labor of cutting everything that does not. A four-page memo that has been ruthlessly edited beats a twelve-page memo that includes everything the analyst knows. The skeptic's time is the scarcest resource in the room; spend it on the call, the variant perception, the bear case, and the numbers, and nothing else.

**"The bear case is where I list the risks."** A risk list is not a bear case. A risk list is a hedge against being blamed — "I told you so" insurance. A bear case is a *coherent argument that you are wrong*, made by a hypothetical adversary who is smart and motivated, and then *rebutted*. The difference is that a risk list never changes your decision, while a real bear case sometimes does — by cutting the size, tightening the stop, or killing the trade. If your bear case has never once changed a decision, you have been writing risk lists.

**"A precise probability is false precision, so I should not put a number on the scenarios."** This objection feels sophisticated and is exactly wrong. Yes, the 50% you assign to the base case is not knowable to the decimal. But refusing to write a number does not make you more honest — it makes you *un-auditable*, because a view with no probability cannot be scored, cannot be compared to the price, and cannot be falsified. The number is not a claim of precision; it is a forcing function. Writing "base case 50%" makes you ask whether you really believe it is a coin flip or whether you have been quietly assuming it is near-certain. The probability-weighted value is more honest than the qualitative version precisely because it exposes the assumptions you would otherwise smuggle in. Put the number down, label it as a judgment, and let the arithmetic check your story.

**"If the idea is good enough, the format does not matter."** Good ideas die in bad memos every week. The committee does not allocate to the idea in your head; it allocates to the argument on the page, and a great thesis buried under a missing variant perception, an absent bear case, and no invalidation reads, from the outside, exactly like a weak thesis. The format is not bureaucracy — it is the interface through which your idea has to pass to become a position. A brilliant idea that cannot survive the format is, for the purposes of getting capital, indistinguishable from a mediocre one. The discipline of the format is how you prove the idea is as good as you think it is.

## How it plays out in real markets

Memos are not a textbook nicety; they are how capital is actually allocated at every serious fund. Here is how the discipline shows up in practice.

**The risk/reward section that sized the bet.** In the run-up to the 2020 COVID crash and the recovery that followed, the funds that did well were not the ones with the best macro forecast — almost nobody forecast a 34% drawdown in five weeks followed by a V-shaped recovery. They were the ones whose memos had already specified, for each position, the reward to a base target and the risk to an invalidation level. When the market fell apart in March 2020, those analysts did not have to think; they had written down in advance where each thesis broke. Positions that hit invalidation were cut mechanically. Positions that were merely cheaper but whose thesis was intact were added to, because the memo had already said "below \$X with the thesis intact, add." The memo turned a panic into a checklist. The analysts who had only a vague bullish feeling, with no written invalidation, froze — and froze is the most expensive thing you can do in a crash.

**The bear case that strengthened the thesis.** Consider an analyst pitching a long in a bank stock in early 2023, just as the regional-bank crisis was breaking. The lazy memo would have buried the deposit-flight risk. The strong memo did the opposite: it stated the steelmanned bear case — *"uninsured deposits are 60% of the base and could flee in days, as they just did at the failed banks"* — at full strength, and then rebutted it with the specific evidence: this bank's deposit base was 70% insured and granular (no single depositor over a small threshold), its securities book was short-duration and marked, and its loan book was not concentrated in the sectors under stress. By stating the scariest version of the risk and then dismantling it with the bank's actual disclosures, the memo did not just survive the obvious objection — it converted the very thing everyone was afraid of into the reason to be long *this* bank rather than the index. The bear case, steelmanned, became the thesis.

**The macro memo that put a number on a surprise.** A memo is not only for single stocks; the same discipline tightens a macro view, where the temptation to hand-wave is even greater. Take an analyst writing a memo in the autumn of 2022, when inflation was the only story and every CPI print moved everything. The weak memo says "inflation is sticky, position for higher-for-longer." The strong memo does what the structure demands: it states what the curve has priced (a terminal Fed-funds rate near 5%), states the variant view (sticky services and a tight labor market argue for a terminal rate closer to 5.45%), and then puts the trade in dollar terms — short the front end, risk \$10,000 to an invalidation if the next core print comes in below 0.3% month-on-month or the 2-year yield falls 25 basis points. The bear case, steelmanned, is "a sharp growth scare collapses the front end before the inflation data matters," and the rebuttal is the labor data that says no recession is imminent. The memo converts a macro feeling into a sized, falsifiable position with a written kill level — which is the only form of a macro view that survives contact with the next data surprise. We treat the mechanics of those reactions in [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises); the memo is where you commit to a dollar bet on them.

**A memo that changed its own author's mind.** This is the quiet, common case, and it is the best argument for the whole practice. An analyst is excited about a long. He starts writing the memo, body first. The evidence section comes together fine. Then he gets to the bull/base/bear, computes the probability-weighted value, and finds it is +12%, not the +50% his gut promised — the bull case was doing all the emotional work and almost none of the probability mass. Then he writes the steelmanned bear case and cannot rebut the valuation leg: the stock was already pricing most of the recovery. By the time he reaches the recommendation paragraph, the honest call is *pass* — the upside is real but thin and the downside is not protected by a margin of safety. He never sends the memo. It did its job: it talked him out of a mediocre bet *before* the capital was at risk, in private, on paper, at a cost of two hours instead of a 15% drawdown. A memo that changes its author's mind is not a failed memo. It is the most valuable kind.

**The IC review that resized rather than rejected.** The memo does not end at your desk; it goes through committee, and the committee's job is to pressure-test it.

![Flow from a finished memo through committee review to approve resize or kill decisions](/imgs/blogs/the-investment-memo-arguing-your-view-on-paper-6.png)

A well-built memo usually does not get killed outright — the bad ideas die at the writing stage, talked out by their own authors. What survives to committee is mostly sound, so the committee's real value is in *resizing*. They attack the bear case and the financial case, and often they find one leg you under-weighted — a customer concentration, a covenant, a regulatory tail. The outcome is rarely a binary yes/no; it is "good idea, but the bear case is more live than your memo allows, so take half the size and add after the catalyst." That conversation is only possible because the memo gave them numbers to argue with. A memo with no scenarios and no invalidation gives the committee nothing to resize — they can only approve or reject a vibe, and most vibes get rejected.

## The playbook

Here is the repeatable process. Run it for every position large enough to matter.

**The memo skeleton (the nine sections, in read-order):**

1. **Recommendation + summary** — action, instrument, size, base-case return, R:R, horizon. Ten-second read.
2. **Thesis** — one or two sentences, with the *why*.
3. **What's priced in / variant perception** — consensus X, my view Y, the gap is Z.
4. **Evidence + analysis** — the 3-5 facts and the financial case.
5. **Bull / base / bear** — \$ targets, probabilities, probability-weighted value.
6. **Catalyst + timeline** — what closes the gap, by when.
7. **Risk / reward + sizing** — reward to base target, risk to invalidation, the R:R ratio, the share count.
8. **Bear case steelmanned, then rebutted** — strongest counter, answered with evidence.
9. **Invalidation + monitoring** — the price and the fact that prove you wrong; what you watch.

**The writing order (this is the discipline):**

- Write the **body first** — sections 4 through 9, roughly in that order. This is where you do the thinking.
- Write the **bear case at full strength** before you write the rebuttal. Steelman, then answer. If you cannot answer a leg, change the size — do not change the bear case.
- Write the **summary and recommendation last**, as a compression of what the body actually proved. The summary reports a conclusion you reached; it does not manufacture one.

**The pre-mortem cross-check (run before circulating):**

Before the memo leaves your desk, run a pre-mortem — the technique from [stress-testing your thesis with a pre-mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem). Assume it is twelve months later and the position lost money. Write down the three most likely reasons it failed. Then check the memo: is each of those three reasons addressed in the bear case and accounted for in the invalidation level? If a pre-mortem reason is *not* in the memo, you have found the hole the skeptic will find — go back and either rebut it or resize for it. The pre-mortem is the last quality gate: it asks "what did I leave out?" precisely when you most want to believe you left nothing out.

**The single test that decides whether the memo is real:** *did writing it change a number?* If the probability-weighted value came out lower than your gut, if the steelmanned bear case cut the size, if the pre-mortem made you tighten the stop — then the memo did its job. It is a thinking instrument, not a sales document. If you wrote four pages and not one number moved from where your enthusiasm started, you did not write a memo. You wrote a brochure, and the skeptic in the room will read it as one.

![Ten-box memo checklist covering the argument the numbers and the exit](/imgs/blogs/the-investment-memo-arguing-your-view-on-paper-7.png)

The memo is the most demanding form your view will ever take, and that is exactly why it is the most valuable. A view in your head can be vague; a view spoken in a meeting can be hand-waved; a view written as a one-page note can skip the hard parts. A memo cannot. It forces the variant perception into a sentence, the scenarios into numbers, the bear case into a steelman, and the risk into a dollar figure and a stop. By the time you have written one honestly, you know your own thesis better than the skeptic ever could — which is the whole point. You are not writing to win the room. You are writing to find out, before the room does, whether you deserve to.

## Further reading & cross-links

- [The Trade-Idea Note: A One-Page Template](/blog/trading/analyst-edge/the-trade-idea-note-a-one-page-template) — the lighter-weight sibling; when the stakes do not justify a full memo, this is the format.
- [Structuring a Thesis: Claim, Evidence, and Catalyst](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst) — the anatomy of the thesis the memo defends.
- [Base, Bull, and Bear: Building Three Scenarios](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios) — the scenario discipline behind the memo's targets.
- [Stress-Testing Your Thesis with a Pre-Mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem) — the cross-check you run before circulating.
- [Variant Perception: Where Real Edge Comes From](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from) — why the priced-in section is where the trade lives.
- [Catalysts and Timing: Why Cheap Can Stay Cheap for Years](/blog/trading/analyst-edge/catalysts-and-timing-why-cheap-can-stay-cheap-for-years) — the catalyst section, in depth.
- [What Would Change My Mind: Defining Invalidation Upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) — the invalidation discipline the memo commits to.
- [Trading Psychology and the Execution Gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap) — why you write invalidation before you enter, not after.
- [Position Sizing and the Kelly Criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — the math under the sizing section.
- [Expectancy: Why Win Rate Lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) — why a 3:1 reward-to-risk lets you be wrong more than half the time and still win.
- [The Surprise, Not the Level: Betas to Data Surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — the mechanics behind the macro memo's variant view on a data print.
