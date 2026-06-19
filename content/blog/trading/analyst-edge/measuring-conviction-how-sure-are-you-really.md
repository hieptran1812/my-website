---
title: "Measuring Conviction: How Sure Are You, Really?"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Conviction is the bridge from a view to a bet size — but most people confuse it with how much they like the trade. Learn to score conviction honestly from evidence, edge, and what is priced in, so it can drive size instead of emotion."
tags: ["analysis", "market-view", "conviction", "position-sizing", "base-rates", "variant-perception", "calibration", "risk-management", "expected-value", "process", "behavioral-bias", "decision-making"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Conviction is the dial that turns a view into a bet size, but most people set the dial by how much they *like* the trade. Real conviction is a measurement, not a mood: it is a function of evidence quality and independence, the strength of your variant perception, the base rate, how much is already priced in, and how many independent lenses agree. Score those inputs honestly and conviction can drive size; skip the scoring and your sizing is random noise dressed up as confidence.
>
> - Conviction is *not* confidence and it is *not* enthusiasm. Enthusiasm asks "how much do I like this?"; conviction asks "how sure does the evidence say I should be?" Two trades can feel identically certain and deserve sizes 4:1 apart.
> - Ungraded conviction produces random sizing. If "high conviction" is just "I really believe it," you will bet biggest exactly when you are most emotionally captured — which is when you are most likely wrong.
> - Build conviction from five graded inputs into a 0–100 score, map the score to a tier (low / medium / high), and map the tier to a dollar size. Re-score on every new piece of evidence; cut the number when you cannot name a falsifier.
> - The one rule to remember: **size the measured conviction, not the feeling — and high conviction never means bet huge, because ruin still binds.**

## Two trades that feel identical and aren't

A trader sits down on a Monday morning with two trades she is, by her own description, "100% sure" about. The first is a long position in a mid-cap industrial. She has spent three weeks on it: she pulled the last eight quarters of segment data, called two former employees who confirmed the new plant is ramping faster than the company guided, cross-checked the order book against a supplier whose shipments she can see in customs data, and noticed that sell-side estimates still model the old, slower ramp. Nobody is talking about this name. Her view is genuinely different from the consensus, and she has independent primary evidence that the consensus is stale. She *feels* very sure.

The second trade is a long position in a megacap AI name that everyone on her timeline is talking about. The thesis is "AI is the biggest secular trend of the decade and this company is the obvious winner." She has read every bullish thread, watched the CEO's keynote twice, and the chart looks like it wants to break out. She also *feels* very sure — arguably surer, because the energy around the trade is intoxicating and a dozen smart people are confirming it back to her every time she opens her phone.

Same feeling. The internal sensation of certainty is, if anything, stronger on the second trade. And yet these two views could not be more different as objects, and they should be sized perhaps four to one apart — the industrial four times larger than the AI name — because honest conviction in the first is genuinely high and in the second is, on inspection, low. The feeling is lying to her about which is which. The entire job of this post is to give you the instrument that does not lie.

![Five graded inputs feed a conviction score that maps to a tier and then to a dollar bet size](/imgs/blogs/measuring-conviction-how-sure-are-you-really-1.png)

Conviction is the single most abused word in the business. "High conviction" gets stamped on whatever a trader most wants to do, and then used to justify an oversized bet, and then — when the bet fails — the post-mortem says "the conviction was there, the market was just wrong," which learns nothing. The problem is that conviction is doing two jobs at once. It is supposed to be the *output* of an honest assessment of how strong your case is, and it is the *input* to how large you bet. When the assessment is replaced by a feeling, the dial that controls your risk gets set by your mood. This post separates the two jobs, defines conviction precisely as a measurement, gives you five inputs to score it from, and shows — with dollars on a real account — how the same beloved trades get sized completely differently once you grade them honestly.

## Foundations: conviction, confidence, and enthusiasm are three different things

Before any technique, we have to be ruthless about three words people use interchangeably and shouldn't: **enthusiasm**, **confidence**, and **conviction**. They are not synonyms, and conflating them is the original sin that makes sizing random.

**Enthusiasm** is how much you *like* a trade — the emotional pull toward it. It is generated by narrative excitement, recency (the trade that just worked feels great), social proof (other people you respect are in it), and the sheer fun of a clean story. Enthusiasm is real and it is information about *you*, but it is not information about *the trade*. A trade can be thrilling and terrible, or boring and excellent. Enthusiasm correlates with how vivid and available the story is, which is almost orthogonal to whether the position has edge.

**Confidence** is how *certain you feel* that you are right — your subjective sense of probability. Confidence is a number your gut is already producing whether you write it down or not: "I'm pretty sure," "I'd be shocked if this didn't work," "it's basically a lock." The catch is that felt confidence is systematically miscalibrated. People are routinely 90% *sure* of things that happen 70% of the time. Confidence is the feeling of probability; it is not the probability. We have a whole sibling treatment of this in [thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions) — the short version is that the feeling and the frequency come apart, badly, exactly when you are most invested.

**Conviction** is the one we are building. Conviction is a *graded assessment of how strong your case actually is* — built from the quality and independence of your evidence, whether your view is both differentiated and likely right, what the base rate says, how much is already priced in, and how many independent lenses agree. Conviction is meant to be the disciplined, written-down answer to "how sure does the *evidence* say I should be?" — as distinct from how sure I *feel* (confidence) or how much I *want it* (enthusiasm). When it is done honestly, conviction can be trusted to set bet size. When it is just confidence wearing a lab coat, it can't.

Here is the relationship that makes the distinction matter, and it is the spine of the whole post: **conviction is the bridge from a view to a bet size.** You form a view (the market is mispricing this industrial). You quantify how strong the case is (conviction). You translate that strength into how much capital you put at risk (size). If the middle step is corrupted — if "conviction" is really "enthusiasm" — then your sizing is driven by your mood, which means you will, with mechanical reliability, bet *biggest* on the trades you are most emotionally captured by. And emotional capture is precisely the condition under which you are most likely to be wrong, because capture suppresses the search for disconfirming evidence. Ungraded conviction doesn't just produce random sizing; it produces *anti-correlated* sizing, where the position is largest exactly when the edge is smallest.

### Calibrated conviction: the number has to mean something

A conviction score is only useful if a "high conviction" trade actually works more often, or pays off bigger, than a "low conviction" one. That property is called **calibration**: across many trades, the ones you graded high should resolve favorably more often than the ones you graded low. If your "high conviction" trades and your "low conviction" trades win at the same rate, your conviction scale is decorative — it is sorting trades by enthusiasm, not by edge. We treat calibration as its own discipline in a forward post, [calibration: keeping score on your own forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts); for now, hold the standard: a conviction scale earns the right to drive size only if, when you look back over fifty trades, the high-conviction bucket genuinely outperformed the low-conviction bucket. Until you have that evidence, treat your scale as provisional and size conservatively.

### Why ungraded conviction produces random sizing

There is a concrete failure mode worth naming. Suppose you size "by feel" — big when you feel sure, small when you don't. Your felt-sureness is driven mostly by recency and narrative (the two loudest emotional inputs). So you will put on your largest positions right after a winning streak (recency makes you feel invincible) and in the trades with the hottest stories (narrative makes you feel sure). Both of those are conditions associated with *worse* forward returns, not better. The streak mean-reverts; the hot narrative is already priced. Feel-based sizing therefore loads your largest bets onto your worst-expectancy trades. This is not a small inefficiency. Sizing is a multiplier on edge — get it backwards and you can turn a positive-edge process into a losing one. The fix is to refuse to let the feeling set the dial, and instead grade the inputs that actually predict the outcome.

## The five inputs to honest conviction

Honest conviction is not one number you introspect; it is a sum of *graded inputs* you assess separately. Separating them is the whole trick, because it forces you to confront the input you are weakest on instead of letting a strong feeling about one input flood the others. Here are the five. Each gets a max score; we will assemble them into a scorecard in the next section.

**1. Evidence quality and independence.** Not "do I have evidence" but "what *kind* of evidence, and is it independent?" Primary evidence you gathered yourself (channel checks, the actual filing, customs data, your own model of unit economics) is worth far more than secondary evidence (a sell-side note, a bullish thread, a headline). And *independent* pieces of evidence — ones that could each be true or false separately — stack into real conviction, while a dozen restatements of the same underlying claim are *one* piece of evidence wearing twelve coats. The classic trap: you read ten bullish articles and feel ten times more sure, but all ten trace back to the same company press release. That is one fact, not ten. The question to score is: how many *genuinely independent, primary* pieces of evidence support the thesis?

**2. Strength of the variant perception.** Edge comes from being *differentiated and correct* — believing something the market doesn't, and being right. We develop this fully in [variant perception: where real edge comes from](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from); here it is an input to conviction. A view that is both consensus and correct earns you nothing — it is already in the price. A view that is differentiated but wrong loses you money. Only the differentiated-and-correct quadrant has edge, and conviction should rise with how confident you are that you sit there. The question to score: in what specific way is my view different from consensus, and what is my evidence that *I* am the one who is right and the crowd is wrong?

**3. Base rate / the outside view.** Before you grade your specific case, ask what *usually* happens in setups like this. If you are betting on a turnaround, the base rate for turnarounds is grim; if you are betting a hyped IPO keeps rising, the base rate is poor; if you are betting on mean reversion in a range that has held for years, the base rate is decent. The base rate is the "outside view" — the statistics of the reference class — and it should anchor conviction *before* the seductive specifics of your case (the "inside view") pull it up. Most narrative-driven conviction fails precisely here: the story is so vivid that the base rate gets ignored. The question to score: does the history of setups like this back my thesis or fight it?

**4. How much is already priced in.** Conviction in your *view* is not the same as conviction in your *trade*, because the trade only pays if your view is more right than what is already in the price. This is the spine of the entire series — what is priced in versus what I believe — and it has its own posts, including [what's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) and an outside treatment of [consensus expectations and being priced in](/blog/trading/event-trading/consensus-expectations-and-priced-in). For conviction, the key move is: a view that everyone already shares and that is fully reflected in the price deserves *low* conviction *as a trade*, no matter how confidently you hold it, because there is no gap left to capture. The question to score: how much headroom is there between the consensus expectation and my view — is the gap wide open or mostly closed?

**5. Breadth of lens agreement.** When you read a market through multiple independent lenses — macro, positioning, valuation, fundamentals, technicals, sentiment — conviction should rise when *independent* lenses point the same way and fall when they conflict. We cover the discipline of handling disagreement in [reconciling conflicting signals when the lenses disagree](/blog/trading/analyst-edge/reconciling-conflicting-signals-when-the-lenses-disagree). The crucial qualifier is *independent*: three lenses that are really the same lens (price momentum, RSI, and "the chart looks strong" are all the price) count as one, not three. The question to score: how many genuinely independent lenses agree, and are any of them screaming the opposite?

Notice what is *not* on this list: enthusiasm, recency, social proof, how clean the story is, how much you'd enjoy being right. None of the actual inputs to conviction are feelings. That is the point of the figure that opened this post — five graded inputs feed the score, and the feeling is explicitly excluded.

### Turning each input into a number you can actually write down

"Score evidence quality out of 25" is useless if you have no idea what a 7 versus a 20 looks like. So each input needs a rough rubric — anchors you can point your specific case at. These are not laws; they are scaffolding to keep your scoring honest and roughly consistent across trades, which is the property that lets you compare conviction between ideas at all.

For **evidence quality (max 25)**, anchor on the *kind* and *independence* of what you have. Zero to five points: the evidence is a headline, a thread, or a single sell-side note — secondary, and easily traced to one source. Six to twelve: you have a real fundamental case built from public filings and your own model, but no proprietary or primary work. Thirteen to nineteen: you have one genuinely primary, independent source — a channel check, alternative data, a satellite count, a conversation with someone close to the operation. Twenty to twenty-five: you have two or more *independent* primary sources that each corroborate the thesis and could each have come back negative. The jump from "I read a lot about it" to "I gathered evidence the consensus doesn't have" is the jump from single digits to the high teens, and it is the single biggest determinant of whether conviction is real.

For **variant perception (max 25)**, anchor on how cleanly you can state the *delta* between your view and consensus, and your evidence that the delta resolves your way. Zero to five: you cannot articulate a specific difference from consensus — your view *is* the consensus, you just hold it strongly. Six to twelve: you have a vague sense the crowd is "too bearish" or "too bullish" but no specific, falsifiable claim about where they are wrong. Thirteen to nineteen: you can name the exact number or fact the consensus has wrong (estimates model an 8% growth rate; you think it's 14% and here's why). Twenty to twenty-five: you can name the specific error *and* explain the structural reason the crowd persists in it (sell-side hasn't updated since the last guide; the data they'd need isn't in their workflow). A variant view you can state as a precise, falsifiable delta scores high; a "vibe that everyone's wrong" scores low.

For **base rate (max 20)**, anchor on whether the reference class supports the trade. Zero to five: the base rate actively fights you (turnarounds, buying hyped IPOs at the top, fading a strong trend with no catalyst). Six to twelve: the base rate is neutral or you genuinely don't know the reference class. Thirteen to twenty: the base rate is on your side (mean reversion in a long-stable range, a supply shock reversing, a confirmed operational improvement flowing to earnings). The discipline here is to compute the base rate *before* you fall in love with the specifics, because once the inside-view story is vivid, you will rationalize away an unfavorable base rate every time.

For **priced-in headroom (max 20)**, anchor on the size of the gap between consensus expectation and your view. Zero to five: the move is essentially fully priced — your view *is* the consensus or close to it, so there is little left to capture. Six to twelve: partially priced — the market has started to move toward your view, so some but not all of the gap remains. Thirteen to twenty: wide open — consensus is positioned the other way or has not woken up to your thesis, so the full gap is available. The trap is to score this off your view's *correctness* rather than its *differentiation from the price*; a correct view that everyone shares scores near zero here, and that is exactly right.

For **lens agreement (max 10)**, anchor on the count of *independent* confirmations. Zero to two: one lens supports you and at least one independent lens screams the opposite. Three to six: two independent lenses agree and none strongly opposes. Seven to ten: three or more genuinely independent lenses agree. The cap is low and deliberate — lenses are the easiest input to inflate by double-counting, so it can never dominate the score.

The reason to write a number *and a one-line justification* for each input is that the justification is where self-deception gets caught. "Evidence quality: 22 — three primary independent sources" survives scrutiny; "evidence quality: 22 — I've read a ton about this" does not, and the act of writing the justification exposes the difference to you before you size the trade on it.

### Building a conviction score and tiers

Now we assemble the five inputs into one number. The exact weights are a judgment call and you should tune them to your own track record, but a defensible starting allocation out of 100 points is: evidence quality 25, variant perception 25, base rate 20, priced-in headroom 20, lens agreement 10. Evidence and variant perception carry the most weight because they are where real edge lives; lens agreement carries the least because lenses are easy to double-count.

![The conviction scorecard grades five inputs and sums to a score out of one hundred](/imgs/blogs/measuring-conviction-how-sure-are-you-really-2.png)

Then you collapse the score into **tiers**, because false precision in the number is its own trap — there is no real difference between a 62 and a 65, and pretending there is invites you to fiddle the inputs to clear a threshold. Three tiers are enough:

- **Low conviction (35–49):** the thesis is plausible but the case is thin — recycled evidence, a weak variant view, an indifferent base rate, or most of the move already priced. Trade it small, if at all.
- **Medium conviction (50–69):** a solid case with real independent evidence and a genuine gap to consensus, but with a meaningful weakness in at least one input. The bread-and-butter tier.
- **High conviction (70–100):** strong on evidence *and* variant perception *and* base rate *and* headroom, with multiple independent lenses agreeing. These are rare. If half your trades are "high conviction," your scale is broken.

And a floor: **below 35, there is no trade.** A score that low means you do not actually have a differentiated, evidenced view with room to be right — you have a hunch. The discipline of a floor is what stops you from "low-conviction-ing" your way into a portfolio of weak bets that collectively bleed you.

#### Worked example: scoring a trade's inputs into a tier and a dollar size

Take the mid-cap industrial from the opening, on a \$100,000 account. We score each input honestly:

- **Evidence quality (max 25):** primary channel checks (two ex-employees, customs data, supplier shipments) plus the observation that estimates are stale. Three genuinely independent primary sources. Score **21**.
- **Variant perception (max 25):** the view (faster ramp than guided) is clearly differentiated — nobody is modeling it — and the evidence that she is right is strong. Score **20**.
- **Base rate (max 20):** capacity ramps that are already shipping (confirmed by customs data) tend to flow through to earnings; the base rate is supportive but not a lock. Score **14**.
- **Priced-in headroom (max 20):** sell-side still models the old ramp, so the gap between consensus and her view is wide open. Score **16**.
- **Lens agreement (max 10):** fundamentals and supply-chain data agree; valuation is neutral; technicals are quiet; no lens screams the opposite. Call it three independent confirmations. Score **8**.

Total: 21 + 20 + 14 + 16 + 8 = **79 → High conviction.** On a \$100,000 account where High tier risks 4% of capital, that is a **\$4,000** risk budget on this position. The score did not come from how excited she was; it came from grading five inputs, and the high score is *earned* by primary evidence and a real, un-priced variant view. The intuition: a 79 is high because four of five inputs are genuinely strong, not because the trade feels good.

## Separating conviction from recency and narrative

The two largest sources of *fake* conviction are recency and narrative, and they deserve direct treatment because they are what the scorecard exists to defend against.

**Recency** is the tendency to over-weight what just happened. After a string of winners, every new idea feels like a high-conviction layup; after a string of losers, even excellent setups feel doubtful. The feeling of conviction is being driven by your recent P&L, which has nothing to do with the merits of the *next* trade. The defense is mechanical: the scorecard contains no input for "how I've been doing lately." If your gut says 80 but the scorecard says 50, the gap is almost always recency or narrative inflating the gut, and the scorecard wins. The execution-side cousin of this — letting your emotional state drive the trade — is covered in [trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap).

**Narrative** is the seductive story. A clean, vivid, causally satisfying story ("rates are falling, so growth stocks rip, and this is the cleanest growth story") generates intense felt confidence because the brain rewards coherent narratives with a feeling of understanding. But narrative coherence is not evidence, and the most-repeated narratives are the most-priced — by the time a story is clean enough to be intoxicating, it is usually in the price. The scorecard defends against this on two fronts: the *priced-in* input directly penalizes consensus narratives, and the *evidence independence* input catches the "ten articles, one press release" trap.

![Enthusiasm is the feeling while conviction is the graded measurement of the same trade](/imgs/blogs/measuring-conviction-how-sure-are-you-really-4.png)

The before-and-after framing above is the whole discipline in one picture: the left column is enthusiasm — "I love this," "everyone smart is saying it," "the chart looks ready," "feels like an 80% sure thing." The right column is the same trade graded against the inputs that matter, and it collapses to a Low tier. The feeling and the measurement disagree, and you size the measurement.

#### Worked example: narrative conviction cut from 80% to 35% by base rates

Return to the megacap AI trade, on the same \$100,000 account. The gut says 80% — it *feels* like a near-lock. Now grade it:

- **Evidence quality (max 25):** the "evidence" is bullish threads, the CEO keynote, and the chart. All secondary, all tracing back to the same consensus narrative. Effectively one non-independent source. Score **8**.
- **Variant perception (max 25):** the view ("AI leader keeps winning") is *consensus*, not variant. There is no specific way her view differs from what everyone already believes. Score **5**.
- **Base rate (max 20):** the base rate for "buy the most-hyped megacap at all-time highs because the story is great" is mediocre; crowded momentum at extremes mean-reverts more than the narrative admits. Score **7**.
- **Priced-in headroom (max 20):** the story is maximally consensus and maximally priced. The gap between her view and the price is nearly closed. Score **5**.
- **Lens agreement (max 10):** technicals and sentiment agree, but they are not independent (both are "the crowd likes it"); positioning is stretched (a lens screaming the *opposite*); valuation is rich. Score **2**.

Total: 8 + 5 + 7 + 5 + 2 = **27 → below the 35 floor.** The gut's 80% becomes a measured 27, which means *no trade* by the floor rule — or, if she insists on expressing it, the smallest possible size. Compare the dollars: at the Low tier she might have risked 0.5% (\$500); the scorecard says even that is generous, versus the \$4,000 the industrial earned. The narrative trade *felt* surer than the industrial and scored a third as high. The intuition: the base rate and priced-in inputs are designed to deflate exactly the trades that feel best, because feeling-best usually means most-crowded.

![Base rates and priced-in cut narrative conviction while evidence-backed conviction survives the cut](/imgs/blogs/measuring-conviction-how-sure-are-you-really-6.png)

The chart above makes the asymmetry vivid: the narrative trade's conviction craters from a gut 80% to a base-rate-anchored 35% (and the dollar size collapses with it), while the evidence-backed trade barely moves from gut 75% to a measured 70%, because its conviction was built on independent evidence that survives contact with the base rate. *That gap is the entire value of measuring conviction instead of feeling it.*

### Why conviction inflates, mechanically

It helps to understand *why* felt conviction runs ahead of measured conviction, because the mechanisms are predictable, which means they are defensible. There are four of them, and each maps to a specific defense in the scorecard.

The first is **confirmation search.** Once you like a trade, you go looking for reasons it's right, and you find them — the market is enormous and you can always assemble a wall of confirming articles. Each one bumps your felt confidence even though, as evidence, they are mostly restatements of one another. The scorecard's *evidence independence* requirement is the antidote: you are forced to ask not "how much supports this?" but "how many *separable* things support this?", and the wall of confirmation collapses to one or two real data points.

The second is **the availability cascade.** A claim repeated by many sources feels true in proportion to how often you've heard it, not how well it's supported — repetition masquerades as evidence. This is why the most-discussed narratives generate the most felt conviction and, simultaneously, the least edge: by the time a story is everywhere, it is both maximally available (so it feels certain) and maximally priced (so there's nothing left). The *priced-in* input directly cancels this — the more available and consensus the story, the lower the headroom score.

The third is **narrative coherence as a confidence signal.** A story with a clean cause-and-effect chain produces a strong feeling of understanding, and the brain reads "I understand this" as "this is likely true." But coherence and truth are different properties; a coherent story can be entirely wrong, and a true situation can be messy and incoherent. The *variant perception* input defends here by demanding not a coherent story but a *specific, falsifiable delta* from consensus — and most intoxicating narratives, when you try to state the precise delta, turn out to be consensus dressed in vivid language.

The fourth is **emotional and financial commitment.** The moment you put the trade on, you are committed, and commitment retroactively raises felt conviction to reduce the dissonance of holding a position you're unsure about. This is why conviction tends to *rise* after you trade, exactly backwards from the Bayesian ideal where it should only move on new evidence. The defense is the pre-committed *falsifier* and the discipline of re-scoring on evidence rather than on how the position feels now.

#### Worked example: two equally-loved trades sized four to one

Here is the opening dilemma made fully numeric, on the \$100,000 account. The trader loves both trades identically — both feel like 100% locks. She grades both:

The **industrial** scores, as computed earlier: evidence 21, variant 20, base rate 14, priced-in 16, lens 8 = **79, High tier → 4% risk → \$4,000.**

The **megacap AI name** scores: evidence 8, variant 5, base rate 7, priced-in 5, lens 2 = **27, below the floor.** If she insists on expressing it at all, she rounds it up to a token Low-tier toe-hold — say 1% → **\$1,000** — but the scorecard's honest verdict is no trade.

If we are generous to the AI trade and let it stand at the bottom of the Low tier at \$1,000, the two equally-loved trades are sized **\$4,000 versus \$1,000 — exactly four to one** — purely because one is built on independent primary evidence and a real variant view, and the other on a hot narrative that is already in the price. The feeling said *equal*; the measurement said *4:1*. And note which direction the feeling erred: it inflated the *worse* trade most, because the AI name had the more available, more coherent, more socially-confirmed narrative. The intuition: felt conviction is loudest exactly where measured conviction is weakest, so sizing by feeling reliably over-bets the trades you should under-bet.

## Your conviction versus the market's conviction

There is a second, subtler layer: your conviction is not the only conviction in the room. The market itself has a conviction — expressed through positioning, options skew, and how violently it reacts to news — and the *interaction* between your conviction and the market's is where the real opportunity (and the real danger) lives.

When the market has *low* conviction (light positioning, wide range of views, fat options premium for uncertainty) and you have *high, well-evidenced* conviction, that is the dream setup: you are differentiated, there is room for the price to move toward your view, and you are not fighting a crowded consensus. When the market has *high* conviction (everyone positioned the same way) and you agree, you are late — the move is priced and the risk is a violent unwind if the consensus is wrong. When the market has high conviction and you *disagree* with strong evidence, you have a potential pain trade in your favor, but you must respect that a crowded consensus can stay wrong longer than you can stay solvent.

This is why *positioning* is a conviction input in disguise. Crowded positioning means the market's conviction is high and the trade is priced; it should *lower* your conviction-as-a-trade even if it doesn't change your conviction-in-the-view. The mechanics of crowding and the violent reversals it produces are covered in [positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade) and from the dealer's side in [how an options market maker thinks: the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade). The discipline: always ask "what is the market's conviction here, and is mine differentiated from it?" A high personal conviction that merely *matches* a high market conviction is not edge — it is consensus you happen to feel strongly about.

### Lens agreement, properly counted

Because lens agreement is the input people most over-credit, it is worth a closer look. The naive view is "more lenses agreeing = more conviction, linearly." The honest view is that agreement among *independent* lenses adds conviction with *diminishing returns*, and agreement among *correlated* lenses adds almost nothing.

![Lens agreement raises conviction with diminishing returns as independent lenses confirm](/imgs/blogs/measuring-conviction-how-sure-are-you-really-5.png)

The first independent confirmation — say, fundamentals after you already had supply-chain data — adds a lot, because it is a genuinely separate way of being right. The second adds less. By the fifth and sixth, you are mostly adding lenses that are correlated with ones you already counted, so they add little. The danger is mistaking *many* lenses for *many independent* lenses: a stock where price momentum, RSI, MACD, and "the chart looks great" all agree has *one* lens (price) agreeing four times, which the [the indicator trap](/blog/trading/technical-analysis/the-indicator-trap) post dissects in detail. Score lens agreement on the count of *independent* confirmations, and cap its contribution — which is exactly why it carries only 10 of the 100 points.

## Updating conviction as evidence arrives

Conviction is not set once and frozen. It is a *running estimate* that should move as evidence arrives — up when independent confirmation stacks, down when a key support fails or a falsifier triggers. This is the same Bayesian habit that underlies probabilistic thinking: start with a prior (your initial score), and update it with each new, independent piece of evidence.

The two failure modes are symmetric and both deadly. The first is *anchoring* — refusing to lower conviction when disconfirming evidence arrives, because you have already committed (psychologically and financially) to the trade. The second is *over-reacting* — slashing conviction on noise that isn't actually evidence (a single down day, a hostile tweet). The discipline that separates the two is the **falsifier**: before you put the trade on, you write down the specific fact that would *cut your conviction in half*. When real evidence arrives, you ask "is this the falsifier, or is it noise?" If it is the falsifier, you cut the score and the size hard; if it is noise, you hold. This is exactly the pre-trade discipline of writing down what would change your mind, which we build out in [stress-testing your thesis with a pre-mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem).

#### Worked example: rising conviction as independent evidence stacks

The industrial trade started at a score of 79 (High), \$4,000 risk on the \$100,000 account. Over the next two weeks, two independent pieces of evidence arrive:

- The company pre-announces a capacity expansion, *independently* confirming the faster ramp her channel checks suggested. This is genuinely new and independent — it is not a restatement of what she already knew. Evidence quality rises from 21 to 24.
- A sell-side analyst quietly raises numbers toward her view. This narrows the priced-in gap (the market is catching up), so priced-in headroom *falls* from 16 to 11 — the trade is becoming more consensus.

Net: evidence +3, headroom −5. New score = 79 + 3 − 5 = **77**, still High tier, so size holds near \$4,000 — but the *composition* has shifted. The view is more confirmed but more priced, which is the natural arc of a winning trade: conviction-in-the-view rises while conviction-as-a-trade slowly erodes as the gap closes. When the headroom input eventually drops enough to pull the total below 70, that is the signal to take the tier down and trim, *before* the thesis is fully consensus. The intuition: rising confirmation and falling headroom partly offset, and tracking both is what tells you when a great view has become a crowded trade.

#### Worked example: conviction tier mapped to a Kelly-fraction size

The cleanest way to make conviction drive size is to map the tier to a *fraction of full Kelly*, the bet size that maximizes long-run growth given your edge — covered mechanically in [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). Full Kelly is famously too aggressive (it tolerates gut-wrenching drawdowns and is brutally sensitive to estimation error), so professionals bet a *fraction* of it. Conviction sets the fraction.

Suppose your edge on a Medium-conviction trade implies a full-Kelly bet of 8% of capital. You do not bet 8%. You map:

- **Low tier → 1/8 Kelly** = 1% of capital = **\$1,000** on the \$100,000 account.
- **Medium tier → 1/4 Kelly** = 2% = **\$2,000**.
- **High tier → 1/2 Kelly** = 4% = **\$4,000**.

The High-conviction industrial gets \$4,000; an otherwise-identical Medium-conviction trade gets \$2,000; the Low-conviction leftovers get \$1,000. The conviction tier is literally the Kelly fraction, and the fraction caps your downside *even at high conviction* — which is the answer to the "high conviction means bet huge" misconception we hit next. The intuition: conviction scales the bet *within* a hard ceiling, so a great idea gets four times the size of a weak one but still cannot blow you up.

![Conviction tiers map to a percentage of a one hundred thousand dollar account](/imgs/blogs/measuring-conviction-how-sure-are-you-really-3.png)

The mapping above is illustrative — your own tier-to-size table depends on your edge, your account, and your risk tolerance — but the shape is universal: a hard floor (below 35, no trade), a modest Low size, a workhorse Medium size, and a capped High size that never approaches "bet the farm." We build the full conviction-to-size translation in the forward post [from conviction to size: the bet-sizing bridge](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge); here the point is just that the tier *determines* the size mechanically, so the dial is set by measurement, not mood.

### Conviction decays — and stale conviction is expensive

One asymmetry deserves its own emphasis: conviction has a *half-life*. Each of the five inputs ages, and most of them age against you. The priced-in headroom shrinks as the market discovers what you discovered — every analyst who upgrades, every fund that buys, closes a little more of your gap. Your variant view stops being variant the moment the consensus moves toward it. Your evidence goes stale as conditions change. Only the base rate is roughly constant. So a trade you scored 75 and put on at High tier may, three months later, genuinely be a 60 — Medium — not because anything went *wrong*, but because the edge has been partly harvested by the market catching up. The danger is that your *felt* conviction does the opposite: as the trade works and you grow attached, the feeling climbs even as the measured edge erodes, so you hold (or add) exactly as the real edge drains away.

This is why re-scoring is not optional housekeeping; it is risk management. A position sized to a stale 75 when the honest current score is 60 is over-sized relative to its remaining edge, and the excess size is pure uncompensated risk. The professional habit is to re-score on a cadence (say, monthly, plus on any material news), and to treat a *downward* tier change as a trim trigger — not a failure, just the natural retirement of an edge that has been mostly captured. Trimming a winner because its conviction has decayed feels wrong (the trade is working, why cut it?), which is exactly why most people don't do it, and exactly why doing it is an edge. You are selling the part of the position whose edge is gone while it still looks like a winner, instead of round-tripping it when the gap finally closes to zero and the trade reverses.

There is a mirror-image error worth naming: *refusing* to let conviction decay when the thesis has actually been confirmed and the gap has closed entirely. At that point conviction-in-the-view might be near-certain — you were right — but conviction-as-a-trade should be near zero, because there is no headroom left. The trade has done its job. Holding it past that point is not conviction; it is inertia, and it is how a brilliant call turns into a flat or losing position as the priced-in input goes to zero and then negative (the trade becomes the new consensus that the next variant view will fade).

## Common misconceptions

**"Conviction equals certainty."** No. Certainty is a feeling of being right; conviction is a graded measure of how strong your case is. You can have high conviction and still expect to be wrong a meaningful fraction of the time — a 70-point conviction is not a claim that you will win 100% of the time, it is a claim that your case is strong across the five inputs. Anyone who treats conviction as certainty will be devastated by the losses that high-conviction trades inevitably take, and will abandon a perfectly good process after a normal losing streak. Conviction lives in probability space; certainty does not exist in markets.

**"If I love it, it's high conviction."** This is the original sin, and the whole scorecard exists to break it. How much you love a trade is enthusiasm, and enthusiasm is generated by narrative and recency — both of which are *negatively* associated with forward returns at the extremes. The trades you love most are disproportionately the most-crowded, most-priced, most-narrative-driven ones. If you let love set conviction, you will systematically over-size your worst-expectancy trades. Love is a signal to *check the scorecard especially carefully*, not a substitute for it.

**"Conviction is a feeling you can't measure."** It is true that your *felt* confidence is a feeling — but conviction-as-we-define-it is explicitly *not* the feeling; it is the score from grading five concrete, answerable inputs. "Is my evidence primary and independent?" is a checkable question. "Is my view differentiated from consensus?" is checkable. "What does the base rate say?" is lookup-able. "How much is priced in?" is estimable. "How many independent lenses agree?" is countable. None of those require introspecting a vibe. The claim that conviction can't be measured is usually a defense of the right to size by feel — and it is exactly the claim this post is written to refute.

**"High conviction means bet huge."** Ruin still binds. Even a perfectly-scored 95-conviction trade can lose — markets are probabilistic, and a 95 is not a 100. If you bet your account on every high-conviction idea, you will eventually hit a high-conviction loser (you will hit several), and a single ruinous loss ends the game regardless of how good the average idea was. This is why the tier-to-size map *caps* the High tier (at, say, 4% or half-Kelly), not "as much as you can." Conviction scales size *within* a survival constraint; it never overrides it. The asymmetry of losses — why a 50% loss needs a 100% gain to recover — is the reason the cap is non-negotiable, and it is covered in [the asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) and the broader survival argument in [risk management: the only free lunch](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine).

**"High win-rate trades are high conviction."** A common conflation: "I'm sure this wins" gets read as "this has high expected value." But win rate is not edge — a trade that wins 90% of the time and loses 10x its gain in the 10% is a disaster, and [expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) makes this concrete. Conviction is about the strength of your *expected-value* case (how right you are *and* how much the payoff is worth *and* how much is priced), not about how often you expect to be directionally correct. A high-conviction trade can have a modest win rate if the payoff asymmetry is large enough.

## How it plays out in real markets

**Late 2018: the volatility spike and the "buy the dip" reflex.** Through 2017 and most of 2018, "buy every dip in US equities" had worked so reliably that it felt like a high-conviction layup every time the market wobbled — a textbook case of recency manufacturing conviction. By the October–December 2018 selloff, traders who scored the inputs honestly would have found a deteriorating picture: the Fed was actively tightening (a macro lens screaming the opposite), positioning was still long and complacent (low priced-in headroom for more upside, the market's conviction was high and one-sided), and the "dip-buying always works" base rate was being computed off an unusually calm sample. The *feeling* (buy the dip, it always works) and the *measurement* (tightening Fed, crowded longs, stretched base rate) had decoupled hard. The traders who sized by feeling got run over in the December plunge; the ones who scored conviction down trimmed into it.

**Early 2020: the COVID crash and the conviction of the consensus.** In February 2020, the consensus conviction that "the virus is a contained, China-only problem" was extremely high — and almost entirely narrative, with thin independent evidence and a base rate (pandemics) everyone was ignoring. A trader scoring the *bearish* view in late February would have found: evidence quality rising (case counts spreading outside China — primary, independent, hard data), a strongly variant view (the market was priced for "contained"), a base rate that supported caution (pandemics are not mild), and enormous priced-in headroom (equities at all-time highs, pricing zero pandemic risk). That is a high-conviction *short/hedge* scorecard precisely when the consensus conviction was loudest in the other direction — a clean illustration of "your conviction versus the market's conviction" paying off.

**October 2022: the CPI print and a variant view on inflation.** By October 2022, the consensus had high conviction that inflation was sticky and the Fed would keep hiking aggressively — a view so widely held it was deeply priced. A trader with a *variant* view (that goods inflation was rolling over faster than the consensus modeled, supported by independent primary data on shipping rates, used-car prices, and inventory gluts) had the makings of a high-conviction scorecard: differentiated view, independent evidence, supportive base rate (supply-shock inflation tends to reverse), and wide priced-in headroom (the market was positioned for "higher for longer"). When the cooler prints eventually arrived, the violent rally rewarded the differentiated-and-evidenced conviction — and punished those whose "high conviction" in sticky inflation was really just consensus they felt strongly about. The role of the *surprise* relative to what was priced, rather than the level, is exactly the mechanism in [the surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).

**A single-name earnings setup: the trap of the "obvious beat."** Take the recurring pattern around a beloved consumer-tech name heading into earnings, where the felt conviction is "they always beat, the product is great, the stock pops." A trader scoring this honestly almost always finds a low number, and the reason is instructive. Evidence quality is thin — "they always beat" is a base-rate observation, not independent primary evidence about *this* quarter, unless you actually have alternative data on the quarter's sales. The variant view is usually nonexistent — "they beat and pop" is the consensus, which means the options market has already priced a large move (the implied move is wide precisely because everyone expects fireworks), so the priced-in headroom is near zero. The base rate for "buy a consensus-beloved name into earnings expecting a pop" is genuinely poor, because the bar is set by the implied move, not by whether the company does well: a strong quarter that merely meets the elevated whisper number sells off. So the scorecard collapses a 90%-feeling trade to a 25, *correctly*, and the dealer's perspective — that the move is already priced into the option premium you'd pay — is exactly the other-side-of-the-trade reasoning in [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade). The high-conviction version of an earnings trade is not "they'll beat"; it's "I have independent data that the quarter is materially different from what the implied move prices" — a variant view with primary evidence and real headroom, which is rare and which is why most earnings trades should be small or skipped.

The thread through all four: the loudest *felt* conviction was consensus conviction, which is the most-priced and therefore the *least* edge-bearing. The measured conviction — built from independent evidence, a variant view, the base rate, and the priced-in gap — pointed the other way, and the gap between the feeling and the measurement was the trade. This is not a coincidence of these particular episodes; it is structural. The very forces that make a trade *feel* high-conviction (a vivid story, broad agreement, a recent run of confirmation) are the same forces that price it in and strip its edge. That inversion — feeling loudest where edge is thinnest — is the reason measuring conviction beats feeling it, and it is why the scorecard is built to penalize exactly the inputs that drive the feeling.

## The playbook: the conviction scorecard

Here is the repeatable process. Run it before you size any position, and re-run it whenever material evidence arrives.

**1. Score the five inputs honestly, one at a time.** Force yourself to write a number and a one-line justification for each. Doing them separately prevents a strong feeling about one from flooding the rest.

- Evidence quality and independence (max 25): how many genuinely independent, primary pieces of evidence?
- Variant perception (max 25): how is my view differentiated, and what is my evidence I'm the one who's right?
- Base rate / outside view (max 20): does the history of setups like this back me or fight me?
- Priced-in headroom (max 20): how wide is the gap between consensus and my view?
- Lens agreement (max 10): how many *independent* lenses agree, and is any screaming the opposite?

**2. Sum to a score and assign a tier.** Below 35 → no trade. 35–49 → Low. 50–69 → Medium. 70–100 → High. Resist fiddling inputs to clear a threshold; if you are tempted to, that is recency or narrative leaking in.

**3. Sanity-check against the feeling — and trust the score.** Write down your gut conviction *separately*. If the gut and the score disagree by a lot, the gap is almost always enthusiasm inflating the gut. Investigate *why* the gut is high (is it recency? narrative? social proof?), but size the score.

**4. Check your conviction against the market's.** Is the trade crowded (high market conviction, low headroom)? A high personal conviction that merely matches a high market conviction is consensus, not edge — dock the priced-in input accordingly.

**5. Map the tier to a size — within a hard cap.** Use a fixed tier-to-size table (e.g. Low 1%, Medium 2%, High 4% of capital, or the equivalent Kelly fractions). The High tier is *capped*; conviction scales size within the survival constraint, never beyond it.

**6. Write the falsifier.** Before the trade goes on, name the single fact that would cut your conviction in half. This is the trigger for re-scoring.

**7. Re-score on new, independent evidence.** When evidence arrives, ask "is this the falsifier, or noise?" Update the relevant input, recompute the score, and re-tier the size. Confirmation raises conviction-in-the-view; a closing gap lowers conviction-as-a-trade. Trim when the total falls a tier; cut hard when the falsifier triggers.

![The honest-conviction checklist with six questions and the sizing rule](/imgs/blogs/measuring-conviction-how-sure-are-you-really-7.png)

The checklist card above is the thing to keep beside your screen. Six questions — evidence, variant view, base rate, priced-in, lens agreement, and the falsifier — and one rule: sum to a score, map the score to a tier, map the tier to a size, and re-score on every new piece of evidence. If you cannot name the falsifier, you do not have conviction; you have enthusiasm, and you should size accordingly.

The deepest lesson is the one the two trades in the opening were built to teach. The industrial and the AI name felt identical — both "100% sure" — and the feeling was not just unhelpful, it was *inverted*, surest on the trade with the least edge. The scorecard does not make you feel more certain; it makes your sizing honest, which is the only thing that survives a long run of trades. You will still love some bad trades and feel lukewarm about some great ones. That is fine. Love them all you want — then grade them, size the grade, write the falsifier, and let the measurement, not the mood, decide how much you have on the line. The forward post [asymmetry and the art of the high-conviction bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet) takes the next step: how to find the rare trades where high *measured* conviction meets a payoff so asymmetric that they deserve the top of your size range.

## Further reading & cross-links

Within this series — *The Analyst's Edge: From Information to a Market Call*:

- [Thinking in Probabilities, Not Predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions) — why conviction lives in probability space, and how felt confidence comes apart from calibrated odds.
- [Variant Perception: Where Real Edge Comes From](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from) — the differentiated-and-correct quadrant that the variant-perception input scores.
- [Reconciling Conflicting Signals When the Lenses Disagree](/blog/trading/analyst-edge/reconciling-conflicting-signals-when-the-lenses-disagree) — how to count lens agreement without double-counting correlated lenses.
- [What's Priced In: The Question Behind Every Trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) — the priced-in headroom input, in depth.
- [Stress-Testing Your Thesis With a Pre-Mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem) — where the falsifier comes from.
- [From Conviction to Size: The Bet-Sizing Bridge](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge) — the full tier-to-size translation.
- [Calibration: Keeping Score on Your Own Forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts) — how to verify your conviction scale actually predicts outcomes.
- [Asymmetry and the Art of the High-Conviction Bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet) — when high measured conviction meets an asymmetric payoff.

Mechanism and craft cross-links:

- [Position Sizing and the Kelly Criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — the math behind mapping a tier to a Kelly fraction.
- [Expectancy: Why Win Rate Lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) — why conviction is about expected value, not win rate.
- [Risk Management: The Only Free Lunch](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — why the High tier is capped: survival is the compounding engine.
- [The Asymmetry of Losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — the reason ruin binds even at high conviction.
- [Positioning and the Pain Trade](/blog/trading/event-trading/positioning-and-the-pain-trade) — how the market's conviction (crowding) interacts with yours.
- [Trading Psychology and the Execution Gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap) — keeping recency and emotion out of the dial.
- [The Indicator Trap](/blog/trading/technical-analysis/the-indicator-trap) — why correlated lenses count as one, not many.
