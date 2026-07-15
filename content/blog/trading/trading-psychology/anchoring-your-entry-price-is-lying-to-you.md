---
title: "Anchoring: Why Your Entry Price Is Lying to You"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A practitioner's deep dive into anchoring — the bias that turns your entry price, a round number, or a stock's old high into a private reference point that quietly wrecks your exits and your sense of 'cheap.' With the science, worked P&L examples, a real case study, and a drill to strip the anchor out."
tags: ["trading-psychology", "anchoring", "behavioral-finance", "cost-basis", "disposition-effect", "breakeven-trap", "cognitive-bias", "kahneman", "tversky", "decision-making", "risk-management", "mark-to-market"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 35
---

> [!important]
> **TL;DR** — The price you paid for a stock is a fact about *your past*, not about the stock's *future*. Anchoring is the glitch that lets that irrelevant number — or a round number, or the 52-week high — secretly set your idea of "cheap," "expensive," and "when I'll sell."
>
> - Anchoring is real and measured. In Tversky and Kahneman's 1974 experiment, a **random** number from a rigged wheel dragged people's factual estimates from a median of **25% to 45%** — a 20-point swing from a number everyone watched being generated at random.
> - Your cost basis is the most expensive anchor of all. It creates the **breakeven trap**: "I'll sell when it's back to what I paid." The market has never heard of your entry price, so waiting for it can turn a planned small loss into a large realized one.
> - Anchoring to a stock's former high makes a **falling knife look cheap**. Cisco topped near **$80** in March 2000 as the world's most valuable company (~$555B market cap), fell more than **80%** to about **$10** by October 2002, and did not set a new record high for roughly **25 years** — every step down looked like "a bargain versus $80."
> - The proof it's an anchor and not analysis: two traders holding the **identical** position at different cost bases will make **opposite** decisions. Only one of them (at most) can be right, and the market treats them identically.
> - The fix is not willpower. It's a mechanical **mark-to-market-blind review**: value every holding as if you were just handed its cash equivalent, and ask "would I buy *this*, at *this* price, with new money today?" If no, you don't own it — you're just anchored.

Here is a question that sounds too simple to matter, until you notice how often you get it wrong: **why won't you sell that stock?**

If the honest answer is some version of "because it's below what I paid," stop and look at that sentence. It contains a number — your entry price — that is doing an enormous amount of work in your decision. And that number is *private to you*. The person who sells you the next share, the algorithm quoting the market, the company itself — none of them know or care what you paid. Your cost basis exists in your brokerage statement and nowhere else in the universe. Yet it is quietly setting your definition of a good outcome, a bad outcome, "cheap," "expensive," and "time to sell."

That is anchoring. It is one of the most reliably demonstrated biases in all of psychology, it operates on experts and amateurs alike, and it survives even when you *know* the anchor is meaningless. This article is about how anchoring works, exactly how it drains trading accounts, and the one drill that actually breaks its grip.

The diagram below is the mental model for the whole piece. Keep it in your head as we go.

![An anchored reference point sits between the trader and fair value: private, irrelevant anchors capture the reference point and dominate a decision that should track today's fair value](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-1.webp)

Read it left to right. On the left is a cluster of *anchors* — numbers that lodge in your mind: your entry price, round numbers like $100, the 52-week high, the IPO or all-time-high price, the analyst's target. Each of them is a fact about the past or about somebody else's math. In the middle, those anchors capture a single **anchored reference point** — the number your mind now treats as "the right price." Meanwhile, off to the side, *new information* (earnings, news, a changed competitive landscape) is supposed to drive your estimate of **fair value** — what the thing is actually worth today. But when it comes time to decide, the anchor **dominates** and fair value gets **ignored**. That imbalance — a private, irrelevant number beating out live, relevant information — is the entire disease. Everything else in this post is a symptom of it.

## Foundations: how anchoring actually works

You need no finance or psychology background for this section — we build every term from zero. By the end you'll understand not just *that* anchoring happens but *why*, which is what lets you catch it in yourself.

### What an "anchor" is

An **anchor**, in the psychology of judgment, is a number that enters your mind before you make an estimate and then *pulls your estimate toward itself* — even when the number is logically irrelevant to what you're estimating. The classic demonstration is almost comically pure.

In 1974, the psychologists Amos Tversky and Daniel Kahneman published a paper in the journal *Science* called "Judgment under Uncertainty: Heuristics and Biases." In one experiment, they spun a wheel of fortune marked 0 to 100 in front of each subject. The wheel was **rigged** to stop only on 10 or 65. Then they asked the subject a factual question: what percentage of African countries are members of the United Nations? First, is it higher or lower than the number on the wheel; then, what's your actual estimate?

The number on the wheel had nothing to do with African countries. Everyone watched it being generated at random. And yet:

![The wheel-of-fortune experiment: a random number from a rigged wheel dragged people's estimates from a median of 25% to 45%, a 20-point swing from a number everyone saw was random](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-2.webp)

The people who saw the wheel land on **10** gave a median estimate of **25%**. The people who saw it land on **65** gave a median estimate of **45%**. Same question, same facts about the world — but a random number, seen for a second, moved the answer by twenty percentage points. Tversky and Kahneman also reported that paying people for accuracy did **not** make the effect go away. The anchor doesn't work because you're lazy or unmotivated. It works on a level underneath motivation.

> An anchor is a number that gets its hooks into your estimate before you've had a chance to think — and then refuses to let go, even when you know it's meaningless.

### Anchoring works on experts, in their own field

Your first defense is probably "sure, but that's undergraduates guessing at trivia — a professional wouldn't fall for it." The research says otherwise, and this is the part that should worry every trader.

In 1987, Gregory Northcraft and Margaret Neale ran a study (in *Organizational Behavior and Human Decision Processes*) where they took **real estate agents** — professionals whose living depends on pricing houses accurately — walked them through an actual house, and handed them a full information packet. The only thing they varied between agents was the printed **listing price**. Agents who saw a higher listing price appraised the house higher, recommended a higher sale price, and set a higher floor for the lowest acceptable offer. The listing price was an anchor, and it moved the professionals' "expert" valuations.

The kicker: when asked afterward what factors influenced their appraisal, only about **19%** of the agents mentioned the listing price at all. (Among the amateurs in the study, a more honest **37%** admitted it mattered.) The experts were *more* anchored and *less* aware of it. Expertise didn't inoculate them; it just gave them more confidence in a number that had been quietly set for them.

If you want the most unsettling version, look at Birte Englich, Thomas Mussweiler, and Fritz Strack's 2006 study, memorably titled "Playing Dice With Criminal Sentences" (*Personality and Social Psychology Bulletin*). Experienced legal experts read a criminal case, then **rolled a pair of dice** — which the experimenters had loaded to come up high or low — and then recommended a sentence. Judges who rolled high recommended longer sentences than judges who rolled low, *even though they had personally thrown the dice and knew the number was random*. Trained judges. Real sentencing decisions. Anchored by dice.

The lesson for you: "I know it's just my entry price, so it won't affect me" is precisely the belief the research demolishes. Awareness is not immunity.

### Why the anchor wins: adjust, then stop too early

So what is actually happening inside your head? There are two complementary mechanisms, and you should understand both because they suggest different defenses.

The first is **anchoring-and-adjustment**, the process Tversky and Kahneman originally described and that Nicholas Epley and Thomas Gilovich later refined. When you have a number in mind and need a different estimate, you *start from the anchor and adjust away from it* — but you stop adjusting as soon as you reach a value that seems merely *plausible*, which is usually far too close to where you started. You land at the near edge of the range of answers you'd accept, not at the center of it.

![Anchoring-and-adjustment versus re-deriving value: anchoring starts at a number and adjusts too little, so the estimate stays stuck near the anchor; the fix starts from what the asset is worth, price-blind](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-3.webp)

The left column is what your brain does by default: it starts at the anchor (your entry, the old high, the analyst target), adjusts a little toward fair value, stops at the first "plausible" number, and the estimate ends up stuck near the anchor. The right column is the fix, which we'll build into a drill later: start from *zero* — what is this actually worth? — build value from cash flows and facts, and only *then* compare it to the price. One process is contaminated by the anchor at step one; the other never lets the anchor in.

The second mechanism is **selective accessibility**, demonstrated by Mussweiler and Strack. Once a number is in your head, it primes your memory to fetch *evidence consistent with that number*. Anchor high, and your mind more easily calls up reasons the true value is high; anchor low, and it dredges up reasons it's low. This is why anchoring feels like *thinking* rather than *bias* from the inside — you genuinely generate arguments, you just generate the ones your anchor recruited. It also explains why paying people for accuracy didn't help in the original studies: the problem isn't that people didn't try, it's that the anchor had already stacked the evidence.

Two mechanisms, one result: **the number you saw first captures the number you end up believing.** Now let's watch it eat money.

## 1. Your cost basis is the most expensive anchor you own

Every trader carries one anchor around at all times, welded to every open position: the price they paid. It shows up in your platform as your **average cost** or **cost basis**, and it generates the single most common self-destructive sentence in all of trading: *"I'll sell when it gets back to what I paid."*

Call it the **breakeven trap**. It feels like prudence — "I'm not going to lock in a loss" — but it is anchoring in its purest financial form. You have taken a number that is a fact about your history and made it the trigger for a decision about the future. The market, which sets the future, has no idea what your breakeven is.

Here's why that's so expensive. The decision to hold or sell a position *should* depend on one thing: what you expect the position to do **from here**. That expectation is a function of today's price, today's fundamentals, and your read of the future — none of which include what you paid. But the breakeven anchor overrides all of it. Below your entry, you refuse to sell no matter what the evidence says, because selling would "make the loss real." Above your entry, you're suddenly happy to sell, because now it's a "win." Your entry price has become a switch that flips your behavior for reasons that have nothing to do with the trade's actual prospects.

![The breakeven trap: anchoring to your entry freezes the exit while the loss compounds far past the planned stop, turning a small planned loss into a large realized one](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-4.webp)

The chart shows the mechanism in motion. You buy at $50; that becomes the breakeven line and the anchor. Your written plan says cut the position at $46 — an 8% stop. But when price actually touches $46, the anchor speaks louder than the plan: *"I'll sell when it's back to $50."* So you freeze. Price grinds through $42 ("it'll bounce"), and you're now well below your stop, holding for a reason that isn't in your strategy — it's in your cost basis. By the time you capitulate at $38, the planned $4,000 loss has become a realized $12,000 loss. Let's put real numbers on it.

#### Worked example: the breakeven trap versus the early cut

Suppose you buy **1,000 shares at $50**, committing **$50,000**. Your written rule is a hard stop at −8%, i.e. at **$46**.

- **The disciplined path.** Price hits $46. You sell. Loss = 1,000 × ($50 − $46) = **−$4,000**. You have $46,000 back in cash and a clear head.
- **The anchored path.** Price hits $46, but the breakeven anchor says "not yet — I'll get out at $50." You hold. It drifts to $42 ("due for a bounce"), then to **$38**, where the pain finally overwhelms the hope and you sell. Loss = 1,000 × ($50 − $38) = **−$12,000**.

Same stock, same information, same starting point. The breakeven anchor turned a **$4,000** scratch into a **$12,000** wound — a **3×** difference — purely by making $50 the number that mattered instead of the plan. And notice the cruel asymmetry: to get back from −$12,000 to breakeven, the stock has to rally from $38 to $50, a **+31.6%** move, while it only fell −24% to get there. The anchor made you demand a bigger recovery than the drop that created the hole.

**Intuition:** your entry price is a fact about your past that the market will never honor; anchoring your exit to it lets a small, planned loss metastasize into a large, unplanned one.

### The disposition effect: this is measurable, at scale

You might think the breakeven trap is a personal failing. It isn't — it's a mass phenomenon with a name and a number. In 1985, Hersh Shefrin and Meir Statman named it the **disposition effect** (in the *Journal of Finance*): the documented tendency of investors to *sell winners too early and ride losers too long*. It is the breakeven anchor, aggregated across a market.

In 1998, Terrance Odean measured it directly. He analyzed the trading records of **10,000 accounts** at a large discount broker from 1987 to 1993 (also in the *Journal of Finance*) and computed two numbers:

- **PGR** — the Proportion of Gains Realized: of the winning positions people *could* have sold on a given day, what fraction did they sell?
- **PLR** — the Proportion of Losses Realized: of the losing positions they could have sold, what fraction did they sell?

He found **PGR = 0.148** and **PLR = 0.098**. In plain terms, these investors were about **1.5 times** more likely to sell a position that was up than one that was down. They cut winners and clung to losers — exactly what the breakeven anchor predicts. And it wasn't smart: the winners they sold went on to **outperform** the losers they held by about **3.4 percentage points** over the following year. The anchor made them sell the wrong things and keep the wrong things, and it cost them measurable return.

We cover the loss-aversion machinery that powers this — why a loss feels so much worse than an equal gain feels good — in [loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect). Anchoring supplies the *reference point* (your entry); loss aversion supplies the *pain* of crossing below it. Together they're the reason your losers turn into long-term relationships.

**What it costs / when it breaks:** the cost-basis anchor is worst exactly when you most need to act — in a losing position that's still deteriorating. That's when "wait for breakeven" feels safest and is most dangerous.

## 2. Anchoring distorts your entire sense of "cheap" and "expensive"

The cost-basis anchor freezes your *exits*. A second family of anchors corrupts something even more fundamental: your judgment of whether something is a good deal at all. Because here's the uncomfortable truth — **"cheap" and "expensive" are meaningless without a reference point**, and your brain will happily grab whatever reference point is lying around, usually a price the stock used to be.

### The falling-knife illusion

The most dangerous version is anchoring to a stock's **former high**. When a stock that was recently $100 is now $60, every instinct screams "40% off — bargain!" But that instinct is doing pure anchoring: it's measuring cheapness against the *old price*, which the market has already rejected. The relevant question — is $60 cheap relative to what the business is *worth now*? — never gets asked, because the anchor answered a different, easier question first.

![The falling-knife illusion: anchoring to the former high makes a collapsing stock look cheap while fair value falls faster than price the whole way down](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-6.webp)

The picture is the trap in one image. The dashed line up top is the anchor — the old all-time high. The solid line is the market price, stair-stepping down. At every level, an anchored buyer sees "it was $80, now it's $25 — a bargain versus the high!" But look at the lower dashed line: the *justified value* of the business was falling **faster** than the price the whole way down. Relative to what it was actually worth, the stock was *expensive at every step* — the "discount to the old high" was an illusion created entirely by the anchor. Catching a falling knife feels like value investing and is usually just anchoring with extra steps.

This is not a toy example. It is the single most reliable way that fortunes evaporate in a bear market, and we'll walk through the canonical case — Cisco — in the case-study section below.

### Round numbers and the 52-week high: anchors the whole market shares

Some anchors are private (your entry price). Others are *public*, which makes them even sneakier because everyone is anchored to the same number at once, and the anchor becomes partly self-fulfilling.

**Round numbers.** Prices cluster around round numbers — $50, $100, 5,000 on an index — far more than chance would predict. The economist Carol Osler documented (in her work on currency order flow, published in the *Journal of Finance* in 2003) that traders' stop-loss and take-profit orders pile up at round numbers, which in turn makes those levels act like support and resistance. A round number is a coordination point, not a valuation. The danger is confusing the two — setting your sell at "$100 because it's a nice round number" when fair value has moved to $85, and refusing to exit until the stock touches a figure that means nothing.

**The 52-week high.** In 2004, Thomas George and Chuan-Yang Hwang showed (in the *Journal of Finance*) that a stock's **52-week high** works as a market-wide anchor. Investors treat nearness to the 52-week high as a reference: when good news should push a stock *through* its old high, people under-react, as if the high were a ceiling. George and Hwang found that a stock's distance from its 52-week high predicted future returns *better* than its past returns did — a direct fingerprint of anchoring on prices.

The anchor is powerful enough to move even the most sophisticated, high-stakes transactions on Earth. In 2012, Malcolm Baker, Xin Pan, and Jeffrey Wurgler studied mergers and acquisitions (in the *Journal of Financial Economics*) and found that **takeover offer prices are biased toward the target's 52-week high** — in fact the single most common offer price *is* the 52-week high — and a deal's probability of acceptance jumps discontinuously the moment the offer crosses that level. Boards, bankers, and billionaires, negotiating deals worth billions, anchor to a salient past price the same way a retail trader anchors to their entry.

Here is the full menagerie of anchors that reliably wreck traders:

![The anchors that wreck traders: every anchor that hijacks a trader is a number the market has no memory of, split between anchors from your own history and anchors from salient market numbers](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-7.webp)

They fall into two families. **From your own history:** your cost basis ("back to breakeven") and the IPO or all-time-high price ("so cheap now"). **From salient market numbers:** round numbers ("$100 feels right"), the 52-week high or low ("underreact to news near it"), and analyst price targets ("someone's stale math"). Every one of them is a number the market has no memory of and no obligation to honor. What they have in common is that they're *available* — easy to see, easy to remember — and availability is exactly what anchoring feeds on.

#### Worked example: anchoring to a price target versus re-deriving value

Suppose an analyst publishes a **$120 price target** on a stock trading at **$150**. The anchored move is to conclude "it's 20% overvalued, I'll wait for $120." Now the company reports a blockbuster quarter — earnings and guidance both jump — but the analyst hasn't updated the note. Fair value, honestly re-derived from the new numbers, is now **$175**. What do you do?

- **Anchored to the target:** you still see $150 as "above the $120 target," so you avoid buying or you sell. You're trading off a number that was someone else's estimate, published before the news, now stale.
- **Re-derived:** you rebuild the valuation from the new earnings, get ~$175, and see $150 as a **~14% discount** to fair value.

The target didn't become wrong because it was low; it became wrong because it was **old**, and anchoring made you keep using it. A price target is a snapshot of one analyst's model at one moment. Treating it as a fixed reference point substitutes their stale math for your fresh judgment.

**Intuition:** an analyst's target is a number with an expiry date; anchoring to it means letting yesterday's estimate veto today's facts.

## 3. The tell that it's an anchor: two traders, one position, opposite decisions

Everything so far could be waved away as "well, sometimes waiting works out." So here is the cleanest possible test of whether you're being ruled by an anchor — a thought experiment that collapses the whole illusion.

Take two traders, **A** and **B**. They own the *same stock*, in the *same size*, right now, at the *same price of $70*. They face the identical future: the same possible outcomes, the same probabilities, the same expected return from here. The **only** difference between them is a fact about the past: A bought at $40, B bought at $100.

![Two traders, one position, opposite decisions: same stock, same price, same future, but the private cost basis flips the decision from 'sell' to 'hold'](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-5.webp)

Watch what their anchors do. Trader A is sitting on a **+$30 (+75%)** gain, so the anchor whispers *"winner — take profit, lock it in."* Trader B is sitting on a **−$30 (−30%)** loss, so the anchor whispers *"loser — hold to $100, don't realize it."* A wants to sell; B wants to hold. **Opposite decisions.**

But look at the last column. The *rational* move for both is identical, because it depends only on the forward risk and reward of holding the same stock at $70 — and that's the same for A and B. The market offers both of them exactly $70 to exit and exactly the same future if they stay. Their cost bases are invisible to it.

So at least one of them is wrong, and probably both are being irrational, because a *correct* decision procedure would give the same answer to the same situation. If your process produces different actions for A and B, your process is anchored to cost basis — a number that, by construction, cannot contain information about the future.

#### Worked example: the same $70, two different verdicts

Let's make the irrationality concrete. Both A and B hold 1,000 shares at $70; both believe (correctly, identically) that the stock is now worth about **$70** — fairly valued, with a coin-flip of ±20% over the next year.

- **Trader A (bought at $40):** unrealized gain **+$30,000**. Anchoring + loss aversion say "protect the win," so A sells. If the stock is truly fairly valued, that's a defensible-but-arbitrary choice — A is selling because of a *gain that already happened*, not because of anything ahead.
- **Trader B (bought at $100):** unrealized loss **−$30,000**. Anchoring says "wait for $100," so B holds a fairly valued stock specifically to avoid booking a loss — the textbook breakeven trap. B is holding because of a *loss that already happened*.

If fair value is genuinely $70, the right answer is the same for both — hold if you'd buy it here, sell if you wouldn't, based only on the forward view. The $60 gap between their entries is pulling them in opposite directions for no reason a rational market would recognize.

**Intuition:** if two people with the identical position *should* act differently only because they paid different prices, at least one of them is obeying an anchor, not the market.

## What it looks like at the screen: the real-time tells

Biases are easy to nod along with in an article and invisible in the moment. So here is what anchoring actually feels like while you're trading — the specific thoughts and micro-behaviors that mean the anchor has the wheel. Learn to notice these the way you'd notice a warning light.

**Your eyes go to the cost-basis column first.** You open your positions blotter and the number your gaze lands on is not the current price or the day's change — it's your **average cost** and the red or green P&L beside it. If the first fact you process about a position is what you paid, that number is already framing everything after it.

**You catch yourself doing "back to X" math.** *"If it just gets back to $50, I'm out."* *"I only need a 20% bounce to be even."* Any sentence where the target is defined by your entry rather than by the stock's prospects is the breakeven anchor talking. The stock does not know it owes you a bounce.

**Red positions feel unsellable; green ones feel like they must be locked in.** You notice you can click "sell" easily on winners and something in your chest resists it on losers — the same size loss and gain feel completely different, and the difference is measured *from your entry*. That asymmetry is anchoring married to loss aversion.

**A round number or an old high has become "the" price.** You're mentally waiting for $100, or treating the 52-week high as a lid, or describing a stock as "cheap" and when you interrogate the word, the comparison turns out to be to a price from six months ago. The reference point crept in and you never chose it.

**You're arguing with the tape.** The position keeps telling you something (it's going down) and you keep generating reasons it's wrong ("oversold," "due for a bounce," "the market doesn't get it"). Remember selective accessibility: once the anchor is set, your mind fetches confirming evidence effortlessly. The *fluency* of your bullish reasons on a falling position is itself a tell — you're not analyzing, you're rationalizing a number.

**You feel relief, not conviction, when you finally act.** When you exit an anchored loser, notice the emotion. If it's mostly *relief that the pain stopped* rather than *conviction that it was the right forward decision*, the anchor was in charge right up until the end.

None of these thoughts announce themselves as bias. They arrive dressed as prudence, patience, and analysis. The skill is recognizing the costume.

## Common misconceptions

**"Waiting to get back to breakeven is just being patient, not a bias."** Patience means giving a *sound thesis* time to work. The breakeven trap is waiting for a *specific number defined by your purchase*, regardless of whether the thesis is still sound. The test: if the stock were sitting at your breakeven for a reason unrelated to your thesis improving, would you still be a holder? If the only thing that changed is the price hitting your entry, you were anchored, not patient.

**"It's cheap — it's down 40% from its high."** Down-from-the-high is a statement about the *past price*, not the *present value*. A stock can be down 40% and still expensive, if the business deteriorated more than 40%. "Cheap" only means anything relative to *what it's worth now*. The former high is an anchor, not a valuation.

**"Averaging down lowers my breakeven, so it's a smart way to recover."** Averaging down (buying more as it falls) does mathematically lower your *average cost*. But your average cost is the anchor, not a real thing — moving it doesn't move fair value one cent. If the stock is genuinely undervalued, buying more can be rational on its own merits; if you're buying *specifically to lower your breakeven so you can get out even sooner*, you're feeding the anchor, not analyzing the trade. The market still owes you nothing at your new, lower breakeven either.

**"I know it's just my entry price, so it doesn't affect me."** This is the belief the research specifically destroys. Northcraft and Neale's real estate agents mostly *denied* the listing price influenced them — and it influenced them anyway. Englich's judges *threw the dice themselves* and were anchored anyway. Knowing an anchor is irrelevant does not remove its pull; only changing your *process* does.

**"Round numbers really are support and resistance, so anchoring to them is rational."** Partly true, and that's the trap. Round numbers *do* attract orders (Osler's research), so they can genuinely act as levels — for *entries and stops*, that's usable information. The error is letting a round number define *value*, or refusing to exit a deteriorating position until it touches $100. A coordination point for order flow is not a statement about what a business is worth.

**"Analyst price targets give me a fair-value anchor to trade around."** A target is one analyst's model output at one point in time, often anchored itself to the current price and updated slowly. Using it as a *fixed* reference means your valuation lags reality by however stale the note is. Targets are inputs to think about, not reference points to obey.

## How it shows up in real markets

Anchoring is not a laboratory curiosity. Here are documented episodes and patterns where it moved real money at scale — every figure below is sourced.

### 1. Cisco Systems and the 25-year round trip

The canonical falling-knife case is **Cisco Systems**. In March 2000, at the peak of the dot-com boom, Cisco's stock reached roughly **$80** a share and the company briefly became the **most valuable public company in the world**, with a market capitalization around **$555 billion** — trading at something like 200 times earnings. Over the next two-plus years the stock collapsed more than **80%**, bottoming near **$10** in October 2002. Roughly **$430 billion** of market value evaporated.

Now watch the anchor do its work on the way down. At $50, Cisco looked "cheap" — it had been $80. At $25, it looked *incredibly* cheap — 70% off the high! At $10, a bargain! But "cheap versus $80" was measuring against a peak the market had permanently repudiated. The relevant question was whether the price was low relative to what Cisco was actually worth as a business — and for a long time it wasn't; the anchor just made it *feel* like a bargain at every step. The most vivid proof of how far the $80 anchor was from reality: Cisco did not set a new all-time high again for roughly **25 years**, finally closing at a record for the first time since 2000 in December 2025. An entire generation of investors who anchored to "it was $80" waited a quarter-century to be made whole — and only if they never sold.

### 2. The disposition effect in 10,000 real accounts

The breakeven anchor isn't anecdotal — it's been measured in the wild. Odean's 1998 study of **10,000 discount-brokerage accounts** (1987–1993) found investors realized gains at a rate of **0.148** versus losses at **0.098** — roughly **1.5×** more willing to sell a winner than a loser — and that the winners they sold beat the losers they kept by about **3.4 percentage points** over the next year. This is the cost-basis anchor, made visible across a whole population of ordinary investors: a systematic, measurable tax on returns paid by people selling winners and marrying losers.

### 3. Home sellers and the purchase-price anchor

Anchoring to what you paid isn't limited to stocks. David Genesove and Christopher Mayer studied the Boston condo market in the 1990s (in the *Quarterly Journal of Economics*, 2001) and found that sellers facing a **nominal loss** — whose home was worth less than they'd paid — set **asking prices higher** by an amount equal to **25–35%** of the gap between the property's expected value and their original purchase price. They then waited longer and were less likely to sell. Their *purchase price* — a fact about their past — anchored their asking price and made them demand more than the market would pay, exactly like a trader refusing to sell below breakeven. Same anchor, different asset class.

### 4. Takeover offers that cluster at the 52-week high

At the most sophisticated end of finance, Baker, Pan, and Wurgler's 2012 study found that acquisition **offer prices are pulled toward the target's 52-week high**, the modal offer *equals* the 52-week high, and acceptance probability jumps when an offer crosses it. Corporate boards and investment bankers — the professionals' professionals — anchor multi-billion-dollar negotiations to a salient past price. If a $50-billion deal can be anchored to a stock's high from eleven months ago, your $5,000 position anchored to what you paid last Tuesday is not a personal weakness. It's the human default.

### 5. The "return to normal" anchor in every crash

A repeating market pattern: after a large fall, investors and commentators anchor to the *pre-crash* price level as "normal" and treat the crash as a temporary deviation to be reversed. Sometimes it is (broad indices historically recover over long horizons). But at the level of an *individual* stock or a *specific* sector, "it'll go back to where it was" is the falling-knife illusion again — anchoring to a former price and assuming the world owes a reversion. The distinction that matters is the one the anchor prevents you from making: is this cheap relative to *fair value now*, or merely relative to *where it used to be*? The former is analysis; the latter is an anchor wearing analysis's clothes.

## The drill: the mark-to-market-blind review

You cannot un-see your cost basis by trying hard, any more than the real estate agents could un-see the listing price. Willpower is the wrong tool. What works is a **process** that structurally removes the anchor from the decision — because, as the two-traders test proved, a correct process gives the same answer regardless of what you paid.

The core move is a thought experiment I call the **mark-to-market-blind review**: for every position you hold, imagine you were just handed its **cash equivalent** today, and ask the one question that contains no anchor —

> **Would I buy *this*, at *this* price, with *new money*, right now?**

If the answer is yes, you keep it — you'd re-buy it, so holding is just re-buying by default. If the answer is no, you sell, because the only thing keeping you in a position you wouldn't re-buy is the anchor. That's it. Notice what the question does *not* contain: it never mentions your entry price, your breakeven, the old high, or your unrealized P&L. Those are all struck from the decision by construction.

![The mark-to-market-blind review: value each holding as if handed its cash today, run the new-money test, and let the answer — not the entry price — decide keep or sell](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-8.webp)

The flow is deliberately mechanical. **Pick a holding. Strike the cost basis from view.** (Literally — hide the cost and P&L columns; see below.) **Value it as cash:** what is this position worth today, in dollars, if I just received it? **Run the new-money test:** would I buy this, here, now? **No → sell and redeploy the cash. Yes → keep.** Run it on every position on a fixed schedule and the anchor never gets a vote.

#### Worked example: the new-money test in action

You hold **300 shares** bought at **$80**, now trading at **$52**. Your statement shows an unrealized loss of 300 × ($80 − $52) = **−$8,400**, and the breakeven anchor is screaming "wait for $80."

Now do the drill. Mark it to market: you're holding **300 × $52 = $15,600** of this stock. Reframe it as cash: *someone just handed you $15,600 and asked what to do with it.* The question is no longer "how do I get back to $80?" It's **"would I buy 300 shares of this company at $52 today with $15,600 of fresh cash?"**

- If **yes** — you genuinely think it's worth more than $52 on the forward evidence — then hold. You're not anchored; you're a willing buyer at the current price. The −$8,400 is irrelevant to that judgment.
- If **no** — you wouldn't put new money into it at $52 — then the *only* reason you still own it is the $80 anchor. Sell the $15,600, put it into something you *would* buy, and you've just converted a dead, anchored position into live capital.

The −$8,400 doesn't enter either branch. It already happened; it's a fact about your past that the market will never refund. **Intuition:** you re-own every position every day whether you act or not, so make it a real decision at today's price — the entry price is not invited.

### Making the drill stick: five concrete practices

1. **Blind the cost-basis column.** Most platforms let you hide or reorder columns. Hide **average cost** and **unrealized P&L** during your review. If you can't hide them, cover them physically. You are removing the anchor from your field of view before you decide, the same way you'd move a distraction off your desk. (Bring the P&L back afterward for tax and record-keeping — just not while deciding.)

2. **Write fair value before you look at price.** For any position or candidate, write down what you think it's worth *first*, from the fundamentals, and *then* reveal the price to compare. This is the "start from zero" path from the mechanism diagram — you can't anchor to a number you haven't looked at yet.

3. **Run the new-money test on a schedule.** Once a week or once a month, go position by position and force the "would I buy this here?" question on each. Anything that gets a "no" is a sale, not a "let's give it more time." Scheduling it removes the in-the-moment emotion.

4. **Pre-commit your invalidation, in writing, before you enter.** Decide *before* you buy what would make the thesis wrong — a price, a fundamental change, a time limit — and write it down. Then honor it regardless of your entry. This is the antidote to selective accessibility: you define the exit conditions while you're still clear-headed, before the anchor can recruit reasons to ignore them. We build this out in [defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront).

5. **Reframe with the coin-flip test.** When you catch yourself doing "back to breakeven" math, ask: *"If I had this exact amount of cash instead of this stock, would I buy this stock with it, or something else?"* If the answer is "something else," your breakeven is the only thing holding you, and it's holding you to a number the market forgot the moment you clicked buy.

Anchoring is one bug among many, and it rarely fires alone — it hands off to loss aversion, confirmation bias, and sunk-cost thinking in a chain. For the full map of how these biases interlock into a single bad trade, see [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders). Anchoring is usually where the chain *starts*, because it sets the reference point everything downstream reacts to.

## When this matters to you

This touches your money constantly, whether you trade actively or hold a long-term portfolio. Every time you look at a position and feel "I can't sell here, I'm down too much," or "I'll take profits, I'm up nicely," or "it's cheap, it used to be double this" — an anchor is in the room, and it's a number about your past or someone else's math, not about the future you're actually betting on. The 401(k) holder who won't rebalance out of a fund because "it was worth so much more" is anchored to the same $80 that trapped the Cisco holder.

The practical takeaway is small and mechanical, which is exactly why it works: **strike your entry price from every hold-or-sell decision, and replace it with the new-money question.** You will not stop *feeling* the pull of the anchor — the research is clear that awareness doesn't dissolve it. But you can build a process that doesn't let the feeling vote. Hide the cost column, write value before price, run the new-money test on a schedule, and pre-commit your exits. Do that and you've quietly removed the single most expensive number in your trading from the decisions where it does the most damage.

One last honest note: none of this is a promise that selling an anchored loser will feel good, or that the stock won't bounce right after you sell — sometimes it will, and the anchor will use that to torment you. This is educational, not individualized advice, and every position that can make money can lose it. But over many decisions, letting a private, irrelevant number set your "cheap," "expensive," and "time to sell" is a measured, repeatable tax — one you now know how to stop paying.

## Sources & further reading

Primary research behind the headline numbers:

- Amos Tversky and Daniel Kahneman, "Judgment under Uncertainty: Heuristics and Biases," *Science* 185 (1974), 1124–1131 — the wheel-of-fortune anchoring experiment (median estimates 25% vs 45% from anchors of 10 vs 65).
- Gregory B. Northcraft and Margaret A. Neale, "Experts, Amateurs, and Real Estate: An Anchoring-and-Adjustment Perspective on Property Pricing Decisions," *Organizational Behavior and Human Decision Processes* 39 (1987), 84–97 — real estate agents anchored to listing price; ~19% of agents acknowledged it.
- Birte Englich, Thomas Mussweiler, and Fritz Strack, "Playing Dice With Criminal Sentences: The Influence of Irrelevant Anchors on Experts' Judicial Decision Making," *Personality and Social Psychology Bulletin* 32 (2006), 188–200 — judges anchored by self-rolled dice.
- Nicholas Epley and Thomas Gilovich, "The Anchoring-and-Adjustment Heuristic: Why the Adjustments Are Insufficient," *Psychological Science* 17 (2006) — mechanism of insufficient adjustment.
- Hersh Shefrin and Meir Statman, "The Disposition to Sell Winners Too Early and Ride Losers Too Long," *Journal of Finance* 40 (1985), 777–790 — names the disposition effect.
- Terrance Odean, "Are Investors Reluctant to Realize Their Losses?" *Journal of Finance* 53 (1998), 1775–1798 — PGR 0.148 vs PLR 0.098 across 10,000 accounts; winners sold beat losers kept by ~3.4 points/year.
- Thomas J. George and Chuan-Yang Hwang, "The 52-Week High and Momentum Investing," *Journal of Finance* 59 (2004), 2145–2176 — the 52-week high as a price anchor.
- Malcolm Baker, Xin Pan, and Jeffrey Wurgler, "The Effect of Reference Point Prices on Mergers and Acquisitions," *Journal of Financial Economics* 106 (2012), 49–71 — M&A offers cluster at the target's 52-week high.
- David Genesove and Christopher Mayer, "Loss Aversion and Seller Behavior: Evidence from the Housing Market," *Quarterly Journal of Economics* 116 (2001), 1233–1260 — sellers facing nominal losses set asking prices 25–35% of the loss higher.
- Carol Osler, "Currency Orders and Exchange-Rate Dynamics: An Explanation for the Predictive Success of Technical Analysis," *Journal of Finance* 58 (2003) — order clustering at round numbers.

On the Cisco figures: Cisco's ~$80 March 2000 peak, its ~$555B peak market capitalization as the world's most valuable company, its ~80%+ decline to roughly $10 by October 2002, and its first new record high in ~25 years (December 2025) are documented in contemporaneous and retrospective market coverage (CNBC, "Cisco's stock closes at record for first time since dot-com peak in 2000," December 2025; Macrotrends historical price data).

Related posts on this blog:

- [Loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect) — the pain mechanism that powers the breakeven trap.
- [The cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders) — how anchoring chains into the other biases in a single bad trade.
- [What would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) — pre-committing exits so the anchor can't rewrite them later.
