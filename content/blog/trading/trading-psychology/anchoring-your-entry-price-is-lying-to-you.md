---
title: "Anchoring: Why Your Entry Price Is Lying to You"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "How the anchoring bias makes your cost basis, round numbers, and the 52-week high hijack your buy and sell decisions, why the market has no memory of your entry, and a drill to strike the anchor from every position."
tags: ["trading-psychology", "anchoring", "behavioral-finance", "cognitive-bias", "disposition-effect", "cost-basis", "loss-aversion", "kahneman", "tversky", "risk-management", "decision-making", "breakeven"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — Your entry price is a private number the market never agreed to honor, yet it silently sets the level at which you feel allowed to sell. That is the anchoring bias, and it is one of the most expensive glitches in trading.
>
> - **Anchoring is real and mechanical**: an arbitrary starting number drags every later estimate toward it, and you *know* it is arbitrary. In the 1974 wheel-of-fortune experiment, a random spin of 10 versus 65 moved the median guess from 25% to 45% (Tversky & Kahneman).
> - **The anchors that wreck traders** are your cost basis ("I'll sell at breakeven"), round numbers, the 52-week high/low, analyst price targets, and the IPO or all-time-high price — all past or arbitrary numbers pretending to be fair value.
> - **The breakeven trap** freezes exits: anchored to a \$50 entry, a trader rides a loser to \$30 waiting to "get back to even," turning a planned \$5 loss into a \$20 one. This is the disposition effect, and Odean's 10,000-account study caught it in the data.
> - **The tell**: two traders holding the *identical* stock at the *same* price will make opposite decisions purely because they bought at different prices. The market cannot see either entry.
> - **The fix** is a drill, not willpower: the mark-to-market-blind review — value every holding as if the broker handed you its cash today, and ask "would I buy *this*, here, at *this* price, with new money?"

You buy a stock at \$50. Three months later it trades at \$42, the story that made you buy it has quietly fallen apart, and a friend asks why you are still holding. You hear yourself say it before you can stop: *"I'll sell when it gets back to fifty."*

Stop and look at that sentence, because it is one of the strangest things a rational person can say. The number 50 is not a fact about the company. It is not the fair value, not the analyst target, not a support level a thousand other traders are watching. It is the price *you personally paid* — a number that exists nowhere in the world except your own brokerage statement. The market has no idea you bought at 50. The buyers and sellers setting the price tomorrow have never seen your fill. And yet that private, irrelevant number has quietly taken control of the single most important decision you will make on this position: when to get out.

That is anchoring. It is not stupidity and it is not lack of discipline — it is a feature of how every human brain estimates uncertain quantities, and it fires just as hard in professionals as in beginners. This post takes it apart: the science of *why* an anchor grips you even when you can see it is meaningless, the specific anchors that cost traders money, exactly how they distort your buys and sells in dollars, what the bias looks like on your screen in real time, and a mechanical drill for cutting the anchor out of the decision entirely.

![Your entry price is private; the market's price is not — a two-column map contrasting the backward-looking numbers your brain fixates on with the forward-looking forces that actually set tomorrow's price.](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-1.webp)

The diagram above is the mental model for the whole article. On the left is what your brain fixates on — your cost basis, the round number, the 52-week high, the analyst target, the old all-time high. Every one of them is *private and backward-looking*: a number about the past, or about you, or about nothing at all. On the right is what actually sets the price tomorrow — future cash flows, interest rates, new information, and the supply and demand of every *other* trader in the market. Notice that nothing on the right can *see* anything on the left. The market has no memory of your entry. Your anchor is a story you are telling yourself, and this post is about how much that story costs.

## Foundations: what an anchor actually is (the science)

Before we talk about markets, we need the mechanism, because anchoring is one of the best-documented findings in all of behavioral science, and understanding *why* it works is what lets you fight it. Let me define the core terms as we go — you need no prior psychology or finance background to follow this.

An **anchor** is any number that lands in your mind *before* you make a numeric judgment, and then pulls your judgment toward itself — even when the number is irrelevant, and even when you know it is irrelevant. The **anchoring-and-adjustment heuristic** (a *heuristic* is a mental shortcut the brain uses to answer a hard question quickly) is the process behind it: faced with an uncertain estimate, your mind grabs a starting value and then adjusts away from it — but the adjustment is almost always *insufficient*. You stop adjusting too soon, still in the gravity well of the anchor.

### The experiment that started it

In 1974, the psychologists Amos Tversky and Daniel Kahneman ran an experiment that has since been replicated hundreds of times. They sat participants in front of a "wheel of fortune" marked 0 to 100 and spun it — but the wheel was secretly rigged to stop on either **10** or **65**. Right after the spin, they asked each person a question with a genuine numeric answer: *what percentage of countries in the United Nations are African?* First, is it higher or lower than the number you just spun? Then, what is your actual best guess?

The wheel had nothing to do with African geography. Everyone could see it was a spinning wheel of chance. And yet: the people who happened to spin **10** gave a median guess of **25%**, while the people who spun **65** gave a median guess of **45%**. A meaningless random number, visibly generated in front of them, moved the average answer by twenty percentage points.

![An anchor you know is random still drags your estimate toward it — the 1974 wheel-of-fortune experiment, in which a spin of 10 versus 65 moved the median UN estimate to 25% versus 45%.](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-2.webp)

Read the figure carefully, because it contains the entire bias in one picture. Two groups, same question, same true answer. The only difference was a number from a wheel. One group anchored on 10 and adjusted *up* — but stopped at 25, far short of where an un-anchored person would land. The other anchored on 65 and adjusted *down* — but stopped at 45, again far short. That gap between where you stop and where you *should* stop is "insufficient adjustment," and it is the reason an anchor never fully lets go. This result was published in *Science* — "Judgment under Uncertainty: Heuristics and Biases," Tversky and Kahneman, 1974 — and it launched the entire field of behavioral economics.

### Why the adjustment always falls short

The wheel experiment shows *that* anchoring happens; a second line of research shows *why* the adjustment stops too early. Nicholas Epley and Thomas Gilovich ("The Anchoring-and-Adjustment Heuristic: Why the Adjustments Are Insufficient," *Psychological Science*, 2006) found that when people adjust away from an anchor, they stop the moment they reach the *first value that seems plausible* — the near edge of the range they would accept, not its center. You start at the anchor and creep outward only until the estimate stops feeling obviously wrong, then you quit.

Map that onto a losing trade and it is almost too on-the-nose. Your anchor is your \$50 entry. As the stock falls, you revise your sense of "fair exit" downward — but you stop at the first price that still feels defensible, which is always *close to \$50*. \$48 feels plausible. \$45 feels like the edge. \$30 feels absurd — right up until the stock trades there. The bias is not that you refuse to adjust; it is that you stop adjusting while still deep in the anchor's gravity. Epley and Gilovich also found that paying people for accuracy widens the adjustment only a little, and only for anchors they generate themselves — which means "just try harder" helps at the margin and nowhere near enough. The reliable fix is to delete the anchor, not to strain against it.

### The anchor bites hardest exactly when you are least sure

One more property matters for traders: anchoring is *strongest when the true answer is most uncertain*. When you have a firm, independent estimate of a value, an irrelevant number bounces off it. When you have no idea — a beaten-down stock in a confusing tape, a company whose story just changed — your mind has nothing to hold onto *except* the anchor, so it grips hardest. That is exactly the situation in which traders reach for their cost basis: a position has gone against them, the future is murky, and the one crisp number in the fog is what they paid. The bias is not random about when it strikes; it strikes worst when you can least afford it.

#### Worked example: the arbitrary anchor on a number you own

Suppose I ask you to write down the last two digits of your phone number, and then ask what you would pay for a bottle of wine you have never tasted. Feels unrelated, right? In one of the most famous demonstrations of anchoring — Dan Ariely, George Loewenstein, and Drazen Prelec, "Coherent Arbitrariness," *Quarterly Journal of Economics*, 2003 — MIT business students wrote down the last two digits of their **Social Security number**, then bid on ordinary goods (wine, chocolates, a keyboard). The students whose digits fell in the top fifth (say, 80–99) bid **as much as 346% more** for the exact same items than students whose digits fell in the bottom fifth (say, 1–20). A number that was literally an ID digit — no information about the wine at all — moved willingness-to-pay several-fold.

Now translate that to a trading screen. Your cost basis is *exactly* this kind of number: personally salient, emotionally sticky, and completely uninformative about what the stock is worth tomorrow. The intuition to carry forward: **an anchor does not need to be relevant to move you — it only needs to arrive first.**

### It grips experts too, and it survives knowing better

The comforting story is that anchoring is a rookie mistake you outgrow. It is not. Two findings kill that hope.

First, **professionals anchor on the job.** Gregory Northcraft and Margaret Neale (1987) took real estate agents through an actual house, gave them the full listing packet, and let them appraise it — but secretly varied the listing price shown to different agents. The agents' appraisals moved *with* the manipulated listing price, on every measure: what they thought the house was worth, what they would list it at, the lowest offer they would accept. And here is the part that should worry every trader: when asked what factors influenced their appraisal, only **19% of the agents even mentioned the listing price.** They were anchored and did not know it. (The study is fittingly titled "Experts, amateurs, and real estate," *Organizational Behavior and Human Decision Processes*, 1987.)

Second, **knowing the anchor is random does not save you.** Birte Englich, Thomas Mussweiler, and Fritz Strack (2006) had experienced criminal judges read a case file, then roll a pair of dice — openly rigged, the judges could see them — before recommending a sentence. Judges who rolled high numbers recommended longer sentences than judges who rolled low ones. Trained legal experts, sentencing real-world crimes, were moved by dice they had watched roll. The paper's title says it all: "Playing Dice With Criminal Sentences," *Personality and Social Psychology Bulletin*, 2006.

Put those together and the lesson for traders is stark. You cannot beat anchoring by being smart, by being experienced, or by *reminding yourself the anchor is irrelevant*. Awareness alone does almost nothing. The only defense that works is structural — a process that removes the anchor from the decision before your brain can grab it. We will build that process at the end. First, we need to name the specific anchors that show up in a trading account.

> An anchor is not a belief you can argue yourself out of. It is a gravitational field. The only way to escape it is to change your orbit.

## 1. The five anchors that wreck traders

Every anchor that costs traders money is a number that is either *about the past* or *about nothing*, dressed up as if it were about fair value. Here are the five that do the most damage.

![The five anchors that quietly wreck traders — cost basis, round numbers, the 52-week high/low, the analyst price target, and the IPO or all-time-high price, each with the reason it is really just a memory.](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-3.webp)

**Your cost basis.** The price you paid. This is the king of trading anchors and the source of the whole "I'll sell at breakeven" family of mistakes. Your entry price is the one number in this list that is not just irrelevant but *literally invisible to the market* — nobody else can see it, and it has zero bearing on future returns. A stock does not know it owes you anything. We will spend the next two sections on the damage this one does.

**Round numbers.** \$100 a share. \$50. The Dow at 40,000. Bitcoin at \$100,000. Round numbers are anchors manufactured by our base-10 number system, not by any economic force. Traders cluster orders at them, set targets at them, and feel that a stock "should" stall at \$100 — but \$100 is a digit, not a valuation. There is real evidence that prices behave slightly oddly near round numbers precisely *because* so many people anchor there, which is a self-fulfilling technical effect, not a statement about value.

**The 52-week high (and low).** The highest (or lowest) price a stock touched in the past year is one of the most powerful anchors in all of investing — powerful enough to move real money in measurable ways. Thomas George and Chuan-Yang Hwang (2004) showed that a stock's *nearness to its 52-week high* predicts its future returns better than its past returns do: investors treat the 52-week high as a ceiling and *underreact* to good news when a stock is near it, so the stock keeps drifting up afterward. ("The 52-Week High and Momentum Investing," *Journal of Finance*, 2004.) The anchor is so strong it even distorts corporate takeovers — more on that in the real-markets section.

**The analyst price target.** A single number an analyst publishes ("12-month target: \$65") becomes a mental anchor even though it is one person's model output with a twelve-month shelf life and a wide error bar. The target feels like authority. It is a guess, and it goes stale the moment the facts change — but the anchor stays put in your head long after.

**The IPO or all-time-high price.** The price a stock first listed at, or the highest it ever traded, becomes a permanent reference point — "it was a \$300 stock once, so at \$80 it's cheap." That reasoning treats a past price as a floor under future value. It is yesterday's mood, priced. As we will see with Cisco, a former high can keep investors anchored — and underwater — for a quarter of a century.

What unites all five: none of them is a forecast of future cash flows, and none of them is visible to the people setting tomorrow's price. They are memories and conventions wearing the costume of analysis.

#### Worked example: the 52-week high as a ceiling in your head

A stock has been grinding higher and just tagged a new **52-week high at \$80** on genuinely good earnings. Your instinct: *"It's already run — I missed it, I'll wait for a pullback."* That instinct is the anchor. You are treating \$80 as a ceiling for one reason only: it is the highest number you have seen this year.

The George–Hwang finding says this is often exactly backwards. Because so many investors anchor to the 52-week high and *underreact* to good news near it, stocks that break to new highs on real news have historically tended to keep drifting higher over the following months rather than snap back — which is *why* "nearness to the 52-week high" carried predictive power in their data. None of that promises any particular breakout keeps going; it is a warning that "it already hit a new high, so it must be expensive" is an anchor talking, not a valuation. The intuition: **a new high is a fact about the past twelve months, not a cap on the next twelve — when you catch yourself saying "it already ran," check whether the ceiling is the company's or just the calendar's.**

### When cost basis is actually allowed to matter

To be fair to the humble entry price: there are two narrow, *mechanical* uses where your cost basis is genuinely relevant — and noticing the difference sharpens the rule rather than weakening it. First, **taxes**: your cost basis determines your realized gain or loss, so it belongs in tax-loss harvesting and in deciding *which lot* to sell for tax efficiency. That is an accounting fact, not a valuation input. Second, **execution near round numbers**: because other traders cluster orders at round numbers, those levels can be real *short-term* magnets and staging points for entries and exits — useful for the mechanics of *how* you transact, never for *whether* the position is worth owning. The tell that you have crossed the line is simple: the moment your entry price or a round number enters the *hold-or-sell* decision — as opposed to the *tax* or *order-placement* decision — you are back to anchoring, and the drill applies.

## 2. The breakeven trap: how anchoring freezes your exits

Here is where anchoring stops being an interesting lab result and starts emptying your account.

The most destructive thing an anchor does is *freeze your exit on a losing trade*. You bought at \$50. The stock is now \$45, the reason you bought is weaker than it was, and by any forward-looking measure you should either add to the position (if you still believe) or cut it (if you don't). But your cost basis has installed a third option that feels like the only reasonable one: **wait until it comes back to \$50, then sell at breakeven.** You are no longer trading the company. You are trading your own entry price.

Watch what that costs.

![The breakeven anchor turns a small planned loss into a big unplanned one — a price chart showing a disciplined exit at the \$45 stop (a \$5 loss) versus riding the position to \$30 waiting for breakeven (a \$20 loss).](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-4.webp)

#### Worked example: the breakeven anchor versus the early cut

You buy 100 shares at **\$50**, so your position is worth **\$5,000**. Before you enter, you had a plan: if it drops to **\$45**, the trade idea is wrong and you are out. That is a **\$5 per share, \$500** loss — 10% of the position. Painful but survivable; it is exactly the size of loss a decent risk plan is built to absorb.

The stock hits \$45. But now the breakeven anchor is talking: *"If I just hold, it'll get back to fifty and I'm even. Selling here locks in a loss."* So you hold. It goes to \$42. "Even more reason to wait — I'm not selling down here." \$38. \$34. Finally, exhausted, you sell at **\$30**.

That is a **\$20 per share, \$2,000** loss — **40%** of the position, and **four times** the loss you had *planned* to take. The stock never came back to fifty. It did not owe you fifty. The only thing that changed between the smart \$500 exit and the ruinous \$2,000 exit was that you let a private anchor override a written plan.

And notice the cruel asymmetry the math imposes: after a 40% loss, the stock has to rise **67%** just to get you back to \$50, because you are now compounding off a smaller base. The deeper you ride a loser waiting for breakeven, the *more* the market has to do to grant you the wish — which is exactly why the wish so rarely comes true. The intuition: **the breakeven anchor converts a small, planned loss into a large, unplanned one, and it does so by making "I'm not selling at a loss" feel like prudence.**

### This has a name in the data: the disposition effect

You might think a story like that is just one undisciplined trader. It is not — it is a measurable, market-wide pattern with a name. The **disposition effect** is the documented tendency to *sell winners too early and hold losers too long*: to realize gains readily and refuse to realize losses.

The landmark study is Terrance Odean's "Are Investors Reluctant to Realize Their Losses?" (*Journal of Finance*, 1998). Odean took the trading records of **10,000 accounts** at a discount brokerage and measured, for every day someone sold, whether they were selling a position that was up or down relative to where they bought it. The result was unambiguous: investors sold their **winners at roughly one and a half times the rate** they sold their losers — even after stripping out taxes and rebalancing, and even though the winners they sold went on to *outperform* the losers they kept. They were not rebalancing. They were anchored.

![The disposition effect: we sell winners and marry losers — a two-by-two matrix of what a trader feels, does, and ignores when a position is up 20% versus down 20%.](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-5.webp)

The matrix above shows the machinery. When a position is a **winner**, the anchor (your entry, now below the price) makes selling feel *good* — you get to book a gain against your cost basis — so you sell early and cap your biggest compounders. When it is a **loser**, selling means marking the loss against that same cost basis and *admitting the entry was wrong*, which feels *bad*, so you hold and hope. Same anchor, opposite behavior, both wrong. The engine underneath is loss aversion — the well-documented finding that a loss feels about **2.25 times** as intense as an equal-sized gain (Tversky & Kahneman, 1992) — and the reference point that defines "loss" is your entry-price anchor. (We go deep on that mechanism in [loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect); here the point is that the anchor is what *sets the zero line* for the whole thing.)

#### Worked example: the anchor even distorts your tax bill

There is a second, quieter cost. Suppose you hold two positions, each up or down \$1,000, in a taxable account. Rationally, if you need to sell *something*, tax rules reward selling the **loser**: realizing a \$1,000 loss can offset other gains and lower your tax bill, while realizing a \$1,000 gain *creates* a taxable event. So the tax-smart move is to harvest losses and let winners run.

The disposition effect makes you do the exact opposite. You sell the \$1,000 winner (triggering, say, a **\$150–\$250** tax bill depending on your rate and holding period) and hold the \$1,000 loser (forgoing a deduction that could have *saved* you a similar amount). Odean's data showed precisely this: the behavior was not only psychologically driven but "sub-optimal and leads to lower after-tax returns." The intuition: **the entry anchor is so strong it will override even your own self-interest on taxes — you pay real money to protect a number that only exists on your statement.**

### Why "getting back to even" feels like winning

There is a deeper reason the breakeven anchor is so sticky, and it comes from how we keep score in our heads. Hersh Shefrin and Meir Statman — who named the disposition effect in "The Disposition to Sell Winners Too Early and Ride Losers Too Long" (*Journal of Finance*, 1985) — traced it to *mental accounting*: each position is a separate mental "account" that stays open until you sell. Selling *closes* the account and forces a verdict. Close a winner and you book a clean win and a hit of pride. Close a loser and you book the loss *and* a stab of regret at having been wrong — so the mind keeps the losing account open, because an open account still holds the hope of ending green.

"Getting back to even" is the fantasy of closing that account at exactly zero: no loss, no regret, thesis quietly forgotten. It feels like winning because it feels like escaping the verdict. But the account's *real* value is the current price, marked every second by the market whether you look or not. Refusing to close it does not protect you from the loss — the loss is already yours. It only keeps your capital hostage to a number that has already changed, and denies it to the position you *would* buy today. Every day you hold for breakeven, you are paying the opportunity cost of your best current idea to protect the pride of your worst past one.

## 3. "Cheap" versus "expensive": how a former high distorts value

The breakeven trap is anchoring on the *sell* side. There is a mirror-image trap on the *buy* side, and it is just as expensive: anchoring makes a falling stock look "cheap" simply because it used to be higher.

A stock that traded at \$300 and now trades at \$80 *feels* cheap. Your brain anchors on \$300, computes "\$80 is 73% off," and files the stock under "bargain." But "73% below its old high" is not a valuation — it is a fact about the past. The stock might be wildly *expensive* at \$80 if the business that justified \$300 is gone. Anchoring to the former high is how a **falling knife** — a stock in a sustained, fundamentally-driven decline — gets mistaken for a discount rack. The old price acts as an invisible floor in your mind that the market feels no obligation to respect.

The same bug distorts the *upside* anchor. Anchor on an analyst's \$65 target and a stock at \$45 looks like it has "\$20 of upside." But the target is a stale number; the honest question is not "how far is the price from the target?" — it is "what is this actually worth *now*, given what I know today?" Those are completely different questions, and the anchor tricks you into answering the easy, backward-looking one.

#### Worked example: anchoring to a target versus re-deriving fair value

You bought a stock at **\$60** partly because a respected analyst had a **\$75** price target. Since then, the company cut its guidance, a key product slipped, and rates rose. The stock is now **\$45**.

**The anchored path:** you look at the \$75 target still sitting in your notes and think, *"Down to \$45 but the target's \$75 — that's 67% upside, I should buy more."* You are measuring the distance to a number that was computed under assumptions that no longer hold.

**The re-derived path:** you throw the old target away and rebuild fair value from *today's* facts. Say the company will now earn about **\$3.00** per share (down from the \$4.50 the old model assumed), and comparable businesses trade around **15 times** earnings in the current rate environment. That is roughly **\$45** of fair value — *the price it is already at.* The stock is not 67% cheap; it is fairly priced for its diminished prospects, and your "bargain" was an illusion manufactured by a dead anchor.

Those two paths can be off by a factor of two on the same stock, on the same day. The intuition: **fair value is something you re-derive from current facts, not a distance you measure from an old number — the moment you find yourself computing "upside to the target," check whether the target is still alive.**

## 4. Same position, opposite decisions — the proof it's the anchor

If you want a single test that proves your entry price is controlling you, here it is. Take two traders who own the *exact same stock*, in the *exact same amount*, at the *exact same current price* — and give them different cost bases. Watch them make opposite decisions. Nothing about the stock's future differs between them. The only thing that differs is a private number in each of their heads.

![Same stock, same price, opposite decisions — two traders both holding shares worth \$60 today, one who bought at \$40 and one who bought at \$90, reaching opposite hold-or-sell conclusions.](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-7.webp)

#### Worked example: two traders, one stock, opposite errors

The stock is **\$60** today. Both traders own 100 shares.

**Trader A** bought at **\$40**. She is sitting on a **+50%** gain. Her anchor (40, now well below the price) makes selling feel like *winning*, so the disposition effect kicks in: "Huge gain — lock it in before it fades." She **sells** — a stock that, if she had no position, she might happily *buy* at \$60.

**Trader B** bought at **\$90**. He is sitting on a **−33%** loss. His anchor (90, now well above the price) makes selling feel like *admitting defeat*, so the breakeven trap kicks in: "Just wait for it to get back to ninety." He **holds** — a stock that, if he had no position, he would *never buy* at \$60.

Same stock, same \$60, same future. One sells, one holds, and *both decisions are driven entirely by a number the market cannot see.* If the stock is genuinely worth buying at \$60, they should both own it; if it isn't, neither should. The correct decision for both is *identical* and has nothing to do with 40 or 90. The intuition: **whenever your cost basis is doing the deciding, you can prove it by asking what a trader with a different entry — or no position at all — would do at this price; if the answer differs, you are trading your anchor, not the stock.**

## What it looks like at the screen

Anchoring rarely announces itself as "I am now being irrational about my cost basis." It shows up as a set of small, physical, almost invisible habits at the screen. Learn to catch these in real time — they are the tells that the anchor has the wheel:

- **You look at the P&L column first.** Before you read the news, check the chart, or reconsider the thesis, your eye goes straight to the green or red number next to each position. That number is computed *from your cost basis*. Leading with it means you are framing every decision around your entry before you have looked at a single fact about the future.
- **The sell button only feels available when the position is green.** You will happily click "sell" on a winner. On a loser at the same conviction level, your hand hesitates — selling would "make the loss real." The loss is already real; the market already marked it. The only thing selling changes is that you stop compounding the mistake.
- **You catch yourself saying "just to breakeven."** Any sentence containing "back to even," "get out at breakeven," "once it recovers to what I paid," or "I'm not selling at a loss" is the anchor talking out loud. It is the single most reliable verbal tell there is.
- **You set an alert at your entry price.** Not at a technical level, not at a fundamental trigger — at *the exact number you paid*. That alert is an anchor with a notification attached.
- **You won't open the position at all when it's red.** Averting your eyes from a losing position — not checking it, not reading news on it — is anchoring in avoidance mode: the position is below your anchor, looking at it hurts, so you don't. Meanwhile the thesis may be breaking in plain sight.
- **You feel radically different about the same price depending on where you bought.** The clearest tell of all: a stock at \$60 feels like a triumph or a disaster to you *personally*, and the difference is your entry. The stock feels nothing. It is \$60 to everyone.

None of these is a moral failing. They are the surface symptoms of a brain doing exactly what brains do with anchors. But once you can *name* the tell in the moment — "that's the P&L-first habit," "that's the breakeven sentence" — you have created the half-second of distance you need to run the drill instead of the reflex.

## Common misconceptions

**"I know my cost basis is irrelevant, so it doesn't affect me."** This is the most dangerous belief in the whole topic, because the research is unanimous that *awareness does not neutralize anchoring*. Judges anchored on dice they watched roll; real estate agents anchored on listing prices and then denied it. Knowing the anchor is meaningless changes your *explanation* of your behavior, not your behavior. The defense has to be structural.

**"Selling at a loss means I was wrong; holding keeps the trade alive."** Holding does not keep anything alive — the loss already happened the moment the price fell. Refusing to sell does not un-lose the money; it just keeps your capital tied up in your worst idea instead of your best one. "Realizing" a loss is an accounting event, not a new loss. The market already realized it for you.

**"A stock that's down a lot is cheap."** Down-a-lot is a fact about the past relative to an old anchor; cheap is a claim about value relative to *future* cash flows. They are unrelated. Plenty of stocks that were "down 80%" went on to fall another 80%. The former high tells you nothing about the floor.

**"Waiting for breakeven is just being patient."** Patience is holding a position because the *thesis is intact and playing out on its expected timeline*. Waiting for breakeven is holding because of *where you bought* — a completely different reason that happens to feel similar. The test: if you had entered at a different price, would you still be holding for the same forward reasons? If the only thing keeping you in is your entry, that is not patience, it is the anchor.

**"Round numbers and 52-week highs are real levels, so anchoring to them is fine."** They are real in the narrow sense that *other people anchor to them too*, which creates genuine order clustering and short-term technical effects. But that is a statement about crowd behavior, not about value, and it cuts both ways: the level is only "real" until the crowd's anchor breaks, at which point it offers no support at all.

**"Averaging down lowers my cost basis, so it's a smart move."** Notice the goal buried in that sentence — it is to move a *private number*, your average cost, not to own more of a good position at a good price. Lowering your cost basis does nothing to the stock; it just spreads your anchor across more shares and increases your exposure to an idea that has, so far, been wrong. Averaging down is only sound when the honest forward test passes: *would I open this position, this size, at this price, with new money?* If the real motive is to make the breakeven number smaller, that is anchoring wearing the costume of conviction.

## How it shows up in real markets

Anchoring is not confined to individual retail accounts. It shows up in professionally-managed money, in corporate takeover battles, and in decades-long episodes that are studied precisely because the anchor was so visible. Here are four, with real numbers and dates.

![Cisco: a quarter-century spent waiting for a breakeven — a timeline from the March 2000 peak near \$80, through the roughly 88% drawdown to 2002, to the first record close since 2000 in December 2025.](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-6.webp)

### 1. Cisco: a 25-year wait for breakeven

At the top of the dot-com bubble, Cisco Systems was the most valuable company on earth. Its split-adjusted stock peaked near **\$80** on March 27, 2000, briefly carrying a market capitalization above **\$500 billion** (data as of the 2000 peak; MacroTrends/CNBC). Then the bubble burst. By 2002 the stock had fallen roughly **88%**, bottoming near **\$8–\$10**. And here the breakeven-recovery math turns brutal at national scale: from an 88% drawdown, a stock has to rise more than **700%** just to reclaim its old high — the exact same compounding asymmetry as our \$50-to-\$30 worked example, only now measured against a former \$500-billion peak.

Here is the part that makes it a perfect anchoring case: *the business did not collapse.* Cisco's revenue was about **\$19 billion** in fiscal 2000, about **\$22 billion** in 2001, and about **\$19 billion** in 2002 — it kept selling the routers and switches that ran the internet. What vanished was the *premium* investors were willing to pay, not the company. And yet, for anyone anchored to that \$80 peak, the stock became a decades-long "I'll sell when it gets back to even" — a wait that lasted about **twenty-five years**: Cisco did not close at a new record high until **December 2025** (CNBC, December 10, 2025). An investor who anchored to the old high spent a quarter of a century holding for a breakeven that a forward-looking process would never have asked them to wait for.

### 2. The 52-week high sets takeover prices

The 52-week-high anchor is powerful enough to move billion-dollar corporate deals. Malcolm Baker, Xin Pan, and Jeffrey Wurgler ("The Effect of Reference Point Prices on Mergers and Acquisitions," *Journal of Financial Economics*, 2012) found that acquisition **offer prices are biased toward the target's 52-week high**, and that the single most common offer price is *exactly* that high — "a highly salient but largely irrelevant past price." Worse, a deal's probability of being accepted **jumps discontinuously** the moment the offer crosses the 52-week high, and acquirers' own shareholders react more negatively as the bid gets pulled up toward it. This is anchoring at the top of the professional food chain — boards, bankers, and CEOs negotiating deals — using a past price as the reference point for what a whole company is worth.

### 3. The disposition effect, live in 10,000 accounts

We already met Odean's 1998 study, but it belongs here too, because it is the anchor caught in the act across a real population. In the trading records of **10,000 brokerage accounts**, investors sold winners at about **one and a half times** the rate they sold losers, and the losers they clung to went on to underperform the winners they dumped. This is not an anecdote or a lab artifact — it is the entry-price anchor showing up as a systematic, money-losing pattern in ordinary people's real portfolios, every day, at scale.

### 4. Analyst targets as sticky reference points

When a stock's fundamentals deteriorate, the analyst price targets attached to it tend to *lag* — they get revised down, but slowly and reluctantly, and investors keep measuring "upside" against numbers that are already stale. The same anchoring machinery that grips individuals grips the sell-side: a published target becomes a reference point that resists updating even as the facts move. The practical takeaway is not "ignore analysts" — it is to treat any target as a dated opinion with a short shelf life, and to notice when your bull case is really just "the price is below the target."

## The drill: the mark-to-market-blind review

Everything above says the same thing: you cannot out-think an anchor, and awareness is not enough. So the fix is not a mindset — it is a *procedure* that removes your cost basis from the decision before your brain can grab it. Here is the drill.

The core move is a reframe. For every position you hold, picture that your broker just **liquidated it and handed you the cash value today**. You now hold cash equal to the position's current market value — with *no cost basis, no history, no entry price attached.* Then ask one question, and only one:

> **"Would I buy *this* position, right now, at *this* price, with this fresh cash?"**

![The mark-to-market-blind review: value every holding as new cash — a decision flow that strikes the cost basis, reframes the holding as its cash value today, and reduces the decision to a single forward-looking buy-or-not question.](/imgs/blogs/anchoring-your-entry-price-is-lying-to-you-8.webp)

That question is forward-looking by construction. It cannot be answered with "but I'm down," because in the reframe you are not down — you are holding cash and deciding what to do with it. It strips the anchor out mechanically.

- **If the answer is yes** — you would buy it here with new money — the position earns its place. Keep it. Your conviction is about the future, not your entry.
- **If the answer is no** — you would *not* put fresh cash into this at this price — then the *only* reason you are still holding is the anchor. You are holding to avoid marking a loss against your entry, not because you believe in the position. Trim it or exit it, and redeploy the cash into something you *would* buy.

Run this on every holding, one at a time, on a fixed schedule (say, monthly, and always after a position moves sharply). A few rules make it work:

1. **Cover the P&L column.** Literally. Do the review with the unrealized gain/loss hidden, so you cannot see whether each position is green or red. If you can see it, you are anchored before you start.
2. **Write the fair value *first*, then look at the price.** Re-derive what the position is worth from today's facts before you check where it trades. If you look at the price first, it becomes the anchor for your "estimate."
3. **The entry price is never allowed in the decision.** If the words "I paid," "breakeven," or "back to even" appear in your reasoning, you have failed the drill. Start over.
4. **Same test for adds as for holds.** "Would I buy more here with new money?" is the honest version of "averaging down." If you would not add fresh cash, you are not seeing a bargain — you are anchoring on the former high.

### Designing your environment against the anchor

Because willpower loses to anchoring, the highest-leverage move is to change what your *screen* shows you, so the anchor is never handed to you in the first place:

- **Hide your cost basis.** Many platforms let you display holdings without the entry price or the unrealized-P&L column. Turn it off. If your broker will not hide it, keep your working watchlist in a separate sheet that lists only ticker, current price, and forward thesis — never what you paid.
- **Journal in forward terms.** For every position, write down *what has to be true* to keep holding and *what would make you sell* — in fundamentals, price levels, and dates, never "back to \$50." Re-read it at each review. A thesis written in forward terms cannot be satisfied by a breakeven.
- **Pre-commit the exit before you enter.** Decide the stop and the invalidation level *at entry*, when no anchor exists yet because you hold no position. A resting stop-loss order executes the decision your calm self made, before your anchored self can argue with it.
- **Separate the sell decision from the buy decision.** "Should I still own this?" and "am I up or down on it?" are different questions. Answer the first without letting the second in the room; the mark-to-market-blind reframe is the tool that keeps them apart.

Epley and Gilovich's finding that accuracy incentives barely move provided anchors is the whole case for this environmental approach: you are not trying to *want* the anchor less, you are arranging your tools so it never reaches the decision at all.

The drill is deliberately mechanical because that is the only thing that beats a mechanical bias. You are not trying to *feel* less anchored — you have read the research, you know you will always feel the pull. You are building a process that makes the decision *without consulting the anchor at all*. This is the same philosophy as [defining your invalidation level up front](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront): decide the forward-looking rule while you are calm, so that in the moment you execute the rule instead of the reflex. And it is one node in the larger [map of biases that chain into a single bad trade](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders) — anchoring rarely acts alone, but it is very often the first link.

## When this matters to you

If you hold any position for more than a day, this bias has your money. It matters most at exactly the moments that decide your returns: when a trade goes against you and you are choosing between the small planned loss and the big unplanned one; when a stock has fallen far and your brain whispers "bargain"; when you are sizing an add on a loser; and when you are looking at a portfolio full of green-and-red numbers deciding what to keep. In every one of those moments, the honest question is the same, and it never contains your entry price: *would I buy this, here, now, with new money?*

Picture your next portfolio review done the anchored way and then the clean way. Anchored: you scan the P&L column, feel good about the greens, avoid the reds, sell a winner or two "to lock in gains," and quietly leave the losers alone to "come back." Clean: you cover the P&L, and for each holding you write today's fair value first, then ask whether you would buy it fresh at today's price. The greens and reds vanish; what is left is a list of positions you would buy again and a list you would not. The first list you keep, the second you trim — and for the first time the shape of your portfolio reflects what you believe about the future instead of the accident of what you paid in the past. Same holdings, same prices, completely different decisions — which is the whole lesson of this article, run on your own account.

A closing note on what this is and isn't. This is an explanation of a decision-making bias and a process for managing it — it is educational, not individualized investment advice, and nothing here says whether any particular stock is worth buying or selling. Every position that can make money can lose it, and no drill changes that. What the drill changes is *why* you hold what you hold: whether your reasons live in the future, where returns come from, or in the past, where your anchor lives. The market has no memory of your entry. The sooner your decisions stop having one either, the sooner your cost basis stops lying to you.

## Sources & further reading

Primary sources behind the headline numbers:

- Amos Tversky and Daniel Kahneman, "Judgment under Uncertainty: Heuristics and Biases," *Science*, vol. 185 (1974) — the wheel-of-fortune anchoring experiment (median estimates of 25% vs 45% from anchors of 10 vs 65).
- Dan Ariely, George Loewenstein, and Drazen Prelec, "Coherent Arbitrariness: Stable Demand Curves Without Stable Preferences," *Quarterly Journal of Economics*, vol. 118, no. 1 (2003) — the Social Security number anchoring of willingness-to-pay (top-quintile bids up to ~346% higher).
- Gregory B. Northcraft and Margaret A. Neale, "Experts, amateurs, and real estate: An anchoring-and-adjustment perspective on property pricing decisions," *Organizational Behavior and Human Decision Processes*, vol. 39, no. 1 (1987) — professional agents anchored on listing price; only 19% acknowledged it.
- Birte Englich, Thomas Mussweiler, and Fritz Strack, "Playing Dice With Criminal Sentences," *Personality and Social Psychology Bulletin*, vol. 32 (2006) — experienced judges anchored by openly random dice rolls.
- Terrance Odean, "Are Investors Reluctant to Realize Their Losses?" *Journal of Finance*, vol. 53, no. 5 (1998) — the disposition effect across 10,000 accounts; winners sold at ~1.5x the rate of losers.
- Thomas J. George and Chuan-Yang Hwang, "The 52-Week High and Momentum Investing," *Journal of Finance*, vol. 59, no. 5 (2004) — nearness to the 52-week high as a return-predicting anchor.
- Malcolm Baker, Xin Pan, and Jeffrey Wurgler, "The Effect of Reference Point Prices on Mergers and Acquisitions," *Journal of Financial Economics*, vol. 106, no. 1 (2012) — M&A offer prices biased toward the target's 52-week high.
- Nicholas Epley and Thomas Gilovich, "The Anchoring-and-Adjustment Heuristic: Why the Adjustments Are Insufficient," *Psychological Science*, vol. 17 (2006) — adjustments from an anchor stop at the first plausible value, and accuracy incentives barely help.
- Hersh Shefrin and Meir Statman, "The Disposition to Sell Winners Too Early and Ride Losers Too Long: Theory and Evidence," *Journal of Finance*, vol. 40, no. 3 (1985) — the disposition effect named, and its roots in mental accounting and regret.
- Amos Tversky and Daniel Kahneman, "Advances in Prospect Theory: Cumulative Representation of Uncertainty," *Journal of Risk and Uncertainty*, vol. 5 (1992) — the ~2.25 loss-aversion coefficient underlying the disposition effect.
- Cisco Systems price history and dot-com peak: MacroTrends (CSCO price history) and CNBC, "Cisco's stock closes at record for first time since dot-com peak in 2000" (December 10, 2025) — the ~\$80 March 2000 peak, ~88% drawdown to 2002, and ~25-year recovery.

Related reading on this blog:

- [Loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect) — the mechanism that makes your entry-price anchor so painful to sell against.
- [The cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders) — how anchoring chains together with confirmation bias, sunk cost, and capitulation into one bad trade.
- [What would change my mind: defining invalidation up front](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) — the pre-commitment habit that turns the mark-to-market-blind drill into a standing rule.
