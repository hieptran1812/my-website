---
title: "Reflexivity: The Feedback Loop Between Mind and Market"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "George Soros's idea that our biased beliefs move prices and that moving prices bend the real fundamentals, so markets loop between mind and reality instead of settling at a fair value — and how to spot the loop before it reverses."
tags:
  [
    "trading-psychology",
    "reflexivity",
    "george-soros",
    "feedback-loops",
    "boom-bust",
    "behavioral-finance",
    "market-psychology",
    "black-wednesday",
    "cost-of-capital",
    "efficient-markets",
    "short-selling",
    "self-reinforcing",
  ]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Reflexivity is George Soros's insight that markets are a two-way loop: our biased perceptions move prices, and moving prices bend the very fundamentals we were trying to value. Because cause and effect run both ways, markets do not reliably settle at a fair value the way the textbook promises.
>
> - **Psychology is not separate from the fundamentals — it is one of them.** A high stock price lowers a company's cost of capital, which lets it fund real growth, which raises real earnings, which "justifies" the high price. The belief helped create the fact.
> - **This is the opposite of the efficient-market view.** The textbook says price passively reflects value and any gap gets arbitraged away. Reflexivity says perception and price actively reshape the value they are measuring, so there is no fixed equilibrium to converge on.
> - **A boom-bust is a reflexive process:** self-reinforcing on the way up (price and story feed each other until price is far above value), then self-defeating on the way down (the same loop runs in reverse and overshoots below value).
> - **The named cases are all one mechanism:** Soros's 1960s conglomerate trade, Black Wednesday in 1992 (he made roughly \$1 billion breaking the pound's peg), the dot-com and SPAC manias, GameStop's short squeeze, and Silicon Valley Bank's 48-hour death spiral in 2023.
> - **The drill is four questions:** Is the price itself changing the fundamental? Is the narrative self-reinforcing? Who is forced to act at each turn? And what breaks the loop? Spot the loop, and you can anticipate the reversal that the crowd never sees coming.

Here is a puzzle the standard theory of markets cannot answer. A company's stock is expensive — priced at forty times its earnings — and the usual explanation is that investors are simply too optimistic, that reality will eventually correct them. But that expensive stock lets the company raise money cheaply. It issues shares at the high price, uses the cash to build factories and buy competitors, and its earnings genuinely rise. A year later the company really is bigger and more profitable, and the high price that looked like a mistake looks, in hindsight, like foresight. Was the market wrong to be optimistic? Or did the optimism, by lowering the company's cost of raising money, help *make itself true*?

This is not a trick question. It is the central puzzle of a theory called **reflexivity**, developed by the investor George Soros and laid out in his 1987 book *The Alchemy of Finance*. Soros ran one of the most successful funds in history — his Quantum Fund famously made around \$1 billion in a single day in 1992 — and he insisted that his edge came not from a better forecasting model but from a better *theory of how markets actually work*. That theory says something the textbooks deny: that in markets, the observer changes the observed. What we believe about a price feeds into the price, and the price feeds back into the reality we were trying to judge. Mind and market are not two separate things, one measuring the other. They are a single loop.

This article builds that idea from the ground up, assuming you have never read a balance sheet or shorted a currency, and takes it to the depth a practitioner respects. We will define every term, ground every claim in a worked example with real numbers, trace the loop through five famous episodes, and finish with a practical drill for spotting reflexivity in the wild — and, harder, for anticipating the moment it reverses. The diagram below is the whole idea on one page. The rest of the post is a guided tour of it.

![The reflexive loop: biased perception moves prices, moving prices bend the fundamentals, the shifted fundamentals validate the belief, and the amplified belief moves prices again](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-1.webp)

Read it clockwise. A biased perception — the crowd's story about an asset — drives buying or selling, which moves the price. The moved price does not just sit there being observed; it *changes the real world*, making capital cheaper or dearer, altering confidence, funding or starving real growth. Those shifted fundamentals then seem to confirm the original belief, which amplifies it, which moves the price again. The loop has no natural resting point. It reinforces itself on the way up until it can't, then reverses and reinforces itself on the way down. Everything that follows unpacks this single picture.

## Foundations: perception, price, and the two-way street

Before we can talk about the loop, we need shared vocabulary. None of it requires prior finance knowledge; if you already know these terms, skim.

### Price, value, and perception

Three words that sound similar and mean very different things.

**Value** is what an asset is fundamentally worth — the cash it will actually produce for its owner over time. A share of a company is worth the stream of profits it will eventually pay out; a bond is worth the interest and principal it will pay; a currency is worth what it can buy. Value is anchored to the real economy. It is estimable, slow-moving, and boring.

**Price** is simply the last number someone paid. It is set at the margin, by the most recent and most eager buyer and seller, and it can wander far from value for a long time. When a stock trades at \$100, that does not mean the world agrees the company is worth \$100 per share; it means the last trade happened at \$100.

**Perception** is what participants *believe* about the value — their expectations, their story, their model of the future. And here is the crucial move that the whole theory turns on: **perception is always at least a little wrong.** No participant has complete information or a perfect model. Soros called this the principle of *fallibility* — in any situation with thinking participants, their view of that situation is inherently partial and biased. Not stupid, not irrational, just incomplete. You cannot fully know a thing you are part of.

The efficient-market picture treats perception as a passive mirror: prices reflect value because smart investors compete away any gap. Reflexivity treats perception as an *active force*: because everyone's view is biased, and because everyone acts on their biased view, those actions push prices around — and, as we will see, push the fundamentals around too.

### The two functions, and why they interfere

Soros described the reflexive relationship as two functions operating at the same time. The names sound academic but the idea is simple.

![The two functions: the cognitive function runs from reality to biased views, the participating function runs from views back to reality, and because both run at once neither has an independent fixed value](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-3.webp)

The **cognitive function** runs from reality to perception: participants look at the world — cash flows, interest rates, growth, the cost of capital — and form beliefs about it. This is the ordinary act of trying to understand a market.

The **participating function** runs the other way, from perception back to reality: participants *act* on their beliefs, and their actions — buying, selling, lending, allocating capital — change the very situation they were trying to understand.

In physics, the thing you study does not care what you believe about it. A falling rock falls at the same rate whether you predict it correctly or not; the cognitive function operates alone, and reality is an independent given you can converge on. In markets, both functions run at once, and they *interfere*. Your belief that a company will thrive leads you to buy its stock, which lifts its price, which lowers its cost of capital, which helps it thrive. There is no untouched, independent reality sitting still for you to measure — the act of measuring, when it triggers action, moves the thing being measured. That interference is the whole ballgame. It is why Soros argued markets can trend far from value and stay there: there is no fixed point they are being pulled toward.

> In natural science, you study a world that does not know you exist. In markets, you are part of the thing you are trying to predict, and your prediction is one of the forces that shapes it.

### Efficient markets versus reflexivity

The dominant academic theory of markets is the **efficient-market hypothesis (EMH)**, most associated with the economist Eugene Fama. In its strong form it says prices already reflect all available information, so you cannot reliably beat the market; any mispricing is instantly arbitraged away by profit-seekers. In this picture the market is a superb measuring instrument, and price is a faithful readout of value.

Reflexivity does not say EMH is stupid — much of the time, in calm markets, prices really do track value closely and the mirror model works fine. Reflexivity says the mirror model is *incomplete*, and that the times it breaks down are exactly the times that matter most: bubbles, crashes, currency crises, bank runs. The contrast is worth drawing sharply.

![Efficient markets versus reflexivity: one view treats price as a passive mirror of value that gets arbitraged to fair value, the other treats perception and price as a feedback loop with no fixed equilibrium](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-2.webp)

| | Efficient-market view | Reflexive view |
|---|---|---|
| Direction of causation | Fundamentals → price (one way) | Perception ↔ price ↔ fundamentals (loop) |
| Role of perception | Passive mirror of value | Active force that bends value |
| Mispricings | Arbitraged away quickly | Can self-reinforce for years |
| Where it heads | Toward a fair-value equilibrium | No fixed equilibrium; trends and reversals |
| Best describes | Quiet, liquid, ordinary markets | Booms, busts, crises, manias |

Notice the row that does the real work: the role of perception. If perception is a mirror, psychology is just noise around a true value, and studying crowd emotion is a distraction from studying fundamentals. If perception is a force that bends the fundamentals, then **psychology is a fundamental** — it belongs in the valuation, not outside it. That claim sounds like a slogan. In the next section we make it literal, with numbers.

## 1. Why psychology IS a fundamental: the cost-of-capital bridge

The most common way people misunderstand reflexivity is to hear "beliefs move prices" and think, *sure, sentiment pushes prices around in the short run, but fundamentals win in the end.* That is still the mirror model — it just adds some short-term fog. Reflexivity makes a stronger and stranger claim: the belief can change the fundamental itself, permanently, through a completely concrete channel. That channel is the **cost of capital**.

The cost of capital is what a company pays to fund itself — the interest on its debt and the return its shareholders demand on their equity. Here is the bridge that connects psychology to reality: **a company's stock price directly sets its cost of equity capital.** When the price is high, the company can raise a lot of cash by selling only a few new shares. When the price is low, funding the same project means selling many more shares, giving away much more of the company. A high price is cheap capital; a low price is expensive capital. And cheap capital lets a company do real things — build, hire, acquire, invest — that a company with expensive capital cannot.

![The cost-of-capital bridge: a high stock price lets a firm issue cheap equity and buy rivals with rich shares, funding real spending that raises earnings, which validates the high price](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-5.webp)

Follow the arrows. A high stock price splits into two funding advantages: the company can *issue new shares cheaply*, and it can *use its richly valued stock as currency to buy other companies*. Both put more real resources in its hands. It spends them on genuine projects — research, factories, acquisitions — that raise its actual earnings and growth. Higher earnings then make the once-lofty price look justified, which supports an even higher price, which makes capital cheaper still. The loop has closed, and each turn is larger than the last. This is not sentiment sloshing around a fixed value. The price is *inside* the fundamentals, changing them.

#### Worked example: one turn of the self-reinforcing loop

Suppose a company — call it Ascend Corp — has 100 million shares outstanding, and each share earns \$2.50 of profit per year, so the company earns \$250 million in total. The market loves its story and prices the stock at \$100, which is 40 times its per-share earnings. (That ratio, price divided by earnings per share, is the **price-to-earnings or P/E ratio**; a P/E of 40 means investors pay \$40 for every \$1 of annual profit.) The whole company is therefore valued at \$100 × 100 million = \$10 billion.

Now watch what the high price does. The flip side of a P/E of 40 is an **earnings yield** of 1 ÷ 40 = 2.5\%: for each dollar of stock it sells, Ascend gives up only 2.5 cents of annual earnings to its new shareholders. That is astonishingly cheap capital. So Ascend issues 10 million new shares at \$100 each and raises \$1 billion in fresh cash. It invests that \$1 billion in real projects that earn a 15\% return, which is \$150 million of brand-new annual profit.

Add it up. Ascend now earns \$250 million + \$150 million = \$400 million. It has 110 million shares. Its new earnings per share are \$400 million ÷ 110 million = \$3.64 — up 45\% from \$2.50, with almost no dilution because the shares were sold at such a rich price. If the market keeps paying 40 times earnings, the price rises to 40 × \$3.64 = \$145. The optimism that priced Ascend at 40× did not float free of reality; it lowered the cost of capital enough to fund growth that *raised the real earnings by 60\%*, which is exactly what a high multiple is supposed to anticipate.

The intuition: a high price is not just an opinion about the fundamentals — it is a lever that reaches back and changes them, which is why psychology cannot be separated out and dismissed as noise.

### Where the bridge breaks

Every reflexive advantage is also a reflexive vulnerability, and it is worth naming the cost right away. The cost-of-capital loop runs both directions. If Ascend's price *falls* to \$25, its earnings yield jumps to 10\%, its capital becomes expensive, and now raising that same \$1 billion means selling 40 million shares instead of 10 million — crushing dilution that the existing shareholders will revolt against. The projects that were cheap to fund at \$100 become impossible to fund at \$25. So a low price does not merely reflect trouble; it *causes* trouble, by starving the company of the cheap capital it had been relying on. Hold onto that symmetry. It is the engine of the death spiral we build in section 4.

## 2. The boom-bust sequence: self-reinforcing, then self-defeating

Once you accept that perception and price bend the fundamentals, the boom-bust cycle stops being a mystery of mass psychology and becomes a *mechanism*. Soros modeled it as a sequence in which a prevailing perception and a prevailing price trend feed each other, first amplifying and then destroying the very conditions that created them.

![The reflexive boom-bust arc: price detaches upward from slowly rising fundamental value in the self-reinforcing phase, peaks at the moment of truth, then reverses and overshoots below value in the self-defeating phase](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-4.webp)

The chart traces price against a slowly rising line of fundamental value. In the beginning the two move together and nobody notices anything unusual. Then a self-reinforcing phase takes hold: the price rises, the rising price improves the fundamentals (cheaper capital, more confidence, more real activity), the improved fundamentals justify a higher price, and the gap between price and value *widens* rather than closing. This is the exact opposite of what the efficient-market arbitrage story predicts — the mispricing does not get competed away, it gets amplified, because betting on the trend helps the trend come true.

Every reflexive boom passes through recognizable stages. Soros broke the process into a sequence; a simplified version is enough to work with:

1. **The trend begins, unrecognized.** Something real changes — a new technology, a policy shift, a fresh source of demand. Price starts to rise, roughly in line with the improving fundamentals. This part is not a bubble; it is a legitimate response to real news.
2. **Acceleration.** Participants notice the trend, extrapolate it, and start buying because it is rising. Now the participating function kicks in hard: the buying lifts price faster than fundamentals alone would, and the higher price starts feeding back — cheaper capital, richer acquisition currency, more real growth. The gap between price and value opens and keeps opening.
3. **The moment of truth.** The price is now stretched far above any defensible value, and reality starts to disappoint the extrapolation. Growth slows; a marginal buyer hesitates. The trend is still up, but its foundation is visibly cracking. Soros called this the moment when the divergence between belief and reality becomes unsustainable.
4. **The reversal (twilight, then crash).** The loop runs in reverse. Falling prices raise the cost of capital, damage confidence, and force leveraged holders to sell — which pushes prices down further. The same feedback that inflated the boom now deflates it, and faster, because fear moves quicker than greed.
5. **Overshoot below value.** Crucially, the loop does not stop at fair value. Just as it overshot on the way up, it overshoots on the way down: forced selling, broken confidence, and expensive capital push the price *below* the fundamental value, sometimes far below, before it eventually recovers.

The asymmetry between the up-slope and the down-slope matters. Booms build slowly because the fundamentals actually have to improve to validate the story; busts happen fast because confidence and leverage can evaporate in days. That is why the right side of the chart is a cliff and the left side is a ramp.

#### Worked example: how far price can detach from value

Suppose an asset is fundamentally worth \$40 per share and, at the start, trades there. A reflexive boom takes hold and the price runs to \$120 over two years — three times fair value. During those two years the improving story and the cheap capital genuinely lifted the fundamentals, so fair value crept up too, from \$40 to \$50. The price-to-value gap is now \$120 − \$50 = \$70 of pure bias, 140\% of the real worth of the thing.

Now the loop reverses. If price simply returned to the new fair value of \$50, that would already be a 58\% crash from \$120. But reflexive busts overshoot: forced sellers and vanished confidence drive the price to \$30, below the \$50 fair value, a 75\% collapse from the peak — and a 40\% discount to what the asset is actually worth. A patient buyer with cash and no leverage now gets a dollar of value for sixty cents, which is why the people who survive the bust intact are the ones positioned to profit from the overshoot.

The intuition: the size of the eventual crash is set by how far the self-reinforcing loop pushed price above value, plus how far the self-defeating loop then pushes it below — the bias does not unwind to zero, it swings through it.

## 3. The conglomerate trick: reflexivity you can compute

The cleanest place to *see* reflexivity turning perception into fundamentals — and the trade that first made Soros's reputation with the theory — is the American conglomerate boom of the 1960s. It is worth its own section because the mechanism is pure arithmetic; there is no hand-waving about sentiment.

A **conglomerate** is a company that owns many unrelated businesses. In the 1960s a handful of them discovered a financial magic trick. Suppose your conglomerate's stock trades at a high P/E ratio because the market believes you are a brilliant, fast-growing acquirer. You can then use your expensive shares as *currency* to buy other companies — ordinary, boring companies trading at low P/E ratios — and the moment you do, your reported earnings per share jump, with no improvement in any underlying business. Higher reported earnings per share seem to confirm that you are a brilliant, fast-growing acquirer, which supports your high P/E, which lets you buy the next company. The belief manufactures the evidence for itself.

#### Worked example: the conglomerate EPS magic trick

Let Apex Industries be a glamour conglomerate: 100 million shares, earning \$1.00 each, so \$100 million in total profit. Because the market believes in its growth, the stock trades at a P/E of 30 — a price of \$30 per share, and a total value of \$3 billion.

Apex targets Dowdy Manufacturing, a dull but solid company that earns \$30 million a year. Because Dowdy is boring, the market only pays a P/E of 10 for it, valuing the whole company at 10 × \$30 million = \$300 million. Apex offers to buy Dowdy for \$300 million, paid entirely in Apex stock: at \$30 per share, that means issuing \$300 million ÷ \$30 = 10 million new Apex shares.

Now do the combined arithmetic. Apex's total earnings become \$100 million + \$30 million = \$130 million. Its share count becomes 100 million + 10 million = 110 million. Its earnings per share are now \$130 million ÷ 110 million = \$1.18 — an 18\% jump from \$1.00, achieved without a single new product, customer, or efficiency. The "growth" came entirely from swapping high-P/E paper for low-P/E earnings. And if the market keeps rewarding Apex with a P/E of 30, the price rises to 30 × \$1.18 = \$35.45. The higher price makes Apex's acquisition currency even richer, so the next deal is even more accretive, and the loop spins faster.

The intuition: the conglomerate's dazzling earnings growth was an accounting illusion minted by the gap between its own high multiple and its targets' low ones — and the whole edifice depended on the market continuing to believe, because the instant it re-rated Apex down to a normal multiple, the reported "growth" reversed into collapse.

That reversal is exactly what happened. Once enough investors understood that the growth was manufactured rather than earned, they re-rated the conglomerates from glamour multiples to skeptical ones. The falling multiple ended the cheap acquisition currency, which ended the earnings "growth," which justified an even lower multiple. The same reflexive loop that built the boom dismantled it. Soros, having understood the mechanism on the way up, also understood it on the way down, and profited from both legs — the pattern he would repeat for the rest of his career.

## 4. The reflexive death spiral: the loop running in reverse

We have met the loop's upward form (the boom) and its arithmetic form (the conglomerate). Its most dangerous form is the downward one, the **death spiral**, because it is fast, it is involuntary, and it destroys people who did nothing wrong except stand next to it. The death spiral is what happens when the participating function is driven not by greedy buyers but by *forced sellers*.

![The reflexive death spiral: a falling price triggers margin calls and broken confidence, which force selling, which pushes the price lower, which deepens the spiral until the position or firm is wiped out](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-6.webp)

The mechanism: a price falls sharply. That fall does two things at once. It triggers **margin calls** — demands from lenders that borrowers who bought with borrowed money post more cash or collateral immediately — and it **breaks confidence**, causing funders, depositors, or counterparties to pull their money out. Both force the same action: selling assets right now, at any price, to raise cash. That forced selling pushes the price down further, which triggers the next round of margin calls and the next wave of withdrawals, which forces more selling. On the way up, higher prices attracted more buyers; on the way down, lower prices *manufacture more sellers*. That is the reflexive signature, and it is why crashes gap rather than glide.

The key thing to understand is that in a death spiral, **the selling is caused by the price, and the price is caused by the selling.** There is no external bad news required after the first shove. The falling price is both the cause and the effect of the forced selling — a loop that feeds on itself until either the forced sellers are all wiped out or an outside force (a central bank, a bailout, a circuit breaker) interrupts it.

#### Worked example: the reflexive death spiral (a margin cascade)

Imagine a stock trading at \$100, widely held by investors who bought it on **50\% margin** — meaning for every share, they put up \$50 of their own cash and borrowed \$50 from their broker. The brokers require a **maintenance margin** of 25\%: if the investor's own equity ever drops below 25\% of the current share price, the broker issues a margin call and, if unpaid, sells the shares.

Do the math on when the call hits. An investor who bought at \$100 has \$50 of equity and a \$50 loan. If the price falls to \$70, their equity is \$70 − \$50 = \$20, which is \$20 ÷ \$70 = 28.6\% — still above the 25\% line, barely. But if the price falls to \$66, equity is \$16, which is \$16 ÷ \$66 = 24.2\% — below the line. Margin call. The investor cannot post cash, so the broker sells.

Here is the reflexive part. That forced selling, multiplied across thousands of margined holders hitting their calls at roughly the same price, pushes the stock from \$66 down to \$55. But \$55 is now below the trigger point for a *different, more conservative* cohort — people who bought at \$80 with more cushion — whose equity has now also breached 25\%. They get called, they are forced to sell, and their selling pushes the price to \$45, which calls in the next cohort. Each price level detonates the next layer of leverage. Nobody chose to sell at \$45 because they thought it was worth \$45; they sold because the price was \$55 a moment ago and the machinery of leverage gave them no choice.

The intuition: on the way down, a lower price does not summon bargain hunters faster than it summons forced sellers, so the ordinary stabilizing logic of markets inverts — falling prices create the very selling that makes them fall further.

This is not a hypothetical curiosity. It is the precise mechanism behind the collapse of the hedge fund Long-Term Capital Management in 1998, the implosion of the family office Archegos in 2021, and — as we will see — the failure of an entire bank in 2023. Whenever you read that a fund "was forced to liquidate," you are reading about a reflexive death spiral.

## 5. Reflexivity in currencies: the self-fulfilling and self-defeating peg

The purest demonstration of reflexivity — and the trade that made Soros a household name — is an attack on a fixed currency peg. It deserves its own section because it shows the loop operating on a government rather than a company, and because it reveals reflexivity's most useful practical gift: an *asymmetric bet*.

A **currency peg** is a promise by a government to keep its currency within a fixed band against another currency, defended by its central bank. To hold the peg when the currency is under selling pressure, the central bank has two tools: spend its foreign-exchange reserves buying its own currency, and raise interest rates to make holding the currency more attractive. Both are costly. Reserves are finite. And high interest rates choke the domestic economy — they raise mortgage costs, squeeze businesses, and deepen any recession.

Here is the reflexive trap. A speculator who bets the peg will break is not making a neutral forecast. The act of betting — selling the currency short, in size — *increases the pressure on the peg*, forcing the central bank to burn more reserves and hike rates higher, which deepens the recession, which makes defending the peg more politically painful, which makes the break more likely. The bet helps cause the outcome the bet is on. And critically, the central bank's own defense is self-defeating: to save the currency it must inflict the exact economic damage that makes saving the currency not worth it.

![Black Wednesday, 16 September 1992: an overvalued pound in the ERM, Soros building a short position, the Bank of England hiking rates from 10 to 12 to a threatened 15 percent, and the UK abandoning the peg by evening](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-7.webp)

The timeline shows the day this played out. Through 1992 the British pound was pegged inside the European Exchange Rate Mechanism (ERM) at a rate most traders thought was too high for Britain's weak economy. Soros's Quantum Fund built an enormous short position against the pound — reportedly increasing it toward \$10 billion. On the morning of 16 September 1992 — "Black Wednesday" — the Bank of England raised its interest rate from 10\% to 12\%, then announced a further rise to 15\% for the next day, a desperate attempt to make holding the pound attractive enough to defend the peg. It did not work; the selling pressure, amplified by the very visibility of the defense, was too great. By 7:00 that evening, the Chancellor announced Britain was leaving the ERM. The pound floated and fell, the promised 15\% rate was cancelled, and Quantum reportedly made around \$1 billion. Soros became "the man who broke the Bank of England."

#### Worked example: the asymmetric reflexive bet

Look at the trade from the speculator's seat, because it explains *why* reflexive situations are so attractive to bet against. Suppose you short \$10 billion of a pegged currency that is sitting near the floor of its band.

Consider your two outcomes. If the peg *holds*, the currency can only rise a little — the band caps it, maybe 2\% of upside before the ceiling. So your maximum loss is roughly 2\% × \$10 billion = \$200 million, and in practice less, because you also earn or pay the interest-rate difference while you wait. If the peg *breaks*, an overvalued currency that has been held up artificially typically falls hard once released — say 15\%. Your gain is then 15\% × \$10 billion = \$1.5 billion.

So you are risking about \$200 million to make about \$1.5 billion: a payoff ratio better than 7 to 1. And this is before the reflexive kicker — because your own selling, and the herd that follows a famous seller, actively *increases* the odds of the outcome you are betting on. You are not passively wagering on a coin flip; you are pressing your thumb on the scale. That combination — a bounded downside, a large upside, and a bet that helps cause its own payoff — is as close to a free option as markets ever offer, which is exactly why Quantum sized it so aggressively.

The intuition: reflexivity turns a bet on a fragile, artificially-held price into a near-free option, because the act of betting is itself one of the forces that breaks the thing you are betting against.

A necessary honesty note: this is a description of a mechanism, not a recommendation to attack currency pegs. The asymmetry is real only when the peg is genuinely overvalued and fragile; bet against a sound peg and you can bleed the carry cost for years while it never breaks. Reflexivity tells you where to *look* for asymmetric setups, not that any particular peg is about to give.

## What it looks like at the screen

The theory is a skeleton. Here is the flesh — what living inside a reflexive process actually feels like, in real time, because recognizing the *feeling* is how you locate the loop before the numbers confirm it.

**In the self-reinforcing phase,** the most disorienting thing is that the optimists keep being *right*. You are skeptical of a stock at forty times earnings, so you stay out — and it doubles, because the high price funded real growth that made the high price look cheap. Every quarter the company reports better numbers, and every quarter your caution costs you. The story is not obviously crazy; it is self-confirming, and the evidence for it accumulates precisely because everyone believes it. You begin to wonder whether you are the fool. That wondering is the loop working on you.

**As it accelerates,** the reasoning quietly inverts. Early on, people bought the asset because its fundamentals were improving. Now they buy it because it is going up, and they point to the price itself as proof of the thesis. Listen for the tell: when "the fundamentals are strong" gets replaced by "look at the chart" and "it keeps going up," the cognitive function has been swamped by the participating one. The price has stopped being evidence about the company and started being evidence about itself.

**At the moment of truth,** nothing dramatic happens, which is the point. The company reports numbers that are merely good instead of spectacular. A famous believer trims a little. The price stops making new highs but does not fall — it churns, and the churn feels like healthy consolidation. It is not. It is the sound of the marginal buyer, the one the whole edifice needed, failing to arrive. Cheap capital is still available but the appetite for it is thinning. The loop has run out of new fuel, and it is running on fumes while looking, on the surface, calm.

**When it reverses,** the speed is the shock. The first sharp down day arrives and you buy it, because every previous dip was a gift — and it bounces, confirming you. Then the second down day is bigger and bounces less. If leverage is anywhere in the picture, the messages from brokers start arriving, and they are not requests. What terrifies is not the size of any single drop but the discovery that the selling is *self-generating*: there is no bad news to point to, just lower prices forcing more selling, which makes lower prices. You keep waiting for the level where value buyers step in, and it keeps not being this one, because the buyers are waiting for the forced selling to exhaust itself and the forced selling is waiting for a floor that its own existence keeps pushing lower.

If any of that reads like a memory rather than a description, that is the point. The reflexive loop is felt before it is measured. By the time a valuation model confirms that price has detached from value, the loop is already several turns along — the model is describing the boom, not warning you about it.

## Common misconceptions

**"Reflexivity just means sentiment moves prices in the short run."** This is the most common way to hear the idea and miss it. Everyone agrees sentiment adds short-term noise around a true value. Reflexivity's actual claim is stronger and stranger: sentiment, through the cost of capital and confidence, changes the true value itself, sometimes permanently. The high price does not just sit above the fundamentals waiting to be corrected; it *reaches down and lifts the fundamentals*. That is the part that makes markets genuinely indeterminate rather than merely noisy.

**"So there is no such thing as fundamental value?"** No — reflexivity needs the concept of value to work. The whole boom-bust arc is defined by price *detaching from* value and then *overshooting* it. If there were no anchor of value, "overshoot" would be meaningless. The point is not that value doesn't exist; it is that value is not fixed and independent, because price partially moves it. Value is real but soft, and the softness is what price exploits.

**"Reflexivity means markets are irrational."** The people inside a reflexive loop are usually behaving quite rationally given their situation. A fund manager who bought the rising conglomerate was right to — it kept rising and its earnings kept growing. A homeowner who bought in a housing boom was responding sensibly to real, rising prices. Reflexivity does not require anyone to be stupid. It requires only that everyone act on incomplete beliefs and that their actions feed back into the facts. Rational individual behavior produces the collectively unstable loop.

**"If Soros could see the loop, he could time the top."** He could not, and said so repeatedly. Knowing that a process is reflexive tells you it will eventually reverse; it does not tell you when. An overvalued market can double again before it breaks. Soros's edge was not calling the exact top but *understanding the structure* — knowing which situations were reflexive, betting on the direction with the asymmetry in his favor, and sizing so he survived being early. Diagnosis is not a stopwatch.

**"Reflexivity only matters for famous crises."** The loop operates at every scale, quietly, all the time. A startup whose rising valuation lets it hire better people, who build a better product, which raises the valuation, is a reflexive loop. A neighborhood whose rising home prices attract renovation and new shops, which raise home prices, is one. A country whose strengthening currency attracts foreign investment, which strengthens the currency, is one. The famous crises are just the loop turned up loud enough to hear.

**"The efficient-market hypothesis is simply wrong, then."** Also too strong. In calm, liquid, well-arbitraged markets, prices really do track value closely, and the efficient-market model is a good approximation most of the time. Reflexivity is not its replacement; it is its correction at the extremes. The honest position is that markets are *near-efficient in the middle and reflexive in the tails* — and the tails, being where money is made and lost fastest, deserve far more attention than their frequency suggests.

## How it shows up in real markets

Five episodes, one mechanism. Real instruments, real dates, real numbers where they can be cited — and the reflexive loop visible in each.

### 1. The conglomerate boom, 1960s — Soros's proving ground

The mechanism from section 3 is not a textbook toy; it is where Soros first put reflexivity to work as a professional investor. In the mid-to-late 1960s, high-multiple conglomerates — many of them defense and technology firms worried about slowing post-Vietnam sales — used their richly valued shares to buy lower-multiple companies, mechanically boosting their reported earnings per share and "validating" the high multiples that made the trick possible. Soros understood the loop on the way up and rode it. He also understood that it depended entirely on the market continuing to award glamour multiples, and when investors finally recognized the earnings "growth" as an artifact of the valuation gap, the multiples collapsed, the cheap acquisition currency vanished, the reported growth reversed, and the stocks fell hard. He profited on both legs. The lesson he drew was the thesis of his whole career: the profitable moment is not the boom or the bust but the *recognition* of where you are in a reflexive process that the crowd still thinks is a one-way trend.

### 2. Black Wednesday, 1992 — breaking the Bank of England

The pound's expulsion from the ERM on 16 September 1992, detailed in section 5, is the most cited reflexive trade in history because every element is visible. The peg was overvalued for Britain's economy, making it fragile — the necessary condition. The Bank of England's defense was self-defeating: to hold the pound it hiked rates from 10\% to 12\% and threatened 15\% (as of 16 September 1992), inflicting exactly the recessionary pain that made holding the peg politically unbearable. The speculative attack was self-fulfilling: heavy short selling, reportedly building toward \$10 billion at Quantum alone, forced the very reserve drain and rate hikes that broke the government's will. By evening Britain floated the pound, and Soros's fund reportedly netted around \$1 billion. The trade is famous for the profit, but the lesson is the structure — a defended price that requires ever-more-costly action to hold is a reflexive short with an asymmetric payoff.

### 3. The dot-com and SPAC manias — the loop as a bubble

The commercial internet of the late 1990s was a genuine displacement, which is what made the reflexive loop around it so powerful. Sky-high stock prices let internet companies raise enormous amounts of cheap capital, which they spent on growth, hiring, and advertising that produced real revenue growth — which seemed to validate the sky-high prices, which let them raise more. "Eyeballs" and "growth" replaced profits as the story, and for a while the story funded itself. Then the marginal buyer ran out, capital stopped being cheap, the companies that had never earned a profit could no longer fund their losses, and the loop reversed into mass failure.

The same script ran, in miniature and at high speed, through the SPAC boom of 2020–2021. A **SPAC** (special purpose acquisition company) is a shell company that raises money in an IPO and then merges with a private business to take it public. The reflexive loop was in the euphoria itself: rising SPAC prices and easy money attracted more sponsors and more capital, which created more deals, which sustained the enthusiasm. U.S. SPAC IPOs jumped from 248 in 2020 to 613 in 2021, and proceeds rose from roughly \$83 billion to over \$160 billion (per market tallies of the period). Then post-merger performance disappointed, the fresh money that the loop depended on dried up, and the market collapsed through 2022 — the self-defeating phase, on schedule.

### 4. GameStop, January 2021 — reflexivity weaponized

The GameStop episode is a reflexive death spiral run *in reverse* — a forced-buying loop rather than a forced-selling one — and it is the clearest modern example of a price manufacturing the trades that move it. Hedge funds had sold GameStop short so heavily that roughly 140\% of its available shares were sold short. When a crowd of retail traders on the forum r/wallstreetbets began buying, the rising price forced the short sellers to buy shares to cover their positions and cap their losses — and that forced buying pushed the price higher, which forced more covering, which pushed it higher still. This is a **short squeeze**, and it is structurally identical to the death spiral with the sign flipped: the price causes the buying and the buying causes the price. The stock ran from around \$17 to an intraday high of \$483 on 28 January 2021 (up from a low near \$2.57 nine months earlier, at the prices quoted at the time, before a later 4-for-1 split). The hedge fund Melvin Capital, caught in the loop, required a combined \$2.75 billion cash injection from Citadel and Point72. The forced actors — the short sellers who *had* to buy — were the fuel, and when they were exhausted, the loop reversed and the price collapsed just as violently.

### 5. Silicon Valley Bank, March 2023 — the death spiral in 48 hours

The failure of Silicon Valley Bank is the death spiral of section 4 running through an entire bank in two days, and it is the cleanest recent proof that a falling price and broken confidence can be self-fulfilling. SVB had invested heavily in long-dated bonds that lost market value as the Federal Reserve raised interest rates. On 8 March 2023 it announced it had sold a bond portfolio at a roughly \$1.8 billion loss and would try to raise \$2.25 billion in fresh capital to plug the hole. That announcement — meant to reassure — instead broke confidence. Depositors, many of them venture-backed startups connected on the same networks, moved to pull their money at once; on 9 March they attempted to withdraw about \$42 billion in a single day, and the bank's stock fell around 60\%. The falling stock and the deposit run fed each other: the lower the price and the louder the panic, the more depositors fled, which forced the bank to contemplate selling more assets at a loss, which confirmed the fear. By 10 March, regulators closed it — the second-largest bank failure in U.S. history at the time. No new fact was needed after the first shove. The run *was* the cause of the failure, and the failure *was* the cause of the run.

## The reflexivity-spotting drill

You cannot forecast the top of a reflexive process, but you can locate yourself inside one and lean your risk the right way. That is the practical payoff. The drill is four questions, and then a discipline for anticipating the reversal.

![The reflexivity-spotting drill: four diagnostic questions — is price changing the fundamental, is the narrative self-reinforcing, who is forced to act, and what breaks the loop — each paired with the tell that the loop is about to reverse](/imgs/blogs/reflexivity-the-feedback-loop-between-mind-and-market-8.webp)

### The four questions

**1. Is the price changing the fundamental?** This is the test for whether reflexivity is even present. Ask: if the price fell 50\% tomorrow, would the company's actual business get worse *because of the price itself*? For a mature, self-funding company with no debt, mostly no — the price is a readout. For a company that depends on issuing stock to fund itself, a bank that depends on confidence, or anyone using leverage, absolutely yes — a lower price would raise their cost of capital, trigger their margin calls, or scare their funders. Where the price feeds back into the fundamentals, you have a reflexive loop; where it does not, you have an ordinary asset the mirror model handles fine.

**2. Is the narrative self-reinforcing?** Listen to *why* people say they are buying. If the reasons are about the underlying business — earnings, cash flow, competitive position — the cognitive function is still in charge. If the reasons have become circular — "it keeps going up," "look at the chart," "everyone's making money," "this is the future" — the participating function has taken over and the price is now its own justification. The moment the evidence for the thesis is the price action itself, you are in the self-reinforcing phase.

**3. Who is forced to act at each turn?** This is the most powerful question and the one amateurs skip. In every reflexive loop, someone is *compelled* — not choosing — to buy or sell at each stage: the short seller who must cover, the margined holder who must meet a call, the depositor who must flee, the index fund that must track, the company that must roll its debt. Map the forced actors. They are the fuel of the loop, and they tell you both its direction and its limit, because a loop runs exactly as long as it has forced actors left to feed it.

**4. What breaks the loop?** Every reflexive process depends on a supply of something — new buyers, cheap capital, believing depositors, greater fools — and it reverses when that supply runs out. Name the fuel, and watch for it thinning. When the last marginal buyer has bought, when cheap capital gets expensive, when the forced buyers have all covered, when the pool of new believers is exhausted, the loop has nothing left to reinforce it and it tips into reverse.

### Anticipating the reversal

The four questions locate the loop. The reversal comes from watching the *fuel gauge* on question four, because a reflexive process does not die of old age or of an outside shock — it dies when it can no longer feed itself. The tells are on the right-hand column of the drill:

- **Cheap capital dries up.** The company that had been funding itself with rising stock or easy debt suddenly can't — an equity raise gets pulled, a bond deal struggles. The cost-of-capital bridge is failing.
- **The story stops working.** Good news stops moving the price up. When an asset reports strong numbers and *falls*, the marginal believer is already all-in; there is no one left to convince.
- **Forced actors get exhausted.** The short interest that powered a squeeze has covered; the leveraged buyers who powered a boom are fully invested. The fuel is spent.
- **The last marginal buyer is gone.** The clearest sign is the eerie calm at the top — the price stops rising not because of bad news but because the flow of new money has quietly stopped, and the churn that looks like consolidation is actually the loop idling with an empty tank.

The practical posture that falls out of this is not "call the top and short it" — that is how you get run over by a loop that has one more turn in it. It is closer to Soros's own: participate in the reinforcing phase if you must, but with the reversal pre-planned; size down as the fuel gauge drops; refuse leverage in the late innings so that you are never a forced actor yourself; and keep dry capital for the overshoot, because the loop that overshot above value will overshoot below it, and that undershoot is where the patient, unleveraged buyer is finally paid. You do not need to time the top. You need to be getting smaller as the loop runs out of fuel, and to never be the person forced to sell at the bottom.

## When this matters to you

You do not have to trade currencies or short stocks for reflexivity to touch your money. It is in the housing market you buy into, where rising prices loosen lending, which raises prices. It is in the job market of a hot industry, where soaring valuations fund hiring that justifies the valuations. It is in the retirement account you hold through a boom and a bust, where the same feedback that made the last three years feel easy is the feedback that can make the next one brutal. The loop is not exotic; it is the default shape of any market where beliefs and prices can move each other, which is to say almost all of them.

What reflexivity offers you is not a prediction but a *posture*. It tells you to stop asking only "what is this worth?" and to also ask "is the price changing what this is worth, and who is being forced to act?" It tells you that the most dangerous moment is not when things look bad but when they look effortlessly good — because that is the self-reinforcing phase, and it is precisely as convincing as it is temporary. It tells you that leverage is not just risky because it magnifies losses, but because it turns *you* into a forced actor, a piece of someone else's death spiral. And it tells you that after a crash, when the loop has overshot below value and everyone is numb, the same mechanism that punished the leveraged buyer will reward the patient one.

None of this is investment advice; it is a description of a mechanism and a way of seeing. Reflexivity will not tell you what will happen next — Soros, who understood it better than anyone, was frequently early and occasionally wrong. What it will do is change the question you ask when a price seems to be defying gravity, from "when will it correct?" to "what is feeding this loop, and what happens when the fuel runs out?" That is a better question, and asking it is most of the edge.

If you want to keep pulling on this thread, the companion pieces are the natural next steps: [the narrative fallacy](/blog/trading/trading-psychology/the-narrative-fallacy-when-a-good-story-beats-the-data) on why the self-reinforcing story is so persuasive, [herding, social proof, and FOMO](/blog/trading/trading-psychology/herding-social-proof-and-fomo) on how the crowd that powers the loop actually forms, and [the anatomy of a bubble](/blog/trading/trading-psychology/anatomy-of-a-bubble-from-tulips-to-crypto) on the five-phase arc that a reflexive boom walks from displacement to revulsion.

## Sources & further reading

The framework and the figures behind the headline numbers:

- George Soros, *The Alchemy of Finance* (1987) — the primary source for reflexivity, the cognitive and participating functions, the fallibility principle, and the boom-bust model. Soros's own later lectures and essays (including his writing at georgesoros.com) restate the theory in condensed form.
- Reflexivity applied to the 1960s conglomerate boom — Soros cites the conglomerate cycle and the early-1970s REIT boom as his two clearest worked examples of a reflexive stock-market process; see *The Alchemy of Finance* and secondary summaries such as Glen Arnold's write-up of the conglomerate case.
- Black Wednesday, 16 September 1992 — the Bank of England raised rates from 10\% to 12\% and announced a further planned rise to 15\% before the UK left the ERM that evening; George Soros's Quantum Fund reportedly built a short position toward \$10 billion and earned roughly \$1 billion. Via standard histories of the event and contemporaneous market records.
- SPAC boom figures — U.S. SPAC IPOs rose from 248 (2020) to 613 (2021), with proceeds climbing from about \$83 billion to over \$160 billion, before the 2022 collapse; via market tallies of SPAC activity for the period.
- GameStop short squeeze, January 2021 — roughly 140\% of the float sold short; the stock reached an intraday high of \$483.00 on 28 January 2021 (from a low near \$2.57 in April 2020, at the prices quoted at the time, before a later 4-for-1 split); Melvin Capital received a combined \$2.75 billion from Citadel and Point72. Via the public record of the episode, including the Wikipedia summary and market reporting.
- Silicon Valley Bank, March 2023 — an announced ~\$1.8 billion bond-sale loss and a \$2.25 billion capital raise on 8 March triggered roughly \$42 billion of attempted deposit withdrawals on 9 March and a ~60\% stock drop; regulators closed the bank on 10 March, the second-largest U.S. bank failure at the time. Via the FDIC record and standard reporting on the collapse.

Companion posts on this blog: [the narrative fallacy](/blog/trading/trading-psychology/the-narrative-fallacy-when-a-good-story-beats-the-data); [herding, social proof, and FOMO](/blog/trading/trading-psychology/herding-social-proof-and-fomo); [the anatomy of a bubble](/blog/trading/trading-psychology/anatomy-of-a-bubble-from-tulips-to-crypto).
