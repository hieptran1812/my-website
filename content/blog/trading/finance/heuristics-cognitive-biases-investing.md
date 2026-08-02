---
title: "Heuristics and cognitive biases: How the brain simplifies investment decisions"
date: "2026-08-01"
description: "A from-zero guide to representativeness, availability, anchoring, confirmation bias, recency, base rates, and limited attention in investment decisions."
tags: ["behavioral-finance", "heuristics", "cognitive-biases", "investor-psychology", "limited-attention", "anchoring", "confirmation-bias", "decision-making"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Heuristics are useful mental shortcuts, but they can make a vivid story, recent price, or familiar pattern feel more informative than the underlying evidence.
>
> - Representativeness makes a company that looks like a winner feel likely to become one, even when the base rate is poor.
> - Availability makes recent, emotional, and easy-to-recall events feel more common than they are.
> - Anchoring causes forecasts to adjust too little from an initial number, including a price target or historical high.
> - Confirmation bias turns research into belief protection when investors search mainly for evidence that supports an owned position.
> - The practical defense is base-rate-first research, explicit disconfirming evidence, ranges instead of false precision, and a pre-committed review rule.

Why can a company with a beautiful product, fast growth, and an inspiring founder feel like a good investment before we have looked at its valuation? Why can one spectacular market crash make the next crash feel imminent, while years of ordinary returns disappear from memory? Why does a price target printed at the top of a research note continue to influence us even after the business has changed?

The common thread is not lack of intelligence. It is compression. The investment world contains more information than one person can process: prices, filings, competitors, macroeconomic data, incentives, taxes, and thousands of possible future paths. The mind uses shortcuts to turn this open-ended problem into a manageable judgment.

Those shortcuts are often sensible. A familiar face may genuinely carry useful information. A recent warning may deserve attention. A prior estimate may be a reasonable starting point. The trouble begins when the shortcut answers a different question from the one we actually need to answer. Resemblance substitutes for probability. Vividness substitutes for frequency. A starting number substitutes for a valuation model. A supporting article substitutes for a test of the thesis.

![Market information passes through limited attention and heuristics before becoming an investment action](/imgs/blogs/heuristics-cognitive-biases-investing-1-pipeline.webp)

The diagram above is the mental model for the article. The market supplies a large information set. Attention selects a small subset. A heuristic compresses that subset into a pattern or story, and the story becomes a buy, sell, or wait decision. The shortcut is not automatically a mistake; it is a place where we should ask what information was excluded.

## Foundations: the information problem

### What is a heuristic?

A **heuristic** is a simple rule that helps a person make a judgment without calculating every relevant detail. “If the smoke alarm is ringing, check for fire” is a useful heuristic. “If a stock is in the news, investigate it” can also be useful. But “if a stock is in the news, buy it” confuses attention with value.

Heuristics reduce cognitive effort. The tradeoff is that they use selected cues rather than the complete probability model. When the selected cue is informative, the shortcut can work. When the environment changes or the cue is noisy, it can create a predictable error.

Tversky and Kahneman’s 1974 paper, “Judgment under Uncertainty: Heuristics and Biases,” described three classic heuristics: representativeness, availability, and anchoring and adjustment. Their central observation was that people often make judgments by similarity, ease of recall, and adjustment from an initial value rather than by applying the rules of probability and statistics. The paper is a foundation for the distinction between a useful mental shortcut and a systematic bias. [Read the paper’s abstract and citation record](https://pubmed.ncbi.nlm.nih.gov/17835457/).

### Bias is not the same as a bad outcome

A **bias** is a systematic tendency in judgment. It is not simply any decision that turns out badly. An investment can lose money even when the decision was well reasoned, because the future is uncertain. An investment can make money even when the decision was careless, because luck can temporarily reward a poor process.

To study bias, we look for a repeatable pattern. Do investors consistently ignore base rates when a story is vivid? Do they trade more after a salient price move? Do they adjust too little from an arbitrary starting number? A single anecdote can illustrate a mechanism, but it cannot establish that the mechanism is present in every person or every market.

### Probability, base rate, and evidence

A **probability** is a numerical expression of how likely an event is under a defined information set. A **base rate** is the frequency or distribution of outcomes in a relevant reference class before we look at the special features of the current case.

If 100 similar companies enter a market and only 10 become large, the base rate of becoming large in that reference class is 10%. A particular company may have better evidence than the average: stronger distribution, lower costs, or a better balance sheet. The evidence should update the base rate. It should not erase the fact that most comparable outcomes did not become large.

The base rate is not a prophecy. It is a prior distribution that helps prevent a vivid story from becoming a certainty.

### Attention is a scarce resource

**Attention** is the limited mental capacity we use to notice, process, and compare information. We can scroll past hundreds of market facts, but we can deeply process only a small subset. A stock that appears in the news, has extreme volume, or is surrounded by social discussion is more likely to enter our consideration set.

This creates an important asymmetry. When buying, an investor can choose among thousands of securities and needs a way to narrow the set. When selling, the investor usually sells from securities already owned. Attention can therefore affect what gets considered for purchase without being equally important for the sell decision.

Peng and Xiong model attention as a scarce cognitive resource. Their paper argues that limited attention can lead investors to process more market-wide and sector-wide information than firm-specific information, helping explain why stocks in the same category move together. [NBER Working Paper 11400](https://www.nber.org/papers/w11400).

#### Worked example: the same screen, different consideration sets

Imagine a hypothetical market with 1,000 listed companies. You have time to investigate 10 of them. A news feed presents 20 stocks with extreme one-day moves, while a fundamentals screen presents 20 stocks with stable cash flow and reasonable leverage.

If you choose only from the news feed, the probability that your initial consideration set contains attention-grabbing stocks is 100% by construction. That does not mean every selected stock is bad. It means your first filter has already changed the distribution of candidates before valuation begins.

If you combine the two screens, you might investigate stocks that are both visible and financially plausible. If you use only the fundamentals screen, you may miss important new information. The process question is therefore not “should I ignore the news?” It is “does the news decide what I investigate, or does it decide what I buy?”

The intuition is that a filter can shape the decision before the investor has made an explicit investment judgment.

## 1. The three classic heuristics

![Representativeness, availability, and anchoring answer different questions but can fail in predictable ways](/imgs/blogs/heuristics-cognitive-biases-investing-2-matrix.webp)

The matrix separates the three shortcuts. Representativeness asks whether the case looks like a prototype. Availability asks what examples or scenarios come to mind easily. Anchoring begins with an initial number and adjusts, often insufficiently.

### Representativeness: “It looks like a winner”

Representativeness is the tendency to judge probability by similarity. If a company resembles a successful technology compounder, an investor may infer that it is likely to become one. If a recent price pattern resembles a familiar bull market, the investor may infer that the same outcome is likely next.

The shortcut is useful when the resemblance is genuinely diagnostic. A company with a recurring revenue model may deserve different analysis from a commodity producer. The error is treating a category label as a probability. “This is a platform business” does not tell us the probability that the platform will earn attractive returns at the price we pay.

### Availability: “I can remember it, so it must be common”

Availability is the tendency to judge frequency or probability by how easily examples come to mind. A dramatic default, an overnight winner, or a crash headline can dominate memory. Quiet periods with ordinary outcomes are less memorable precisely because nothing spectacular happened.

Availability is especially powerful when information is vivid, recent, emotional, or repeated. A story that appears in multiple feeds may feel like independent confirmation even when every article traces back to the same original source.

### Anchoring and adjustment: “Start here and move a little”

Anchoring occurs when an initial value influences a later estimate, even when the initial value is arbitrary or no longer relevant. In investing, anchors include purchase price, previous high, analyst price target, round-number index level, P/E multiple, and the first forecast in a meeting.

Adjustment is not always bad. Forecasting often needs a starting point. The bias appears when the adjustment is too small, when the anchor is not updated after new evidence, or when several people in a group share the same stale starting number.

#### Worked example: a three-question diagnosis

Suppose a stock trades at $80. Ask three questions:

1. Is it attractive because it is below its previous $120 high?
2. Is it attractive because recent headlines make its turnaround easy to imagine?
3. Is it attractive because forward cash flows justify $80?

The first question tests anchoring. The second tests availability and representativeness. The third asks for the economically relevant evidence. If the answer to the third question is unclear, the first two questions should not be allowed to create certainty.

The intuition is that a low price, a vivid story, and a familiar pattern are clues—not valuation conclusions.

![Three heuristic clues about the $120 high, turnaround headlines, and a familiar pattern converge with the valuation evidence question on a single decision gate that forks into acting or holding off](/imgs/blogs/heuristics-cognitive-biases-investing-8-graph.webp)

## 2. Representativeness and the story that looks like the answer

### Prototypes are efficient but lossy

A **prototype** is a mental model of what a category typically looks like. We recognize a “growth company” through cues such as expanding revenue, a large addressable market, a charismatic founder, or a new product. Prototypes help us sort information quickly. They become dangerous when the visible cues are easier to observe than the less exciting variables that determine returns: price, competition, dilution, cash needs, and failure probability.

An investor can correctly identify that a business resembles a growth company and still make a poor investment. The category may be right while the valuation is wrong. The business may grow while returns are low because the market paid too much for that growth.

### The conjunction trap in investment stories

A **conjunction** is the joint occurrence of two events. The probability of two events happening together cannot be higher than the probability of either event alone. Yet a detailed story can feel more likely than a broad description because it is easier to imagine.

![A detailed investment story branches into four cumulative conditions with shrinking joint probability, contrasted against the single condition taken alone at the highest probability](/imgs/blogs/heuristics-cognitive-biases-investing-9-tree.webp)

Imagine two forecasts:

- The company will grow revenue quickly.
- The company will grow revenue quickly, win a major contract, expand margins, and become the market leader.

The second is more vivid, but it contains more conditions. Its probability cannot exceed the probability of the first statement. A narrative can gain psychological plausibility as it gains detail while becoming statistically less likely.

### Growth, quality, and price are separate variables

Investors often compress three questions into one adjective:

| Question | What it measures | Why it matters |
|---|---|---|
| Is the business growing? | Revenue, users, earnings, or cash flow | Operating trajectory |
| Is the business high quality? | Returns on capital, durability, balance sheet | Economic resilience |
| Is the stock attractive? | Price relative to future cash flows | Expected investment return |

A company can score well on the first two and poorly on the third. This is one reason representativeness is so costly: the investor sees a good company and unconsciously treats “good company” as “good stock.”

#### Worked example: growth does not determine return

Consider a hypothetical company with $10 of earnings per share. An investor buys at a P/E ratio of 30, so the price is

$$
30 \times \$10 = \$300.
$$

Suppose earnings grow 10% to $11, but the market now values the company at a P/E of 20. The new price is

$$
20 \times \$11 = \$220.
$$

The business earnings increased by 10%, but the share price fell from $300 to $220, a decline of $80 or approximately 26.7%. The calculation is hypothetical; the point is that operating progress and investment return are not identical.

The intuition is that a good business can be a poor investment when the starting price already assumes too much success.

### Base rates: the statistics we forget

The **base-rate neglect** error occurs when a specific description dominates a relevant prior frequency. In finance, the base rate might be the historical distribution of outcomes for comparable firms, the typical survival rate of new ventures, or the normal range of margins in an industry.

The base rate should not be used mechanically. A firm with unusually strong evidence deserves an update. The discipline is to make the update visible: start from a reference class, list the evidence that makes this case different, and state how much the evidence changes the range.

![Base-rate-first reasoning starts with a peer distribution before updating with company-specific evidence](/imgs/blogs/heuristics-cognitive-biases-investing-3-before-after.webp)

The before-and-after figure shows the difference between story-first and base-rate-first reasoning. Story-first reasoning begins with the special case and treats failure as an exception. Base-rate-first reasoning begins with the distribution, then asks whether the company has evidence that justifies moving toward the favorable end.

## 3. Availability, recency, and the market headline

### Memory is a search engine, not a database

When you ask “How likely is another crash?” your mind does not retrieve every historical market day and calculate a frequency. It searches memory for examples. If the last crash was heavily covered, the example is easy to retrieve. The ease of retrieval feels like evidence of frequency.

This is not because memory is useless. Memory contains valuable information. The problem is that retrieval is influenced by salience, repetition, emotion, and recency. A rare event can be easier to recall than a common event.

### Recency changes the perceived normal

After a long period of rising prices, recent gains can become the investor’s definition of normal. A flat month feels like weakness. After a crash, ordinary volatility can feel like the beginning of another crisis. The recent sample becomes the forecast.

Recency also affects earnings interpretation. A company that just reported a strong quarter may be described as a high-growth company even if the longer history is mixed. A weak quarter can make a durable business look permanently impaired. The correct analysis asks whether the recent result is a new regime, a seasonal observation, or noise.

### Salience and probability are different

A dramatic event deserves attention, but attention does not establish probability. A plane crash is newsworthy because it is dramatic and rare. A large number of ordinary safe flights are not newsworthy. Similarly, a spectacular stock winner can dominate investor discussion while thousands of unremarkable investments remain invisible.

The same asymmetry affects risk. A visible failure can cause investors to overestimate that exact risk, while an invisible slow deterioration can be underestimated because it lacks a single dramatic headline.

![A recent headline can become a trade before the investor checks the longer evidence history](/imgs/blogs/heuristics-cognitive-biases-investing-4-timeline.webp)

The timeline illustrates a common sequence: headline, memory retrieval, story, trade, and only later verification. A better process moves the evidence check earlier. The goal is not to trade slowly for its own sake. It is to stop the first vivid example from becoming the entire reference class.

#### Worked example: an evidence window with recency bias

Assume a hypothetical stock has five annual returns: -20%, +5%, +7%, +8%, and +10%. The simple average is

$$
\frac{-20 + 5 + 7 + 8 + 10}{5} = \frac{10}{5} = 2\%.
$$

Now suppose the investor remembers only the last three years. The average of +7%, +8%, and +10% is

$$
\frac{7 + 8 + 10}{3} = \frac{25}{3} \approx 8.3\%.
$$

Neither average is automatically the correct forecast. The five-year window may mix regimes, and the three-year window may be more relevant if the business changed. But the large difference shows why a short recent window can create a much more optimistic baseline.

The intuition is that the lookback window is itself an assumption, not a neutral fact.

### Attention and underreaction

Limited attention can also produce underreaction. If an earnings announcement is released when investors are distracted or when the information is difficult to process, the price response may be delayed. DellaVigna and Pollet studied Friday earnings announcements and reported that the delayed response represented 60% of the total response for Friday announcements versus 40% on other weekdays. They also reported abnormal trading volume around Friday announcements was 10% lower. These are results from their study design and sample, not universal constants. [NBER Working Paper 11683](https://www.nber.org/papers/w11683).

The behavioral lesson is subtle. Attention can cause overreaction to vivid information and underreaction to information that is hidden, complex, or inconveniently timed. “The market has not moved” is not always evidence that information does not matter; it may mean that processing is incomplete.

## 4. Anchoring: the first number is rarely innocent

### Why anchors work

An anchor provides a starting point. Starting points reduce effort, which is useful when the answer is uncertain. The problem is that adjustment from the anchor may be insufficient. The initial number remains embedded in the estimate even when it should have been discarded.

In investment research, anchors enter everywhere:

- a previous high becomes “fair value”;
- an analyst target becomes a default forecast;
- a round-number index level becomes a support or resistance story;
- last year’s margin becomes a long-run assumption;
- the purchase price becomes the minimum acceptable sale price.

The anchor can be informative when it is tied to cash flows or a stable economic relation. It is dangerous when it is merely the first number we saw.

### Price anchors and valuation anchors

A price anchor answers “what did this asset cost?” A valuation anchor answers “what multiple or cash flow assumption did we start with?” Both can persist after the evidence changes.

Suppose a company’s earnings decline and its competitive position weakens. A P/E of 25 from the original research note may remain the default multiple. The analyst may adjust earnings down but not the multiple, producing a forecast that still reflects the old optimism. This is anchoring in a model, not only in a chart.

### Anchoring in groups

Anchors become stronger when a group discusses the first estimate together. A meeting may start with a $500 price target. Every later estimate is then described as “above” or “below” $500 rather than being rebuilt from cash flows. The group appears to debate, but it may only be adjusting around the same starting point.

One practical defense is to collect independent estimates before sharing the initial number. Another is to ask for a forecast range and the assumptions at the endpoints, not just one target.

#### Worked example: adjusting from a stale target

Imagine an analyst’s original target is $120. New evidence reduces expected earnings by 20%. If the analyst adjusts the price target by the same 20%, the new target is

$$
\$120 \times (1 - 0.20) = \$96.
$$

That may be reasonable if the valuation multiple is unchanged. But suppose the new evidence also increases competitive risk and the appropriate multiple falls from 20 to 15. The combined valuation effect is

$$
\text{new value} = \text{old value} \times 0.80 \times \frac{15}{20} = \$120 \times 0.80 \times 0.75 = \$72.
$$

The $96 target reflects one adjustment. The $72 value reflects two. The difference is not a claim about a real company; it illustrates how a stale anchor can cause the analyst to adjust too little.

The intuition is that updating one input while leaving the old framework unchanged can preserve the original conclusion in disguise.

## 5. Confirmation bias and research as identity protection

**Confirmation bias** is the tendency to seek, interpret, and remember information in ways that support an existing belief. It becomes especially strong after we own an asset because the position can become part of our identity: “I am the kind of investor who saw this early.”

### The asymmetry of search

Suppose your thesis is “this company will expand margins.” You search for evidence and find a positive management presentation, a supportive analyst note, and a customer quote. You also find that costs are rising and a competitor is discounting. If the positive evidence feels like confirmation while the negative evidence feels like an exception, the search has become asymmetric.

Confirmation bias does not require lying to yourself. The investor may sincerely believe the positive evidence is more important. The issue is that the process gives disconfirming evidence a higher burden of proof.

![Confirmation bias filters search toward supporting evidence and reinforces the owned position](/imgs/blogs/heuristics-cognitive-biases-investing-5-graph.webp)

The graph shows a loop: an initial thesis selects sources; selected sources create confidence; confidence supports holding or adding; ownership strengthens the thesis. Contradicting evidence can be routed into a dismissal branch. A research process should deliberately add a disconfirming branch before the position becomes emotionally expensive to change.

### The ownership problem

Ownership changes the question. Before buying, you ask “why might this be attractive?” After buying, you may ask “how can I defend this purchase?” Those questions are not equivalent. The first is an open search; the second is a defense brief.

The cure is not to avoid conviction. Conviction is useful when it is tied to explicit evidence. The cure is to write the thesis, key risks, and invalidation conditions before the trade. Then review the position against those conditions rather than against your emotional attachment.

#### Worked example: a falsification budget

Assume a hypothetical thesis has three pillars:

- revenue growth of 15% or more;
- gross margin at least 40%;
- net debt below $100 million.

You assign each pillar a one-third weight. After a review, growth is 18%, margin is 35%, and net debt is $120 million. A simple score gives one pillar passing and two failing, or approximately 33% of the original checklist.

The score is not a valuation model. It is a forcing device. Without it, the investor may focus on the 18% growth rate and call the other two items temporary. With it, the investor must explain why the failed pillars should not invalidate the thesis.

The intuition is that disconfirming evidence should be recorded before it becomes convenient to reinterpret it.

## 6. Base rates, Bayesian updating, and false precision

### A plain-language Bayesian update

**Bayesian updating** is the process of starting with a prior belief and changing it when new evidence arrives. In simplified form,

$$
P(H \mid E) = \frac{P(E \mid H)P(H)}{P(E)},
$$

where $H$ is a hypothesis, $E$ is evidence, $P(H)$ is the prior probability, and $P(H \mid E)$ is the updated probability after seeing the evidence.

In investment language: start with the base rate for a class of outcomes, then ask how likely the observed evidence would be if the favorable hypothesis were true. The formula is less important than the order of operations. The prior comes before the story.

### Why strong evidence can move a weak prior

A low base rate is not a permanent “no.” Suppose only 5% of similar projects succeed, but a particular project has unusually strong evidence: committed customers, low funding needs, and a proven distribution channel. The evidence can justify moving the probability above 5%. The question is how much, not whether the story sounds exciting.

### Why precise forecasts are often misleading

A forecast of 73% can look more scientific than a range of 30%–60%, but precision may only reflect the analyst’s formatting. If the inputs are uncertain, the output should preserve that uncertainty. A range is not weakness when it is connected to scenarios and evidence.

One way to reduce false precision is to forecast states rather than one number:

| State | Operating result | Valuation implication | Decision question |
|---|---|---|---|
| Bear | Demand weakens | Multiple contracts | Can the balance sheet survive? |
| Base | Growth normalizes | Fair multiple | Is the return adequate? |
| Bull | Evidence supports acceleration | Multiple expands | What probability is justified? |

The table does not remove uncertainty. It makes the uncertainty visible.

#### Worked example: updating a base rate without erasing it

Imagine a hypothetical reference class of 100 companies. Ten eventually reach a target scale. The base rate is therefore 10%.

Now suppose a new company has a distribution partnership. Among companies with a similar partnership, 30% reach the target scale. If the partnership is genuinely comparable and the data are reliable, the evidence should move the estimate upward from 10% toward 30%.

But the partnership is not the only variable. If the company also faces a much larger competitor, its conditional probability may be lower than the reference group. If the reference sample contains 20 companies, the 30% estimate may itself be unstable. The correct result may be a range rather than a single posterior number.

The intuition is that evidence should update a base rate, while the quality and relevance of the evidence determine the size of the update.

## 7. What heuristics do well: ecological rationality and signal quality

It would be a mistake to turn behavioral finance into a catalogue of human defects. A shortcut can be rational in an environment where it uses a reliable signal and the cost of calculating everything is high. A firefighter who sees a particular smoke pattern does not need to estimate every possible source before moving. An investor who knows a company’s industry deeply may recognize a meaningful change before a generic model does.

### The environment decides whether a shortcut works

The same heuristic can be useful in one environment and misleading in another. “Use the recent price to estimate liquidity” may be reasonable for a highly traded instrument in ordinary conditions. It may fail during a market shock, when the last transaction price is not a reliable estimate of the price at which a large position can be liquidated.

“Choose the familiar brand” can work when product familiarity is evidence about repeat purchases. It can fail when the stock price already embeds an optimistic outcome. “Follow the sector” can work when a common macro factor dominates earnings. It can fail when balance-sheet differences decide which firms survive.

This is sometimes described as **ecological rationality**: the quality of a heuristic depends on the structure of the environment. We should therefore ask two questions:

1. What signal is this shortcut using?
2. How reliable is that signal in the current market regime?

### Signal, noise, and incentive

An investment signal is information that changes the distribution of possible outcomes. Noise is movement or data that does not improve the forecast. The same observation can be signal for one decision and noise for another.

A sudden volume increase may signal new information, forced trading, index rebalancing, or attention-driven speculation. A price increase may signal better future cash flows, a lower discount rate, short covering, or simply a temporary order imbalance. The heuristic “price up means business improving” is not a complete causal model.

Incentives matter too. A company presentation is designed to communicate a favorable interpretation. An analyst note may be shaped by client relationships or career incentives. A social-media post may be shaped by attention and identity. This does not make every source false. It means the source should be interpreted together with its incentive and its distance from the underlying evidence.

### A source hierarchy for behavioral research

When a story makes a claim, place the claim in a source hierarchy:

| Claim | Best first source | Common shortcut to avoid |
|---|---|---|
| Company revenue or debt | Company filing or audited report | Repeating a summary article |
| Index level or volume | Exchange or market-data source | Quoting an unsourced chart |
| Regulation | Regulator or legal text | Relying on a secondary post |
| Psychological mechanism | Original paper | Treating a popular summary as evidence |
| Market interpretation | Several independent sources | Counting repeated commentary as confirmation |

The hierarchy is itself a heuristic, but it is a deliberately designed one. It reduces the chance that a vivid summary outranks the primary evidence. It also makes a research note auditable: another reader can inspect where each important claim came from.

#### Worked example: useful shortcut or dangerous shortcut?

Consider two hypothetical rules:

- Rule A: “If a company misses earnings once, sell immediately.”
- Rule B: “If a company misses earnings, check whether the miss changes the long-run cash-flow thesis, balance-sheet safety, or valuation.”

Rule A is fast, but it treats every miss as the same event. Rule B is slower, but it preserves the question that matters. If a company earns $9 instead of an expected $10 because of a temporary timing issue, the long-run thesis may be unchanged. If it earns $9 because pricing power collapsed, the thesis may be materially worse.

Neither rule guarantees a good return. Rule B is better designed because it uses the shortcut only to trigger analysis rather than to replace analysis.

The intuition is that a high-quality heuristic points toward the evidence; a low-quality heuristic jumps directly to the action.

### When speed is part of the risk

Speed can be valuable in a market, but speed also magnifies the effect of a bad first impression. A fast decision is most dangerous when the position is large, the asset is illiquid, the probability estimate is weak, or the decision is difficult to reverse.

A useful risk rule is to match research depth to irreversibility. A small, liquid, easily reversible experiment may justify a lightweight screen. A concentrated position, a leveraged trade, or a long-lived private investment deserves a slower base-rate and disconfirmation process.

This rule also protects against a common mistake: treating the urgency of a notification as evidence that the underlying decision is urgent. A message can be immediate while the investment opportunity is not.

## 8. Attention, category learning, and why sectors move together

Investors often process information at the category level because categories reduce search costs. A headline about interest rates becomes a banking-sector signal. A commodity price becomes a mining-sector signal. A new technology becomes a technology-sector signal. This is efficient when the category truly shares an economic driver.

The risk is **category substitution**: the investor uses the sector story as a substitute for firm analysis. Two banks can have different asset quality, capital, funding, and management. Two technology companies can have different pricing power and cash needs. A sector label is a first-pass hypothesis, not a complete valuation.

Peng and Xiong’s category-learning result is useful here. Limited attention can produce stronger co-movement because investors process common category information more than firm-specific information. In plain language, the market may first move “the group” and only later separate the winners from the laggards.

![A new company can be judged by prototype resemblance or by a base-rate range](/imgs/blogs/heuristics-cognitive-biases-investing-6-tree.webp)

The tree shows the danger of stopping at resemblance. A new company can look like a known winner, but the relevant comparison is a distribution of peer outcomes. The research question is not “which famous success does this resemble?” It is “what happened to the full set of companies with comparable starting conditions?”

#### Worked example: sector exposure hiding inside a diversified list

Suppose a hypothetical portfolio holds 10 stocks, each with a 10% weight. Six are banks or property developers whose earnings depend on credit conditions. The portfolio has 10 names, but the economic exposure to one macro driver is 60%.

If the investor says “I am diversified because I own 10 stocks,” the count is true but incomplete. A sector or factor view may reveal concentration. The relevant question is the number of independent risks, not only the number of tickers.

This is where behavioral simplification can create a false sense of safety. The list looks varied; the cash flows may not be.

The intuition is that category labels can both clarify a common driver and conceal correlated exposure.

## 9. Measuring a bias without overclaiming

Behavioral explanations are attractive because they make a confusing event feel understandable. That attractiveness creates a second risk: after seeing an outcome, we may label every decision with a bias and mistake a plausible story for evidence. A careful analysis separates three layers.

### Description, mechanism, and prediction

The first layer is **description**: what happened? A stock appeared in the news, volume rose, and individual investors bought it. The second layer is **mechanism**: limited attention may have made visible stocks more likely to enter the buying set. The third layer is **prediction**: if the mechanism is active, similar attention shocks should be associated with similar buying behavior after controlling for other explanations.

One event can support a mechanism without proving the prediction. A market can rise after a popular story because the story contains genuine information, because investors imitate one another, or because both occur together. The analyst should state which conclusion the evidence can support.

### Selection effects

Suppose we study investors who bought a stock after reading about it. We cannot automatically compare them with an imaginary group that saw the same article but felt no interest. The investors selected themselves into attention and trading. Their behavior may reflect information, preferences, experience, wealth, or all of these.

This is a **selection effect**: the observed sample is not random. In an individual portfolio, selection is even more obvious. You remember the visible stock because it entered your feed; you do not observe the thousands of stocks that never entered your consideration set. A process review should record rejected candidates and the reason for rejection, not only the positions eventually bought.

### Correlation is not a causal chain

If attention and trading rise together, attention may cause trading. But trading itself may cause attention through price moves and media coverage. A third variable, such as a company announcement, may cause both. The direction matters.

A practical way to slow down causal overreach is to draw the proposed chain in words:

- announcement changes expected cash flows;
- investors notice the announcement;
- investors revise valuations;
- price and volume change.

Then list an alternative chain:

- price moves for an unrelated reason;
- media reports the move;
- attention attracts short-term traders;
- volume increases without a durable cash-flow change.

Both chains can generate the same headline. The evidence needed to distinguish them is different.

### Process metrics for an individual investor

You do not need a laboratory to inspect your own heuristics. Keep a monthly decision log with four fields:

| Field | Example | Bias it can expose |
|---|---|---|
| Discovery source | News, screen, filing, conversation | Availability and attention |
| Initial reason | Growth, valuation, turnaround | Representativeness |
| Contrary evidence | Margin pressure, debt, competition | Confirmation bias |
| Review result | Thesis intact or changed | Recency and outcome bias |

After enough decisions, look for patterns. Are purchases concentrated in assets that appeared in the news? Do you use a precise target without a range? Do you record risks only after a position loses money? The log does not prove a bias, but it gives the hypothesis something observable to attach to.

#### Worked example: the evidence-log audit

Imagine a hypothetical investor records 20 purchase decisions. Twelve began with a social-media post, five with a screen, and three with a company filing. Of the 12 social-media discoveries, eight were purchased within one day. Of the eight filing discoveries, two were purchased within one day.

The raw counts do not prove that social media caused bad returns. The groups may differ in company type and market regime. But they do reveal an attention pattern: social visibility is more likely to produce immediate action. The investor can now add a cooling-off rule or require a primary-source check for the social-media group.

The intuition is that measuring the process can reveal a bias before measuring the outcome reveals a loss.

### The danger of bias labels

Labels can help communication, but they can also stop analysis. “Anchoring” should lead to the question “what evidence would make the original number irrelevant?” “Confirmation bias” should lead to “which disconfirming fact did I actively test?” “Recency” should lead to “what happens in a longer window?”

If the label does not change the next research step, it is probably just a story about the investor rather than a useful diagnosis.

## 10. How heuristics interact rather than appear one at a time

Real decisions rarely contain one isolated bias. A company can be available because it is in the news, representative because it resembles a famous winner, anchored to a high price target, and protected by confirmation bias after purchase.

### A compound example

Imagine a stock doubles in a few months after a popular product launch. The investor sees the price move repeatedly on social media: availability. The company resembles a previous technology winner: representativeness. A $200 target appears in a headline: anchoring. After buying, the investor follows accounts that agree with the thesis: confirmation bias. The investor now feels that several independent signals point in the same direction, even though they may all originate from the same price move and story.

This is a **correlated-bias problem**. Adding more evidence does not necessarily improve the decision if the evidence is generated by the same underlying event. Ten articles repeating one announcement are not ten independent observations.

### A source-independence check

For each important claim, ask:

1. What is the original source?
2. Is this source independent of the other sources?
3. Is the evidence about the business, the price, or other people’s opinions?
4. What observation would contradict the claim?

The check is deliberately boring. Boring is useful because it interrupts the emotional momentum created by a compelling story.

## 9. How it shows up in real markets

### Case study 1: Attention-driven buying

Barber and Odean’s “All That Glitters” studies the relation between attention and the buying behavior of individual and institutional investors. The paper’s central hypothesis is structural: individuals face a search problem when choosing what to buy from thousands of possible stocks, but they usually sell stocks already in their portfolios. Stocks in the news, with high abnormal volume, or with extreme one-day returns are more likely to enter the buying choice set.

This mechanism is different from saying that investors are attracted to bad companies. An attention-grabbing stock may contain important information. The behavioral risk is that attention determines the candidate set before the investor asks whether the security’s price reflects the information. The paper therefore connects a cognitive constraint—limited search—with a market behavior—net buying of visible stocks.

The practical lesson is to separate discovery from decision. Use news to find questions. Use filings, cash-flow analysis, valuation, and risk limits to decide whether the question deserves capital.

Source: [Barber and Odean, “All That Glitters”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1151595).

### Case study 2: Friday earnings announcements and delayed attention

DellaVigna and Pollet examined earnings announcements released on Fridays. Their study reports that the delayed return response was 60% of the total response for Friday announcements, compared with 40% on other weekdays. Abnormal trading volume around Friday announcements was 10% lower, and the authors interpreted the findings as evidence consistent with limited attention.

The point is not that Friday automatically creates a trade. It is that the timing and complexity of information can affect when investors process it. A delayed response can be misread in two ways: a trader may think the initial price is fully informative, or a trader may treat the later movement as new news when it is partly delayed processing of old news.

The case also shows why event studies need careful interpretation. A statistical pattern in a sample is evidence about an average relationship, not a guarantee about the next announcement. The relevant behavioral concept is limited attention, not a calendar superstition.

Source: [DellaVigna and Pollet, “Investor Inattention, Firm Reaction, and Friday Earnings Announcements”](https://www.nber.org/papers/w11683).

### Case study 3: Vietnam’s 2020–2021 sector narratives

HOSE’s 2021 annual report records that the VN-Index ended 2021 at 1,498.28 points, up 35.7% from the end of 2020. It reports average daily trading value of nearly VND 21,997 billion and total trading value of VND 5,499,240 billion for 2021. These figures describe a period of unusually high market activity and provide a useful setting for studying attention and category learning.

In a strong market, sector labels can become powerful shortcuts. Investors may first select “banking,” “property,” or “brokerage” as a theme, then search for a stock that looks most representative of the theme. A rising sector creates more news, more social proof, and more recent examples, which can make the sector’s success feel self-explanatory.

That does not mean the sector move was only psychological. Liquidity, earnings, interest rates, credit, and pandemic recovery all matter. The behavioral question is how investors moved from a valid macro observation to a specific stock decision. A category can be economically relevant and still be too coarse for valuation.

Source: [HOSE Annual Report 2021](https://staticfile.hsx.vn/Uploads/UploadDocuments/1641899/Bao%20cao%20thuong%20nien%202021.pdf).

### Case study 4: Vietnam’s 2022 correction

The State Securities Commission reported that the VN-Index was 960.65 on November 21, 2022, down 35.9% from the end of 2021. It also reported average trading value falling from VND 26,299 billion per session in April to VND 12,124 billion in November. The update discussed declining confidence in the corporate-bond market and reported private-placement issuance of VND 329,296 billion through November 11, down 28.5% from the same period of 2021.

This episode illustrates how a category narrative can reverse. When the market was rising, a sector label could serve as a shortcut for opportunity. When liquidity and confidence weakened, the same label could become a shortcut for danger. Investors who had anchored to recent highs or recent liquidity may have interpreted the correction through the freshest headline rather than through a longer distribution of market regimes.

The lesson is not to ignore current information. It is to ask whether current information changes the business, the financing constraint, the valuation, the liquidity of the position, or only the emotional atmosphere around it. Those are different variables.

Source: [State Securities Commission market update](https://ssc.gov.vn/webcenter/portal/ssc/pages_r/l/chitit?dDocName=APPSSCGOVVN1620126529).

## 10. A better process for using heuristics without being ruled by them

The goal is not to eliminate shortcuts. A person who tried to model every possible outcome before making every decision would never finish. The goal is to know when the shortcut is being used and add a slower check when the stakes are high.

![A structured research note adds base rates, disconfirming evidence, and a review rule before capital is committed](/imgs/blogs/heuristics-cognitive-biases-investing-7-before-after.webp)

### Step 1: Define the question before collecting stories

Write the decision as a question that can be tested. “Is this exciting?” is not testable. “Can this company grow free cash flow while maintaining its balance sheet over the next five years?” is better. The question determines what evidence belongs in the research note.

### Step 2: Start with the reference class

Find comparable companies, projects, or historical episodes. Record the range of outcomes before reading the most attractive narrative about the current case. The comparison need not be perfect. It needs to be explicit enough that you can explain why the current case belongs near one end of the range.

### Step 3: Separate discovery sources from decision sources

News, social media, and screeners are discovery tools. They help you find candidates and questions. Decision sources should be closer to the underlying claim: filings for reported financials, exchange data for prices and volume, official statistics for macro data, and primary research for academic findings.

### Step 4: Write the strongest disconfirming case

Do not write a generic “risks include competition.” Write the specific observation that would make the thesis wrong. If the thesis requires margin expansion, identify the cost or pricing evidence that would falsify it. If the thesis requires a sector recovery, identify the credit or demand variable that would invalidate the recovery.

### Step 5: Use ranges and scenarios

Replace a single forecast with bear, base, and bull states. Tie each state to observable conditions. A range makes it harder for an anchor to hide inside one exact target and makes uncertainty part of the decision rather than a footnote.

### Step 6: Record the decision before the outcome

Write why you acted, what you believed, what you did not know, and what would change your mind. Later, evaluate the process separately from the return. A gain does not prove the thesis; a loss does not disprove a disciplined process.

### Step 7: Review the source concentration

Count not only the number of links but the number of independent facts. Five articles can repeat one press release. Five indicators can measure the same macro driver. If the evidence is correlated, confidence should not rise as much as the raw count suggests.

This is educational, not individualized financial advice. The appropriate process depends on objectives, liquidity, time horizon, taxes, and the ability to absorb losses.

## 11. Regime changes: when yesterday’s shortcut stops working

Markets change the reliability of signals. In a calm market, recent volume may contain information about liquidity. During a stress event, volume can be dominated by forced selling. In a low-rate environment, a familiar valuation multiple may be supported by discount rates. When rates, funding, or regulation change, the same multiple may imply a very different price.

This is why a heuristic should carry a validity condition. “High volume deserves investigation” is more robust than “high volume means buy.” “A strong brand deserves a premium” is more robust than “a strong brand deserves any premium.” The first version points to a question; the second hides a conclusion.

#### Worked example: updating a shortcut after the regime changes

Suppose a hypothetical investor has used a 20× earnings multiple as a rough reference for a profitable sector. A change in funding conditions increases uncertainty and reduces expected growth. The investor should not automatically replace 20× with 10× or 30×. The correct response is to rebuild a range, identify the changed assumptions, and ask whether the old reference class still applies.

The intuition is that a shortcut is not a permanent law; it is a rule with an environment attached.

The review should therefore include a simple question: “What changed in the environment since I adopted this rule?” Possible answers include liquidity, rates, competition, accounting quality, regulation, investor composition, or the time horizon of the decision. If nothing material changed, the shortcut may still be useful. If the environment changed, the investor should treat the old rule as a hypothesis that needs revalidation.

That final step turns a bias audit into a repeatable discipline. It does not promise perfect decisions; it makes it harder for yesterday’s convenient explanation to survive unchanged after today’s evidence has moved.

The best process is therefore modest: notice the shortcut, name its signal, test its limits, and keep the decision reversible when the evidence is weak.

## Common misconceptions

### “Heuristics are always bad.”

Heuristics are necessary because attention and time are limited. A shortcut can be accurate when the environment is stable and the cue is informative. The problem is using a shortcut outside the environment where it works or treating a cue as a complete valuation.

### “More information automatically reduces bias.”

More information can help, but it can also increase noise, confirmation, and false confidence. If every new article supports the same story, the investor may be accumulating repetition rather than independent evidence.

### “A base rate tells me what will happen.”

A base rate is a prior distribution, not a forecast of one case. Specific evidence can move the estimate. The discipline is to show the update instead of allowing the story to erase the prior.

### “A recent market move contains the most relevant information.”

Recent information can matter a great deal, especially when it signals a regime change. But recency is not a substitute for relevance. Ask whether the move changes cash flows, discount rates, liquidity, or only the attention environment.

### “Confirmation bias means I should seek the opposite view every time.”

Seeking an opposite view mechanically can create another ritual rather than better reasoning. The useful test is whether the contrary evidence is specific, independent, and capable of changing the decision.

### “A precise model defeats cognitive bias.”

Precision in a spreadsheet can hide uncertainty in the inputs. A model is a tool for exposing assumptions, not proof that the assumptions are correct. Ranges and sensitivity analysis are often more honest than a single exact output.

## When this matters to you

Heuristics appear whenever you decide what deserves attention:

- opening a brokerage app after a dramatic headline;
- treating a high-growth company as a high-return stock;
- using a previous high as fair value;
- buying a sector because its recent winners are visible;
- interpreting a research note through sources you already agree with;
- deciding that a risk is common because you have seen it repeatedly in the news;
- calling a portfolio diversified because it contains many tickers.

The practical habit is simple but demanding: let stories generate questions, let base rates define the starting distribution, let primary evidence update the range, and write down what would prove the thesis wrong.

> A heuristic is a shortcut through the information problem. It becomes a bias when we forget which road it skipped.

## Sources & further reading

- [Tversky and Kahneman, “Judgment under Uncertainty: Heuristics and Biases”](https://pubmed.ncbi.nlm.nih.gov/17835457/), *Science*, 1974.
- [Tversky and Kahneman, “Availability: A Heuristic for Judging Frequency and Probability”](https://www.sciencedirect.com/science/article/pii/0010028573900339), *Cognitive Psychology*, 1973.
- [Peng and Xiong, “Investor Attention: Overconfidence and Category Learning”](https://www.nber.org/papers/w11400), NBER Working Paper 11400, 2005.
- [Barber and Odean, “All That Glitters”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1151595), *Review of Financial Studies*, 2008.
- [DellaVigna and Pollet, “Investor Inattention, Firm Reaction, and Friday Earnings Announcements”](https://www.nber.org/papers/w11683), NBER Working Paper 11683, 2005.
- [Hong and Stein, “Disagreement and the Stock Market”](https://www.aeaweb.org/articles?id=10.1257%2Fjep.21.2.109), *Journal of Economic Perspectives*, 2007.
- [HOSE Annual Report 2021](https://staticfile.hsx.vn/Uploads/UploadDocuments/1641899/Bao%20cao%20thuong%20nien%202021.pdf).
- [State Securities Commission market update, November 2022](https://ssc.gov.vn/webcenter/portal/ssc/pages_r/l/chitit?dDocName=APPSSCGOVVN1620126529).
- Related reading: [Prospect theory](/blog/trading/finance/prospect-theory-behavioral-finance), [sector rotation](/blog/trading/vietnam-stocks/sector-rotation-explained-leaders-and-laggards), and [valuation by sector](/blog/trading/vietnam-stocks/valuation-by-sector-pe-pb-nav-ev-ebitda).
