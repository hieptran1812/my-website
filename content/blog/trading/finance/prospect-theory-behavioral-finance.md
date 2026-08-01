---
title: "Prospect theory: The investor is not an expected-value calculator"
date: "2026-08-01"
description: "A from-zero, deeply practical guide to prospect theory, reference points, loss aversion, probability weighting, framing, and the way these mechanisms shape investment decisions."
tags: ["behavioral-finance", "prospect-theory", "loss-aversion", "decision-making", "investor-psychology", "risk", "probability-weighting", "portfolio-management"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Prospect theory explains why investors do not evaluate money like a spreadsheet: we compare outcomes with a reference point, feel gains and losses asymmetrically, and mentally distort probabilities.
>
> - The same $1,000 change can feel completely different depending on whether it is framed as a gain, a loss, or a recovery.
> - A loss is not simply a negative gain: the value curve is usually steeper below the reference point.
> - People often overweight small probabilities and compress the difference between medium and high probabilities.
> - “I will hold until I get back to breakeven” is a psychological response to a reference point, not automatically an investment thesis.
> - The practical defense is to pre-commit the reference point, decision rule, time horizon, and evidence that would change your mind.

Why can a person refuse a guaranteed $500 gain but accept a gamble with a possible $1,000 gain? Why does a stock bought at $100 feel “still bad” at $90 even if the company has improved? Why does the same investor buy lottery-like shares while also paying too much to insure against a remote disaster?

These are not merely stories about weak discipline. They are recurring patterns in how people transform uncertain outcomes into choices. The transformation happens before a spreadsheet can calculate an expected return. We first decide what counts as a gain or a loss, how strongly each outcome matters, and how likely it feels. Only then do we act.

![Pipeline from cash outcome through reference point and subjective value to an investment decision](/imgs/blogs/prospect-theory-behavioral-finance-1-pipeline.webp)

The diagram above is the mental model for this article. A market price is an outcome in dollars, but the decision-maker does not experience the dollar amount in isolation. The outcome is compared with a reference point; the difference is transformed by a curved value function and a probability-weighting function; framing then helps determine the action. The rest of the article tours those transformations from the clean benchmark of expected utility to practical portfolio decisions.

## Foundations: the building blocks

Before we can talk about bias, we need a simple vocabulary. Behavioral finance is not the claim that every investor is irrational all the time. It is the study of how actual preferences and beliefs differ from the simplified decision-maker used in many economic models, and how those differences can affect prices, trading, and household outcomes.

### Outcomes, wealth, and returns

An **outcome** is what happens to your money: you receive $100, lose $50, or finish the year with $11,000. **Wealth** is the total value of what you own at a point in time. A **return** is the percentage change in an investment over a period. If one share moves from $100 to $110, the dollar gain is $10 and the simple return is

$$
r = \frac{110 - 100}{100} = 0.10 = 10\%.
$$

Here, the numerator is the change in price and the denominator is the starting price. The arithmetic is objective. If the share is $110, the market does not need to know whether you feel proud, relieved, or disappointed.

But the decision you make next is not determined by the arithmetic alone. You might compare $110 with the $100 you paid, with the $140 high you remember, with the $120 target in your plan, or with the $105 you could have earned in a savings account. Each comparison creates a different psychological outcome from the same market price.

### Risk, uncertainty, and probability

An outcome is **certain** when you know what will happen. It is **risky** when there are several possible outcomes and you can attach probabilities to them. It is **ambiguous** when you do not know the probabilities themselves.

Suppose an investment has a 60% chance of gaining $100 and a 40% chance of losing $50. Its **expected monetary value** is

$$
E[X] = 0.60 \times 100 + 0.40 \times (-50) = 60 - 20 = \$40.
$$

The expected value is a probability-weighted average. It is not a promise that you will receive $40. You receive either $100 or lose $50; $40 is the long-run average across many repetitions of the same gamble.

That distinction matters. If this gamble is offered once, you cannot eat the average. You must choose between a certain outcome and uncertain outcomes, and the emotional meaning of each outcome may matter more than its average dollar value.

### The clean benchmark: expected utility

Classical decision theory makes one important improvement over raw expected money. It assumes that people maximize **utility**, a function that represents how valuable an outcome is to them, rather than maximizing dollars directly. A risk-averse person may prefer a certain $40 to a gamble whose expected monetary value is $40 because the downside hurts more than the upside pleases.

If a gamble produces outcomes $x_1, x_2, \ldots, x_n$ with objective probabilities $p_1, p_2, \ldots, p_n$, expected utility is

$$
EU = \sum_{i=1}^{n} p_i u(x_i),
$$

where $u(x_i)$ is the utility assigned to outcome $x_i$. The probabilities add to one, so $\sum_{i=1}^{n} p_i = 1$.

The benchmark is powerful because it separates two ideas: probability tells us how often an outcome occurs, and utility tells us how much the outcome matters. It can represent diminishing marginal utility of wealth: the first $1,000 may change your life more than the tenth $1,000.

The limitation is descriptive. People often do not evaluate final wealth in a stable way. They evaluate changes relative to a reference point, treat probabilities as psychologically transformed, and respond differently to an equivalent gain and loss. Prospect theory was developed to describe those patterns.

![Classical absolute-wealth evaluation compared with reference-dependent evaluation](/imgs/blogs/prospect-theory-behavioral-finance-2-before-after.webp)

The figure contrasts the clean benchmark with the behavioral lens. The classical view ranks final wealth. Prospect theory asks what the change means relative to a reference point, and then allows gains and losses of the same size to receive different subjective weight.

#### Worked example: expected money is not the same as a comfortable choice

Imagine two choices:

1. Receive a guaranteed $40.
2. Receive $100 with probability 60%, and lose $50 with probability 40%.

The gamble has expected value

$$
0.60 \times \$100 + 0.40 \times (-\$50) = \$40.
$$

If you choose the certain $40, you are not making a mathematical mistake. You are revealing that the security of certainty has value to you. If the gamble is repeated many times and the probabilities are reliable, its average dollar outcome is $40. But a single household decision has a time horizon, a budget, and consequences that are not captured by the average.

The intuition is simple: expected money describes the average outcome; utility describes how the decision-maker experiences the possible outcomes.

## 1. Reference points: every gain and loss is relative

A **reference point** is the benchmark against which an outcome is judged. It can be the price you paid, the wealth you had yesterday, a target return, a previous high, a promised salary, or the status quo. Prospect theory does not insist that one reference point is always correct. It says the reference point is behaviorally important.

### The cost basis is not the company

Suppose you buy one share at $100. The company does not know your purchase price, and its future cash flows do not become better or worse because you entered at $100. Yet your experience at different prices changes:

- At $80, you see an unrealized loss of $20.
- At $90, you still see an unrealized loss of $10.
- At $100, you feel relief because the account is back to breakeven.
- At $120, you see a $20 gain.

The **cost basis** is the accounting reference used to calculate your gain or loss. It is also often a psychological anchor. The cost basis is useful for taxes and record-keeping, but it is not automatically a useful forecast of future value. A stock does not become more attractive merely because it is below what you paid.

![A purchase price becomes a reference point across later market prices](/imgs/blogs/prospect-theory-behavioral-finance-5-timeline.webp)

The timeline shows how one purchase can create a sequence of emotional labels. The price at $90 is a loss relative to the cost basis, even though it might be a good investment at $90 if the expected future cash flows justify it. Conversely, $120 can be a gain and still be a bad investment if the market value is far above reasonable cash-flow estimates.

### Reference-point dependence creates path dependence

Two investors can own the same share at the same market price but make different choices because their histories differ. Investor A bought at $100 and sees a loss at $90. Investor B bought at $60 and sees a gain. The market price is identical; the reference points are not.

This is **path dependence** in behavior: the current decision depends partly on the route taken to reach the current state. If the share first rose from $60 to $120 and then fell to $90, an investor who bought at $60 may still feel successful. If it fell from $120 to $90, an investor who bought at $120 may feel trapped. The next dollar of expected return is the same in both cases, but the mental account is different.

Reference points can also move. After a long bull market, the previous high may become the implicit benchmark. A 10% return can feel disappointing if the market rose 25%. After a crash, a 5% return can feel wonderful if the alternative was another year of losses. This moving benchmark helps explain why satisfaction and risk-taking change across market regimes.

#### Worked example: the same current price, three different stories

Assume a share is now $90. Consider three investors:

- Investor A bought at $100: current result is $90 - $100 = -\$10$, or -10%.
- Investor B bought at $60: current result is $90 - $60 = +\$30$, or +50%.
- Investor C has no position and is deciding whether to buy: current result is neither a gain nor a loss.

The company, market price, and future distribution are the same. But A may hold to avoid realizing a loss, B may sell to lock in a gain, and C may ask whether $90 is attractive from today forward. Investor C is closest to the economically relevant question: if you had $90 in cash today, would you buy this share? That is the **fresh-capital test**.

The intuition is that the cost basis explains the past; it does not, by itself, determine the best use of capital today.

### Reference points beyond the brokerage account

Investors carry several reference points at once:

| Reference point | Question it creates | Common behavioral consequence |
|---|---|---|
| Purchase price | Am I up or down? | Disposition effect and breakeven fixation |
| Previous high | How far have I fallen? | Anchoring and regret |
| Portfolio target | Am I on plan? | Rebalancing or frustration |
| Risk-free alternative | What did I give up? | Opportunity-cost awareness |
| Personal need | Can I pay the bill? | Forced selling at a bad time |

There is no universal rule saying one reference point must dominate. A retirement portfolio should be judged against the retirement goal and time horizon, not only against the last quoted price. A short-term trading position may have a clearly defined invalidation level. A household emergency fund should be judged by liquidity and safety, not by whether it outperformed a stock index.

The mistake is allowing an accidental reference point to control a decision that belongs to a different time horizon.

## 2. The value function: why losses hurt more than equal gains please

Prospect theory’s value function maps a change relative to a reference point into subjective value. It has three features that matter in finance:

1. It is defined around a reference point.
2. It is usually concave for gains and convex for losses.
3. It is steeper on the loss side.

### Concave gains, convex losses

For gains, the curve is **concave**: the difference between $0 and $100 feels larger than the difference between $1,000 and $1,100. For losses, the curve is often **convex**: the subjective difference between a $100 loss and a $200 loss may feel smaller than the difference between no loss and a $100 loss.

This does not mean a $200 loss is harmless. It means the marginal psychological impact can diminish as losses accumulate. That shape can create risk aversion over gains and risk seeking over losses.

The intuition is familiar. If someone offers you a certain $100 gain or a 50–50 gamble for $0 or $200, the guaranteed gain may feel attractive. But if you are already down $1,000, you may prefer a gamble that can erase the loss or produce a larger loss over accepting a smaller certain loss.

### Loss aversion is a relative slope, not a magic constant

**Loss aversion** means that a loss of a given size often carries more subjective pain than an equal gain carries pleasure. A popular shorthand is that losses “feel about twice as large” as gains, but the exact ratio depends on the experiment, the reference point, the stakes, the framing, and the domain. We should not treat a single coefficient as a law of human behavior.

The important claim is qualitative: the slope below the reference point is steeper. That asymmetry is enough to generate many observed decisions.

![Expected utility and prospect theory compared across reference, probability, value, and predicted choice](/imgs/blogs/prospect-theory-behavioral-finance-3-matrix.webp)

The matrix is a map, not a claim that every person uses the same function. Expected utility usually keeps objective probabilities and evaluates final wealth through a utility function. Prospect theory adds a reference point, a separate value function for changes, and a probability transformation.

#### Worked example: a symmetric gamble with asymmetric feelings

You can choose between:

- A guaranteed $100 gain.
- A 50% chance of gaining $200 and a 50% chance of gaining $0.

Both have expected monetary value $100. A risk-averse decision-maker may prefer the guaranteed gain. Now move the reference point so that the choice is framed as avoiding a loss:

- Accept a guaranteed $100 loss.
- Take a 50% chance of losing $200 and a 50% chance of losing $0.

The expected monetary value is -$100 in both cases. Yet a person who is risk seeking in the loss domain may choose the gamble, hoping for the zero-loss outcome.

The probabilities and dollar averages have not changed. The location relative to the reference point has changed the shape of the decision.

The intuition is that a risky choice can be attractive when it offers a chance to escape a salient loss, even if the expected money is no better.

### The disposition effect as a portfolio consequence

The **disposition effect** is the tendency to realize gains more readily than losses. It is one behavioral route from the value function to actual trading. Selling a winner converts a paper gain into a realized gain and closes a pleasant mental account. Selling a loser makes the loss official.

That emotional asymmetry can produce a portfolio with two undesirable properties:

- winners are removed before their thesis has matured;
- losers remain because breakeven has become the implicit objective.

The cure is not “always sell losers” or “always hold winners.” The correct question is forward-looking: given today’s price, information, risk, and opportunity set, would you initiate the position now? If not, the fact that you have not yet recovered the cost basis is not a reason by itself to keep it.

## 3. Probability weighting: the mind does not use probabilities like a calculator

Prospect theory also allows people to transform objective probabilities into **decision weights**. A decision weight is not necessarily a person’s stated belief about frequency. It is the psychological influence an outcome has on the choice.

### Small probabilities can be overweighted

A 1% chance is not experienced as exactly one hundredth of a certain event. Rare, vivid, emotionally charged outcomes may receive disproportionate attention. This can help explain why people buy lottery-like payoffs and why they may overpay for protection against dramatic but remote events.

Overweighting a small probability does not mean that everyone always overweights every rare event. Context matters. A rare event can be ignored when it is abstract, delayed, or hard to imagine; it can dominate attention when it is vivid and available in memory.

### Medium and high probabilities can be compressed

People may also fail to distinguish sharply between probabilities such as 50%, 60%, and 70% when the information is uncertain. If the probability estimate itself is noisy, a precise numerical difference can create false confidence. This is one reason to separate:

- the probability in the model;
- the quality of the evidence behind the probability;
- the size of the payoff if the event occurs.

![Objective probabilities flow through rare-event salience and probability compression before a choice](/imgs/blogs/prospect-theory-behavioral-finance-6-graph.webp)

The figure deliberately uses qualitative labels rather than a universal weighting curve. The empirical shape varies by model and context. The robust lesson is that subjective decision weight can differ from objective probability, especially at the extremes.

#### Worked example: a lottery-like investment

Imagine a speculative asset with a 1% chance of becoming worth $10,000 and a 99% chance of becoming worth $0. Ignore fees and assume the purchase price is $50.

The expected value of the payoff is

$$
0.01 \times \$10{,}000 + 0.99 \times \$0 = \$100.
$$

The expected profit before the $50 purchase price is $50. But the expected value does not describe the distribution: 99 out of 100 comparable trials end at zero. If the probability is reliable and the gamble can be repeated independently, an expected-value argument may be relevant. For a household making one irreversible bet, the concentration risk and loss of $50 may matter much more.

Now imagine the probability is not actually known. The 1% is a promotional estimate, based on a small sample and a compelling narrative. The most dangerous error is to treat the number as precise merely because it is written with two decimal places.

The intuition is that a large payoff can attract attention while hiding the fact that the probability estimate may be weak and the downside concentrated.

### Insurance and lottery behavior can coexist

It is tempting to label people inconsistent when they buy both lottery tickets and insurance. Prospect theory offers a more nuanced explanation. The lottery creates a small chance of a life-changing gain; insurance creates a small chance of avoiding a catastrophic loss. Both small probabilities can receive high decision weight because the outcomes are vivid and the reference point is emotionally important.

The financial cost can still be poor. A decision weight is not an expected return. When a person overweights a remote upside, they may pay too much for a lottery-like payoff. When they overweigh a remote downside, they may buy more protection than their budget and actual exposure justify.

## 4. Framing: the same outcome can produce a different choice

**Framing** is the way a choice is presented. A frame can emphasize gains, losses, avoided losses, probabilities, certainty, or the status quo. Prospect theory predicts that framing can change preferences even when the underlying outcomes are mathematically equivalent.

### Gain frame versus loss frame

Consider an investment decision with two ways to describe the same account:

- “Your portfolio is up $2,000 from the beginning of the year.”
- “Your portfolio is $3,000 below the high-water mark.”

Both statements can be true at the same time. The first establishes the start-of-year balance as the reference point. The second establishes the high-water mark. The investor may become risk seeking under the second frame, trying to recover the missing $3,000, even though the high-water mark is not a contractual claim.

This is why a high-water mark can be a dangerous benchmark for personal decisions. It is useful for describing drawdown, but it should not silently become a promise that the portfolio must return to a particular number.

![Absolute wealth and reference-dependent value produce different interpretations of the same $1,000 change](/imgs/blogs/prospect-theory-behavioral-finance-2-before-after.webp)

### Narrow framing versus broad framing

**Narrow framing** evaluates a decision in isolation. **Broad framing** evaluates it as part of a portfolio or lifetime plan. Buying one volatile stock can look attractive when framed as a small chance of a large gain. The same purchase may look less attractive when framed as an addition to an already concentrated portfolio.

Narrow framing also makes daily price changes emotionally salient. If you check an account every hour, each fluctuation becomes a separate gain or loss. If you evaluate a long-term plan at a pre-committed interval, the same market noise is aggregated into a longer horizon.

Broad framing does not mean ignoring risk. It means putting the risk in the unit that matters: total portfolio, annual spending need, debt service, or probability of meeting a goal.

#### Worked example: one portfolio, two reference points

Suppose you invest $10,000. After a period of volatility, the account is worth $10,500.

- Relative to the initial deposit, the account is up $500, or 5%.
- Relative to a temporary peak of $12,000, the account is down $1,500, or 12.5%.

The arithmetic is not contradictory. The two percentages answer different questions. The first asks whether the account grew from the starting point. The second asks how far it is from its peak.

If you frame the portfolio as “up 5%,” you may be willing to follow the plan. If you frame it as “down $1,500,” you may take a risky bet to recover the peak. A disciplined process records both numbers but assigns each a clear purpose: total return for performance, drawdown for risk, and goal progress for suitability.

The intuition is that a benchmark can be informative without being a valid target for risk-taking.

## 5. Four building blocks of prospect theory

The theory is easier to remember as four questions rather than as a single formula.

![Four questions behind a risky choice: reference, value, probability weight, and frame](/imgs/blogs/prospect-theory-behavioral-finance-4-tree.webp)

### Question one: Compared with what?

Identify the reference point. Is the decision about wealth relative to today, the cost basis, a target, an obligation, or a previous high? If the reference point is hidden, the emotional reaction can look irrational because the observer is using a different benchmark.

### Question two: How much does the change matter?

Map the change through the value function. The first dollars of gain or loss may have greater marginal impact than later dollars. Gains and losses may have different slopes. The same numerical change may therefore produce different subjective value.

### Question three: How likely does the event feel?

Separate objective probability from decision weight. Ask what evidence supports the probability and whether vividness, recent news, or social proof is doing more work than base rates.

### Question four: What frame is controlling the choice?

Rewrite the decision in at least two ways: gain and loss; narrow and broad; current capital and historical cost basis. If the decision changes when only the words change, that is a signal to slow down.

This four-question process is not a mechanical cure. It is a way to expose the hidden inputs that a normal expected-return calculation may omit.

## 6. From individual psychology to market behavior

Prospect theory is a theory of individual choice, but markets are made from individual choices. The bridge from a person to a market pattern is not automatic. A bias can cancel out across investors, be arbitraged away, or be amplified by leverage, attention, institutional constraints, and social interaction.

### Trading volume and disagreement

If investors have different reference points and different probability weights, they can disagree about the same information. One investor sees a stock at $90 as a bargain relative to a $120 high; another sees it as expensive relative to a $60 cost basis; a third sees neither reference point as relevant. Disagreement creates trade.

A market can therefore have high volume even when the underlying information is not changing rapidly. The buyers and sellers need different beliefs, objectives, liquidity needs, or reference points. That does not prove that every trade is irrational. It shows why trading volume should not be read as a direct measure of informed conviction.

### Underreaction and overreaction

Reference dependence and probability weighting can push prices in opposite directions. Investors may underreact to abstract, difficult information because it is hard to process. They may overreact to vivid, salient news because it receives too much attention. In the medium term, gradual information flow and attention can support momentum; over a longer horizon, extreme narratives may reverse when fundamentals reassert themselves.

Hong and Stein describe a related framework in which disagreement, gradual information diffusion, and limited attention help connect momentum, post-earnings drift, and long-run reversal. Their point is not that every reversal is psychological, but that a market model must explain both price movement and trading volume, not just one or the other.

### Limits to arbitrage

If a price is wrong, why does a rational trader not immediately correct it? **Limits to arbitrage** are the risks and costs that make correcting a mispricing difficult. A trader can be early, face funding pressure, endure further price movement, or be unable to short the asset. A behavioral mispricing can survive because the arbitrageur’s horizon is shorter than the market’s correction time.

This matters for individual investors because “the price is irrational” is not a timing signal. A price can remain away from your estimate longer than your liquidity, patience, or risk budget can survive.

## 7. Real-market case studies

The mechanisms become clearer when we keep the facts and the interpretation separate.

### Case study 1: GameStop and attention in January 2021

GameStop is often summarized with one phrase, “short squeeze,” but the SEC staff report describes a more complicated episode. The report says the stock experienced a confluence of large price moves, large volume changes, large short interest, frequent Reddit mentions, and significant mainstream-media coverage. On January 13, 2021, the closing price rose from $19.95 to $31.40 and volume reached approximately 144 million shares, compared with approximately 7 million shares the previous day. On January 27, the stock closed at $347.51; on January 28, it reached an intraday high of $483.00. These are dated historical observations from the SEC report, not a forecast or a repeatable expected return.

Behaviorally, the episode illustrates attention-driven buying and social proof. Investors who would not search through thousands of securities encountered GameStop because it was salient. A visible price increase became evidence that other people were buying. The price increase generated more attention, and the attention generated more trading. That is a feedback loop, not proof that the underlying business value changed at the same speed.

The SEC report also cautions against a single-cause narrative. Its staff concluded that positive sentiment, rather than short sellers buying to close positions, sustained the weeks-long price appreciation. The lesson for prospect theory is not “never buy a volatile stock.” It is to identify which reference point you are using, what probability you assign to the payoff, and whether the probability is based on evidence or on the crowd’s excitement.

Source: [SEC Staff Report on Equity and Options Market Structure Conditions in Early 2021](https://www.sec.gov/files/staff-report-equity-options-market-struction-conditions-early-2021.pdf).

### Case study 2: Vietnam’s 2020–2021 liquidity bull market

The HOSE Annual Report 2021 records a sharp expansion in activity. The VN-Index ended 2021 at 1,498.28 points, up 35.7% from the end of 2020. Average daily trading value on HOSE was nearly VND 21,997 billion, while total 2021 trading value was VND 5,499,240 billion. The report also records market capitalization above VND 5,800,000 billion and a record session on December 23, 2021, with more than VND 45,371 billion in value.

These facts do not prove that the market was a bubble or that every participant was driven by one bias. They do provide a setting in which attention, social proof, and reference points could reinforce one another. New investors saw rising prices and increasing liquidity. A previous day’s gain became a reference point for the next day’s expected gain. A sector that had already risen became evidence of where the crowd was going next.

Prospect theory helps explain why investors may accept more risk after gains: a paper profit can feel like “house money,” making a new gamble feel less painful. It also explains why a later correction can cause an abrupt change in behavior. The same investor who treated a position as risk capital at the peak may treat the remaining balance as money that must be recovered after a fall.

The correct interpretation is mechanism-level: a rising market can make risk-taking easier to justify, while liquidity and attention can turn individual decisions into a collective trend. It does not follow that every price change is irrational or that the same numerical return can be expected again.

Source: [HOSE Annual Report 2021](https://staticfile.hsx.vn/Uploads/UploadDocuments/1641899/Bao%20cao%20thuong%20nien%202021.pdf).

### Case study 3: Vietnam’s 2022 correction and changing reference points

In a November 2022 market update, Vietnam’s State Securities Commission reported that the VN-Index stood at 960.65 points on November 21, 2022, down 35.9% from the end of 2021. The same update reported that average trading value declined from VND 26,299 billion per session in April to VND 12,124 billion in November. It also described declining investor confidence in the corporate bond market and reported that privately placed corporate-bond issuance through November 11 was VND 329,296 billion, down 28.5% from the same period of 2021.

The behavioral point is not that the correction had no fundamental or regulatory causes. Market structure, credit conditions, disclosure, liquidity, and regulation matter. The point is that falling prices change the reference point of the participant. A person who bought near the end of 2021 may see a loss and hold for breakeven. A person who held cash may see a different opportunity. A person using leverage may face a cash constraint that turns a preference into a forced sale.

This is where individual prospect theory meets market mechanics. Loss aversion can delay selling in an unlevered account; leverage and margin calls can force selling regardless of preference. A theory of behavior must therefore be paired with the balance sheet and liquidity constraints around the investor.

Sources: [State Securities Commission market update](https://ssc.gov.vn/webcenter/portal/ssc/pages_r/l/chitit?dDocName=APPSSCGOVVN1620126529), [SSC corporate-bond market conference](https://ssc.gov.vn/webcenter/portal/ssc/pages_r/l/chitit?dDocName=APPSSCGOVVN1620133618).

### Case study 4: Trading results and the cost of an active reference point

Barber and Odean studied account data from 66,465 households at a large discount broker during 1991–1996. The households that traded most earned an annual return of 11.4%, while the market return was 17.9% in their sample. The interpretation is not that every trade loses money, nor that market conditions are unchanged since the study. It is evidence that high activity can be associated with lower net performance after the costs and mistakes that accompany trading.

Prospect theory helps us ask what happens before a trade. A recent winner can make an investor feel unusually skilled and increase risk-taking. A recent loser can create a desire to “make it back” through a larger bet. Both choices use the previous outcome as a reference point. The investor is no longer asking only whether the next position has a good forward distribution; they are also trying to repair or confirm a self-image.

The practical lesson is to record the decision before the outcome is known. A decision journal separates process quality from result quality. A good process can produce a loss; a bad process can produce a gain. Without that separation, one lucky result becomes evidence for overconfidence.

Source: [Barber and Odean, “Trading Is Hazardous to Your Wealth”](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00226).

## 8. A closer look at the formal model

The prose so far has been deliberately intuitive. Now we can write down a compact version of the model, with one warning: the following equation is an explanatory abstraction of the canonical prospect-theory structure. It is not a claim that every investor uses the same estimated parameters.

![The prospect-theory model adds a reference point, a curved value function, and decision weights to the expected-utility benchmark](/imgs/blogs/prospect-theory-behavioral-finance-3-matrix.webp)

The matrix above is the navigation map for the formal model: reference point, value transformation, probability transformation, and the resulting choice.

![Reference points and value curvature are the two distinct transformations before a risky choice](/imgs/blogs/prospect-theory-behavioral-finance-4-tree.webp)

For an outcome $x$ relative to a reference point, a common value-function abstraction is

$$
v(x) =
\begin{cases}
x^\alpha, & x \ge 0, \\
-\lambda(-x)^\beta, & x < 0,
\end{cases}
$$

where $\alpha$ controls curvature for gains, $\beta$ controls curvature for losses, and $\lambda$ captures the relative weight placed on losses. The key restrictions are usually $0 < \alpha \le 1$, $0 < \beta \le 1$, and $\lambda > 1$. The restrictions describe diminishing sensitivity and a steeper loss side; they do not tell us that one exact value of $\lambda$ applies to all decisions.

![The value function changes slope around the reference point and treats gains and losses differently](/imgs/blogs/prospect-theory-behavioral-finance-2-before-after.webp)

For a small set of outcomes, a prospect-value abstraction is

$$
PV = \sum_{i=1}^{n} \pi(p_i) v(x_i),
$$

where $p_i$ is the objective probability of outcome $x_i$ and $\pi(p_i)$ is its decision weight. The function $\pi$ represents the psychological transformation of probability. It is not the same as saying that the person believes the event occurs with probability $\pi(p_i)$ in a statistical sense.

![Probability weights and value transformations are separate stages in a prospect evaluation](/imgs/blogs/prospect-theory-behavioral-finance-6-graph.webp)

### Why the reference point belongs inside the equation

If final wealth is $W$ and the reference point is $W_0$, the relevant change is $x = W - W_0$. Changing $W_0$ changes the sign and magnitude of $x$ without changing $W$.

Suppose final wealth is $10,500:

- with $W_0 = \$10,000$, the change is $x = +\$500$;
- with $W_0 = \$12,000$, the change is $x = -\$1,500$.

The account statement is the same. The argument entering the value function is not. This is why a behaviorally realistic model cannot simply take the final wealth number and apply one permanent utility curve without specifying the benchmark.

### Why probability weighting is not a forecasting license

Probability weighting can explain why a rare event has more influence than its objective probability would suggest. It does not imply that an investor should deliberately inflate every small probability. The model describes a possible source of error and a possible source of preference; it does not create information about the actual odds.

This distinction is especially important in markets. If you do not know whether an event has a 1% or a 10% probability, the problem is not merely that your decision weight is distorted. The probability estimate itself is uncertain. Treating an uncertain estimate as a known probability can make a model look more rigorous while making the decision less honest.

### A sensitivity table beats a single behavioral coefficient

Rather than asking whether your loss-aversion coefficient is 2 or 2.25, test how the decision changes under several plausible assumptions:

| Assumption | Question | What to inspect |
|---|---|---|
| Reference point | Cost basis, goal, or current wealth? | Does the action change? |
| Loss sensitivity | Small, medium, or high? | How much downside is tolerable? |
| Probability | Base case or a range? | Is the thesis robust to worse odds? |
| Framing | Gain, loss, or portfolio? | Does the narrative drive the action? |

This table is more useful than pretending to know a psychological parameter to two decimal places. The purpose of behavioral analysis is often to locate fragile decisions, not to produce a false point estimate of emotion.

![A risky choice branches through probability weights before subjective value is compared](/imgs/blogs/prospect-theory-behavioral-finance-6-graph.webp)

#### Worked example: a two-outcome prospect value, clearly labeled as an abstraction

![A two-outcome prospect is evaluated by combining weighted gains and weighted losses](/imgs/blogs/prospect-theory-behavioral-finance-1-pipeline.webp)

Consider a hypothetical gamble with a $100 gain and a $50 loss, each with probability 50%. To illustrate the structure, assume $\alpha = \beta = 0.88$, $\lambda = 2.25$, and no probability weighting, so $\pi(0.5) = 0.5$. These numbers are illustrative parameters for arithmetic, not an empirical estimate for you.

The gain contribution is approximately

$$
0.5 \times 100^{0.88} \approx 0.5 \times 57.5 = 28.75.
$$

The loss contribution is approximately

$$
-0.5 \times 2.25 \times 50^{0.88} \approx -0.5 \times 2.25 \times 30.0 = -33.75.
$$

The illustrative prospect value is therefore approximately

$$
PV \approx 28.75 - 33.75 = -5.00.
$$

The expected monetary value of the same gamble is

$$
0.5 \times \$100 + 0.5 \times (-\$50) = \$25.
$$

The two calculations answer different questions. Expected value says the average dollar outcome is positive. The illustrative prospect value says the loss can receive enough subjective weight that the gamble is unattractive. Change the reference point or parameters and the conclusion can change.

The intuition is that a positive expected dollar outcome can coexist with a negative subjective evaluation when the downside is psychologically and financially important.

This calculation is not meant to encourage fitting a personal utility function from one trade. Its purpose is diagnostic: it shows exactly where an apparently positive expected dollar outcome can become unattractive after reference dependence and loss sensitivity are introduced. In practice, the uncertainty around the inputs is often more important than the fourth decimal place of the output.

### What the parameters can and cannot tell us

The parameterized form is useful because it exposes assumptions. If $\lambda$ is larger, the same loss receives more weight. If $\alpha$ is smaller, sensitivity to additional gains falls faster. If $\pi(p)$ bends sharply near zero, a rare event receives more decision weight. But the equation does not observe your mind directly. Parameters are estimated from choices under a particular framing, sample, and stake size.

That creates three practical cautions. First, parameter estimates are not portable without argument: a laboratory gamble, a retirement choice, and a concentrated stock position may not activate the same reference point. Second, the model is not a substitute for cash-flow analysis. A psychologically attractive outcome can still be financially destructive. Third, an equation can clarify a decision even when it cannot predict the exact choice of one person.

The right use is comparative. Ask how the decision changes when the reference point changes, when the loss coefficient is higher, or when the probability is expressed as a frequency rather than a percentage. If a small change in assumptions flips the action, the decision is fragile and deserves a larger margin of safety.

## 9. Reference points across a household balance sheet

Investors rarely make decisions in a single account. They have cash, debt, housing, retirement savings, insurance, and future income. Each can create a reference point, and mental accounting can prevent the household from seeing the full risk picture.

### The portfolio reference point

For a portfolio, a useful reference point may be a target allocation rather than a purchase price. If the target is 60% equities and 40% bonds, an equity rally can move the portfolio to 70% equities. The investor may feel good because the account is up, while the portfolio has become more exposed to an equity drawdown.

This is one reason rebalancing can feel uncomfortable. It asks you to sell part of an asset that has produced a gain and buy an asset that has lagged. The emotional frame is “sell the winner, buy the loser.” The portfolio frame is “return the risk exposure to the level chosen before the latest price movement.” Neither frame guarantees a good outcome, but only the second refers to the portfolio policy.

### The debt reference point

Debt creates another asymmetry. A borrower may treat the original loan balance as the reference point and feel no progress until principal falls below a round number. Alternatively, the borrower may compare the interest rate with an investment return and take additional risk to “beat” the debt. That is a framing decision involving certainty, taxes, liquidity, and downside—not simply a comparison of two percentages.

### The housing reference point

Homeowners often anchor on a purchase price or an unrealized property value. A falling quoted value can feel like a loss even when the home continues to provide housing services. A rising value can feel like wealth available to spend even when selling would require moving, paying costs, or taking on new debt.

The lesson is not that one account should always be consolidated. Different accounts can have legitimate purposes. The lesson is that the reference point should match the obligation: retirement wealth should be measured against future spending, emergency cash against near-term needs, and speculative capital against the amount the household can genuinely lose.

### A whole-balance-sheet example

Imagine a hypothetical household with $20,000 in cash, $50,000 in diversified investments, and $30,000 of high-interest debt. Its gross financial assets are $70,000, but its net financial position before other assets is $40,000.

If the household frames a $5,000 investment gain against the $50,000 portfolio, the gain is 10% on that account. If it frames the same gain against the whole net financial position, it is 12.5%. If it frames the household against the $30,000 debt, paying down debt may be the most salient improvement in financial security.

None of those percentages is the single “true” frame. They answer different questions. The mistake would be to use the most flattering percentage to justify a decision that increases the household’s exposure to a risk it cannot absorb.

The intuition is that a portfolio decision is often a balance-sheet decision in disguise.

## 10. Why behavior is stable enough to study but variable enough to respect

Behavioral finance needs a middle position between two extremes. The first extreme says people are perfectly rational, so observed mistakes are irrelevant. The second says people are irrational in a fixed, universal way, so one behavioral coefficient explains everything. Neither is useful.

### Stable patterns

Some patterns recur across experiments and account data:

- outcomes are evaluated relative to a benchmark;
- losses can have greater subjective impact than equal gains;
- attention is limited;
- salient events attract trading;
- overconfidence can increase activity;
- past outcomes influence future risk-taking.

These regularities justify studying them. They also explain why a process can be designed around known failure modes.

### Context-dependent behavior

The magnitude and direction of a bias can vary with:

- experience and financial sophistication;
- whether the decision is repeated or one-off;
- whether the stakes are small or life-changing;
- the availability of liquidity;
- tax and institutional rules;
- the social setting;
- the investor’s current gains or losses.

An investor can be disciplined in a retirement account and reckless in a small trading account. The same person can be risk averse with money needed for rent and risk seeking with a windfall. This is not a refutation of behavioral finance; it is a reminder that the reference point and domain are part of the model.

### The difference between a bias and a bad outcome

A loss does not prove that a decision was biased. A sound decision under uncertainty can lose. Likewise, a gain does not prove skill. To identify a behavioral problem, look for a repeatable process pattern: selling winners for emotional relief, increasing risk after losses to recover a benchmark, or buying only assets that happen to be visible in the news.

The correct unit of analysis is therefore the decision rule over a sample of decisions, not one trade in isolation.

### A practical test for a suspected bias

When you suspect that a decision is being driven by a bias, do not begin by arguing with yourself about whether the bias is “real.” Run a small counterfactual exercise instead.

Write the decision as if you owned nothing. Then write it as if the price had never reached its recent high. Then write it as if a trusted person, rather than you, owned the position. Finally, write the decision as a portfolio choice in which the capital could be allocated to any reasonable alternative. If the answer changes each time, the historical path is doing substantial work.

This test does not tell you which action is correct. It identifies where the reasoning is sensitive to framing. You can then add the missing information: expected cash flows, downside scenarios, liquidity needs, taxes, correlation with the rest of the portfolio, and the date on which the thesis should be reviewed.

It is also useful to distinguish a **preference** from a **bias**. Choosing a certain return because you value sleep, liquidity, or a known obligation is a preference. Choosing the same return only because an account is below a round-number benchmark, while ignoring a changed risk profile, may be a reference-point error. The difference is not whether the choice looks conservative or aggressive; it is whether the reason survives a transparent change in frame.

Behavioral finance therefore gives us a disciplined way to be humble. We do not need to declare ourselves irrational, and we do not need to pretend that a model removes uncertainty. We need to make the hidden benchmark visible, test the probability estimate, and choose a process that remains usable when the account is moving against us.

## Common misconceptions

### “Prospect theory says people are irrational.”

It is better understood as a descriptive model of systematic patterns in judgment and choice. People can be rational relative to their goals, information, and constraints while still using reference points and decision weights that differ from an idealized expected-utility model.

### “Loss aversion means never sell a losing position.”

That conclusion confuses emotional pain with economic value. A losing position can recover, continue falling, or remain below the return available elsewhere. The relevant comparison is the forward distribution from today, not the desire to erase a historical mark.

### “If a probability is 1%, it should matter only 1%.”

Objective probability and decision weight are different concepts in prospect theory. A 1% chance can receive too much or too little attention depending on salience, evidence, and framing. The answer is not to ignore probabilities; it is to inspect how the probability was estimated and how the payoff is distributed.

### “A stock below my purchase price is cheaper.”

It is cheaper than your purchase price, but that says nothing by itself about intrinsic value or expected return. The purchase price is a historical reference point. A fresh-capital test asks whether you would buy the asset today with cash you did not already own in it.

### “Diversification removes behavioral risk.”

Diversification can reduce the damage from one position, but it does not remove framing, overconfidence, panic, or bad probability estimates. An investor can be diversified across many highly correlated assets or can sell the entire portfolio at the worst moment.

### “More data will solve the problem.”

More data can help, but it can also create false precision and selective attention. Behavioral defenses include a clear decision rule, a defined time horizon, a pre-committed review schedule, and a record of what evidence would change the thesis.

## A practical decision protocol

The goal is not to suppress every emotion. Emotions contain information about what matters to you, but they should not silently rewrite the investment problem.

![Unmanaged behavior compared with a checklist and pre-commitment process](/imgs/blogs/prospect-theory-behavioral-finance-7-grid.webp)

### Step 1: Name the reference point

Write down the benchmark before looking for a decision. It might be current wealth, a liability, a retirement goal, a cost basis, or a portfolio allocation. If you use several reference points, label their jobs. Do not let the previous high become a target without deciding that it should be one.

### Step 2: Rewrite the decision forward from today

Ask: “If I held cash instead of this asset, would I initiate the position at today’s price?” This does not mean ignoring taxes, transaction costs, or portfolio context. It means stopping the cost basis from acting as an unexamined veto.

### Step 3: Separate probability from story

List the possible outcomes, the evidence for each probability, and the uncertainty around the estimate. A range is often more honest than a single point. If the payoff is extremely asymmetric, show the probability of permanent loss, not only the attractive upside.

### Step 4: Change the frame

Describe the same decision in four ways:

- gain frame: what can I gain?
- loss frame: what can I lose?
- broad frame: how does this change total portfolio and goals?
- narrow frame: what happens to this one position?

If your answer changes dramatically, pause. The inconsistency may be revealing a hidden reference point rather than a new fact.

### Step 5: Pre-commit the review rule

Define what you will review, when you will review it, and what evidence would invalidate the thesis. A price threshold can be useful, but price alone is not always enough. Business deterioration, balance-sheet stress, a change in time horizon, or a need for liquidity can matter more than a percentage drawdown.

### Step 6: Size the position so emotion is survivable

Position sizing is a behavioral tool. If a normal daily move makes you abandon a sound plan, the position may be too large for your risk capacity or emotional tolerance. The right size is not determined only by the asset’s expected return; it is determined by what lets you continue following the process through uncertainty.

This is educational, not individualized financial advice. A suitable process depends on your goals, liquidity, tax situation, time horizon, and ability to absorb loss.

## When this matters to you

Prospect theory shows up whenever money is compared with a benchmark:

- deciding whether to hold a losing stock;
- choosing between paying down debt and investing;
- evaluating a portfolio after a drawdown;
- deciding whether a “discount” is real or merely below a previous price;
- buying a lottery-like asset because the upside is vivid;
- paying for protection because a remote loss feels certain;
- changing a long-term plan after a short-term headline.

The useful habit is not to ask whether you have biases. You do, as does everyone else. Ask which reference point, probability weight, or frame is controlling this decision, and whether it belongs to the goal you are actually trying to achieve.

> A price tells you where the market is. A reference point tells you why the same price feels different to different people.

## Sources & further reading

- [Kahneman and Tversky, “Prospect Theory: An Analysis of Decision under Risk”](https://www.jstor.org/stable/1914185), *Econometrica*, 1979.
- [Barber and Odean, “Trading Is Hazardous to Your Wealth”](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00226), *Journal of Finance*, 2000.
- [Barber and Odean, “Boys Will Be Boys”](https://academic.oup.com/qje/article-abstract/116/1/261/1939000), *Quarterly Journal of Economics*, 2001.
- [Peng and Xiong, “Investor Attention, Overconfidence and Category Learning”](https://www.nber.org/papers/w11400), NBER Working Paper 11400, 2005.
- [Barber and Odean, “All That Glitters”](https://academic.oup.com/rfs/article-abstract/21/2/785/1607197), *Review of Financial Studies*, 2008.
- [Hong and Stein, “Disagreement and the Stock Market”](https://www.aeaweb.org/articles?id=10.1257%2Fjep.21.2.109), *Journal of Economic Perspectives*, 2007.
- [SEC Staff Report on Equity and Options Market Structure Conditions in Early 2021](https://www.sec.gov/files/staff-report-equity-options-market-struction-conditions-early-2021.pdf), October 18, 2021.
- [HOSE Annual Report 2021](https://staticfile.hsx.vn/Uploads/UploadDocuments/1641899/Bao%20cao%20thuong%20nien%202021.pdf).
- [State Securities Commission market update, November 2022](https://ssc.gov.vn/webcenter/portal/ssc/pages_r/l/chitit?dDocName=APPSSCGOVVN1620126529).
- Related reading: [Mean-variance optimization and the efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants), [Covariance matrices](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants), and [the Vietnam sector investing playbook](/blog/trading/vietnam-stocks/capstone-a-full-vietnam-sector-investing-playbook).
