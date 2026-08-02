---
title: "Loss aversion and the disposition effect: Why investors sell winners and hold losers"
date: "2026-08-01"
description: "A from-zero, evidence-based guide to loss aversion, the disposition effect, breakeven fixation, tax motives, opportunity cost, and better sell decisions."
tags: ["behavioral-finance", "loss-aversion", "disposition-effect", "portfolio-management", "investor-psychology", "risk-management", "capital-gains", "decision-making"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — The disposition effect is the tendency to realize gains too readily and hold losses too long because a realized loss feels like an admission of failure.
>
> - A cost basis is useful for accounting, but it is not a forecast of what an asset is worth today.
> - “I will hold until breakeven” converts a historical price into a future objective and can hide opportunity cost.
> - Odean’s study of 10,000 brokerage accounts found that investors realized gains more readily than losses; the winners they sold subsequently outperformed the losers they kept.
> - Taxes, rebalancing, liquidity, and genuine mean-reversion beliefs can explain some sales, so the disposition effect must be tested rather than assumed.
> - The practical defense is a fresh-capital test: if you had cash today, would you buy this position at today’s price?

Have you ever sold a stock after it rose 20% because you wanted to “lock in the profit,” then watched it rise another 50%? Have you kept a stock that fell 40% because selling would make the loss real? Both decisions can feel sensible in the moment. One protects a gain; the other preserves hope.

The uncomfortable possibility is that the two decisions are not independent. A gain is closed to produce relief. A loss is left open to avoid regret. The portfolio then becomes a record of emotional bookkeeping: realized winners, unrealized losers, and a growing collection of positions whose main investment thesis is “wait until I get back to where I started.”

This pattern is called the **disposition effect**. It is not the claim that every winner should be held forever or every loser should be sold immediately. It is a documented tendency to sell winning investments too early and retain losing investments too long, after accounting for plausible alternatives such as rebalancing and taxes.

![A purchase price becomes a gain or loss label before it becomes a hold, sell, or add decision](/imgs/blogs/loss-aversion-disposition-effect-1-pipeline.webp)

The diagram above is the mental model for this article. You buy at a price, the market moves, and the current price is compared with the historical cost basis. That comparison creates a mental label—winner, loser, or breakeven—which can influence the action more than the forward value of the asset. We will build the mechanism from zero, inspect the evidence, work through the arithmetic, and design a decision process that keeps the past from silently dictating the future.

## Foundations: the position and its reference point

### What is a position?

A **position** is an investment you currently own or owe. If you buy one share of a stock, you have a long position. If you sell a borrowed share, you have a short position. This article focuses mainly on long positions because the winner/loser asymmetry is easiest to see there.

The market value of a long position is the current price multiplied by the number of shares. If you own one share priced at $80, the position is worth $80. Your **cost basis** is the amount used to calculate the gain or loss, usually including the purchase price and sometimes transaction costs depending on the accounting or tax system.

The cost basis is a historical fact. The market value is a current fact. The difference between them is the unrealized gain or loss.

### Paper gains, realized gains, and realized losses

An **unrealized gain** is a position worth more than its cost basis while you still own it. An **unrealized loss** is a position worth less than its cost basis. The gain or loss becomes **realized** when you sell the position.

Suppose you buy one share at $100:

- at $120, you have an unrealized gain of $20;
- at $80, you have an unrealized loss of $20;
- if you sell at $120, the $20 becomes a realized gain;
- if you sell at $80, the $20 becomes a realized loss.

The market value has already changed before the sale. Selling changes the ownership and the accounting label; it does not create the economic loss from nothing. This distinction is central. The decision to sell should be about future value, risk, liquidity, taxes, and alternatives—not only about whether the account statement will display a red or green number.

### The cost basis as a reference point

In the previous article on [prospect theory](/blog/trading/finance/prospect-theory-behavioral-finance), we saw that people evaluate outcomes relative to reference points. A cost basis is one especially powerful reference point because it is precise, personal, and visible in the brokerage account.

The market does not know your cost basis. Two investors can hold identical shares at the same current price and face the same future cash flows while feeling different things. One bought at $60 and sees a gain at $80. The other bought at $100 and sees a loss at $80. The asset is the same; the mental account is different.

### The economic question is forward-looking

Once a position exists, the economically relevant question is:

> If I had the current market value in cash today, would I choose this asset over the available alternatives?

This is the **fresh-capital test**. It does not require pretending that taxes, transaction costs, or portfolio constraints do not exist. It simply prevents the purchase price from becoming the only reason to continue holding.

![The fresh-capital test replaces the historical cost frame with a comparison among today’s alternatives](/imgs/blogs/loss-aversion-disposition-effect-4-before-after.webp)

#### Worked example: the same share, three different investors

Assume one share is currently worth $80.

- Investor A bought at $100 and has a $20 unrealized loss, or -20%.
- Investor B bought at $60 and has a $20 unrealized gain, or approximately +33.3%.
- Investor C has $80 in cash and no position.

The market gives all three the same forward opportunity set. Investor A may want to wait for $100. Investor B may want to sell at $80 to protect the gain. Investor C asks whether $80 is attractive today. The third question is the cleanest starting point because it does not contain a historical emotional commitment.

The intuition is that the cost basis explains how you arrived at today; it does not automatically tell you what to do from today.

## 1. The disposition effect in one picture

![Winners and losers create different emotional questions, but both should be evaluated with the same forward-looking question](/imgs/blogs/loss-aversion-disposition-effect-2-matrix.webp)

The matrix compares two positions. A winner invites “Should I lock it in?” A loser invites “Should I wait until breakeven?” Both questions are emotionally natural. Neither is the central investment question. The central question is whether the current asset is the best use of capital given today’s information.

### The classic pattern

The disposition effect predicts two asymmetric actions:

- sell winners too early, realizing gains before the investment thesis has matured;
- hold losers too long, delaying a sale because realization creates regret.

The word **too** matters. Selling a winner can be rational if the price becomes excessive, the portfolio becomes concentrated, the thesis is complete, or cash is needed. Holding a loser can be rational if the forward expected return is attractive and the risk is acceptable. The disposition effect is a statistical tendency, not a commandment against selling winners or losers.

### Why gains and losses are not psychologically symmetric

A realized gain can produce pride, relief, and a feeling of competence. A realized loss can produce regret, embarrassment, and the painful conclusion that the earlier decision was wrong. If the investor can avoid realizing the loss, the emotional cost is postponed.

This is where loss aversion meets mental accounting. The investor treats each position as a separate account and evaluates the account against its purchase price. A portfolio-level gain may coexist with a painful single-position loss. A sale closes one account and changes the emotional composition of the portfolio.

### Breakeven as a moving target

Breakeven sounds objective, but it is only objective relative to a chosen reference point. If a share was bought at $100, breakeven is $100 before costs and taxes. If the investor received a dividend, added shares, or paid fees, the relevant accounting basis may differ. If the investor’s opportunity cost is a broad index, “breakeven” may mean something else entirely: matching the return that could have been earned elsewhere.

The emotional version of breakeven is especially dangerous because it can survive after the reason for owning the position disappears. The investor no longer asks whether the company is attractive; the investor asks whether the account can return to a number written in the past.

#### Worked example: the recovery arithmetic

Suppose a share falls from $100 to $70. The loss is $30, or 30% of the original price. To return from $70 to $100, the share must rise by

$$
\frac{100 - 70}{70} = \frac{30}{70} \approx 42.9\%.
$$

The required recovery is larger than the original percentage loss because the denominator has changed. If the share falls to $50, the loss is 50%, but the required recovery is

$$
\frac{100 - 50}{50} = 100\%.
$$

At $20, the loss is 80%, and the required recovery is

$$
\frac{100 - 20}{20} = 400\%.
$$

These are arithmetic facts, not forecasts. A position can recover, remain stagnant, or fall further. The calculation simply shows why “wait until breakeven” can require a very different future return from the one the investor originally expected.

The intuition is that the size of the historical loss does not tell you the probability of recovery; it only tells you how large a future gain would be required.

## 2. What the evidence says

### Odean’s brokerage-account study

Terrance Odean’s 1998 paper, “Are Investors Reluctant to Realize Their Losses?”, is one of the central empirical studies of the disposition effect. Odean analyzed trading records for 10,000 accounts at a large discount brokerage, using data from 1987 through 1993. The dataset contained 162,948 records, including account identifiers, trade dates, buy/sell indicators, quantities, commissions, and principal amounts.

The key result was that investors showed a strong preference for realizing winners rather than losers. Odean examined alternative explanations: rebalancing, the possibility that low-priced stocks were more expensive to trade, and beliefs that current losers would later outperform current winners. The pattern remained after controlling for rebalancing and share price. The winning investments investors sold continued to outperform the losing investments they kept in subsequent months.

For taxable investments, Odean also found that tax-motivated selling was most evident in December. This matters because it shows why the disposition effect cannot be interpreted as “every sale of a winner is irrational.” Taxes create a legitimate reason to realize some positions at particular times. The research asks whether tax and portfolio explanations account for the whole pattern; in the study, they did not.

Source: [Odean, “Are Investors Reluctant to Realize Their Losses?”](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00072).

### Shefrin and Statman’s theory

Shefrin and Statman’s 1985 paper placed the pattern in a broader framework involving mental accounting, regret aversion, self-control, and tax considerations. Their title—“The Disposition to Sell Winners Too Early and Ride Losers Too Long”—captures the behavioral asymmetry, but the theory is not limited to one emotional mechanism.

Regret aversion matters because selling a losing position confirms that the earlier decision was wrong. Self-control matters because the investor may understand the forward-looking logic but still find it difficult to act. Taxes matter because the timing of realization changes after-tax outcomes. Mental accounting matters because the investor evaluates a position separately instead of viewing the portfolio as one allocation of capital.

Source: [Shefrin and Statman, “The Disposition to Sell Winners Too Early and Ride Losers Too Long”](https://doi.org/10.1111/j.1540-6261.1985.tb05002.x).

### Evidence outside the original US sample

Grinblatt and Keloharju used a unique Finnish dataset to monitor buys, sells, and holds of individuals and institutions on a daily basis over a two-year period. Their logit regressions found evidence that investors were reluctant to realize losses, engaged in tax-loss selling, and were influenced by past returns and historical price patterns such as monthly highs and lows.

The cross-market evidence is useful for two reasons. First, it suggests the pattern is not an artifact of one US brokerage sample. Second, it shows the behavior coexists with market institutions and tax rules. The magnitude and interpretation can vary across countries, investor types, and time periods; the existence of a recurring pattern does not imply identical behavior everywhere.

Source: [Grinblatt and Keloharju, “What Makes Investors Trade?”](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00338).

### What a study can and cannot prove

An account-data study can reveal a relationship between unrealized gains/losses and realization decisions. It cannot observe every internal thought. A realized gain may reflect a valuation judgment, a liquidity need, a tax rule, or emotional relief. Researchers therefore test alternative explanations and compare subsequent performance.

The correct takeaway is evidence of a systematic trading tendency, not a claim that researchers can read a person’s mind. That distinction makes the result stronger, not weaker: the pattern appears in behavior even when motive is not directly observable.

## 3. Why investors hold losers

### Avoiding the admission of error

Selling a loser turns a private possibility into a public fact on the account statement. The investor must acknowledge that the asset is worth less than the amount paid. Holding preserves the possibility of recovery and protects the self-image of being right eventually.

This is not necessarily conscious deception. The mind can reinterpret new evidence to make holding feel analytical: “The market is wrong,” “the loss is temporary,” or “I will wait for the next catalyst.” Sometimes those statements are correct. The problem is when the investor would not buy the stock today but keeps it solely because selling would feel bad.

### Mean-reversion beliefs

Some investors hold losers because they believe prices revert toward a long-run value. That can be a valid strategy when the asset is mean reverting and the investor has the capital and time to wait. But a stock can be a loser because the long-run cash flows deteriorated, not because the price temporarily deviated from them.

The investor needs to define the source of mean reversion. Is it accounting value, replacement cost, earnings power, sector cyclicality, or simply the historical price? A return to the old price is not a law of nature. Bankruptcy, dilution, technological substitution, and permanent demand loss can prevent it.

### Mental accounting and the closed account

A position is often treated as a separate mental account. The investor may accept a loss in one account only after a gain in another account offsets it. This creates a **break-even effect**: the desire to return to zero can dominate the decision to maximize future wealth.

The account can also be framed differently depending on whether the position is viewed alone or inside a portfolio. A $1,000 loss in a $100,000 portfolio is a 1% portfolio impact. The same loss can feel much larger when it is displayed as a red line next to the purchase price. Broad framing can reduce the emotional intensity, but it should not conceal concentration or risk.

### The sunk-cost mistake

A **sunk cost** is a cost already incurred and unrecoverable. The money paid for a share is sunk from the perspective of today’s decision. The current market value is not sunk: it can be invested elsewhere, held in cash, or used to meet a liability.

The cost basis remains relevant for taxes and record-keeping. It is not irrelevant in every context. The behavioral mistake is using a sunk historical cost as if it were a claim on the future.

![A purchase at $100 creates a path through loss, breakeven, and gain labels as the market price changes](/imgs/blogs/loss-aversion-disposition-effect-3-timeline.webp)

![A loss can lead to regret, tax considerations, or a forward-looking review; only the last one asks what the position is worth today](/imgs/blogs/loss-aversion-disposition-effect-5-graph.webp)

The graph separates mechanisms that can converge on the same action. Regret can encourage holding, tax timing can encourage either realization or deferral, and a forward review can justify selling, holding, or resizing. The observed action alone does not identify the motive.

#### Worked example: holding for breakeven versus reallocating

Imagine a hypothetical portfolio with a $1,000 position now worth $700. You can either hold it or sell and invest the $700 in an alternative.

If the original position has an expected return of 2% over the next year, its expected value after one year is

$$
\$700 \times 1.02 = \$714.
$$

If the alternative has an expected return of 8%, its expected value is

$$
\$700 \times 1.08 = \$756.

$$

The difference is $42 before taxes, fees, and uncertainty. The original $1,000 cost basis does not change either forward calculation. If the original position has a stronger risk-adjusted opportunity, holding may be sensible. If not, waiting for $1,000 is an emotional target rather than an economic reason.

The intuition is that opportunity cost begins with the current market value, not the amount originally paid.

## 3.5 Realization utility: why closing the account feels different

The disposition effect becomes easier to understand when we distinguish the value of owning an asset from the value of realizing the outcome. A paper gain can disappear. A realized gain can be mentally consumed, celebrated, or moved into a “safe money” account. A paper loss can remain psychologically reversible; a realized loss cannot.

This idea is sometimes called **realization utility**: the act of closing a position can carry its own psychological value or cost. It is an explanatory framework, not a universal law. Some investors feel relief when they close a losing position because uncertainty ends. Others feel regret. The direction depends on the reference point, the investor’s self-image, and the surrounding portfolio.

### Realization changes the story, not just the ledger

Before selling, an investor can say, “The market has not recognized the value yet.” After selling, the investor may say, “I was wrong.” The second statement is not always true—the investment may have been reasonable when made—but the emotional narrative is more final.

This finality can create two opposite mistakes. The investor may close a winner to create a completed success. Or the investor may avoid closing a loser to preserve an unfinished possibility. Both actions optimize the emotional state of the account rather than the expected use of capital.

### Realization and information quality

A sale also creates information about the investor’s own decision. If the sale is based on a thesis failure, realizing the loss is useful feedback. If it is based only on discomfort, the feedback is ambiguous. The investor may learn “losses are painful” rather than “the thesis was wrong.”

This is why decision journals should record the reason for realization. A later outcome cannot tell you whether the original reasoning was good unless the reasoning was written down before the result.

### Realization across a whole portfolio

Investors often treat a realized gain as available “house money” and a realized loss as a permanent reduction in wealth. Both are accounting frames. In a portfolio, wealth is the total current value plus liabilities, not the emotional sum of closed and open mental accounts.

If one position rises $200 and another falls $200, selling the winner does not make the portfolio whole. It only changes which position is closed. A broad portfolio review asks whether the combined exposures still fit the objective.

#### Worked example: two mental accounts, one economic position

Imagine a hypothetical portfolio with two shares:

- Share A was bought at $100 and is now $120, a $20 paper gain.
- Share B was bought at $100 and is now $80, a $20 paper loss.

The total invested amount was $200 and the current value is also $200. If the investor sells A and keeps B, the account now shows a realized $20 gain and an unrealized $20 loss. If the investor sells B and keeps A, it shows a realized $20 loss and an unrealized $20 gain. The economic value is the same before taxes and costs.

The psychological experience is not the same. The first version feels successful because the realized account is green. The second feels like failure because the realized account is red. The portfolio-level wealth has not changed; the mental accounting has.

The intuition is that realization can rearrange the emotional labels without improving the portfolio.

## 4. Why investors sell winners

### The pleasure of realization

Selling a winner provides a clear, completed success. The investor no longer has to worry that the paper profit will disappear. A realized gain can be mentally booked as evidence of skill, even when the sale was premature.

### Risk reduction can be rational

A winner may become too large a share of the portfolio. Selling part of it can reduce concentration. A position may also have reached a valuation target, changed in risk, or become unsuitable for the investor’s time horizon. These are legitimate reasons to sell a winner.

The test is whether the sale is connected to the portfolio and evidence. “It is up, so I should sell” is not a complete rule. “It is now 35% of the portfolio, while my policy limit is 15%” is a risk-management reason.

### The opportunity cost of cutting winners

The disposition effect can remove positive exposure from the portfolio. If the investor sells every winner after a small gain, long-run compounding is interrupted. The investor may then buy a new position, incur costs, and repeat the cycle.

This is not an argument for never taking profits. It is an argument for distinguishing profit realization from thesis completion. A company can be a winner because fundamentals improved, because valuation expanded, or because the entire market rose. The next decision depends on what created the gain and whether that cause persists.

#### Worked example: compounding interrupted by an early sale

Suppose a hypothetical investment grows from $100 to $120. You sell and realize a 20% gain. A different investor holds the position while it grows another 25% from $120 to $150.

The holder’s total gain is

$$
\frac{150 - 100}{100} = 50\%.
$$

The seller’s realized gain is 20%, but the seller also has $120 that must be redeployed. If the replacement investment earns 0%, the seller finishes with $120 while the holder has $150. If the replacement earns 25%, the seller finishes with $150 before costs and timing differences. The sale itself is not the problem; the reinvestment decision determines whether it was helpful.

The intuition is that selling a winner is only half of a capital-allocation decision; the destination of the proceeds matters.

## 5. Taxes, rebalancing, and legitimate reasons that resemble a bias

### Tax-loss selling

Taxes can create a rational incentive to realize losses or defer gains. The exact rules depend on jurisdiction, account type, holding period, and the investor’s tax position. A loss may be useful if it offsets taxable gains, but selling solely for a tax benefit can still be poor if the replacement asset has worse risk or if transaction costs erase the benefit.

Odean’s study found tax-motivated selling was most visible in December. This seasonal pattern is important because it demonstrates that realization behavior can contain both psychological and institutional components. Researchers should not call every year-end loss sale a disposition effect.

### Rebalancing

Rebalancing means adjusting a portfolio back toward a target allocation. A winner can be sold because it has grown beyond its intended weight. A loser can be bought because its weight has fallen below target. The actions look like selling winners and holding or buying losers, but the motivation is portfolio construction.

The cleanest way to separate rebalancing from disposition is to record the target weights before prices move. If the policy says an asset should be 10% and it rises to 15%, a partial sale follows from the policy. If the policy is invented only after the sale, it may be a post-hoc explanation.

### Liquidity needs

An investor may sell a winner because cash is needed for a bill, or hold a loser because selling would create a tax or liquidity problem. A household’s financial decision is constrained by obligations. A portfolio cannot be evaluated separately from the balance sheet.

The presence of a legitimate constraint does not make behavior irrational. It changes the objective. The investor may be maximizing financial resilience rather than expected portfolio return.

### Portfolio insurance and risk limits

A winner can be sold to reduce volatility, meet a risk budget, or comply with an investment mandate. A loser can be held because it hedges another exposure. The same position-level action can be correct or incorrect depending on the whole portfolio.

This is why the disposition effect should be measured at the position and portfolio levels. A trade that looks like premature profit-taking in isolation may be sensible when it removes a concentration risk.

| Reason for sale | Position-level appearance | What to verify |
|---|---|---|
| Emotional relief | Winner sold | Was there a forward thesis review? |
| Rebalancing | Winner sold | Was a target allocation defined before? |
| Tax management | Winner or loser sold | Does the after-tax benefit exceed costs? |
| Thesis failure | Winner or loser sold | What evidence changed? |
| Liquidity need | Any position sold | Is the cash need real and time-bound? |

## 6. Institutional and professional versions of the same problem

The disposition effect is not limited to inexperienced individuals. Professional investors operate under mandates, reporting dates, bonuses, tax rules, and career concerns. Those constraints can create different reference points: a fund’s purchase price, the start-of-year NAV, a benchmark, a high-water mark, or the manager’s previous recommendation.

### Reporting and career reference points

A manager may avoid realizing a loss near a reporting date because it makes performance look worse, or sell a winner to show a realized success. A fund may also realize losses to offset gains or rebalance. The observed trade is therefore the result of both psychology and institutional design.

This does not weaken the behavioral argument. It shows that reference points are created by organizations as well as individuals. A quarterly performance table can make a calendar date emotionally important even when the underlying investment horizon is multi-year.

### High-water marks and incentive asymmetry

A **high-water mark** is a previous peak value used in some performance-fee arrangements. It can create a reference point above the investor’s original capital. If the fund is below the high-water mark, a manager may face incentives to reduce risk, increase risk to recover, or change positions for reputational reasons. The appropriate behavior depends on the mandate and the contract, but the reference point is real and consequential.

Individual investors create informal high-water marks too. The highest account value becomes an implicit promise, and a fall from that peak feels like a loss even if the portfolio remains above the original contribution. A risk policy should state whether drawdown is measured from the peak, the starting balance, or the goal; otherwise each stressful period can change the benchmark.

### Why experience may help but not eliminate the effect

Experience can improve awareness of costs and tax implications. It may also teach an investor to use predetermined exit rules. But experience can create new anchors: a previous successful trade, a remembered recovery, or confidence in a familiar sector. The goal is not to become immune. It is to build a process that makes the behavior visible.

#### Worked example: the benchmark is part of the mandate

Imagine a hypothetical fund starts a year at NAV $100, rises to $120, and later falls to $110. A limited view says the fund is down $10 from its peak. A starting-capital view says it is up $10. A benchmark view may say it outperformed or underperformed depending on the benchmark return.

The three numbers are not interchangeable. If the mandate is to protect capital, peak drawdown may dominate. If the mandate is long-term growth, total return from the starting capital may dominate. If the mandate is relative performance, the benchmark matters. The same fund can be described as “down 8.3% from peak” and “up 10% from start” without contradiction.

The intuition is that a reference point is part of the objective function; changing it changes the decision problem.

## 7. A decision tree for selling

![A sell decision should separate thesis change, portfolio risk, cash need, and mere regret](/imgs/blogs/loss-aversion-disposition-effect-6-tree.webp)

The tree gives four branches. A changed thesis can justify selling. A changed portfolio risk can justify resizing. A cash need can override return optimization. If the only reason is regret or the desire to see a green number, slow down and apply the fresh-capital test.

### Branch one: did the thesis change?

A thesis is the causal explanation for why an investment should produce an acceptable return. It can involve earnings growth, valuation, asset value, a catalyst, or a risk premium. The thesis changes when evidence weakens the causal chain.

Examples include:

- expected demand is permanently lower;
- competition removes pricing power;
- management changes capital allocation;
- debt or dilution changes the payoff;
- the valuation no longer compensates for the risk.

The purchase price is not evidence that the thesis remains intact.

### Branch two: did portfolio risk change?

A position can become inappropriate even if the company remains attractive. A strong winner may become too large. Correlations can rise during a crisis. A change in income, debt, or family obligations can reduce the investor’s ability to absorb volatility.

This branch often leads to a partial sale rather than an all-or-nothing decision. The purpose is to make the portfolio survivable, not to predict the exact top.

### Branch three: is cash needed?

If an investor needs cash for a known obligation, liquidity can dominate expected return. The correct preparation is to match the time horizon of the asset with the time horizon of the liability. Selling a long-duration asset to pay a near-term bill can be rational even at a loss.

The behavioral mistake is not “selling at a loss.” It is allowing a known cash need to remain unplanned until the market chooses the sale price.

### Branch four: is the only reason emotional relief?

Regret is information about discomfort, not automatically about value. If the thesis is intact, portfolio risk is acceptable, and cash is not needed, a sale motivated only by the desire to remove a red number deserves a pause. Conversely, discomfort can reveal that the position is too large. The response may be resizing rather than pretending emotion has no information.

## 8. The forward-looking mathematics of opportunity cost

### Current value is the starting capital

Suppose a position cost $1,000 and is now worth $700. The opportunity-cost question compares the expected distribution from $700 in the current asset with the distribution from $700 in alternatives. The $300 historical loss is relevant to performance measurement and taxes, but it is not an additional $300 that must be recovered before the next decision can be made.

This is easy to say and difficult to feel. The account statement combines historical and current information in one line. A decision note should separate them.

### Expected return is not enough

The alternative with the highest expected return may have unacceptable downside, liquidity, or correlation. A forward-looking comparison should include:

- expected return range;
- probability of permanent loss;
- time to liquidity;
- transaction costs and taxes;
- effect on total portfolio risk;
- evidence quality.

The fresh-capital test is not a command to chase the asset with the biggest forecast. It is a way to compare choices on the same current-capital basis.

#### Worked example: opportunity cost with uncertainty

Assume a hypothetical $700 position has three possible one-year outcomes:

- $840 with probability 50%;
- $700 with probability 30%;
- $420 with probability 20%.

Its expected value is

$$
0.50 \times 840 + 0.30 \times 700 + 0.20 \times 420 = 420 + 210 + 84 = \$714.
$$

Now compare an alternative with outcomes:

- $805 with probability 70%;
- $560 with probability 30%.

Its expected value is

$$
0.70 \times 805 + 0.30 \times 560 = 563.50 + 168 = \$731.50.

$$

The alternative has a higher expected value in this hypothetical example, but its suitability still depends on liquidity, correlation, and the investor’s loss tolerance. The historical $1,000 cost basis does not change either expected-value calculation.

The intuition is that opportunity cost compares the current $700 with alternatives; it does not compare the alternatives with the old $1,000.

### The cost of waiting

Holding a loser has a cost even when no cash fee is charged. Capital is unavailable for another opportunity, the position may consume attention, and the investor may become more concentrated. Waiting can be correct when it buys time for a thesis to work. It is costly when it merely postpones an uncomfortable decision.

The cost of waiting should be expressed in a decision rule. “I will wait” is incomplete. “I will review after the next filing; if margin remains below the thesis threshold, I will exit” turns waiting into a testable plan.

## 9. Taxes, averaging down, and measurement traps

The clean fresh-capital test is a starting point, not the entire decision. Real portfolios contain tax lots, transaction costs, cash-flow needs, and rules about concentration. These frictions can make the economically best action different from the emotionally easiest action. They can also make a sensible decision look like the disposition effect from the outside.

### Tax-loss selling is a calculation, not a mood

Suppose a taxable account owns a position with a $2,000 unrealized loss. Selling may create a tax loss that can offset taxable gains, subject to the rules of the investor’s jurisdiction. The value of that tax asset should be compared with the cost of selling, the risk of the position, and the replacement investment. The correct question is not “Can I avoid realizing the loss?” It is “What is the after-tax value of each available path?”

An illustrative calculation makes the distinction concrete. Assume a hypothetical $2,000 loss produces a tax benefit worth $400, while selling and replacing the position costs $30 in spread and fees. The immediate net benefit is not $2,000; it is approximately $370 before considering price movement, holding-period rules, or restrictions on repurchasing a substantially identical security. If the position is highly risky and the replacement has similar exposure, the tax benefit may support a sale. If the replacement changes the portfolio’s risk dramatically, the tax benefit is only one input.

The timing can matter too. A loss harvested in December and a loss harvested in June may have different value depending on realized gains, income, filing rules, and the probability that the investor can use the deduction. Tax advice is jurisdiction-specific; the behavioral lesson is general: tax language should be translated into an explicit cash-flow estimate.

### Averaging down can be rational—or a disguised need to be right

Adding to a losing position lowers the average purchase price. That accounting result is certain. The investment result is not. A lower average basis can make breakeven appear closer while increasing exposure to the same thesis.

Consider a hypothetical investor who buys 10 shares at $100 and then 10 more at $60. The total cost is $1,600 for 20 shares, so the average basis is $80. If the price later reaches $80, the account appears to have recovered to breakeven before fees. But the investor has not escaped the original decision; the investor has doubled the capital exposed to the asset. The relevant question at $60 was whether the next $600 should be invested there, not whether the average basis would become more comfortable.

Averaging down deserves a fresh thesis. The investor should be able to explain why the expected return improved, why the risk of permanent impairment did not rise more than proportionally, and why the resulting position size remains acceptable. If the only argument is “the loss is larger, so I should buy more,” the arithmetic is serving the reference point rather than the investment case.

### Performance measurement can reward the wrong behavior

Realized-profit reports are incomplete. An account that realizes many small gains and leaves a few large losses open can look successful on a transaction list while producing poor total returns. Conversely, an account that realizes a loss to remove a deteriorating position may look worse in the month of the sale while improving its future risk.

Use at least three views:

1. **Total-return view:** include realized and unrealized gains, income, fees, and cash flows.
2. **Allocation view:** measure the return and risk contribution of each position relative to the portfolio.
3. **Decision view:** record what was known when the action was taken and whether the rule was followed.

The third view prevents hindsight from turning every profitable sale into a mistake or every lucky recovery into proof of skill. If a sold winner rises afterward, the decision may still have been correct if valuation or concentration risk justified the sale. If a held loser recovers, holding may still have been undisciplined if it was based only on breakeven.

### A useful audit table

For each position under review, write down the following before looking at the realized P&L:

| Question | Evidence to record | Decision implication |
| --- | --- | --- |
| What is the asset worth today? | Cash flows, valuation, credit, or market data | Establish a forward estimate |
| What changed since purchase? | New filings, prices, rates, competition, liquidity | Separate thesis change from price change |
| What is the next-best use of capital? | Alternative return, risk, fees, taxes | Measure opportunity cost |
| How large should the position be? | Portfolio weight, correlation, liability horizon | Consider resizing |
| What would prove the view wrong? | A number, event, or date | Make holding falsifiable |

The table does not eliminate uncertainty. It makes the uncertainty comparable across winners and losers. That symmetry is the practical opposite of the disposition effect.

![A five-row audit matrix pairs each question—value today, what changed, next-best use, position size, invalidation—with the shallow cost-basis-only answer it replaces](/imgs/blogs/loss-aversion-disposition-effect-8-matrix.webp)

## 10. How it shows up in real markets

### Case study 1: Odean’s 10,000-account evidence

Odean’s account data are the central case because they make the disposition effect observable at scale. The study analyzed records from 10,000 accounts at a large discount broker over 1987–1993. Investors realized gains more readily than losses. After controlling for rebalancing and share price, the behavior remained. More importantly, the winners selected for sale subsequently outperformed the losers investors retained.

That last result challenges the comforting story that investors simply sold winners because those winners had lower future prospects. If the sold winners later did better than the kept losers, the realization pattern was not a reliable way to identify inferior assets.

The paper also records tax-motivated selling, especially in December. This is a model example of careful behavioral research: identify the pattern, test reasonable alternatives, and report where institutional incentives explain part of the result without explaining all of it.

Source: [Odean, “Are Investors Reluctant to Realize Their Losses?”](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00072).

### Case study 2: Finnish investors and historical price patterns

Grinblatt and Keloharju studied Finnish stock-market trading with daily observations of individuals and institutions over two years. Their analysis found that past returns and historical price patterns, including whether a stock was near a monthly high or low, affected buying and selling. The evidence also supported reluctance to realize losses and tax-loss selling.

The Finnish case helps separate two ideas. The disposition effect is not only about a US investor looking at a green or red position. Historical price location itself can influence action. A monthly high can invite profit-taking; a monthly low can invite either bargain hunting or avoidance. The same price pattern can trigger opposite actions depending on the investor’s existing position and reference point.

The lesson is to treat chart location as a description of the path, not as a sufficient reason for a decision. A stock at a monthly low may be cheap, impaired, or both.

Source: [Grinblatt and Keloharju, “What Makes Investors Trade?”](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00338).

### Case study 3: Vietnam’s 2020–2021 winners and 2022 losers

HOSE reported that the VN-Index ended 2021 at 1,498.28 points, up 35.7% from the end of 2020. It also reported average daily trading value of nearly VND 21,997 billion in 2021. In a market with strong liquidity and rising prices, investors had many opportunities to create paper gains and new reference points.

When the market later corrected, individual positions did not all behave alike. An investor who bought a sector early might still see a gain while another investor in the same sector saw a loss. The label “winner” or “loser” therefore depended on entry date, not only on the business.

This is where the disposition effect can interact with sector rotation. Early winners may be sold to lock in gains even while the sector’s earnings thesis remains intact. Late entrants may hold losers to recover their cost basis even after the sector’s risk has changed. The proper decision requires separating entry price from sector evidence and portfolio exposure.

Source: [HOSE Annual Report 2021](https://staticfile.hsx.vn/Uploads/UploadDocuments/1641899/Bao%20cao%20thuong%20nien%202021.pdf).

### Case study 4: Vietnam’s 2022 correction and the cost of waiting

The State Securities Commission reported that the VN-Index stood at 960.65 on November 21, 2022, down 35.9% from the end of 2021. It reported average trading value falling from VND 26,299 billion per session in April to VND 12,124 billion in November. The same update discussed declining confidence in corporate bonds and reported private-placement issuance of VND 329,296 billion through November 11, down 28.5% from the same period of 2021.

A behavioral reading should not erase the market’s financing and regulatory realities. A holder of a losing stock or bond may have been responding to credit conditions, liquidity, or disclosure risk rather than simply refusing to admit a mistake. But the reference-point mechanism still matters: “wait for the old price” can keep capital in an asset while the underlying financing environment changes.

The lesson is to review the thesis and liquidity conditions, not only the distance from the purchase price. A loss can be temporary, but the reason for owning the asset can also be permanently impaired.

Source: [State Securities Commission market update](https://ssc.gov.vn/webcenter/portal/ssc/pages_r/l/chitit?dDocName=APPSSCGOVVN1620126529).

## 11. A practical protocol for resisting the disposition effect

![A disposition-resistant review uses thesis status, opportunity cost, and a pre-committed review date instead of only realized P&L](/imgs/blogs/loss-aversion-disposition-effect-7-before-after.webp)

The figure contrasts a P&L-led decision with an evidence-led decision. The goal is not to make every decision emotionless. It is to place the emotion next to the analysis rather than let it replace the analysis.

### Step 1: Record the position without the purchase price

Write the current market value, number of shares, portfolio weight, and forward thesis. Hide the cost basis for the first pass if necessary. This is a practical way to force the fresh-capital question.

### Step 2: State what would make you buy today

Write the reasons a new investor would initiate the position now. If the reasons are only “it is below my cost” or “it used to trade higher,” you have identified an anchor, not a thesis.

### Step 3: State what would invalidate the thesis

Choose observable conditions: revenue, margin, leverage, customer retention, competitive position, valuation, or a date-based catalyst. A thesis without an invalidation condition can absorb any evidence and become impossible to disprove.

### Step 4: Compare alternatives at current value

Compare the current position with cash, a diversified asset, debt repayment, or another investment. Include risk, liquidity, taxes, and correlation. Do not compare every alternative with the original cost basis; that makes the old position look like the default winner or loser.

### Step 5: Decide whether the action is full, partial, or none

Not every disposition problem requires a complete sale. Rebalancing can be partial. A thesis may be weakened but not destroyed. Liquidity needs may require selling the most liquid portion. The important thing is to connect the size of the action to the evidence.

### Step 6: Set a review date

If the thesis needs time, specify the time. “Hold for now” becomes more disciplined when it means “review after the next annual report” or “review when the contractual milestone is reached.” Time without a review rule is often just avoidance.

### Step 7: Evaluate the process after the outcome

Record whether the thesis was correct, whether the decision followed the rule, and whether the outcome was influenced by luck. A sale that is followed by a price increase is not automatically a mistake. A hold that is followed by a recovery is not automatically a good decision. Process evaluation protects against hindsight bias.

This is educational, not individualized financial advice. Tax treatment, position sizing, and suitable risk depend on the investor’s circumstances and jurisdiction.

## 12. A final diagnostic: bias, information, or constraint?

Not every asymmetric trade is evidence of a psychological error. A useful diagnosis separates three causes that can produce the same visible action.

### Bias

The action is bias-driven when the investor’s reason is anchored to the cost basis, the desire to avoid regret, or the desire to create a realized gain. Typical language includes “I only need to get back to even,” “I do not want to admit I was wrong,” or “I should sell because it is green.” The test is counterfactual: if the position were offered today at its current price, would the investor choose it for the same amount of capital?

### Information

The action is information-driven when new evidence changes the estimate of future cash flows or risk. A weaker balance sheet, a lost customer, a new competitor, or a changed interest-rate environment can justify selling a loser. A winner can also be held when new evidence improves the thesis. The price label is incidental; the information is causal.

### Constraint

The action is constraint-driven when taxes, liquidity, mandate limits, concentration, or a liability horizon changes the feasible choices. A taxable investor may harvest a loss. A household may sell an appreciated asset to fund tuition. A fund may trim a position because its weight exceeds policy. These are not automatically behavioral errors, though the constraint should be recorded before the trade rather than invented afterward.

### The counterfactual sale test

Before executing, write two sentences:

1. “If I did not own this position, I would [buy/not buy] it today because …”
2. “I am selling or holding despite that answer because …”

The first sentence tests the forward thesis. The second identifies the friction. If the answers conflict, the conflict is useful: perhaps taxes justify waiting, perhaps the position is too large, or perhaps the investor is protecting a self-image. A disciplined process does not require the friction to be zero; it requires the friction to be visible and priced.

This diagnostic also improves conversations between investors. Instead of arguing “sell the loser” versus “hold the loser,” ask whether the disagreement is about value, risk, taxes, or emotion. Two people can reach different actions rationally when their constraints differ. The behavioral warning applies when the stated reason is merely a historical price that no longer affects the forward opportunity set.

The deepest lesson is modest: the past should be recorded, but it should not be allowed to vote alone. A cost basis is evidence about the path taken. It is not evidence that the same path remains the best route from today.

That distinction is small in wording but large in practice: it turns “wait until breakeven” into a hypothesis that can be tested against alternatives, risks, and evidence.

The practical payoff is a cleaner decision: own the asset for what it may become, not for what the account once displayed.

## Common misconceptions

### “The disposition effect means always sell losers.”

No. A loser can be the best forward opportunity in the portfolio. The effect is a tendency to hold losers because they are losers, not a rule that every losing position is bad.

### “Realized gains are better than paper gains.”

Realization changes ownership and accounting, not the underlying quality of the next opportunity. A realized gain can be useful for taxes, liquidity, or risk control, but realization alone does not create value.

### “If the stock returns to my cost basis, the decision worked.”

Returning to breakeven may feel satisfying, but the position may have consumed years of capital and attention. Compare the result with alternatives and with the risk taken during the waiting period.

### “Taxes are just an excuse for holding losers.”

Taxes are a real constraint, and tax-loss selling can be rational. But the after-tax benefit should be compared with the investment’s forward risk, trading costs, and replacement opportunity. Tax language should explain the calculation, not end the discussion.

### “Selling a winner early is always a mistake.”

A winner can become too large, too expensive, or inconsistent with the portfolio’s risk budget. The mistake is selling only because the position is green, without examining the forward thesis and alternatives.

### “The cost basis does not matter at all.”

It matters for taxes, performance attribution, and understanding the history of the position. It should not be the only reason to hold or sell. Accounting relevance and decision relevance are different.

## When this matters to you

The disposition effect appears whenever the account’s historical label competes with today’s opportunity:

- holding a stock because it is below your purchase price;
- selling a fund because it finally recovered;
- refusing to rebalance a winner;
- adding to a loser solely to lower the average cost;
- waiting for a bond or property investment to return to par;
- keeping a position because selling would make a mistake visible;
- treating a realized gain as proof that the research process worked.

The useful habit is to keep two records. The first is historical: what did you buy, when, and at what cost? The second is forward-looking: what is the current value, what evidence supports the thesis, what could invalidate it, and what else could the capital do?

> Breakeven is a number in the past. Capital allocation is a decision about the future.

## Sources & further reading

- [Odean, “Are Investors Reluctant to Realize Their Losses?”](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00072), *Journal of Finance*, 1998.
- [Shefrin and Statman, “The Disposition to Sell Winners Too Early and Ride Losers Too Long”](https://doi.org/10.1111/j.1540-6261.1985.tb05002.x), *Journal of Finance*, 1985.
- [Grinblatt and Keloharju, “What Makes Investors Trade?”](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00338), *Journal of Finance*, 2001.
- [Barber and Odean, “Trading Is Hazardous to Your Wealth”](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00226), *Journal of Finance*, 2000.
- [Barber and Odean, “The Behavior of Individual Investors”](https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/behavior%20of%20individual%20investors.pdf), review chapter.
- [HOSE Annual Report 2021](https://staticfile.hsx.vn/Uploads/UploadDocuments/1641899/Bao%20cao%20thuong%20nien%202021.pdf).
- [State Securities Commission market update, November 2022](https://ssc.gov.vn/webcenter/portal/ssc/pages_r/l/chitit?dDocName=APPSSCGOVVN1620126529).
- Related reading: [Prospect theory](/blog/trading/finance/prospect-theory-behavioral-finance), [mean-variance optimization](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants), and [the Vietnam sector investing playbook](/blog/trading/vietnam-stocks/capstone-a-full-vietnam-sector-investing-playbook).
