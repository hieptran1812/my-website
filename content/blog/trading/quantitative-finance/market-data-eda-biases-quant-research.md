---
title: "Working with market data: EDA, survivorship bias, and look-ahead bias for quant research"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Before any alpha exists you have to understand and clean your data. This is a hands-on, first-principles tour of market data, exploratory analysis, and the three silent biases — survivorship, look-ahead, and non-point-in-time fundamentals — that inflate a backtest and that a quant-researcher take-home case is built to expose."
tags:
  [
    "market-data",
    "survivorship-bias",
    "look-ahead-bias",
    "point-in-time-data",
    "exploratory-data-analysis",
    "backtesting",
    "quant-research",
    "quant-interviews",
    "data-cleaning",
    "as-of-join",
    "quantitative-finance"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Before you can find a signal, you have to trust your data; the biases that quietly inflate a backtest are exactly what a quant-researcher take-home case is designed to expose.
>
> - **Market data comes in four families** — prices/quotes, fundamentals, corporate actions, and alternative data — and each has its own units, frequency, and traps. You cannot compute a single correct return without handling splits and dividends first.
> - **Survivorship bias** is the silent deletion of every company that died. A strategy backtested only on names that still trade today can look like it earns **+14.0%** a year when the honest number is **+7.5%**, and its Sharpe ratio can fall from **1.30** to **0.55** once you put the dead names back.
> - **Look-ahead bias** is trading on information you could not have known at the time. Using a *restated* earnings number instead of the one actually filed is the single most common leak, and it can manufacture a Sharpe ratio out of thin air.
> - **Point-in-time data** records what was knowable on each date; a *restated* database silently overwrites history. The fix is the **as-of join** — attach to every trade date only the latest value whose knowledge date is already in the past.
> - **A clean holdout** is tested exactly once. Snoop it a hundred times and your out-of-sample test quietly becomes in-sample, and the live Sharpe collapses toward zero.
> - The one habit to remember: **for every number, ask "when did I actually know this?"** If the answer is "later," it is a leak.

Here is a number that should make every aspiring quant uncomfortable. You can take a perfectly ordinary momentum strategy, backtest it on twenty years of US stock data, and get a Sharpe ratio of 1.3 — the kind of number that gets you a second-round interview. Then you can fix a *single* property of the dataset, change nothing about the strategy, and watch that same Sharpe ratio fall below 0.6. The strategy did not change. The data did. More precisely, the *honesty* of the data did.

This is the part of quant research nobody puts on a poster. The glamorous part is the model. The part that decides whether the model is real is the data underneath it — and almost every catastrophic backtest failure traces back not to a bug in the model but to a bias in the data that made the past look more predictable than it actually was. A take-home data case from Two Sigma, Citadel, DE Shaw, AQR, or Point72 is rarely "build the best model." It is far more often "here is a messy dataset; show me you know where the bodies are buried."

![The market-data taxonomy: every quant dataset is one of four families, each with its own units, frequency, and traps.](/imgs/blogs/market-data-eda-biases-quant-research-1.png)

The diagram above is the mental model for the whole post: every dataset a quant touches is one of four families — prices and quotes, fundamentals, corporate actions, and alternative data — and each family carries its own units, its own update frequency, and its own way of lying to you. We are going to build all of it from zero. We will define every term the first time it appears. We will work every important idea through with real dollar figures you can check in your head. And we will spend most of our time on the three biases that separate a backtest you can trust from one that is quietly cheating: survivorship, look-ahead, and non-point-in-time fundamentals.

By the end you will be able to take a raw CSV of prices and fundamentals and answer the question that actually matters in an interview room: *is the edge in this backtest real, or is it an artifact of how the data was assembled?*

## Foundations: what market data actually is

Let us start with the rawest possible question. When a quant says "I have the data," what is in the file?

### A price is a number attached to a moment

A **price** is what someone paid (or is willing to pay) for one unit of an asset at one instant. That sounds trivial, but the word "price" hides three different things that beginners routinely confuse:

- The **bid** is the highest price a buyer is currently willing to pay. The **ask** (or *offer*) is the lowest price a seller is currently willing to accept. The gap between them is the **bid-ask spread** — the small, ever-present cost of trading. If the bid is $99.98 and the ask is $100.02, the spread is $0.04, or 4 cents. You buy at the ask and sell at the bid, so you start every round-trip already $0.04 in the hole on a $100 stock.
- The **last traded price** is the price of the most recent actual transaction. It sits somewhere between (or at) the bid and ask.
- The **mid price** is just the average of bid and ask — here $(99.98 + 100.02)/2 = \$100.00$. Quants often use the mid as a "fair" reference because it is not biased toward buyer or seller.

A **tick** is a single event in this stream: one quote update or one trade, time-stamped to the microsecond. Tick data is the most granular thing you can get, and a single liquid US stock can throw off millions of ticks a day. Most research does not use raw ticks; it uses *bars*.

### OHLCV: the daily summary every backtest starts with

An **OHLCV bar** compresses all the ticks in a period (usually one day) into five numbers:

- **Open** — the first traded price of the period.
- **High** — the highest traded price.
- **Low** — the lowest traded price.
- **Close** — the last traded price.
- **Volume** — the total number of shares (or contracts) traded.

A year of daily OHLCV for one stock is about 252 rows (there are roughly 252 trading days in a US year). This is the workhorse format: small enough to load instantly, rich enough to compute returns, volatility, and most signals. When a take-home hands you a CSV, it is almost always OHLCV.

### Fundamentals describe the company, not the trade

**Fundamentals** are the accounting facts about the business behind the stock: revenue, earnings, assets, debt, cash flow. They come from the company's filings — in the US, the quarterly 10-Q and annual 10-K reports submitted to the Securities and Exchange Commission (the SEC, the federal regulator of public markets). The two core statements are the **income statement** (what the company earned over a period — revenue minus costs equals profit) and the **balance sheet** (what it owns and owes at a single point in time — assets on one side, liabilities plus equity on the other).

Fundamentals update on a completely different clock from prices. Prices tick continuously; fundamentals arrive in lumps, four times a year, *weeks after* the period they describe. That lag is the seed of look-ahead bias, and we will come back to it.

### Corporate actions: the events that rewrite the price

A **corporate action** is something the company does that mechanically changes its share price or share count without changing the value of what you own. The two that matter most for data hygiene:

- A **stock split** multiplies the number of shares and divides the price by the same factor. In a 2-for-1 split, one $100 share becomes two $50 shares. You own the same $100 of company either way — but the raw price series shows a sudden 50% "drop" that is not a loss.
- A **dividend** is a cash payment to shareholders, usually quarterly. When a company pays a $2 dividend, $2 of cash leaves the company per share, so on the **ex-dividend date** (the first day a buyer is *not* entitled to the dividend) the price drops by roughly $2 — again, not a loss to an existing holder, who got the $2 in cash.

A third corporate action, the **delisting**, is where survivorship bias lives. A delisting is when a stock stops trading on its exchange — because the company went bankrupt, was acquired, merged, or fell below listing standards. The name simply disappears from the live data feed. If your dataset only contains names that are *still trading today*, every delisted name has been silently deleted, and that deletion is not random.

### Price return vs total return

A **return** is the percentage change in your wealth over a period. There are two flavors, and confusing them is a classic interview trip-wire.

- The **price return** uses only the change in price: $R_{\text{price}} = (P_{\text{end}} - P_{\text{start}}) / P_{\text{start}}$.
- The **total return** adds in any cash you received along the way (dividends): $R_{\text{total}} = (P_{\text{end}} - P_{\text{start}} + D) / P_{\text{start}}$, where $D$ is the dividend per share.

For a stock that pays no dividend, the two are identical. For a dividend-payer, total return is always higher, and over decades the gap is enormous: roughly a third of the long-run return of the US stock market has come from dividends, not price appreciation. A backtest that uses price return when it should use total return will systematically understate the performance of dividend-paying stocks — and *over*state the apparent edge of any strategy that happens to avoid them.

![Price return versus total return on a $100 stock: a dividend leaves total return unchanged even though the raw price drops.](/imgs/blogs/market-data-eda-biases-quant-research-2.png)

The timeline above traces a single share through a year, and it is worth walking through because it ties the whole foundations section together. We will make it the first worked example.

#### Worked example: total return through a dividend and a split

You buy one share on January 1 for **$100.00**. Here is everything that happens to it over the year.

1. **March 15 — ex-dividend.** The company pays a **$2.00** dividend. On the ex-date the price mechanically drops by the dividend, from $103 (say it had drifted up to $103 the day before) to **$101**. You now hold one share worth $101 *plus* $2 in cash. Your wealth did not fall — it moved from "all in the share" to "share plus cash."
2. **June 30 — drift.** Through ordinary trading the price rises to **$108.00**. Nothing mechanical here; this is real economic gain.
3. **September 1 — 2-for-1 split.** Your one $108 share (say it is $108 at the split) becomes **two shares at $54** each. You hold $108 of stock either way. The share *count* doubled; the *value* did not.
4. **December 31 — sell.** Each share is now worth **$58**. You sell both for $58 × 2 = **$116**.

Now compute the return two ways.

The **naive price return** off the raw close: you "bought at $100" and the final *raw* close per share is $58. That looks like $(58 - 100)/100 = -42\%$. That is nonsense — it has not accounted for the split, which artificially halved the price.

The **correct total return**: your ending wealth is the $116 from selling the two shares *plus* the $2 dividend you banked in March = **$118**. You started with $100. So your total return is $(118 - 100)/100 = +18\%$.

The lesson in one sentence: **the raw price series is not a return series — you must adjust for every split and dividend before a single percentage means anything.**

## Exploratory data analysis: look before you model

Before you fit anything, you look. **Exploratory data analysis** (EDA) is the disciplined habit of staring at your data — its shape, its gaps, its outliers — until you understand how it was generated and where it might be lying. In quant research, EDA is not optional warm-up; it is where most of the real findings (and most of the disqualifying mistakes) live.

### Return distributions: the bell curve is a lie

The single most important EDA plot is the **histogram of daily returns** — a chart that counts how many days fell into each return bucket. The instinct from a statistics class is to expect a **normal distribution** (the symmetric "bell curve"), because so much of classical statistics assumes it. Real equity returns are not normal, and the difference is the difference between surviving a crash and being wiped out by one.

![A daily-return distribution has a tall central peak and fat tails, so a normal model badly understates crash risk.](/imgs/blogs/market-data-eda-biases-quant-research-4.png)

The panel above shows what a real daily-return distribution looks like (the blue bars) against the normal curve a textbook would fit (the dashed line). Two features jump out:

- **A taller, narrower peak.** Most days, the market barely moves. Returns cluster near zero far more tightly than a bell curve predicts. This is sometimes called *excess kurtosis* — kurtosis being a measure of how "peaked" a distribution is.
- **Fat tails.** Out at the extremes — the days the market falls 5% or more — there are *far* more observations than the normal curve allows. The red bars on the left are crash days the bell curve says should essentially never happen.

How much more? Under a normal distribution, a daily move of more than five standard deviations should occur about once every **3.5 million years**. In real equity markets, five-sigma days show up every few years. The October 19, 1987 crash, when the S&P 500 fell about 20% in a single session, was something like a 20-sigma event under a normal model — which is to say, the model was not wrong by a little, it was wrong by an amount with no physical analogue. The practical consequence: any risk number you compute assuming normality will *understate* the chance of a disaster, often by orders of magnitude.

#### Worked example: why the standard deviation alone misleads you

Suppose you measure a strategy's daily returns and find a mean of **+0.04%** per day and a standard deviation of **1.0%** per day. A naive analyst annualizes and reports a tidy story: expected return ≈ $0.04\% \times 252 \approx +10\%$ a year, volatility ≈ $1.0\% \times \sqrt{252} \approx 15.9\%$ a year, Sharpe ≈ $10/15.9 \approx 0.63$.

Now look at the actual worst days. Under the normal assumption, your worst day in a year (252 draws) should be about $-2.7$ standard deviations, i.e. roughly **−2.7%**. But when you sort the real daily returns, you find the worst day was **−7.0%** — nearly *seven* standard deviations of loss in a single session. On a $1,000,000 book, the model told you to brace for a $27,000 bad day; the real bad day cost **$70,000**. If you had sized your positions off the normal model, you were carrying more than twice the tail risk you thought.

The lesson: **the mean and standard deviation summarize the calm middle of the distribution; the money is made and lost in the tails the bell curve cannot see.** Always look at the empirical extremes directly.

### Missing data: separate "absent" from "never existed"

The second pillar of EDA is **missing data** — cells in your table with no value. In market data, "missing" is not one thing; it is at least three, and conflating them is how leaks creep in.

![A missing-data heatmap separates a true delisting from a benign vendor gap at a glance.](/imgs/blogs/market-data-eda-biases-quant-research-5.png)

The heatmap above plots eight tickers (rows) against twelve months (columns); green means data is present, red means it is absent. Three completely different stories produce red cells:

- **A delisting.** Ticker CCC trades cleanly through May and then goes solid red from June onward. The company died — bankrupt, acquired, or delisted — and the data ends because the name ceased to exist. This red block is *meaningful*: it is precisely the kind of name survivorship bias deletes. You must keep it, and you must record the delisting return (often a large negative, sometimes −100%).
- **A pre-existence gap.** Ticker DDD is red until May, then green. The company had not gone public yet — it IPO-ed in May. The red here is *absence of the entity*, not a data error. You should not forward-fill it or treat it as a missing observation; before May, this stock simply was not in your investable universe.
- **A vendor gap.** Ticker FFF has scattered red cells (March, July) surrounded by green. The company traded fine; the data vendor just dropped a value. This is a genuine *gap* you might fill (carefully) or drop.

The reason this matters: a common beginner reflex is to "clean" a dataset by dropping every row with a missing value, or by forward-filling every gap. Drop the delisting rows and you have just *reintroduced survivorship bias*. Forward-fill a pre-IPO period and you have invented a price for a stock that did not trade. The correct handling depends entirely on *why* the value is missing, and a missing-data heatmap is the fastest way to tell the three apart.

```python
import pandas as pd

  # panel: rows = dates, cols = tickers, values = adjusted close
panel = prices.pivot(index="date", columns="ticker", values="adj_close")

  # Where is data missing, and is it a delisting or a gap?
is_missing = panel.isna()
  # A delisting: missing from some date onward and never returns.
last_valid = panel.apply(lambda c: c.last_valid_index())
delisted = last_valid[last_valid < panel.index.max()]
print(delisted)   # these names MUST stay in the universe with their final return
```

### Outliers: a real crash or a fat-fingered tick?

An **outlier** is a value far from the rest of the distribution. In returns, a single day of +400% is almost never a real move; it is usually a bad tick, an unadjusted split, or a data error. But you cannot just delete every extreme value, because *some* extremes are real crashes you must keep. The discipline is to investigate each one:

- A return of exactly −50% or +100% on a single day is a screaming sign of an **unadjusted split** (a 2-for-1 split looks like a −50% return if you forgot to adjust). Check the corporate-action calendar before deleting.
- A return of +1,900% followed the next day by −95% is usually a bad print — a single erroneous tick that got recorded as a close. These you clean.
- A return of −22% on a known crash date is real. You keep it.

A simple, robust screen is to flag any daily return outside, say, ±50% and route it to manual review against the split/dividend calendar — never auto-delete.

### Split and dividend adjustment: the back-adjustment that makes returns honest

We met splits and dividends in the foundations; now we make them operational. The goal of **adjustment** is to produce a single continuous price series whose day-to-day percentage changes equal the *true* total return, with all the mechanical jumps removed.

![Back-adjusting a price series removes the split and dividend jumps so the return series reflects only real economic change.](/imgs/blogs/market-data-eda-biases-quant-research-3.png)

The chart above shows the idea. The solid line is the **raw close** — it jumps *down* at the dividend (green band) and *halves* at the split (amber band). The dashed line is the **adjusted close** — smooth, with the jumps removed, so that its percentage changes are the real returns. The standard technique is **back-adjustment**: you take the most recent price as truth and walk *backward*, scaling every earlier price by the cumulative split and dividend factors, so the series is continuous and the latest price is unchanged.

#### Worked example: building the adjustment factor

Take a stock that closes at **$50** the day before a 2-for-1 split, then **$25** the day after (the split halved it). Without adjustment, the day's return looks like $(25 - 50)/50 = -50\%$ — a fake crash. To fix it, you multiply every price *before* the split by the **split factor** $1/2 = 0.5$. Now the pre-split $50 becomes an adjusted $25, and the day's adjusted return is $(25 - 25)/25 = 0\%$ — correct, because nothing actually happened to your wealth.

Dividends work the same way but with a smaller factor. If a $25 stock pays a $0.50 dividend, the **dividend adjustment factor** for prices before the ex-date is $1 - D/P = 1 - 0.50/25 = 0.98$. You multiply all earlier prices by 0.98, which lifts the *returns* on dividend days by the right amount so that the adjusted series captures total return. Vendors usually ship a single **cumulative adjustment factor** per day that folds in every split and dividend up to the present; the **adjusted close** is just the raw close times that factor.

```python
  # Total daily return from adjusted close handles splits AND dividends at once.
panel["ret"] = panel["adj_close"].pct_change()

  # Sanity check: a raw -50% one-day move that vanishes after adjustment was a split.
raw_ret = panel["close"].pct_change()
suspicious = raw_ret[(raw_ret < -0.45) & (panel["ret"].abs() < 0.05)]
print(suspicious)   # these dates had a split the raw series didn't account for
```

The one-sentence intuition: **always compute returns from the adjusted close, never the raw close — the adjusted series is the only one whose percentage changes are real.**

## Survivorship bias: the names that died

Now we reach the first of the three big biases, and the one that most often shows up in a take-home case. **Survivorship bias** is the error of analyzing only the things that *survived* a selection process, when the things that did *not* survive carry exactly the information you need.

The canonical illustration comes from the Second World War. Engineers studying returning bombers wanted to add armor where the planes showed the most bullet holes — the wings and fuselage. The statistician Abraham Wald pointed out the fatal flaw: they were only looking at the planes that *came back*. The planes hit in the engines and cockpit did not return to be studied. The armor belonged where the surviving planes had *no* holes, because hits there were lethal. The data they could see was systematically missing the cases that mattered most.

### Why a stock universe must include the dead

In market data, the "planes that did not come back" are delisted companies — the bankruptcies, the failed mergers, the names that fell off the exchange. If you build a stock universe by downloading "all stocks currently in the S&P 500" and then pulling their history back twenty years, you have committed the exact error. The names that *left* the index — because they collapsed — are not in your list. Your historical universe is a curated collection of *winners*, because membership today is conditioned on having survived.

![Survivorship shrinks the past universe: filtering to names that still trade today silently deletes every company that died.](/imgs/blogs/market-data-eda-biases-quant-research-6.png)

The chart above shows the mechanism. The solid line is the **true universe** — the number of names that actually existed and traded in each past year, roughly flat at around 650. The dashed line is the **survivors-only sample** — the subset of those names that are *still trading today*. As you walk backward in time, the survivors-only line shrinks, because more and more of the companies that existed back then have since died. The shaded red region between the two lines is the deleted gap: about **150 names** — that delisted, merged, or went bankrupt — that a survivors-only dataset silently throws away.

That deletion is not random. The deleted names are disproportionately the *worst performers* — the firms that lost so much value they got delisted. Remove the worst performers from history and the average performance of what remains rises mechanically. Your backtest is now running on a universe that was hand-picked, after the fact, to look good.

#### Worked example: how much does survivorship inflate a $10,000,000 backtest?

Let us make this concrete with a number you can carry into an interview. You are backtesting an equal-weighted long-only strategy on a **$10,000,000** book over one year, picking from a universe of **650 names** that existed at the start of the year.

The honest, point-in-time universe has all 650 names. Suppose **150** of them performed badly enough that, by today, they have delisted; their average return over the test year was a brutal **−40%** (many of them on their way to zero). The surviving **500** names averaged a healthy **+15%**.

The **honest** backtest weights all 650 names equally. Its return is the blend:

$$ R_{\text{honest}} = \frac{500}{650} \times 15\% + \frac{150}{650} \times (-40\%) = 11.54\% - 9.23\% = +2.3\% $$

On $10,000,000 that is a gain of **$230,000** — modest, and honest.

Now the **survivorship-biased** backtest. You built the universe from "names that still trade today," so the 150 delisted names are gone before you start. You only hold the 500 survivors, and your return is simply their average:

$$ R_{\text{biased}} = +15\% $$

On $10,000,000 that is a gain of **$1,500,000**. The bias did not change your strategy by one line of code; it changed your reported return from **$230,000** to **$1,500,000** — it inflated the profit by **$1,270,000**, more than 6×, purely by deleting the losers from the past. An interviewer who hands you a "great" backtest and asks "what's wrong with this?" is very often pointing at exactly this gap.

The Sharpe ratio is hit just as hard. Recall the **Sharpe ratio** is the average excess return divided by the volatility of returns — a measure of return per unit of risk. The deleted names were not just low-return; they were *high-volatility* (collapsing companies swing wildly). Removing them lowers both the numerator (less true downside) and, deceptively, the apparent risk. In the before/after figure earlier, the same strategy went from a hireable-looking **Sharpe 1.30** on the survivors-only universe to a barely-a-signal **Sharpe 0.55** on the point-in-time universe.

![Survivorship bias inflates both the return and the Sharpe ratio of the same strategy.](/imgs/blogs/market-data-eda-biases-quant-research-7.png)

The before/after above is the survivorship story in one frame: drop the dead names (left, amber) and the same strategy shows **+14.0%** and Sharpe **1.30**; keep them in a point-in-time universe (right, green) and it collapses to **+7.5%** and Sharpe **0.55**. The one-sentence intuition: **a backtest universe must contain every name that existed *as of the test date*, dead ones included — anything else is grading the strategy on a test it already knows the answers to.**

#### Worked example: building a point-in-time universe as of a past date

So how do you build the honest universe? The rule is to reconstruct membership *as it was* on the date in question, using only information available then. Suppose you want the investable universe as of **March 31, 2015**. The naive (biased) query is "all stocks in the index today." The correct query is:

1. Start from a database that records, for each stock, its **first trading date** and its **delisting date** (if any).
2. Keep a name if and only if it satisfies: `first_trade_date <= 2015-03-31` **and** (`delist_date is null` **or** `delist_date > 2015-03-31`). In words: it had already started trading by your date, and it had not yet delisted.
3. Crucially, do *not* filter on anything that happened *after* March 31, 2015 — not "still trading today," not "in the index now."

```python
asof = pd.Timestamp("2015-03-31")
universe = securities[
    (securities["first_trade_date"] <= asof) &
    (securities["delist_date"].isna() | (securities["delist_date"] > asof))
]
print(len(universe))   # names alive AND tradeable on 2015-03-31 — dead names of the
                       # future are still here, because on this date they were alive
```

A company that thrived until 2018 and then went bankrupt is *in* your 2015 universe, because on March 31, 2015 it was alive and tradeable. A company that IPO-ed in 2017 is *out*, because it did not exist yet. This is the same logic the as-of join uses for fundamentals, which we turn to next. The lesson: **point-in-time membership asks "was this name alive and tradeable on the date?" — never "did it survive to today?"**

## Look-ahead bias: using data you could not have known

The second big bias is the subtlest and the most dangerous because it can manufacture a signal out of pure noise. **Look-ahead bias** is using, in a decision dated time $t$, any piece of information that was not actually available until *after* $t$. You are letting your backtest peek at the future.

It sounds like something you would obviously never do. In practice it is astonishingly easy to do by accident, because of one fact about fundamental data: **the number is finalized weeks before it is released, and revised weeks after.**

### The reporting lag, and the two dates every fact has

Every fundamental fact has (at least) two dates attached:

- The **event date** (or period-end date) — when the thing being measured actually happened. A company's Q1 revenue is "as of" March 31, the last day of the quarter.
- The **knowledge date** (or filing/announcement date) — when the public could first *know* the number. The company files its Q1 results with the SEC several weeks after the quarter ends — typically late April for a calendar-quarter company.

The gap between them, usually four to eight weeks, is the **reporting lag**. If your backtest attaches Q1 revenue to *March 31* and starts trading on it then, you are trading on a number that nobody would actually possess until late April. That is look-ahead bias, and it is the most common reason a "great" fundamental strategy evaporates in live trading.

![The look-ahead leak: trading on a number before its public release date is how a backtest silently cheats.](/imgs/blogs/market-data-eda-biases-quant-research-8.png)

The timeline above lays out the leak precisely. The quarter *ends* on March 31 (the event). The earnings are not *filed* until April 28 (the knowledge date). The earliest a strategy can honestly act on the number is April 29 — the next open after the filing. The leak (red) is any trade placed in that April 1-to-April 27 window, when the backtest "knows" a number the market did not yet have. And there is a second, nastier wrinkle: on August 10 the company *restates* the Q1 number — revises it after the fact — which brings us to the most expensive mistake of all.

### Restated vs as-reported: the leak that fabricates Sharpe

Here is the trap that catches even experienced people. Many fundamental databases are **restated** — they show you the *most recent, corrected* version of every historical number. If a company reported Q1 earnings of $1.20 in April and later restated them up to $1.60 in an August amendment, a restated database shows you **$1.60** for Q1, overwriting the $1.20 that was actually known at the time.

A **point-in-time** (PIT) database, by contrast, preserves what was knowable on each date: it shows $1.20 if you query "as of May," and only updates to $1.60 once your query date passes the August restatement.

![A restated database overwrites history while a point-in-time vintage preserves what you could actually know.](/imgs/blogs/market-data-eda-biases-quant-research-10.png)

The matrix above contrasts them row by row. The point-in-time column (green) shows the as-filed values — $1.20 for Q1 EPS, $400M revenue — knowable on the filing date, safe to trade. The restated column (red) shows the revised values — $1.60, $440M — that were only knowable months later, and that therefore leak the future if you use them in a backtest.

Why is using restated numbers so destructive? Because restatements are *correlated with the stock's future*. Companies that later restate earnings *upward* tended to be doing well; companies that restate *downward* were often in trouble (sometimes heading for an accounting scandal). A backtest that uses the restated number is implicitly using knowledge of how things turned out. It is not a small effect.

#### Worked example: the Sharpe inflation from a restated-earnings leak

Build a simple strategy: each quarter, go long the stocks with the highest reported earnings surprise (actual EPS minus the consensus estimate). Backtest it two ways.

*Version A (leaked, restated data).* You use a restated database. For a basket of names that later got their earnings revised *upward*, your "surprise" is computed against the inflated restated number, so you systematically buy exactly the names whose restatements signaled strength. Suppose this version posts an average quarterly excess return of **+3.0%** with a quarterly volatility of **3.0%**. Annualizing, the Sharpe is roughly $\frac{3.0\% \times 4}{3.0\% \times \sqrt{4}} = \frac{12\%}{6\%} = 2.0$. A Sharpe of 2.0 is spectacular — the kind of number that should make you suspicious *because* it is so good.

*Version B (honest, point-in-time data).* Now you rebuild it using only as-reported numbers known on each filing date. The restatement edge vanishes — you can no longer see which names will be revised up. The average quarterly excess return falls to **+0.6%** with the same 3.0% volatility. The annualized Sharpe is $\frac{0.6\% \times 4}{6\%} = \frac{2.4\%}{6\%} = 0.4$.

The leak turned a real-but-weak Sharpe of **0.4** into a fantasy Sharpe of **2.0** — a fivefold inflation, on a strategy that is genuinely marginal. On a $25,000,000 book, the leaked version promised roughly $3,000,000 of annual excess profit; the honest version delivers about $600,000, and the $2,400,000 difference was never real money — it was the backtest reading the answer key. The lesson: **use the number that was *filed*, not the number that was later *corrected*; a Sharpe that looks too good is usually a date error, not a discovery.**

## Timestamp alignment and the as-of join

Everything above — survivorship, reporting lag, restatements — reduces to one operation done correctly: **aligning every piece of data to the moment you actually knew it, and joining it to your trade dates accordingly.** The tool for this is the **as-of join** (also called a "point-in-time join" or, in some libraries, `merge_asof`).

A normal join matches rows by an exact key. An **as-of join** matches each row in your left table (your trade dates) to the *most recent* row in the right table (your data) whose knowledge date is *less than or equal to* the trade date. In plain English: for each day you want to trade, attach the latest fact you could possibly have known by that day — and nothing newer.

![An as-of join attaches to each trade date only the latest value whose knowledge date already lies in the past.](/imgs/blogs/market-data-eda-biases-quant-research-9.png)

The diagram above shows the logic for a signal dated April 29. The as-of cutoff is "knowledge date ≤ April 29." Three candidate values exist: Q4 EPS (known January 30 — valid but stale), Q1 EPS (known April 28 — valid and freshest), and the Q1 restatement (known August 10 — *invalid*, because its knowledge date is in the future relative to April 29). The as-of join picks Q1 EPS (the latest *valid* value) and *rejects* the restatement as a future leak. This single operation, done on the *knowledge* date rather than the *event* date, is what makes a fundamental backtest honest.

#### Worked example: an as-of join that prevents the leak

In pandas, `merge_asof` does exactly this — but only if you point it at the right column. The bug that creates look-ahead bias is joining on the **event date** (period-end) instead of the **knowledge date** (filing date).

```python
import pandas as pd

  # Trade dates: when we want to act.
signals = pd.DataFrame({"trade_date": pd.to_datetime(
    ["2015-04-15", "2015-04-29", "2015-05-15"])}).sort_values("trade_date")

  # Fundamentals carry BOTH dates. knowledge_date is the filing date.
fundamentals = pd.DataFrame({
    "period_end":     pd.to_datetime(["2014-12-31", "2015-03-31"]),
    "knowledge_date": pd.to_datetime(["2015-01-30", "2015-04-28"]),  # filing dates
    "eps":            [1.05, 1.20],
}).sort_values("knowledge_date")

  # CORRECT: join on knowledge_date — attach only what was filed by the trade date.
joined = pd.merge_asof(
    signals, fundamentals,
    left_on="trade_date", right_on="knowledge_date",
    direction="backward",          # most recent value at or before the trade date
)
print(joined[["trade_date", "eps"]])
  # 2015-04-15 -> eps 1.05  (Q1 not filed until Apr 28, so we only know Q4)
  # 2015-04-29 -> eps 1.20  (Q1 now filed; we may use it)
  # 2015-05-15 -> eps 1.20
```

Notice the April 15 trade gets the *Q4* number (1.05), not Q1 (1.20) — because on April 15 the Q1 filing had not happened yet. Join on `period_end` instead and April 15 would wrongly pick up the Q1 number two weeks early. The lesson: **always join on the knowledge date and always use `direction="backward"` — the as-of join is your structural defense against look-ahead bias.**

### Timezones: the other silent misalignment

A close cousin of the reporting lag is the **timezone** trap. Markets around the world close at different wall-clock times. If you join a US signal computed at the US close (4:00 p.m. Eastern) with a European price stamped at the European close (which already happened, around 11:30 a.m. Eastern that same day), the European data is *stale* — fine. But if you accidentally join *tomorrow's* Tokyo open (which, in UTC, has a date that looks "same-day" from a US perspective) you have leaked the future across the international date line. The fix is mundane but essential: **store every timestamp in UTC, record the exchange's local close, and only ever join data whose timestamp precedes your decision time in absolute (UTC) terms.**

## Data snooping vs a clean holdout

The final discipline is about *how you test*, not how you assemble data — but it fails for the same reason: it lets information leak from where it should not. **Data snooping** (also called *p-hacking* or *overfitting to the test set*) is the practice of trying so many strategies, parameters, or variations against your evaluation data that one of them looks good *by chance alone*.

### Why a holdout exists

The defense is a **holdout** (or *out-of-sample*) set: a slice of data you set aside at the very beginning, do not look at, and use exactly once at the end to estimate how the strategy will perform on data it has never seen. The data you *do* develop on is the **training** (or *in-sample*) set.

![A clean holdout sits in the future of the training data with a gap, and is touched only once.](/imgs/blogs/market-data-eda-biases-quant-research-11.png)

The figure above shows the clean construction. The **training** block (blue) is the earlier ~75% of history — 2005 to 2020 — where you fit and tune everything. Then comes a small **embargo** (amber) — a one-month gap — to stop information bleeding across the boundary (a feature computed with a 20-day window near the end of training would otherwise overlap into the holdout). Finally the **holdout** (green) is the most recent ~25% — 2021 to 2025 — which you test *once*. The honest path (green box) runs the holdout a single time and reports that number. The data-snooping path (red box) tests a hundred variants against the holdout and keeps the best, which quietly turns the out-of-sample test back into in-sample.

#### Worked example: how 20 tries manufacture a false discovery

The arithmetic of snooping is worth knowing cold, because interviewers love it. A standard significance test uses a threshold of **p < 0.05** — meaning that if a strategy were truly worthless, there is only a 5% chance it would *look* this good by luck. So if you test *one* worthless strategy, you have a 5% false-positive rate. Fine.

But now suppose you test **20** independent worthless strategies and keep the best. The chance that *at least one* clears p < 0.05 by pure luck is:

$$ P(\text{at least one false positive}) = 1 - (1 - 0.05)^{20} = 1 - 0.95^{20} = 1 - 0.358 = 0.642 $$

There is a **64%** chance you "discover" a significant strategy even though every one of them is garbage. Test 100 variants and the probability of at least one spurious "winner" rises to $1 - 0.95^{100} \approx 99.4\%$ — you are essentially *guaranteed* a false discovery. This is why a strategy that was tuned heavily on the holdout has a live Sharpe that collapses toward zero: you did not find an edge, you found the luckiest draw among many.

The defenses are concrete: test the holdout *once*; if you must run multiple hypotheses, correct for it (e.g. a Bonferroni correction divides your 0.05 threshold by the number of tries, so 20 tries require p < 0.0025); and prefer strategies with an *economic reason* to work over ones found by search. The lesson: **every extra look at the holdout spends some of its statistical power; a holdout you have peeked at a hundred times is worth nothing.** For the deeper machinery here see [hypothesis testing and p-values for quant interviews](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) and the way bias and variance trade off in [estimators, MLE, bias and variance](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews).

## In the interview room and the take-home

Now let us put it together the way a take-home case actually arrives. You are handed a dataset and a vague prompt — "explore this and tell us if there's a tradeable signal" — and the entire exercise is a test of whether you can find the traps before you find the (often illusory) alpha. Here are five fully-worked problems of the kind these cases pose.

#### Worked example: "this backtest shows 25% a year — what's wrong with it?"

The interviewer shows you a long-only US equity backtest, 2005 to 2025, equal-weighted, rebalanced monthly, posting **+25%** annualized with a Sharpe of 1.8. The universe was built as "current S&P 500 constituents, pulled back to 2005."

Your answer, in order: **survivorship bias is the prime suspect.** "Current S&P 500 constituents pulled back" means the universe is the *winners* — every company that fell out of the index (often because it collapsed) is missing. To quantify it, rebuild the universe point-in-time: include every name that *was* in the index on each historical date, dead ones included, using their actual delisting returns. I would expect the +25% to fall by a third to a half. As a second check, confirm the returns come from the *adjusted* close (so splits and dividends are handled) and that the rebalance does not trade on month-end prices that were not yet known. A Sharpe of 1.8 on a simple long-only strategy is itself a red flag — real long-only equity Sharpe ratios cluster well below 1. The number is too good, which usually means a date or a universe error rather than a discovery.

#### Worked example: "we use the latest fundamentals — is that okay?"

The candidate dataset has a `revenue` column and a `report_date` column, but the documentation says values are "as currently reported." The trap: "as currently reported" means **restated** — the numbers have been revised since they were first filed.

Your answer: this is a look-ahead leak waiting to happen. The `revenue` shown for Q1 2018 may include corrections made in 2019 or later, and restatements are correlated with the stock's future, so any strategy using these numbers is peeking. The fixes, in order of preference: (1) obtain a **point-in-time** vintage of the data that records the as-filed value and its knowledge date; (2) if only restated data exists, apply a *conservative reporting lag* — never use a fundamental until at least, say, 45-90 days after the period-end, which at least prevents the worst leaks even if it cannot undo restatements; (3) flag in your writeup that the restatement bias cannot be fully removed without PIT data, and discount the backtest accordingly. The honest move is to *name the limitation*, not to hide it.

#### Worked example: "build the universe as of June 30, 2010"

The task: from a securities master table with `ticker`, `first_trade_date`, and `delist_date`, return the investable universe as of June 30, 2010.

Your answer is the point-in-time filter from earlier, stated crisply: keep a name if `first_trade_date <= 2010-06-30` and (`delist_date is null` or `delist_date > 2010-06-30`). The two failure modes to call out: (1) filtering on "still active today" — which deletes the dead and reintroduces survivorship; (2) filtering on `first_trade_date` alone while forgetting the delist condition — which would wrongly include companies that had *already* delisted before your date. The correct universe contains exactly the names that were alive *and* tradeable on the date, including the ones that would later die. On a $50,000,000 portfolio, getting this filter wrong is not a rounding error — it is the difference between a real estimate and a fantasy.

#### Worked example: "catch the leak in this feature pipeline"

You are shown a feature built as `momentum = price[t] / price[t-252] - 1`, joined to a label `forward_return = price[t+21] / price[t] - 1`, and a fundamental feature `pe_ratio` joined on `period_end`. Find the leak.

Two leaks. First, the **fundamental join is on `period_end`**, not the filing date — so the P/E ratio for Q1 is attached to March 31, weeks before it was knowable. Fix: join on the **knowledge date** with a backward as-of join. Second, watch the **label boundary**: `forward_return` looks 21 days ahead, so the *last* 21 rows of any training period must be embargoed from the holdout, or the 252-day momentum window and the 21-day label will overlap the train/test boundary and leak. The price-momentum feature itself (`price[t]/price[t-252]`) is fine — it only uses the past — provided the prices are adjusted. State both leaks, give the one-line fix for each, and you have nailed the question.

#### Worked example: "quantify the survivorship inflation on this $10M strategy"

The interviewer hands you the biased and unbiased universes and asks for the dollar impact. You already have the template from earlier: blend the survivors' return with the delisted names' return at their true weights. With 650 names, 150 delisting at −40% and 500 surviving at +15%, the honest return is +2.3% (**$230,000** on $10,000,000) versus the biased +15% (**$1,500,000**). State the inflation as a number — **$1,270,000**, a 6.5× overstatement — and explain *why* it is one-directional: the deleted names are non-randomly the worst performers, so removing them can only inflate. Bonus points for noting that the Sharpe inflation is even larger because the deleted names were also the most volatile, so dropping them shrinks the apparent risk too.

A few more rapid-fire prompts these cases love, with the one-line answer each wants:

- **"Why is your worst day −7% when the model says −2.7%?"** Returns are fat-tailed; the normal model understates the tails, so size off the empirical extremes, not the standard deviation.
- **"Your strategy avoids dividend payers and beats the market — real?"** Check that you are comparing *total* return, not price return; price return understates dividend payers and flatters anything that avoids them.
- **"You forward-filled missing fundamentals — problem?"** Forward-filling across a delisting invents data for a dead company; forward-filling across a gap is fine, but you must distinguish the two first.
- **"Sharpe 2.0 on a fundamental signal — convince me."** Default suspicion is a restated-data leak; rebuild on point-in-time data and watch it fall.

## Common misconceptions

**"Missing data just means drop the row."** This is the single most common way beginners *reintroduce* survivorship bias. A missing value because a company *delisted* is the most informative observation in your dataset — it is a name that died — and dropping it deletes exactly the losers you need. A missing value because the company had not *IPO-ed* yet is not missing at all; the entity did not exist. Always diagnose *why* a value is absent before touching it.

**"The latest, most accurate data is the best data."** For backtesting, the *most accurate* data is precisely the wrong data, because accuracy here means *corrected after the fact*. A restated earnings number is more accurate than the as-filed one, and using it is a textbook look-ahead leak. In research you want the *least* hindsight-contaminated data — what was knowable then — not the most polished version available now.

**"A higher backtest Sharpe is always better."** A suspiciously high Sharpe is more often evidence of a leak than of a discovery. Real, capacity-meaningful equity strategies live in the Sharpe 0.5-1.5 range after costs; a long-only Sharpe of 2+ almost always means survivorship, look-ahead, or snooping. The correct reaction to a beautiful number is to hunt for the bug, not to celebrate.

**"Returns are roughly normal, so standard deviation captures the risk."** Daily equity returns have fat tails; extreme days happen orders of magnitude more often than a normal model predicts. Sizing positions or computing value-at-risk off the standard deviation alone will systematically understate the chance of a large loss. Look at the empirical tails directly.

**"Adjusting for splits is enough; dividends are small."** Over short windows, sure. Over decades, dividends compound into roughly a third of total US equity return. A backtest that uses split-adjusted-but-not-dividend-adjusted prices (price return instead of total return) will understate every dividend payer and quietly bias any cross-sectional comparison. Use the fully adjusted close, which folds in both.

**"If I cross-validate, I can reuse the test set freely."** Cross-validation reduces variance in your estimate, but every time you look at the same evaluation data to make a decision, you spend statistical power. Run 50 variants through the same cross-validation and pick the best, and you have snooped just as surely as with a single holdout. The discipline of "touch the final holdout once" is not replaced by cross-validation; it sits on top of it.

## How it shows up in real research

**The Long-Term Capital Management collapse (1998).** LTCM's convergence-arbitrage models were calibrated on historical relationships that, in the data, looked beautifully stable and mean-reverting. The historical data, however, was thin on true crises — the sample under-represented the fat-tail days when correlations rush to one and liquidity vanishes. When Russia defaulted in August 1998, the "five-sigma" moves the models deemed near-impossible arrived in clusters, and the fund lost roughly $4.6 billion in months. The lesson is the EDA lesson: the tails the normal model cannot see are exactly where the firm-ending losses live.

**Mutual fund performance studies.** Academic researchers studying mutual fund returns found that early databases only contained funds that *still existed*, deleting the ones that had closed — almost always the poor performers. Studies that corrected for this survivorship bias found that average fund performance was meaningfully lower than the survivors-only databases suggested, by on the order of 1-1.5% a year. The funds that died took their bad returns out of the dataset with them, flattering everyone who remained — the bomber-armor problem, in finance.

**The point-in-time data industry.** The entire existence of vendors like Compustat's point-in-time database, S&P Capital IQ's PIT fundamentals, and specialized providers exists *because* look-ahead bias is so destructive and so easy to commit. These products are expensive precisely because reconstructing "what was knowable on each historical date" — preserving every as-filed value and every restatement vintage with its own knowledge date — is laborious. A serious quant fund pays for PIT data because a single restatement leak can turn a marginal strategy into a backtest fantasy worth millions on paper and nothing in production.

**The reproducibility crisis in factor research.** A wave of academic finance papers claimed to discover hundreds of return-predicting "factors." When researchers re-examined them with proper out-of-sample discipline and multiple-testing corrections, a large fraction failed to replicate — they were the lucky survivors of data snooping across thousands of tested signals. The arithmetic from our worked example scales brutally: test enough variants and you are *guaranteed* false positives, and a literature that does not correct for the number of tests will fill up with them. This is why hiring committees test your instinct for snooping so hard.

**Alt-data hygiene at modern funds.** As funds moved into alternative data — satellite images of parking lots, credit-card transaction panels, web-scraped prices — every classic bias reappeared in new clothing. A credit-card panel that *backfills* corrected transaction data has a restatement-style look-ahead leak. A satellite dataset that only covers stores *still open today* has survivorship bias. A web-scraping signal tested across a thousand product categories until one "works" is snooping. The instruments changed; the questions a take-home asks did not.

**Crypto and the 24/7 timestamp trap.** Crypto markets never close, trade across every timezone, and report data from exchanges with inconsistent clocks. A surprising number of crypto backtests leak the future simply by joining a signal to a price stamped in the wrong timezone, or by using a "closing" price on a market that has no close. The discipline is identical to equities — store everything in UTC, only join data that genuinely precedes your decision in absolute time — but the absence of a clean daily boundary makes the leak easier to commit and harder to spot. For the broader picture of how these markets are structured, see [how 2026 transformed crypto](/blog/trading/finance/how-2026-transformed-crypto-vcs-into-architects-of-digital-finance).

## From raw feed to a number you can trust

![Three silent biases sit between a raw data feed and a backtest number you can trust; each must be removed in order.](/imgs/blogs/market-data-eda-biases-quant-research-12.png)

The pipeline above is the whole post as a single checklist. A raw vendor feed enters on the left. Two cleaning tracks run in parallel: the **survivorship track** (add the delisted names back, then lag everything to its knowledge date so the data is point-in-time) and the **adjustment track** (adjust splits and dividends correctly, then run an as-of join on every feature). Both converge on locking a clean holdout, and only *then* does a backtest produce a Sharpe you can actually trust. Skip any box and the number coming out the right side is fiction.

If you internalize one reflex from all of this, make it this: **for every value your strategy touches, ask "when did I actually know this?"** When did this price reflect a real, adjusted economic value? When was this earnings number actually filed, and has it been restated since? Was this company alive and tradeable on this date, or am I only seeing it because it survived? Did I look at this holdout before? Every one of the biases in this post — survivorship, look-ahead, restatement, snooping — is a different way of accidentally answering "later" to that question and pretending the answer was "then."

### Where this touches your work, and what to read next

If you are preparing for a quant-researcher take-home, build the muscle on real, messy data: download a few years of daily prices *including delisted names* (CRSP and some commercial vendors provide them; many free sources do not, which is itself a lesson), compute total returns off the adjusted close, and deliberately introduce a look-ahead leak so you can see what it does to a Sharpe ratio. Then remove it. Doing this once, by hand, teaches more than any amount of reading.

From here, the natural next steps deepen the statistical machinery these biases interact with: the way correlations behave (and misbehave) is in [covariance, correlation, and their pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews); the regression tools you will use to test a signal are in [linear regression for quant interviews](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews); the bias-variance lens on why over-tuned models fail out of sample is in [estimators, MLE, bias and variance](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews); and the significance-testing arithmetic behind the snooping problem is in [hypothesis testing and p-values](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews).

This is educational material, not investment advice — and the deepest reason these biases matter is not that they cost you points in an interview, but that a fund that fails to catch them deploys real capital against a signal that was never there. The data hygiene *is* the edge. Find the bodies before you find the alpha.
