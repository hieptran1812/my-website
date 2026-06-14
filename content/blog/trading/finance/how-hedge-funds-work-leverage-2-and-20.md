---
title: "How Hedge Funds Really Work: Leverage, 2-and-20, and the Bridgewater Playbook"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A plain-English guide to what a hedge fund actually is, how shorting and leverage work, why the 2-and-20 fee model shapes everything, and why most funds disappoint investors after fees."
tags: ["hedge-funds", "investing", "leverage", "short-selling", "fees", "risk-parity", "financial-institutions", "asset-management", "alpha", "bridgewater"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A hedge fund is a lightly regulated private investment pool that can do what mutual funds cannot — sell short, borrow to use leverage, and charge a slice of profits — in pursuit of returns that do not move with the stock market; the fee structure called "2 and 20" explains both the fortunes its managers make and why most funds disappoint their investors once fees are subtracted.
>
> - A hedge fund pools money from accredited and institutional investors and tries to make money whether markets go up or down, not just to beat a benchmark.
> - "2 and 20" means a 2% annual fee on all the money managed plus 20% of the profits; this wedge eats a large share of returns and is the single most important number to remember.
> - Shorting and leverage are the tools that let a fund profit from falling prices and amplify gains — but a short can lose more than 100% and leverage amplifies losses just as fast as gains.
> - Ray Dalio's Bridgewater built the biggest hedge fund on earth around "risk parity," the All Weather portfolio, and a framework for how the economy works as a machine.
> - The number to remember: from 2008 to 2017 Warren Buffett bet that a plain S&P 500 index fund would beat a basket of hedge funds — and the index won by a wide margin, returning about 125% versus roughly 36% for the funds-of-funds.

How can a money manager become a billionaire while, on average, the funds in their industry have a hard time beating a cheap index fund that a teenager could buy with one tap on a phone? That single contradiction is the whole story of the hedge fund business. The answer is not that hedge fund managers are frauds, and it is not that they are geniuses. The answer is hidden inside the structure — who is allowed to invest, what the fund is allowed to do, and above all how the manager gets paid. The diagram above is the mental model: every hedge fund strategy is a specific bet paired with a specific way it can blow up, and the manager collects a fee whether the bet wins or loses.

![Matrix of hedge fund strategies with the bet and the main risk for each](/imgs/blogs/how-hedge-funds-work-leverage-2-and-20-1.png)

By the end of this post you will understand what a hedge fund actually is, how short selling and leverage work from scratch, why the "2 and 20" fee model bends every incentive in the industry, what the major strategies are betting on, how Ray Dalio's Bridgewater became the largest hedge fund in history, and why — despite all the mystique — most hedge funds have struggled to justify their fees. We will work through real dollar arithmetic at every step, and we will name the risk beside every upside. None of this is investment advice; it is an attempt to make a famously opaque corner of finance legible.

## First principles: what a hedge fund actually is

A **hedge fund** is a private pool of money, run by a professional manager, that invests on behalf of a small number of wealthy or institutional clients and is allowed to use aggressive techniques — selling short, borrowing money, and concentrating bets — that ordinary public funds cannot. The word "fund" just means a pot of pooled money. The word "hedge" is a historical accident we will return to, and it badly misleads people, so set it aside for now.

Compare it to the fund most people already know. A **mutual fund** (and its cousin the exchange-traded fund, or ETF) is a public pool. Anyone with a few dollars can buy in. In exchange for that openness, regulators in the United States — chiefly the Securities and Exchange Commission, the SEC — impose tight rules. A mutual fund mostly has to own things outright (it is "long only"), it faces strict limits on borrowing, it must publish its holdings, and it must let investors take their money out daily. It is a safe, transparent, heavily supervised vehicle for the general public.

A hedge fund is the opposite trade-off. Because it only takes money from a narrow set of investors who are presumed sophisticated enough to fend for themselves, it escapes most of those rules. In the U.S., the typical investor in a hedge fund must be an **accredited investor** (broadly, an individual with over \$1 million in net worth excluding their home, or income above \$200,000 a year) or a **qualified purchaser** (generally over \$5 million in investments), or an institution such as a pension fund or university endowment. Because the fund is "private" and sold only to these people, it does not have to register as a public investment company, does not have to publish daily holdings, and can lock up investor money for months or years.

That regulatory freedom is the whole point. It lets the manager do four things a mutual fund largely cannot:

1. **Go short** — bet that a price will fall, and profit when it does.
2. **Use leverage** — borrow money to control a position larger than the fund's own capital.
3. **Concentrate** — put a large share of the fund in a handful of bets rather than spreading thinly across hundreds of stocks.
4. **Charge a performance fee** — take a cut of the profits, not just a flat management fee.

### Absolute return versus beating a benchmark

Here is a distinction that sits at the heart of the hedge fund pitch. Most public funds are sold on **relative return** — the goal is to beat a **benchmark**, usually a market index like the S&P 500. If the S&P 500 falls 20% and the fund falls only 18%, the manager can claim success: they beat the benchmark by two points, even though the client lost money.

A hedge fund classically promises **absolute return** instead — the goal is to make money in dollars, in any market, up or down. A good absolute-return year is a positive number, full stop. To deliver that, the fund tries to produce returns that do not move in lockstep with the stock market. The technical word for "moves with the market" is **beta**, and the word for "return earned above and beyond the market's own movement, from genuine skill" is **alpha**.

Put it this way: if you owned the whole market and it went up 8%, you earned 8% of beta with zero alpha — you just rode the tide. If a manager produced 8% in a year the market was flat, that 8% is pure alpha. Hedge funds are, in theory, alpha machines. The hard truth we will reach later is that alpha is scarce, it is fiercely competed for, and it tends not to persist.

### The supporting cast: prime broker, margin, high-water mark, drawdown

Four pieces of plumbing show up constantly, so let me define each inline before we lean on them.

A **prime broker** is the big bank (Goldman Sachs, Morgan Stanley, JPMorgan and the like) that services the hedge fund: it holds the fund's securities, lends it shares to short, lends it cash to leverage, executes trades, and tracks the collateral. The prime broker is the fund's landlord, banker, and lender all at once — and, as we will see in the Archegos story, the prime broker's risk department is sometimes the only thing standing between a fund and catastrophe.

**Margin** is borrowed money from the prime broker used to buy or short securities, with the securities themselves pledged as collateral. If you put up \$20 and borrow \$80 to buy a \$100 position, you are trading "on margin."

A **high-water mark** is the highest value an investor's stake has ever reached in the fund. It matters because the performance fee is only charged on gains above that prior peak — a rule designed so the manager cannot charge you twice for earning back money they already lost.

A **drawdown** is the drop from a peak to a trough — how far the fund fell from its highest point before recovering. A fund that goes from \$100 to \$70 has suffered a 30% drawdown. Drawdown is the number that frightens investors, because it is the experience of watching wealth evaporate.

With those defined, we can build the rest of the machine.

## The fee model: what "2 and 20" really takes

The phrase **"2 and 20"** describes the classic hedge fund fee schedule. It has two parts:

- A **management fee** of (traditionally) 2% per year, charged on the total assets under management — the whole pot — regardless of performance. This pays the manager's salaries, rent, data, and lights. The investor pays it in good years and bad.
- A **performance fee** (also called the incentive fee or "carry") of (traditionally) 20% of the profits the fund generates. This is the part that makes managers rich.

Two and twenty is the headline, though competition has pushed many funds to lower numbers — "1.5 and 15" or even less — while the most sought-after funds have charged far more. The principle, not the exact percentages, is what matters: the manager is paid a slice of your assets and a slice of your gains.

Let me make this concrete.

#### Worked example: 2-and-20 on a \$100 million fund that returns 10%

Suppose a fund manages \$100 million and earns a 10% gross return over the year — meaning the fund's investments produced a profit of \$10 million before any fees.

- **Management fee:** 2% of the \$100 million in assets = \$2 million. This is charged no matter what.
- **Performance fee:** 20% of the \$10 million profit = \$2 million.
- **Total fees:** \$2 million + \$2 million = \$4 million.

So the fund made \$10 million for its investors and kept \$4 million of it. The investors are left with \$6 million of profit on their \$100 million — a **net return of 6%**, not the 10% gross. The manager captured 40% of the year's profit through fees.

The one-sentence intuition: even in a good year, the fee wedge can quietly take nearly half of what the fund earned for you.

![Pipeline showing the 2-and-20 fee waterfall from gross profit to investor net](/imgs/blogs/how-hedge-funds-work-leverage-2-and-20-3.png)

The figure above is the fee waterfall: the \$10 million gross profit flows in at the top, the 2% management fee and the 20% performance fee are skimmed off in the middle, and the \$6 million net is what reaches the investor. Notice that the management fee is charged on the entire \$100 million, while the performance fee is charged only on the \$10 million of profit. That asymmetry becomes brutal in a flat or down year, which is where the high-water mark enters.

### The high-water mark: no double-dipping on losses

The performance fee would be a scam without a guard rail. Take a fund that loses 20% one year, then gains 20% the next. Roughly speaking, the investor is back to break-even (actually slightly below, because a 20% loss requires a 25% gain to recover). It would be outrageous for the manager to charge a 20% performance fee on that second-year "gain" when the investor has made no money over the two years combined. The **high-water mark** prevents exactly that.

The rule: the manager earns the performance fee only on gains that lift the investor's stake above its prior all-time peak. Below the high-water mark, the manager works for free on the performance side (they still collect the management fee).

#### Worked example: the high-water mark after a down year

Start with \$100 million at a high-water mark of \$100 million.

- **Year 1:** the fund loses 15%. It falls to \$85 million. No performance fee is owed — there were no profits. The manager still charges the 2% management fee, roughly \$1.7 million on the \$85 million average, so the investor's pain is compounded slightly by fees. The high-water mark stays at \$100 million.
- **Year 2:** the fund gains 10%, rising from \$85 million to \$93.5 million. Even though this is a +10% year, the fund is still below its \$100 million high-water mark. So **no performance fee is charged** on that \$8.5 million of recovery. The manager only collects the management fee.
- **Year 3:** the fund gains another 15%, rising from \$93.5 million to \$107.5 million. Now it has cleared the \$100 million high-water mark. The performance fee applies only to the \$7.5 million of gains above \$100 million — 20% of \$7.5 million = \$1.5 million — not to the full year's gain.

The one-sentence intuition: the high-water mark means a manager who loses your money must earn it all back before they can charge a performance fee again — which is exactly why managers who suffer a deep drawdown sometimes shut the fund and reopen a fresh one, resetting the mark to zero.

### Hurdle rates

Some funds add a **hurdle rate**: a minimum return the fund must clear before the performance fee kicks in. If the hurdle is 5%, the manager earns the 20% incentive fee only on returns above 5% (or, in a "hard hurdle," only on the portion above 5%; in a "soft hurdle," on everything once 5% is cleared). The logic is that an investor could have earned a risk-free return just by holding Treasury bills, so the manager should only be rewarded for beating that baseline. Hurdle rates are more common in private equity and credit funds than in classic equity hedge funds, but they belong in your vocabulary.

The deeper point of this whole section is that the fee model is not a footnote — it is the gravitational center of the industry. The 2% management fee guarantees the manager gets paid even if they lose money. The 20% performance fee gives the manager a call option on your capital: they share in the upside but not the downside. That asymmetry shapes how much risk managers are willing to take, which funds survive, and why — as we will see — the math is stacked against the investor over the long run.

## Going short: profiting when prices fall

To understand a hedge fund you must understand **short selling**, because shorting is the technique that separates a hedge fund from an ordinary investor. Most people only know how to make money when prices rise: buy low, sell high. That is **going long**. Shorting is the mirror image — it is how you make money when a price falls.

Here is the mechanism, built from scratch. To short a stock you:

1. **Borrow** the shares from someone who owns them, arranged through your prime broker. You pay a small borrowing fee for this loan.
2. **Sell** the borrowed shares immediately at today's price, putting cash in your pocket.
3. **Wait** for the price to fall (you hope).
4. **Buy back** the same number of shares later at the lower price — this is called "covering."
5. **Return** the shares to the lender, and keep the difference between what you sold for and what you paid to buy back.

You sold high and bought low, but in that order. You profited from the decline.

#### Worked example: a short sale on a \$50 stock

Suppose you think a stock trading at \$50 is overpriced and will fall.

- You borrow 100 shares and sell them at \$50 each, receiving **\$5,000** in cash.
- Unfortunately you are wrong. The stock rises to \$80.
- To return the borrowed shares you must now buy 100 shares at \$80, costing **\$8,000**.
- You received \$5,000 and paid \$8,000, so you lost **\$3,000**.

Now compare the worst cases. If you had gone *long* — bought 100 shares at \$50 for \$5,000 — the absolute worst that could happen is the company goes bankrupt and the stock falls to \$0. You would lose your \$5,000 and not a penny more. A long position has a floor: you can lose 100% of what you put in, but no more.

A short has no such floor. A stock can rise to \$80, \$120, \$500 — there is no upper limit on a price. So your loss on a short is theoretically **unlimited**. In our example the stock only had to rise 60%, to \$80, to lose you \$3,000 — more than half of what a total wipeout on the long would have cost, and the price could keep climbing.

The one-sentence intuition: a long position can fall to zero and stop, but a short position bleeds more and more as the price rises without limit — which is why shorting is the most dangerous tool in the kit.

![Before and after comparison of a long position payoff versus a short position payoff](/imgs/blogs/how-hedge-funds-work-leverage-2-and-20-2.png)

The figure above lays the two side by side. The long column (buy at \$50) caps the loss at \$5,000 — the price can only fall to zero. The short column (sell borrowed shares at \$50) caps the *gain* at \$5,000 — the most you can make is if the stock goes to zero — but leaves the loss open-ended as the price rises. This asymmetry is the single most important risk fact about shorting, and it is why a crowded short trade going wrong can become a **short squeeze**: as the price rises, shorts rush to buy back and cut their losses, and all that buying pushes the price even higher, which forces yet more shorts to cover. We will see that exact dynamic crush a hedge fund in the GameStop story.

A hedge fund that holds both longs and shorts is genuinely "hedged" in one sense: if the whole market falls, the longs lose but the shorts gain, partially offsetting each other. That is the original meaning of "hedge," and it is why a hedge fund's market exposure (its net beta) can be much lower than a mutual fund's. But — and this matters — being hedged does not mean being safe. A fund can lose badly if its specific longs fall *and* its specific shorts rise at the same time, which is precisely what happens when a strategy gets crowded and unwinds.

## Leverage: borrowing to amplify both ends

The second superpower is **leverage** — using borrowed money to control a position larger than your own capital. Leverage does not create returns out of thin air; it multiplies whatever return you were going to get, in both directions.

The mechanics run through the prime broker. The fund posts its own cash (its **equity**) as collateral, and the prime broker lends it additional cash (margin) to buy more. A fund with \$20 million of its own money might borrow \$80 million from the prime broker to control a \$100 million position — that is **5:1 leverage** (\$100 million of assets for every \$20 million of equity). Some strategies, especially those trading tiny, "safe-looking" spreads, have historically run at 20:1, 30:1, or higher. Leverage is measured as the ratio of total position size to equity.

![Graph of leverage through the prime broker with a margin-call branch](/imgs/blogs/how-hedge-funds-work-leverage-2-and-20-4.png)

The diagram above shows the structure: the fund's \$20 million equity plus \$80 million of borrowed margin from the prime broker fund a \$100 million position. If the position rises, the gains accrue on the full \$100 million but are measured against the \$20 million of equity, so the percentage return on equity is amplified. If the position falls far enough, the broker issues a **margin call** — a demand for more collateral — and can force the fund to sell at the worst possible moment.

#### Worked example: 5:1 leverage turns a 4% move into ~20%

A fund has \$20 million of equity and uses 5:1 leverage to control a \$100 million position (borrowing \$80 million).

- **If the position gains 4%:** the \$100 million position rises by \$4 million. Subtract the interest on the \$80 million loan — say the borrowing cost is roughly \$2 million a year at a 2.5% rate, but to keep the arithmetic clean assume the gain is measured before financing for a short horizon — the fund's equity rises from \$20 million to about \$24 million. A 4% gain on assets became roughly a **20% gain on equity**. Leverage multiplied the return fivefold.
- **If the position loses 4%:** the \$100 million position falls by \$4 million. The fund's equity drops from \$20 million to \$16 million — a **20% loss on equity** from a mere 4% market move. The same multiplier works in reverse.

The one-sentence intuition: leverage is a magnifying glass held over your returns — it makes a small gain look brilliant and a small loss look catastrophic, and the market does not care which one shows up.

Leverage interacts viciously with shorting and with margin. As losses mount, the fund's equity shrinks, which raises its effective leverage (the same loan now sits on a smaller equity base), which makes the next loss hurt even more. At some point the prime broker's risk system demands the fund top up its collateral.

#### Worked example: a margin call when equity falls below maintenance

Continue with the \$100 million position funded by \$20 million of equity and \$80 million of borrowed margin. Say the prime broker requires a **maintenance margin** of 15% — the fund's equity must stay at least 15% of the position's current value, or \$15 million on a \$100 million position.

- The position falls 8%, from \$100 million to \$92 million. The \$80 million debt is unchanged, so the fund's equity is now \$92 million − \$80 million = **\$12 million**.
- Required maintenance is 15% of \$92 million = about \$13.8 million. The fund has only \$12 million. It is below maintenance.
- The prime broker issues a **margin call**: deposit more cash immediately, or we sell your positions to bring the account back into line.
- If the fund cannot find the cash, the broker liquidates positions — often dumping them into a falling market, locking in the loss and sometimes pushing the price down further, which can trigger margin calls at *other* leveraged funds holding the same assets.

The one-sentence intuition: leverage does not just amplify your losses, it can take the timing of the exit out of your hands entirely — the margin call forces you to sell at the bottom.

This forced-selling cascade is the mechanism behind nearly every famous hedge fund blow-up. Hold that thought; it returns in the LTCM, quant quake, and Archegos stories.

It is worth pausing on *why* a fund leverages at all, because the temptation is so strong. Recall the fee model. The manager earns 20% of profits but bears none of the losses. Leverage multiplies the profits the manager shares in, while the downside falls entirely on the investor's capital. From the manager's seat, modest leverage on a sound strategy is rational, and excessive leverage on a risky one can still be rational *for the manager* — heads they get rich, tails the investor loses and the manager simply closes the fund. This is the dark side of the performance-fee asymmetry: it quietly rewards taking more risk with someone else's money. The high-water mark and the manager's own capital invested alongside clients ("skin in the game") are the partial brakes on this incentive, which is exactly why sophisticated investors insist on both.

## The major strategies: what hedge funds actually bet on

"Hedge fund" is a legal and fee structure, not a single strategy. Under that umbrella sit wildly different approaches. Here are the major families, each a distinct bet with a distinct way to fail — the matrix from the top of this post.

**Long/short equity** is the oldest and most common style. The manager goes long the stocks they think are underpriced and short the stocks they think are overpriced. If the analysis is good, the longs outperform the shorts whether the overall market rises or falls. The bet is on relative value between companies. The main risk is that in a panic, correlations spike — good and bad stocks fall together, the shorts get squeezed, and the longs sink anyway.

**Global macro** trades the big picture: interest rates, currencies, government bonds, commodities, and broad equity indexes across countries, based on a view about economies and policy. George Soros's famous 1992 bet against the British pound was a macro trade. The bet is on the direction of entire markets or currencies. The main risk is that macro calls are notoriously hard to time, and macro funds often use heavy leverage, so being right eventually but wrong for now can be fatal.

**Quantitative and statistical arbitrage** uses computer models to find tiny, fleeting pricing relationships — for instance, two historically linked stocks that have temporarily drifted apart — and trades thousands of them at once, expecting the gaps to close on average. Individually each edge is minuscule; the strategy relies on doing it across a huge number of positions, often with significant leverage to make the small spreads worthwhile. The bet is that statistical regularities persist. The main risk is **crowding**: when many quant funds run similar models and one is forced to sell, they all sell the same names at once, and the "uncorrelated" strategy suddenly correlates perfectly with disaster — exactly the 2007 quant quake.

**Event-driven and merger arbitrage** bets on corporate events. In **merger arb**, when company A announces it will buy company B for, say, \$50 a share, B's stock usually jumps to just below \$50 — say \$48 — reflecting the chance the deal falls through. The arbitrageur buys B at \$48, expecting to collect the full \$50 when the deal closes, pocketing the \$2 spread. The bet is that announced deals close. The main risk is **deal breaks**: if regulators block the merger or the buyer walks away, B's price collapses back toward where it started and the spread blows out into a loss.

**Distressed debt** buys the bonds or loans of companies in or near bankruptcy, often at deep discounts — paying 40 cents for a dollar of face value — betting the eventual recovery (through restructuring or liquidation) will be worth more, say 70 cents. The bet is that the market is too pessimistic about the wreckage. The main risk is that bankruptcies drag on for years, recoveries come in lower than hoped, and the investor's capital is locked up illiquid the whole time.

These are not exhaustive. Three more deserve a mention because they dominate today's flows.

**Activist funds** buy a large stake in a public company and then agitate for change — board seats, a breakup, a buyback, a sale — to force the share price up. The bet is that the manager can improve a business the rest of the market has given up on; the risk is that management digs in, the campaign drags through proxy fights and lawsuits, and the stake stays underwater for years while capital is tied up.

**Volatility and tail-risk funds** trade options to bet on how *much* prices will move, not which direction. Some sell insurance against crashes and collect steady premiums in calm times — a strategy that looks brilliant for years and then loses catastrophically in a single panic. Others, the "tail-risk" funds, do the reverse: they pay small premiums year after year, bleeding slowly, in exchange for a giant payoff when markets crash. The bet and the failure mode are mirror images of each other, which is exactly why they are sold as offsets.

**Multi-strategy "pod shops"** — Citadel, Millennium, Point72 and their peers — are the dominant force in the modern industry. Instead of one big bet, the firm runs dozens or hundreds of small, tightly risk-managed teams ("pods"), each given capital and a strict stop-loss: a pod that loses a set percentage is cut. The firm allocates leverage across pods and aims for a smooth, market-neutral return stream. The bet is on process and diversification across many small edges; the risk is that the whole machine runs on heavy leverage and shared financing, so a shock that hits many pods at once — the way August 2007 hit the quants — can force deleveraging across the entire firm.

Across every one of these, the core idea repeats: a strategy is a bet, and every bet has a built-in failure mode. The label on the fund tells you the bet. The fine print tells you how it loses.

## The Bridgewater playbook: Dalio, risk parity, and All Weather

No discussion of hedge funds is complete without **Bridgewater Associates**, the firm Ray Dalio founded in 1975 — literally out of his apartment — and grew into the largest hedge fund in the world, managing well over \$100 billion at its peak. Bridgewater is worth studying not because it is typical (it is not) but because its ideas became influential across the whole industry.

Dalio's central insight is captured by **risk parity**, the engine behind Bridgewater's **All Weather** portfolio. A traditional portfolio — say the classic 60% stocks, 40% bonds — *looks* diversified, but it is not. Because stocks are so much more volatile than bonds, that 60/40 mix gets something like 90% of its actual ups and downs from the stock portion. You think you are balanced; you are really making a concentrated bet on stocks.

Risk parity flips the question. Instead of balancing the *dollars* across assets, it balances the *risk* each asset contributes. It asks: which assets do well in each kind of economic "weather"? Stocks like rising growth. Long-term government bonds like falling growth (and falling rates). Commodities and inflation-linked bonds like rising inflation. Nominal bonds like falling inflation. By holding a slice tuned to each environment — and using modest leverage on the lower-risk pieces to bring their risk contribution up to par with stocks — the portfolio aims to perform reasonably no matter which way the economy turns. Hence "All Weather."

![Stack of the All Weather risk buckets for different economic environments](/imgs/blogs/how-hedge-funds-work-leverage-2-and-20-5.png)

The figure above shows the four economic-weather buckets that the All Weather idea balances: a growth-up bucket built around stocks, a growth-down bucket leaning on long Treasuries, an inflation-up bucket of commodities and inflation-linked bonds, and an inflation-down bucket of nominal bonds. The key move is that the buckets are sized so each contributes a similar share of the portfolio's total risk, rather than letting stocks dominate. Risk parity is not magic — it depends on leverage being cheap and on bonds and stocks staying negatively correlated, an assumption that broke badly in 2022 when both fell together — but it reframed how a generation of allocators thought about diversification.

Alongside All Weather, Bridgewater runs **Pure Alpha**, an actively managed global macro fund that makes discretionary and systematic bets across dozens of markets, trying to generate alpha uncorrelated with everything else. Underpinning both is Dalio's **economic-machine framework** — his attempt to model the economy as a small set of mechanical relationships (productivity growth, a short-term debt cycle, and a long-term debt cycle) that repeat across history. Whether or not one buys the framework, it produced a discipline: write down the rules, test them against history, and let the system, not the gut, make the call.

Bridgewater is also famous — or notorious — for its culture of **radical transparency**: meetings are recorded, employees rate each other in real time on tablets, and disagreements are aired openly rather than smoothed over. Dalio's argument is that an "idea meritocracy" needs brutal honesty to function; critics call it a stressful environment that not everyone survives. For our purposes the cultural point is secondary. The investing lesson is the one to keep: the most durable hedge fund did not chase hot stocks; it built a framework for balancing risk across all economic environments and held it with discipline.

## Why most hedge funds underperform net of fees

Now the uncomfortable part. The hedge fund industry sells the promise of alpha — superior, market-beating, uncorrelated returns. As a group, over the past two decades, it has largely failed to deliver that promise *after fees*. There are four reasons, and they compound.

**First, fee drag.** Return to the 2-and-20 arithmetic. We saw that a 10% gross year became a 6% net year — the fees took 40% of the profit. Now compound that over many years. A fund that earns 8% gross every year for a decade, charging 2 and 20, delivers far less than 8% to the investor; the fees act like a heavy tax that erodes the compounding engine year after year. To merely match a cheap index fund net of fees, the hedge fund manager has to *substantially* beat that index gross — a high bar that most clear only briefly, if at all.

![Matrix of gross fund return versus investor net return across scenarios](/imgs/blogs/how-hedge-funds-work-leverage-2-and-20-7.png)

The figure above shows the wedge across scenarios: a +20% gross year becomes +14% net, a +10% year becomes +6%, a flat year becomes −2% (the management fee is still charged), and a −15% year becomes −17%. In every row the investor keeps less than the fund earned, and in down years the management fee makes the loss worse. The fee structure is not symmetric — the manager shares your upside but not your downside.

**Second, crowding.** When a strategy works, money floods in and other managers copy it. The edge that one clever fund found gets competed away as everyone piles into the same trades. The famous spreads shrink, the easy alpha disappears, and what is left is a crowded position vulnerable to a violent unwind — the recurring theme of the blow-up stories below.

**Third, the difficulty of persistent alpha.** Markets are close to efficient, which means consistent outperformance requires a genuine, repeatable edge — and edges decay. The manager who was brilliant for five years often reverts to the average, because their advantage was partly luck, partly a market regime that ended, partly an inefficiency that got arbitraged away. Studies of hedge fund returns repeatedly find that past performance is a weak predictor of future performance, and that the average fund's alpha, after fees, hovers near zero.

**Fourth, survivorship bias inflates the reported numbers.** Industry indexes tend to drop funds that blow up and stop reporting, so the published average looks better than the experience of the average dollar invested. The fund that lost 60% and shut down quietly vanishes from the statistics; the fund that got lucky keeps reporting. The real, all-in record is worse than the brochure.

#### Worked example: the Buffett bet, 2008–2017

The most public demonstration of all this is a wager Warren Buffett made. In 2007, Buffett bet \$1 million (to charity) that over the ten years from 2008 through 2017, a simple, low-cost S&P 500 index fund would outperform a hand-picked basket of hedge funds, chosen by a firm that runs **funds of funds** — a fund that invests in other hedge funds, adding a second layer of fees on top of the underlying funds' 2-and-20.

- The asset manager Protégé Partners took the bet, selecting five funds of funds.
- Over the decade, the **S&P 500 index fund returned roughly 125%** in total — an average of about 8.5% a year.
- The five funds of funds returned, on average, only about **36%** in total — roughly 3.2% a year.
- Buffett won decisively. The index nearly tripled the hedge funds' result.

Why such a gap? The funds of funds were carrying two layers of fees — the underlying 2-and-20 plus the fund-of-funds' own fees — against a market that, over that particular decade, rose steadily. The gross returns of the underlying funds may have been respectable; the *net* returns, after two coats of fees, were not.

The one-sentence intuition: over a long horizon, the simplest cheapest market exposure beat an expensively assembled basket of professional managers — because the fees were a certainty and the alpha was not.

None of this means *no* hedge fund is worth it. A handful — Renaissance Technologies' Medallion fund is the legend — have produced extraordinary, durable returns, which is why Medallion stopped taking outside money and charges fees that would be absurd anywhere else. But these are rare exceptions, usually closed to new investors, and survivorship bias makes them loom larger in the imagination than they should. For the typical investor in the typical fund, the math above is the base rate.

## Who invests in hedge funds, and why

If the average fund disappoints after fees, why does roughly \$4 trillion sit in hedge funds worldwide? Because the buyers are not retail investors chasing returns — they are large institutions with specific needs, and they are not all chasing the same thing.

**Pension funds** — the pools that pay retirees — invest in hedge funds partly for diversification: a return stream that does not move with their large stock and bond holdings can smooth the overall portfolio, even if its standalone return is mediocre. A pension that must earn, say, 7% a year to meet its promises and fears a stock crash may pay for an absolute-return strategy as insurance.

**University endowments** — Yale's and Harvard's are the famous ones — pioneered heavy allocations to hedge funds and other "alternatives." The Yale model, built by the late David Swensen, leaned on illiquid and uncorrelated assets precisely because an endowment has a near-infinite time horizon and can tolerate lock-ups in exchange for diversification and access to top managers.

**Sovereign wealth funds** — state-owned investment pools like Norway's or Singapore's GIC — deploy national savings across the globe and use hedge funds for the same diversification and access reasons, at enormous scale.

**Wealthy individuals and family offices** — the original hedge fund clients — invest for return, diversification, status, and access to managers their financial advisers cannot otherwise reach.

The common thread: these investors are large, sophisticated, patient, and seeking something other than pure return — usually diversification, downside protection, or access to scarce talent. The pitch that survives the fee critique is not "we will beat the market" but "we will give you a return that zigs when your other assets zag." Whether the funds reliably deliver even that is hotly debated, especially after 2022, when many "diversifiers" fell alongside everything else.

There is also a less flattering reason large institutions buy hedge funds: **career risk**. A pension officer or endowment committee that allocates to a famous fund and loses money has cover — everyone did the same thing, the consultant recommended it, the manager is a household name. The same officer who skips hedge funds entirely, puts the money in a cheap index fund, and underperforms in a year when alternatives shine looks like they were asleep at the wheel. The institutional incentive to *look* prudent can diverge from the incentive to *be* low-cost, and a chunk of the industry's \$4 trillion is explained by that gap as much as by genuine expected alpha.

#### Worked example: the diversification math an institution actually buys

Suppose a pension holds \$1 billion, split \$700 million in stocks and \$300 million in bonds. In a crash, stocks fall 30%, so the stock sleeve loses \$210 million and the whole fund drops about 18% even after the bonds hold steady. Now the pension moves \$200 million into a hedge fund that, by design, returns roughly 0% in that same crash (its longs and shorts offset). In the crash the stock sleeve is now \$500 million and loses \$150 million; the hedge fund loses nothing; the bonds hold. The fund falls about 15% instead of 18%. The pension paid 2-and-20 not for a higher return but to shave three points off the worst-case drawdown — and whether three points of crash protection is worth roughly \$4 million a year in management fees on \$200 million is the entire debate in miniature. The one-sentence intuition: institutions are often buying insurance, not alpha, and like all insurance it is worth it only if the disaster it covers actually arrives.

## Common misconceptions

**"Hedge funds always beat the market."** They do not, and as a group they have not for most of the past two decades after fees. The Buffett bet is the cleanest public proof. A few stars produce extraordinary returns, but the average fund, net of 2-and-20, has struggled to keep up with a cheap index fund — and the funds you can actually invest in are rarely the stars.

**"Hedge funds are reckless gamblers."** Some take wild, concentrated, leveraged bets, and those make headlines when they blow up. But the word "hedge" points the other way: many funds are explicitly built to *reduce* market exposure by pairing longs with shorts, and the largest, like Bridgewater, are obsessive about balancing risk. The industry contains both the cautious and the cowboys; the blow-ups are memorable precisely because they are not the norm.

**"'Hedge' means low-risk."** This is the most damaging misreading of the name. "Hedge" originally referred to A.W. Jones pairing long and short positions to neutralize broad market moves — a specific technique, not a promise of safety. Plenty of hedge funds run enormous risk through leverage and concentration. The name describes a technique that *some* funds use; it is not a risk rating.

**"Short selling is un-American or just destructive."** Politicians periodically attack shorting as a vandalism of markets, and exchanges sometimes ban it in a panic. But short sellers add information: they are the investors with an incentive to dig into frauds and overvaluations, and several major accounting scandals (Enron among them) were flagged early by shorts. Shorting also provides liquidity and helps prices reflect bad news, not just good. It can be abused, but it is a normal, useful part of a functioning market.

**"You can invest in any hedge fund you want."** You generally cannot. U.S. rules restrict hedge fund investment to accredited investors and qualified purchasers, and the best funds are frequently *closed* — they refuse new money because more assets would dilute their edge. Medallion has not taken outside capital in decades. The funds available to a new investor are, by selection, rarely the ones you would most want.

**"The 2% fee is the small part — it's the 20% that costs you."** In good years the performance fee dominates, but over a full cycle the seemingly modest 2% management fee, charged every single year on the entire pot regardless of results, is a relentless drag that compounds against the investor. In flat and down years it is the *only* fee — and it still gets charged while you lose money.

## How it shows up in real markets

The mechanisms in this post — shorting, leverage, margin calls, crowding — are not abstractions. They have repeatedly moved markets and destroyed funds. Here are the episodes worth knowing.

![Timeline of hedge fund history from A.W. Jones in 1949 to the 2021 meme squeeze](/imgs/blogs/how-hedge-funds-work-leverage-2-and-20-6.png)

The timeline above traces the arc: A.W. Jones invents the hedged fund in 1949, Long-Term Capital Management blows up on leverage in 1998, the quant quake and Paulson's subprime short hit in 2007, the 2008 crisis prompts the Buffett bet, and the 2021 meme-stock squeeze maims a prominent short seller. Each milestone is one of the stories below in compressed form.

### A.W. Jones invents the hedge fund, 1949

The first recognizable hedge fund was launched in 1949 by Alfred Winslow Jones, a sociologist and journalist, not a financier. His innovation was to combine three ideas that define the structure to this day: he held *both* long and short positions to neutralize broad market swings (the original "hedge"), he used *leverage* to amplify his stock-picking, and he took a 20% cut of profits as his fee. Jones structured the fund as a private partnership to escape regulation, the same legal trick used today. For years his approach was a secret; a 1966 magazine profile revealing how far he had outperformed mutual funds set off the first hedge fund boom. The lesson is foundational: every element of the modern hedge fund — long/short, leverage, performance fee, private structure — was present at the creation, 75 years ago.

### Long-Term Capital Management, 1998

LTCM is the cautionary tale every risk manager memorizes. Founded by a star bond trader and including two Nobel laureates, the fund ran sophisticated arbitrage trades on tiny price discrepancies in bonds worldwide. Because the spreads were so small, LTCM piled on enormous leverage to make them profitable — at its peak it controlled positions of well over \$100 billion on a few billion of equity, with leverage ratios reported around 25:1 and far higher counting derivatives. The models said the positions were nearly riskless because they were diversified across many uncorrelated bets. Then in 1998 Russia defaulted on its debt, panic spread, and all those "uncorrelated" trades moved against LTCM at once — exactly the crowding-plus-leverage trap. As losses mounted, the fund's shrinking equity sent its effective leverage spiraling and margin calls loomed. The Federal Reserve organized a \$3.6 billion rescue by a consortium of banks to prevent a forced liquidation that could have cascaded through the whole financial system. The lesson: leverage turns a survivable loss into an existential one, and "diversified" bets stop being diversified in a crisis. (This one deserves its own full post — the mechanics of how a Nobel-decorated fund destroyed itself are a masterclass in tail risk.)

### The quant quake, August 2007

For a few days in August 2007, many of the most successful quantitative long/short equity funds suffered sudden, severe, simultaneous losses — despite the broad market being relatively calm. The cause was crowding. So many quant funds were running similar statistical-arbitrage models that they held nearly identical positions. When one large fund, likely facing losses elsewhere in the subprime mess, began rapidly liquidating, it pushed down exactly the stocks the other quants were long and pushed up the ones they were short. That moved every similar fund's portfolio against it, triggering more forced selling in a feedback loop. Strategies that had looked uncorrelated for years turned out to be the *same* strategy, and they all unwound together over about a week before partially recovering. The lesson: a strategy is only as uncorrelated as the number of people *not* running it, and leverage turns a crowded exit into a stampede.

### John Paulson's subprime short, 2007

While the quants were getting crushed, John Paulson was making one of the most lucrative trades in history. Convinced that the U.S. housing market was a bubble built on bad mortgages, Paulson's fund bought **credit default swaps** — contracts that pay out if the underlying mortgage bonds default — effectively a giant, cheap short on subprime housing. When the mortgage market collapsed in 2007–2008, those swaps exploded in value. Paulson's funds reportedly made roughly \$15 billion in 2007, with Paulson personally earning several billion. This is the upside the whole industry advertises: an out-of-consensus view, expressed through a structure (shorting via derivatives) that ordinary investors could not access, paying off enormously. The lesson — and the asterisk — is that this kind of home run is spectacularly rare, was a bet against a once-in-a-generation bubble, and that Paulson's *later* funds did not repeat it, a vivid illustration of how poorly alpha persists.

### Archegos Capital Management, 2021

Archegos was technically a **family office** (a private vehicle managing one wealthy family's money, even less regulated than a hedge fund), but its blow-up is a pure illustration of hidden leverage and the prime broker's role. Run by Bill Hwang, Archegos took massive, concentrated, leveraged bets on a handful of stocks — using **total return swaps**, derivatives arranged through multiple prime brokers, so that no single bank could see the *total* size of its position across all of them. Effectively Archegos controlled tens of billions of dollars of stock on a far smaller equity base, with the leverage scattered and disguised. When a few of its concentrated holdings dropped in March 2021, the margin calls came, Archegos could not meet them, and the prime brokers raced to dump the stock. Credit Suisse alone lost about \$5.5 billion; other banks lost billions more. The lesson: leverage hidden across multiple counterparties is leverage no one is managing, and a single family office's collapse can punch holes in the balance sheets of the world's largest banks. (Archegos, too, is worth a full post — the failure of prime-broker risk management is a story in itself.)

### GameStop and Melvin Capital, 2021

In January 2021, a swarm of retail investors organized online noticed that the stock GameStop — a struggling video-game retailer — was heavily shorted by hedge funds, including Melvin Capital. They began buying the stock and its call options en masse. As the price rose, the shorts faced exactly the asymmetric, unlimited-loss trap from our shorting section: to cap their losses they had to buy back shares, but that buying drove the price higher, forcing more covering — a textbook **short squeeze**. GameStop rocketed from under \$20 to an intraday peak around \$483. Melvin Capital, which had a large short position, lost roughly 50% of its capital in a single month and required a multibillion-dollar cash injection from other funds to survive; it ultimately closed in 2022. The lesson: the unlimited-loss geometry of shorting is not theoretical, and in a connected world a crowd can deliberately weaponize it against a leveraged, concentrated short.

### The 2008 crisis and the case for and against the industry

The broad 2008 financial crisis is the backdrop against which the Buffett bet was struck. Many hedge funds lost heavily in 2008; the average fund fell, though by less than the stock market, which is the diversification case the industry makes. But many also "gated" — suspended investor withdrawals — trapping clients' money exactly when they needed it, which is the liquidity risk of giving up the daily redemption that mutual funds offer. Coming out of the crisis, with the index fund then compounding for a decade in a long bull market, the funds' net-of-fee underperformance became the central critique of the industry. The lesson is balanced: hedge funds did cushion the 2008 drop somewhat, validating the diversification pitch, but the decade that followed exposed how expensive that cushion was relative to simply riding the recovery in cheap index funds.

## When this matters to you, and where to read next

For most individual investors, the single most useful takeaway is also the most deflating: you almost certainly cannot, and probably should not, try to invest in hedge funds. The rules largely exclude you, the best funds are closed, and the math of 2-and-20 means the average accessible fund is fighting a strong headwind just to match a near-free index fund. Understanding hedge funds matters less because you will buy one and more because they shape the markets you *do* invest in — they are often the marginal buyer or short-seller setting prices, the force behind sudden squeezes and unwinds, and the entities whose forced liquidations turn a wobble into a crash.

If you ever do encounter a hedge fund pitch — through a wealth manager, an inheritance, or a job — the questions this post equips you to ask are the right ones. What is the gross return *and* the net return after all fees? What is the leverage, and who is the prime broker? What is the worst historical drawdown? Is there a high-water mark and a hurdle? How correlated is this return to my existing stocks and bonds — am I paying for diversification, or paying 2-and-20 for hidden beta I could get for free? And what, specifically, is the bet, and how does it lose? Every strategy has a built-in failure mode; a manager who cannot describe theirs clearly is one to avoid.

The deeper habit worth carrying away is to look past the label to the structure. "Hedge fund" tells you about fees and freedom, not about safety or skill. The fee model, the leverage, the short book, and the crowding risk are what actually determine outcomes — and they are knowable, if you ask.

To place hedge funds in the wider financial system, the companion posts in this series are the natural next stops. Start with the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) for the map of who is who, then read [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) to understand the prime brokers that lend hedge funds their leverage. The piece on the [Big Three asset managers — BlackRock, Vanguard, and State Street](/blog/trading/finance/big-three-blackrock-vanguard-state-street) shows the index-fund giants on the other side of the active-versus-passive divide that the Buffett bet dramatized. And because leverage and margin calls live and die by the cost of borrowing, [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) explains the rate that sits underneath every leveraged trade in this post.

The hedge fund is one of the most glamorized and least understood institutions in finance. Strip away the mystique and you are left with three tools — shorting, leverage, and a fee structure — and one hard truth: those tools can build a fortune for the manager far more reliably than they build one for the investor.
