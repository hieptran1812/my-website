---
title: "Terminal Value: The Part That Dominates the Answer (and the Traps)"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "In most discounted cash flow models, the terminal value — the lump that stands in for every cash flow beyond the explicit forecast — is 60–80% of the entire valuation, which means a DCF is mostly a bet on one far-future assumption. This post builds terminal value from zero, both methods, all the traps."
tags: ["equity-research", "corporate-finance", "terminal-value", "dcf", "valuation", "gordon-growth", "perpetuity", "exit-multiple", "roic", "wacc", "intrinsic-value"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — In most discounted cash flow models, **60–80% of the value sits in a single number called the terminal value** — the lump that represents every cash flow after the explicit forecast ends. That means a DCF is mostly a bet on one far-future assumption, and small changes to it swing the entire answer. Master terminal value and you master where DCFs actually live or die.
>
> - **You need a terminal value because you cannot forecast forever.** After five or ten explicit years, you stop projecting line by line and collapse everything that comes after into one number sitting at the horizon. For our company, Northwind, that one number turns out to be **86% of the whole valuation** — so it deserves 86% of the scrutiny, which it almost never gets.
> - **There are two methods, and they are two different bets.** *Gordon growth* (perpetuity growth) assumes cash flows grow forever at a rate `g`: `TV = FCF_(next year) / (WACC − g)`. *Exit multiple* assumes a buyer pays a market multiple of terminal earnings: `TV = terminal EBITDA × a multiple`. The first is an intrinsic bet; the second is a relative-value bet. The discipline is to compute both and make them agree.
> - **The growth rate `g` must be tiny — no higher than the long-run economy.** A company cannot grow faster than the economy forever, or it would eventually *become* the economy. So `g` is capped at roughly long-run nominal GDP (about 2–3%). And because the denominator is the small gap `(WACC − g)`, the terminal value is violently sensitive to `g`: nudging it from 2% to 4% can swing the value by a third.
> - **Growth in the terminal value only creates value if returns beat the cost of capital.** Terminal reinvestment is `g / ROIC`; if terminal ROIC exactly equals WACC, growth adds *nothing* to the terminal value — the famous "convergence" result. Growth is only worth paying for when ROIC exceeds WACC, which ties directly back to the [ROIC–WACC spread](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value).
> - This is the lump that sits at the end of the [explicit DCF forecast](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) and gets discounted at the [WACC](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) using the [discounting](/blog/trading/equity-research/time-value-of-money-discounting-for-investors) machinery. Get the terminal value wrong and a beautiful forecast and a perfect discount rate just deliver a precisely wrong answer.

In the [first DCF post](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) we built a five-year forecast of Northwind Industries' free cash flow, and in the [second](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) we built the discount rate to turn those future dollars into present value. If you stopped there, you would have a DCF that valued *five years* of a company that will, with luck, throw off cash for fifty. That is the problem this post solves. A business does not stop existing at the end of your forecast horizon; it keeps generating cash, year after year, long after you have stopped projecting line by line. **The terminal value is the single number that captures all of that — every cash flow from the horizon to the end of time, collapsed into one lump sitting at the edge of the forecast.**

Here is the fact that should reorganize how you think about valuation: in a typical DCF, that one lump is *most of the answer.* Not a footnote, not a rounding adjustment — the majority of the value. For Northwind, as we will compute, the terminal value is about **86%** of the total. The five years we labored over in the first two posts, the years where all the judgment and detail went, contribute barely a seventh of the valuation. The rest is the terminal value, and the terminal value rests on a tiny handful of assumptions about a far-future steady state that no one can actually forecast.

This is the central, slightly uncomfortable truth of discounted cash flow valuation: **a DCF is mostly a bet on the terminal value, and the terminal value is mostly a bet on one growth rate.** People spend hours arguing about whether revenue grows 9% or 11% in year three — a decision that moves the valuation by a percent or two — and then wave through a terminal growth rate that, moved by a single point, swings the whole answer by a quarter. The figure below shows the imbalance for Northwind, and it is the orientation for everything that follows: the thin blue slice of explicit-forecast value, and the towering yellow column of terminal value that dwarfs it.

![A stacked bar chart of one discounted cash flow showing a thin lower segment for the present value of the five explicit forecast years at about two hundred sixty-five million dollars and a much taller upper segment for the present value of the terminal value at about sixteen hundred million dollars, together making an enterprise value near eighteen hundred seventy million, with a side note warning that eighty-six percent of the answer rests on one far-future assumption](/imgs/blogs/terminal-value-the-part-that-dominates-1.png)

The thesis of this post is one sentence: **because the terminal value dominates the valuation and depends on a single far-future assumption, mastering it — both how to compute it and how to keep it honest — is the highest-leverage skill in intrinsic valuation.** We will build it from zero: why we need it, the two ways to compute it, why the growth rate must be small, how growth and returns interact inside it, how to discount it back correctly, and the classic traps that turn a terminal value into a fantasy. Throughout, we continue with Northwind, so the numbers connect to the forecast we already built.

## Foundations: the building blocks of a terminal value

Before we compute anything, let's pin down the vocabulary. Most of these terms appeared in the earlier DCF posts, but terminal value uses them in a specific way, so let's be precise. Each definition is the minimum you need to follow the build.

**Discounted cash flow (DCF).** A method that values a business as the present value of all the cash it will ever generate for its investors. You forecast the cash flows, discount them to today, and sum them. The full machine has two cash-flow parts: the *explicit forecast* (the years you project in detail) and the *terminal value* (everything after). This post is about the second part.

**Explicit forecast horizon.** The number of years — usually 5 to 10 — that you forecast *explicitly*, line by line. The horizon should run until the business reaches **steady state**, a condition where growth, margins, and returns have settled into rates the company can hold forever. For Northwind we used five years, by which point growth had faded to a sustainable 4% and margins had stabilized at 15%.

**Terminal value (TV).** The value, *as of the horizon year*, of every cash flow that comes *after* the horizon. It exists because you cannot forecast year-by-year to infinity, and you don't need to once the business is in steady state — at that point all the future years follow a simple pattern (a constant growth rate, or a stable level of earnings), and a simple pattern can be summed with a formula. Terminal value sits at the *end* of the explicit forecast — for a five-year forecast, it sits at year 5 — and then gets discounted back to today like any other future amount.

**Perpetuity.** A stream of cash flows that continues *forever.* The remarkable fact, which makes terminal value possible, is that an infinite stream of cash flows can have a *finite* present value, because each future dollar is discounted more heavily than the last, and the discounted amounts shrink fast enough to sum to a finite total. A perpetuity that pays a constant `C` every year, discounted at rate `r`, is worth exactly `C / r`. That single formula is the seed of the Gordon-growth method.

**Gordon growth model (perpetuity growth).** The first method for terminal value. It assumes that after the horizon, free cash flow grows *forever* at a constant rate `g`. The present value (at the horizon) of a cash flow stream that starts at `FCF_(next year)` and grows forever at `g`, discounted at `WACC`, is:

$$\text{TV} = \frac{\text{FCF}_{\text{next year}}}{\text{WACC} - g}$$

This is just the constant-perpetuity formula `C / r` with the growth baked into the denominator, turning `r` into `(WACC − g)`. We derive *why* the growth lands in the denominator shortly.

**Exit multiple method.** The second method. Instead of assuming the business runs forever as an intrinsic perpetuity, it assumes the business is *sold* at the horizon for a price equal to a market multiple of its terminal earnings — most commonly **EBITDA** (earnings before interest, taxes, depreciation, and amortization, a rough proxy for operating cash generation). So:

$$\text{TV} = \text{terminal EBITDA} \times \text{exit multiple}$$

If terminal EBITDA is \$240M and comparable businesses sell for 10× EBITDA, the terminal value is \$2,400M. This is a *relative-value* bet — it imports the market's pricing of similar companies rather than deriving value from cash flows directly.

**Terminal growth rate (`g`).** The rate at which free cash flow is assumed to grow *forever* in the Gordon model. The single most important — and most abused — number in a terminal value. It must be small: no business can durably grow faster than the economy it operates in, so `g` is capped at roughly long-run **nominal GDP growth** (real growth plus inflation, historically around 2–3% for a developed economy). A `g` of 5% or 6% is almost always a mistake, for reasons we will hammer.

**WACC (weighted average cost of capital).** The discount rate for a firm-level DCF, built in [Part 2](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) — the blended cost of all the capital the business runs on. For Northwind it is about **8.4%**. The terminal value uses WACC twice: once inside the Gordon formula's denominator `(WACC − g)`, and once to discount the terminal lump back to today.

**ROIC (return on invested capital).** The after-tax operating profit a company earns on each dollar of capital invested in the business, `NOPAT / invested capital`. For Northwind, **16%**. ROIC governs how much value growth creates — including, crucially, growth in the terminal period. (Full treatment in [returns on capital](/blog/trading/equity-research/returns-on-capital-roic-roe-roa) and the [ROIC–WACC spread post](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value).)

**Reinvestment rate.** The fraction of after-tax operating profit a company plows back to fund growth. In the terminal period, where growth is `g` and returns are ROIC, the reinvestment rate the growth *requires* is `g / ROIC` — the same identity that anchored the [forecasting post](/blog/trading/equity-research/building-a-dcf-part-1-forecasting). This is what links terminal growth to terminal cash flow, and it is where most terminal values quietly cheat.

**NOPAT (net operating profit after tax).** Operating profit (EBIT) taxed as if the firm had no debt: `EBIT × (1 − tax rate)`. The seed of free cash flow. In the terminal period, terminal NOPAT minus terminal reinvestment is the terminal free cash flow that feeds the Gordon formula.

### Where we left Northwind

We carry forward the company from the DCF posts. Here is Northwind at the end of the explicit forecast, in year 5:

| Northwind — end of explicit forecast (Year 5) | Value |
|---|---:|
| Revenue | \$1,466.8M |
| Operating (EBIT) margin | 15.0% |
| EBIT | \$220.0M |
| NOPAT (after 25% tax) | \$165.0M |
| Free cash flow to the firm (FCFF), Year 5 | \$123.8M |
| ROIC | 16% |
| WACC | 8.4% |

By year 5, Northwind has reached steady state: growth has faded to 4%, margins have stabilized at 15%, and free cash flow has grown to \$123.8M as the reinvestment burden of its high-growth years has lifted. This is exactly the condition that lets us stop forecasting explicitly and switch to a terminal value. Everything after year 5 follows a stable pattern, and a stable pattern is summable. Now let's sum it.

## Why we need a terminal value at all

Start with the obvious question: why not just forecast every year until the company dies? Two reasons, one practical and one conceptual.

The **practical** reason is that you cannot forecast a hundred individual years with any credibility. Your year-3 revenue estimate is a guess; your year-30 estimate is a fantasy with extra steps. Building out fifty explicit rows does not make the distant years more accurate — it just dresses guesswork in the costume of precision. Past the point where the business has reached steady state, there is nothing left to forecast *differently* year to year; every year just repeats the same pattern. So forecasting them individually adds labor without adding information.

The **conceptual** reason is the more important one: once a business is in steady state, its future cash flows follow a *rule* simple enough to sum with a formula. If free cash flow grows at a constant `g` forever, you don't need to list out the infinite stream — the perpetuity formula collapses it into one number. The terminal value is not a shortcut or an approximation we tolerate; it is the *correct* way to value the steady-state tail, because the steady state genuinely is a simple, summable pattern.

So the structure of every DCF is: forecast explicitly through the messy, changing years until the business settles; then capture the settled, forever-after tail in a single terminal value at the horizon; then discount both pieces back to today. The terminal value is where the "forever" of "the present value of all future cash flows" actually lives. The explicit forecast handles the near, knowable years; the terminal value handles the rest of eternity.

And eternity, it turns out, is most of the value. This is not an accident or a quirk of Northwind — it is structural, and worth understanding *why*. A business that is still growing throws off relatively little free cash flow in its early years (it is reinvesting to grow, as we saw at length in the forecasting post) and *more* free cash flow later, once growth slows and reinvestment falls. The big, fat free cash flows are in the *future*, in the steady-state years — which are exactly the years the terminal value captures. The explicit forecast often catches a company during its cash-hungry growth phase; the terminal value catches it in its cash-generative maturity. So even before discounting, the terminal period holds most of the cash. That is the deep reason terminal value dominates.

#### Worked example: the share of Northwind's value in the terminal value

Let's make the dominance concrete by computing both pieces of Northwind's enterprise value. From the [forecasting post](/blog/trading/equity-research/building-a-dcf-part-1-forecasting), the five explicit years of free cash flow are \$25.2M, \$44.4M, \$67.4M, \$94.5M, and \$123.8M. Discount each back to today at the 8.4% WACC (divide year-`t` cash flow by `1.084^t`):

| Year | FCFF (\$M) | Discount factor (1.084^t) | PV (\$M) |
|---|---:|---:|---:|
| 1 | 25.2 | 1.084 | 23.2 |
| 2 | 44.4 | 1.175 | 37.8 |
| 3 | 67.4 | 1.274 | 52.9 |
| 4 | 94.5 | 1.381 | 68.4 |
| 5 | 123.8 | 1.497 | 82.7 |
| **PV of explicit years** | | | **\$265M** |

So the five explicit forecast years, discounted, are worth about **\$265M** today. Now the terminal value: we will compute it properly in the next section, but the answer is a terminal value of \$2,407M sitting at year 5, which discounted back five years (÷1.497) is **\$1,608M** today. Add the two pieces:

$$\text{Enterprise value} = \$265M + \$1,608M = \$1,873M$$

And the terminal value's share of the total is:

$$\frac{\$1,608M}{\$1,873M} = 86\%$$

**Eighty-six percent of Northwind's entire valuation is the terminal value.** The five years we forecast in painstaking detail — every margin assumption, every reinvestment line — account for just 14% of the answer. The other 86% rests on three terminal assumptions: the terminal cash flow, the growth rate `g`, and the discount rate. This is the single most important number to internalize about DCFs, because it tells you where to spend your skepticism.

*A DCF feels like a forecast of the next five years, but it is mostly a bet on a far-future steady state you summarized in a single formula — so the terminal value deserves far more scrutiny than it usually gets.*

## The two methods, side by side

There are two standard ways to compute a terminal value, and they come from genuinely different philosophies of what a company is worth at the horizon. Both are legitimate; the best practice is to compute *both* and reconcile them, because each is a check on the other. The figure below lays them out side by side.

![A side-by-side comparison of the two terminal-value methods, with the Gordon growth perpetuity method on the left assuming free cash flow grows forever at g and computing terminal value as next year's free cash flow divided by WACC minus g equals one hundred thirty over five point four percent, and the exit multiple method on the right assuming a buyer pays a market multiple of terminal EBITDA of two hundred forty million times ten, both landing near twenty-four hundred million but one being a perpetuity bet and the other a relative-value bet](/imgs/blogs/terminal-value-the-part-that-dominates-2.png)

**Method one: Gordon growth (perpetuity growth).** This treats the business as an intrinsic, going concern that generates cash forever, with that cash growing at a steady `g`. It is the more *theoretically pure* method, because it derives value from the company's own cash flows rather than from what other companies trade for. Its weakness is its violent sensitivity to `g` and `WACC`, which we will dissect.

**Method two: exit multiple.** This treats the horizon as a *sale*: at year 5, you imagine selling the whole business to a buyer who pays a market multiple of its terminal earnings. It is the more *market-grounded* method, because it imports the actual prices that comparable businesses command — it anchors the terminal value to reality rather than to a formula. Its weakness is that it imports the market's *current* mood, which may be a bubble or a trough, and projects it onto your horizon; it also smuggles in a growth-and-return assumption implicitly (a multiple *is* a bet on future growth and returns), just less transparently than Gordon growth does.

The deep point is that these two methods are *not independent* — they are two views of the same underlying reality, and a good analyst forces them to agree. Every Gordon-growth terminal value implies an exit multiple (just divide it by terminal EBITDA), and every exit multiple implies a growth rate (solve the Gordon formula backwards). If your Gordon-growth `g` of 3% implies a 10× exit multiple, and comparable companies actually trade at 10×, your two methods are telling a consistent story and you can trust the terminal value more. If your Gordon `g` implies a 25× multiple that no real company commands, one of your assumptions is broken. We will do exactly this reconciliation later; for now, hold the idea that the two methods are mutual lie detectors.

## Gordon growth: where the (WACC − g) denominator comes from

The Gordon-growth formula looks like it fell from the sky:

$$\text{TV} = \frac{\text{FCF}_{\text{next year}}}{\text{WACC} - g}$$

But it is just the perpetuity formula with growth, and seeing where it comes from is what lets you feel its danger in your bones. Start with a perpetuity that pays a *constant* cash flow `C` every year forever, discounted at rate `r`. Its present value is the sum of `C/(1+r) + C/(1+r)² + C/(1+r)³ + …`, an infinite geometric series that sums to exactly:

$$\text{PV} = \frac{C}{r}$$

A \$100 forever, discounted at 8%, is worth \$100 / 0.08 = \$1,250 today. Now let the cash flow *grow* at `g` each year: the payments are `C, C(1+g), C(1+g)², …`. Re-sum the (now growing) geometric series, and the growth subtracts from the discount rate in the denominator:

$$\text{PV} = \frac{C}{r - g}$$

That is the entire derivation. **Growth in the cash flows acts like a reduction in the discount rate.** Intuitively: if your cash flows are growing at `g`, then in *real terms relative to the growth*, you are only discounting at the leftover rate `(r − g)`. A stream growing at 3% discounted at 8.4% feels, to your wallet, like a flat stream discounted at 5.4%. That is why the effective denominator shrinks from `WACC` to `(WACC − g)`. The figure below builds this intuition and then shows the menace hiding in that small denominator.

![A two-panel figure, the left panel building the intuition that a non-growing perpetuity is worth free cash flow divided by WACC and that letting it grow at g effectively shrinks the discount rate to WACC minus g, and the right panel showing why the gap is dangerous because at WACC eight point four percent and g three percent the gap is only five point four points, so dividing by a small number makes terminal value huge and twitchy, jumping twenty-three percent when g moves to four percent and blowing up to infinity as g approaches WACC](/imgs/blogs/terminal-value-the-part-that-dominates-3.png)

Here is the menace. The denominator `(WACC − g)` is the *difference* between two numbers that are both small and not far apart. For Northwind, it is `8.4% − 3% = 5.4%`. That 5.4% is what you divide the terminal cash flow by, and dividing by a small number produces a large, unstable result. Worse: because it is a *difference*, a small absolute change in `g` is a *large proportional* change in the denominator. Move `g` from 3% to 4% — a single point — and the denominator shrinks from 5.4% to 4.4%, a 19% reduction, which inflates the terminal value by about 23%. Move `g` all the way to WACC and the denominator goes to *zero*, and the terminal value goes to *infinity*. The formula literally explodes. This is not a quirk; it is the defining behavior of terminal value, and it is why `g` is the most dangerous knob in valuation.

#### Worked example: Northwind's terminal value via Gordon growth

Let's compute it. We need the terminal cash flow — the free cash flow in the *first year after the horizon*, year 6. Northwind's year-5 FCFF is \$123.8M, and in steady state it grows at the terminal rate. We will normalize the terminal free cash flow to a round **\$130M** to represent a clean steady-state starting point (the year-5 \$123.8M grown modestly into a normalized terminal figure — using a tidy round number here keeps the arithmetic transparent, and it is within a hair of \$123.8M × 1.03 ≈ \$127.5M). We set the terminal growth rate at **`g` = 3%**, comfortably within long-run nominal GDP, and discount at **WACC = 8.4%**. Plug in:

$$\text{TV}_{\text{year 5}} = \frac{\text{FCF}_{\text{year 6}}}{\text{WACC} - g} = \frac{\$130M}{0.084 - 0.03} = \frac{\$130M}{0.054} = \$2,407M$$

So at the end of year 5, Northwind's terminal value is **\$2,407M** — the value, as of year 5, of every free cash flow from year 6 to eternity, assuming they grow at 3% forever. Notice the magnitude: \$130M of cash flow becomes a \$2,407M lump, a multiple of about 18.5×, entirely because we divided by the small 5.4% gap. That 18.5× is the price of a perpetuity that grows; it is *enormous*, and it is why the terminal value will dominate the valuation. Note also that this \$2,407M sits *at year 5* — it is a future amount, not a present value. We still have to discount it back, which we do next.

*The Gordon terminal value turns a single year's cash flow into a giant lump by dividing it by a small gap, which is exactly why both the cash flow and the gap have to be right.*

## Discounting the terminal value back to today

The terminal value we just computed, \$2,407M, is a *future* number — it is what the post-horizon cash flows are worth *as of year 5*. To add it to today's valuation, we must discount it back to the present, exactly like any other future cash flow. And here lurks a subtle, common, costly mistake: **discounting the terminal value back the wrong number of years.**

The rule is simple but easy to fumble: the terminal value computed by the Gordon formula sits *at the end of the explicit horizon*. For a five-year forecast, it sits at the end of year 5, so it is discounted back **five** years — the same number of years as the year-5 cash flow. Not six, not four. The reason it is five and not six is that the Gordon formula's numerator is the year-6 cash flow, but the formula already values that growing stream *as of year 5* (a perpetuity formula values the stream one period before its first payment). So the lump lives at year 5, and gets discounted five years.

The off-by-one error in either direction is surprisingly common. Discount it six years and you understate the value (you have pushed the lump one year too far into the future); discount it four years and you overstate it. Because the terminal value is most of the valuation, a one-year error in its discount period is a meaningful error in the whole answer — typically several percent. It is the kind of mistake that is invisible in a spreadsheet (the formula looks fine) and only caught by checking that the terminal value is discounted by the *same* factor as the final explicit year.

#### Worked example: discounting Northwind's terminal value back five years

Northwind's terminal value is \$2,407M at year 5. The five-year discount factor at 8.4% is:

$$(1 + \text{WACC})^5 = 1.084^5 = 1.497$$

So the present value of the terminal value is:

$$\text{PV of TV} = \frac{\$2,407M}{1.497} = \$1,608M$$

The terminal value, worth \$2,407M at year 5, is worth **\$1,608M** today. That is the number that goes into the enterprise value. Combine it with the \$265M present value of the explicit years from earlier, and Northwind's enterprise value is **\$1,873M**, of which the terminal value's \$1,608M is **86%**.

Now watch the off-by-one. If we had carelessly discounted the terminal value back *six* years instead of five — `2,407 / 1.084^6 = 2,407 / 1.623 = \$1,483M` — we would have understated the terminal value by \$125M, dropping the enterprise value to \$1,748M, a **7% error** in the entire valuation, from a single mis-typed exponent. And discounting back only *four* years — `2,407 / 1.084^4 = 2,407 / 1.381 = \$1,743M` — would *over*state it by \$135M. Because the terminal value is the bulk of the value, its discount period is one of the highest-leverage details in the model, and one of the easiest to get wrong.

*The terminal value lives at the horizon year and is discounted exactly that many years — the same factor as your final explicit cash flow — and getting that exponent wrong by one swings most of the valuation.*

## Why a huge terminal-value share is a warning, not a comfort

It is tempting to look at "86% of value in the terminal value" and shrug — that's just how DCFs work, right? Partly. But the *size* of the terminal-value share is itself diagnostic information, and a very high share is a yellow flag that should make you nervous, not complacent.

Think about what a high terminal-value share *means*. If 86% of the value is in the terminal value, then 86% of your valuation rests on assumptions about a steady state you cannot observe and can barely forecast — the terminal growth rate, the terminal margin, the terminal return on capital. The explicit forecast, the part you can actually reason about with company-specific detail, is contributing only 14%. The valuation is mostly a bet on the far future, dressed up with a near-future forecast that barely moves the needle. The higher the terminal-value share, the more the whole valuation is a single fragile assumption wearing a spreadsheet.

A terminal-value share creeps *up* for two reasons, one benign and one alarming. The benign reason is a **short explicit horizon**: if you only forecast three years, more of the company's life falls into the terminal value, mechanically raising its share. The fix is to forecast longer, until the business genuinely reaches steady state, which pulls more value into the explicit (and more defensible) period. The alarming reason is a **high-growth company whose value is all in the distant future**: a business reinvesting everything today to grow, with its big cash flows decades away, will have a terminal value that is 90%+ of the valuation no matter how long you forecast — and that valuation is correspondingly fragile, which is exactly why high-growth-stock valuations are so contentious and so volatile. When the terminal-value share is very high, the honest response is to widen your sensitivity ranges and trust the point estimate less.

The practical discipline: always report the terminal-value share alongside the valuation, and treat a share above ~80% as a signal to (a) check that your horizon is long enough, and (b) stress-test the terminal assumptions hard, because they *are* the valuation. A model that hides the terminal-value share is hiding how much of its answer is a far-future guess.

## The growth rate `g`: why it must be small

We have seen that the terminal value is violently sensitive to `g`, and that a higher `g` produces a higher value. This creates an irresistible temptation: bump `g` up a little, and watch the valuation climb. Resisting that temptation is the single most important discipline in terminal value, and the discipline rests on one ironclad principle.

**A company cannot grow faster than the economy forever.** This is not a rule of thumb; it is arithmetic. If a company grows even 1% faster than the overall economy, then — given enough time — it eventually becomes *larger than the entire economy*, which is impossible. A business growing 6% forever in a 4%-nominal-growth economy would, over a century or two, come to represent an absurd, impossible fraction of all economic activity. So the terminal growth rate `g`, which by definition is the *forever* growth rate, is hard-capped by the long-run growth of the economy the company operates in — roughly **nominal GDP**, which is real growth (~2%) plus inflation (~2%), so about 4% as a ceiling, and most analysts use something more conservative like 2–3% to leave a margin of safety.

This is why a terminal `g` of 5% or 6% is almost always wrong, and a `g` of 8% or 10% is a flashing red light. Such rates assume the company outgrows the world economy in perpetuity, which no company has ever done or can ever do. The fact that a higher `g` produces a more attractive valuation is precisely the trap: the number that flatters the valuation most is the number that is least defensible. The figure below plots the explosion directly — terminal value as a function of `g`, holding everything else fixed — and you can see the curve bend gently at low `g` and then rocket toward infinity as `g` climbs toward WACC.

![A line chart with terminal growth rate g on the horizontal axis running from zero to the eight point four percent WACC and terminal value on the vertical axis, showing a curve that rises gently and almost flat at low growth rates of two and three percent then bends sharply upward and explodes toward infinity as g approaches the WACC, with a shaded sane region near long-run GDP of two to three percent and a shaded danger region at high growth where terminal value runs away from reality, and a dashed vertical asymptote at g equals WACC where terminal value is infinite](/imgs/blogs/terminal-value-the-part-that-dominates-4.png)

The shape of that curve is the whole lesson. In the *sane region* — `g` at or below long-run GDP, the leftmost portion — the curve is gentle: moving `g` from 2% to 3% changes the terminal value modestly. But as `g` climbs into the danger zone toward WACC, the curve goes nearly vertical: now a tiny change in `g` produces an enormous change in value, and at `g = WACC` the value is literally infinite. The lesson: **stay on the flat part of the curve.** A terminal `g` near long-run GDP keeps you in the gentle region where the valuation is stable; a terminal `g` reaching for WACC puts you on the cliff face where the valuation is meaningless. Any analyst who needs a high `g` to make the valuation "work" is standing on the cliff and calling it solid ground.

#### Worked example: the value swinging when `g` goes from 2% to 4%

Let's quantify the sensitivity with Northwind's numbers, holding the terminal cash flow at \$130M and WACC at 8.4%, and moving only `g`:

| Terminal `g` | Denominator (WACC − g) | TV at year 5 (\$M) | PV of TV (÷1.497, \$M) |
|---|---:|---:|---:|
| 2% | 6.4% | 2,031 | 1,357 |
| 3% | 5.4% | 2,407 | 1,608 |
| 4% | 4.4% | 2,955 | 1,974 |

Walk through it. At `g = 2%`, the terminal value is \$130M / 0.064 = \$2,031M, worth \$1,357M today. At `g = 4%`, it is \$130M / 0.044 = \$2,955M, worth \$1,974M today. So moving `g` from 2% to 4% — a two-point change in a single far-future assumption — swings the present value of the terminal value from \$1,357M to \$1,974M, a jump of \$617M, or about **45%**. And since the terminal value is most of the enterprise value, that 45% swing in the terminal value translates into roughly a **40% swing in the entire valuation**, from one analyst choosing 2% and another choosing 4%. Two people who agree on every single line of the five-year forecast, every margin and every reinvestment number, can still produce valuations 40% apart purely from a two-point disagreement about a growth rate decades in the future that neither can actually forecast.

*The terminal growth rate is the most leveraged number in a DCF — a swing well within the range of "reasonable" assumptions moves the whole valuation by tens of percent, which is why honest DCFs always show the answer as a range across `g`, never a single point.*

## The convergence insight: growth in the terminal value is only worth something if ROIC beats WACC

Here is the most subtle and important idea about terminal value, and the one that separates people who *understand* DCFs from people who merely *operate* them. It is tempting to think a higher terminal growth rate is always better — more growth, more value, full stop. **That is false.** Growth in the terminal value creates value *only* if the company earns a return on its growth capital above its cost of capital. If terminal ROIC merely *equals* WACC, growth adds *exactly nothing* to the terminal value. This is the **convergence** result, and it falls straight out of the reinvestment identity.

Recall from the [forecasting post](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) that to grow at rate `g` with a return on invested capital of ROIC, a company must reinvest a fraction `g / ROIC` of its operating profit. So the terminal free cash flow is not the full terminal NOPAT — it is terminal NOPAT *minus* the reinvestment that the terminal growth requires:

$$\text{FCF}_{\text{terminal}} = \text{NOPAT} \times \left(1 - \frac{g}{\text{ROIC}}\right)$$

Now substitute this into the Gordon formula. The terminal value becomes:

$$\text{TV} = \frac{\text{NOPAT} \times \left(1 - \dfrac{g}{\text{ROIC}}\right)}{\text{WACC} - g}$$

Stare at this. Growth `g` appears in *two* places now: it *increases* the terminal value through the shrinking denominator `(WACC − g)`, but it *decreases* the terminal value through the reinvestment drag `(1 − g/ROIC)` in the numerator, because faster growth eats more cash in reinvestment. Whether growth helps or hurts depends entirely on which effect wins — and the tipping point is exactly where **ROIC equals WACC.** When terminal ROIC = WACC, the two effects cancel *perfectly*: the value you gain from the smaller denominator is exactly offset by the cash you lose to reinvestment, and the terminal value is the same *regardless of `g`*. Growth becomes value-neutral. The figure below shows this across the three cases.

![A three-by-three matrix showing how terminal growth affects value at three levels of terminal return on invested capital, with the top row where ROIC of twenty percent exceeds WACC showing terminal value rising from fifteen hundred forty-eight million at zero growth to two thousand forty-six million at three percent growth so growth creates value, the middle row where ROIC of eight point four percent equals WACC showing terminal value unchanged at fifteen hundred forty-eight million whether growth is zero or three percent so growth adds nothing, and the bottom row where ROIC of five percent is below WACC showing terminal value falling from fifteen hundred forty-eight million to nine hundred sixty-three million as growth rises so growth destroys value](/imgs/blogs/terminal-value-the-part-that-dominates-5.png)

Read the three rows. When **ROIC > WACC** (top row), growth *creates* value: each dollar reinvested earns more than it costs, so more growth means more value, and the terminal value rises with `g`. When **ROIC = WACC** (middle row), growth is *neutral*: the terminal value is identical whether you assume 0% growth or 3% growth, because the company is just running to stand still, earning exactly its cost of capital on every reinvested dollar. And when **ROIC < WACC** (bottom row), growth actively *destroys* value: every reinvested dollar earns less than it costs, so the faster the company grows, the more value it incinerates, and the terminal value *falls* as `g` rises. This is the single most counterintuitive and important fact in valuation: **for a company earning below its cost of capital, growth is a liability, and a shrinking such business can be worth more than a growing one.**

The practical implication for terminal value is enormous. A terminal growth rate is only *worth* assuming if you are also assuming a terminal ROIC above WACC. If you bump `g` up to flatter the valuation but your company's terminal ROIC is only at or below WACC, the bump does *nothing* (at ROIC = WACC) or *hurts* (at ROIC < WACC) — even though the naive Gordon formula, which ignores the reinvestment drag, would show the value rising. This is why the most rigorous DCFs build the terminal value from the `NOPAT × (1 − g/ROIC)` form, not the bare `FCF / (WACC − g)` form: the rigorous form *forces* the reinvestment drag of growth to show up, so you cannot get free value from growth you haven't earned the returns to justify.

#### Worked example: Northwind's convergence case, where growth stops adding value

Let's make convergence concrete. Take Northwind's terminal NOPAT at \$130M (using a round figure for clarity), WACC at 8.4%, and ask what the terminal value is at two different terminal ROICs, comparing zero growth to 3% growth.

**Case A — terminal ROIC = 16% (Northwind's actual, well above WACC).** At `g = 0%`, reinvestment is zero, so terminal FCF = NOPAT = \$130M, and TV = \$130M / 0.084 = **\$1,548M**. At `g = 3%`, reinvestment rate is `3% / 16% = 18.75%`, so terminal FCF = \$130M × (1 − 0.1875) = \$105.6M, and TV = \$105.6M / 0.054 = **\$1,956M**. Growth *raised* the terminal value by \$408M, because Northwind earns 16% on capital that costs 8.4% — the 7.6-point spread makes growth genuinely valuable.

**Case B — terminal ROIC = 8.4% = WACC (the convergence case).** At `g = 0%`, terminal FCF = NOPAT = \$130M, and TV = \$130M / 0.084 = **\$1,548M**. Now at `g = 3%`, reinvestment rate is `3% / 8.4% = 35.7%`, so terminal FCF = \$130M × (1 − 0.357) = \$83.6M, and TV = \$83.6M / 0.054 = **\$1,548M** — *identical.* The growth did *nothing*. The smaller denominator (0.054 instead of 0.084) made the value want to rise, but the heavier reinvestment (35.7% of NOPAT) took exactly that much cash away, and the two effects cancelled to the dollar. When a company earns exactly its cost of capital, growing it is pure motion without progress: it gets bigger, ties up more capital, and is worth not one cent more.

*Growth is only worth paying for when returns exceed the cost of capital; in the terminal value, a high `g` paired with a terminal ROIC no better than WACC is an illusion that the rigorous formula erases — the reinvestment drag eats every dollar the shrinking denominator tries to add.*

## Cross-checking Gordon growth against an implied exit multiple

We now have a powerful internal check, born from the fact that the two terminal-value methods are two views of the same reality. Take your Gordon-growth terminal value, divide it by terminal EBITDA, and you have backed out the **implied exit multiple** — the EBITDA multiple your perpetuity assumptions are secretly assuming a buyer would pay. Then ask the killer question: *would the market actually pay that multiple for a business like this?* If yes, your two methods agree and you can trust the terminal value more. If the implied multiple is absurd, your Gordon assumptions are absurd, even if each one looked reasonable in isolation. The figure below runs this reconciliation for Northwind.

![A before-and-after figure showing a terminal-value reconciliation, the left side starting from Gordon growth with terminal free cash flow of one hundred thirty million at WACC eight point four percent and g three percent giving a terminal value of twenty-four hundred seven million at year five, then dividing by terminal EBITDA of about two hundred forty million to back out an implied exit multiple of ten times EBITDA, and the right side sanity-checking by asking whether mature peers trade near ten times which they do so it passes, while a higher g of five percent would imply eighteen times which if peers trade at nine times would expose the growth rate as a fantasy, concluding that two methods that agree are a stronger answer than either alone](/imgs/blogs/terminal-value-the-part-that-dominates-6.png)

#### Worked example: backing out Northwind's implied exit multiple and reconciling

Northwind's Gordon-growth terminal value is \$2,407M at year 5. Its terminal EBITDA — operating profit plus depreciation, roughly \$220M of EBIT plus ~\$20M of D&A — is about **\$240M**. So the implied exit multiple is:

$$\text{Implied multiple} = \frac{\text{TV}}{\text{terminal EBITDA}} = \frac{\$2,407M}{\$240M} = 10.0\times$$

Our Gordon assumptions (3% growth, 8.4% WACC) are implicitly assuming that at year 5, a buyer would pay **10× EBITDA** for Northwind. Now the sanity check: do mature industrial-manufacturing companies trade around 10× EBITDA? Broadly, yes — 8× to 11× is a typical range for stable, mid-quality industrials. So our two methods *agree*: the intrinsic Gordon value and the relative-value exit multiple tell a consistent story, and we can trust the terminal value more than we would trust either method alone.

Now contrast a broken case. Suppose an analyst, hungry for a higher valuation, sets `g = 5%`. The terminal value becomes \$130M / (0.084 − 0.05) = \$130M / 0.034 = \$3,824M, implying a multiple of \$3,824M / \$240M = **15.9×**. If comparable industrials trade at 9–10×, that 15.9× implied multiple is a flashing alarm: the 5% growth rate is assuming the market would pay a premium no peer commands. The implied-multiple cross-check caught a `g` assumption that, on its own line in the spreadsheet, looked like just another number. *This is the entire value of the reconciliation: it converts an abstract, hard-to-judge growth rate into a concrete multiple you can hold up against the market and immediately see is wrong.*

The exit-multiple method run forward has its own discipline, worth stating: choose the multiple from *mature, steady-state* peers, not from high-growth darlings, because your terminal company is by assumption mature. And apply the multiple to a *normalized* terminal EBITDA, not a boom-year or trough-year figure — applying a peak multiple to a peak EBITDA double-counts the cycle and produces a terminal value that evaporates when the cycle turns. The two methods, reconciled, are far stronger than either alone, because the perpetuity grounds the multiple in cash-flow logic and the multiple grounds the perpetuity in market reality.

## The classic terminal-value traps

After you have built and reviewed enough DCFs, you stop seeing infinite variety in how terminal values go wrong and start seeing the same handful of mistakes. Almost every blown terminal value commits one of six recognizable traps, each with a tell you can spot in seconds and a discipline that fixes it. The figure below catalogs them.

![A two-row grid of six terminal-value traps with their fixes, the top row in red showing g set too high which claims the firm outgrows the economy forever and is fixed by capping g near two to three percent, g and reinvestment mismatched which implies infinite ROIC and is fixed by reinvesting g over ROIC, and terminal margins and returns that never fade from a peak which is fixed by using mid-cycle figures, and the bottom row in amber showing discounting the terminal value back the wrong number of years which is fixed by discounting it exactly the horizon number of years, no cross-check against an implied multiple which is fixed by backing out and sanity-checking the implied multiple, and an exit multiple taken off a boom year which double-counts the cycle and is fixed by normalizing both the multiple and the earnings](/imgs/blogs/terminal-value-the-part-that-dominates-7.png)

Walk through them, because each maps to a discipline we have built. **`g` too high** is the cardinal sin — a terminal growth rate above long-run GDP claims the company outgrows the economy forever; the fix is to cap `g` at roughly 2–3%. **`g` and reinvestment mismatched** is the subtler cousin: growing `g` while holding reinvestment near zero implies an infinite terminal ROIC (the [forecasting post's](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) cardinal sin, now in the tail); the fix is the `NOPAT × (1 − g/ROIC)` form that forces reinvestment to scale with growth. **Terminal margins and returns that never fade** assume the company holds peak profitability forever despite competition; the fix is to use mid-cycle, defensible terminal margins and a terminal ROIC that has converged toward a sustainable level. **Discounting the terminal value the wrong number of years** is the off-by-one we dissected; the fix is to discount it exactly the horizon number of years, the same factor as your final explicit cash flow. **No cross-check** leaves a Gordon value that might imply an absurd multiple unexamined; the fix is the implied-multiple reconciliation. And **an exit multiple off a boom year** applies a peak-cycle multiple to a peak-cycle EBITDA, double-counting the cycle; the fix is to normalize both the multiple and the earnings to mid-cycle.

Notice the pattern: the traps are almost all *consistency* failures — the terminal value asserts something (a growth rate, a margin, a return) that some other part of the model, or the outside market, contradicts. That is why the master discipline of terminal value, like the master discipline of forecasting, is *internal consistency checked against external reality*: every terminal assumption must be compatible with every other (growth with reinvestment, margins with competition) and with what the market actually pays (the implied multiple). A terminal value that passes all those checks is one you can defend; one that fails any of them is a fantasy with a large number attached.

## Common misconceptions

**"The terminal value is a minor adjustment at the end of the model."** It is the *opposite* — it is usually the *majority* of the valuation, 60–80% in a typical DCF and 86% for Northwind. The explicit forecast, where all the detailed work goes, is often the minority of the value. Treating the terminal value as a tidy-up at the end, computed with a quick formula and minimal thought, is treating most of your answer as an afterthought. The terminal value deserves the *most* scrutiny, not the least, precisely because it carries the most weight.

**"A higher terminal growth rate always makes the company more valuable."** Only if the company earns a return above its cost of capital on the growth. If terminal ROIC equals WACC, growth adds *exactly nothing* (the convergence result); if terminal ROIC is below WACC, growth *destroys* value, and a higher `g` makes the company worth *less*. The naive Gordon formula, which ignores reinvestment, hides this by letting growth inflate the value for free. The rigorous formula, `NOPAT × (1 − g/ROIC) / (WACC − g)`, exposes it. Growth is not intrinsically good; growth at returns above the cost of capital is good.

**"You can pick the terminal growth rate to make the valuation come out where you want."** You can, and it is the most common way DCFs are abused — but it is also the most catchable. A `g` reverse-engineered to hit a target price will almost always (a) exceed long-run GDP, exposing it as economically impossible, and (b) imply an exit multiple no comparable company trades at, exposing it as market-impossible. The two cross-checks — GDP ceiling and implied multiple — are designed precisely to catch a `g` that was chosen for the answer rather than derived from the business. An honest `g` is small and survives both checks.

**"The two terminal-value methods are alternatives — pick the one you like."** They are not alternatives to choose between; they are *cross-checks* to reconcile. Every Gordon value implies a multiple, and every multiple implies a growth rate. Computing only one and ignoring the other throws away the single best consistency check you have. The disciplined practice is to compute both, back out what each implies about the other, and only trust the terminal value when the intrinsic (Gordon) and relative (multiple) views agree. Two methods that converge are far stronger than either alone.

**"EBITDA is the right terminal cash flow for the Gordon formula."** No — EBITDA is *not* free cash flow, and feeding it into the Gordon formula massively overstates the terminal value. EBITDA ignores taxes, capex, and the reinvestment that even a steady-state company needs to maintain and modestly grow its asset base. The Gordon numerator must be *free cash flow* — NOPAT minus terminal reinvestment — not EBITDA, not NOPAT, and certainly not EBIT. EBITDA belongs in the *exit-multiple* method (as the thing you multiply), not in the perpetuity. Confusing the two is a common and expensive error.

**"If the explicit forecast is good, the terminal value will take care of itself."** The explicit forecast and the terminal value are *different bets* requiring *different* assumptions. A perfect five-year forecast tells you almost nothing about the right terminal growth rate, terminal margin, or terminal ROIC — those are assumptions about a steady state *beyond* the forecast, and they have to be made and defended separately. Since the terminal value is most of the answer, "the forecast is good" is not reassurance about the valuation; the terminal assumptions are a whole separate front where the valuation can still go badly wrong.

## How it shows up in real markets

**High-growth technology valuations and the terminal cliff.** The fiercest valuation debates in markets — over high-multiple software and platform companies — are almost entirely debates about the terminal value, even when the participants think they are arguing about next year's revenue. For a company reinvesting everything to grow, with its large free cash flows decades away, the terminal value can be 90%+ of the valuation no matter how long the explicit horizon. That is *why* these stocks are so volatile: a small shift in the assumed terminal growth or terminal margin, or a move in interest rates that changes WACC, swings most of the valuation. The bulls and bears usually agree on the next two years and disagree violently about the steady state — which is to say, about the terminal value. When you hear that a stock is "priced for perfection," it means its terminal assumptions are pinned at the optimistic end of the explosion curve, where small disappointments produce large repricings. (These dynamics are illustrative of the pattern, not a claim about any specific company's exact figures.)

**The interest-rate sensitivity of long-duration assets.** Because the terminal value is discounted at WACC and sits far in the future, it is acutely sensitive to changes in the discount rate — and the WACC moves with interest rates. When rates rose sharply in 2022, the assets that fell hardest were precisely the long-duration ones: high-growth, low-current-cash-flow companies whose value was almost all in the terminal lump, discounted back many years. A one-point rise in WACC, compounded over the many years to the terminal value and applied to a number that is most of the valuation, produces an outsized hit. The terminal value is where a DCF's interest-rate duration concentrates, which is why "long-duration equity" reprices like a long bond when rates move.

**Mature, cash-cow businesses and the small terminal-value share.** At the other extreme, a mature consumer-staples or utility business — slow-growing, fully in steady state, throwing off big free cash flows *today* — has a *low* terminal-value share, because much of its value is in the near, knowable cash flows rather than the distant tail. These are the businesses [value investors like Warren Buffett](/blog/trading/finance/warren-buffett-berkshire-value-investing) prize, and the terminal-value lens explains part of why: a valuation that is *less* dependent on the terminal value is *less* dependent on unforecastable far-future assumptions, and therefore more trustworthy. A boring business whose value is mostly in the next decade's cash flows is a more reliable DCF than an exciting one whose value is all in a terminal lump you cannot see.

**Private equity and the exit-multiple discipline.** Private equity firms live and die by the exit multiple, because their terminal value is a *literal* sale: they buy a company, hold it for five years, and sell it. Their entire return model hinges on the exit-multiple assumption — and the cardinal PE sin is *multiple expansion in the model*, assuming they sell at a higher multiple than they bought at without a concrete reason. A PE model that buys at 9× and assumes a 12× exit is doing exactly the "exit multiple off a boom year" trap, baking in market appreciation it cannot control. The disciplined PE underwrite assumes a *flat or conservative* exit multiple and earns its return from operating improvement and debt paydown, not from a terminal multiple it hopes the market will hand it. The exit multiple, for them, is not a modeling convenience — it is the bet.

**Accounting frauds and the cash that never materializes.** The terminal value assumes a steady state of *real* free cash flow stretching to infinity — which makes it worthless if the cash flows are fake. The companies at the center of the great frauds, [Enron](/blog/trading/finance/enron-2001-accounting-fraud) and [Wirecard](/blog/trading/finance/wirecard-the-german-fintech-fraud), reported growing earnings while their actual cash was thin or fictional. A terminal value built on reported earnings would have valued an infinite stream of cash that did not exist. This is the deep reason terminal value must be built on *free cash flow*, not earnings, and why the [quality-of-earnings](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) work matters so much before you ever reach the terminal value: a terminal value is the present value of forever, and forever is a very long time to be capitalizing a number that isn't real.

## When this matters and further reading

Terminal value is where most of a DCF's value lives, which makes it where most of a DCF's *error* can live too. Once you can build it both ways, keep `g` below the economy's growth, force reinvestment to scale with growth through `g / ROIC`, discount the lump the right number of years, and cross-check the intrinsic value against the multiple it implies, you have mastered the part of valuation that actually moves the answer. The five-year forecast is the part everyone obsesses over; the terminal value is the part that decides the verdict.

The terminal value is one piece of the DCF machine; the others complete the picture:

- **[Building a DCF, Part 1: Forecasting](/blog/trading/equity-research/building-a-dcf-part-1-forecasting)** — the explicit forecast that produces the cash flows up to the horizon, and the steady state that the terminal value picks up from.
- **[Building a DCF, Part 2: Cost of Capital (WACC and CAPM)](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm)** — the discount rate that appears twice in the terminal value, in the `(WACC − g)` denominator and in discounting the lump back.
- **[Time Value of Money: Discounting for Investors](/blog/trading/equity-research/time-value-of-money-discounting-for-investors)** — the perpetuity math that makes an infinite stream have a finite value, the foundation of the Gordon formula.

And for where this terminal value goes next, and the principle that governs whether its growth is worth anything:

- **[Enterprise Value to Per-Share: The Bridge](/blog/trading/equity-research/enterprise-value-to-per-share-the-bridge)** — how the enterprise value we built here (explicit PV plus terminal PV) becomes a per-share price you can compare to the market.
- **[The ROIC–WACC Spread: The Engine of Intrinsic Value](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value)** — the principle behind the convergence result: growth, including terminal growth, only creates value when returns beat the cost of capital.

Build the terminal value with discipline — a small `g`, reinvestment tied to returns, the right discount period, and an implied multiple that survives contact with the market — and the most leveraged number in your valuation becomes one you can defend. Build it carelessly, and the 80% of your answer that lives in the terminal value will be 80% guess, expressed to four decimal places.
