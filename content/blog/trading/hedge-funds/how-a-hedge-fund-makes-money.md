---
title: "How a hedge fund makes money: the management fee, the incentive fee, and the business you really own"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A hedge fund earns returns for its investors, but the manager earns money two ways: a management fee that pays the bills and an incentive fee that builds the wealth. Understand the cash flows and you understand the whole business you are about to start."
tags: ["hedge-funds", "fund-management", "asset-management", "management-fee", "incentive-fee", "high-water-mark", "fund-economics", "management-company", "two-and-twenty", "founder-playbook"]
category: "trading"
subcategory: "Hedge Funds"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A hedge fund earns *returns for its investors*; the *manager* earns money two ways, and the difference between those two sentences is the whole business.
>
> - The **management fee** is a small annual percentage of assets under management (AUM), historically 2%, now closer to 1.3–1.5%. It is predictable, it scales straight-line with size, and its only job is to keep the lights on.
> - The **incentive fee** is roughly 20% of the *profit above a high-water mark*. It is the real wealth, it scales with performance, and it pays exactly zero when you are underwater. It is shaped like a call option you wrote to yourself.
> - The business you own is the **management company**, not the fund. The fund holds the investors' capital; the management company collects the fees, pays the cost stack, and distributes what is left to you.
> - The one number to remember: at a 1.5% management fee against roughly USD 1.2M of fixed costs, you need about **USD 80M of AUM just to break even** — and in a flat year, that fee is all you earn.

Maya had run a long/short equity book for nine years. She had a Sharpe she was proud of, a two-page tear sheet that allocators nodded at, and, after eighteen months of legal bills, a Delaware limited partnership called the Meridian Fund with USD 40 million of day-one capital. On launch morning she expected to feel like an owner. Instead she felt like she had just signed up for a very expensive job.

Here is the arithmetic that ruined her launch-week glow. Her management fee was 1.5% of AUM. On USD 40 million that is USD 600,000 a year — which sounds like a lot until you put it next to the bill stack: an administrator, an auditor, outside counsel, a compliance consultant, a Bloomberg terminal and an order-management system, an analyst's salary, her own draw, the prime broker's minimums, the office, the insurance. Her fixed run-rate was about USD 1.2 million. The management fee covered *half* of it. And the incentive fee — the famous 20%, the part everyone thinks of when they hear "hedge fund" — was zero, because three weeks into trading she was down 4% and the fund had a high-water mark she had to climb back over before she could charge a cent of it.

She had spent a decade learning how to *run money*. Nobody had ever sat her down and explained how a fund manager actually *makes* money — that there are two revenue lines with completely different shapes, that one of them is an option that can pay nothing for years, that the asset she now owned was not the fund at all but a separate company sitting beside it, and that the entire business is a race to reach the AUM where the fees finally exceed the costs before the cash runs out.

This post is that explanation. We are going to follow every dollar from the investor's capital to Maya's bank account, and by the end you will understand the economics of a hedge fund the way an owner does, not the way a headline does. Figure 1 is the whole story on one page; we will spend the rest of the post earning the right to read it.

![A stacked diagram showing how a hedge fund manager gets paid from two revenue lines, a steady management fee on assets and a performance fee on profits above the prior peak](/imgs/blogs/how-a-hedge-fund-makes-money-1.png)

## Foundations: the two revenue lines and the two entities

Before any of the nuance, you have to get four definitions and one distinction firmly in your head. Everything else in this post — every worked example, every misconception, every survival calculation — is a recombination of these five ideas. If you already trade for a living, the *markets* part of a fund is familiar to you; it is this *business* layer that is usually a blank.

**Assets under management (AUM).** The total capital the manager is responsible for — the money the investors have entrusted to the fund. AUM is the base that the management fee is charged on, so it is the single most important number in the manager's business. A USD 40M fund and a USD 500M fund can run an identical strategy with identical returns; their *businesses* are nothing alike, because the fee revenue is more than ten times larger for the second one. AUM grows two ways: by performance (the fund makes money, so the asset base swells) and by net inflows (new investors subscribe faster than old ones redeem). It shrinks the same two ways in reverse.

**Net asset value (NAV).** The value of the fund per unit of ownership, struck on a schedule (monthly for most hedge funds). If you own one share/interest of the fund and the NAV per share is USD 1,050 today versus USD 1,000 at the start of the year, your investment is up 5% before fees. The administrator computes NAV; it is the number every fee, every subscription, and every redemption is calculated against. Keep NAV-per-share separate from total AUM in your head: NAV-per-share is the *price* of the fund, AUM is the *price times the number of shares outstanding*.

**The management fee.** A fixed annual percentage of AUM, accrued and charged regardless of performance. The classic figure is 2% — so on USD 100M of AUM, USD 2M a year, typically accrued monthly (about 0.167% a month) and deducted from NAV. As of 2026 the *average* has compressed: industry fee data (HFR and others) put the typical management fee closer to **1.3–1.5%** as allocators pushed back over the last decade. This fee is the predictable revenue line. It does not care whether you are up or down; it cares only how much money you manage.

**The incentive fee (also called the performance fee).** A percentage — classically 20% — of the *profit* the fund generates, charged only on gains, and only on gains *above a high-water mark*. This is the variable, performance-linked revenue line. In a year where the fund returns 15%, the manager keeps 20% of that 15% (before we adjust for the high-water mark and any hurdle). In a flat or down year, the incentive fee is zero. This is the line that makes founders rich — and the line that pays nothing for a manager in a slump. The current industry average has drifted to roughly **15–17%**, but "twenty" is still the mental anchor and we will use 20% in the examples.

**The high-water mark.** The previous peak NAV per share that the fund must exceed before the manager can charge another incentive fee. If your fund hits USD 1,200 NAV per share, charges its incentive fee, then falls to USD 1,000, you cannot charge an incentive fee again until the NAV climbs back above USD 1,200 — even though, on the way up from USD 1,000, your investors are making money. The high-water mark exists so that investors never pay a performance fee twice on the same dollar of gains. It is the single most important reason the incentive fee can be zero for a long time, and we devote a whole section to it below. (Its companion, the **hurdle rate**, sets a *minimum* return — say the cash rate, or a fixed 5% — below which no incentive fee applies at all; many funds have a high-water mark but no hurdle, and we will treat the hurdle as an optional add-on.)

That is the four-term vocabulary of the *revenue*. Now the distinction that confuses almost everyone, and the one this whole post is built around.

### The fund is not the business. The management company is.

When people say "I want to start a hedge fund," they picture one thing — a pool of money they trade. In reality you are creating **at least two separate legal entities**, and they have completely different jobs.

**The fund** is the investment vehicle, usually structured as a limited partnership (a Delaware LP for US investors, with a Cayman company as the offshore feeder for non-US and tax-exempt investors — we cover that machinery in [what a hedge fund actually is](/blog/trading/hedge-funds/what-a-hedge-fund-actually-is)). The fund holds the *investors'* capital and the *positions* the strategy puts on. The investors are **limited partners (LPs)**: they put in money, they share in the gains and losses, and their liability is limited to what they invested. The fund does not "earn fees" — it *pays* them. The fund's purpose is to compound the LPs' capital.

**The management company** (often just "the ManCo," or the "investment adviser") is the *operating business*. It is the entity you, the founder, actually own. The management company employs the staff, signs the Bloomberg contract, and — crucially — **receives the fees**. The management fee and the incentive fee flow *out of the fund* and *into the management company*. The management company pays the cost stack out of that revenue, and whatever is left over is the profit the founding partners distribute to themselves. When a GP-stakes buyer like Petershill or Blackstone's Strategic Capital writes a check to "buy a piece of a hedge fund," they are almost always buying a slice of the **management company's revenue**, not a slice of the fund's assets. We preview that valuation in [capital structure, evolution, and GP stakes](/blog/trading/hedge-funds/capital-structure-evolution-and-gp-stakes); for now, just hold the fact that the management company is the thing with enterprise value.

**The general partner (GP)** is a third entity (frequently a separate LLC) that legally controls the fund and bears the liability that the limited partners are shielded from. In a lot of small launches the GP and the ManCo are effectively the same people wearing two hats, and the GP entity exists mostly to isolate liability. The detail that matters for *this* post is simpler: the **fund holds the assets, the management company earns the fees, and the GP carries the legal risk.** Three boxes, three jobs.

Why split them at all? Three reasons, all of which an allocator's operational due diligence team will check (see [why operations and ODD make or break a launch](/blog/trading/hedge-funds/what-a-hedge-fund-actually-is)). First, **liability isolation**: if the management company gets sued, the fund's assets — the LPs' money — should not be exposed, and vice versa. Second, **clean economics**: the fund's books show only investments and the fees it pays; the management company's books show the actual operating business, which is what you would eventually sell or take seed capital against. Third, **tax and partnership mechanics**: structuring the incentive fee as an *allocation* to the GP rather than a fee can change its tax character, and keeping the entities separate makes that clean. You do not need to master the tax here; you need to internalize that there are separate boxes and the money moves *between* them in a specific direction.

With those five ideas in hand — AUM, NAV, the management fee, the incentive fee, the high-water mark, and the fund-versus-ManCo split — we can now follow the money.

## The management fee: keeping the lights on

The management fee is the boring, beautiful, predictable revenue line. It is a fixed percentage of AUM, it arrives whether you are up or down, and its entire job is to **cover the cost of being in business** so that the manager is not forced to shut down during a bad stretch. It is the salary; the incentive fee is the bonus.

Because it is a flat percentage of AUM, the management fee scales *linearly* with the size of the fund. Double the AUM and you double the fee. There is no leverage in it, no convexity, no surprise — it is the most predictable cash flow in the entire business. Figure 2 shows the line: management-fee revenue at a 1.5% rate as AUM sweeps from USD 25M to USD 1B. It is a straight line through the origin, and the three points marked on it tell the whole story of the management-fee business at three different scales.

![A line chart showing management-fee revenue rising in a straight line with assets under management, marked at forty million, two hundred fifty million, and one billion dollars](/imgs/blogs/how-a-hedge-fund-makes-money-2.png)

Read off the points. At USD 40M, a 1.5% fee is USD 600,000 a year — Maya's situation, where the fee covers about half the bills. At USD 250M, it is USD 3.75M a year — enough to run a real operation comfortably. At USD 1B, it is USD 15M a year — a serious business *before you count a single dollar of performance*. That last figure is why critics complain that very large funds can get lazy: at a billion dollars, the management fee alone is a fortune, and a manager could in principle coast on it. (This is the heart of the misconception we tackle below, that "the management fee is profit." It is revenue, not profit, and at small scale it is barely revenue.)

The straight-line shape has a second, subtler consequence that every founder feels in year one. The fee scales with size, but **the cost of running a credible institutional fund does not scale down nearly as fast.** You still need an independent administrator, an annual audit, real compliance, and institutional-grade technology even at USD 40M — an allocator's ODD team will reject you without them, regardless of returns. So the small fund pays roughly the same fixed costs as the medium fund but earns a fraction of the fee. That gap between a linear fee and a sticky fixed cost base *is* the survival problem, and we will quantify it as the break-even AUM in a moment.

#### Worked example: the management fee on the Meridian Fund

Maya's Meridian Fund launches with USD 40 million of AUM at a 1.5% management fee. The annual management fee is:

| Step | Calculation | Result |
|---|---|---|
| Management fee = AUM × management rate | \$40,000,000 × 1.5% | **\$600,000 per year** |

That USD 600,000 is accrued monthly — about USD 50,000 a month — and pulled out of the fund's NAV before any performance is measured. It is hers (well, the management company's) regardless of whether Meridian is up or down. Now put it next to her cost stack of roughly USD 1.2 million a year, and the unglamorous truth of a small launch appears: the predictable revenue line covers half the bills, and the *other* half has to come from either the incentive fee (which may be zero) or her own pocket and any launch reserve she raised. *The management fee guarantees you can keep the doors open at a given size — but only if that size is large enough, and USD 40M is not.*

The other thing to know about the management fee is that it is **under permanent downward pressure**. The classic "2 and 20" was an industry norm in an era when hedge funds were scarce and exclusive. As the industry grew past USD 4 trillion and as allocators got more sophisticated and more numerous, the bargaining power shifted. Pension consultants and funds-of-funds now routinely negotiate the management fee down, especially for large early tickets, and the industry average drifted from 2% toward the 1.3–1.5% range over the 2010s and into the 2020s. A new manager often introduces a **founders share class** to lock in early capital — a discounted fee (say 1.5% / 15%, or even 1% / 10%) for investors who come in during the first year or above a size threshold, sometimes capped by a time window or an AUM ceiling on the class. The trade is explicit: you give up fee revenue per dollar in exchange for getting the dollars in the door sooner, which matters enormously when you are below break-even. We dig into the full terms negotiation in [fees and terms and the high-water mark](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark).

#### Worked example: the founders-class trade-off

Maya is at USD 40M and bleeding reserve. A funds-of-funds offers to write a USD 30M ticket — but only into a **founders share class** at 1% / 10% instead of her standard 1.5% / 20%, locked for two years. Is the discount worth it? Compare the management-fee revenue with and without the ticket:

| Scenario | Calculation | Management-fee revenue |
|---|---|---|
| Without the ticket (standard class only) | \$40,000,000 × 1.5% | \$600,000 per year |
| With the ticket — standard slice | \$40,000,000 × 1.5% | \$600,000 |
| With the ticket — founders slice | \$30,000,000 × 1.0% | \$300,000 |
| **With the ticket — total** | sum of the two slices | **\$900,000 per year** |

The founders-class dollars come in at a lower rate, but they are *additional* dollars. Her management-fee revenue jumps from USD 600k to USD 900k — a 50% increase — and her total AUM rises to USD 70M, much closer to the USD 80M break-even. The discount cost her USD 150k a year of fee she would have earned had that USD 30M come in at the full 1.5% (30M × 0.5% = 150k), but the alternative was not "the USD 30M at full freight" — it was *no USD 30M at all*. Below break-even, the founders class is almost always the right trade. *When you are sub-scale, the question is never "what fee can I charge" but "what gets the capital in the door before the reserve runs out" — and a discounted dollar beats a full-fee dollar you never receive.*

So the management fee is predictable, it scales straight-line with AUM, it covers the cost of staying alive, and it is slowly being negotiated down. It will not make you rich. That job belongs to the other line.

## The incentive fee: the real upside

If the management fee is the salary, the incentive fee is the entire reason to start a fund instead of taking a salaried PM seat at someone else's shop. It is a share — classically 20% — of the *profits* the fund generates for its investors. On a USD 250M fund that returns 12% in a year, the gross profit is USD 30 million, and the manager's 20% cut is USD 6 million. That single number can dwarf a year's worth of management fee. The incentive fee is where the wealth is.

But it has a shape that the management fee does not, and that shape is the most important thing in this section. The incentive fee is **one-sided**. It pays a percentage of gains and *nothing* on losses. You participate in your investors' upside and you do not refund them on the downside (beyond forgoing the fee). That is the payoff diagram of a **call option** — and in fact the academic literature treats the incentive fee as exactly that: the manager holds a free call option on the fund's performance, struck at the high-water mark. Figure 4 draws it.

![A payoff chart showing the incentive fee is zero for fund returns below the high-water mark and rises at twenty percent of the gain above it, with the plus twelve percent point earning a six million dollar fee on a two hundred fifty million dollar fund](/imgs/blogs/how-a-hedge-fund-makes-money-4.png)

The kink at zero is the whole point. To the left of the high-water mark — in the red region, where the fund is below its prior peak — the incentive fee is a flat line at zero. The manager earns the management fee and nothing else. To the right, in the green region, the line rises at a 20% slope: every extra dollar of gain above the mark hands the manager twenty cents. A long/short equity manager who has internalized [position sizing and risk of ruin](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) will recognize the danger immediately — a one-sided payoff *tempts* the holder to take more risk, because the downside of a big swing is borne mostly by the investors while the upside is shared. The high-water mark and the manager's own capital in the fund are the two main brakes on that temptation, which is why allocators care so much that the founder has meaningful personal money alongside the LPs.

### The high-water mark: why the option can pay nothing for years

The high-water mark is the rule that keeps investors from paying a performance fee twice on the same gains, and it is the single biggest reason a manager's incentive fee can vanish for an extended period. The mechanic is simple to state and brutal in effect: **the incentive fee is charged only on NAV per share above the previous peak.**

Imagine Meridian's NAV per share runs from USD 1,000 to USD 1,200 over a great first year. The manager charges 20% on the USD 200 of gain and the high-water mark is now USD 1,200. Year two is ugly and the NAV falls to USD 1,000 — back where it started. The investors have lost the entire year-two gain. Now year three: the fund climbs from USD 1,000 back to USD 1,150. The investors are *up* 15% on the year and feel like the manager earned a fee. But the high-water mark is still USD 1,200, and USD 1,150 is below it. **The incentive fee is zero.** The manager only starts charging again once the NAV clears USD 1,200. Between the USD 1,200 peak and the moment it is reclaimed, the manager runs the entire operation on the management fee alone — which, as we saw, may not cover the bills.

This is the part that founders underestimate. A drawdown does not just hurt your investors; it can erase your *largest* revenue line for a year or more, precisely when your cost stack is unchanged. A 20% drawdown requires a 25% gain to get back to par — because gaining back a loss is harder than losing it. (Lose 20% and you are at 0.80 of where you started; to get back to 1.00 you need to multiply by 1.25, a 25% gain.) That is the asymmetry every fund manager learns to fear; it is the same asymmetry that disciplines the whole business of [not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up).

### Crystallization: when the fee actually becomes cash

The incentive fee *accrues* continuously against NAV, but it **crystallizes** — turns into cash the manager can actually take out — on a schedule, usually annually (December 31 for most funds). Until it crystallizes, an accrued incentive fee is a number on the books that can still evaporate if the fund gives back gains before year-end. A manager who is up huge in October and gives most of it back by December crystallizes much less than the October paper figure. This is why a strong finish to the year matters so much to the *manager's* income, not just the investors': the crystallization date is when the option pays.

Some funds crystallize more often (quarterly) or use a **clawback** or deferral mechanism so the manager cannot bank a fee in a good year and walk away before a bad one. The direction of all of these refinements is the same: align the manager's payday with the *durable* gains the investors actually keep, not the transient ones. Allocators reward funds whose fee terms make that alignment tight.

### The hurdle rate: making the manager earn more than cash

The high-water mark protects investors from paying twice; the **hurdle rate** protects them from paying for returns they could have earned risk-free. A hurdle sets a *minimum* return — say the cash rate, a Treasury-bill benchmark, or a fixed 5% — below which no incentive fee applies, even on a gain above the high-water mark. The logic is that an investor should not pay a performance fee for a 3% year when a money-market fund would have paid 4% for no risk. The hurdle says, in effect, "you only share in the *excess* over what I could have earned sitting in cash."

Hurdles come in two flavors that make a real difference to the manager's cut. A **soft hurdle** means that once the fund clears the hurdle, the incentive fee applies to the *entire* gain (including the part below the hurdle); the hurdle is just a gate that must be passed. A **hard hurdle** means the incentive fee applies *only* to the gain *above* the hurdle, with the below-hurdle return excluded entirely. On a 10% year with a 5% hard hurdle, the manager charges 20% of only the top 5 percentage points; with a 5% soft hurdle, the manager clears the gate and charges 20% of the full 10%. That distinction is worth a lot of money over time, and it is one of the most-negotiated lines in the term sheet.

Many hedge funds — especially in equity strategies — have a high-water mark but *no* hurdle, on the argument that beating the broad market is the relevant bar, not beating cash. Credit and macro funds, and almost all private-equity-style vehicles, are more likely to carry an explicit hurdle. The thing to internalize for *this* post is that a hurdle is a second condition stacked on top of the high-water mark: the fund must be above its prior peak *and* above the hurdle return before a dollar of incentive fee is earned. Each condition makes the manager's option harder to get into the money, which is exactly why allocators push for both. We work through the full menu of hurdle structures in [fees and terms and the high-water mark](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark).

#### Worked example: the incentive fee on a 12% year, then minus costs

Meridian has grown. It now runs USD 250 million, returns 12% for the year, and starts the year at its high-water mark (so the entire gain is fee-eligible). Walk the cash flows:

| Line | Calculation | Amount |
|---|---|---|
| Gross gain | \$250,000,000 × 12% | \$30,000,000 |
| Incentive fee | 20% × \$30,000,000 | \$6,000,000 |
| Management fee | 1.5% × \$250,000,000 | \$3,750,000 |
| **Total fee revenue to the management company** | incentive + management | **\$9,750,000** |
| Less fixed costs (illustrative) | — | −\$2,500,000 |
| **Management-company profit before partner tax** | revenue − costs | **\$7,250,000** |

(The cost stack grows with the fund — a USD 250M operation runs richer than Maya's USD 40M launch — so we use roughly USD 2.5M here rather than USD 1.2M.) The partners distribute that USD 7.25 million as the income of the business. Notice the composition: the incentive fee (USD 6M) is the *larger* of the two revenue lines and it is the one that turned a comfortable-but-modest fee business into a genuinely lucrative one. *In a good year the incentive fee is the wealth; in a flat year, as we are about to see, it disappears and the same business barely breaks even.*

#### Worked example: a flat year, where the incentive fee is zero

Same Meridian Fund, same USD 250M, but this year the fund is flat — up 0.3%, essentially unchanged. The high-water mark is not exceeded, so:

| Line | Calculation | Amount |
|---|---|---|
| Gross gain | flat year, ≈ 0 | ≈ \$0 |
| Incentive fee | 20% × \$0 | \$0 |
| Management fee | 1.5% × \$250,000,000 | \$3,750,000 |
| **Total fee revenue to the management company** | incentive + management | **\$3,750,000** |
| Less fixed costs (illustrative) | — | −\$2,500,000 |
| **Management-company profit** | revenue − costs | **\$1,250,000** |

The same fund, the same staff, the same desks — and the income dropped from USD 7.25 million to USD 1.25 million because the option paid nothing. The business did not break; the management fee kept it alive. But every founder needs to feel this in their bones before launch: **a year with no performance is a year with almost no profit, and you cannot budget on the incentive fee.** You budget on the management fee and treat the incentive fee as the bonus it is. *The fund survives a flat year on the boring fee; it gets rich on the exciting one.*

#### Worked example: the year-two effect of a high-water mark after a drawdown

Now the hardest case. Meridian runs USD 250M and has a high-water mark from a prior peak. This year the strategy struggles and the fund is **down 15%**. Next year it recovers strongly, **up 18%**. Two years, two very different stories for the manager:

**Year 1 (down 15%):**

| Line | Calculation | Amount |
|---|---|---|
| Gross gain | \$250,000,000 × −15% | −\$37,500,000 |
| Incentive fee | no fee on a loss | \$0 |
| Management fee | 1.5% × ≈\$232,000,000 | ≈\$3,480,000 |

The high-water mark now sits 15% above the year-end NAV. **Year 2 (up 18% off the lower base):** the start-of-year NAV is 0.85 of the old peak, and 0.85 × 1.18 = 1.003, so the NAV ends only about 0.3% above the old peak. The fee-eligible gain above the high-water mark is just that sliver above the old peak.

| Line | Calculation | Amount |
|---|---|---|
| Fee-eligible gain | only the sliver above the old peak | small |
| Incentive fee on that sliver | 20% × the sliver | ≈ \$0, near zero |
| Management fee | 1.5% × ≈\$270,000,000 | ≈\$4,050,000 |

Here is the cruelty of the high-water mark. The fund delivered a *great* 18% in year two — investors who stayed are thrilled — and the manager still earns almost no incentive fee, because almost the entire 18% was spent climbing back to the prior peak rather than making new high ground. The investors got most of their money back; the manager got mostly the management fee for two straight years. *A drawdown does not just cost you one bad year of incentive fee; it can cost you the recovery year too, because the high-water mark makes you climb back over the old peak for free.* This is why a deep drawdown is an existential event for a small fund's *business*, not only for its returns, and why so many funds that suffer one quietly wind down rather than grind back over the mark for free — the topic of [surviving the J-curve and break-even AUM](/blog/trading/hedge-funds/surviving-the-j-curve-break-even-aum).

## Where every dollar goes

We have the two revenue lines. Now follow the money the rest of the way, because revenue is not income — the cost stack stands between them. Figure 3 traces a dollar of fees from the fund, into the management company, through the costs, and out to the partners.

![A pipeline diagram tracing a dollar of fees from the fund out to the management company, through the fixed cost stack, leaving a residual that the founding partners distribute as profit](/imgs/blogs/how-a-hedge-fund-makes-money-3.png)

A dollar of fees is *earned by the fund's investors' capital*, *charged to the fund*, *received by the management company*, *spent on the cost stack*, and only then is the residual *distributed to the partners*. The order matters: costs come out first. In a small fund the residual can be negative — the costs exceed the fees — and the founders fund the shortfall out of reserves or their own pockets. Here is the cost stack, roughly in order of size, with illustrative US figures for an institutional-grade small fund (these vary widely with strategy and complexity; treat them as ranges, not gospel):

- **Compensation** is almost always the biggest line. Salaries for the investment team, operations, and any business-development hire, plus the founders' own draw. Even a lean three-person shop is paying real salaries, and talent is mobile — a founder who cannot pay competitively watches her best analyst get poached by a pod shop that can (see [how quant firms make money](/blog/trading/quant-careers/how-quant-firms-actually-make-money) and [quant compensation demystified](/blog/trading/quant-careers/quant-compensation-demystified) for what that market pays).
- **Administration.** The fund administrator independently strikes the NAV, processes subscriptions and redemptions, and keeps the books — roughly USD 30k–100k+ a year for a small fund. This is non-negotiable: an allocator's ODD team will reject a fund that prices its own book, because self-administration is one of the canonical fraud red flags (it was central to Madoff).
- **Audit.** An annual financial-statement audit by a recognized firm — roughly USD 30k–60k. Again, independent and non-negotiable for institutional capital.
- **Legal and compliance.** Outside counsel for fund documents and ongoing matters, plus a compliance program (often outsourced to a consultant for a small fund) — commonly USD 50k–150k a year on top of the one-time USD 50k–150k+ formation bill for the LP, the offshore feeder, the PPM, the LPA, and the investment-management agreement.
- **Technology, market data, and execution.** The order-management system, the risk system, market-data feeds, a Bloomberg terminal or two — this line runs from USD 50k to USD 250k+ depending on how data- and tech-heavy the strategy is. A systematic fund's tech bill dwarfs a discretionary stock-picker's.
- **Prime brokerage and operations.** The prime broker provides financing, margin, securities lending so you can short, custody, and capital introduction. Sub-USD 100M funds often start with a mini-prime or introducing broker because the bulge-bracket primes set minimums. The PB does not usually charge a flat fee — it earns on financing spreads and stock-loan — but it shapes your economics and your counterparty risk, which after Lehman in 2008 is why most serious funds run multi-prime.
- **Office, insurance, and the rest.** Rent, D&O insurance, BCP and cyber, travel for the capital-raising road show. Smaller lines individually, real money together.

Add it up and an institutional-grade small fund commonly carries **USD 0.5M to 2M+ a year** in fixed costs. The word that matters is *fixed*: most of these costs do not fall when your AUM falls or your returns go flat. That is what makes the break-even calculation the most important survival number in the business.

#### Worked example: the break-even AUM

What AUM does Maya need so that the management fee *alone* covers her fixed costs — so the fund survives a flat year without the incentive fee and without burning her reserve? The formula is just the cost stack divided by the management-fee rate:

| Step | Calculation | Result |
|---|---|---|
| Break-even AUM = fixed costs ÷ management-fee rate | \$1,200,000 ÷ 1.5% | — |
| Same thing, rate as a decimal | \$1,200,000 ÷ 0.015 | **\$80,000,000** |

Maya needs roughly **USD 80 million of AUM** for the 1.5% management fee to cover her USD 1.2M of fixed costs. She launched at USD 40M — half of break-even. Figure 6 draws it: the rising management-fee revenue line crosses the flat fixed-cost line at USD 80M, splitting the world into a red sub-scale zone where the fee cannot cover costs and a green zone above where it can. If she had negotiated a 2% fee instead of 1.5%, break-even would fall to USD 60M (1.2M ÷ 0.02); if her costs were leaner at USD 0.9M, it would fall to USD 60M at the 1.5% rate. *The break-even AUM is the line every founder is racing to cross before the launch reserve runs out, and below it the management fee is a promise the business cannot yet keep.*

![A line chart showing management-fee revenue crossing the flat fixed-cost run-rate at about eighty million dollars of assets, with a red loss zone below and a green profit zone above](/imgs/blogs/how-a-hedge-fund-makes-money-6.png)

This is also where the famous "USD 100M threshold" comes from. USD 100M is the rough level the industry treats as institutional viability — comfortably above the break-even for a typical cost stack, and the AUM at which an SEC adviser registration is generally required (regulatory AUM at or above USD 100M, below which you register with the state). Many large allocators and their consultants will not even look at a fund below USD 250M–500M, regardless of how good the returns are, because the operational risk of a sub-scale fund is too high and the ticket they want to write would be too large a share of your book. The gap between where a fund can break even (USD 80M-ish) and where institutional capital will engage (USD 250M+) is the no-man's-land that kills a lot of promising launches.

## The asset you own: the management company

Now we can answer the question Maya got wrong on launch morning. She thought she owned a hedge fund. She owns a **management company**, and understanding the difference is the difference between thinking like an employee and thinking like a founder.

Figure 5 lays out the three entities and the direction the money moves. The limited partners put capital into the fund. The general partner controls the fund and bears the liability. The fund holds the capital and the positions — but the fund *pays* fees; it does not earn them. Those fees flow into the management company, which is owned by the founder and the team. The management company is the only one of the three boxes that *earns* anything. It is the business.

![A graph showing limited partners and the general partner connected to the fund, the fund paying fees that flow to the management company, which is owned by the founder and team who take the profit](/imgs/blogs/how-a-hedge-fund-makes-money-5.png)

Why does this distinction have teeth? Because the management company is the thing with **enterprise value** — the thing someone will pay you a multiple of earnings to own a slice of, the thing you can sell, the thing that survives even if a particular fund vintage closes. The fund's assets belong to the LPs; you never owned them. What you built is a *revenue engine* that earns 1.5% of those assets plus 20% of their gains, and that engine has a market price.

This is exactly what **GP-stake buyers** value. Firms like Goldman Sachs's Petershill, Blackstone's Strategic Capital, Investcorp, and others have built a whole business out of buying minority stakes in the *management companies* of established alternative managers. They are not buying the funds; they are buying a perpetual share of the management-company revenue — a slice of that 1.5%-plus-20% engine. When such a buyer values a hedge-fund management company, they are essentially capitalizing the two revenue lines: the stable, predictable management-fee stream (which they value at a higher multiple because it is recurring and low-volatility) and the lumpy, cyclical incentive-fee stream (valued at a lower multiple because it is performance-dependent and can be zero). A management company whose revenue is mostly stable management fee on sticky, locked-up capital is worth more *per dollar of revenue* than one whose revenue is mostly volatile performance fee — even at the same total revenue. We work through the actual mechanics and multiples of these deals in [capital structure, evolution, and GP stakes](/blog/trading/hedge-funds/capital-structure-evolution-and-gp-stakes).

To make this concrete: consider a management company earning USD 7.5M of management fee and, on average, USD 5M of incentive fee a year, on USD 500M of locked-up AUM. A GP-stakes buyer will not slap one multiple on the USD 12.5M total. They will capitalize the two streams separately — perhaps a higher multiple on the recurring USD 7.5M management fee (it is contractual and predictable) and a markedly lower multiple on the USD 5M incentive fee (it is performance-dependent and can be zero next year). The same buyer valuing a different USD 12.5M-revenue firm whose *mix* is USD 4M management fee and USD 8.5M incentive fee would pay *less* for the whole thing, because the bulk of the revenue is the volatile kind. This is why two managers with identical total revenue can have very different enterprise values, and why the founder who wants a valuable, sellable business obsesses over fee *mix* and capital stickiness, not just this year's return. (The exact multiples and deal structures live in the GP-stakes post; the point here is the principle: stable revenue is worth more per dollar than volatile revenue.)

This reframes the founder's strategic problem. Yes, you need returns — without them the fund dies and the management company is worthless. But the *value* of the business you own is maximized by growing **stable, sticky management-fee revenue** on a large, locked-up asset base, not just by posting a great year. That is why founders chase AUM and longer lock-ups so hard once they are past survival: it is not greed for the management fee for its own sake, it is that recurring fee on sticky capital is what makes the management company a valuable, sellable, durable asset rather than a year-to-year bet on performance. The founder who understands this manages the *business* — capital stickiness, fee mix, cost discipline — as carefully as she manages the *portfolio*.

## Common misconceptions

The way a hedge fund makes money is so widely misunderstood that the misconceptions are worth naming and killing one by one. Each of these is a mistake an outsider — and a surprising number of insiders — actually makes.

**"The fund is the business you own."** No. You own the management company. The fund holds the investors' capital and pays you fees; the management company collects them and is the entity with enterprise value. A GP-stakes buyer purchases a slice of your management company, never your fund. Confusing the two is the most fundamental error a founder can make, and it leads directly to the next one.

**"The management fee is profit."** The management fee is *revenue*, and at small scale it is barely that. Costs — comp, admin, audit, legal, tech, prime — come out first, and they are largely fixed. A USD 40M fund's USD 600k management fee does not cover a USD 1.2M cost stack; the "profit" is negative. Profit is what is left after the cost stack, and below break-even there isn't any. The management fee becomes meaningful profit only well above the break-even AUM.

**"20% means you keep a fifth of everything."** You keep a fifth of the *gains above the high-water mark*, not a fifth of the assets and not a fifth of every up move. On a USD 250M fund up 12%, the incentive fee is 20% of the USD 30M gain — USD 6M — which is 2.4% of AUM, not 20% of it. And after a drawdown you may keep a fifth of *nothing* for a year or more while you climb back over the mark. The 20% is a slice of a sometimes-empty pie.

**"The incentive fee always pays."** It is an option that pays exactly zero whenever the fund is below its high-water mark — in a flat year, a down year, and the entire recovery up to the prior peak. A manager can run a perfectly good operation for two or three years and earn almost no incentive fee because the fund spent that whole time underwater or climbing back to even. The incentive fee is the bonus, never the budget.

**"A great year makes you rich; you can budget on performance fees."** You cannot budget on the incentive fee, because you cannot predict it and it can be zero. Prudent founders budget the *business* on the management fee — the predictable line — and treat every dollar of incentive fee as upside to be partly reinvested and partly saved against the inevitable flat or down year. The funds that die are usually the ones that staffed up and spent against a great year's performance fee and then could not cover the costs when the next year was flat.

**"2 and 20 is still the standard, so the headline fees are huge."** The headline compressed years ago. Industry averages as of the mid-2020s sit closer to 1.3–1.5% management and 15–17% performance, with founders share classes discounting further to win early capital. Fee pressure is a permanent feature of a USD 4-trillion-plus industry with sophisticated, numerous allocators. The economics of a new launch are tighter than the "2 and 20" cliché suggests.

## How it plays out in the real world

The abstractions above are visible in the shape of the actual industry. A few real patterns, with the numbers flagged as the estimates they are.

**Most funds are small, and the giants hold almost everything.** The global hedge-fund industry manages roughly USD 4.5–5.0 trillion as of 2025–2026 (HFR, Preqin, BarclayHedge — estimates vary by source and methodology), across something like 8,000–10,000 funds. But the distribution is extraordinarily skewed: the "billion-dollar club" of funds above USD 1B controls the large majority of the assets — on the order of 90% — while the median fund runs far less than USD 100M. That skew is the management-fee math made flesh. The billion-dollar manager earns USD 15M+ a year in management fee alone and runs a comfortable, durable business; the median manager is fighting to reach break-even. The same fee terms produce two utterly different businesses depending only on which side of the AUM divide you land on — exactly the before-and-after of Figure 7.

![A side-by-side comparison of a sub-scale forty million dollar fund where the management fee barely covers costs and partners net a loss, versus a scaled five hundred million dollar fund where the incentive fee dominates and partners take home millions](/imgs/blogs/how-a-hedge-fund-makes-money-7.png)

**The before-and-after of scale.** Figure 7 puts Maya's launch beside the fund she is trying to become. On the left, the USD 40M Meridian: a USD 600k management fee against a USD 1.2M cost stack, no incentive fee while underwater, partners netting a loss. On the right, a USD 500M version of the same fund returning 12%: a USD 7.5M management fee, roughly USD 12M of incentive fee in a good year, partners taking home many millions after costs. *Identical strategy, identical fee terms — the only thing that changed is the AUM, and it changed everything about the business.* The entire founder's game is to get from the left panel to the right panel before the cash runs out, which is why the early years are a J-curve of burning reserve while you grind toward scale.

**Launches and liquidations run roughly in balance.** HFR's quarterly reports show, in recent years, hundreds of launches and hundreds of liquidations per quarter industry-wide — roughly "launches ≈ liquidations," sometimes net-negative. And the mortality is real: something like 7–11% of hedge funds liquidate in a given year (spiking in 2008 and 2020), and a large fraction of launches do not survive three to five years. The proximate cause is usually not a single catastrophic trade; it is the slow bleed of a sub-scale fund that never crossed break-even, gave back a year to a drawdown, watched the high-water mark turn the recovery into free work, and quietly wound down when the founder's reserve ran out. The economics in this post *are* the survival statistics.

**The fee terms are negotiated, not posted.** In practice a new manager rarely charges a clean 2/20. A funds-of-funds writing an early ticket negotiates a founders class at 1.5/15; a large pension demands a most-favored-nation clause and a fee break for size; a seeder takes a discount plus a slice of the management-company economics in exchange for day-one capital. Every one of those negotiations is a trade between *fee revenue per dollar* and *getting the dollars sooner* — and below break-even, sooner usually wins. The headline "2 and 20" is a starting point in a negotiation, not a fact about what the manager actually earns.

**The investor sees the returns net of both fees — and the gap is large.** When a fund reports "we returned 12%," that is usually the *gross* number, before fees. The investor's *net* return is what is left after the management fee and the incentive fee are deducted from NAV. On a 12% gross year at 1.5/20, the investor pays roughly 1.5% management fee plus 20% of the gain — so the net return lands closer to 8%, and the manager's two fee lines captured something like a third of the gross gain. This is not a small wedge, and it is exactly why allocators negotiate so hard and why the SEC's Marketing Rule (Rule 206(4)-1, in force since late 2022) requires a fund advertising performance to show *net-of-fee* numbers alongside gross. For the founder, the lesson is that your fees are a visible, scrutinized cost to your investors; the better your returns, the more fee they will tolerate, and the relationship between gross alpha and what you are allowed to charge is the quiet governor on the whole business. A manager whose net-of-fee returns do not beat a cheap index will not keep the capital that makes the fees worth having.

**When the option-like incentive fee bites the manager.** Funds that suffer a deep drawdown face a stark choice created entirely by the high-water mark. After Amaranth's roughly USD 6.6 billion loss in 2006, or in the many quieter cases of a fund down 30–40%, the manager faces years of running the operation on the management fee alone while the team's incentive fee is zero until the NAV claws back over the old peak. The rational team often leaves for a shop where their option is struck at-the-money again, and the fund frequently shuts down and re-launches a clean vehicle — a "high-water-mark reset" by another name. The incentive fee's one-sided payoff, so attractive in a good year, becomes the thing that empties the building in a bad one. This is the human face of the year-two worked example above.

## When this matters / Further reading

Understand the two revenue lines and the two entities and you understand the entire business you are about to start — not the headlines, the actual cash flows. The management fee is the predictable slice of AUM that keeps the lights on, scales straight-line with size, and is under permanent downward pressure. The incentive fee is the option-like 20% of profits above the high-water mark that builds real wealth, pays nothing while you are underwater, and crystallizes only on schedule. The costs come out first, so revenue is not income, and below roughly USD 80M of AUM there is no income at all. And the asset you own is not the fund — it is the management company, the revenue engine a GP-stakes buyer would actually pay a multiple to own.

This matters most at three moments in a founder's life. First, **before launch**, when you build the budget: budget on the management fee, treat the incentive fee as upside, and know your break-even AUM cold. Second, **during a drawdown**, when the high-water mark turns your recovery into free work and you have to decide whether the business can survive the climb back over the mark. Third, **when you grow or sell**, when the value of the management company — driven by stable, sticky management-fee revenue on a large locked-up base — becomes the prize. A founder who manages the business with the same discipline she brings to the portfolio is the one who is still standing in five years.

**Further reading in this series:**

- [What a hedge fund actually is](/blog/trading/hedge-funds/what-a-hedge-fund-actually-is) — the structure these fees flow through: the LP, the offshore feeder, the GP, and the service-provider stack.
- [Fees and terms and the high-water mark](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark) — the full terms negotiation: hurdles, crystallization, founders classes, clawbacks, and the most-favored-nation clause.
- [Surviving the J-curve and break-even AUM](/blog/trading/hedge-funds/surviving-the-j-curve-break-even-aum) — the survival math of the early years, when you burn reserve grinding toward the AUM where the fees finally exceed the costs.
- [Capital structure, evolution, and GP stakes](/blog/trading/hedge-funds/capital-structure-evolution-and-gp-stakes) — how buyers value the management company and what selling a slice of your revenue actually means.

**Further reading across the blog:**

- [Quant compensation demystified](/blog/trading/quant-careers/quant-compensation-demystified) — what the talent you must pay actually earns, and why comp is the largest line in your cost stack.
- [How quant firms actually make money](/blog/trading/quant-careers/how-quant-firms-actually-make-money) — the same fee-and-cost lens applied across the firm landscape you are about to compete in.
- [Risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up) — why the one-sided incentive fee makes risk control a business problem, not just a portfolio problem.
- [Position sizing and risk of ruin](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — the asymmetry of a drawdown, which the high-water mark turns into the manager's problem too.
