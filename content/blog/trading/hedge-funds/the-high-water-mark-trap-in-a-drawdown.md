---
title: "The high-water-mark trap: why a drawdown can starve the firm exactly when it needs cash"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The incentive fee is an option that pays zero below the high-water mark, so a drawdown shuts off performance-fee cash at the exact moment a fund most needs to retain talent and defend the book. This post traces the recovery math, the talent death spiral, and the three brutal choices underwater: grind back, reset the mark, or close."
tags: ["hedge-funds", "fund-management", "asset-management", "high-water-mark", "drawdown", "incentive-fee", "talent-retention", "fund-survival", "fund-economics", "recovery-math"]
category: "trading"
subcategory: "Hedge Funds"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The incentive fee is an option you wrote that pays zero below the high-water mark, so a drawdown shuts off your performance-fee cash at the exact moment you most need it to keep your team and defend the book.
>
> - Below the high-water mark you earn **zero incentive fee** on every dollar of recovery — and the climb back is **convex**: down 20% needs **+25%**, down 25% needs **+33%**, down 50% needs **+100%**.
> - At a steady 8% net, clawing back a 25% drawdown takes roughly **3.7 years** of zero performance fee while the bills run unchanged — that is the trap.
> - With no incentive-fee cash, the bonus pool empties, the best trader gets poached, returns weaken, and the drawdown deepens: a self-reinforcing **death spiral** that a cash buffer is built to break.
> - Underwater you have three real options — **grind back** (keeps the promise, starves the firm), **reset the mark** (restores cash, breaks investor trust), or **close** — and the one number to remember is that a 25% drawdown means **zero incentive fee until +33%**.

By the spring of the Meridian Fund's fourth year, Maya was down 22% from her old peak and her best trader was on the phone with a recruiter. She knew because he had told her — honestly, almost apologetically — that a multi-manager platform had offered him a guaranteed two-year package worth more than he had made at Meridian in three. He did not want to leave. He believed in the book. But he had a mortgage and two kids and a number in front of him that Maya, looking at her own financials, could not come close to matching. Her incentive fee for the year was going to be exactly \$0, because the fund's net asset value per share was still well below its old high-water mark, and the documents she had signed three years earlier said — in language she had read a hundred times and only now truly felt — that until the NAV climbed back above that peak, every dollar of recovery belonged to her investors, fee-free.

The cruelty of it was geometric. She had earned a large incentive fee in years one and two, crystallized it, and watched the high-water mark step up to the new peak. Then year three brought a 22% drawdown, and the mark stayed frozen at the top while the fund fell beneath it. Now, to pay the retention bonus that would keep her best trader, she needed performance-fee cash. To earn performance-fee cash, she needed to climb back above the mark. To climb back above the mark from 22% down, she needed a gain of roughly 28% — and at the kind of returns a battered book actually produces, that would take *years*, not months. The fee she needed to retain the talent who would *produce* the recovery was locked behind the recovery itself. She was, in the most literal sense, being asked to bootstrap her way out of a hole with a shovel she could only buy once she was already out.

This is the high-water-mark trap, and it is the single most underappreciated failure mode in the business of running a fund. It is not a market risk or a strategy risk; it is a *structural* risk baked into the fee contract itself. The high-water mark is a beautiful piece of investor protection — it guarantees an investor never pays a performance fee twice on the same dollar — and it is, from the founder's side of the table, an option that pays zero across an entire region of outcomes. When you are underwater, that region is exactly where you live. Figure 1 is the whole trap on one page: the NAV path, the frozen mark, and the shaded years of zero fee. The rest of this post traces the mechanism, the math, and the choices it forces.

![A four-year NAV per share path that peaks at the high-water mark, falls about 22% into a shaded underwater region of zero incentive fee, then grinds back above the mark roughly thirty-six months later](/imgs/blogs/the-high-water-mark-trap-in-a-drawdown-1.png)

This post builds directly on the fee mechanics laid out in [fees and terms: the high-water mark](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark). There we defined the management fee, the incentive fee, the high-water mark, the hurdle, and crystallization, and traced how a good year can still pay you nothing. Here we take the *drawdown* seriously as a business event — what it does to your cash, your team, your investors, and your range of choices — because a fund rarely dies from the down year itself. It dies from what the down year does to the *firm* while the fee is switched off.

## Foundations: the incentive fee as an option

Before the trap, the vocabulary. We will define each term once, precisely, and then never hand-wave it. If you have read the fees-and-terms post, this is a fast recap; if you have not, this section makes the post self-contained.

**Net asset value (NAV) and NAV per share.** The NAV is the total value of the fund's assets minus its liabilities. Divide by the number of shares (or units, or interests) outstanding and you get the **NAV per share** — the price of one unit of the fund. If a share launched at \$1,000 and is \$1,150 a year later, the fund is up 15% before fees. The fund's independent administrator strikes this number on a schedule (monthly for most hedge funds), and every fee, subscription, and redemption is computed against it. The high-water mark lives at the *per-share* level, which is where most confusion starts.

**Assets under management (AUM).** The total capital the manager is responsible for — investor money plus accumulated gains, minus losses and redemptions. AUM is the base the management fee is charged on. It is NAV per share multiplied by the number of shares, so it moves with both performance and flows. A drawdown shrinks AUM through the price, and redemptions shrink it through the share count — and both shrink the management fee.

**The management fee.** A fixed annual percentage of AUM, accrued and deducted from NAV regardless of performance — historically 2%, compressed toward roughly 1.3–1.5% by 2026 (HFR fee data). It is the salary, not the bonus: predictable, performance-blind, and — crucially for this post — the *only* revenue line that survives a drawdown. We will use 1.5% in the examples.

**The incentive fee (or performance fee).** A percentage — classically 20%, now closer to 15–17% on average — of the fund's profit, charged only on gains, and only on gains *above the high-water mark*. In a great year it is most of what a founder earns; in a down or flat-while-underwater year it is zero. We use 20% as the mental anchor because it makes the option shape clearest.

**The high-water mark (HWM).** The previous peak NAV per share that the fund must exceed before the manager can charge another incentive fee. Hit \$1,280, crystallize the fee, then fall to \$1,000, and you cannot charge an incentive fee again until the NAV climbs back above \$1,280 — even though, on the way up from \$1,000, your investors are making real money. The mark exists so an investor never pays a performance fee twice on the same dollar of gain.

**Underwater.** The state of being below the high-water mark. While underwater, the incentive fee is zero no matter how positive the year's return is, because the year's gains are clawing back ground the investor already paid for once. A fund can be *up 8% for the year and underwater at the same time* — that is precisely the situation that pays \$0.

**Drawdown.** The peak-to-trough decline in NAV per share, measured from the high-water mark. A 22% drawdown means the NAV sits 22% below its prior peak. The drawdown is the *depth* of the hole; the gain-to-recover is the *climb* out, and as we will see those two numbers are not the same — the climb is always larger.

**Crystallization.** The moment the accrued incentive fee stops being a bookkeeping estimate and becomes a real, locked-in amount paid to the management company — and the moment the high-water mark resets to the new peak. Almost always annual (typically December 31). A redeeming investor crystallizes their share of the fee on the way out.

**High-water-mark reset (or HWM reset).** A deliberate, negotiated lowering of the mark — usually after a deep drawdown — so the manager can begin charging an incentive fee again before the NAV has fully recovered its old peak. It is the most controversial term in the business because it shifts the cost of the recovery from the manager back onto investors who already paid for the lost ground once. We give it a full section.

That is the vocabulary. Now the part that matters most for everything below: the incentive fee is not just *like* an option — it *is* one, in the precise financial sense, and treating it as one explains every dynamic in this post.

## The high-water mark as an option

Write down the manager's incentive-fee payoff as a function of the fund's NAV per share, and the shape is unmistakable. Let the high-water mark be \$H and the current NAV per share be \$S. The incentive fee the manager collects at crystallization is:

| Region | NAV per share | Incentive fee payoff |
|---|---|---|
| Underwater | S ≤ H | 0 |
| Above the mark | S > H | rate × (S − H) × shares |

That is a **call option** written on the fund's own NAV, struck at the high-water mark \$H, with the manager holding the long position. Below the strike the payoff is flat at zero; above the strike the payoff rises linearly at the incentive-fee rate. The high-water mark is the strike. The fund's NAV is the underlying. The incentive-fee rate is the *participation* in the upside. A founder who has traded options for a living already knows everything they need to know about their own compensation — they have just never drawn it on the right axes.

Seeing it as an option immediately explains the trap, because options have a property that linear payoffs do not: **the payoff is zero across a whole region, and the further you are below the strike, the more the underlying has to move before the option pays anything at all.** A fund 5% underwater is a slightly out-of-the-money option; a small recovery puts it back in the money. A fund 40% underwater is *deeply* out of the money; it needs an enormous move before the option is worth a cent, and in the meantime the manager — the option holder — collects nothing while still paying the full cost of staying in business. The option does not expire, which is the one mercy, but an option you cannot exercise pays no rent, and rent is exactly what a fund's fixed costs demand every month.

There is a second, subtler consequence of the option framing that founders miss. Because the manager holds a call, the manager's *interests* are convex in the same way the payoff is: the manager has an incentive to take more risk when deep underwater, because only a large move recovers the option's value, and a small steady return — while best for the investor — leaves the manager unpaid for years. This is the moral-hazard reading of the high-water mark, and it is exactly why operational due diligence and risk committees watch a deeply underwater manager so closely. The fee structure that aligns the manager beautifully near the strike can *mis*-align them badly when the fund is far below it. We return to this when we discuss grinding back versus gambling.

#### Worked example: the option payoff on the Meridian Fund

Maya's Meridian Fund manages \$200M at a 1.5% / 20% structure with a high-water mark and no hurdle, on 200,000 shares struck at \$1,000. Her peak NAV per share — the current high-water mark — is \$1,280, set at the end of year two. Track the incentive fee as the NAV moves.

| NAV per share | Position vs mark | Incentive fee = 20% × (NAV − 1,280) × 200,000 shares |
|---|---|---|
| \$1,000 (down 22%) | \$280 underwater | \$0 |
| \$1,150 (down 10%) | \$130 underwater | \$0 |
| \$1,280 (at the mark) | at the strike | \$0 |
| \$1,330 (new peak) | \$50 above | 20% × \$50 × 200,000 = **\$2,000,000** |
| \$1,400 (new peak) | \$120 above | 20% × \$120 × 200,000 = **\$4,800,000** |

Notice the cliff. From \$1,000 to \$1,280 — a 28% gain on the fund, a heroic year for any manager — the incentive fee is *flat at zero the entire way*. The fee only switches on at the strike, and then it rises fast: the first \$50 above the mark is worth \$2M to the management company. *The incentive fee is a call option struck at the high-water mark, so the entire 28% climb from down-22% back to the strike pays the manager nothing, and only the gains above the old peak put a dollar in the firm's pocket.*

One more property of the option framing is worth making explicit, because it is where the manager's option differs from a tradable one and where the trap gets its teeth. A market call option has *time value* — even when it is out of the money, it is worth something today, because there is time for the underlying to move before expiry, and the holder can sell that time value to someone else. The manager's incentive-fee option has no such liquidity. The manager cannot sell the option, cannot hedge it, and cannot borrow against its time value to make payroll this month. The option is worth a great deal in theory — a deeply underwater fund with a real edge holds an instrument with substantial expected value — but that value is locked up entirely in the future, unrealizable until the NAV clears the mark, while the firm's costs demand realized cash *now*. An asset that is valuable on paper and worthless in the bank account is precisely the kind of asset that starves a business, and that is the manager's position underwater: long a valuable, illiquid, unhedgeable option whose time value cannot be spent. The good news is that the option does not expire — a fund can stay open and keep the option alive for years. The bad news is that staying open *costs cash the option cannot provide*, so the founder is forced to fund the option's carry out of reserves, out of the shrinking management fee, or out of pocket. The whole survival problem reduces to one question: can the firm afford to hold its own option long enough for it to come back into the money?

## The drawdown trap

Now assemble the trap from its parts. The trap is not the drawdown. Drawdowns are normal; every honest manager has them, and a well-run fund survives them routinely. The trap is the *interaction* between three things that all hit at once when the fund goes underwater:

1. **The incentive fee goes to zero** — the option is out of the money — so the firm's largest, most variable revenue line switches off entirely.
2. **The management fee shrinks** — because the drawdown cut the asset base, and because some investors redeem, lowering AUM further — so even the predictable revenue line falls.
3. **The fixed costs do not move** — the administrator, the auditor, the compliance counsel, the market-data bill, the office, and above all the salaries are contractually or practically fixed, and they keep arriving every month.

Stack those together and the picture is stark: the moment the fund needs cash the most — to retain people, to defend the book, to ride out the storm without being a forced seller — is the exact moment the fee structure throws off the *least* cash. Revenue collapses precisely when the demand on it peaks. That is the trap, and Figure 1 shows where you live while it holds: in the shaded band below the mark, for as long as the recovery takes.

The phrase "starves the firm" is not rhetorical. Recall the spine of this whole series: a fund is a business that must keep its promise *credible*, its structure *sound*, and its capital *sticky*, long enough to compound. A drawdown attacks all three at once. Credibility takes a hit because the returns are bad. The structure is strained because the cash that pays for sound operations is shrinking. And the capital gets twitchy because investors who are underwater are exactly the investors most likely to redeem. The high-water mark is the mechanism that turns a *market* event — a bad year — into a *business* event that can kill an otherwise-viable firm.

#### Worked example: the cash shortfall against payroll in the drawdown

Maya's Meridian Fund peaked at \$200M of AUM (NAV per share \$1,280 on 200,000 shares, less rounding). A 22% drawdown takes the NAV per share to about \$1,000, and a wave of nervous investors redeems 15% of the shares at the next quarterly window. Trace the revenue and the costs through the down year.

| Line | Calculation | Result |
|---|---|---|
| AUM after the drawdown (before redemptions) | \$200M × (1 − 0.22) | \$156M |
| AUM after a 15% redemption | \$156M × (1 − 0.15) | \$132.6M |
| Management fee on the shrunken, average AUM (~\$144M blended) | \$144M × 1.5% | ~\$2.16M |
| Incentive fee (underwater the entire year) | 20% × \$0 fee-eligible profit | **\$0** |
| **Total revenue for the year** | \$2.16M + \$0 | **~\$2.16M** |
| Fixed costs (admin, audit, compliance, data, rent, base salaries) | institutional-grade run-rate | ~\$2.0M |
| Discretionary cash left for bonuses and retention | \$2.16M − \$2.0M | **~\$0.16M** |

At the peak, with a 1.5% fee on \$200M plus a large incentive fee, Maya had a comfortable seven-figure bonus pool. One down year later she has *roughly \$160,000* of slack — against a retention package for one trader that the rival platform has set above \$1M. The arithmetic does not work. *A 22% drawdown plus a 15% redemption cut the management fee by a third and the incentive fee to zero, so the bonus pool that retains talent collapses from millions to a rounding error in a single year.*

The redemption line deserves emphasis because it is the part founders underweight. A drawdown does not only cut the *price* (NAV per share); it cuts the *quantity* (shares outstanding), because underwater funds see net outflows. The two compound: a 22% price decline and a 15% redemption together take AUM from \$200M to about \$133M — a 34% cut to the base the management fee is charged on. The fund's only surviving revenue line shrinks by a third at the same moment the variable line goes to zero. This is why the [liquidity terms](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark) — lock-ups, notice periods, gates — are not investor-hostile fine print but survival infrastructure: they slow the quantity leg of the collapse long enough for the price leg to recover.

## The recovery math

Here is the piece of arithmetic that every founder should have tattooed somewhere visible: a drawdown and the gain needed to recover it are *not* the same number, and the gap between them widens fast. If you fall by a fraction \(d\) of your peak, the gain \(g\) you need to get back to the peak is:

$$ g = \frac{d}{1 - d} $$

The reason is mechanical. A loss is taken on the larger, pre-loss base; the recovery gain is earned on the smaller, post-loss base. Lose 50% of \$100 and you have \$50; to get back to \$100 you must *double* — earn 100% — because your 100% is now measured against the smaller \$50. The deeper the hole, the more the denominator shrinks, and the more violently the required gain accelerates. Figure 2 plots the curve against the naive 1-to-1 line a tired brain assumes; the gap between them is the convexity that bites.

![A convex curve showing the percentage gain required to recover a drawdown, rising far above a straight one-to-one reference line, with marked points at down 20 percent needs 25 percent, down 25 percent needs 33 percent, and down 50 percent needs 100 percent](/imgs/blogs/the-high-water-mark-trap-in-a-drawdown-2.png)

Run the headline cases:

| Drawdown \(d\) | Gain to recover \(g = d/(1-d)\) | The math |
|---|---|---|
| Down 10% | +11.1% | 0.10 ÷ 0.90 |
| Down 20% | +25% | 0.20 ÷ 0.80 |
| Down 25% | +33.3% | 0.25 ÷ 0.75 |
| Down 33% | +50% | 0.33 ÷ 0.67 |
| Down 50% | +100% | 0.50 ÷ 0.50 |
| Down 60% | +150% | 0.60 ÷ 0.40 |

Down 20% needs +25%. Down 50% needs +100%. Down 60% needs +150%. The convexity is why deep drawdowns are so often terminal: a fund down 50% must *double* before it earns a single dollar of incentive fee, and a manager who can reliably double a book would not have lost half of it in the first place. The recovery requirement runs away from the realistic return faster than any manager can chase it.

Two things make the recovery math worse in practice than this clean formula suggests. First, the management fee keeps being deducted *during* the recovery, so the fund must out-earn its own fee drag just to stand still relative to the mark — the gross return needed is a touch higher than the net figures above. Second, and far more important, the recovery is measured in *time*, and time is what the firm does not have, because every month underwater is a month of full costs against zero incentive fee.

#### Worked example: down 25% on a \$200M fund — the +33% and the zero-fee period

Take Meridian down exactly 25% from its \$1,280 peak. The NAV per share falls to \$960. To get back to the \$1,280 mark:

| Step | Calculation | Result |
|---|---|---|
| Drawdown depth | (1,280 − 960) ÷ 1,280 | 25% |
| Gain required to recover | 0.25 ÷ (1 − 0.25) | **+33.3%** |
| NAV per share at recovery | \$960 × 1.333 | \$1,280 (the old mark) |
| Incentive fee earned on the entire +33.3% climb | 20% × \$0 (all below the mark) | **\$0** |
| First dollar of incentive fee | only on NAV above \$1,280 | after full recovery |

The +33.3% is not the headline — the *\$0* is. Maya must produce a 33% gain, the kind of year that would normally crystallize a multi-million-dollar fee, and earn precisely nothing on it, because every dollar of that gain merely repays ground the investors already paid for once. *A 25% drawdown demands a 33% recovery, and the high-water mark means the manager works through that entire 33% year for zero incentive fee — a full, fee-free year of the firm's best possible performance.*

Now put the recovery on a clock, because the time dimension is where the trap actually kills firms. How long does +33% take? It depends on the net annual return the battered book can produce, and the relationship is brutal: at low returns the recovery stretches over many years, each one paying zero. Figure 6 shows it.

![A bar chart of the years required to recover a 25 percent drawdown at various annual net returns, from about ten years at three percent down to under two years at twenty percent, with the eight percent case highlighted at about three point seven years](/imgs/blogs/the-high-water-mark-trap-in-a-drawdown-6.png)

The number of years \(t\) to earn the +33.3% at a steady net return \(r\) is \(t = \ln(1.333) / \ln(1 + r)\):

| Net annual return while recovering | Years to climb +33% back to the mark |
|---|---|
| 3% | ~9.7 years |
| 5% | ~5.9 years |
| 8% | ~3.7 years |
| 10% | ~3.0 years |
| 12% | ~2.5 years |
| 15% | ~2.1 years |
| 20% | ~1.6 years |

At a realistic post-drawdown return of 8% net, recovering a 25% drawdown takes **about 3.7 years** — and every one of those years pays zero incentive fee while the firm's costs run unchanged. That is not a market problem the manager can trade their way out of; it is a *cash-runway* problem the manager must *fund* their way through. The fund that survives a deep drawdown is almost never the one with the best recovery returns. It is the one that arranged, before the drawdown, to have enough cash to pay its people through three or four fee-free years.

It is worth holding the time figures and the depth figures side by side, because their product is what actually kills firms. A shallow 10% drawdown needs +11% and, at 8% net, recovers in well under two years — survivable on the management fee for most funds that reached scale. A moderate 25% drawdown needs +33% and roughly 3.7 fee-free years — survivable only with reserves. A deep 50% drawdown needs +100% and, at 8% net, takes about *nine years* — a horizon over which essentially no standalone fund survives without resetting the mark or recapitalizing, because nine years of zero incentive fee against full costs exhausts any plausible reserve. The depth of the drawdown does not just set the size of the climb; it sets the *duration* of the cash drought, and duration is what runs the firm out of money. This is the deepest reason risk discipline is a survival function and not a performance preference: a manager who caps drawdowns at, say, 15% is buying a recovery measured in months and payable from the management fee, while a manager who lets a drawdown reach 40% has bought a recovery measured in many years and payable only from reserves they almost certainly do not have.

#### Worked example: a two-year recovery with zero incentive fee while costs continue

Suppose Maya does well after the 25% drawdown — she produces +15% net in year one of the recovery and +16% net in year two, which compounds to about +33% and just clears the old mark by the end of year two. A genuinely strong two years. Tally the firm's economics across them.

| Year | Net return | NAV per share | Above mark? | Incentive fee | Management fee (~\$135M avg) | Fixed costs | Net cash |
|---|---|---|---|---|---|---|---|
| Recovery yr 1 | +15% | \$960 → \$1,104 | no (below \$1,280) | \$0 | ~\$2.0M | ~\$2.0M | ~\$0 |
| Recovery yr 2 | +16% | \$1,104 → \$1,281 | just clears | ~20% × \$1 × 200k ≈ \$40k | ~\$2.0M | ~\$2.0M | ~\$40k |

Two excellent years — a 15% and a 16% — and the firm runs at roughly breakeven through both, earning a token \$40,000 of incentive fee only in the final month when the NAV finally pips the mark. *A founder can post two of the best years of her career and still take home almost no performance pay, because the high-water mark routes the entire recovery to the investors and leaves the firm running on the management fee alone for the duration.* The takeaway is not that the structure is unfair — it is doing exactly what it promised the investors — but that the founder must plan the firm's finances around the *possibility* of multiple fee-free years, or the firm dies during the very recovery that would have made it whole.

#### Worked example: the cumulative cash shortfall against a competitive bonus pool

The single-year shortfall understates the danger, because the drought is cumulative and the rival platforms reset their packages every year. Suppose Maya grinds through a four-year recovery from down-25% at a steady 8% net — the figure 6 case. Set the firm's fixed costs at \$2.0M a year, the management fee on a recovering ~\$140M average AUM at 1.5% (so ~\$2.1M), and the *competitive* bonus pool she would need to fully match outside offers at ~\$2.5M a year. Tally the four-year gap between what the firm can pay and what it would take to hold the whole team.

| Year | Mgmt fee | Incentive fee | Fixed costs | Cash left for bonuses | Competitive pool needed | Shortfall |
|---|---|---|---|---|---|---|
| 1 | ~\$2.1M | \$0 | \$2.0M | ~\$0.1M | \$2.5M | ~\$2.4M |
| 2 | ~\$2.1M | \$0 | \$2.0M | ~\$0.1M | \$2.5M | ~\$2.4M |
| 3 | ~\$2.1M | \$0 | \$2.0M | ~\$0.1M | \$2.5M | ~\$2.4M |
| 4 | ~\$2.1M | ~\$0.5M (clears mark late) | \$2.0M | ~\$0.6M | \$2.5M | ~\$1.9M |
| **Total** | — | ~\$0.5M | — | ~\$0.9M | \$10M | **~\$9.1M** |

Over the four-year grind the firm can fund roughly \$0.9M of bonuses against a roughly \$10M competitive pool — a cumulative shortfall near **\$9M** that the founder must either absorb from a reserve built in the good years, replace with deferred-equity comp that costs no current cash, or accept will cost her the people who leave. *The cost of the high-water-mark trap is not one missed bonus; it is the compounding multi-year gap between the cash a fee-starved firm can pay and the cash a competitor with no firm-wide mark can offer — a gap that runs to millions and that only a pre-funded reserve or non-cash equity can close.*

Figure 7 is the same point as a revenue stack: in a good year, three layers of cash pile up — the incentive fee, fresh subscriptions, and the management fee. Underwater, the top two vanish and the bottom one shrinks, and all that is left is the bills.

![A vertical stack showing that a drawdown removes the incentive fee layer and the new-subscriptions layer, shrinks the management fee layer because the asset base fell, and leaves only the fixed bills that do not pause](/imgs/blogs/the-high-water-mark-trap-in-a-drawdown-7.png)

## Retaining talent while underwater

A hedge fund is people. The strategy lives in heads, not in a building, and the moment those heads start to leave, the franchise begins to unwind. This is why the talent problem is the most dangerous expression of the high-water-mark trap — far more dangerous than the founder's own foregone pay, which a founder can absorb. The founder *chose* this risk; the analysts and traders did not, and they have a market value that does not care whether the fund is underwater.

Trace the dynamic and it is a textbook self-reinforcing loop, drawn in Figure 4. The drawdown zeroes the incentive fee. With no performance-fee cash, the bonus pool — which at most funds is funded almost entirely from the incentive fee — empties. A senior trader who delivered alpha all year sees a bonus near zero, through no fault of their own, while a rival multi-manager platform dangles a guaranteed package. The trader leaves. With the trader goes a slice of the book's edge, its risk capacity, and its institutional memory. Returns weaken. The drawdown deepens. The fund falls *further* below the mark, which makes the next year's incentive fee even less likely, which empties the next bonus pool, and the loop tightens. Each turn makes the next turn more likely. This is the death spiral, and it has ended more funds than any single bad trade.

![A seven-stage pipeline of the underwater death spiral, from drawdown to zero incentive fee to an empty bonus pool to a poached lead trader to a thinning team to weaker returns to a deepening drawdown that loops back](/imgs/blogs/the-high-water-mark-trap-in-a-drawdown-4.png)

The cruelty is that the loop attacks the *cause of the recovery*. The people who would dig the fund out are exactly the people the structure cannot pay to keep. A platform, by contrast, can offer guaranteed pay precisely because it does not run a single high-water mark across the whole firm — it pays each pod from its own crystallized P&L and absorbs the others' drawdowns at the center. The standalone fund has no such cross-subsidy. Its high-water mark is one mark for the whole vehicle, and when it is underwater, *everyone* is unpaid at once.

The timing of the poaching makes it worse. Rival platforms do not recruit at random; they recruit *into* a competitor's drawdown, because that is exactly when a talented PM is most reachable — underpaid through no fault of their own, watching a fee-free year stretch into a fee-free era, and rationally weighing a guaranteed package against a deferred bet on a recovery that may take years. The founder is therefore defending the team at the worst possible moment: cash at its thinnest, morale at its lowest, and the outside offer at its most aggressive. A retention plan improvised *during* the drawdown almost always fails, because the cash to fund it is precisely what the drawdown removed. The plans that work are the ones written into the firm's structure *before* the drawdown — the reserve policy, the deferral schedule, the equity grants — so that when the platform calls, the founder has something concrete to put on the table that does not depend on a fee that no longer exists.

The only defenses are arranged in advance:

- **A cash buffer at the management company.** The single most important survival reserve a founder can build. Retain a portion of the good-year incentive fee at the management company — do not distribute every dollar — so that in a drawdown you can pay retention bonuses out of accumulated reserves rather than out of a fee that no longer exists. Figure 3 contrasts the buffered firm and the fee-starved firm in the same down year. The difference between them is not skill; it is whether the founder saved.
- **Deferred and clawback-able comp** that vests over multiple years, so a trader who leaves mid-recovery forfeits unvested upside — this raises the cost of being poached and buys retention without current cash.
- **Phantom equity or a points-on-the-management-company structure**, giving key people a stake in the *firm's* long-run value rather than only the current year's bonus, so their incentive survives a zero-fee year.
- **Honest, frequent communication** that frames the recovery as a shared mission with a defined finish line (the mark), so the best people stay for the upside of being there when the fee switches back on.

![A before-and-after comparison of two firms in the same 22 percent drawdown, the fee-starved firm with an empty reserve losing its best PM into a death spiral, and the buffered firm paying retention from twelve to eighteen months of reserves and keeping the team](/imgs/blogs/the-high-water-mark-trap-in-a-drawdown-3.png)

#### Worked example: an HWM reset and what it costs investors

Maya is two years into a recovery from a 25% drawdown, the NAV per share has climbed from \$960 to \$1,150, and she is still \$130 below the \$1,280 mark. Her best two people are wavering. She approaches her largest investors with a proposal: reset the high-water mark from \$1,280 down to \$1,150 (the current NAV) so the firm can begin charging an incentive fee again on further gains, in exchange for cutting the incentive-fee *rate* from 20% to 10% until the old \$1,280 mark is genuinely reclaimed. What does the reset cost the investors?

| Item | Without reset | With reset (rate cut to 10% up to \$1,280) |
|---|---|---|
| Fee on the \$130 climb from \$1,150 to \$1,280 (per share) | \$0 (below old mark) | 10% × \$130 = \$13 per share |
| On 200,000 shares | \$0 | \$13 × 200,000 = **\$2,600,000** |
| Who pays | nobody — investors recover free | investors pay \$2.6M on ground they already paid for once |

The reset hands the management company roughly \$2.6M of incentive fee on a stretch of recovery the investors were contractually entitled to receive fee-free — a direct transfer from investors to manager, justified (if at all) only by the argument that without it the manager loses the team and nobody recovers anything. *A high-water-mark reset is a negotiated transfer of the recovery's economics from investors back to the manager, and even a softened version with a halved rate can cost investors millions on ground they already paid for — which is exactly why it is the most contentious term in the business.* We unpack the mechanism and the controversy next.

## Resetting the high-water mark

A high-water-mark reset — sometimes called a "modified high-water mark," a "high-water-mark holiday," or in its harshest form simply "lowering the strike" — is a deliberate change to the fee contract that lets the manager charge an incentive fee before the NAV has fully recovered its old peak. It is the manager's escape hatch from the trap, and it is loaded with controversy because every dollar it earns the manager is a dollar taken from investors on ground they already paid for once.

The mechanics come in a few flavors, ordered roughly from least to most investor-hostile:

- **A reset to the current NAV with a reduced rate.** The mark drops to today's NAV, but the incentive-fee rate is cut (say from 20% to 10%) until the old mark is genuinely reclaimed, so the manager shares the recovery rather than capturing all of it. This is the version Maya proposed above, and it is the most defensible.
- **A reset spread over time (amortized).** The old mark is not erased but lowered in steps over two or three years, so the manager earns a partial fee on the recovery while still owing investors most of the lost ground.
- **A reset tied to a new lock-up.** Investors accept a lowered mark only in exchange for the manager extending the lock-up, aligning the manager's renewed fee with a commitment of stickier capital.
- **A clean reset to the current NAV.** The old mark is simply erased and replaced by today's NAV. The manager charges a full incentive fee on the entire recovery. This is the version that draws the most fire, because the investor pays twice — once on the original gains that set the old mark, and again on the recovery back up to it.

Why would an investor ever agree? Only one argument holds water: the alternative is worse. If a clean refusal means the manager loses the team and either gambles the book to chase the option's value or closes the fund and returns capital at the bottom, an investor may rationally prefer a reset that keeps a competent, motivated manager working the recovery. The reset is, at best, a recognition that a manager who cannot pay their people will not recover the investor's capital either. The investor is buying *alignment and retention* with a slice of the recovery's economics.

But the controversy is real and the abuses are well documented. A clean reset can reward exactly the wrong behavior: a manager who takes a huge swing, blows up, and then resets the mark has used the high-water mark's own logic against the investor — heads I take 20%, tails I lower the strike and try again. This is the moral-hazard reading of the option we flagged earlier, made concrete. Sophisticated allocators and the operational-due-diligence process treat reset clauses as a serious governance question: who can trigger one, on what terms, with what investor consent, and with what disclosure. A fund whose documents let the manager unilaterally reset the mark is a fund whose alignment can be revoked at the manager's discretion — and that is precisely the kind of structural weakness an [operational due diligence](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark) review exists to catch. The honest framing for a founder is this: a reset is sometimes the lesser evil, but it spends trust that took years to build, and it should be the last lever pulled, negotiated transparently with the largest investors, never imposed.

## Grinding back vs closing

Strip away the euphemisms and a founder underwater has exactly three real options. Figure 5 lays them out against their consequences, and the honest truth is that none of them is good — they are three different ways of paying for the same hole.

![A three-by-three matrix of the options underwater, grind back, reset the mark, and close, against their effects on fee cash, on investors, and on the firm, with grind back aligning investors but starving the firm, reset restoring cash but breaking trust, and close ending the franchise](/imgs/blogs/the-high-water-mark-trap-in-a-drawdown-5.png)

**Grind back.** Keep the mark where it is, run the strategy, earn nothing on the recovery, and climb back to the surface on the management fee plus reserves. This keeps the promise to investors perfectly — they pay no fee on ground they already paid for — and it is the option that preserves the founder's reputation and franchise value if it works. But it starves the firm: it is the multi-year, zero-fee slog we costed above, and it only works if the founder built a cash buffer in advance and the team is willing to stay for deferred upside. The grind is the *right* answer when the drawdown is moderate, the buffer is real, and the strategy's edge is intact. It is the wrong answer when the drawdown is so deep that the recovery would take a decade — at which point the founder is asking the team to volunteer for years of below-market pay in pursuit of an option that may never pay off.

**Reset the mark.** Restore the firm's fee cash by lowering the strike, at the cost of investor trust, as dissected above. This is sometimes the lesser evil but should be the rarely-pulled lever, negotiated openly. A founder who resets is buying time and team retention with reputation; the question is always whether the reputation hit is survivable, and for a young fund still building a track record it often is not.

**Close.** Wind the fund down, return capital, and stop the bleed. This is the option founders resist longest and sometimes should choose soonest. If the drawdown is deep enough that the recovery is genuinely improbable, every additional month is a month of fixed costs burning the founder's own reserves and the investors' patience for no expected return. A clean, honest wind-down — returning capital promptly, crystallizing fees only on actual gains, communicating transparently — protects the founder's reputation for a *next* fund far better than a slow, grinding death that ends in forced liquidation. The founder taxonomy of [how funds die](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark) shows again and again that the firms that wound down cleanly produced founders who launched successful second funds, while the firms that ground on to the bitter end produced founders nobody would back again.

The decision among the three turns on three questions, answered honestly: How deep is the hole (and so how long is the realistic recovery)? How much runway does the firm have (reserves plus management fee minus costs)? And is the edge still real, or did the drawdown reveal that the strategy stopped working? A moderate drawdown, real reserves, and an intact edge point clearly to grinding back. A deep drawdown, thin reserves, and a strategy that has stopped working point — however painful — to closing. The reset sits uncomfortably in the middle, available only to a founder whose investor relationships are strong enough to survive the ask.

There is a darker fourth path that the option framing predicts and that every governance structure is built to prevent: **gamble for resurrection.** A deeply underwater manager holds a call option that only a large move can put back in the money, and the temptation is to take outsized risk — to swing for a return big enough to clear the mark in one year rather than grind for five. This is rational for the *option holder* and ruinous for the *investor*, who bears the downside of the gamble while the manager keeps the upside. It is the LTCM dynamic, the Amaranth dynamic, the pattern behind a long list of blow-ups: a manager far below the mark, reaching for a recovery that the prudent return cannot deliver, taking on the concentration and leverage that turn a recoverable drawdown into a terminal one. The healthy fund's risk committee and independent governance exist precisely to take this option off the table when the founder is most tempted to reach for it.

## Common misconceptions

The high-water mark generates more confused intuitions than any other term in the business. Here are the ones that cost founders real money.

**"A drawdown just means a year of no bonus — I'll make it back next year."** This treats the drawdown as a single bad year when it is a multi-year fee-free *period* whose length is set by the convex recovery math. Down 25% is not "skip one bonus"; it is roughly 3.7 years of zero incentive fee at an 8% recovery return, while the firm's full costs run the whole time. The founders who fail are the ones who budgeted for the down year and not for the long climb out of it.

**"The high-water mark protects me too, because it stops me from double-charging."** The high-water mark protects the *investor*; for the *manager* it is an option you wrote with a strike that ratchets up and never down. It costs you money in exactly the state where you can least afford it. It is investor protection, full stop — a feature of the contract you signed, not a feature that works in your favor.

**"If I have a great recovery year, the fee comes roaring back."** Only the part *above the old mark* pays. A 33% recovery year that climbs from down-25% exactly back to the mark pays *zero* incentive fee, because every dollar of it is below the old peak. The fee does not come back when your *returns* recover; it comes back when your *NAV* clears the mark — and those can be a year or more apart.

**"A high-water-mark reset is a normal, no-big-deal adjustment."** A reset is a transfer of the recovery's economics from investors back to the manager on ground the investors already paid for once. A clean reset can cost investors millions and is the most contentious term in the business; operational due diligence treats unilateral reset rights as a serious governance red flag. Treating it as routine is how a founder destroys the trust that took years to build.

**"The management fee will carry me through the drawdown."** The management fee shrinks during a drawdown — the NAV fell, cutting the price leg, and redemptions cut the share-count leg — so the only surviving revenue line is *smaller* exactly when it has to do all the work. On the Meridian example, a 22% drawdown plus a 15% redemption cut the management fee by a third. The fee that was designed to "keep the doors open" keeps them open only if the firm reached real scale before the drawdown hit.

**"Being underwater just lowers my pay; it doesn't threaten the firm."** Being underwater threatens the firm through the *team*, not the founder's wallet. The founder can absorb a lean year; the analysts and traders have a market value that does not care about the mark, and a rival can pay them guaranteed cash the underwater fund cannot match. The drawdown's real danger is the talent death spiral, and that is a firm-ending risk, not a pay cut.

## How it plays out in the real world

The high-water-mark trap is not a theoretical edge case; it is a visible pattern across the industry's recorded failures and survivals.

**The mortality data.** Roughly 7–11% of hedge funds liquidate in a given year (HFR), with spikes in stress years like 2008 and 2020, and a large fraction of *launches* do not survive three to five years. A meaningful share of those liquidations are not blow-ups in the dramatic sense — no fraud, no single catastrophic trade — but exactly the slow death this post describes: a fund that took a drawdown, went underwater, lost its people to better-paying rivals while the fee was switched off, and wound down before it could climb back. The trap is a quiet, common killer, not a rare one.

**The platform contrast.** The rise of the multi-manager platforms — the "pod shops" — is in large part a structural answer to the high-water-mark trap. A platform pays each portfolio manager on their *own* crystallized P&L and absorbs drawdowns at the center, which means a single PM's drawdown does not zero the whole firm's bonus pool, and the platform can offer the guaranteed packages that poach talent away from standalone funds underwater. This is exactly why the talent flow runs from drawn-down boutiques to platforms in a stress year: the platform's compensation structure does not have a single firm-wide high-water mark to trap it. The standalone founder competes against an org designed specifically to be immune to the trap that threatens them.

**The deep-drawdown blow-ups.** The famous failures — LTCM losing roughly \$4.6B in months on leveraged convergence trades, Amaranth losing about \$6.6B (around 65% of its assets) in a week on concentrated natural-gas bets — are, read through the option lens, partly stories of the gamble-for-resurrection dynamic. A manager deep below the mark holds an option that only a large move recovers, and the structures that should restrain that reach — independent risk, real governance, the discipline of not running a single concentrated book — failed. The high-water mark did not cause those blow-ups, but the incentive it creates when a fund is far underwater is part of why a recoverable drawdown becomes a terminal one. The lesson the survivors took is the [risk discipline of not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up): keep the drawdowns shallow enough that the recovery is plausible at *prudent* returns, so you are never tempted to reach.

**The survivors.** The funds that ride out deep drawdowns and come back share a profile: they reached real scale before the drawdown (so the management fee alone covered costs), they retained reserves at the management company in the good years (so they could pay retention out of savings), they had sticky capital with real lock-ups (so the redemption leg of the collapse was slowed), and they communicated honestly with investors about the recovery as a shared, finite mission. Note what is *not* on that list: a great recovery return. The recovery return matters, but it is downstream of survival, and survival is a *business* outcome arranged before the drawdown, not a *trading* outcome produced during it.

## When this matters / Further reading

The high-water-mark trap matters most at two moments, and a founder should think about it at both. The first is *at launch*, when the fee structure, the lock-up terms, and above all the management company's reserve policy are being set: the founder who plans for the possibility of multiple fee-free years — by building real scale before relying on the incentive fee, by retaining reserves in good years, by structuring deferred comp that survives a zero-fee year, and by negotiating lock-ups that slow redemptions — is the founder who survives the drawdown that eventually comes. The second is *in the drawdown itself*, when the three options come into focus and the founder must answer honestly how deep the hole is, how much runway remains, and whether the edge is still real — and then choose to grind, reset, or close before the death spiral chooses for them.

The single number to carry out of this post: a 25% drawdown means **zero incentive fee until +33%**, and at realistic returns that climb takes years. The high-water mark is a promise to investors that is genuinely good for them; it is also an option you wrote that pays nothing in the region where you most need cash. Run the firm so you can survive that region, because the market will eventually put you in it.

To go deeper, start with the term itself in [fees and terms: the high-water mark](/blog/trading/hedge-funds/fees-and-terms-the-high-water-mark), which builds the management fee, the incentive fee, the hurdle, and crystallization from zero. The cash-runway side of the trap — how much AUM and how much reserve you actually need to ride out a fee-free stretch — is the subject of [surviving the J-curve and break-even AUM](/blog/trading/hedge-funds/surviving-the-j-curve-break-even-aum). The investor side — how to keep capital sticky and communicate through a drawdown so redemptions do not compound the price decline — is [investor relations and retention](/blog/trading/hedge-funds/investor-relations-and-retention). The team side — who you hire, how you structure their comp, and how that comp survives a down year — is [building the team: who to hire and when](/blog/trading/hedge-funds/building-the-team-who-to-hire-and-when). And the endgame, when the trap wins, is mapped in [how hedge funds die: the failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy). For the broader discipline of keeping drawdowns shallow enough that recovery stays plausible, the careers-side treatment is [risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up).
