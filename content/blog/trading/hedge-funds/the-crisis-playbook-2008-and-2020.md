---
title: "The crisis playbook: gating, communication, and the franchise-vs-LP tension in 2008 and 2020"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "What a real market crisis forces a fund manager to decide that no model prepares them for: whether to gate redemptions, how to communicate when everyone is afraid, how to meet margin calls without a fire sale, when to use a side pocket, and the agonizing tension between protecting the management company and protecting the limited partners, with 2008 and 2020 as the templates."
tags: ["hedge-funds", "fund-management", "crisis-management", "redemptions", "gates", "side-pockets", "margin-calls", "liquidity-risk", "investor-relations", "asset-management"]
category: "trading"
subcategory: "Hedge Funds"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A market crisis forces a founder into choices no backtest prepares them for: whether to gate redemptions, how to speak when everyone is afraid, how to meet a margin call without selling into a vacuum, when to use a side pocket, and the agonizing tension between protecting the management company (the franchise) and protecting the limited partners. The crisis does not decide whether the fund survives; the response does.
>
> - **The crisis sequence is fixed**: assess losses and liquidity, de-risk early, communicate before LPs ask, meet margin without a fire sale, then decide on a gate and a side pocket. Funds that skip steps die.
> - **A gate is a one-time weapon**: it stops a redemption cascade but spends reputation you can never fully earn back, so it is the lever of last resort, never first.
> - **Communication is the cheapest crisis tool and the first one founders drop**: the silence after a bad month is what turns a drawdown into a redemption run.
> - **The number to remember**: roughly **7–11%** of hedge funds liquidate in a normal year, and that rate spiked in both **2008** and **2020** — the crisis is when funds die, and most of those deaths were response failures, not trade failures.

It is 2:00 a.m. on a Thursday in March 2020. Maya, who runs a long/short-equity fund she calls the Meridian Fund, is sitting at her kitchen table with two screens open. One shows futures that have gone limit-down again. The other shows an inbox: three of her limited partners have submitted redemption requests totaling \$30M against a fund that, six weeks ago, held \$200M and now holds something closer to \$150M after a drawdown she still cannot fully size because half her book has not printed a real bid all day.

She has to decide something by the morning open in Asia. If she sells into tomorrow's session to raise the cash to meet those \$30M of redemptions, she will be dumping her most liquid names — the only ones she *can* sell — into a market with no buyers, locking in losses on the very positions she would want to keep, and leaving the remaining LPs holding a more concentrated, less liquid, more impaired book. If she gates those redemptions — invokes the clause in her fund documents that lets her limit how much capital leaves in a single period — she stops the bleed, but she also tells every investor she has, and every investor she will ever try to raise from, that the Meridian Fund is a place your money can get trapped. There is no option that is merely good. Every choice on the table is a choice about who absorbs the pain.

This is the part of running a fund that the pitch book never mentions and the backtest cannot model. A crisis is not a bigger version of a bad month. It is a regime where the things you assumed were independent become the same thing, where the liquidity you priced as free evaporates, and where your investors, your prime broker, and your own balance sheet all pull on the same rope at the same time. Figure 1 lays out the sequence of decisions Maya is about to walk through — the crisis decision tree that every founder runs, in roughly the same order, every single time.

![A seven-step pipeline of the founder's crisis decisions from assessing losses and liquidity through de-risking, communicating, meeting margin, deciding on a gate, deciding on a side pocket, and surviving to preserve the franchise](/imgs/blogs/the-crisis-playbook-2008-and-2020-1.png)

The goal of this post is to make that sequence concrete: to define every term you will hear thrown around in a crisis, to walk the actual mechanics of gating and side-pocketing and margin-meeting, and then to sit honestly inside the tension that defines the whole exercise — the conflict between protecting the franchise that pays your salary and protecting the investors who trusted you with their capital. We will use 2008 and 2020 as the two canonical templates, because they failed funds in different ways, and a founder who understands both has seen most of what a crisis can throw.

## Foundations: what a crisis does to a fund

Before any of the decisions make sense, the vocabulary has to be precise. In calm markets these words are footnotes in the fund documents. In a crisis they become the only words that matter.

**A crisis, for a fund, is a liquidity event, not a loss event.** Funds survive losses constantly — a 10% drawdown is a Tuesday. What kills funds is the moment when three things happen at once: the value of the book falls sharply, investors try to leave, and the financing that supports the positions gets pulled or repriced. A loss alone is recoverable. A loss that arrives with a wall of redemptions and a margin call is what turns into a death spiral. The 2008 and 2020 events were crises precisely because all three pressures hit together.

**Net asset value (NAV)** is the fund's total assets minus its liabilities — the number the administrator strikes, usually monthly, that tells each investor what their stake is worth. In a crisis the NAV becomes hard to compute honestly, because some of the positions have no observable price. **NAV per share** (or per unit) is what each investor's slice is worth; redemptions are paid at the NAV per share struck on the redemption date.

**A redemption** is an investor asking for their money back. Funds do not offer daily liquidity the way a bank account does; they offer **redemption frequency** (often quarterly), require a **notice period** (commonly 30–90 days), and may impose a **lock-up** (often around a year) before any redemption is allowed at all. These terms exist precisely so the fund is not forced to sell on a day when selling is ruinous. We covered the full set in [liquidity terms: lock-ups, gates, side pockets, and the mismatch that blows funds up](/blog/trading/hedge-funds/liquidity-terms-lockups-gates-side-pockets); here the focus is what happens when the crisis tests them all at once.

**A gate** is a contractual limit on how much capital can leave in a single redemption period. There are two kinds, and the distinction is the single most important structural fact in this whole post:

- An **investor-level gate** caps how much any *one* investor can pull in a period — commonly **25%** of that investor's stake, so a full exit takes four periods. It is fair across investors and is the gate most funds reach for first.
- A **fund-level gate** caps total redemptions across *all* investors in a period — for example, if redemption requests exceed 25% of total NAV, the fund honors requests pro-rata up to that 25% and pushes the rest to the next period. It protects the remaining investors from a fire sale but treats redeemers as a group rather than rewarding whoever filed first.

**A margin call** is the prime broker demanding more collateral against the fund's positions. As prices fall, the broker raises **haircuts** (the discount it applies to collateral value), so the fund must post more cash exactly when its cash is most precious. We cover the financing relationship in [cash, treasury, and counterparty risk](/blog/trading/hedge-funds/cash-treasury-and-counterparty-risk); in a crisis the margin call is the gun to the head that forces selling.

**A fire sale** is selling assets fast and cheap because you have no choice. The hallmark of a fire sale is that the act of selling moves the price against you: you dump a position into a thin market, the price drops, your remaining positions get marked down, and your margin gets worse. A fire sale is how a solvent fund becomes an insolvent one.

**A side pocket** is an accounting structure that segregates a fund's illiquid or hard-to-value positions into a separate sub-account that investors cannot redeem from until those positions are eventually realized. New redemptions are paid out of the liquid main book; the illiquid positions sit in the side pocket and are paid out only as they are sold. It stops a fire sale of the illiquid assets and stops early redeemers from getting paid at a stale price while late redeemers are stuck with the impaired stuff.

**The franchise** is the management company — the business Maya owns. It earns the management fee and the performance fee, employs her team, and is the entity that survives or dies as a going concern. **The limited partners (LPs)** are the investors in the fund. The franchise and the LPs are *different legal entities with different interests*, and in a crisis those interests can point in opposite directions. Protecting the franchise can mean gating to keep AUM (and therefore fee revenue) in place; protecting the LPs can mean letting them leave even though it shrinks the business. This is the core tension, and we will spend a full section on it.

**A liquidity spiral** is the self-reinforcing loop where selling drives prices down, which triggers more margin and more selling, which drives prices down further. It is the mechanism by which a localized shock becomes a market-wide crisis. When everyone is forced to sell the same assets at once, correlations that looked like 0.3 in the backtest snap to 1.0 — a phenomenon we treat in depth in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

With the vocabulary in place, walk the decisions in order. The first one is not whether to gate. It is whether you have de-risked early enough that you never have to.

## The gating decision

Gating is the decision that defines a crisis manager's reputation, so it deserves the most careful treatment — and the most discipline about *when* it is even on the table.

Start with what a gate is for. A gate exists to prevent a **first-mover advantage** among your investors. In a crisis, an investor who redeems first gets paid at a NAV that may not yet reflect the full impairment of the illiquid book, and gets paid in the fund's most liquid assets — leaving the investors who did not redeem holding a smaller, more concentrated, less liquid, more impaired fund. That dynamic is a run: the rational move for every investor, knowing others are redeeming, is to redeem first, which guarantees everyone redeems. A gate breaks the run by making "redeem first" not pay off, because nobody gets all their money out at once.

That is the noble case for a gate, and it is real. The ignoble case — and the one allocators are watching for — is gating to protect the management fee. If Maya gates \$30M of redemptions on a \$150M fund, she keeps charging her management fee on capital that wanted to leave. An LP who is told "you cannot have your money back" while the manager continues to bill them is going to read that as the franchise protecting itself at the LP's expense, and they will never forget it.

So the gating decision is governed by three questions, in order:

1. **Do my documents allow it, and exactly how?** A gate is only available if the limited partnership agreement (LPA) and offering documents grant it, and the precise trigger matters: is it investor-level or fund-level, what is the threshold, does it require board or independent-director approval, and does invoking it require notice? You cannot invent a gate in a crisis; it has to have been written into the documents at launch. Founders who skipped or softened the gate language to win allocators in calm times discover in a crisis that they have no gate to invoke.
2. **Is the redemption pressure a run, or just normal redemptions?** A gate is the right tool for a *run* — redemptions far above the fund's normal flow, driven by fear rather than by individual portfolio decisions. It is the wrong tool for ordinary redemptions you simply do not want to pay. If two LPs want out for idiosyncratic reasons in a normal quarter, you pay them. You gate only when paying everyone who has asked would force a fire sale that harms the LPs who stayed.
3. **Have I exhausted the cheaper levers first?** A gate is the *last* line, not the first. Before gating, you de-risk to raise cash, you draw on cash reserves, you negotiate with the largest redeemers for a staged exit, and you communicate. If you can meet redemptions out of cash and orderly sales, you do not gate. The gate is reserved for the case where meeting the requests in full would require selling into a vacuum.

The mechanics of invoking a gate are unglamorous and must be done by the book. You notify investors in writing, citing the specific provision; you apply the gate **uniformly** (a fund-level gate is pro-rata across all redeemers — you cannot quietly let your biggest, most relationship-important LP out while gating the small ones); you involve the fund's governing body (the board of directors of the offshore fund, or the GP with independent oversight) so the decision is documented as being in the fund's interest, not the manager's; and you set out clearly what happens to the gated portion — typically it carries to the next redemption period.

#### Worked example: gating \$30M of redemption requests

Back to Maya's 2:00 a.m. table. The Meridian Fund's NAV has fallen to roughly \$150M. She has received \$30M of redemption requests for the upcoming quarter-end — that is **20%** of NAV. Her documents include a **fund-level gate** that triggers at **25%** of NAV.

First test: is she even over the gate threshold? No — \$30M is 20% of \$150M, below her 25% trigger. So a fund-level gate is not contractually available for this round. That is a critical fact: gates have thresholds for a reason, and 20% does not clear hers.

Second test: can she meet \$30M without a fire sale? She holds \$15M in cash and unencumbered T-bills. Her book is roughly 60% in large-cap names she can sell in a few days at a manageable cost, and 40% in less liquid positions. To raise the remaining \$15M she would sell large-caps, costing perhaps **1.5%** in market impact in this environment — about \$0.2M of impact on \$15M sold. That is painful but survivable. Meeting the redemptions in full is the *right* answer here: she is not over the gate threshold, and the cost of paying is a manageable \$0.2M of impact, not a ruinous fire sale.

Now change one number. Suppose the requests had been \$45M — **30%** of the \$150M NAV, over her 25% trigger. Now the fund-level gate is available. She honors **25%** (\$37.5M) pro-rata and carries the remaining \$7.5M to next quarter. An LP who asked for \$15M gets \$12.5M now (25% of NAV × their pro-rata share) and \$2.5M next quarter. The gate has stopped her from having to sell the last \$7.5M into the worst of the vacuum, preserving the book for the LPs who stayed.

*The gate is a threshold weapon, not a mood: if the documents say 25% and the pressure is 20%, you do not gate — you pay, even when paying hurts.*

The reputational cost of a gate is the hidden term in this math. Surveys of allocators after 2008 consistently found that *having gated* was one of the most durable black marks a fund could carry — many institutions added "did you gate or suspend in 2008?" as a standing question in their operational due diligence, and a "yes" without an airtight, LP-protective rationale was frequently disqualifying. We treat that screening process in [operational due diligence: the veto](/blog/trading/hedge-funds/operational-due-diligence-the-veto) — the people who decide whether to give you capital next time are explicitly grading how you behaved last time.

## Communication under fire

The cheapest, most powerful crisis tool is also the one founders abandon first, because it is the one that feels worst: talking to your investors when you have nothing good to say.

Here is the dynamic that destroys funds. The fund has a terrible month. The founder, embarrassed and overwhelmed and busy managing the actual portfolio, goes quiet — no letter, no call, just a NAV print that lands in the administrator's monthly statement. The LPs, who now know only that they lost money and that the manager has gone silent, fill the silence with the worst possible story: *the manager is hiding something, the losses are worse than the statement, get out now.* The silence does not calm anyone; it confirms the fear. And the fund that could have survived the drawdown does not survive the run that the silence caused.

The discipline is the opposite of the instinct. In a crisis you communicate **more** than in good times, **earlier** than the LPs ask, and with **more candor** than feels comfortable. The principles:

- **Get ahead of the NAV.** The worst way for an LP to learn the fund is down 20% is from the administrator's statement. Reach them first, with your own framing, before the number lands. An LP who hears it from you reads it as a manager in control; an LP who hears it from a statement reads it as a manager who went dark.
- **Be specific about what happened and why.** "Markets were volatile" is the language of someone hiding the ball. "We were long quality cyclicals and short high-multiple growth; the de-grossing shock hit both legs at once, costing us X on the long book and Y on the short book" is the language of someone who knows exactly what they own and why it moved. Specificity is the proof of competence that calms fear.
- **State what you are doing about it, concretely.** "We have cut gross exposure from 200% to 120%, raised cash to 18% of NAV, and confirmed our margin coverage with both prime brokers" tells the LP you are acting. Vague reassurance ("we remain confident in the portfolio") does the opposite — it sounds like denial.
- **Never surprise an LP with a gate.** If you may need to gate, the LP should hear that it is *possible* before it happens. A gate that arrives as a surprise is a betrayal; a gate that an LP was warned about, and understands the rationale for, is a hard but defensible decision. The surprise, not the gate, is what breaks the relationship.
- **Tell the truth even when the truth is "I don't know yet."** In the worst of a crisis you genuinely cannot mark half the book. Saying "we cannot strike a reliable NAV on the illiquid sleeve until prices normalize, here is our valuation policy and here is our best estimate of the range" is honest and survivable. Pretending to a precision you do not have is how a valuation scandal starts.

The asymmetry is brutal and worth internalizing: communication that goes well buys you patience you desperately need; communication that goes badly, or does not happen, manufactures a redemption run out of a recoverable loss. There is no cheaper insurance, and founders under-buy it precisely when the premium is lowest — at the start of the crisis, when a single honest call could keep an LP in the fund. The retention mechanics of these relationships, in calm and in crisis, are the subject of [investor relations and retention](/blog/trading/hedge-funds/investor-relations-and-retention).

Figure 2 contrasts the two paths through the same drawdown — the fund that goes silent and sells into the vacuum against the fund that de-risks early and communicates honestly — and the survival outcomes they reach.

![A before-and-after comparison showing a crisis managed badly on the left ending in liquidation versus a crisis managed well on the right ending with the franchise surviving](/imgs/blogs/the-crisis-playbook-2008-and-2020-2.png)

## Meeting margin calls

A redemption you can sometimes gate. A margin call you cannot. When the prime broker demands more collateral, you post it by the deadline — usually the same day — or the broker liquidates your positions for you, at prices it chooses, in whatever order is most convenient for the broker. The margin call is the one crisis pressure with no negotiating room and no gate, which is why it dictates the order of operations.

The mechanism is the part founders under-appreciate until they live it. In calm markets the prime broker finances the book at a given haircut — say it lends against 85% of the value of a liquid long position. In a crisis the broker, managing its own risk, **widens the haircut**: now it lends against only 70%, or 60%, of a value that has *itself* fallen. So the required margin climbs for two compounding reasons at once — the positions are worth less, and the broker is financing a smaller fraction of that smaller value. A book that was comfortably margined on Monday can be deeply short of collateral by Wednesday without the fund having traded a single share.

This is why **cash is the only thing that meets a margin call cleanly.** You meet the call by posting cash or unencumbered high-quality collateral (T-bills). If you do not have it, you raise it by selling — which is exactly the fire sale that the whole playbook is trying to avoid. The fund that holds a cash buffer through the crisis meets its calls and lives; the fund that was fully invested, squeezing every basis point of return out of a zero-cash book in the good times, has to sell into the vacuum and dies. The discipline of holding "wasted" cash in calm markets is the premium you pay for the option to survive a crisis, and it is the single most reliable predictor of which leveraged funds make it through.

The structural defense is **multi-prime**. After Lehman Brothers failed in September 2008, funds that had concentrated their entire book with one prime broker discovered that their assets — including assets the prime had **rehypothecated** (pledged onward as collateral for the prime's own borrowing) — were frozen in the bankruptcy estate, inaccessible for months or longer. A fund that could not access its own collateral could not meet calls at its *other* relationships and was forced into liquidation by the failure of a counterparty, not by a bad trade. The lesson the industry took was to spread financing across two or more primes, cap rehypothecation, and hold a meaningful cash buffer outside the prime entirely. Single-prime concentration is now treated as an operational red flag in due diligence for exactly this reason.

#### Worked example: a margin call and a forced de-risk on a \$200M book

Take Maya's fund at the start of the 2020 gap-down, before the redemptions, when it still held \$200M and ran at roughly **5×** gross (so about \$1B of gross exposure, long and short). The book is financed across two prime brokers. She holds a cash buffer of \$18M — about 9% of NAV.

Monday, the market gaps down. Her required margin starts the week around \$12M and her \$18M of cash comfortably covers it. But the prime brokers widen haircuts into the drawdown. By the table below, required margin climbs while her available cash drains as losses bite:

| Day | Margin required (USD) | Cash available (USD) | Position |
|---|---|---|---|
| Mon | \$12M | \$18M | covered |
| Tue | \$14M | \$16M | covered, tightening |
| Wed | \$17M | \$13M | short \$4M — forced sale begins |
| Thu | \$20M | \$11M | short \$9M |
| Fri | \$21M | \$14M | gap closing after de-risk |

By Wednesday the required margin (\$17M) exceeds her available cash (\$13M) — she is \$4M short and the broker is on the phone. She has no choice but to sell to raise cash. The question is not *whether* to sell but *what*. The right move is a **deliberate de-risk**: cut the gross exposure across the whole book proportionally, selling the most liquid names first to raise cash with the least market impact, rather than dumping whatever is easiest at any price. By Friday she has cut gross from 5× to roughly 3×, raised enough cash to close the margin gap, and — critically — done it as a controlled de-grossing rather than a panicked fire sale. The de-risk costs her: she has locked in losses and capped her rebound if the market snaps back. But it has kept her solvent and out of the broker's hands.

Figure 6 shows that cascade — the required margin climbing while the cash buffer drains, the forced-sale zone where the two cross, and the de-risk that finally closes the gap.

![A line chart showing required margin climbing from 12 to 21 million dollars while available cash falls from 18 to 11 million dollars across a gap-down week, with the deficit zone shaded red and the de-risk closing the gap on the final day](/imgs/blogs/the-crisis-playbook-2008-and-2020-6.png)

*A margin call has no gate: you post cash or the broker sells for you, so the cash you "wasted" holding in calm markets is the only thing that lets you de-risk on your own terms instead of theirs.*

The risk discipline that prevents this from becoming terminal — sizing positions so a crisis-grade move never forces liquidation — is the subject of [risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up). The crisis is where that discipline either pays off or its absence becomes fatal.

## The side pocket

Some positions cannot be sold at any sensible price in a crisis — a private placement, a stake in a thinly traded credit, a position so large relative to its market that liquidating it would itself crater the price. For these, the founder has a tool that is neither a gate nor a sale: the side pocket.

A side pocket carves the illiquid positions out of the main fund into a separate sub-account. Once a position is side-pocketed, three things change. First, **new redemptions are paid only out of the liquid main book** — an LP who redeems gets their pro-rata share of the liquid assets in cash and a *continuing* interest in the side pocket that pays out only as those illiquid positions are eventually realized. Second, **management and performance fees on the side pocket are usually frozen or restructured** — you do not get to charge a performance fee on a paper mark of an asset you cannot sell. Third, **valuation of the side pocket is governed by a documented policy** rather than a live market price, because by definition there is no reliable live price.

The purpose is fairness across investors in time. Without a side pocket, the LP who redeems early gets paid in cash at a NAV that includes a possibly-too-high mark on the illiquid sleeve, walking away whole while the LPs who stay are left holding the impaired, unsellable positions. The side pocket stops that: it freezes the illiquid positions so that *whoever owned them when the crisis hit bears their eventual outcome*, redeemer and stayer alike, pro-rata. It also stops a fire sale of assets that would fetch pennies if dumped but might recover substantial value if held to an orderly realization.

Side pockets are powerful and dangerous in equal measure, which is why they are heavily scrutinized. The danger is abuse: a manager can be tempted to side-pocket a position simply because it is *losing*, not because it is genuinely illiquid, to keep it out of the redeemable NAV and out of the high-water-mark math. After 2008, the SEC brought enforcement actions against managers who side-pocketed liquid positions, mis-valued side-pocketed assets, or failed to disclose side pockets properly. The honest use is narrow: a side pocket is for assets that are *genuinely* illiquid and hard to value, it is governed by the fund documents and a written valuation policy, it is disclosed, and the fee treatment is fair to LPs. Used that way it is a legitimate crisis tool. Used to hide losses it is a path to an enforcement action and the end of the franchise.

#### Worked example: the franchise-versus-LP trade-off in dollars

This is the example that exposes the whole tension, so set it up carefully. The Meridian Fund holds \$150M after the drawdown. Of that, \$120M is liquid and \$30M sits in a single illiquid credit position that Maya believes is worth \$30M if held to maturity but would fetch maybe \$18M if dumped today into a frozen market.

She has \$45M of redemption requests — 30% of NAV, over her gate trigger. She has three broad paths:

- **Path A — fire-sale everything to pay redemptions.** She sells the illiquid position for \$18M (a \$12M loss versus her held-to-maturity estimate) plus \$27M of liquid assets to fund the \$45M. The redeemers get paid in full and walk away. The LPs who stayed are left with a \$105M fund that has crystallized a \$12M loss on the credit position they will never recover. *The franchise keeps the redeeming LPs happy but impairs the loyal ones — and shrinks AUM to \$105M.*
- **Path B — gate, no side pocket.** She gates at 25% (\$37.5M), pays that pro-rata out of liquid assets, carries \$7.5M to next quarter. She avoids selling the illiquid position. But the redemptions are still being paid at a NAV that values the illiquid sleeve at the full \$30M — so redeemers exit at a mark that may prove too high, and stayers eat the difference if the credit underperforms. *Fairer than a fire sale, but the timing inequity remains.*
- **Path C — side-pocket the illiquid position, then meet redemptions out of the liquid book.** She moves the \$30M credit position into a side pocket. Redemptions are now satisfied from the \$120M liquid book: she pays the \$45M (or gates it if over threshold) entirely from liquid assets, and *every* investor — redeemer and stayer — keeps their pro-rata interest in the side pocket and shares its eventual \$30M-or-\$18M outcome. No fire sale of the credit, no timing inequity, and her performance fee on the side pocket is frozen until it is realized. *The side pocket is the only path that is fair across both time and investors — at the cost of complexity and a frozen fee.*

Run the franchise math underneath. Maya's management fee is 1.5%. On the \$105M post-fire-sale fund (Path A), her annual management revenue is about \$1.6M. On a \$150M fund where she side-pockets and retains more AUM (Path C), it is about \$2.25M. The franchise has a direct financial interest in *not* shrinking — which is exactly why every crisis decision must be checked against the question "am I doing this for the LPs or for my fee?" The side pocket happens to align the two here, which is why it is the right answer; but the founder must be honest that the franchise's incentive is always pulling toward keeping AUM in place.

*A side pocket is the one tool that can be fair to redeemers and stayers at the same time — but the very fact that it also protects the manager's AUM is why it must be used only when it is genuinely fair, and disclosed when it is.*

## Franchise vs LPs: the core tension

Everything above converges on one structural fact that the founder must hold clearly in their head, because it is the source of every hard crisis decision: **the management company and the limited partners are different parties with different interests, and the founder is the agent of both.**

The franchise — Maya's management company — earns the management fee and the performance fee, employs her team, carries her name, and is the thing that either survives as a going concern or shuts down. Its interest is to **keep AUM high** (more fee revenue), **keep the track record alive** (so it can raise again), and **stay in business**. The LPs' interest is to **get the best risk-adjusted outcome on their specific capital**, which sometimes means being allowed to leave, being paid promptly, and not being trapped in a fund whose manager is preserving the franchise at the LPs' expense.

In calm markets these interests are aligned: the way to keep AUM high and the track record alive is to compound the LPs' capital well, so doing right by the LPs *is* doing right by the franchise. The crisis is exactly the regime where they diverge, and they diverge along every lever:

- **Gating** keeps AUM in place (good for the franchise) but traps capital that wanted to leave (bad for the redeeming LP). The franchise's incentive is always nudging toward "gate"; the LP-protective rationale must be real, not a rationalization of the fee motive.
- **Selling to meet redemptions** honors the redeemers (good for those LPs) but can impair the stayers and shrink the franchise (bad for both the franchise and the loyal LPs). The franchise's incentive nudges toward *not* selling.
- **Side-pocketing** can be fair to LPs *and* protect AUM — the rare case where the interests align — which is precisely why it is so tempting to abuse: a manager can dress up a fee-protecting move as an LP-protecting one.
- **Communication** costs the franchise nothing financially but costs the founder emotionally; here the franchise's long-run interest (retain trust, raise again) aligns with the LPs', so the failure to communicate is a failure of nerve, not of incentive.

The honest framing is this: in a crisis the founder must repeatedly ask, for every decision, *"would I make this same choice if I earned no fee on the trapped capital?"* If the answer is yes, the decision is LP-protective and defensible. If the answer is no — if the only reason to gate is to keep billing the management fee — then the decision is franchise-protective at the LPs' expense, and it is the kind of decision that, even when it is legal, ends careers when the LPs and the allocators figure out what happened. The funds that survive a crisis *and* keep their reputation are the ones whose founders resolved the tension in the LPs' favor when it was close, and accepted a smaller franchise as the price of a clean reputation.

There is a long-game argument that makes this less saintly than it sounds. A founder who protects the LPs at the franchise's short-term expense in a crisis builds the one asset that is hardest to manufacture and most valuable in the next raise: a reputation for behaving well when it cost them money. Allocators talk to each other. The fund that gated for fee reasons in 2008 was, in many cases, unable to raise meaningfully afterward; the fund that let LPs out gracefully, ate the smaller AUM, and communicated honestly was the one allocators came back to. Doing right by the LPs in the crisis *is* the franchise's best long-run strategy — it just does not feel that way at 2:00 a.m. with the fee revenue draining. This is the same kind of long-horizon, reputation-as-capital reasoning that separates senior decision-makers from junior ones, treated in [decision-making under uncertainty: the senior's edge](/blog/trading/quant-careers/decision-making-under-uncertainty-the-seniors-edge).

Figure 4 lays the four crisis levers against what each one buys and what each one costs, so the trade-offs are visible at a glance.

![A matrix showing the four crisis levers de-risk, communicate, side pocket, and gate against what each one buys and what each one costs](/imgs/blogs/the-crisis-playbook-2008-and-2020-4.png)

### The liquidity spiral the founder is fighting

The reason every lever is a trade-off, and the reason the founder cannot simply "wait it out," is that the crisis is not a static loss to be endured. It is a dynamic, self-reinforcing process — a liquidity spiral — that gets worse the longer the founder is passive and the more everyone is forced to sell the same assets at once.

The loop runs like this. A price shock hits the book and triggers a margin call. The margin call forces selling. The selling, in a market where many leveraged players are being forced to sell the same things at the same time, pushes prices down further. The lower prices trigger the *next* margin call, which forces more selling, which pushes prices down again. Each turn of the loop is rational for the individual fund — you have to meet the call — and catastrophic in aggregate, because everyone meeting their call at once is what makes the prices fall that triggers the next round of calls.

This is why de-risking *early* matters so much: the fund that cuts gross at the first sign of stress steps out of the spiral while there are still buyers, while the fund that waits gets forced to sell at the bottom along with everyone else. And it is why a gate can be a market-stabilizing tool and not just a fund-stabilizing one: a gate that stops a redemption-driven fire sale removes one fund's forced selling from the spiral. The tools to break the loop are the same two that keep a single fund alive — holding cash so you are not forced to sell, and gating so redemptions do not force a sale. Figure 5 traces the spiral and the two points where it can be broken.

![A graph showing the liquidity spiral where a price shock triggers a margin call that forces selling that lowers prices and feeds back into the next margin call, with cash and a gate breaking the loop](/imgs/blogs/the-crisis-playbook-2008-and-2020-5.png)

The 2008 mechanics of this spiral — why diversification stopped diversifying, why everything fell together — are exactly the regime where correlations go to one, which is treated from the cross-asset side in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

## The 2008 and 2020 templates

The two crises that every fund founder studies failed funds in *different* ways, and a founder who understands both has a fuller map than one who has internalized only one.

**2008 was a slow-motion solvency-and-liquidity crisis.** It built over months — Bear Stearns in March, the slow bleed through the summer, then the violence of September with Lehman's failure and the near-collapse of the financial system. The defining features for funds were: a *prime-broker counterparty failure* (Lehman) that froze the assets of funds concentrated with it, including rehypothecated collateral; a wave of redemptions as funds-of-funds and institutions pulled capital both because they were afraid and because their *own* investors were redeeming from *them* (a redemption cascade that ran up the chain); and a liquidity freeze in credit and structured products that made huge swaths of fund portfolios impossible to mark or sell. 2008 was the crisis of **gates and side pockets and suspensions** — an unprecedented number of funds gated, side-pocketed, or suspended redemptions, and the reputational fallout reshaped how allocators evaluate liquidity terms to this day. It was also the crisis that taught the industry that **single-prime concentration is a fatal risk** and made multi-prime standard.

**2020 was a fast, violent, V-shaped liquidity shock.** In March, as the pandemic shut down the world, markets did not bleed for months — they gapped down in days, with even the "safest" assets (Treasuries, gold) selling off in the dash for cash, and then, after extraordinary central-bank intervention, rebounded almost as fast. The defining features for funds were: a *speed* that left no time to deliberate — margin calls arrived in days, not weeks; a *flight to cash* so universal that normally-uncorrelated assets all fell together (correlations went to one); and then a *sharp reversal* that punished the funds that had panic-sold at the bottom and rewarded the ones that had de-risked in a controlled way and kept enough exposure to participate in the rebound. 2020 was the crisis of **the margin call and the de-risk** — the funds that died were largely the over-levered ones forced to liquidate at the lows, and the funds that thrived were the ones with cash buffers and the discipline to de-gross deliberately rather than dump.

The synthesis a founder should carry is that **2008 tests your liquidity terms and your counterparty structure; 2020 tests your cash buffer and your speed of decision.** A fund built only for a 2008-style slow crisis (good gates, side pockets, multi-prime) but running with no cash and too much leverage gets killed by a 2020-style fast shock before its gates ever matter. A fund built only for a 2020-style fast shock (big cash buffer, low leverage) but with weak liquidity terms and a single prime gets killed in a 2008-style slow grind when its single prime fails and it has no gate to stop the run. You need both defenses, because you do not get to choose which crisis arrives.

Figure 3 sketches the two shapes overlaid — the kind of drawdown path each produces and how the redemption pressure builds against it.

![A chart showing a fund drawdown reaching a trough of negative twenty-eight percent at month five overlaid with quarterly redemption requests that spike to thirty percent of NAV at the trough](/imgs/blogs/the-crisis-playbook-2008-and-2020-3.png)

#### Worked example: the March 2020 liquidity squeeze and the cost of selling into it

Make the 2020 cost concrete on Maya's book. In the worst week of March 2020, the Meridian Fund needs to raise \$20M to meet a combination of margin calls and the most urgent redemptions. She has two ways to raise it.

**Selling into the vacuum (the wrong way).** She dumps \$20M of positions into the worst of the illiquidity. Bid-ask spreads that are normally a few basis points have blown out to **2–4%**, and her own selling moves the price. Realistically she pays around **3%** in spread and impact — about **\$0.6M** of pure transaction cost to raise \$20M — and she sells at prices near the absolute bottom. When the market rebounds 20% over the following weeks, the positions she sold would have recovered roughly \$4M of value she no longer owns. Total cost of selling into the vacuum: about **\$0.6M** of impact plus roughly **\$4M** of forgone rebound — call it **\$4.6M** destroyed on a \$20M raise.

**Drawing the cash buffer and de-risking deliberately (the right way).** Because she held an \$18M cash buffer going in, she meets most of the \$20M from cash and raises only the small remainder by trimming her most liquid names at a manageable cost. She participates in the rebound on the positions she kept. The "cost" was the drag of holding \$18M in cash through the calm years before — perhaps **\$0.5M** of forgone return per year of carrying idle cash — which now looks like the best insurance premium she ever paid.

*The cost of selling into a 2020-style vacuum is not the loss on the day; it is the spread you pay at the bottom plus the rebound you forfeit, which is why the cash buffer that felt like dead weight in the calm years is what saves the fund in the crisis.*

## Common misconceptions

**"A gate protects the fund, so gating is the responsible thing to do in a crisis."** A gate protects the *remaining* investors from a fire sale, which is sometimes responsible — but it is the lever of last resort, not first, and gating when you are below your contractual trigger, or gating to keep your management fee on capital that wanted to leave, is exactly the abuse that ends careers. The responsible default is to meet redemptions out of cash and orderly sales and to gate only when paying in full would force a fire sale that harms the stayers. The gate is a weapon you can fire once before your reputation bears the cost; treat it that way.

**"If my returns are good, my LPs will stay through a crisis."** Returns buy you nothing once an LP is afraid and the manager has gone silent. Investors redeem in crises for reasons that have nothing to do with your alpha — their *own* investors are redeeming from them, their risk committee mandated de-risking across all managers, they need liquidity somewhere and you are the line item they can actually sell. Retention in a crisis is a function of trust and communication far more than of the trailing return number. The fund with the better track record and worse communication loses more LPs than the fund with the worse track record and better communication.

**"A side pocket is a way to hide losses until they recover."** A side pocket used to hide losses — to move a merely *losing* position out of the redeemable NAV and the high-water-mark math — is securities fraud, and the SEC has brought enforcement actions for exactly this. A side pocket is legitimate only for genuinely *illiquid and hard-to-value* assets, governed by a documented valuation policy, properly disclosed, with fair fee treatment. The test is liquidity, not direction: you side-pocket what you cannot sell, never what you do not want to mark.

**"The crisis is what kills the fund."** The crisis is the *trigger*; the *response* is what kills the fund. Funds survive enormous drawdowns when they de-risk early, hold cash, communicate honestly, and use their liquidity terms as designed. Funds die when they sell into the vacuum, go silent, run with no cash buffer, or surprise their LPs with a gate. Two funds in the identical drawdown reach opposite outcomes based entirely on how the founder behaves — which is the whole reason a playbook exists.

**"I can figure out my crisis response when the crisis comes."** Every crisis decision is made under time pressure you cannot fabricate in calm — a margin call due the same day, a redemption deadline, a market with no bids. The decisions that survive are the ones already made: the gate language already in the documents, the cash-buffer target already set, the prime-broker structure already diversified, the LP contact list already prioritized, the honest message already drafted in outline. The founder who waits to decide until 2:00 a.m. in March decides badly. The playbook is something you build in the calm and execute in the storm.

**"Protecting the franchise and protecting the LPs are basically the same thing."** In calm markets, yes — compounding the LPs' capital well *is* how the franchise prospers. In a crisis they diverge sharply: gating, not selling, and keeping AUM in place all favor the franchise's fee revenue at the redeeming LPs' expense. The founder is the agent of both parties, and the discipline is to resolve close calls in the LPs' favor — both because it is right and because, in the long game, a clean crisis reputation is the franchise's single most valuable asset.

## How it plays out in the real world

The named events are the ones to study, because each one is a different way the playbook was followed or broken.

**Long-Term Capital Management (1998)** is the leverage-and-correlation template. LTCM ran balance-sheet leverage around **25–30×** on top of roughly **\$1.25 trillion** in derivatives notional, on convergence trades that assumed historically uncorrelated spreads. When Russia defaulted, those spreads all widened together — correlations went to one — and the fund lost roughly **\$4.6B** in months. The positions were so large relative to the market that LTCM *could not* exit without moving prices against itself, so it was trapped in its own liquidity spiral; the resolution was a **\$3.6B** recapitalization organized by the New York Fed with fourteen banks, because a disorderly liquidation threatened the whole system. The lesson the founder takes is that **leverage plus concentration plus correlated tails is the combination that removes every exit** — by the time LTCM wanted to de-risk, it was too big to sell.

**Amaranth Advisors (2006)** is the concentration template. A single trader's concentrated natural-gas spread bets lost roughly **\$6.6B** — about **65%** of the fund's \$9B-plus AUM — in roughly a week. There was no crisis playbook that could have saved a fund whose entire risk was in one book that gapped against it overnight; the failure was upstream, in letting a single concentrated position grow to where a normal adverse move was terminal. The lesson is that the best crisis management is the position sizing that ensures no single shock can end you — a discipline covered in [risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up).

**The 2008 redemption wave** is the cascade template. As markets fell, funds-of-funds faced redemptions from *their* investors and had to redeem from the underlying funds to raise cash, which forced those funds to sell, which drove prices down, which triggered more redemptions up and down the chain. An unprecedented number of funds responded with gates, side pockets, and outright suspensions — and the allocator community's reaction was permanent. Operational due diligence after 2008 added explicit questions about crisis behavior, and a fund that had gated for fee-protective reasons, or suspended without a clear LP-protective rationale, frequently found itself unable to raise afterward. The reputational accounting of the crisis lasted far longer than the drawdown.

**March 2020** is the speed template. The shock was so fast and the dash for cash so universal that even Treasuries sold off, and over-levered funds got margin-called into liquidation within days, selling at the lows just before the central-bank-driven rebound. The funds that came through best were the ones with cash buffers and the discipline to de-gross deliberately rather than dump — and many of them posted strong full-year returns precisely because they survived March with enough exposure to capture the recovery. 2020 rewarded preparation and punished leverage with brutal speed, validating the cash-buffer discipline that had looked like dead weight in the long calm bull market before it.

Across all of these, the pattern that determines survival is the same: the funds that died were, overwhelmingly, the ones that entered the crisis over-levered with no cash buffer, then compounded the problem by selling into the vacuum and going silent. The funds that lived had cash, had diversified financing, de-risked early, and talked to their investors. The crisis sorted funds not by the cleverness of their strategy but by the soundness of their preparation and the discipline of their response — which is exactly what the playbook is built to deliver. The full taxonomy of the ways funds end is the subject of the companion post [how hedge funds die: the failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy).

Figure 7 collapses the whole playbook onto one card — the nine things a prepared founder has already decided before the morning the market gaps down.

![A nine-cell grid checklist of crisis preparations including knowing daily liquidity, mapping margin terms, pre-drafting gate language, a side-pocket policy, an LP contact tree, stress testing the book, a cash-buffer target, multi-prime setup, and an honest message draft](/imgs/blogs/the-crisis-playbook-2008-and-2020-7.png)

## When this matters / Further reading

This matters the moment you run leveraged, externally-funded capital — which is to say, the moment you are a hedge fund rather than a separately managed account you control entirely. The crisis playbook is not advanced material to get to later; it is the thing that determines whether "later" exists. A founder can do everything else in this series right — structure the fund cleanly, raise from good LPs, build a sound operation, pass operational due diligence — and still lose all of it in a single mishandled crisis. The work is to build the playbook in the calm: write the gate language into the documents, set and hold the cash buffer, diversify the prime brokers, draft the LP contact tree, and decide in advance the principle that you will resolve close calls in the LPs' favor. Then, when the market gaps down and it is 2:00 a.m., you are executing a plan instead of improvising a tragedy.

The crisis is also where the abstract tension at the heart of this whole series becomes concrete and personal: the fund is a promise to compound the LPs' capital inside a structure that protects them, and the crisis is the test of whether you keep that promise when keeping it costs you. The founder who keeps it builds the reputation that lets the franchise live to compound another decade. The founder who breaks it to protect the fee may survive the quarter and lose the career.

For the pieces this post builds on and points to:

- [Liquidity terms: lock-ups, gates, side pockets, and the mismatch that blows funds up](/blog/trading/hedge-funds/liquidity-terms-lockups-gates-side-pockets) — the full set of liquidity tools the crisis tests.
- [Cash, treasury, and counterparty risk](/blog/trading/hedge-funds/cash-treasury-and-counterparty-risk) — the cash buffer and prime-broker structure that decide whether you meet margin on your own terms.
- [Investor relations and retention](/blog/trading/hedge-funds/investor-relations-and-retention) — the relationship work that determines whether communication under fire lands.
- [How hedge funds die: the failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy) — the broader catalog of fund deaths this crisis playbook is meant to prevent.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — the market mechanism that makes the liquidity spiral and the diversification failure happen.
- [Operational due diligence: the veto](/blog/trading/hedge-funds/operational-due-diligence-the-veto) — the process by which allocators grade exactly how you behaved in the last crisis.
- [Decision-making under uncertainty: the senior's edge](/blog/trading/quant-careers/decision-making-under-uncertainty-the-seniors-edge) — the long-horizon, reputation-as-capital reasoning that resolves the franchise-versus-LP tension.
- [Risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up) — the position sizing that keeps a crisis from ever becoming terminal.
