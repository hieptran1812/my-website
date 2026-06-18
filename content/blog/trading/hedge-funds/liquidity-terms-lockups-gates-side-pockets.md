---
title: "Liquidity terms: lock-ups, gates, side pockets, and the mismatch that blows funds up"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How lock-ups, redemption frequency, notice periods, gates, side pockets, and suspension rights align the speed at which investors can leave with the speed at which a fund can sell, and what happens in 2008 when the two diverge."
tags: ["hedge-funds", "fund-management", "liquidity", "redemptions", "lock-up", "gates", "side-pockets", "asset-management", "fund-terms", "risk-management"]
category: "trading"
subcategory: "Hedge Funds"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The deadliest structural mistake in fund management is promising investors more liquidity than your assets can deliver; lock-ups, redemption frequency, notice periods, gates, side pockets, and suspension rights are the tools that align how fast investors can leave with how fast you can sell.
>
> - **Liability liquidity** (how fast investors can pull money out) must match **asset liquidity** (how fast you can sell without crushing the price). When the door is faster than the sale, a redemption wave becomes a forced fire-sale and then a run.
> - The toolkit is a ladder of friction: a **lock-up** bars exit for a period, a **notice period** demands warning, a **gate** caps how much can leave per dealing day, a **side pocket** carves illiquid positions out of redemptions, and **suspension** freezes the door entirely.
> - These terms are not a tax the manager levies on investors out of greed; they protect the *remaining* investors from being diluted by a fire-sale triggered by the leavers.
> - In the 2008–2009 crisis, an unusually large share of funds gated, suspended, or side-pocketed — roughly a quarter at the peak (illustrative) — because the industry had sold monthly and quarterly liquidity on books that needed months to unwind.

It is November 2008. Maya runs a credit fund she will call the Meridian Fund — \$220M of structured credit, leveraged loans, and a sleeve of private placements she bought because they yielded more than anything trading on-screen. Her returns have been excellent for three years. Her terms, set the day she launched, offered investors **quarterly redemptions with 45 days' notice** because a competitor down the hall offered quarterly liquidity and her seed investor's consultant said monthly would have been even better for the raise.

This morning her administrator forwards the redemption notices that arrived before the quarter-end cutoff. They total **\$30M** — about 14% of the fund. Some are from funds-of-funds whose own investors are redeeming; some are from a family office that needs cash; one is from her anchor seed investor, who is "rebalancing." None of them are angry. They are simply leaving, the way the documents say they are allowed to.

Maya pulls up her book and does the arithmetic that will define the next eighteen months of her life. The on-screen positions she can sell in a week — but the bid is gone; dealers are quoting her marks 8 to 12 points below where her administrator struck NAV at month-end. The leveraged loans settle T+20 on a good day and the secondary desk is not picking up the phone. The private placements cannot be sold at any honest price for *months*. She has promised her investors \$30M of cash in 45 days against a book that, if she is honest, takes six months to liquidate without setting fire to the marks. The gap between *how fast they can leave* and *how fast she can sell* is the whole problem. It has a name — **liquidity mismatch** — and it is, far more often than a bad trade, what actually kills funds. Figure 1 lays out the tools she should have used to close that gap before she ever took the first dollar.

![Matrix of five liquidity tools showing what each one does and who it protects: lock-up, notice period, gate, side pocket, and suspension](/imgs/blogs/liquidity-terms-lockups-gates-side-pockets-1.png)

This post is about those tools — what each one is, why it exists, how it works mechanically, and how they fit together into a single discipline: **matching the liquidity you promise your investors to the liquidity your assets can actually deliver.** Get that match right and a redemption is a routine cash-flow event. Get it wrong and a redemption is the first domino in a run that ends with a gate, a side pocket, an angry investor base, and — often — a wind-down. The Meridian Fund is fictional, but every number and every failure mode in it is drawn from what genuinely happened to credit and structured-product funds in 2008.

## Foundations: the liquidity vocabulary

Before anything else, fix the words. Almost every disaster in this area comes from a founder who knew the *terms* but never internalized the *mismatch they create*. Build the vocabulary from zero.

**Liquidity** is the speed at which you can convert something into cash *at a fair price*. The "fair price" qualifier is the entire point. Anything can be sold instantly if you accept a bad enough price; a position is liquid if you can sell it quickly *without* moving the price against yourself. A large-cap stock is liquid: you can sell \$5M of Apple in minutes near the screen price. A tranche of a bespoke structured-credit deal is illiquid: there may be no buyer this month at any price you would call fair, and finding one takes weeks of calls.

A fund has liquidity on **two sides**, and the founder's whole job here is to keep them aligned:

- **Asset liquidity** — how fast you can sell what you own without crushing the price. This is set by your *strategy*: a liquid macro book is days; a distressed-debt or private-credit book is months.
- **Liability liquidity** — how fast your investors can get their money out. This is set by your *terms*: the lock-up, the redemption frequency, the notice period, the gates. Investors are the fund's liabilities — they have a claim on the fund's cash — so "liability liquidity" is just "how fast those claims come due."

**The liquidity mismatch** is the gap between the two: offering monthly redemptions on a book that takes six months to sell. It is the single most dangerous structural error a fund can make, because it is invisible in calm markets — everyone gets their money, on time, every month — and catastrophic in a stressed one, when everyone wants out at once and the assets cannot be sold fast enough to pay them.

Now the tools that close the gap. Each is a different lever on liability liquidity:

**Lock-up** — a period after an investor subscribes during which they *cannot* redeem at all. A one-year lock-up means money invested in January cannot leave until the following January at the earliest. A **hard lock-up** forbids redemption entirely during the period. A **soft lock-up** allows redemption but charges a penalty fee — typically **2% to 5%** of the redeemed amount, often paid *into the fund* (so it benefits remaining investors rather than the manager).

**Redemption frequency** — how often the fund offers an exit. Common frequencies are **monthly, quarterly, semi-annual, and annual**. The dates the fund processes redemptions are **dealing days** (or "redemption dates"). Less frequent redemption is friendlier to the manager and to the remaining investors; more frequent is friendlier to the leaver.

**Notice period** — how far in advance an investor must tell the fund they intend to redeem. A **60-day notice** on a quarterly fund means you must file by the end of January to get out at the March 31 dealing day. The notice period gives the manager *time to sell in an orderly way* before cash is due.

**Gate** — a cap on how much can be redeemed at a single dealing day. There are two kinds, and the distinction matters enormously:

- An **investor-level gate** caps how much *any single investor* can redeem at one date — e.g. 25% of their stake per quarter, so a full exit takes four quarters.
- A **fund-level gate** caps *total* redemptions across all investors at one date — e.g. 25% of fund NAV — and pro-rates the available amount across everyone who asked, deferring the rest.

**Side pocket** — a mechanism that carves an *illiquid* position out of the main, redeemable book into a separate sub-account. Investors who are in the fund when the side pocket is created keep their share of it, but they *cannot redeem it* until the underlying position is realized (sold or matured). New investors do not participate in it. The side pocket exists so that a leaver does not get paid out at a fake "fair" price on an unmarkable asset, leaving the stayers holding the loss.

**Suspension** — the nuclear option: the manager invokes a right (written into the fund documents) to *suspend redemptions entirely*, usually also suspending the calculation of NAV. No one gets out until the suspension is lifted. It is used only when paying redemptions would harm the remaining investors so badly that freezing everyone is the lesser evil.

Two more terms you will meet: **NAV** (net asset value) is the fund's assets minus liabilities; **NAV per share** is that divided by the number of shares, and it is the price at which investors subscribe and redeem. **In-kind redemption** means paying a redeeming investor with a slice of the actual securities instead of cash — a rarely-used escape hatch for illiquid books.

That is the whole vocabulary. The rest of this post is about how these pieces fit together and what happens when they do not.

## Why liquidity terms exist

It is tempting to read the toolkit above as a list of ways the manager makes it *hard* for you to get your money — friction the manager imposes because illiquid investors are convenient. That reading is wrong, and getting it right is the foundation for everything else. Liquidity terms exist to solve a genuine structural problem, and the problem belongs to the *investors* at least as much as to the manager.

The problem is this: a hedge fund is a **commingled pool**. Many investors share one book. When one investor redeems, the manager must raise cash to pay them — by selling assets, or by holding cash that would otherwise be invested. If the manager has to sell into a thin market to fund a redemption, the sale moves the price *down*, and that lower price is struck into the NAV that *everyone else* still owns. The leaver got out near the old, higher mark; the stayers eat the markdown caused by the leaver's exit. This is **redemption-driven dilution**, and it is a transfer of wealth from the patient investors to the impatient one.

Worse, it is self-reinforcing. If investors suspect that early redeemers get paid at good prices while late redeemers eat the fire-sale, the rational move is to redeem *first* — which is exactly the logic of a bank run. Everyone races for the exit, the manager is forced to dump the whole book at distressed prices, and a fund that was perfectly solvent in a calm market is destroyed by the *dynamics of its own redemption queue*. The assets did not become worthless. The fund failed because its liability structure let a stampede form.

Liquidity terms are the brake on that stampede. A lock-up keeps the capital base stable long enough to invest it properly. A notice period gives the manager time to sell in an orderly way rather than at a fire-sale. A gate stops a single dealing day from forcing a liquidation of the whole book. A side pocket ensures the leaver and the stayer share an illiquid loss *fairly* rather than the leaver escaping it. Suspension freezes the door when the alternative is destroying value for everyone.

Frame it against the series spine: a fund sells an *aligned promise* to compound capital inside a structure that protects the investor. Liquidity terms are a load-bearing part of that structure. They are not the manager protecting *himself* from investors — they are the manager protecting the *investor collective* from the few who would otherwise impose a fire-sale on the many. The honest version of the pitch is: "Your terms are designed so that no other investor's exit can hurt you, and so your exit cannot hurt them." Figure 2 shows the two worlds side by side — the matched fund where a redemption is orderly, and the mismatched fund where it becomes a run.

![Before-and-after comparison of a mismatched fund forced into a fire-sale and a run versus a matched fund that sells calmly into its notice window](/imgs/blogs/liquidity-terms-lockups-gates-side-pockets-2.png)

There is a second, quieter reason these terms exist: they let the manager *actually run the strategy*. Many of the best risk-adjusted returns in the industry come from harvesting an **illiquidity premium** — getting paid extra to hold assets that others cannot or will not hold because they are hard to sell. Distressed debt, private credit, special situations, certain structured products: these earn their returns *because* they are illiquid, and you cannot harvest that premium with a book you might have to liquidate next month. Long lock-ups and infrequent redemptions are the price of admission to those returns. A fund that promises daily liquidity simply cannot hold those assets, and so cannot earn those returns. The terms and the strategy are two ends of the same decision.

## Lock-ups

A **lock-up** is the most fundamental liability-liquidity tool: a period after subscription during which capital cannot leave. It buys the manager a stable base.

The mechanics are simple but the variants matter. A **hard lock-up** forbids redemption outright for the period — typically **one year**, sometimes two or three for less liquid strategies. A **soft lock-up** allows redemption during the period but charges an **early-redemption penalty**, commonly **2% to 5%** of the amount redeemed, and crucially that penalty is usually paid *into the fund* — it accrues to the remaining investors as compensation for the disruption, not to the manager as a fee. Some funds use a **rolling lock-up** (each subscription is locked for a year from *its own* date) versus a **hard-start lock-up** (every investor is locked until one common date). And **founders share classes** — the discounted fee class offered to early or large investors — frequently come with a *longer* lock-up as part of the trade: you get 1.5/15 instead of 2/20, but you commit your capital for two years instead of one.

Why would an investor accept a lock-up at all? Three reasons. First, it is often the only way to access an illiquid-premium strategy at all. Second, a lock-up *protects the locked investor too* — it stops the fund from being drained by faster-moving money. Third, a manager who insists on an appropriate lock-up is signaling that the strategy is real: a founder who offers daily liquidity on a distressed-credit book is either lying about the strategy or about to blow up.

But a lock-up is a promise the *manager* makes, too — a promise to deploy that locked capital into the strategy that justifies locking it. Locking up investors for a year and then running a liquid, market-neutral book that could have offered monthly liquidity is a mismatch in the *other* direction: you have taken away flexibility the investor did not need to give. Allocators notice. The lock-up must fit the asset.

#### Worked example: lock-up plus notice timing for an investor who wants out

Maya's earlier fund (before the Meridian credit fund) was a long/short equity book, and she set its terms correctly: a **one-year hard lock-up**, **quarterly** redemptions, **60-day notice**. An investor — call him the Hartwell Family Office — subscribes **\$5M** on **January 1 of Year 1**.

Hartwell decides in mid-Year-1 that he wants his money back. When can he actually have it? Trace the calendar:

- The **lock-up** runs to **January 1 of Year 2**. He cannot redeem at all before then. The first dealing day his money is eligible for is the one *after* the lock-up expires.
- Redemptions are **quarterly** — March 31, June 30, September 30, December 31. The first eligible dealing day after January 1, Year 2 is **March 31, Year 2**.
- To redeem at the March 31 dealing day he must give **60 days' notice** — so he must file his redemption request by **January 31, Year 2** (60 days before March 31, give or take the documents' day-counting).
- If he files on time, he is paid out at the **March 31 NAV per share**, with cash typically wired within the documents' settlement window — often **15 to 30 days** after the dealing day, sometimes with a small **holdback** (5% to 10%) released after the year-end audit confirms the NAV.

So money Hartwell put in on January 1 of Year 1, and *decided* to pull mid-Year-1, does not reach his bank account until roughly **mid-April of Year 2** — about fourteen to fifteen months after he made the decision. None of this is a trick; it is exactly what the subscription documents said. Figure 5 lays the calendar out.

![Timeline showing an investment, a one-year lock-up expiry, a redemption notice filing, and the quarterly dealing day fourteen months later](/imgs/blogs/liquidity-terms-lockups-gates-side-pockets-5.png)

*The lesson is that the day an investor decides to leave and the day cash actually arrives can be a year or more apart, and that gap is the manager's entire margin for selling assets in an orderly way.*

## Redemption frequency and notice periods

If the lock-up sets *when the door first opens*, redemption frequency and the notice period set *how often it opens* and *how much warning the manager gets each time*.

**Redemption frequency** is the cadence of dealing days. The common menu:

| Frequency | Dealing days per year | Typical strategy fit |
|---|---|---|
| Monthly | 12 | Liquid: macro, CTA, liquid equity L/S |
| Quarterly | 4 | The default for most hedge funds |
| Semi-annual | 2 | Less liquid credit, event-driven |
| Annual | 1 | Distressed, private credit, illiquid |

The trade is direct: more frequent redemption is friendlier to the investor who might need cash, but it forces the manager to keep the book more liquid (to be ready to pay) and exposes the fund to more frequent stampede risk. Less frequent redemption lets the manager hold less-liquid, higher-returning assets, but asks the investor to commit for longer.

**The notice period** is the lead time the manager gets before cash is due, and it is the single most underrated term in the whole set. A 90-day notice on a quarterly fund means the manager always knows, three months ahead, exactly how much cash they must raise — which means they can sell into the *next ninety days of normal trading* rather than dumping everything on the dealing day. A long notice period is, in effect, a small lock-up that resets every time: it converts a sudden cash demand into a scheduled, plannable one. Funds running anything less than perfectly liquid books should fight for the longest notice period the market will bear, precisely because it is the cheapest insurance against a fire-sale.

Notice periods commonly run **30 to 90 days**; 45 to 60 is typical. There is a subtlety: notice is often **irrevocable** once given — file your redemption and you are committed, even if the fund rallies before the dealing day. Some funds allow a short window to rescind; many do not, because revocable notice destroys the planning value (the manager raised cash for a redemption that then evaporated).

A second subtlety is the relationship between frequency, notice, and the *real* worst-case liability. The number that matters is not "monthly liquidity" but "what is the most cash I could be forced to pay on the soonest dealing day, and can I raise it without a fire-sale?" That is the question Figure 3 walks through — the actual path a single redemption request travels.

A third subtlety is the **holdback**. When a fund pays a redemption, it usually does not pay 100% on the settlement date. It pays most of it — say **90% to 95%** — within the settlement window after the dealing day, and **holds back** the remaining 5% to 10% until the **year-end audit** confirms the NAV the investor redeemed at was correct. The holdback exists because the monthly NAV is an estimate; the audited year-end NAV is the true one, and if the estimate was too high, the fund must claw the overpayment back from the holdback rather than chasing the departed investor for a refund. A redeeming investor should expect, then, that even after the dealing day and the settlement window, a small slice of their capital stays in the fund for months until the audit closes. This is normal and protective — it stops a redeemer who left at an overstated NAV from being overpaid at the stayers' expense.

A fourth subtlety is **equalization and series accounting** — the machinery that keeps the high-water mark and incentive fee fair across investors who subscribed at different NAVs. It is mostly an administrator's problem rather than a liquidity term, but it interacts with redemptions: an investor's *individual* high-water mark, not just the fund's, determines what incentive fee crystallizes when they redeem, and the administrator must track each series or each investor's equalization credit through every subscription and redemption. A founder does not need to run the arithmetic, but should know it exists and that a competent independent administrator is the one who gets it right — another reason the administrator is a first-class part of the structure, not a back-office afterthought.

There is one more dynamic worth making explicit, because it is the engine of the run: the **first-mover advantage** in a mismatched fund. If redeemers are paid in order and the manager funds early redemptions by selling the most liquid assets first, then the investor who redeems *first* gets paid at a clean NAV out of the liquid sleeve, while the investor who redeems *last* gets paid out of whatever illiquid assets are left, at fire-sale marks. That asymmetry is precisely what makes the rational move "redeem before everyone else does," and it is precisely what gates and pro-rata payment exist to neutralize. A fund-level gate that pays everyone the same proportion on the same day destroys the first-mover advantage at its root: there is no longer any benefit to being first in the queue, so the incentive to stampede disappears. Understanding *why* the gate works — that it removes the queue-position advantage — is more useful than memorizing the cap percentage.

![Pipeline diagram of a redemption request flowing through lock-up check, notice period, the dealing day tally, a gate check, and either full payment or deferral](/imgs/blogs/liquidity-terms-lockups-gates-side-pockets-3.png)

#### Worked example: the liquidity-mismatch run

Return to the Meridian credit fund and make the mismatch concrete. Maya set **quarterly redemptions, 45-day notice** on a **\$220M** book whose honest liquidation profile is: \$60M sellable in a week, \$90M in one to two months, and **\$70M** of private placements and bespoke structured credit that take **six months or more** to sell at a fair price. Call the blended honest liquidation horizon "about six months for the whole book."

Now the November 2008 redemption wave hits: **\$30M** of requests for the December 31 dealing day, filed within the 45-day window. Trace the squeeze:

- She has 45 days to raise \$30M. The \$60M liquid sleeve can cover it — *if she is willing to sell her most liquid, highest-quality positions first.* She does, because she has no choice.
- But selling the liquid sleeve to pay leavers means the *remaining* fund is now disproportionately illiquid: the patient investors who stayed are left holding a book that is now **80%+ hard-to-sell**. She has improved her near-term cash at the cost of making the stayers' position worse — the exact dilution the terms were supposed to prevent.
- Word gets around (it always does). The **March 31** dealing day brings **\$45M** of new requests, because investors now suspect that whoever redeems first gets the good assets. She no longer has a liquid sleeve to sell. To pay, she must hit bids on the private placements at fire-sale prices — marks 15 to 25 points below where they were struck.
- Selling at fire-sale prices crushes the NAV, which scares more investors, which brings *more* redemptions at June 30. This is the run. The assets were never worthless; the *liability structure* let a stampede form on a book that could not be sold fast enough to satisfy it.

Had Maya matched her terms to her book — a **two-year hard lock-up, annual redemptions with 90-day notice, and a side-pocket provision for the private placements** — the same \$30M of underlying investor need would have arrived as a *manageable, scheduled* demand against a book she had a year to position. The strategy did not fail. The liquidity match did.

*The lesson is that monthly or quarterly liquidity is not a feature you add to make investors happy; it is a promise about how fast you can sell, and a book that takes six months to liquidate cannot honestly offer it.*

## Gates

A **gate** is a cap on how much can be redeemed at a single dealing day. It is the circuit-breaker that stops one redemption wave from forcing a liquidation of the entire book. Of all the tools, the gate is the one most likely to be misunderstood — investors hear "gate" and think "the manager is trapping my money" — so the mechanics deserve real care.

There are two distinct kinds, and conflating them is a classic error.

An **investor-level gate** caps how much *each individual investor* can take out per dealing day. A 25% investor-level gate means any single investor can redeem at most 25% of their holding each quarter, so a full exit takes four quarters. The point is to stretch each investor's exit over time so the manager can sell in an orderly way. It treats every investor identically and does not depend on what anyone else does.

A **fund-level gate** caps *total* redemptions across all investors at a single dealing day — e.g. 25% of fund NAV. If requests exceed the cap, the available amount is **pro-rated** across everyone who asked, and the unfilled portion is **deferred** to subsequent dealing days (often with priority over later requests). The fund-level gate protects the book as a whole: no single dealing day can drain more than the cap, no matter how many investors run for the door at once.

The distinction is load-bearing. Under an *investor-level* gate, a fund where everyone redeems still loses 25% of NAV per quarter — the gate limits each person but not the aggregate. Under a *fund-level* gate, the fund loses at most 25% of NAV per quarter *in total*, which is the real protection against a run. Most modern fund documents include a **fund-level** gate precisely because the run is an aggregate phenomenon.

Gates are typically set at **10% to 25%** of NAV per dealing day, and the manager usually has *discretion* to invoke them (a "soft" or discretionary gate) or they trigger automatically once requests cross the threshold (a "hard" gate). A discretionary gate is more flexible but puts the decision — and the reputational cost — squarely on the manager.

#### Worked example: a 25% fund-level gate on \$100M of redemption requests

Make the gate arithmetic explicit. The Meridian Fund has **\$100M** of NAV at a December 31 dealing day. Its documents include a **25% fund-level gate**: total redemptions at any one dealing day are capped at 25% of NAV.

This quarter, redemption requests total **\$40M** — 40% of NAV, well over the cap. The gate binds. Walk the mechanics:

- The cap is **25% of \$100M = \$25M**. That is the maximum the fund will pay at this dealing day.
- The \$25M is **pro-rated** across the redeeming investors. Three investors asked: Investor A for \$20M, B for \$12M, C for \$8M — \$40M total. Each receives **25M ÷ 40M = 62.5%** of what they asked: A gets **\$12.5M**, B gets **\$7.5M**, C gets **\$5M**, summing to \$25M.
- The unpaid **\$15M** is **deferred** to the next dealing day (March 31), typically with priority — the deferred requests are honored before any new ones. So A's remaining \$7.5M, B's \$4.5M, and C's \$3M roll forward.
- At March 31, if no new requests arrive and the fund can pay, the deferred \$15M clears. If the gate binds *again* (new requests plus the deferred \$15M exceed 25% of the now-smaller NAV), the deferral cascades further.

Who gets paid and who waits? *Everyone* who asked gets 62.5% now; everyone waits for the rest. The gate does not pick winners — it spreads the pain proportionally and buys the manager time to sell the deferred portion in an orderly way rather than fire-selling \$40M of assets in 45 days. Figure 4 shows the same numbers as a chart.

![Bar chart showing 40 million dollars of redemption requests capped to 25 million dollars paid under a 25 percent gate, with 15 million dollars deferred](/imgs/blogs/liquidity-terms-lockups-gates-side-pockets-4.png)

*The lesson is that a fund-level gate converts a 40% redemption shock into a 25%-per-quarter orderly unwind, paid pro-rata so no investor jumps the queue, which is precisely the dilution-and-run protection the term exists to provide.*

The reputational reality of gating, though, is brutal and worth stating plainly: invoking a gate is a near-irreversible signal. Even when it is the *correct* decision — even when it protects the remaining investors exactly as designed — investors read a gate as "this fund is in trouble," and the gated capital, once it can finally leave, usually does, plus a chunk of the capital that was not even trying to leave. A gate stops the immediate run at the cost of the fund's long-term franchise. This is why gates are a tool of last resort *before* suspension, not a routine lever, and why getting the *lock-up and notice* right up front — so you never have to gate — is the real skill.

## Side pockets

A **side pocket** solves a different problem from the gate. The gate caps *how much* can leave; the side pocket fixes *what an illiquid position is worth* when someone leaves.

The problem: suppose the fund holds an illiquid position — a private placement, a distressed claim, a litigation stake — that has no reliable market price. The administrator has to strike a NAV every month, so the position carries *some* mark, but that mark is an estimate, and it could be badly wrong in either direction. Now an investor redeems. If the position is overmarked, the leaver gets paid out at the inflated price and the stayers are left holding the eventual write-down. If it is undermarked, the leaver is shortchanged and the stayers get a windfall. Either way, the *act of redeeming an unmarkable asset transfers value* between the leaver and the stayer, and there is no fair price to redeem at.

The side pocket removes the asset from that problem entirely. When the manager designates a position as a side pocket, it is **carved out of the main, redeemable book into a separate sub-account**, usually by converting each investor's pro-rata interest in the position into a *special, non-redeemable share class*. The rules that follow:

- Investors who are in the fund *at the moment the side pocket is created* keep their pro-rata interest in it. They cannot redeem that interest — it is locked until the position is **realized** (sold, matured, or written off).
- The main book — the liquid portion — continues to offer normal redemptions at a NAV that *excludes* the side-pocketed position. So redemptions are paid out of the liquid sleeve, at a clean mark, and no one is forced to value the unmarkable asset to process an exit.
- **New investors** who subscribe after the side pocket is created do *not* participate in it. They buy into the liquid book only. This is fair: they were not there when the illiquid bet was made.
- When the position is finally realized, the proceeds (gain or loss) are distributed to the investors who held the side-pocket interest, and any **incentive fee** on a side-pocketed gain typically crystallizes only at realization — not on the interim, made-up mark.

Side pockets are usually **capped** in the documents — e.g. no more than 10% to 20% of NAV may be side-pocketed — precisely because the mechanism is so powerful that an unscrupulous manager could abuse it to hide losses or trap capital. (The abuse case: marking a souring position "illiquid," side-pocketing it, and thereby freezing investor money in a loss they cannot escape. This is why allocators scrutinize side-pocket provisions hard during operational due diligence.)

A few mechanical details separate a clean side-pocket provision from a sloppy one, and they are worth knowing because allocators check them. First, **management fees on a side pocket** should be charged on the *original cost or a conservative mark*, not on an optimistic interim valuation — charging a full fee on a marked-up illiquid position lets the manager collect on gains that have not been realized and might never be. Second, the **trigger for creating a side pocket** should be objective and disclosed (a position becomes side-pocketable when it meets a defined illiquidity test), not a discretionary lever the manager can pull at will on any position that turns against them. Third, the side pocket should **convert back** or distribute cleanly on realization, with the gain or loss flowing only to the investors who held the interest when it was created. A provision that nails these three points is a sign of a manager who has thought about fairness; a vague provision that lets the manager side-pocket anything, charge full fees on guessed marks, and distribute at discretion is a red flag that the tool exists to trap capital rather than to protect it.

It is also worth distinguishing the side pocket from its cousin, the **gate**, because founders sometimes reach for the wrong one. A gate is the right tool when the *whole book* is temporarily hard to sell fast enough — the assets are fine, there is just too much redemption demand for the dealing day. A side pocket is the right tool when a *specific position* is structurally unmarkable — there is no honest price at which a redeemer could fairly exit it. A well-run illiquid fund typically writes *both* into the documents: a fund-level gate to handle aggregate redemption surges, and a side-pocket provision to handle the individual positions that have no market. They solve different problems and a fund touching illiquid assets generally needs both.

#### Worked example: side-pocketing a \$5M illiquid position

The Meridian Fund holds **\$100M** of NAV, of which **\$5M** is a private placement in a fintech lender — a position with no market quote, marked at cost. Redemptions are looming and Maya does not want the leavers to either escape or absorb the eventual outcome of that position unfairly. She side-pockets it.

- The \$5M position is converted into a **side-pocket share class**. Each current investor's pro-rata share of the fintech position moves into it. An investor who owned **10%** of the fund now owns 10% of the **\$5M** side pocket (\$500k of interest) *and* 10% of the **\$95M** liquid book.
- The fund's **redeemable NAV** is now **\$95M** — the liquid sleeve. All redemptions are processed against that clean number. An investor redeeming gets paid their share of the \$95M; their \$500k side-pocket interest stays put.
- Eighteen months later the fintech lender is acquired and the position realizes at **\$8M** — a \$3M gain. That \$3M is distributed to the investors who held the side-pocket interest *at the time it was created* — including investors who have since fully redeemed their liquid stake. The incentive fee on the \$3M gain crystallizes now, at realization, on a real number rather than a guessed mark.

The structure is shown in Figure 6: the \$5M carved out of the \$100M main book into a sub-account that pays out only on realization, while the \$95M liquid sleeve handles redemptions.

![Graph showing a 100 million dollar main book splitting into a 95 million dollar liquid sleeve that handles redemptions and a 5 million dollar side pocket that pays out only when the illiquid stake is sold](/imgs/blogs/liquidity-terms-lockups-gates-side-pockets-6.png)

*The lesson is that a side pocket lets a fund hold genuinely illiquid positions inside a redeemable structure by ensuring the leaver and the stayer share the eventual outcome fairly, rather than the act of redeeming forcing a fake price on an unmarkable asset.*

## Suspension: the nuclear option

When the gate is not enough — when paying even the gated amount would require a fire-sale that destroys value for everyone — the manager invokes the last tool: **suspension** of redemptions, usually paired with suspension of NAV calculation and subscriptions.

Suspension freezes the entire door. No one redeems; often no one can even get an official NAV. It is invoked under a right written into the fund documents (the "suspension of dealings" clause), typically triggered by conditions like: markets for a material part of the portfolio have closed or become disorderly; the fund cannot reasonably value its assets; or processing redemptions would be materially prejudicial to remaining investors. The board of directors (for an offshore fund, the Cayman board) usually has to approve or ratify the suspension, which is one reason an independent board matters.

Suspension is genuinely the nuclear option because of what it costs. A gate signals trouble; a suspension signals *crisis*. It almost always precedes one of two outcomes: an orderly **managed wind-down** (the fund stops trading, sells the book over months or years, and returns capital as assets liquidate), or, rarely, a recovery and re-opening that the investor base nonetheless flees. Very few funds suspend and emerge with their franchise intact. Suspension is the tool you reach for when the choice is no longer "good outcome vs bad outcome" but "destroy value for everyone in a fire-sale vs freeze everyone and unwind slowly."

There is an important honesty point here. Some of 2008's suspensions were exactly the right call — freezing a book whose assets genuinely could not be sold, protecting investors from a forced fire-sale at the bottom. And some were abuse — managers freezing redemptions to keep collecting management fees on trapped capital, or to avoid recognizing losses. The difference is visible in the *details*: a legitimate suspension comes with frequent communication, a clear realization plan, a *suspended* management fee or a fee charged only on realizable NAV, and an independent board overseeing it. An abusive one comes with silence, a full fee on a frozen NAV, and a manager who will not commit to a wind-down timeline. Allocators learned to read those signals the hard way.

#### Worked example: gate, then suspend, then wind down

Trace the Meridian Fund's full path through 2008–2009 to see how the tools escalate. NAV starts at **\$220M**.

- **Q4 2008:** \$30M of requests against a 45-day notice. No gate in the documents (Maya's first mistake). She sells the \$60M liquid sleeve to pay in full. NAV drops to roughly **\$185M** after the payout and markdowns, now disproportionately illiquid.
- **Q1 2009:** \$45M of requests — more than 24% of NAV. *Now* she invokes a **discretionary gate** at 25%, paying about **\$46M** pro-rated (the gate barely binds) and deferring the rest. But paying even that requires hitting bids on private placements at 20-point discounts. NAV craters to **~\$120M** on the markdowns.
- **Q2 2009:** the markdowns scare everyone; requests exceed **50%** of the now-\$120M NAV. Paying the gated 25% would still force a fire-sale of the most illiquid assets at any price, crushing the stayers. The board approves a **suspension**. Redemptions and NAV calculation freeze.
- **2009–2011:** the fund enters a **managed wind-down**. Maya stops taking new risk, sells the book over two years as markets recover, charges a reduced fee on realizable NAV, sends quarterly wind-down letters, and returns capital in tranches as assets realize. Investors ultimately recover far more than they would have in a Q2-2009 fire-sale — but the franchise is gone, and so is Maya's fund.

*The lesson is that the tools form an escalation ladder — notice, then gate, then suspension, then wind-down — and a fund that lacked the lower rungs (Maya had no gate and too-short notice) is forced to climb straight to the top, where the franchise dies even if the investors are eventually made whole.*

## Matching asset and liability liquidity

Everything above reduces to one discipline, and it is worth stating as a rule the founder can actually use: **the redemption terms you offer must be no faster than the speed at which you can liquidate your book at fair prices, with a real margin of safety.** This is the founder's liquidity-matching test, and it should be run *before* the fund launches, not discovered during a crisis.

How to run it, concretely:

1. **Profile your book's liquidation horizon.** For each sleeve of the portfolio, estimate honestly how long it takes to sell *in a stressed market* — not a calm one — without moving the price more than a tolerable amount. Liquid equities and futures: days. Liquid credit: weeks. Less-liquid credit, event-driven, small-cap: one to three months. Distressed, private credit, structured products, litigation: six months to years. The stressed-market estimate is the one that matters, because that is exactly when redemptions cluster.

2. **Set the terms to the *slowest sleeve you cannot side-pocket*.** Your redemption frequency and notice period together must give you enough warning to fund the worst plausible dealing-day redemption by selling assets *within their stressed liquidation horizon*. If your book takes three months to sell in stress, a 90-day-notice quarterly fund is roughly matched; a monthly fund is not.

3. **Use the tools to bridge any residual gap.** A lock-up smooths the early period when the book is being built. A fund-level gate caps the worst-case single-day drain. A side pocket carves out the genuinely unmarkable positions so the rest can offer cleaner liquidity. Suspension is the backstop you hope never to use.

4. **Add a margin of safety.** The mistake is to match the terms to the *median* liquidation speed. Match them to the *stressed* speed, then add a buffer, because correlations go to one in a crisis: the moment you need to sell, so does everyone else holding the same assets, and the liquidation horizon you estimated in calm markets triples. (This is the cross-asset reality of liquidity dry-ups; see [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) for why the diversification that holds in normal times evaporates exactly when you are relying on it.)

The asymmetry to internalize: **erring toward less liquidity is recoverable; erring toward more is fatal.** A fund that offered a longer lock-up than it strictly needed will lose a few investors at the margin and leave some return on the table — annoying, survivable. A fund that offered more liquidity than its assets can deliver will, the first time markets seize, face the run-gate-suspension-winddown cascade that ends the firm. When in doubt, promise less liquidity than you think you can deliver. The structure is the part of the promise that, once broken, cannot be repaired.

This is also where liquidity terms tie directly into the rest of the fund's survival machinery. The terms are written into the [fund documents — the PPM, LPA, and subscription agreement](/blog/trading/hedge-funds/the-fund-documents-ppm-lpa-subscription); they are operationalized through the fund's [cash, treasury, and counterparty-risk](/blog/trading/hedge-funds/cash-treasury-and-counterparty-risk) management (which determines whether you can actually raise the cash a redemption demands); and they are the central decision in [the crisis playbook for 2008 and 2020](/blog/trading/hedge-funds/the-crisis-playbook-2008-and-2020). Get the match right and the other three become routine; get it wrong and they become emergencies.

## Common misconceptions

The terms in this post are surrounded by folk beliefs that get founders and allocators into trouble. Six worth correcting.

**"Long lock-ups are just the manager being greedy — trapping my capital so they keep collecting fees."** Sometimes, but usually the opposite. A lock-up that *matches the strategy* protects the locked investor by keeping the capital base stable and preventing faster money from draining the fund. The greed case is the *mismatch in the other direction* — a long lock-up on a liquid book that did not need it. The honest test is whether the lock-up fits the asset: a one-year lock-up on a distressed-credit book is alignment; a one-year lock-up on a daily-liquid macro book is a money-grab. Judge the *fit*, not the length.

**"A gate is a sign of fraud or that the manager has stolen the money."** No. A gate is a *liquidity* tool, not a *solvency* statement. A perfectly honest fund holding perfectly good but illiquid assets will gate in a redemption surge precisely *because* it is being responsible — refusing to fire-sale good assets to pay early leavers at the expense of everyone who stayed. Madoff did not gate; he paid every redemption on demand, which is exactly what a Ponzi must do to avoid suspicion. The fraud signal is *too much* liquidity offered too smoothly, not a gate invoked in a crisis. That said, a gate *can* be a sign of trouble — the trouble being a liquidity mismatch — so it is a yellow flag worth investigating, not a verdict.

**"Monthly liquidity is always investor-friendly — I should prefer the fund that lets me out fastest."** This is the most dangerous misconception of all, because it is exactly backwards in the case that matters. Monthly liquidity is friendly *to you* only if the fund's *assets* can actually be sold monthly without a fire-sale. If the manager offered monthly liquidity on a book that takes six months to sell, that "friendly" term is a trap that will spring on every investor — you included — the moment markets seize. The fund offering *less* liquidity, matched to its book, is the safer place for your capital. A redemption term you can never safely exercise in a crisis is not a benefit; it is a lie about the strategy. Prefer the fund whose terms match its assets, not the one whose terms are loosest.

**"Side pockets are a scam to lock up my money and hide losses."** They can be abused, which is why allocators scrutinize them — but the mechanism itself is the *fairest* available way to hold an unmarkable asset in a commingled fund. Without a side pocket, every redemption forces a guessed price on the illiquid position, transferring value between leavers and stayers. The side pocket removes that transfer. The abuse case is real (marking a losing position "illiquid" to trap capital), but the defense is in the *details*: a capped side-pocket provision, an independent administrator and valuation policy, realization-based fee crystallization, and board oversight. A clean side-pocket provision is a sign of a *well-run* fund, not a poorly-run one.

**"If a fund suspends, the investors are going to lose everything."** Not necessarily — and often the opposite. A legitimate suspension *protects* value by preventing a fire-sale at the bottom; the suspended-and-wound-down funds of 2008–2009 frequently returned far more to investors than they would have recovered in a Q4-2008 panic sale. What the investors lose is *access* (their capital is frozen, possibly for years) and the *franchise* (the manager's business rarely survives). Those are real costs. But "suspension equals total loss" confuses a liquidity event with an insolvency event. The asset value is usually still there; it is the *liquidity* that vanished.

**"Liquidity terms are boilerplate — copy a competitor's documents and move on."** This is how funds die. Liquidity terms are the single most strategy-specific section of the fund documents, because they must encode *your* book's stressed liquidation horizon, not your competitor's. Copying a liquid equity fund's monthly-liquidity terms onto a credit book is precisely the mismatch that destroyed Maya's Meridian Fund. The terms are not boilerplate; they are the structural expression of your strategy's liquidity, and they deserve as much thought as the strategy itself.

## How it plays out in the real world

Everything above was a live, industry-wide event in 2008–2009, and it is the canonical case study for every term in this post.

Going into the crisis, large parts of the hedge fund industry had drifted into liquidity mismatch. The 2003–2007 bull market and a wave of new capital pushed managers toward higher-yielding, less-liquid assets — structured credit, leveraged loans, mortgage products, private placements, asset-backed securities — while *competition for capital* pushed redemption terms in the *opposite* direction, toward the monthly and quarterly liquidity that allocators (especially funds-of-funds) wanted. Funds were buying six-month assets and selling monthly liquidity. The mismatch was invisible while markets rose and redemptions were calm.

Then Lehman Brothers failed in **September 2008**, the markets for exactly those less-liquid assets seized, and the redemption wave arrived. Funds-of-funds, facing their *own* redemptions, pulled capital from underlying managers. Investors who needed cash anywhere sold what they could, which meant redeeming from their most liquid holdings — including hedge funds that had advertised liquidity. The redemption requests hit books that could not be sold fast enough to meet them, and the run dynamics took over: those who redeemed first got the liquid assets at decent marks; those who waited faced the fire-sale.

The industry's response was the **gating wave**. An unusually large share of funds — across the credit, structured-product, and even some equity strategies — imposed gates, suspended redemptions, or side-pocketed illiquid assets. Estimates of the share of funds restricting redemptions in some form at the peak run roughly from one in five to one in three depending on the strategy mix and the source; the contemporaneous reporting from industry bodies and the financial press put it in that rough range. Figure 7 shows the shape — a normal-year baseline of a few percent spiking to roughly a quarter of funds at the 2008–2009 peak. Treat the exact heights as illustrative; the *shape* — a sharp, broad-based spike when liquidity mismatch met a redemption surge — is the well-documented part.

![Bar chart showing the share of funds restricting redemptions spiking from a few percent in a normal year to roughly a quarter at the 2008 to 2009 peak](/imgs/blogs/liquidity-terms-lockups-gates-side-pockets-7.png)

The aftermath reshaped the industry's terms. Allocators came out of 2008 deeply attuned to liquidity matching — operational due diligence began scrutinizing the gap between a fund's asset liquidity and its redemption terms as a first-order risk, not a footnote. Funds-of-funds, which had been the worst victims of the mismatch (offering their *own* investors monthly liquidity while invested in funds that gated), were structurally damaged and many never recovered. The terms themselves evolved: investor-level and fund-level gates became standard documentation rather than exotic; side-pocket provisions became routine for any fund touching illiquid assets; notice periods lengthened; and the worst mismatches — daily-and-weekly liquidity on illiquid books — largely disappeared from credible launches.

It is worth naming the deeper lesson the named blow-ups reinforce. The funds that *blew up* on a single bad trade — Amaranth on natural gas in 2006, LTCM on convergence trades in 1998 — are the famous cases. But far more funds died quietly in 2008–2009 not because their assets were *wrong* but because their *liquidity terms were wrong*: the assets were fine and even recovered, but the fund did not survive the gap between how fast investors could leave and how fast the book could be sold. The trade was not the killer. The mismatch was. That is the asymmetry every founder should carry: you can survive a bad year of returns, but a liquidity mismatch in a crisis ends the firm even when the portfolio is sound.

The 2020 COVID liquidity shock was a fast-motion replay with a happier ending for most. The March 2020 dash-for-cash hit liquidity hard, some funds gated again, but central-bank intervention restored market liquidity within weeks rather than the years 2008 took. The lesson held in both directions: funds with matched terms and adequate notice periods rode it out; funds that had crept back toward mismatch in the long 2010s bull market got a sharp, brief reminder of why the terms exist.

## When this matters / Further reading

Liquidity terms matter most at two moments, and they are the worst two moments to discover you got them wrong. The first is *the week before launch*, when you set the redemption frequency, notice period, lock-up, gate, and side-pocket provisions in the fund documents — and when the temptation to offer looser liquidity to win the raise is strongest. The second is *the first real crisis*, when the gap between your liability liquidity and your asset liquidity becomes the difference between an orderly quarter and the run-gate-suspension-winddown cascade. By the second moment, the terms are fixed; all you can do is invoke the tools you wrote into the documents at the first.

The discipline is one sentence: **never promise more liquidity than your assets can deliver, and match the terms to the stressed-market liquidation speed of your slowest non-side-pocketable sleeve, with a margin of safety.** Lock-ups, notice periods, gates, side pockets, and suspension are not investor-hostile friction; they are the structural half of the aligned promise, the part that protects the patient investor from the impatient one and lets you hold the assets that actually earn the return. Get the match right and a redemption is a cash-flow event. Get it wrong and it is the end of the firm.

For the connected pieces of the founder's playbook:

- These terms are drafted and disclosed in [the fund documents — the PPM, LPA, and subscription agreement](/blog/trading/hedge-funds/the-fund-documents-ppm-lpa-subscription), which is where the liquidity provisions become legally binding.
- The ability to actually *fund* a redemption without a fire-sale lives in [cash, treasury, and counterparty risk](/blog/trading/hedge-funds/cash-treasury-and-counterparty-risk) — the operational machinery behind the terms.
- Keeping capital from running in the first place is the subject of [investor relations and retention](/blog/trading/hedge-funds/investor-relations-and-retention) — the human side of liability liquidity.
- When a real crisis hits, [the crisis playbook for 2008 and 2020](/blog/trading/hedge-funds/the-crisis-playbook-2008-and-2020) walks through the gating decision and the investor communication in real time.
- And for why the liquidation horizons you estimate in calm markets are wrong in exactly the moment you need them, see [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — the cross-asset reality that turns a matched book into a mismatched one overnight.
