---
title: "Mortgage-backed securities: bonds with negative convexity"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into mortgage-backed securities: how a pool of home loans becomes a pass-through bond, why prepayment risk turns falling rates into bad news, what negative convexity really means, agency versus non-agency, the dollar roll, and how MBS holders can move the whole Treasury market."
tags: ["fixed-income", "bonds", "mbs", "mortgage-backed-securities", "negative-convexity", "prepayment-risk", "securitization", "agency-mbs", "interest-rates", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — a mortgage-backed security (MBS) is a bond built from a pool of home loans, and it behaves like an ordinary bond turned partly inside out: when rates fall, it can actually *hurt* you.
> - An MBS bundles thousands of mortgages into one tradeable **pass-through** bond that forwards homeowners' principal and interest to investors every month.
> - The twist is **prepayment risk**: a homeowner can refinance and repay early, and they do exactly that when rates fall — handing your principal back at the worst possible moment.
> - That gives an MBS **negative convexity**: when rates fall its price flattens into a ceiling (your upside is capped) while a Treasury keeps soaring; when rates rise it falls about as much as anything else. Heads you barely win, tails you lose.
> - **Agency** MBS (Ginnie, Fannie, Freddie) carry a credit guarantee, so you bear mostly rate and prepayment risk; **non-agency** MBS hand you the credit risk too — the fuse that lit 2008.
> - MBS investors hedge their shifting duration by trading Treasuries, and that "convexity hedging" can amplify the very rate moves that started it — so this is not just an investor's quirk, it can move the whole market.

Here is a fact that breaks most people's intuition about bonds. You own a bond. Interest rates fall. Normally that is a celebration — falling rates push bond prices *up*, and the longer the bond, the bigger the party. But you own a particular kind of bond, and instead of soaring, its price barely budges. Worse, a chunk of your money comes back early, in cash, and the only place to put it is into new bonds that now pay almost nothing. Rates fell, and you are *poorer* for it. What kind of bond does that?

A mortgage-backed security. The thing you own is, underneath, a few thousand people's home loans, and those people have a right you cannot take away from them: the right to refinance. When rates drop, they exercise it. They pay off their old expensive mortgages and take out cheap new ones, and the cash that flows back to you — early, unwanted, at the worst possible time — is the early repayment of loans you were counting on to keep paying you 6.5% for years. This single feature, the homeowner's option to prepay, is what makes the MBS the biggest, strangest, most quietly important corner of the bond market after Treasuries themselves.

![A pool of three home loans of three hundred thousand and two hundred fifty thousand and four hundred thousand dollars flowing into a one billion dollar mortgage pool, then wrapped by an agency guarantee into a pass through bond that pays principal and interest to two investors each month](/imgs/blogs/mortgage-backed-securities-bonds-with-negative-convexity-1.png)

The diagram above is the mental model for the whole post. Many small, ordinary home loans go in on the left. They get pooled — bundled into one big basket of similar loans. An agency wraps a guarantee around the basket, and out the other side comes a single bond, the **pass-through**, that you can buy a slice of. Each month, as homeowners make their mortgage payments, the cash flows through the structure to you: a little interest, a little principal, repeated for as long as the loans last. Hold that picture. Everything that follows — prepayment, average life, negative convexity, the dollar roll — is just a consequence of the fact that the cash flows in this bond come from human beings who get to decide when to pay you back. (Everything here is educational, not investment advice; the goal is to understand the machine, not to tell you what to buy.)

## Foundations: the building blocks you need first

Let's assemble the vocabulary from zero. Some of this overlaps with the rest of [the bond series](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction); if a term is familiar, skim it, but do not skip, because the whole strangeness of an MBS lives in how these pieces fit together.

**A bond is a promise to pay a stream of cash.** When you buy a bond you are lending money to an **issuer** in exchange for a schedule of future payments: periodic **coupons** (the interest) and the **face value** (also called **par** — almost always \$1,000 per bond) returned at **maturity** (the final date). A plain US Treasury is the cleanest example: it pays you a fixed coupon twice a year and then, on one known date, hands back every dollar of principal in a single lump. A bond shaped like that — all the principal at the end, on a known date — is called a **bullet bond**. Hold that word; the MBS is the opposite of a bullet, and the contrast is the whole story.

**Price and yield move on a seesaw.** The coupon printed on a bond never changes. What changes every second is its **price** and, mirror-image to it, its **yield** — the single annual return that makes the bond's future cash flows worth exactly its current price. When market yields rise, the price of an existing bond *falls*; when yields fall, its price *rises*. That inverse link is [the price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), and it is the engine behind every chart in this post.

**Duration is how hard the seesaw tips.** **Modified duration** (written `ModDur`) has a concrete meaning: it is roughly the *percentage price change for a 1% change in yield*. A bond with a duration of 5 loses about 5% of its value when yields rise 1%, and gains about 5% when yields fall 1%. Duration is the single most useful risk number in fixed income — but, as the post on [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story) shows, it draws a straight line through a relationship that is actually curved.

**Convexity is the curve in that line.** For an ordinary bond the price–yield relationship is not a straight line but a gentle bow that always sits *above* duration's straight-line estimate. That curvature is **positive convexity**, and it is a quiet gift: it means your gains come out a little bigger than duration promised and your losses a little smaller than duration threatened. This whole post is about a bond where that gift runs in *reverse*.

**A basis point** is one hundredth of a percent — 0.01%. Rates move in basis points ("bps"): a 25 bps cut is a quarter of one percent; a "1% rate move" is 100 bps, a large move that mostly happens in cycles and crises. And **the mortgage rate** itself is just a Treasury yield plus a spread — the extra yield lenders demand to lend against a house instead of to the government. When the 10-year Treasury yield falls, mortgage rates usually follow, which is exactly why a Treasury move ends up reaching into your MBS through millions of refinancing decisions. (For the policy plumbing behind that, see [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).)

**Securitization** is the act in the diagram: pooling many small, illiquid loans and issuing a single tradeable security backed by them. A **pass-through** is the simplest securitized product — it literally passes the pool's cash through to investors, pro-rata, with no reshuffling. (Reshuffled versions, called CMOs, come later.)

With those in hand, here is the one sentence that motivates the entire post: **an MBS is a bond whose borrowers hold an option to repay early, and they exercise it against you exactly when a normal bond would be doing best.** That option is the source of every weird behavior we are about to unpack.

## How a pool of mortgages becomes a bond

Start with a single mortgage. A family borrows \$300,000 to buy a house, agreeing to pay it back over 30 years at a fixed rate of, say, 6.5%. Each month they send the bank a check that covers two things: the **interest** on the balance they still owe, and a slice of **principal** that chips away at the balance. Early on, almost all of the check is interest; near the end, almost all of it is principal. That blended monthly payment — interest plus a bit of principal, every month — is the cash flow at the heart of everything.

One mortgage is a terrible investment to hold by itself. It is illiquid (you cannot sell one family's loan on an exchange), it is lumpy (if that one family defaults or pays off, your whole investment changes), and it is a hassle (someone has to collect the checks). So the system pools them. A bank or an agency gathers thousands of similar loans — same rough rate, same type, same vintage — into one big basket worth, say, \$1 billion, and issues a single security against the whole basket. Now the cash flows are averaged across thousands of households: one family prepaying barely registers, defaults are diluted, and the security is liquid because it trades as one standardized instrument. That is the figure-1 pipeline: small loans in, one pass-through bond out.

The word **pass-through** is literal. Every month the loan servicer collects all the homeowners' checks, takes a small fee, and *passes the rest through* to the bondholders in proportion to how much of the bond they own. If you own 1% of the pass-through, you get 1% of every dollar of interest and 1% of every dollar of principal the pool collects that month — including 1% of any principal that comes back early because someone refinanced, sold their house, or simply paid extra. This is the crucial difference from a Treasury. A Treasury pays you interest and returns all your principal on one date you knew in advance. An MBS dribbles your principal back to you month after month, in amounts nobody can predict, because they depend on what thousands of strangers decide to do with their houses and their loans.

#### Worked example: one month of a pass-through

*Setup.* You own \$100,000 of a \$1 billion agency pass-through. The pool's loans carry a 6.5% coupon (after the servicing fee, say you receive a 6% **pass-through rate**). In a normal month, the pool collects its scheduled interest, its scheduled principal, and a little extra principal from people who prepaid.

*Step 1 — your share.* You own \$100,000 / \$1,000,000,000 = 0.01% of the pool. Whatever the pool collects, you get 0.01% of it.

*Step 2 — the interest piece.* At a 6% annual pass-through rate, one month of interest on your \$100,000 balance is \$100,000 × 6% / 12 = \$500. That part is clean and predictable.

*Step 3 — the principal piece.* Suppose the pool returns 0.8% of its remaining principal this month (scheduled amortization plus some prepayment). On your \$100,000 that is \$800 of principal handed back. Your balance is now \$99,200, so next month's \$500 interest will be slightly smaller.

*Step 4 — the catch.* That \$800 of principal is *real cash in your pocket*, but it is no longer invested at 6%. You have to find somewhere new to put it. *An MBS quietly shrinks itself every month, and the rate at which it shrinks — which you do not control — is the entire ballgame.*

The number that summarizes "how fast is the pool shrinking" is the **prepayment speed**. The market quotes it two main ways. **CPR** (conditional prepayment rate) is the annualized fraction of the pool's principal that prepays in a year — a 6% CPR means roughly 6% of the remaining balance is expected to vanish to prepayment this year. **SMM** (single monthly mortality) is the monthly version. And the industry's reference yardstick is the **PSA** model: "100% PSA" is a standard assumed ramp where prepayments start near zero on a brand-new pool and rise to a steady 6% CPR by month 30. "150% PSA" means one-and-a-half times that speed; "50% PSA" means half. You do not need the arithmetic; you need the idea: **prepayment speed is a guess about human behavior, and the whole valuation of the bond hangs on that guess.**

#### Worked example: what a CPR number actually does to your money

*Setup.* You hold \$100,000 of an agency pool. We will look at the same pool under three prepayment speeds — slow (6% CPR), normal (15% CPR), and a refinance-wave spike (40% CPR) — and ask how much principal comes back to you in the first year from prepayment alone (ignoring the small scheduled amortization, to isolate the option's effect).

*Step 1 — slow, 6% CPR.* Roughly 6% of your balance prepays: about \$6,000 comes back early this year. Your bond stays long and your high coupon keeps working on \$94,000 of it. This is the quiet, range-bound world where MBS feel like generous bonds.

*Step 2 — normal, 15% CPR.* About \$15,000 comes back. You are reinvesting a meaningful chunk every year, but the bond still has years of life left.

*Step 3 — spike, 40% CPR.* About \$40,000 — nearly half your position — comes back in a single year, and the wave usually runs for more than one year. Within two to three years most of your \$100,000 is gone, returned as cash into the low-rate market that caused the wave.

*Step 4 — the takeaway.* The exact same bond hands you back \$6,000 or \$40,000 depending only on what rates did to refinancing incentives. *The CPR is not a property of the bond; it is a forecast of a crowd, and the crowd moves hardest in exactly the direction that hurts you.*

| Scenario | Prepayment speed (CPR) | Principal back in year 1 (per \$100k) | What it means for you |
|---|---|---|---|
| Rates rise, no refi incentive | ~4–6% | ~\$5,000 | Bond stays long; you keep the high coupon — but you are stuck if rates rose |
| Range-bound, normal turnover | ~10–15% | ~\$12,000 | Steady trickle back; manageable reinvestment |
| Rates fall, refinance wave | ~30–45% | ~\$40,000 | Half your money back fast, to reinvest at the new low rate |

## Prepayment: why falling rates return your money early

Now we reach the option at the center of everything. A mortgage borrower has the right to pay off their loan early, in full, at any time, with no penalty (in the US market, for the standard conforming loan). They will do it for three reasons, and the order matters.

The first and biggest is **refinancing**. If a family is paying 6.5% and rates fall so that they could get a new loan at 4.5%, they refinance: they take out a fresh 4.5% loan, use it to pay off the old 6.5% loan in one lump, and lower their monthly payment. From your seat as the bondholder, that 6.5% loan you owned a slice of just *disappeared* — repaid in full — and the principal landed back in your lap. The second reason is **housing turnover**: people sell their houses (job moves, upsizing, downsizing, divorce, death), and selling a house pays off its mortgage regardless of rates. The third is **curtailment**: people who simply pay a little extra each month. Turnover and curtailment churn along at a slow background rate no matter what; refinancing is the wild card, and it is the one that responds violently to interest rates.

![A before and after comparison showing that when rates stay high a homeowner keeps a cheap six point five percent loan and you collect coupons for years, but when rates fall the homeowner refinances into a four point five percent loan, the old loan is paid off early, and your principal comes back years early to reinvest at the new low rate](/imgs/blogs/mortgage-backed-securities-bonds-with-negative-convexity-2.png)

The figure above is the whole asymmetry in one picture. On the left, rates stayed high: the homeowner has no reason to refinance a loan that is already cheaper than the market, so they keep paying, and you keep collecting your 6.5% for years — slow, long, friendly cash flow. On the right, rates fell: the homeowner refinances, the old loan is paid off in full, and your principal comes back early, just as new loans (and new bonds) pay much less. The same loan, the same homeowner, two opposite outcomes — and notice that the *good* outcome for you (keeping the high coupon) only happens when rates *rise or stay put*, while the *bad* outcome (early repayment into a low-rate world) happens precisely when rates *fall*. That is backwards from how a normal bond treats you, and it is the seed of negative convexity.

#### Worked example: the refinance wave hits your position

*Setup.* You hold \$100,000 of a 6.5% pass-through. Today the prevailing 30-year mortgage rate is also about 6.5%, so prepayments run at a sleepy "background" pace — say 6% CPR, mostly from people moving house. The bond's modeled average life is around seven years.

*Step 1 — rates fall 2%.* The 30-year mortgage rate drops to 4.5%. Suddenly every homeowner in your pool is paying 2 percentage points more than the market. The incentive to refinance is enormous, and refinancing waves are fast: CPR can spike from 6% to 30%, 40%, even higher in a sharp rally.

*Step 2 — what that does to your cash.* At 40% CPR, roughly 40% of the pool's remaining balance prepays in a year. Within a couple of years, most of your \$100,000 of principal has come back to you in cash. The seven-year bond you thought you owned behaved more like a two- or three-year bond.

*Step 3 — the reinvestment trap.* You now hold a big pile of returned principal, and the world you are reinvesting it into pays 4.5%, not 6.5%. The very thing that made your old loans valuable — their high coupon — is exactly why they got refinanced away. *You were forced to sell your best asset, early, into the worst possible market, and nobody asked your permission.* This is [reinvestment risk](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield) in its purest, most painful form.

The mirror image matters just as much. Suppose instead rates *rose* 2%, to 8.5%. Now nobody refinances a cheap 6.5% loan — why would you trade a 6.5% mortgage for an 8.5% one? Prepayments slow to a trickle (turnover only), the pool's principal comes back even more slowly than expected, and your money stays locked up in a below-market bond for *longer* than you planned. So when rates rise, the bond gets *longer* (bad — you are stuck); when rates fall, it gets *shorter* (also bad — you are flushed out). The MBS lengthens when you wish it were short and shortens when you wish it were long. There is a name for each half of that, and we will get to them. First, the centerpiece.

## Negative convexity: the price ceiling that ambushes you

Here is the claim, stated plainly: **as rates fall, a normal Treasury's price keeps climbing, but an MBS's price flattens out into a ceiling.** The reason is the option you now understand. When rates fall, a normal bond's fixed cash flows become more valuable and its price soars. But an MBS's cash flows are *not* fixed — falling rates trigger the refinancing wave that returns your principal at par (around \$1,000 per bond), so the bond cannot rise much above par. Why would anyone pay \$1,150 for a bond that is about to repay them \$1,000? The looming prepayment caps the price. The upside gets chopped off right where a normal bond's upside takes flight.

![Two bond price curves plotted against the interest rate with rates falling to the left, a solid Treasury curve that keeps rising steeply toward the upper left and a dashed mortgage backed security curve that flattens into a ceiling near par as rates fall, with the gap between them labeled as lost upside](/imgs/blogs/mortgage-backed-securities-bonds-with-negative-convexity-3.png)

This is the figure to tattoo on your memory, and it is the direct sequel to the negative-convexity chart in [the convexity post](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story). The horizontal axis is the interest rate, with rates *falling to the left*. The vertical axis is price. The solid line is a normal Treasury: as you move left (rates down) it bows *upward*, climbing faster and faster — that is positive convexity, the friendly curve. The dashed line is the MBS: starting from today's par price, it rises a little as rates fall, then *flattens*, running into a ceiling near par because the refinancing wave is about to hand you your money back. The shaded region between the two curves, on the left side, is your **lost upside** — the gain a normal bond would have delivered that the MBS swallows. To the right, where rates rise, the two curves fall together: when rates rise nobody refinances, so the MBS behaves much like a normal bond and drops in price. You keep the full downside and lose most of the upside. *That* is negative convexity: the curve bends the wrong way.

Recall the friendly behavior of a normal bond from the convexity post: it gains *more* than duration predicts and loses *less* than duration predicts — a one-sided error that always works in your favor. An MBS reverses it. It gains *less* than duration predicts (the ceiling) and loses about as much, or more, than duration predicts (the extension we will see next). The error works *against* you on both sides. In the language of options, you are **short an option** — you have effectively sold the homeowner the right to repay early, and that option gets exercised against you exactly when it costs you most.

#### Worked example: \$100,000 across a 1% move each way

*Setup.* You hold a \$100,000 MBS position and a \$100,000 position in a plain Treasury with the same starting price (par) and roughly the same headline duration of about 7. We will move rates 1% in each direction and reprice both, the way a risk desk would.

*Step 1 — rates fall 1%.* The Treasury, with positive convexity, rises a touch more than 7%: to about \$107,300. The MBS, with its prepayment ceiling, rises only to about \$101,500 — a measly +1.5%. The refinancing wave that a 1% drop kicks off caps the gain. You made \$1,500 where the Treasury made \$7,300.

*Step 2 — rates rise 1%.* Now nobody refinances. The MBS extends and falls almost the full amount: to about \$92,800, a loss of \$7,200. The Treasury, cushioned by positive convexity, falls a bit less: to about \$93,300, a loss of \$6,700.

*Step 3 — read the asymmetry.* The Treasury gained \$7,300 but only lost \$6,700: heads-you-win-a-little-extra. The MBS gained only \$1,500 but lost \$7,200: heads-you-barely-win, tails-you-lose-it-all. *Same duration on paper, opposite curvature in reality — and the difference, roughly \$5,800 of foregone gain on the rally, is the price of being short the homeowner's option.*

This is why duration alone is a lie for an MBS, and why the desk that values these bonds uses a different number. The duration you compute from a *fixed* cash-flow schedule is meaningless for a bond whose schedule shape-shifts with rates. Instead, traders use **effective duration** and **effective convexity**: they bump rates down a bit, let the prepayment model re-forecast the cash flows and reprice the bond; bump rates up, re-forecast, reprice; and read the curvature straight off the two new prices. (The convexity post walks through that bump-and-reprice computation in detail.) For an MBS, that measured effective convexity comes out *negative* — the formal fingerprint of the ceiling you see in the chart.

The cleanest way to see the whole trap at once is to lay all three scenarios side by side: rates down, flat, and up, with the prepayment speed, the average life, and both bond prices in each row.

![A table of one MBS across three rate moves, when rates fall one percent prepayment is fast at about twenty five percent CPR the average life contracts to about three years the MBS price rises only to one thousand fifteen while the Treasury rises to one thousand seventy three, when rates are flat both sit at par, and when rates rise one percent prepayment is slow the average life extends to about ten years and the MBS falls to nine hundred twenty eight while the Treasury falls only to nine hundred thirty three](/imgs/blogs/mortgage-backed-securities-bonds-with-negative-convexity-7.png)

Read the table top to bottom and the trap is undeniable. On the rally (top row) the prepayment speed jumps, the average life collapses, and the MBS price barely lifts off par while the Treasury runs away to the upside. On the selloff (bottom row) prepayments stop, the average life drags out, and the MBS falls about as hard as the Treasury — slightly harder, in fact, once you account for the extension. The flat middle row is the only place the two instruments look alike. Every cell in that table is a restatement of one fact: **you are short the homeowner's option, and the option is worth most to them precisely when its absence would have been worth most to you.**

### The homeowner's option is worse than a callable bond's

It is worth pausing on *why* the MBS's negative convexity is, if anything, nastier than a [callable bond](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story)'s — because both are "short an option," but the options are not equally well-behaved. A callable corporate bond gives the *issuer* the right to buy the bond back at a set call price, and a corporate treasurer is a sophisticated, rational actor: they call when, and only when, it is clearly profitable for them, and they do it cleanly, on a call date, all at once. You can model that decision with cold precision. The homeowner's option is messier in every direction, and the messiness mostly costs you.

Homeowners are *not* perfectly rational. Some refinance the instant rates drop a quarter point; some never refinance even when it would obviously save them thousands, out of inertia, fear of paperwork, or not realizing they can. Some prepay for reasons that have nothing to do with rates at all — a job move, a divorce, a death, a windfall — so principal comes back even when no rational rate-driven exerciser would send it. This **prepayment "burnout"** (the tendency of a pool that has already seen a refinance wave to prepay more slowly afterward, because the rate-sensitive borrowers have already left) and this irrational dispersion mean the option is exercised in a fuzzy cloud rather than at a clean threshold. The result is that the MBS's price ceiling is *soft and uncertain* rather than a hard line, and a whole industry of prepayment modelers exists to forecast a crowd's behavior that no formula can fully capture. You are short an option whose exercise depends on millions of human decisions — and you carry the risk that your model of those decisions is wrong, on top of the convexity itself.

## Extension and contraction: the two faces of the same risk

The price ceiling has a twin: the bond's *life* moves the wrong way too. Because an MBS pays principal back gradually and unpredictably, it does not have a fixed maturity in any useful sense. Instead, traders summarize it with **average life** (also called **weighted-average life**): the average number of years until each dollar of principal is expected to come back, weighted by how much comes back when. A 30-year mortgage pool might have an average life of seven years, because so much principal trickles back early through turnover and curtailment that you do not actually wait three decades for your money.

The problem is that average life is not a constant — it *moves with rates*, and it moves in the direction that hurts.

![A single downward sloping curve of average life in years against the interest rate with rates falling to the left, showing the average life shrinking from about ten years to about three years as rates fall, with a contraction zone marked where rates fall and prepayments speed up and an extension zone marked where rates rise and prepayments stop](/imgs/blogs/mortgage-backed-securities-bonds-with-negative-convexity-4.png)

Read the curve from the center outward. Today, at a 4% rate, the bond's average life is about seven years. Move *left* — rates fall — and prepayments speed up, so the principal floods back fast and the average life *shrinks*, down toward three years. That is **contraction risk**: the bond gets shorter just when you would have wanted it long (because long bonds win most when rates fall). Move *right* — rates rise — and prepayments dry up, so the principal trickles back slowly and the average life *stretches*, out toward ten years or more. That is **extension risk**: the bond gets longer just when you would have wanted it short (because long bonds lose most when rates rise). The bond contracts into your face on the way down and extends into your face on the way up. Both faces of the same option, both pointing the wrong way.

#### Worked example: the average life that won't sit still

*Setup.* You buy an MBS today expecting a seven-year average life. You are a pension fund that needs to match a seven-year liability — you want your money back in roughly seven years, not three, not twelve.

*Step 1 — rates fall 2%.* A refinance wave hits. Average life collapses to about three years. Most of your money is back in cash within three years, and you are reinvesting it at the new low rate — your carefully matched seven-year asset turned into a three-year asset and broke your hedge. *(This is precisely the problem [immunization and duration matching](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge) tries to avoid, and why MBS make the matching job genuinely hard.)*

*Step 2 — rates rise 2%.* Refinancing stops. Average life stretches to about ten or eleven years. Now your money is locked up for far longer than your liability, in a bond that is worth less than you paid, and you cannot get out without selling at a loss.

*Step 3 — the bind.* In *both* scenarios your seven-year asset turned into the wrong length at the wrong time. *Extension and contraction are not two separate risks — they are the same negative-convexity option viewed from the two ends of a rate move, and there is no rate path on which the option works in your favor.*

## The uncertain cash-flow stream: an MBS is not a bullet bond

Step back and compare the *shape* of the cash flows, because that shape is the deepest difference between an MBS and a Treasury, and it is the thing that confuses every newcomer.

![A two panel cash flow comparison, the top panel a bullet Treasury bond paying six steady forty dollar coupons then a single large one thousand forty dollar lump of principal and coupon at maturity, the bottom panel an MBS paying lumpy unpredictable amounts including a large three hundred dollar refinancing spike that returns capital years early and then a declining tail of smaller payments](/imgs/blogs/mortgage-backed-securities-bonds-with-negative-convexity-5.png)

The top panel is a bullet Treasury: a row of identical, predictable coupons, and then one big block at the end — all the principal plus the last coupon, on a date you knew the day you bought it. There is no guessing. You can build a portfolio around it because you know exactly what it pays and when. The bottom panel is the same dollar amount invested in an MBS, and look how different it is: every payment blends interest and principal, the amounts shrink over time as the balance amortizes, and — crucially — there is an unpredictable *spike* in the middle (drawn in amber) where a refinancing wave returns a chunk of principal years early. You did not choose that spike, you could not predict its timing, and it landed in a year when reinvesting was unattractive.

This is why an MBS is genuinely harder to own than a Treasury even before you get to the negative convexity. With a bullet, the only real risk is that rates move and the price moves. With an MBS, the *cash flows themselves* are a moving target. The bond can mature in three years or twelve. The reinvestment problem is constant rather than concentrated at maturity. And every analytic — yield, duration, convexity — has to be computed *through a prepayment model* that is itself a guess. The whole instrument is a bet not just on rates, but on a forecast of human behavior in response to rates.

#### Worked example: same coupon, different fates

*Setup.* You put \$100,000 into a 6% Treasury and \$100,000 into a 6% MBS on the same day. Both yield 6% on paper. Three years pass and rates fall 2%.

*Step 1 — the Treasury.* It paid you \$6,000 a year, on schedule, and its price rose because rates fell. Your \$100,000 of principal is still invested at 6%, still earning the high coupon, and the bond is worth *more* than you paid. Falling rates were good news, exactly as the textbook promises.

*Step 2 — the MBS.* The 2% drop set off a refinance wave. Over those three years, perhaps \$70,000 of your \$100,000 principal came back early. You collected your 6% only on a shrinking balance, and you had to reinvest the returned \$70,000 at the new 4% rate. Your blended return drifted down toward 4–5%, and the bond's price never got to enjoy the rally — the ceiling held it near par.

*Step 3 — the lesson.* Same headline yield, same falling-rate scenario, opposite outcomes. *The Treasury's yield was a promise; the MBS's yield was a hope, contingent on homeowners not doing the one thing that falling rates make them want to do.*

## Agency versus non-agency: who stands behind the loans

So far we have ignored the other big risk a bond can carry: the risk that the borrower simply doesn't pay — [credit risk](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back). Whether your MBS carries credit risk depends entirely on one fork in the road: agency versus non-agency.

![A comparison matrix of agency versus non agency mortgage backed securities across who guarantees it, the credit risk to you, the underlying loans, the yield and spread, and the main risk you bear, showing agency MBS guaranteed by Ginnie Fannie and Freddie with near zero credit risk and non agency private label MBS leaving you the credit risk of jumbo subprime and Alt-A loans](/imgs/blogs/mortgage-backed-securities-bonds-with-negative-convexity-6.png)

**Agency MBS** are backed by one of three entities. **Ginnie Mae** (the Government National Mortgage Association) carries the *full faith and credit* of the US government — its guarantee is as good as a Treasury's. **Fannie Mae** and **Freddie Mac** (the two big "government-sponsored enterprises," or GSEs) carry an *implicit* government backstop that became very explicit when the government placed both into conservatorship in 2008 and stood behind them. All three guarantee that you get your principal and interest *even if homeowners default* — they make you whole on credit losses. So when you own agency MBS, you have effectively been handed a credit guarantee, and the only risks left to you are the rate risk and the prepayment/negative-convexity risk this whole post is about. That is why agency MBS are considered nearly as safe, credit-wise, as Treasuries — and why the Federal Reserve buys them by the trillion (more on that later).

**Non-agency MBS** (also called **private-label** MBS, or PLS) have no such wrap. A private bank or issuer pools loans that do not qualify for an agency guarantee — **jumbo** loans (too big for agency limits), **subprime** loans (borrowers with weak credit), or **Alt-A** loans (somewhere in between, often with missing documentation) — and sells securities backed by them with *no* government guarantee. Here you bear the credit risk yourself: if homeowners default and the foreclosure sales don't cover the balance, *you* eat the loss. To make that palatable, non-agency deals are usually carved into **tranches** (the topic of CMOs below), where junior tranches absorb the first losses to protect senior ones, and the whole structure pays more yield than agency paper as compensation. The extra yield is the price of the credit risk you are taking on. And as 2007–2008 demonstrated with brutal clarity, that credit risk — concentrated in subprime non-agency MBS — is exactly what detonated the global financial crisis.

#### Worked example: the yield pickup and what it pays for

*Setup.* An agency MBS yields 5.5%. A comparable non-agency MBS, backed by lower-quality loans, yields 7.5%. You are tempted by the extra 2 percentage points — on \$100,000 that is \$2,000 a year more.

*Step 1 — what the 2% extra is paying for.* It is not free money. It is compensation for the credit risk the agency wrap would have removed. If 5% of the pool's loans default and recoveries are 50%, the pool loses 2.5% of principal — wiping out more than a year of that extra yield.

*Step 2 — the tail.* In a normal year, defaults are low and you happily pocket the extra \$2,000. In a housing downturn, defaults cluster (everyone gets squeezed at once), and the losses can run to 10%, 20%, or more of a junior tranche's principal — far more than the yield pickup ever paid you.

*Step 3 — the lesson.* *The agency-versus-non-agency choice is the credit-risk dial: agency hands you a guarantee and leaves only rate and prepayment risk; non-agency pays you more to take the credit risk yourself — and credit risk, like prepayment, shows up all at once at the worst time.* For the broader version of this trade-off, see [investment grade vs high yield](/blog/trading/fixed-income/investment-grade-vs-high-yield-the-great-divide).

## CMOs: slicing the pool to redistribute the risk

A plain pass-through gives every investor the same blend of prepayment risk — everyone gets the same pro-rata trickle of early principal. But not every investor wants the same thing. A money-market fund wants its cash back fast and certain; a pension fund wants a long, stable stream. The **collateralized mortgage obligation (CMO)** exists to serve them both out of the same pool, by *redistributing* the prepayment risk rather than removing it.

A CMO carves the pool's cash flows into sequential **tranches**. In the simplest "sequential-pay" structure, all the principal — scheduled *and* prepaid — goes to Tranche A first until A is completely paid off; only then does principal start flowing to Tranche B, then C, and so on. Tranche A therefore has a short, relatively certain life (it gets the early money, including the prepayment spikes), while the last tranche, often called the **Z-tranche**, has a long life and absorbs the leftover uncertainty. More elaborate structures create a **PAC** (planned amortization class) tranche that promises a stable, scheduled paydown across a *range* of prepayment speeds, with the variability dumped onto a paired **support** (or "companion") tranche that soaks up whatever is left. The support tranche has wildly uncertain cash flows and brutal negative convexity; the PAC, in exchange, behaves almost like a normal bond — as long as prepayments stay inside the band the structure was designed for.

The essential point: **a CMO does not make prepayment risk disappear; it concentrates it.** Every dollar of negative convexity the pool contains still exists — it has just been pushed onto whoever bought the support tranche. Buy a PAC and you have bought relative safety; buy the support tranche and you have bought a magnified version of the very ceiling-and-extension behavior this post is about, at a higher yield. The crisis-era cautionary tale is the **interest-only (IO) strip**: a tranche that receives *only* the interest the pool throws off and *no* principal. Because interest accrues only on the balance that is still outstanding, an IO strip is destroyed by fast prepayment — when everyone refinances, the balance vanishes and the interest stream with it. An IO is so prepayment-sensitive that its price often *rises* when rates rise (slower prepays, more interest collected), making it one of the few bonds with a negative duration — useful as a hedge, lethal as a naive holding.

#### Worked example: who eats the refinance wave in a CMO

*Setup.* A \$1 billion pool is split into a \$700 million PAC tranche and a \$300 million support tranche. The PAC promises a stable paydown as long as prepayments run between roughly 100% and 250% PSA.

*Step 1 — normal speeds.* Prepayments come in at 150% PSA, inside the band. The PAC pays down exactly on its planned schedule; the support tranche absorbs the modest variability. Everyone gets what they expected.

*Step 2 — a refinance wave.* Rates plunge, prepayments spike to 500% PSA. A flood of early principal arrives. The PAC still pays down on schedule (that was the promise), so *all* the excess early principal is shoved onto the support tranche, which gets paid off far faster than its holder wanted — its life contracts violently and it gets flushed out into a low-rate market.

*Step 3 — extension.* Later, rates rise and prepayments collapse to 50% PSA. Now there is *too little* principal; the support tranche, already shrunken, extends dramatically to make up the difference. *The PAC holder bought stability by selling the worst of the negative convexity to the support holder — the risk did not vanish, it found a new owner.*

## The dollar roll: a financing trade unique to MBS

There is one more piece of MBS machinery worth understanding, because it is unique to this market and it explains a lot of how the market actually trades: the **dollar roll**. Most agency MBS trade in a forward market called **TBA** ("to-be-announced"), where you agree to buy a certain type of MBS — a 30-year 6% Ginnie pool, say — for delivery next month, *without knowing the exact pools you will receive*. You know the coupon, the agency, and the face amount; the specific loans are announced just before settlement. This standardization is what makes agency MBS so liquid: millions of slightly different pools all trade as one fungible contract.

A dollar roll is a pair of TBA trades that together act like a short-term loan. You *sell* a TBA position for this month's settlement and simultaneously agree to *buy* the same position back for next month's settlement, at a slightly lower price. In effect, you have handed your bonds to a counterparty for a month and gotten cash, agreeing to take the bonds back later — exactly like using the bonds as collateral for a one-month loan (a [repo](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction), in spirit). The price difference between the two legs — the "drop" — is your financing cost, and because the buyer of your bonds collects a month of coupon and principal that you give up, the economics can sometimes make rolling *cheaper* than holding outright. When demand to borrow specific MBS is high, a roll can even trade "special," financing your position below the normal rate. Mortgage REITs and dealers use dollar rolls constantly to finance large MBS holdings cheaply; it is one of the quiet plumbing trades that keeps the multi-trillion-dollar agency market liquid.

#### Worked example: financing a position with a roll

*Setup.* You own \$10 million face of a 30-year 6% agency TBA and you want to keep the exposure but free up cash for a month.

*Step 1 — the two legs.* You sell the \$10 million TBA for September settlement at \$101.00 and simultaneously buy \$10 million for October settlement at \$100.75. You receive cash now and will pay it back (plus take the bonds) next month.

*Step 2 — the drop.* The \$0.25 price difference (the "drop") is the cost of the roll, but in exchange you skipped a month of owning the bonds — you gave up roughly a month of the 6% coupon and any principal, which the buyer collects. Whether the roll beats holding depends on whether the drop is smaller or larger than the carry you gave up.

*Step 3 — when it pays.* If demand to borrow this exact coupon is high, the drop shrinks (or the roll trades "special"), and you end up financing your \$10 million position *below* the prevailing repo rate. *The dollar roll turns a standardized, liquid bond into a cheap financing tool — a perk that exists precisely because the TBA market makes one MBS interchangeable with another.*

## Why this all loops back to the whole market

We have been treating you as a lone investor reacting to rates. But MBS holders are not passive — they *hedge*, and their hedging is large enough to move the very rates they are reacting to. This is the feedback loop that makes negative convexity a macro phenomenon, not just a portfolio quirk.

Here is the mechanism. A bank, insurer, or agency holding a giant MBS book wants to keep its interest-rate exposure (its duration) roughly constant. But we just learned that an MBS's duration *changes with rates*: it shortens when rates fall (contraction) and lengthens when rates rise (extension). So to stay hedged, the holder has to trade *against* those moves. When rates fall and their MBS duration drops, they are suddenly under-exposed to rates, so they *buy* duration — they buy Treasuries (or receive in swaps) to top up. That buying pushes Treasury yields *down further*, which triggers *more* refinancing, which shortens MBS *more*, which forces *more* Treasury buying. A self-reinforcing rally. When rates rise and MBS duration extends, the same logic runs in reverse: holders are suddenly over-exposed, so they *sell* Treasuries to shed duration, pushing yields *up further* and amplifying the selloff.

This is **convexity hedging**, and it is one of the genuine ways the tail wags the dog in fixed income. The mortgage market is so large — agency MBS alone is a multi-trillion-dollar market, comparable in size to the Treasury market — that the hedging flows of its holders can measurably accelerate Treasury moves. It turns the MBS investor's private problem (my bond keeps changing length) into a market-wide accelerant (everyone's hedging in the same direction at the same time). When you read that "the bond market moved more than the news justified," convexity hedging by MBS holders is often part of why.

## Common misconceptions

**"Falling rates are always good for a bond."** True for a bullet Treasury, false for an MBS. Falling rates trigger the refinancing wave that caps the MBS's price near par and returns your principal early into a low-rate world. For a mortgage investor, a sharp rate rally is often the *worst* scenario, not the best — the precise opposite of the textbook reflex.

**"An MBS yields more than a Treasury, so it's a better deal."** The extra yield is not a gift; it is the premium you collect for being *short the homeowner's prepayment option* (and, for non-agency, for taking credit risk too). You are getting paid to absorb negative convexity. Whether that pay is adequate depends entirely on how violently rates move — you are quietly selling volatility, and you find out the price the next time rates lurch. The yield looks like free income right up until the option is exercised against you.

**"Negative convexity is just a quant abstraction."** It is the most concrete thing in the world: it is the reason your mortgage fund underperforms a Treasury fund in a rate rally, and the reason your "seven-year" bond turns into a three-year bond or a twelve-year bond depending on which way rates broke. Anyone who owns a mortgage fund owns negative convexity whether they have heard the term or not.

**"Agency MBS are risk-free because the government guarantees them."** The guarantee covers *credit* risk — you will get your principal and interest back even if homeowners default. It does *nothing* about rate risk or prepayment risk. Agency MBS lost serious value in 2022 when rates rose sharply (the same selloff that hit Treasuries), guarantee and all. Safe from default is not the same as safe from loss.

**"Prepayment just means you get your money back, so what's the harm?"** The harm is *when* and *into what*. You get your money back early specifically when reinvesting is worst (rates just fell), and you keep your money locked up specifically when you most want out (rates just rose). The timing is adversarial by construction — that is the whole definition of a short option position.

**"An MBS and a CMO tranche are basically the same risk."** A CMO *redistributes* the pool's prepayment risk, it does not reduce it. A PAC tranche can be nearly as stable as a normal bond, while the support tranche or an IO strip paired with it carries a *concentrated, magnified* version of the negative convexity. Two tranches from the same pool can have wildly different — even opposite — responses to a rate move.

## How it shows up in real markets

**The 2003 refinancing wave and the convexity-hedging spike.** In the summer of 2003, US mortgage rates fell to then-historic lows, and refinancing exploded. MBS portfolios shortened dramatically as homeowners refinanced en masse, and holders rushed to buy Treasuries and receive in swaps to replace the lost duration. The hedging flow was so large it whipsawed the Treasury and swap markets — yields fell sharply, then snapped back violently when rates reversed and the same players had to sell. It became the textbook case of MBS convexity hedging amplifying a rate move: the mortgage market's private hedging need spilled out and moved the world's most liquid bond market. Round numbers; the episode is well documented, and the mechanism is exactly the feedback loop above.

**The 2007–2008 subprime detonation.** The non-agency side of the MBS market was the fuse of the global financial crisis. Pools of subprime loans were securitized into private-label MBS and re-securitized into CDOs, sold as safe because models assumed home prices would not fall nationwide all at once. When they did, defaults clustered, the junior tranches were wiped out, and losses chewed up into tranches that had been rated AAA. The lesson was about *credit* risk in non-agency structures, not prepayment — but it is the reason the agency/non-agency distinction is the first question any MBS buyer asks, and why the agency guarantee is worth so much. (For the rating-agency angle, see [credit rating agencies](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch).)

**The Fed as the world's largest MBS buyer.** Since 2008, the Federal Reserve has bought agency MBS by the trillion as part of quantitative easing, holding well over \$2 trillion at its peak. By buying MBS, the Fed pushed mortgage rates down directly — bypassing the Treasury-to-mortgage spread — to support housing. When the Fed later began shrinking that portfolio ("QT"), it removed a giant, price-insensitive buyer from the market, and mortgage spreads widened. The biggest holder of negative convexity on earth is a central bank that does not hedge it the way a private investor would, which itself changes how the whole market behaves. (See [the central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).)

**The 2022 rate shock and "extension" everywhere.** When the Fed hiked aggressively through 2022 and long rates rose by roughly 2.5 percentage points, refinancing stopped cold — almost every outstanding mortgage was now below the market rate, so nobody refinanced. MBS *extended*: their average lives stretched out, their durations lengthened, and their prices fell hard, right alongside Treasuries. Banks holding large agency-MBS and long-Treasury books — most infamously [Silicon Valley Bank](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — sat on enormous unrealized losses on supposedly "safe" agency paper. The guarantee paid every dollar of principal and interest as promised; it did nothing to stop the price from falling. Extension risk, on the books of leveraged holders, helped turn a rate shock into a banking scare.

**Mortgage REITs and the leverage on negative convexity.** Mortgage REITs (agency mREITs) make their living owning agency MBS, financing them cheaply via repo and dollar rolls, and pocketing the spread — which means they are running a *leveraged* short-convexity position. In calm, range-bound markets the carry is generous and the dividends are fat. In a sharp rate move — either direction — the negative convexity bites, the value of the assets swings against the leverage, and several mREITs have had to slash dividends, sell assets into falling markets, or de-lever abruptly. They are the purest publicly traded expression of "you are paid to be short the homeowner's option, until the day you aren't."

## When this matters to you, and further reading

If you own a bond fund, you may already own negative convexity without knowing it: a mortgage fund, a "total bond market" index fund (which is heavily MBS), or a fund chasing yield through agency or non-agency paper. The thing to internalize is that those funds will *quietly underperform* a pure-Treasury fund in a sharp rate rally — not because the manager did anything wrong, but because the bonds are built to cap their own upside. In a selloff they will fall about as much as anything else. The extra yield they paid you in the calm years was the compensation; the capped rally is the bill.

The deeper payoff is conceptual. Once you see that an MBS is "a normal bond minus an option you sold to the homeowner," the whole instrument becomes legible: the higher yield is the option premium, the price ceiling is the option being exercised, extension and contraction are the option's two payoff regions, and convexity hedging is everyone trying to re-cover that option at once. It is the cleanest real-world example of how an embedded option reshapes a bond's entire personality.

To go deeper: the foundation for the price ceiling is [convexity: why duration is not the whole story](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story), which derives the curvature this whole post relies on; the reinvestment problem is [reinvestment risk and the two faces of yield](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield); the matching problem MBS makes hard is [immunization and duration matching](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge); and for the heavy machinery of pricing cash flows through a model, see the quant-side [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics). For where mortgage rates come from in the first place, start with [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).
