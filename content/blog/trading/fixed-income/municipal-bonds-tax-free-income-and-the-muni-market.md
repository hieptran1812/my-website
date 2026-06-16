---
title: "Municipal bonds: tax-free income and the muni market"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "What municipal bonds are, why their interest is exempt from federal tax, how to use tax-equivalent yield to compare a tax-free muni with a taxable bond, and why the muni market is a fragmented, retail-heavy corner where a 3.5% coupon can beat a 4.8% one."
tags: ["fixed-income", "bonds", "municipal-bonds", "tax-equivalent-yield", "general-obligation", "revenue-bonds", "muni-treasury-ratio", "tax-exemption", "credit-risk", "retail-market"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — a municipal bond is a loan to a US state or local government whose interest is exempt from federal income tax, which lets it pay a lower headline yield and still leave a high-bracket investor with more money than a taxable bond.
> - Munis come in two flavors: a **general obligation (GO)** bond backed by the issuer's full taxing power, and a **revenue bond** backed only by the cash flow of one project (a toll road, a water system, an airport). The GO claim is broader and usually yields a touch less.
> - The whole subject turns on one number, the **tax-equivalent yield (TEY)**: a tax-free yield divided by one minus your tax rate. A 3.5% muni, for someone in the 37% federal bracket, is worth 3.5% / (1 − 0.37) = 5.56% in taxable terms — and that beats a 4.8% taxable corporate.
> - The same muni is worth *more the higher your bracket*. At 12% it grosses up to just 3.98% and loses to the corporate; at 37% it grosses up to 5.56% and wins. Munis are a tax play, so they live in the portfolios of high-bracket individuals and the funds that serve them.
> - The market gauges muni value with the **muni/Treasury yield ratio**: when a tax-free muni out-yields a taxable Treasury (ratio above ~100%) munis are cheap; when the ratio falls toward 65–80%, munis are rich.
> - Munis are not default-free. **Detroit** (2013, the largest US city bankruptcy) and **Puerto Rico** (a US territory with \$70B+ of unpayable debt) both restructured and handed bondholders losses, and the market is a fragmented, retail-heavy place of ~50,000 issuers and ~1 million bonds where most issues rarely trade.

Here is a puzzle that breaks most people's intuition about bonds. A high-earning doctor is comparing two bonds. One is a perfectly solid corporate bond yielding **4.8%**. The other is a municipal bond — a loan to a city — yielding only **3.5%**. The corporate pays more. It is from a large, profitable company. By every number on the screen it looks like the better deal. And yet, for this particular doctor, the 3.5% bond is the one that leaves more money in the bank at the end of the year.

How can a bond that pays *less* be worth *more*? The answer is the one feature that makes municipal bonds their own corner of the market: the interest a muni pays is **exempt from federal income tax**, and often from state tax too. The doctor keeps every penny of the muni's 3.5%. On the corporate's 4.8%, the taxman takes a 37% bite, leaving only about 3.0% in hand. Suddenly the "lower-yielding" muni is the higher-paying one, and the whole comparison flips.

![A general obligation bond backed by a city's full taxing power beside a revenue bond backed only by one project's own cash flow](/imgs/blogs/municipal-bonds-tax-free-income-and-the-muni-market-1.png)

The diagram above is the mental model for the first half of this post: municipal bonds split into two big families. On the left is a **general obligation** bond — the city pledges its full taxing power to repay you, so the claim is broad and strong. On the right is a **revenue bond** — only one project's income (the tolls, the water bills, the airport fees) stands behind it, so the claim is narrower and the yield a bit higher. Both share the magic ingredient, the federal tax exemption, and that exemption is what the back half of the post is really about: how to price it with **tax-equivalent yield**, how to spot when munis are cheap with the **muni/Treasury ratio**, who actually buys these bonds, and why "tax-free" never meant "risk-free." This is the tax-advantaged corner of the bond market, and it rewards anyone willing to do a little arithmetic the rest of the market won't.

## Foundations: what a municipal bond actually is

Before we can price the tax break, we need to be precise about the instrument. None of this is hard, but munis carry a few terms that don't appear anywhere else in fixed income, so let's define each from scratch.

A **bond** is a tradable loan. You hand the issuer money today (you *buy* the bond), and in return the issuer promises a fixed stream of payments — the **coupons** — plus the return of the original amount, the **principal** or **par value**, at a set future date called **maturity**. A plain bond might be "a 10-year \$1,000 par note with a 3.5% coupon": you pay roughly \$1,000 today, you receive \$35 a year (3.5% of \$1,000) for ten years, and you get your \$1,000 back at the end. (For how those pieces fit together in general, see [anatomy of a bond: par, coupon, maturity, issuer](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer).)

A **municipal bond** — a "muni" — is simply a bond where the **issuer is a US state or local government, or an entity created by one.** That covers an enormous range: a state (California, Texas), a county, a city, a school district, a transit authority, a water utility, a public hospital, a port authority, an airport. When any of these needs to build something — a school, a bridge, a sewer system, a stadium — it often borrows the money by selling bonds to investors, and those bonds are munis. The thing that unites them all is not the issuer's size but its *type*: it is a sub-national public body in the United States.

What makes a muni special is the **tax exemption.** Under US federal tax law, the interest you earn on most municipal bonds is **exempt from federal income tax.** If you collect \$35 of muni coupon, you keep all \$35 — none of it appears on your federal tax return as taxable income. Contrast that with a corporate bond or a US Treasury, whose interest *is* taxable: collect \$48 of corporate coupon and you owe ordinary income tax on it, so a top-bracket investor keeps only about \$30. The exemption is the entire reason munis exist as a distinct asset class, and it is worth understanding *why* it exists, which we'll come to in a moment.

There is a second layer of exemption worth flagging now. Many states also exempt their *own* residents from **state** income tax on **in-state** munis. A California resident who buys a California muni typically pays no federal *and* no California tax on the interest — a "double exempt" bond. Buy an out-of-state muni and you usually still get the federal exemption but owe your home state's tax on it. This is why muni funds often come in state-specific flavors ("the California muni fund," "the New York muni fund"): for a resident of a high-tax state, the in-state exemption is a meaningful extra sweetener.

A few more terms you'll meet:

- A **basis point** (*bp* or *bps*) is one hundredth of a percentage point — 0.01%. A yield difference of "130 basis points" is 1.30%. Credit and rate people quote everything in bps, so 100 bps = 1.00%.
- A **marginal tax rate** (or **tax bracket**) is the rate you pay on your *next* dollar of income. The US federal system is progressive — 10%, 12%, 22%, 24%, 32%, 35%, 37% — and the bracket that matters for a muni decision is your *top* one, because muni interest, if it were taxable, would stack on top of all your other income and be taxed at that highest rate.
- **Yield** is the annual return a bond pays relative to its price. For our purposes, a bond bought at par yielding 3.5% pays \$35 a year on \$1,000. (For the family of yield measures, see [the many yields: current yield, YTM, and yield to call](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call).)

With those in hand, the whole subject reduces to one sentence: **a muni pays a lower yield than a comparable taxable bond, but you keep all of it, so for a high-bracket investor it can leave more money in hand.** Everything that follows is machinery for measuring exactly that.

### Why the tax exemption exists

The federal exemption isn't an accident or a loophole that slipped through — it's a deliberate, century-old policy with a clear logic, and understanding the logic explains both why munis are reliable and why they're occasionally politically vulnerable.

The original idea rests on the principle of **reciprocal immunity**: in the early US constitutional understanding, the federal government and the states were seen as separate sovereigns that shouldn't tax each other's core functions. Taxing the interest on a state's bonds would, in effect, be the federal government taxing the states' ability to borrow — to fund schools, roads, and water. So municipal interest was left untaxed when the federal income tax was created in 1913, and it has stayed that way ever since (with some carve-outs we'll meet later).

The practical effect — and the reason the exemption survives politically — is that it is a **subsidy for state and local infrastructure that flows through the bond market instead of through Washington.** Because investors will accept a lower yield on a tax-free bond, a city can borrow more cheaply than it otherwise could. That lower borrowing cost is, in economic terms, a federal subsidy: the US Treasury forgoes the tax it would have collected, and the saving shows up as a lower interest bill for the city. The bondholder and the city split the value of the exemption between them — the bondholder gets a better after-tax yield than a taxable bond of the same risk, and the city gets a lower interest cost than a taxable issuer of the same risk. That split is the heart of the muni market, and the tax-equivalent-yield math we're about to do is just a way of measuring exactly how the spoils are divided.

This also tells you who the exemption is *worth most to*: investors in the highest tax brackets, because they're the ones who would otherwise lose the most to tax. A muni is, by design, a high-bracket investor's instrument. We'll see that fall out of the arithmetic in a few sections.

## The two families: general obligation vs revenue bonds

Almost every muni belongs to one of two families, and the distinction is the single most important thing to know about a muni's *credit* — its chance of paying you back. The split is about **what stands behind the bond.**

A **general obligation bond (GO)** is backed by the **full faith and credit** of the issuing government — which, in plain English, means its **power to tax.** A city that issues a GO bond is pledging that it will, if necessary, raise property taxes to make the payments. The repayment doesn't depend on any single project succeeding; it draws on the whole tax base of the community. Because that claim is broad and backed by the coercive power to tax, GO bonds are generally considered the *safest* munis, and they typically pay a slightly *lower* yield. In our running numbers, a 10-year GO might yield around **3.3%**.

A **revenue bond** is backed only by the **revenue of a specific project or enterprise.** A toll-road authority issues bonds and pledges the tolls to repay them. A water utility pledges water-bill revenue. An airport pledges landing fees and concession income. Crucially, there is usually **no tax backstop** — if the project's revenue falls short, bondholders have no claim on the city's general tax base. The bond lives or dies on whether that one enterprise throws off enough cash. Because the claim is narrower and the risk concentrated in a single project, revenue bonds generally pay a slightly *higher* yield to compensate. A comparable revenue bond might yield around **3.7%**, a touch above the GO.

The figure at the top of the post captures the contrast: the GO leans on the city's full taxing power (green, the stronger claim), while the revenue bond leans on one project's own receipts (amber, the narrower claim) with no tax to fall back on (red).

That difference matters enormously when things go wrong, and it doesn't always break the way you'd expect. Conventional wisdom said GO bonds were nearly bulletproof because a city can *always* raise taxes. But Detroit's bankruptcy (which we'll examine in detail later) showed that in a deep enough crisis, even "full faith and credit" GO holders can be forced to take losses — while some revenue bonds, tied to a healthy, essential enterprise like a water system people will always pay for, came through better. The family a bond belongs to is the *starting point* for its credit, not the last word; you still have to ask whether the taxing power or the project revenue is actually strong.

#### Worked example: the same project, GO vs revenue

A city wants to build a \$100 million water-treatment plant and is deciding how to finance it.

**Option A — a GO bond.** The city issues \$100 million of general obligation bonds at a **3.3%** coupon. Annual interest: \$3.3 million. The pledge is the city's full taxing power; if water revenue ever fell short, property taxes would cover the bonds. Investors see a broad, strong claim and accept the low 3.3% yield.

**Option B — a revenue bond.** The city instead has its water authority issue \$100 million of **revenue** bonds at a **3.7%** coupon, repaid only from water bills. Annual interest: \$3.7 million — \$400,000 more per year than the GO. Why would the city pay more? Because a revenue bond doesn't count against the city's general debt limits, doesn't require voter approval in many states, and ring-fences the risk to the project. Investors demand the extra 0.4% (40 bps) because if the plant underperforms, there's no tax backstop.

The 40-bp gap *is the price of the missing tax pledge.* The city is, in effect, paying \$400,000 a year to keep this debt off its general books and tied to the project — and investors are collecting that \$400,000 as compensation for giving up the city's taxing power as a backstop.

*A GO bond rents the city's whole tax base; a revenue bond rents one project's cash flow — and the yield gap between them is the market's price for that difference in backing.*

### What actually makes a muni strong or weak

The GO-vs-revenue split is the starting point, but it doesn't tell you whether a *specific* bond is sound. Credit analysts dig into a handful of fundamentals, and they're worth knowing because they explain why two GO bonds — or two revenue bonds — can yield very differently.

For a **GO bond**, the question is the health of the **tax base.** A growing, diversified, wealthy community with a rising population can comfortably raise taxes if it must; a shrinking, single-industry town with a falling population cannot — there's no one left to tax. Analysts look at population trends, income per capita, the diversity of the local economy, and the size of the existing debt and pension burden relative to the tax base. Detroit was the cautionary tale: decades of population loss hollowed out the tax base until the "power to tax" was a power to tax almost no one. A GO pledge from a thriving suburb and a GO pledge from a dying mill town are not the same bond, even if both say "full faith and credit."

For a **revenue bond**, the question is the **essentiality and economics of the project.** A water or sewer system is about as safe as revenue bonds get, because people will pay their water bill before almost anything else and the service is a monopoly — there's no competitor to switch to. A toll road through a growing corridor is solid; a toll road built on optimistic traffic forecasts that never materialized is dangerous (several have defaulted when the cars simply didn't come). A sports stadium financed on the promise of a team's success is among the riskiest, because the revenue depends on attendance and the team staying put. The key questions are: is the service essential, is demand reliable, and does the revenue comfortably cover the debt payments — the **debt-service coverage ratio**, the project's net revenue divided by its annual debt payment? A coverage ratio of 2.0 (revenue is twice the debt bill) is comfortable; 1.1 is precarious.

#### Worked example: two revenue bonds, same yield, different safety

You're shown two revenue bonds, both yielding 3.7%. Bond A is a **water-system** revenue bond with net revenue of \$20 million against \$10 million of annual debt service — a debt-service coverage ratio of \$20M / \$10M = **2.0**. Bond B is a **convention-center** revenue bond with net revenue of \$11 million against \$10 million of debt service — coverage of just **1.1.**

Both pay the same 3.7% today, but their margins for error are wildly different. Bond A can lose *half* its revenue and still cover the bonds; the service is essential and the demand monopolistic. Bond B can lose barely 10% of its revenue — one bad year of bookings — before it can't make the payment, and convention-center demand is discretionary and cyclical. At the same yield, Bond A is the far better risk-adjusted buy. The market *should* charge Bond B a higher yield; if it isn't, the bond is mispriced and you're not being paid for the thinner coverage.

*Two revenue bonds at the same yield are not the same bet — the debt-service coverage ratio tells you how much revenue can vanish before you stop getting paid.*

## Tax-equivalent yield: the one number that matters

Now we get to the concept that the entire muni market revolves around. You cannot compare a tax-free muni to a taxable bond by looking at their yields side by side — that's comparing a number you keep entirely with a number you only keep part of. To compare them honestly, you have to put them in the same units: **after-tax dollars.** The tool that does this is the **tax-equivalent yield (TEY)**.

The TEY answers a precise question: *what would a taxable bond have to yield to leave me with the same after-tax income as this tax-free muni?* The formula is one of the most useful in personal finance, and it's almost embarrassingly simple:

$$
\text{Tax-Equivalent Yield} = \frac{\text{Muni Yield}}{1 - \text{Tax Rate}}
$$

Here the muni yield is the tax-free yield you're offered, and the tax rate is your **marginal** (top-bracket) rate expressed as a decimal. The denominator, $1 - \text{tax rate}$, is the fraction of a *taxable* bond's interest you actually get to keep after tax — your **after-tax keep rate.** Dividing the muni yield by that keep rate "grosses it up" into the bigger, pre-tax taxable number it's equivalent to.

![Tax-equivalent yield grosses a tax-free muni yield up by dividing by one minus the investor's tax rate to compare it with a taxable bond](/imgs/blogs/municipal-bonds-tax-free-income-and-the-muni-market-2.png)

The pipeline above walks the calculation end to end with our running numbers: a 3.5% muni, an investor in the 37% bracket (so the keep rate is 1 − 0.37 = 0.63), divide to get a tax-equivalent yield of 5.56%, then compare that to the 4.8% taxable corporate — and the muni wins by 0.76% a year.

Let's make sure the intuition is airtight before we lean on it. Why divide rather than multiply? Because we want the *taxable* yield that, after losing its tax, equals the muni. If a taxable bond yields $Y$ and you keep $(1 - t)$ of it, your after-tax yield is $Y \times (1 - t)$. We want that to equal the muni's tax-free yield $M$:

$$
Y \times (1 - t) = M \quad\Longrightarrow\quad Y = \frac{M}{1 - t}
$$

So $Y$ — the taxable yield that ties the muni — is exactly $M / (1 - t)$, the TEY. The division is just algebra: we're solving for the pre-tax number whose after-tax remainder matches the muni.

#### Worked example: the doctor's decision, fully worked

Let's resolve the puzzle from the opening with every step shown. Our doctor is in the **37% federal bracket** and is choosing between a **3.5% tax-free muni** and a **4.8% taxable corporate**, each \$10,000 invested.

**The muni.** It pays 3.5% tax-free. On \$10,000 that's **\$350 a year**, and the doctor keeps all \$350 — no federal tax.

**The corporate.** It pays 4.8% taxable. On \$10,000 that's **\$480 a year** before tax. But the doctor owes 37% of that to the IRS: \$480 × 0.37 = \$177.60 in tax. After tax, the corporate leaves:

$$
\$480 \times (1 - 0.37) = \$480 \times 0.63 = \$302.40
$$

So the head-to-head, in dollars the doctor actually keeps, is **\$350 (muni) vs \$302.40 (corporate).** The "lower-yielding" muni wins by **\$47.60 a year** on \$10,000.

Now confirm it with the TEY formula. The muni's tax-equivalent yield is:

$$
\text{TEY} = \frac{3.5\%}{1 - 0.37} = \frac{3.5\%}{0.63} = 5.56\%
$$

That means the 3.5% muni is equivalent, for this doctor, to a **5.56% taxable** bond. Since the actual taxable corporate yields only 4.8%, the muni is the clear winner — by 5.56% − 4.8% = **0.76% a year**, which on \$10,000 is \$76 of *yield* advantage… and indeed \$350 vs the \$302.40 keep matches once you reconcile the rounding (the muni's \$350 is 5.56% of the after-tax base the corporate delivers). Either way you slice it — dollars kept or tax-equivalent yield — the muni is ahead.

*For a top-bracket investor, a 3.5% tax-free coupon punches like a 5.56% taxable one, which is how a "lower-yielding" bond ends up paying you more.*

## The tax break is worth more the higher your bracket

The doctor's win wasn't a property of the bond — it was a property of the doctor's *bracket*. Because the TEY divides by $(1 - t)$, a bigger $t$ (a higher bracket) means a smaller denominator and therefore a *bigger* tax-equivalent yield. The exact same 3.5% muni is worth dramatically more to a high earner than to a low one. This is the deepest fact about munis: **they are a tax play, and the value of the play scales with your tax rate.**

![Tax-equivalent yield of a single 3.5 percent muni rising bracket by bracket from below to above the taxable corporate line](/imgs/blogs/municipal-bonds-tax-free-income-and-the-muni-market-4.png)

The bar chart above shows the same 3.5% muni grossed up across the federal brackets, with the 4.8% corporate drawn as a horizontal line. At the low brackets the bars sit *below* the line — the corporate wins. Only at the 32% and 37% brackets do the bars rise *above* the 4.8% corporate, where the muni wins. The bracket isn't a footnote to the muni decision; it *is* the decision.

Let's read the numbers straight off the formula for each bracket, all for the same 3.5% muni:

| Tax bracket | Keep rate (1 − t) | Tax-equivalent yield | vs 4.8% corporate |
|---|---|---|---|
| 12% | 0.88 | 3.5% / 0.88 = **3.98%** | corporate wins |
| 22% | 0.78 | 3.5% / 0.78 = **4.49%** | corporate wins |
| 24% | 0.76 | 3.5% / 0.76 = **4.61%** | corporate wins |
| 32% | 0.68 | 3.5% / 0.68 = **5.15%** | muni wins |
| 37% | 0.63 | 3.5% / 0.63 = **5.56%** | muni wins big |

![A table of the tax-equivalent yield of one tax-free muni across five tax brackets with the verdict against the taxable corporate in each row](/imgs/blogs/municipal-bonds-tax-free-income-and-the-muni-market-7.png)

The same numbers laid out as a verdict table make the crossover unmistakable: the muni's tax-equivalent yield climbs row by row, and the right column flips from "corporate wins" to "muni wins" exactly where your bracket crosses the break-even rate.

The pattern is stark. For someone in the 12% bracket, the muni is equivalent to a 3.98% taxable bond — well short of the 4.8% corporate, so they should buy the corporate and pay the tax. The crossover happens between the 24% and 32% brackets; above it, the muni dominates. This is exactly why muni funds market themselves to high earners and why a low-bracket retiree is often *wasting* the tax exemption by holding munis — they're accepting the low headline yield without getting enough tax benefit to justify it.

#### Worked example: the break-even bracket

At what exact tax rate does our 3.5% muni tie the 4.8% corporate? Set the TEY equal to the corporate yield and solve for $t$:

$$
\frac{3.5\%}{1 - t} = 4.8\% \quad\Longrightarrow\quad 1 - t = \frac{3.5\%}{4.8\%} = 0.729 \quad\Longrightarrow\quad t = 0.271 = 27.1\%
$$

So the **break-even tax rate is about 27.1%.** An investor whose marginal rate is above 27.1% should prefer the muni; below it, the corporate. The 24% bracket (TEY 4.61%) is just under break-even and loses; the 32% bracket (TEY 5.15%) is comfortably over and wins. You can run this calculation in reverse for any pair of bonds: the break-even rate is $1 - (\text{muni yield} / \text{taxable yield})$, and comparing it to your own bracket tells you instantly which bond to own.

*The muni-vs-taxable choice has a single switch — your break-even tax rate — and a muni only makes sense if your bracket is on the high side of it.*

#### Worked example: the in-state double exemption

Now layer on state tax. Suppose our doctor lives in California, top state rate ~13.3%, and is comparing an *in-state* California muni at 3.5% (exempt from both federal *and* California tax) against the same 4.8% taxable corporate (taxable at both levels). The doctor's combined marginal rate is roughly 37% federal + 13.3% state = ~50% (ignoring the small interaction between them for simplicity). The in-state muni's tax-equivalent yield is now:

$$
\text{TEY} = \frac{3.5\%}{1 - 0.50} = \frac{3.5\%}{0.50} = 7.00\%
$$

For a top-bracket Californian, the *same* 3.5% in-state muni is equivalent to a **7.00% taxable** bond — it now towers over the 4.8% corporate. The state exemption nearly doubled the apparent yield versus the federal-only case (5.56%). This is why high-tax-state residents are the most natural muni buyers of all, and why "the California fund" and "the New York fund" exist: stacking the state exemption on top of the federal one turns a modest tax-free coupon into a yield no taxable bond of similar risk can match.

*Stacking a state exemption on the federal one compounds the tax break — for a top-bracket resident of a high-tax state, a tax-free coupon can be worth nearly double its headline rate.*

## The muni/Treasury ratio: is the whole market cheap or rich?

So far we've compared one muni to one taxable bond for one investor. But traders need a single gauge for whether munis *as a class* are cheap or expensive right now. That gauge is the **muni/Treasury yield ratio** — often just "the M/T ratio" or "the muni ratio" — and it is the most-watched number in the muni market.

The ratio is exactly what it sounds like: the yield on a high-grade (AAA) municipal bond divided by the yield on a US Treasury of the *same maturity*, expressed as a percentage.

$$
\text{Muni/Treasury Ratio} = \frac{\text{AAA Muni Yield}}{\text{Treasury Yield}} \times 100\%
$$

Here both yields are for the same maturity (say 10 years), the muni is top-rated to strip out credit differences, and the result is a percentage. If the 10-year AAA muni yields 3.5% and the 10-year Treasury yields 4.0%, the ratio is 3.5 / 4.0 = **87.5%.**

Here's why the ratio is so revealing. In a tax-free-versus-taxable world, you'd *expect* a muni to yield *less* than a Treasury — the muni's interest is exempt, the Treasury's is taxable (at the federal level), so investors will accept a lower yield on the muni and still come out ahead after tax. A ratio comfortably **below 100%** is therefore the normal, healthy state: munis yield, say, 80–90% of Treasuries, and the gap is the value of the tax exemption being shared between issuer and investor.

The signal comes from the *level*:

- **Ratio above ~100% → munis are cheap.** When a tax-*free* muni yields *more* than a taxable Treasury, something is off. You're being paid a higher yield *and* you don't pay tax on it — a double win. This happens when munis are being dumped (in a panic, in a tax-selling rush, when a wave of new issuance floods the market) and is a classic "munis are cheap" signal.
- **Ratio low, toward 65–80% → munis are rich.** When the ratio is well below normal, munis are expensive: their tax-free yield has been bid down so far that even after accounting for the tax break, you're not getting paid much extra to own them. This happens when demand is hot (lots of high-bracket money chasing tax-free income) or supply is scarce.

![The muni to Treasury yield ratio over two decades, spiking above one hundred percent in crises when munis are cheap and falling below eighty percent when munis are rich](/imgs/blogs/municipal-bonds-tax-free-income-and-the-muni-market-3.png)

The centerpiece chart above plots the 10-year AAA muni/Treasury ratio over roughly two decades (the path is illustrative but the shape and the crisis spikes are real and well-documented). The dashed orange line at **100%** is the cheap/rich threshold; the dashed green line near **80%** marks the lower edge of the normal band. For most of the time the ratio oscillates between ~80% and ~95% — munis a bit cheaper than Treasuries on a pre-tax basis, as theory predicts. Then come the violent exceptions:

- In the **2008 financial crisis**, forced selling — leveraged muni funds unwinding, insurers dumping bonds, a freeze in the bond-insurance industry — spiked the ratio toward **190%.** Tax-free munis briefly yielded almost twice the taxable Treasury, an extraordinary "munis on sale" moment for anyone with cash.
- In **March 2020 (COVID)**, the same dynamic recurred even more sharply: as investors fled to cash, munis were sold indiscriminately and the ratio gapped to roughly **200%.** Then the Federal Reserve stepped in to backstop short-term muni funding, and within months the panic reversed.
- In **2021**, the after-shock was the opposite extreme: a flood of stimulus cash, near-zero interest rates, and high earners hunting for tax-free income drove the ratio down toward **65%** — munis historically rich, barely paying a premium for the tax exemption.

The ratio is, in effect, the muni market's fear-and-greed gauge. It tells you not whether *a* muni is a good buy for *you* (that's the TEY's job) but whether the *whole asset class* is cheap or dear relative to the risk-free taxable benchmark.

Why does the ratio move at all, rather than sitting at a stable level set by the tax exemption? Because the muni market's supply and demand are *seasonal and lumpy* in ways the Treasury market isn't. On the **demand** side, muni buying surges in the spring around tax season (when high earners, freshly reminded of their tax bills, hunt for shelter) and ebbs at other times. On the **supply** side, issuance comes in waves — cities tend to bring deals in concentrated windows, and a heavy month of new bonds can swamp demand and push yields up (ratio up, munis cheap), while a quiet month with redemptions and coupon payments flooding cash back to investors can starve them of bonds to buy (ratio down, munis rich). Layer on the fact that the marginal buyer is a *retail* investor or fund — more prone to panic selling and yield-chasing than a disciplined institution — and you get a market that overshoots in both directions. The ratio's swings are the visible result of a thin, retail, seasonal market repricing a fixed tax benefit against a constantly moving Treasury yield.

#### Worked example: reading the ratio as a real after-tax edge

Suppose the 10-year AAA muni yields 4.5% and the 10-year Treasury yields 4.0%, a ratio of 4.5 / 4.0 = **112.5%** — above 100%, so the signal says "munis cheap." Let's verify what that means in after-tax dollars for a 37%-bracket investor.

The muni pays 4.5% tax-free — kept in full. The Treasury pays 4.0% taxable; after 37% tax the investor keeps 4.0% × 0.63 = **2.52%.** So the after-tax comparison is **4.5% (muni) vs 2.52% (Treasury)** — the muni delivers nearly *double* the after-tax yield of the supposedly identical-risk Treasury. The ratio being above 100% wasn't a quirk; it was a genuine, large after-tax edge sitting in plain sight. That is exactly the kind of dislocation that draws crossover buyers — taxable-bond investors who don't even use the tax break — into munis, because a tax-free yield above the taxable Treasury is a free lunch even before the exemption.

*When the muni/Treasury ratio climbs above 100%, a tax-free bond is out-yielding a taxable one — a dislocation so favorable it pulls in buyers who don't even need the tax break.*

## Who buys munis, and why it shapes the market

The tax math tells you *who should* own munis — high-bracket investors — and that single fact shapes the entire structure of the market. Munis are overwhelmingly a **retail** asset: owned by individuals, directly or through funds, far more than by the institutions that dominate Treasuries and corporates.

The reason is the tax exemption itself. A pension fund, an endowment, or a foreign central bank pays *no US income tax* (or isn't subject to it), so the muni exemption is worthless to them — they'd rather hold a higher-yielding taxable bond. The exemption only has value to a taxable US investor in a high bracket. So the natural owners of munis are:

- **High-income individuals** holding bonds directly in taxable accounts — the classic muni buyer, a wealthy retiree or professional clipping tax-free coupons.
- **Municipal bond mutual funds and ETFs**, which pool many investors' money to buy a diversified basket of munis — the dominant way most people get muni exposure, since buying individual munis well is hard (more on that below).
- **Property & casualty insurers and some banks**, taxable institutions that hold munis for the after-tax yield.

Notice who's *missing*: the giant tax-exempt institutions (pensions, endowments, sovereign wealth funds) and foreign buyers who anchor the Treasury market. Their absence is why munis are a smaller, more fragmented, more retail-driven world.

![A grid contrasting the municipal market's fifty thousand issuers and retail ownership against the single issuer and deep liquidity of the Treasury market](/imgs/blogs/municipal-bonds-tax-free-income-and-the-muni-market-6.png)

The grid above lays the muni market against the Treasury market on the dimensions that matter for an investor. The contrasts are dramatic:

- **Issuers.** The Treasury market has exactly *one* issuer — the US government. The muni market has roughly **50,000** distinct issuers: every state, county, city, school district, water authority, and special district that has ever borrowed. There is no single "muni" the way there is a single 10-year Treasury; there are tens of thousands of unrelated credits.
- **Bonds.** Because each issuer often sells many separate maturities, there are something like **1 million** distinct muni bonds (CUSIPs) outstanding, versus a few hundred active Treasury issues. The muni universe is a vast long tail of tiny, idiosyncratic bonds.
- **Owners.** Roughly **two-thirds** of munis are held by households and the funds that serve them — a retail-dominated ownership base — versus the institutional and foreign giants that own Treasuries.
- **Liquidity.** This is the painful part. Most munis **trade rarely** — many sit untraded for months or years after issuance — and when they do trade, the **bid-ask spread** (the gap between the price you can sell at and buy at) is wide, often a percent or more for small retail lots. The Treasury market is the deepest, most liquid market on earth, with razor-thin spreads. A muni is the opposite: easy to buy at issuance, sometimes hard and costly to sell.

The market is roughly **\$4 trillion** in size — large in absolute terms but a fraction of the ~\$27 trillion Treasury market, and spread across vastly more issuers and bonds. That fragmentation and illiquidity is the muni market's defining structural feature, and it has a direct practical consequence: **most individuals are better off owning munis through a fund than picking individual bonds.** A fund diversifies across thousands of issuers (so one Detroit doesn't sink you), and it absorbs the liquidity cost of trading at institutional scale instead of paying the punishing retail bid-ask on a single \$10,000 lot.

#### Worked example: the hidden cost of buying one muni

You want \$25,000 of muni exposure. Compare two routes.

**Route 1 — buy one individual muni.** You buy a single \$25,000 revenue bond at a 3.7% yield. But as a retail buyer in a fragmented market, your dealer marks it up: you pay perhaps **1.0% above** the bond's fair value at purchase, and if you ever need to sell early, you might give up another **1.0–2.0%** to the bid-ask spread. That ~1% entry cost alone is equivalent to losing about *a third of a year's coupon* up front, and you hold the undiversified, single-issuer risk of that one project.

**Route 2 — buy a muni fund.** You put \$25,000 into a muni ETF yielding 3.5% with an expense ratio of **0.15% a year** (\$37.50 on \$25,000). You get instant diversification across thousands of bonds, tight on-exchange trading, and no punishing one-off markup. Over a multi-year hold, the fund's small annual fee is far cheaper than the retail bid-ask you'd eat buying and selling individual bonds — and you've shed the concentration risk entirely.

The headline yields look close (3.7% vs 3.5%), but after the retail trading costs and the concentration risk, the fund usually wins for an ordinary investor. The 0.2% lower yield is the price of diversification and liquidity, and in a market this fragmented it's a bargain.

*In a market of 50,000 issuers and million-bond illiquidity, the cost of buying munis badly often dwarfs the yield difference between bonds — which is why most muni money sensibly flows through funds.*

## Munis can default: the credit risk that "tax-free" hides

It is tempting to read "exempt from tax" as "safe." It isn't. A muni is still a **loan**, and the borrower can still fail to pay. **Credit risk** — the chance you don't get paid back in full — applies to munis just as it does to corporates (the full framework is in [credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back)). Muni defaults are *rarer* than corporate defaults, especially among GO bonds and essential-service revenue bonds, but they are not zero, and when a big one happens it reshapes how the whole market prices risk.

The reasons munis default less than corporates are real: a city can't go out of business the way a company can, essential services (water, sewer) generate reliable revenue, and GO bonds can in principle raise taxes. Historically, investment-grade muni default rates have been a fraction of comparably-rated corporate rates. But "rarer" is not "never," and two modern episodes — Detroit and Puerto Rico — taught the market hard lessons about what can go wrong.

![A timeline of major municipal credit events from the 2008 crisis through Detroit's 2013 bankruptcy and Puerto Rico's restructuring](/imgs/blogs/municipal-bonds-tax-free-income-and-the-muni-market-5.png)

The timeline above traces the modern muni credit story. The 2008 crisis strained budgets everywhere; **Detroit** filed for bankruptcy in 2013; **Puerto Rico**'s crisis built through 2015–16 and led to a bespoke federal restructuring. Let's take the two big ones in turn, because each broke a different piece of muni conventional wisdom.

**Detroit, 2013.** After decades of population loss and industrial decline, the city of Detroit filed for Chapter 9 municipal bankruptcy in July 2013 — at roughly **\$18–20 billion** of obligations, the largest US municipal bankruptcy in history. The shock for the market wasn't that a struggling city had trouble; it was *who took the losses.* Detroit's emergency manager initially proposed treating **unlimited-tax general obligation bonds** — the supposedly bulletproof "full faith and credit" pledge — as **unsecured** claims, putting them in line with other creditors rather than first. GO holders, who thought a city's taxing power made them nearly senior, faced real haircuts; in the final plan, some GO classes recovered only a fraction of par (figures around the 70s of cents on the dollar circulated for certain classes), while pensions and other creditors negotiated their own outcomes. The lesson seared into the market: **a GO pledge is only as good as a bankruptcy judge says it is**, and "full faith and credit" can be contested when a city is truly insolvent.

**Puerto Rico, 2015–2022.** Puerto Rico is a US *territory*, not a state, and it had borrowed enormously — well over **\$70 billion** of bond debt plus large pension liabilities — much of it bought by mainland muni-fund investors precisely *because* Puerto Rico bonds were "triple tax-exempt" (free of federal, state, *and* local tax in every US state). When the island's economy buckled, that debt became unpayable. But Puerto Rico, as a territory, had **no access to Chapter 9** bankruptcy. Congress had to pass a special law, **PROMESA**, in 2016, creating a federal oversight board and a bespoke restructuring process. The eventual restructuring (largely completed in 2022) **cut the debt sharply**, with many bonds recovering well below 100 cents on the dollar — a major loss for the retail investors and funds that had reached for that juicy triple-exempt yield. The lesson: **the tax exemption is not a credit guarantee**, and the most tax-advantaged bonds can carry the most credit risk precisely because the tax sweetener lured buyers into overlooking the danger.

#### Worked example: expected loss on a muni vs the extra yield

Apply the credit framework to a single muni. You're weighing a BBB-rated revenue bond yielding **4.2%** against a AAA GO yielding **3.3%** — a 90-bp pickup for the lower-rated bond. Is the extra 0.9% enough to cover the extra default risk?

Use **expected loss = probability of default × loss given default**. Suppose the BBB muni has a 1% annual probability of default and, because munis often recover more than corporates (essential assets, ongoing revenue), a loss-given-default of 50% (i.e. you'd recover ~50 cents). Its expected annual credit loss is:

$$
1\% \times 50\% = 0.5\% \text{ per year}
$$

The AAA GO's expected loss is negligible, call it ~0.0%. So you're being paid **0.9%** of extra yield to bear about **0.5%** of expected annual credit loss — leaving roughly **0.4%** as a risk premium for the *uncertainty* (defaults cluster in recessions, and the loss is lumpy). Whether that 0.4% is enough depends on your appetite for the tail: most years the BBB muni simply pays you 0.9% more, but in the rare bad year you take a real loss. The 90-bp spread isn't free money — it's the price of credit risk, the same arithmetic as any corporate, just with muni-flavored default and recovery numbers.

*A higher muni yield is compensation for credit risk, not a gift — and "tax-free" does nothing to change the PD-times-LGD math behind a default.*

## A few wrinkles that trip people up

Two features of munis deserve a flag because they catch even experienced investors.

**The Alternative Minimum Tax (AMT).** Not all muni interest is fully exempt. A subset of munis are **private activity bonds** — bonds where the proceeds substantially benefit a private entity (a stadium financed for a team, an airport facility leased to airlines). The interest on many of these is **subject to the AMT**, a parallel tax system that claws back some exemptions for certain taxpayers. For an investor caught by the AMT, an "AMT bond" is less tax-free than it looks, and these bonds usually yield a bit more to compensate. Always check whether a muni is "AMT" or "non-AMT" — it changes the TEY math for affected investors.

**The de minimis rule and buying munis below par.** If you buy a muni in the secondary market at a discount to par, part of your gain at maturity can be taxed as **ordinary income** rather than tax-free interest, under the "market discount" and **de minimis** rules. This is a trap for investors who think *every* dollar from a muni is tax-free; the *coupon* interest is exempt, but a price gain on a discounted bond may not be. It's a reminder that the exemption covers *interest*, not necessarily *capital gains*.

#### Worked example: an AMT bond's real tax-equivalent yield

You're a 37%-bracket investor *also subject to the AMT* (a 28% AMT rate applies to your situation), comparing two munis: a regular non-AMT muni at 3.4% and a higher-yielding **private-activity AMT bond** at 3.8%. The AMT bond looks better by 40 bps — but its interest is hit by your 28% AMT.

The non-AMT muni is fully tax-free: effective yield **3.4%.** The AMT bond's interest is taxed at 28%, so you keep only 3.8% × (1 − 0.28) = **2.74%** of it after the AMT. Suddenly the "higher-yielding" AMT bond delivers *less* after-tax (2.74%) than the plain muni (3.4%). For an AMT-exposed investor, the extra 40 bps of headline yield was an illusion — the AMT ate it and more.

*"Tax-free" has fine print: a private-activity AMT bond can quietly become partly taxable, and for the wrong investor its higher yield turns into a lower one.*

## Common misconceptions

**"Tax-free means risk-free."** No — the exemption is about *taxes*, not *default*. A muni is a loan to a city or authority that can run into trouble, and munis carry the same interest-rate risk as any bond (their price falls when rates rise; see [duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income)) *plus* credit risk. Detroit's GO holders and Puerto Rico's bondholders took real losses despite the tax exemption. "Tax-free" describes the coupon's treatment, nothing more.

**"Munis always yield less than taxable bonds, so they're a worse deal."** They yield less *on the screen* precisely because you keep all of it. The right comparison is the tax-equivalent yield, and for a high-bracket investor a 3.5% muni can beat a 4.8% taxable bond hands down. Judging munis by their headline yield is exactly the mistake the muni exemption is designed to exploit.

**"Munis are great for everyone who wants safe income."** They're great for *high-bracket* investors. A retiree in the 12% bracket grossing a 3.5% muni up to only 3.98% is leaving money on the table versus a 4.8% taxable bond — and is worse off still holding munis inside a tax-deferred account like an IRA, where the exemption is wasted entirely (you don't pay tax on IRA interest anyway, so the muni's lower yield is pure loss). Munis belong in *taxable* accounts of *high-bracket* investors, and almost nowhere else.

**"A general obligation bond can't default because cities can always raise taxes."** Detroit demonstrated otherwise. In a genuine insolvency, a bankruptcy court can subordinate even "full faith and credit" GO claims, and the political will (or legal ability) to raise taxes on a shrinking population has limits. GO bonds are *safer* than revenue bonds on average, but "can't default" is wrong — the pledge is strong, not absolute.

**"All muni interest is completely tax-free."** Most is federally exempt, but private-activity bonds may be subject to the AMT, out-of-state bonds are usually taxable by your home state, capital gains on munis are taxable, and discounted munis can trigger ordinary-income tax under the de minimis rule. The exemption covers ordinary *interest* on most munis — not every dollar in every situation.

**"Buying individual munis is the cheap way to get tax-free income."** For most retail investors it's the *expensive* way. The market's fragmentation and illiquidity mean you pay a wide dealer markup to buy and a wide spread to sell a single \$10,000 lot, often more than a fund's annual fee over a realistic holding period — and you take concentrated single-issuer risk. Funds, despite their fees, usually deliver cheaper, more diversified muni exposure for ordinary investors.

## How it shows up in real markets

**The 2008 muni-insurance collapse and the ratio spike.** Before 2008, a large share of munis were "wrapped" by **bond insurers** (MBIA, Ambac, FGIC) that guaranteed payment for a fee, letting weak issuers borrow at AAA rates. When those insurers' *other* business — guaranteeing subprime-mortgage securities — blew up, their own credit collapsed, and the AAA wrap on thousands of munis became worthless overnight. Forced selling by leveraged muni funds and the loss of the insurance backstop drove the muni/Treasury ratio toward ~190% — tax-free bonds yielding nearly double taxable Treasuries. Investors who recognized that the *bonds themselves* were mostly fine (it was the *insurers* that failed, not the cities) bought munis at a once-in-a-generation discount. The episode permanently shrank the bond-insurance industry and taught the market to look through the wrapper to the underlying issuer.

**Detroit's bankruptcy, 2013–2014.** Detroit's Chapter 9 was the muni market's stress test for the GO pledge. The city's proposal to treat unlimited-tax GO bonds as unsecured claims — putting bondholders who thought they had a near-senior claim in line with pensioners and vendors — sent a chill through the market and forced a wholesale reassessment of how "safe" GO debt really is in a true insolvency. The eventual plan of adjustment imposed losses on some GO classes while protecting essential pieces, and it established that a federal bankruptcy court, not the bond's covenant language, has the final word on recoveries. Spreads on weaker-city GOs widened for years afterward as investors repriced the risk that "full faith and credit" might not mean first in line.

**Puerto Rico and the triple-exempt trap, 2015–2022.** Puerto Rico bonds were beloved by mainland muni-fund managers because their interest was exempt from federal, state, *and* local tax in *every* state — a "triple-exempt" prize that pushed yields down and let the island borrow far more than its economy could support. When the debt became unpayable, the very feature that made the bonds attractive (their tax appeal had masked deteriorating credit) turned into a multi-year restructuring under the special PROMESA law, since a territory had no Chapter 9 access. Many bonds were ultimately cut to recoveries well below par. The episode is the textbook case of **reaching for tax-advantaged yield and overlooking credit** — and a reminder that the most tax-favored bonds can carry hidden danger precisely because the tax break dulls investors' scrutiny.

**The March 2020 COVID dislocation and the Fed.** As the pandemic froze markets, investors fled to cash and dumped munis indiscriminately, gapping the muni/Treasury ratio to roughly 200% in a matter of days — tax-free bonds yielding twice taxable Treasuries even for AAA credits. The Federal Reserve, for the first time, stood up a **Municipal Liquidity Facility** to backstop short-term muni funding, signaling it would not let the muni market seize up. The panic reversed within months, and by 2021 the ratio had swung all the way to ~65% (munis historically *rich*) as stimulus cash and yield-hungry high earners flooded back in. The whole episode — from 200% to 65% in roughly a year — is the muni/Treasury ratio doing its job as a fear-and-greed gauge, and a case study in how central-bank intervention can short-circuit a forced-selling spiral.

**The 2017 tax law and the demand for tax-free income.** When the 2017 federal tax overhaul capped the **state-and-local-tax (SALT) deduction** at \$10,000, residents of high-tax states (California, New York, New Jersey) suddenly faced higher effective tax burdens — which *increased* the value of tax-free muni interest to exactly those investors. Demand for in-state munis from high-bracket residents of high-tax states rose, helping push muni/Treasury ratios lower (munis richer) in the years that followed. It's a clean illustration of the core principle: the value of a muni's exemption rises and falls with the tax rates of its natural buyers, so changes in tax law move muni prices directly.

## When this matters to you, and where to go next

Municipal bonds touch your life the moment your income climbs into the higher tax brackets. If you're a high earner with money in a *taxable* brokerage account looking for steady income, the muni-versus-taxable decision is one of the few places in personal finance where a little arithmetic — the tax-equivalent yield — reliably finds free money the rest of the market overlooks. And every time you read that "munis are cheap" or "rich," you now know it's the muni/Treasury ratio talking: above 100% means a tax-free bond is out-yielding a taxable one, a genuine dislocation.

The single idea to carry forward is the **tax-equivalent yield**: a tax-free yield divided by your after-tax keep rate, $\text{TEY} = \text{muni yield} / (1 - t)$. Compute it before you ever compare a muni to a taxable bond, remember that it rises with your bracket, and never let "tax-free" lull you into forgetting that a muni is still a loan with credit and rate risk attached.

From here, the natural next steps deepen each piece. For the credit machinery behind a muni default — probability of default, loss given default, recovery — see [credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) and [credit spreads: pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default). For who else lends to governments and companies, and why, see [who buys bonds: the global demand for safe income](/blog/trading/fixed-income/who-buys-bonds-the-global-demand-for-safe-income) and [who issues bonds and why: governments, companies, and cities](/blog/trading/fixed-income/who-issues-bonds-and-why-governments-companies-and-cities). For how the ratings that drive muni credit are assigned, see [credit rating agencies: Moody's, S&P, Fitch](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch). And to keep the macro context in view — why all bond yields beat to the rhythm of rates and supply — return to [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [deficits, debt, and bond supply](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields).

*This is educational material about how municipal bonds and their taxation work, not advice to buy or sell any security or a substitute for personalized tax guidance.*
