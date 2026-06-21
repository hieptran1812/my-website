---
title: "Lehman Brothers, 2008: Leverage, Repo 105, and the Run on an Investment Bank"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a 158-year-old investment bank with 30 times leverage, funded overnight in the repo market and hiding billions with an accounting trick, ran out of cash in a single weekend and froze the world's financial system."
tags: ["banking", "lehman-brothers", "repo", "leverage", "bank-run", "wholesale-funding", "repo-105", "2008-crisis", "investment-bank", "systemic-risk", "money-market-funds"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Lehman Brothers did not die because it lost money slowly; it died because the professionals who lent it cash overnight stopped lending, and a bank with 30.7 times leverage and no sticky deposits cannot survive even a few days of that.
>
> - Lehman financed \$639 billion of assets on roughly \$21 billion of equity — about **30.7 times leverage**, three times a normal commercial bank. A fall in asset value of just **3.26%** is enough to wipe out all the equity at that ratio.
> - It funded itself not with insured deposits but with **short-term wholesale money** — mostly overnight repo. When lenders doubted its subprime and commercial-real-estate collateral, they raised haircuts and refused to roll the loans. That is a *run on repo*, and it is faster and harder to stop than a deposit run.
> - The accounting trick called **Repo 105** let Lehman shuffle about \$50 billion of assets off its balance sheet at each quarter-end, making its reported leverage look safer than it was.
> - When Lehman filed for bankruptcy on **September 15, 2008** — the largest filing in US history — its debt sat inside a money-market fund that then "broke the buck," and the short-term credit the whole economy runs on froze. One bank's funding failure became everyone's.

It was the weekend of September 13-14, 2008. On the thirty-second floor of the Federal Reserve Bank of New York, the heads of every major Wall Street firm sat in shirtsleeves around a table, summoned by the Treasury Secretary and the New York Fed president to do one thing: keep Lehman Brothers alive until Monday. Lehman was a 158-year-old institution that had survived the Civil War, the Great Depression, two World Wars, and the dot-com bust. It had \$639 billion of assets on its books. And it was about to run out of cash.

The arithmetic was brutal and simple. Lehman did not have depositors whose money sat patiently in checking accounts. It borrowed most of the cash it needed *every single day* from other financial institutions, pledging its bonds and mortgages as collateral. By that Friday those lenders had started to say no. They looked at the subprime mortgages and the commercial real estate piled on Lehman's balance sheet, decided they might not be worth what Lehman claimed, and quietly declined to roll over the loans. A bank that has to refinance hundreds of billions of dollars by tomorrow morning, and cannot, is finished — no matter how the income statement looks.

The diagram above the next section is the mental model for this whole post: a timeline that runs from the year Lehman's leverage peaked to the Monday it filed for bankruptcy. Notice that almost everything that *matters* happens in the last week. That is the signature of a funding run. Earnings erode over months; trust evaporates over days; cash evaporates over hours. By Monday, September 15, Lehman had filed for Chapter 11 — the largest bankruptcy in American history — and within twenty-four hours the failure had reached into an ordinary money-market fund and frozen the plumbing the entire economy uses to borrow short-term. This is the story of how one over-leveraged, wholesale-funded bank took the system down with it.

![Timeline of Lehman Brothers from peak leverage in 2007 to the September 15 2008 bankruptcy filing](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-1.png)

## Foundations: investment banks, wholesale funding, and the run on repo

Before we touch a single Lehman number, we need five ideas defined from zero. Skip nothing here; everything that follows is built from these blocks.

### An investment bank is not a deposit bank

When you think "bank," you probably picture the place that holds your checking account, gives you a debit card, and pays you a little interest. That is a **commercial bank** (or *retail bank*) — its defining feature is that it takes **deposits** from the public. Deposits are an unusual and wonderful kind of funding: they are cheap (banks pay you almost nothing on a current account), and they are *sticky* — most people leave most of their money in the bank most of the time, and in many countries the government **insures** deposits up to a limit (in the United States, \$250,000 per depositor per bank, through the FDIC). That insurance is the whole point: a saver who knows the government stands behind their \$250,000 has no reason to rush to the door in a panic.

An **investment bank** is a different animal. It does not (traditionally) take retail deposits. Instead it does things like trade securities, make markets, underwrite bond and stock issues, advise on mergers, and — this is the part that killed Lehman — hold a large inventory of securities on its own balance sheet, financed with borrowed money. Lehman, along with Bear Stearns, Merrill Lynch, Goldman Sachs, and Morgan Stanley, was one of the five big independent US investment banks in 2008. None of them had the deposit base of a JPMorgan or a Bank of America. They funded themselves a completely different way. (For the broader picture of what an investment bank does and how it earns money, see the companion piece on the [business of an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).)

### Wholesale funding: borrowing from professionals, not from the public

If you don't have deposits, where do you get the hundreds of billions of dollars you need to hold your bond inventory? You borrow it in the **wholesale funding markets** — the markets where banks and big institutions lend cash to each other. This is *professional* money: short-term loans from other banks, from money-market funds, from corporate treasurers parking spare cash, all in large blocks.

Wholesale funding has one enormous, fatal difference from deposits: it is **not sticky and not insured**. A money-market fund lending Lehman \$5 billion overnight is run by professionals whose entire job is to avoid losing the principal. At the first whiff of trouble they will pull their money — there is no insurance to make them patient, and there is no loyalty to a 158-year-old name. Worse, wholesale lenders watch each other. If one big lender pulls back, the rest see it and pull back too, because no professional wants to be the last one still funding a name everyone else has abandoned. Deposits trickle out; wholesale funding can vanish in a day.

### Repo: the overnight loan against collateral

The single most important wholesale funding tool — the one at the center of Lehman's death — is the **repurchase agreement**, universally called **repo**. A repo sounds complicated but is actually just a *secured loan*. Here is the whole thing in one sentence: I sell you a bond today for cash, and I promise to buy it back tomorrow for slightly more.

That "slightly more" is the interest. The bond I "sold" is really **collateral** — if I fail to buy it back, you keep the bond. So from the lender's point of view, a repo is the safest loan there is: even if the borrower vanishes, you are holding their bond. From the borrower's point of view, repo is the cheapest way to finance a portfolio of securities you already own. Lehman owned a vast inventory of bonds; every night it repo'd a big chunk of them out for cash, and every morning it repaid the cash and got the bonds back, then did it all again. (For the full mechanics — how repo settles, who the lenders are, why it sits at the heart of modern finance — see [the repo market and how banks fund overnight](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight), and for the system view, [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).)

The catch is the **haircut**. A lender won't give you \$100 of cash for \$100 of bonds — what if the bond falls in value before you buy it back? Instead, they lend you, say, \$98 against \$100 of bonds. That \$2 gap is the haircut: the lender's safety margin. In calm times, haircuts on high-quality collateral are tiny — 1% or 2%. But here is the key: **the haircut is a dial the lender controls, and they turn it up the instant they get nervous.** A jump in haircuts from 2% to 25% is, in effect, a lender saying "I no longer trust this collateral, so I'll lend much less against it." That dial is the throttle on a repo run.

### Leverage: the amplifier under everything

**Leverage** is the ratio of a bank's total assets to its own equity (its own money, the shareholders' stake). If a bank has \$100 of assets funded by \$10 of equity and \$90 of borrowing, its leverage is 10 times (\$100 ÷ \$10). Leverage is a multiplier in both directions. If those \$100 of assets rise 1%, the bank gains \$1 on \$10 of equity — a 10% return on equity. Magic. But if the assets *fall* 1%, the bank loses \$1, and its equity drops from \$10 to \$9 — a 10% loss. Leverage doesn't change your luck; it changes how hard your luck hits you. (We build the full leverage-and-capital picture in [bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion); here we just need the multiplier.)

The crucial threshold: at leverage of *L* times, an asset fall of **1/L** wipes out the entire equity. At 10 times, a 10% fall is fatal. At 30 times, only a 3.3% fall is fatal. Hold that number — we'll do the exact arithmetic for Lehman in a moment.

### Repo 105 and window dressing: making the picture lie

Finally, a term we will return to: **window dressing**. Banks report their balance sheet on specific dates — the end of each quarter. If a bank's leverage looks scary, it has an incentive to make the *reported* number on those specific dates look better than its true average. Doing something temporary right before the reporting date purely to flatter the published figures is called **window dressing**. **Repo 105** was Lehman's particular, and infamous, way of doing exactly this — using an accounting quirk in how a repo is classified to make about \$50 billion of assets disappear from the balance sheet on reporting day, then reappear days later. We'll dissect the trick in its own section.

With those five ideas in hand — investment bank vs deposit bank, wholesale funding, repo and haircuts, leverage, and window dressing — we can now read Lehman's collapse not as a mystery but as a mechanism.

## The leverage: 30.7 times, and why that number is the whole story

Let's start with the single most important number. At the end of 2007, Lehman Brothers' leverage was **30.7 times**. Its total assets were about **\$639 billion**, and the equity underneath that whole tower was only about **\$21 billion**. Put differently, for every dollar of its own money, Lehman controlled almost thirty-one dollars of assets.

Compare that to a plain commercial bank. A typical large deposit-funded bank runs at roughly 10 to 12 times leverage — it funds itself mostly with cheap, sticky, insured deposits and keeps an equity cushion of around 8% of assets. Lehman ran at three times that. The chart below puts the two side by side.

![Bar chart comparing Lehman 30.7 times leverage to a typical commercial bank at 10 times](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-2.png)

Why would a bank choose to run so hot? Because in a rising market, leverage is a money machine. Suppose Lehman could earn a 1% spread on its asset portfolio — borrow at 4%, hold bonds yielding 5%. On \$639 billion of assets, that 1% spread is \$6.4 billion of gross profit. Sitting on \$21 billion of equity, \$6.4 billion is a 30% return on equity. Investors *loved* that. Through the mid-2000s, every quarter the leverage went up and the profits went up and the bonuses went up, and the people pointing out that the same multiplier works in reverse were ignored as killjoys. This is the eternal temptation at the heart of the banking spine: a bank is a leveraged machine, and the more leverage you bolt on, the better it looks — right up until the moment the assets stop rising.

Now let's do the reverse math, because the reverse math is the entire point.

#### Worked example: how a 3.26% asset fall wipes out Lehman

Lehman has \$639 billion of assets and \$21 billion of equity (we'll use the rounder figure \$20.8 billion, which is \$639 billion ÷ 30.7, to keep it consistent with the leverage ratio). The other \$618 billion is borrowed.

Now suppose the assets fall in value. The losses come straight out of equity first — that is what equity *is*, the first-loss layer. How big a fall does it take to erase all \$20.8 billion of equity?

Loss to wipe equity = \$20.8 billion. As a fraction of \$639 billion of assets, that is:

\$20.8bn ÷ \$639bn = 0.0326 = **3.26%**.

So a fall of just 3.26% in the value of Lehman's assets is enough to leave the firm with *zero* equity — insolvent. To check it the other way: at 30.7 times leverage, the wipe-out fall is 1 ÷ 30.7 = 0.0326 = 3.26%. The two roads give the same answer.

Let's make it concrete with one more step. If Lehman's assets fall 3.3% — barely more than three cents on the dollar — the loss is \$639bn × 0.033 = \$21.1 billion, which is *more* than the \$20.8 billion of equity. The firm is not just dented; it is underwater. Its assets are worth less than its debts.

The one-sentence intuition: **at 30.7 times leverage, the entire margin between solvency and insolvency is a 3% move in your assets — and in the housing crash of 2008, mortgage-related assets fell far more than 3%.** Leverage didn't cause the losses; it made a survivable loss into a fatal one.

The chart below shows this as a cliff. As the asset fall climbs from 0% toward 3.26%, the equity remaining drops in a straight line — and the moment it crosses 3.26%, there is nothing left.

![Bar chart showing equity remaining falling to zero as Lehman assets drop 3.26 percent at 30.7 times leverage](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-3.png)

This is why leverage is described as *fragility*, not just risk. A bank at 10 times leverage has a 10% buffer before insolvency — room to be wrong, room to wait out a bad quarter. A bank at 30 times has a 3% buffer. It is a building with no margin in its foundations: the smallest tremor and it goes.

## The funding model: an investment bank with no deposits to lean on

Leverage tells you how *thin* Lehman's cushion was. The funding model tells you how *fast* it could be pushed over. These are two separate vulnerabilities, and Lehman had both.

Recall the foundations: a commercial bank funds itself mostly with deposits — sticky, insured, patient money. Lehman had almost none of that. Its \$618 billion of borrowing was overwhelmingly **wholesale**: short-term loans from other financial institutions, and the single biggest slice of that was **overnight repo**. Lehman was, in a real sense, financing a thirty-year asset (a mortgage bond) with a one-day loan. Each morning it had to convince the market to lend it the money all over again.

Think about what that means structurally. A deposit-funded bank that loses the confidence of its customers faces a *queue* — savers show up, withdraw, and the bank pays them out of its cash and liquid assets. It's bad, but it has friction; insured savers mostly stay put, and there are physical and psychological limits to how fast a deposit base drains. A wholesale-funded bank that loses confidence faces a *cliff*. There is no insured, sticky money to slow the drain. The lenders are professionals who can decline to roll the loan with a single phone call, and who all watch each other. The bank's funding doesn't drain; it switches off.

There is a subtler trap hidden inside "overnight" funding that's worth naming. When you finance a long asset with a one-day loan, you have not borrowed for one day — you have borrowed for *thirty years, one day at a time*, and you are betting that you can re-borrow tomorrow, and the day after, and every day until the asset matures. In calm markets that bet is invisible and free; rolling the loan is automatic. But the bet is real, and it comes due every single morning. This is called **rollover risk**, and it is the precise mechanism by which a maturity mismatch turns lethal: the longer-dated your assets and the shorter-dated your funding, the more often you have to win the same coin flip — "will they lend to me again today?" — and you only have to lose it *once*. Lehman had to win that coin flip on hundreds of billions of dollars every business day. The first morning it lost, the firm was over.

This is also why a bank can be killed by a problem that has nothing to do with its own books. In a wholesale panic, lenders don't carefully distinguish the strong borrowers from the weak; they retreat from a whole *category*. When subprime fear gripped the repo market, lenders pulled back from anyone holding mortgage collateral, sound or not, because in a panic the cheapest defense is to stop lending against the whole asset class. A bank that did nothing wrong can be caught in a run aimed at its neighbors — which is exactly what made the wholesale-funding model a *systemic* fragility, not just a one-firm risk.

#### Worked example: \$639bn of assets on a \$21bn sliver

Let's see the funding stack as a picture. Lehman's \$639 billion of assets were funded by:

- **Borrowed money: about \$618 billion** — most of it short-term wholesale funding, a large chunk of it overnight repo that had to be refinanced *every day*.
- **Equity: about \$21 billion** — the shareholders' own money, the only true cushion.

So the cushion was \$21 billion out of \$639 billion, or 3.3% of the balance sheet. The chart below draws it to scale: a towering orange column of debt with a thin green band of equity on top. That thin green band is everything standing between Lehman and insolvency — and \$618 billion of the orange column had to be re-borrowed, much of it tomorrow morning.

![Stacked bar showing Lehman 639 billion assets as 618 billion debt and a thin 21 billion equity band](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-6.png)

The one-sentence intuition: **a deposit bank that loses trust faces a slow leak it can plug; a wholesale-funded bank that loses trust faces a switch that flips off overnight — and Lehman's whole funding base was that switch.** The 3.3% equity and the overnight funding were not two problems. They were the same problem, viewed from two sides: a tiny cushion under an enormous pile of money that had to be re-borrowed daily from people who could leave at will.

There is a name for the role wholesale funding played in 2008: the **shadow banking system**. Lehman did exactly what a deposit bank does — borrow short and lend long, transforming one-day money into thirty-year assets — but it did it *outside* the deposit-insurance-and-lender-of-last-resort safety net that protects ordinary banks. It got all the fragility of maturity transformation with none of the backstops. (That whole system is the subject of [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).)

## The assets: subprime mortgages and commercial real estate

A thin cushion and flighty funding are dangerous in the abstract. They become lethal when the assets you're holding start to fall. Lehman's assets started to fall hard, and the reason is the central villain of the 2008 crisis: real estate.

Through the boom, Lehman had pushed aggressively into the mortgage business and into commercial real estate. It originated mortgages, packaged them into securities, and — crucially — held a great deal of this inventory on its *own* balance sheet rather than selling it all on. It also made big, illiquid bets on commercial property, including a roughly \$22 billion buyout of the apartment company Archstone near the absolute top of the market in 2007. These were precisely the assets that the housing crash hit hardest.

Here is why the *type* of asset matters so much for a repo-funded bank. Repo lenders accept your bonds as collateral, but how much they'll lend against a given bond depends on how confident they are in its value. Pristine collateral — US Treasury bonds — gets a tiny haircut, because everyone agrees on what a Treasury is worth. Subprime mortgage securities and commercial real estate are the opposite: their value is *uncertain and falling*, and worse, *nobody is quite sure what they're worth* because there are suddenly no buyers. The moment repo lenders look at Lehman's collateral and think "I'm not sure these mortgages are worth what Lehman's books say," two things happen at once. They raise the haircut (lend less against the same bonds), and Lehman has to come up with the difference in cash. And cash was the one thing Lehman didn't have a stable source of.

This is the deadly feedback loop, and it's worth stating precisely because it is the engine of every modern bank failure:

1. The assets fall in value (or are merely *feared* to have fallen).
2. The thin equity cushion shrinks toward zero.
3. Funders — repo lenders — get nervous about the collateral.
4. They raise haircuts and pull funding, forcing the bank to sell assets to raise cash.
5. Forced selling into a falling market pushes the assets down *further*, confirming the original fear.
6. Go to step 1, faster.

Notice that this loop can run even if the bank is technically still solvent at the start. *Doubt* about the assets is enough to trigger the funding pull, and the funding pull forces the fire sale that makes the doubt come true. This is the difference between **insolvency** (your assets are genuinely worth less than your debts) and a **liquidity crisis** (you can't raise the cash you need *right now*, even if you might be solvent on a calm day). For a bank at 30 times leverage funded overnight, the two collapse into each other: a liquidity crisis forces a fire sale, and a fire sale at 30 times leverage creates the insolvency. Liquidity *is* solvency when your cushion is 3%.

#### Worked example: how a real-estate bet eats the cushion

Take Lehman's commercial-real-estate exposure, of which the Archstone apartment deal was the headline. Lehman committed roughly \$22 billion to that buyout near the 2007 top. Suppose, conservatively, that commercial property values then fell 15% — a mild estimate for what actually happened to US commercial real estate after 2007. The loss on that single exposure is:

\$22bn × 0.15 = \$3.3 billion.

Now set that against the cushion. Lehman's *entire* equity was about \$20.8 billion. So this *one* category of bet, on a 15% decline, eats \$3.3 billion — about 16% of the firm's total equity — before we count a single dollar of subprime-mortgage losses. Stack the mortgage book's losses on top, and you can see how a firm with a 3.3% cushion runs out of cushion fast when *several* asset categories all fall at once. The assets didn't need to fall uniformly by 3.26%; a few concentrated bets falling 15-30% did the same arithmetic to the equity.

The one-sentence intuition: **at 30 times leverage you don't need a market-wide crash to go insolvent — a couple of large, concentrated, illiquid bets falling 15-20% will eat the whole cushion by themselves.** Concentration and illiquidity are leverage's accomplices.

## The run on repo: how the funding actually disappeared

We've assembled the ingredients. Now watch them ignite. The thing that actually killed Lehman in the second week of September 2008 was a **run on repo** — a wholesale-funding run. It is the exact analogue of a 1930s bank run with savers lined up around the block, except the "savers" are professional lenders, the "withdrawal" is refusing to roll an overnight loan, and the whole thing happens in days, not weeks.

The diagram below traces the mechanism. Read it left to right: it is the deadly feedback loop from the last section, drawn as the specific chain that played out on Lehman's funding desk.

![Pipeline of the run on repo from doubt to rising haircuts to lenders pulling to fire sale to insolvency](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-4.png)

Let's walk each step with the real story behind it.

**Doubt the assets.** By the summer of 2008, after Bear Stearns had already failed in March (rescued by JPMorgan in a Fed-backed shotgun marriage), the market's eyes turned to the next-weakest of the big investment banks. Lehman reported a \$2.8 billion loss for its second quarter — its first loss as a public company — and announced it was raising capital. That announcement, meant to reassure, did the opposite: it confirmed that the losses on the mortgage and real-estate book were real and large. The doubt was now public.

**Haircuts jump.** As confidence fell, Lehman's repo counterparties demanded more collateral for the same loans — they raised haircuts. Where a bond might once have funded at a 2% haircut, lenders now wanted far more, especially against the mortgage and real-estate collateral. Every increase in haircut was a direct cash drain: Lehman had to post more collateral or repay part of the loan, both of which consumed the cash it was desperately trying to conserve.

**Lenders pull.** Then came the worst step. Repo lenders began declining to roll over the loans at all. The clearing banks that sat in the middle of Lehman's repo arrangements demanded ever more collateral to keep processing its trades. Hedge-fund clients who used Lehman as their prime broker — and whose cash balances Lehman had been using as a funding source — yanked their money out, fearing it would be trapped if Lehman failed. Each departure was visible to the others, and each one made the next more likely.

#### Worked example: the mechanics of a rising haircut

Suppose, before the panic, Lehman funds \$10 billion of mortgage bonds at a 2% haircut. It pledges \$10 billion of bonds and receives \$9.8 billion of cash (the 2% haircut means the lender keeps a \$0.2 billion margin). Lehman uses that \$9.8 billion to fund its operations.

Now the panic hits and the lender raises the haircut on the *same bonds* to 25%. Lehman re-pledges \$10 billion of bonds and now receives only \$7.5 billion of cash. Overnight, the same collateral that produced \$9.8 billion now produces \$7.5 billion — a **\$2.3 billion cash hole** that Lehman must fill *today*, on this one block of collateral alone. Multiply that across hundreds of billions of repo'd assets, and across multiple lenders all tightening at once, and you have a cash demand of tens of billions of dollars appearing in a matter of days.

The one-sentence intuition: **a repo run doesn't need the lenders to withdraw a single existing loan — they only have to lend less against the same collateral, and a 30-times-levered bank with no deposit buffer is instantly short tens of billions in cash it cannot raise.** The haircut is the dial, and turning it from 2% to 25% is a run even if no one technically "withdraws."

**Fire sale and the death spiral.** To meet the cash drain, Lehman had to sell assets — but into a market where everyone knew it was a forced seller, prices were terrible, and selling its better assets only left a worse pile behind. Each sale confirmed that the marks on its books were too high, which deepened the doubt, which raised haircuts further. By the second week of September, Lehman's stock had collapsed (it fell roughly 45% on a single day, Tuesday September 9, after talks to raise capital from a Korean bank fell through), and its access to funding was evaporating. **Out of cash.** A bank that cannot fund itself by tomorrow morning is dead, regardless of what its balance sheet claims today.

## Repo 105: the accounting trick that hid the leverage

Here is where the story turns from tragedy to something darker. For years before the collapse, Lehman knew its leverage was a problem — analysts and investors watched the number, and a falling leverage ratio reassured them. So at the end of each quarter, when the books were photographed for the public, Lehman used an accounting maneuver called **Repo 105** to make about \$50 billion of assets temporarily vanish.

To understand the trick, we have to understand a quiet rule about how repos are recorded. A normal repo, as we saw, is economically a *loan*: you "sell" a bond and promise to buy it back, so it's really borrowing against collateral. Accountants treat a normal repo correctly — as a financing. The assets stay on your balance sheet, and you show a borrowing against them. Nothing disappears.

But there was an accounting standard (FAS 140) that said: if the value of the collateral you pledge is *high enough* relative to the cash you receive — specifically, if you pledge at least 105% of the cash value (hence "Repo 105"; some larger versions were "Repo 108") — then, under certain interpretations, the transaction could be treated as a **true sale** rather than a loan. And a true sale means the assets come *off* the balance sheet entirely.

That distinction is the whole trick. Lehman would, in the last days of a quarter, do enormous Repo 105 transactions: pledge, say, \$50 billion of bonds, receive cash, and — because the deal was structured to qualify as a "sale" — remove that \$50 billion of assets from its books. It used the cash to pay down other borrowings, shrinking the balance sheet on both sides. The published quarter-end balance sheet looked \$50 billion smaller, and the leverage ratio looked meaningfully lower. Then, a few days into the next quarter, Lehman would buy the bonds back — and the \$50 billion would quietly reappear on the books, where it sat for the rest of the quarter until the next reporting date came around.

![Before and after balance sheet showing Repo 105 moving 50 billion of assets off the books at quarter-end](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-5.png)

#### Worked example: Repo 105 lowering reported leverage

Let's see exactly how much the dial moved. Suppose on the day before quarter-end Lehman's *real* balance sheet is:

- Assets: \$690 billion
- Equity: \$22 billion
- Reported leverage: \$690bn ÷ \$22bn = **31.4 times**

That 31.4 looks alarming, and the market is watching. So Lehman does \$50 billion of Repo 105 transactions right before the reporting date. It pledges \$50 billion of bonds, receives \$50 billion of cash (treated as a sale, so the bonds leave the books), and uses that cash to repay \$50 billion of its borrowings. Both sides of the balance sheet shrink by \$50 billion. Now the *reported* picture is:

- Assets: \$640 billion (the \$50 billion is "gone")
- Equity: \$22 billion (unchanged — equity doesn't move)
- Reported leverage: \$640bn ÷ \$22bn = **29.1 times**

The published leverage dropped from 31.4 to 29.1 — a couple of full turns of leverage — without Lehman changing a single thing about its actual risk. The assets weren't sold to anyone permanently; they came back days later. The risk never left the building. Only the photograph lied.

The one-sentence intuition: **Repo 105 didn't reduce Lehman's risk by one dollar — it reduced the *reported* risk, which is exactly what window dressing is: making the number on the reporting date lie about the number on every other date.** By some accounts Lehman moved around \$50 billion off-balance-sheet at quarter-ends in 2008. The bankruptcy examiner, Anton Valukas, later concluded in a famous 2,200-page report that there were grounds for legal claims over the practice, and that Lehman's own auditor and senior management knew. The number you were shown was not the number that was true.

This matters beyond Lehman. It is a permanent lesson for anyone reading a bank: **a balance sheet is a snapshot on one chosen day, and a bank with an incentive to flatter that snapshot can do so.** The reported leverage of a wholesale-funded bank at quarter-end may be the *best* it looks all quarter. When you read a bank's filings, the question is never just "what does the leverage say?" but "what was the leverage on the other 89 days?"

## The failed rescue: the weekend nobody could save

By Friday, September 12, 2008, Lehman was hours from running out of cash, and the only paths left were a buyer or a government rescue. The weekend at the New York Fed was about finding one of them. Neither appeared.

There were two serious suitors. **Bank of America** looked at Lehman's books, saw the size of the hole in the mortgage and real-estate assets, and decided it would rather buy Merrill Lynch instead — which it did that very weekend, a deal that itself signaled how scared everyone was. **Barclays**, the British bank, was genuinely interested and negotiated hard. But a Barclays acquisition needed two things: a guarantee of Lehman's trading obligations to get from Friday to the closing of the deal, and the approval of British regulators. The British regulator (the FSA) declined to waive the requirement for a shareholder vote on such short notice, effectively requiring Barclays to guarantee a book it hadn't fully measured, with no time. Barclays walked.

And the US government — the Treasury and the Fed — declined to put public money in to plug the gap, as they had for Bear Stearns in March and would for AIG just two days later. The reasons are still debated. Officials argued they lacked the legal authority to lend to an insolvent firm with insufficient collateral; critics argue they were swayed by political anger at "bailouts" and underestimated how violently the system would react. Whatever the mix of cause, the result was decisive: there was no buyer and no backstop.

So at 1:45 a.m. on Monday, September 15, 2008, Lehman Brothers Holdings filed for **Chapter 11 bankruptcy protection** — listing \$639 billion in assets, the largest bankruptcy filing in US history, then or since. (For contrast: Washington Mutual, which failed days later and is the largest *bank* failure, had \$307 billion in assets. Lehman was twice that.) A 158-year-old firm with tens of thousands of employees was gone, undone not by a single catastrophic loss but by the simple fact that on Monday morning, no one would lend it the cash it needed to open.

The deepest lesson of the failed rescue is about the nature of the asset itself. The thing that could have saved Lehman was *confidence* — a buyer willing to vouch for it, or a government willing to backstop it. Confidence is the only asset a bank truly cannot manufacture for itself. A bank can hold more capital, sell assets, cut risk — but it cannot, on its own, decide to be trusted. And once the wholesale market decided Lehman was not to be trusted, no amount of internal effort could refinance \$618 billion by Monday. The bank ran on the one fuel it could not control.

There is a genuine policy dilemma buried here, and it's worth being fair to both sides. Every time the government rescues a failing bank, it teaches every *other* bank's managers and lenders that risk is one-directional: keep the gains in good times, hand the losses to the taxpayer in bad ones. That expectation — the **moral hazard** of guaranteed rescue — encourages exactly the reckless leverage that caused the problem. Letting Lehman fail was, in part, an attempt to break that expectation, to remind the market that some firms are allowed to die. The trouble is that the choice was framed as binary — rescue or disorderly bankruptcy — when the real failure was the absence of a *third* option: a way to wind a giant, interconnected bank down in an orderly fashion, imposing losses on its owners and creditors without freezing the system around it. That missing third option is precisely what post-crisis "resolution" regimes and living wills were built to create. Lehman's collapse is the most expensive lesson in modern finance about what happens when the only tools available are "save everything" or "let it explode."

## The systemic freeze: how one bankruptcy stopped the world

If Lehman's failure had stayed Lehman's problem, it would be a cautionary tale about leverage and nothing more. What made September 2008 the defining financial event of a generation was what happened *next* — the way a single bank's funding failure propagated into a freeze of the entire short-term credit system. The chain is worth tracing precisely, because it is the textbook example of **systemic risk**: the risk that one institution's failure cascades into everyone's.

![Pipeline showing Lehman failure breaking a money market fund then a credit freeze across the system](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-8.png)

**Step one: a money-market fund "breaks the buck."** To finance itself, Lehman had issued short-term IOUs called **commercial paper** — unsecured corporate borrowing that matures in days or weeks. A great deal of that paper was held by **money-market funds**: investment funds that hold ultra-safe, short-term assets and promise savers they can always get their money back at a stable \$1.00 per share. Money-market funds are the closest thing in the shadow system to a bank deposit — millions of people and corporations park cash in them precisely because they're supposed to never lose value. One of the oldest and largest, the Reserve Primary Fund, held \$785 million of Lehman commercial paper. When Lehman defaulted, that paper became near-worthless, and the fund's value dropped below \$1.00 per share — to about \$0.97. In the language of the industry, it had **broken the buck**, something that had happened essentially once before in history.

**Step two: savers flee all money-market funds.** The instant one supposedly safe fund broke the buck, every saver in every money-market fund asked the same terrified question: *is mine next?* They didn't wait to find out. A run on money-market funds began — hundreds of billions of dollars (estimates around \$300 billion in the days that followed) pulled out as fast as the funds could be redeemed. The "deposit-like" safe asset of the shadow system was suddenly not safe, and the panic spread to all of them at once.

**Step three: the commercial-paper market dies, and credit freezes.** Money-market funds were the biggest buyers of commercial paper — the short-term IOUs that ordinary, healthy, non-financial companies use to fund payroll, inventory, and day-to-day operations. As the funds faced redemptions, they stopped buying commercial paper from *anyone*, even from blue-chip industrial firms with nothing to do with mortgages. The commercial-paper market — the short-term funding plumbing of the real economy — seized up. Suddenly, completely solvent companies could not roll over the routine borrowing they relied on. Banks stopped lending to each other because no one knew who was next to fail; the rate banks charge each other overnight spiked. The credit that the entire economy runs on quietly stopped flowing.

This is the moment "systemic risk" stopped being a textbook phrase and became a felt reality. The Federal Reserve and Treasury, having declined to save Lehman, now had to rescue *everything else* to stop the cascade — guaranteeing money-market funds, standing up emergency facilities to buy commercial paper, and pushing through the \$700 billion TARP program within weeks. The cost of *not* backstopping one \$639 billion bank was a near-collapse of the whole financial system and the deepest recession since the 1930s.

#### Worked example: why \$785 million broke a system

The Reserve Primary Fund held \$785 million of Lehman commercial paper inside a fund of around \$62 billion. That is a loss of roughly 1.2% of the fund — by itself, not a catastrophe. So how did a 1.2% loss in one fund freeze global credit?

Because the *promise* the fund made was that you could never lose a cent. A money-market fund is supposed to return exactly \$1.00 per share, always. Breaking that promise — even by 3 cents, to \$0.97 — destroyed the one thing the entire \$3+ trillion money-market industry was built on: the belief that the dollar in there was as safe as a dollar in a bank. Once that belief cracked in *one* fund, it cracked everywhere, because the belief was the product. Savers don't run on a fund because they calculated a 1.2% loss; they run because the thing they were promised could never happen, just happened.

The one-sentence intuition: **a confidence-funded system can survive real losses but not the destruction of the belief that makes it work — and \$785 million was enough to destroy that belief for a \$3 trillion industry.** This is the banking spine writ large: every part of the financial system that borrows short and promises safety lives or dies on confidence, and confidence is binary. It is there until the instant it is not.

## Common misconceptions

**"Lehman failed because it lost too much money."** Not exactly. Lehman certainly had large losses on its mortgage and real-estate assets, but plenty of firms had losses and survived. What killed Lehman *on that weekend* was that it could not refinance its short-term funding — a liquidity crisis, not (only) an insolvency crisis. The losses created the doubt; the doubt triggered the funding run; the funding run forced the fire sale that finished the job. A better-capitalized, deposit-funded bank with the *same* asset losses could have ridden it out. The fatal combination was thin equity *plus* flighty funding, not losses alone.

**"It was just like a normal bank run — depositors panicking."** No. Lehman had essentially no retail depositors to run on it. Its run was a *wholesale* run: professional lenders in the repo and commercial-paper markets declining to roll over short-term loans. That distinction is the heart of this post. A deposit run is partly slowed by insurance and stickiness; a wholesale run has neither and moves at the speed of a phone call. The same mechanism, with very different speed and very different defenses — which is exactly why comparing Lehman to a deposit-funded failure like SVB is so illuminating.

**"Repo 105 was the reason Lehman collapsed."** No — Repo 105 was an accounting deception that *hid* the leverage, but it didn't create it. Lehman would have been just as fragile at 30.7 times leverage whether or not it window-dressed the quarter-end number. What Repo 105 did was deny outside observers a clear view of the danger, delaying the market's reckoning and deepening the eventual shock. The trick was a symptom of a culture that prioritized the optics of the balance sheet over its reality — but the real disease was the leverage and the funding model underneath.

**"The government should obviously have saved Lehman."** This is genuinely debated, not obvious. Officials argued they lacked legal authority to lend against insufficient collateral to an insolvent firm. And there's a real argument that *some* failure was necessary to break the assumption that the government would catch every fall — the "moral hazard" of guaranteed rescue. But the counterargument is just as strong: the systemic damage from letting Lehman go was so severe that the government had to rescue almost everything else within days, at far greater cost. The honest conclusion is that letting a deeply interconnected, \$639 billion bank fail without a plan for the fallout was a catastrophic miscalculation of how the system would react.

**"At least the equity holders were the ones who took the loss — that's how it should work."** They did lose nearly everything, which is correct in principle. But the failure also vaporized the savings of money-market fund holders who thought they were perfectly safe, froze credit for solvent companies that had nothing to do with Lehman, and threw millions out of work in the recession that followed. The point of resolving a failing bank in an *orderly* way — rather than a disorderly Chapter 11 — is precisely to impose losses on the people who took the risk *without* spreading the damage to everyone who didn't. Lehman's collapse did the opposite.

## How it shows up in real banks

The Lehman mechanism — thin equity plus runnable funding plus falling assets — is not a museum piece. It recurs, and the differences between cases are as instructive as the similarities.

### The post-Lehman freeze: the cost of letting it fail

The clearest "real banks" lesson is what the Lehman failure did to *other* banks. In the days after September 15, interbank lending — banks lending to each other overnight, the most basic plumbing of the system — nearly stopped. The rate at which banks were willing to lend each other dollars (the LIBOR-OIS spread, a standard gauge of bank funding stress) blew out to extraordinary levels, because every bank suddenly suspected that every other bank might be the next Lehman, hiding its own pile of bad mortgage assets behind its own window-dressed balance sheet. The lesson regulators took: a single large, interconnected institution cannot be allowed to fail *disorderly*, because its counterparties freeze in fear of who's next. This insight drove the entire post-crisis reform agenda — higher capital requirements, mandatory liquidity buffers, and "living wills" that map out how a giant bank can be wound down without freezing the system.

### Money-market funds breaking the buck: the run on "safe"

The Reserve Primary Fund breaking the buck is the cleanest illustration in financial history of how a confidence-funded asset dies. After 2008, regulators rewrote the rules for money-market funds — requiring some of them to float their share price (admitting that "always \$1.00" was a fiction), imposing liquidity fees and redemption gates, and toughening what they can hold. The deeper lesson stuck: anything that *promises* perfect safety while *holding* imperfect assets is a run waiting to happen, because the promise, not the asset, is what investors are buying. The same logic now drives the debate over stablecoins, which promise a stable \$1.00 while holding portfolios that can wobble. (We'll meet that argument again when this series reaches modern banking and deposit disintermediation.)

### SVB 2023: the deposit run that rhymed with Lehman

Fifteen years later, Silicon Valley Bank failed in a way that *rhymed* with Lehman but differed in a crucial dimension. SVB had \$209 billion in assets. Its problem was a duration mismatch — it had bought long-dated bonds that fell in value as rates rose, sitting on about \$17 billion of unrealized losses. When word spread, **\$42 billion of deposits were withdrawn in a single day** (March 9, 2023), with another \$100 billion queued for the next morning, and the bank failed within roughly 36 hours. The matrix below lays the two failures side by side.

![Matrix comparing a deposit-funded run at SVB to a wholesale-funded run at Lehman](/imgs/blogs/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank-7.png)

The deep contrast: SVB's run came from *depositors* — but a very specific, un-sticky kind. Fully 94% of SVB's deposits were *uninsured* (above the \$250,000 FDIC limit), held by tech startups who all banked together, talked to each other constantly, and could move money with a phone app. So SVB's "deposits" behaved less like patient retail savings and more like Lehman's wholesale funding: concentrated, sophisticated, and able to flee instantly. Lehman ran on repo lenders; SVB ran on uninsured tech depositors; but in both cases the funding was *runnable* — not the sticky, insured, granular deposit base that lets an ordinary bank sleep at night. The lesson is that the label "deposit" doesn't make funding safe; *stickiness* makes funding safe, and an investment bank's repo or a startup bank's uninsured millions are both fragile in the same way. (For the full SVB and Credit Suisse story at the system level, see [SVB and Credit Suisse, 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

### Bear Stearns 2008: the dress rehearsal

Six months before Lehman, Bear Stearns — a smaller investment bank with the same model (high leverage, repo funding, mortgage assets) — suffered the identical run. Its repo lenders pulled, its cash drained, and over a single weekend in March 2008 it was sold to JPMorgan for \$2 per share (later raised to \$10) in a deal backed by \$30 billion of Fed financing. Bear was the dress rehearsal, and its rescue arguably *worsened* Lehman's fate: the market assumed the government would do the same for Lehman, so it didn't demand Lehman fix itself fast enough, and when the rescue didn't come, the shock was greater. Bear and Lehman together are the proof that the investment-bank-on-repo model was systemically fragile — not one firm's mistake, but a structural flaw shared across the whole independent-broker-dealer business. Within a week of Lehman's fall, the two surviving big independents, Goldman Sachs and Morgan Stanley, converted into bank holding companies to gain access to the Fed's safety net. The standalone, wholesale-funded investment bank, as a category, did not survive 2008.

### Northern Rock 2007: the same disease, a year early, across the ocean

A full year before Lehman, the British mortgage lender Northern Rock failed in the same way for the same reason. Northern Rock funded its mortgage book not with deposits but with short-term wholesale borrowing — and when the wholesale markets froze in August 2007, it couldn't refinance. It became the first British bank run since 1866, with savers queuing outside branches. Northern Rock is proof that Lehman's mechanism wasn't unique to American investment banks or to subprime: it was the generic vulnerability of *any* institution that funds long assets with runnable short money, on either side of the Atlantic. (Northern Rock gets its own deep dive later in this series.)

## The takeaway / How to use this

The durable lesson of Lehman is not "leverage is bad" or "regulate banks more," though both have their place. It is something more precise and more useful: **a bank is a confidence-funded machine, and the question that decides whether it lives or dies is not how good its assets are, but how fast its funders can leave.**

Read any bank — any financial institution at all — through that single lens, and you'll see the same structure Lehman had:

- **How thin is the equity cushion?** Lehman's was 3.3% of assets, meaning a 3.26% asset fall was fatal. The thinner the cushion, the smaller the loss that ends the firm. When you see leverage above 20 times, you are looking at a firm that needs the world to stay calm to survive.
- **How runnable is the funding?** This is the question Lehman teaches that the balance sheet alone won't tell you. Deposits — granular, insured, sticky — are slow money. Wholesale funding — repo, commercial paper, uninsured concentrated deposits, overnight anything — is fast money. A bank can be perfectly solvent and still die if its funding is fast enough and its losses are doubtful enough. *Always look at the liabilities side as hard as the assets side.*
- **Is the reported picture the true picture?** Repo 105 is the permanent warning that a quarter-end snapshot can be the most flattering view a bank shows all year. When a number looks suspiciously well-managed right at the reporting date, ask what it looked like on the other days.
- **What happens to everyone else if it fails?** Lehman's deepest cost was not its own — it was the money-market funds that broke, the credit that froze, the recession that followed. The more interconnected a firm's short-term funding is with the rest of the system, the more its private fragility becomes a public danger.

If you remember one thing, remember the weekend. A 158-year-old firm with \$639 billion of assets, profitable for most of its history, run by people who were not fools, was destroyed in *days* — not because it suddenly lost \$639 billion, but because the people who lent it cash overnight decided, almost all at once, to stop. Everything else in this post — the 30.7 times leverage, the subprime assets, the rising haircuts, the Repo 105 trick — is just the explanation of *why* they stopped and *how fast* the bank fell once they did. The maturity-transformation trade that makes a bank useful is the same trade that makes it fragile: borrow short, lend long, and pray the short lenders stay. Lehman is the story of what happens the instant they don't.

This is educational material about how banks work and fail, not investment advice. The point is to read a bank's fragility, not to trade on it.

## Further reading & cross-links

- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the full mechanics of leverage, the loss-absorbing role of equity, and why a small asset fall can erase a bank. Lehman is the extreme case of this math.
- [The repo market and how banks fund overnight](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight) — repo mechanics, haircuts, and rehypothecation in depth; the funding tool that Lehman lived and died on.
- [The anatomy of a bank run: from whisper to collapse](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse) — the general run mechanism, the self-fulfilling panic, and the difference between a slow leak and a cliff.
- [Silicon Valley Bank 2023: the duration trap and the 36-hour digital run](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run) — the modern deposit run that rhymed with Lehman, and the crucial difference in who the funders were.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the system-level view of maturity transformation done outside the deposit safety net, which is exactly what Lehman was.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — what an investment bank actually does, and why its business model lacks the sticky deposit base that protects a commercial bank.
