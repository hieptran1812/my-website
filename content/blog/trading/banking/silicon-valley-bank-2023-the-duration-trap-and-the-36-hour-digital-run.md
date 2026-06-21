---
title: "Silicon Valley Bank 2023: The Duration Trap and the 36-Hour Digital Run"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a fast-growing, profitable bank parked cheap deposits in long bonds at the bottom of the rate cycle, let an unrealized loss quietly swallow its equity, and then died in thirty-six hours when 94%-uninsured depositors ran as one crowd."
tags: ["banking", "silicon-valley-bank", "bank-run", "duration-risk", "interest-rate-risk", "deposit-insurance", "asset-liability-management", "bank-failure", "liquidity", "held-to-maturity"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Silicon Valley Bank did not fail because it made bad loans. It failed because it took a flood of cheap, flighty deposits and parked them in long-dated bonds at the very bottom of the rate cycle, so when rates jumped the bonds lost value faster than the bank's thin equity could absorb — and a concentrated, 94%-uninsured depositor base ran as one crowd in thirty-six hours.
>
> - SVB had \$209bn in assets and \$175bn in deposits; it had put about \$91bn into held-to-maturity bonds whose unrealized loss had grown to roughly \$17bn — about the size of its entire \$16bn equity cushion.
> - The trap was a **duration mismatch**: deposits that can leave tomorrow funding bonds that mature in years. When the Fed lifted rates from near zero to over 5%, the market value of those bonds collapsed.
> - Because 94% of deposits were above the \$250,000 insurance limit and the depositors all knew each other, the run was instant: about \$42bn left on Thursday and another \$100bn was queued for Friday before regulators seized the bank.
> - The one number to remember: **94% uninsured**. Insured depositors have no reason to run; uninsured ones who can lose real money — and who all share the same group chat — run first and ask questions later.

## The weekend the tech world held its breath

On the afternoon of Thursday, March 9, 2023, the founders and finance chiefs of hundreds of startups were staring at the same message in the same Slack channels and the same group chats: *get your money out of Silicon Valley Bank, now.* By the time the West Coast went to bed, depositors had tried to pull about \$42 billion out of the bank in a single day. By Friday morning, the people charged with stopping the bleeding had to confront a darker number: roughly \$100 billion more was lined up to leave the moment the doors opened. There was not going to be a Friday. At a little after 8 a.m. Pacific time, California's financial regulator closed Silicon Valley Bank and handed it to the Federal Deposit Insurance Corporation. It was the second-largest bank failure in American history and, by the speed of its collapse, almost certainly the fastest.

What makes SVB worth dwelling on is not that a bank failed — banks fail in every cycle. It is that this one looked, on paper, like a strong bank right up until it wasn't. It was profitable. Its loan book was clean; it had almost no defaults. It had ridden the 2020–2021 technology boom to become the 16th-largest bank in the United States, the financial home of roughly half the country's venture-backed startups. And it died in a day and a half. The figure below is the whole story in miniature: a Wednesday capital raise, a Thursday run, a Friday seizure, and a government backstop on Sunday night.

![Silicon Valley Bank collapse timeline March 8 to 12 2023](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-1.png)

The diagram above is the mental model for the rest of this piece. SVB did not break because it lost money on risky bets. It broke because of two quieter, more boring decisions that compounded: it invested short-term money in long-term bonds, and it let its funding sit in the hands of a small, tightly networked group of depositors who could move as one. The first decision turned a rate increase into a balance-sheet hole. The second turned that hole into a stampede. This is a deep dive into exactly how each of those mechanisms works, with the numbers, so that the next time you read that a bank is "well capitalized," you know which footnote to check.

If you want the compact, system-level version of SVB and the Credit Suisse failure that followed it the same month, the one-pager [SVB and Credit Suisse 2023: the bank runs that shook the system](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) is the place to start. This post goes much deeper on the bank's own balance sheet — the duration trap, the accounting, the run mechanics, and the systemic-risk decision.

## Foundations: the four ideas you need before the story makes sense

SVB's failure is often told as a morality tale about reckless tech bankers, but the truth is more interesting and more useful: every step of it followed from a handful of plain mechanical facts about how banks work. Before we walk through the collapse, let's build those facts from zero. A reader who already knows what duration and held-to-maturity accounting mean can skim; everyone else should not skip this.

### What a bank actually is, in one sentence

A bank is a machine that **borrows short and lends long**. It takes money you can demand back at any moment — your checking and savings deposits — and it puts that money into things that pay it back slowly: loans that run for years, bonds that mature in a decade. The gap between the low rate it pays you on deposits and the higher rate it earns on those longer assets is the bank's profit, called the *net interest margin*. (A *margin* here just means the spread between two rates, expressed as a percentage of the bank's earning assets.)

This trick has a name — **maturity transformation** — and it is genuinely useful to society: it lets short-term savings fund long-term investment. But it carries a permanent, structural fragility. The bank has promised to give depositors their money back on demand, while its money is tied up in assets it cannot instantly turn into cash at full value. As long as depositors trust the bank and don't all ask for their money at once, the trick works. The day they stop trusting it, the machine breaks. Hold that thought; it is the spine of the entire story.

### A bank's balance sheet, and where the cushion is

A balance sheet has three parts, and they always satisfy one equation:

$$
\text{Assets} = \text{Liabilities} + \text{Equity}
$$

For a bank, the **assets** are the things it owns and earns from: cash, loans it has made, and securities (bonds) it has bought. The **liabilities** are what it owes to others — overwhelmingly, deposits. And the **equity** is what's left over for the owners after you subtract what's owed from what's owned. Equity is the bank's shock absorber: if assets lose value, the loss eats into equity first, before it can ever reach depositors.

Here is the part that trips up newcomers: a bank runs on *very little* equity. A typical large commercial bank funds itself with roughly 8% equity and the rest borrowed — call it 92% deposits and other debt. That means the bank is *levered* about 12.5 to 1: every \$1 of the owners' money supports about \$12.50 of assets. Leverage is what makes banking profitable, because the owners earn a return on the whole \$12.50, not just their \$1. But leverage is also what makes banking deadly: if the assets fall in value by just 8%, the entire equity cushion is gone, and the bank is insolvent — its assets are worth less than what it owes. A small percentage loss on a big, levered balance sheet is an enormous loss relative to the thin sliver of equity underneath it. Remember 8% and 12.5×; they explain why a bank can look fine and be one bad quarter from death.

### Duration: why a long bond is dangerous when rates rise

A bond is a loan you make to a borrower — a government, a company — that pays you fixed interest (the *coupon*) for a set number of years and then returns your principal at the end. The crucial, counterintuitive fact about bonds is this: **when market interest rates rise, the price of an existing bond falls.**

Why? Suppose you bought a 10-year bond last year that pays a 1.5% coupon, because that's what 1.5% was the going rate. This year, new bonds pay 5%. Nobody will buy your old 1.5% bond at the price you paid — why accept 1.5% when 5% is on offer? The only way your bond sells is at a discount steep enough that the buyer's *total* return (your low coupon plus the bargain price) matches the 5% they could get elsewhere. So your bond's market price drops.

*Duration* is the number that measures how much. (Duration is, roughly, the weighted-average number of years until you get your money back, and it doubles as the sensitivity of the bond's price to rates: a bond with a duration of 7 falls about 7% for every 1 percentage-point rise in yield.) The longer the bond's maturity, the higher its duration, and the more violently its price moves when rates change. A 30-year mortgage bond is far more rate-sensitive than a 2-year Treasury note. If you want the full treatment of why duration is the single most important number in bond investing, the fixed-income deep dive [Duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) covers it; here we only need the one fact that long bonds fall hard when rates rise.

### Held-to-maturity vs available-for-sale: the accounting that hides the loss

When a bank buys bonds, accounting rules let it sort them into two buckets, and the difference is at the heart of the SVB story.

- **Available-for-sale (AFS)** securities are marked to market: their reported value moves up and down with the market, and the gains or losses flow into a part of equity called *accumulated other comprehensive income*. The loss shows up on the balance sheet (though it can be excluded from the headline regulatory capital ratio of smaller banks).
- **Held-to-maturity (HTM)** securities are different. If a bank declares its honest intention and ability to hold a bond until it matures and gets its full principal back, it can carry that bond at *amortized cost* — essentially, the price it paid — and never mark it to market on the balance sheet at all. The market value can collapse, and the reported value doesn't move. The unrealized loss is disclosed only in a footnote.

You can already feel the trap. HTM accounting is reasonable *if* the bank really can hold to maturity. But that ability depends on never being forced to sell — and a bank is forced to sell when depositors run. The moment SVB had to sell those "held-to-maturity" bonds to raise cash, the footnote loss became a real, realized loss, and the gap between the comforting book value and the brutal market value snapped shut. Keep these four ideas — maturity transformation, the thin equity cushion, duration, and the HTM footnote — in your head. The rest of the story is just these four colliding.

## How SVB got the deposits: the tech boom flood

Every bank's story starts with its funding, and SVB's funding came from a single, concentrated, and unusually flighty source: the venture-capital ecosystem. SVB banked startups, the VC funds that financed them, and the people who ran both. When a startup raised a funding round, the cash landed in an SVB account. When a VC fund closed, its committed capital sat at SVB. The bank's whole identity was being the default bank of the technology economy.

That model was a quiet machine for years. Then 2020 and 2021 happened. With interest rates slashed to near zero and a torrent of capital chasing technology, venture funding exploded. Startups raised enormous rounds; the money flowed straight into SVB. The bank's deposits roughly tripled in about two years — from around \$60 billion at the start of 2020 to about \$175 billion by the end of 2021. That is a staggering rate of growth for a bank. Deposits are the raw material of banking, and SVB suddenly had a mountain of it.

Here is the first thing that should make a careful reader nervous, even before any bonds are bought. These were not sticky retail deposits — the granular checking accounts of millions of households who rarely move their money and are fully insured. These were large corporate balances: a single startup might hold \$50 million or \$200 million in one account. They were *operating* cash that companies needed available, parked by a small number of sophisticated treasurers who all watched the same signals. And because almost every account was far above the \$250,000 federal insurance limit, the depositors had every incentive to bolt at the first sign of trouble. SVB had taken on the most run-prone kind of funding there is, in enormous size, very fast.

It's worth pausing on *why* this kind of deposit is so different from the household savings that fund a normal bank, because the contrast is the whole game. A bank's funding has a property called **deposit stickiness** — the tendency of deposits to stay put even when the depositor could earn a little more elsewhere, or when there's a scare. Stickiness comes from three things: small balances (it isn't worth a household's time to chase a few extra dollars or to panic over a bank below the insurance limit), large numbers of independent depositors (so their decisions don't correlate), and a relationship that's costly to move (your salary, your mortgage, your direct debits all run through one account). A classic retail bank scores high on all three. Its deposits are *behaviorally* long-term even though they are *contractually* on demand, and that behavioral stickiness is what lets the bank safely fund long assets. The whole maturity-transformation trick rests on it.

SVB's funding scored near zero on all three. The balances were huge, so chasing yield or fleeing risk was very much worth the depositor's time. The depositors were few and *correlated* — they took advice from the same venture funds and moved in herds rather than independently. And moving the money was nearly frictionless: a corporate treasurer can wire a balance to another bank in minutes, with none of the switching cost that pins a household in place. SVB had built a deposit base that was contractually on demand *and* behaviorally on demand. There was no behavioral stickiness underneath the contractual fragility — and that, far more than the size of any single number, is the structural reason it could die in a day. We'll quantify exactly how run-prone shortly. First, what did the bank *do* with all that money?

## The bond bet: parking cheap money in long bonds at the worst possible time

A bank with a flood of new deposits has to put the money somewhere that earns more than it pays out. The obvious place is loans. But SVB's customers — cash-rich startups that had just raised money — didn't need to borrow much; they were depositing, not borrowing. So SVB had far more deposits than it had good loans to make. The excess had to go into securities.

This is where the fatal decision was made. With rates near zero in 2020–2021, short-term bonds paid almost nothing. To earn a respectable yield, SVB reached for *longer-dated* bonds — mostly long-term U.S. Treasuries and government-backed mortgage securities — and it put a huge share of them into the held-to-maturity bucket. By the end of 2022, SVB held roughly \$91 billion in HTM securities, on top of its available-for-sale book. These were safe in the sense that the borrowers (the U.S. government and its agencies) were never going to default. But "safe from default" is not the same as "safe from rate moves," and SVB had loaded up on exactly the assets most sensitive to rates, at the exact moment rates had nowhere to go but up.

Think about the position it had built. On one side of the balance sheet: deposits that could leave tomorrow, costing almost nothing. On the other side: tens of billions in bonds that wouldn't mature for years, with prices that would crater if rates rose. The bank was earning a spread on the difference — borrow at near-zero, lend long at 1.5% or 2% — and that spread looked like easy profit while it lasted. But it had quietly become one of the most rate-exposed large banks in the country. The pipeline below traces the trap from the deposit flood all the way to the seizure; everything after this section is the mechanism of each arrow.

Two further choices deepened the trap. The first was the decision to load so much of the book into the *held-to-maturity* bucket rather than available-for-sale. The benefit, in the short run, was cosmetic stability: HTM bonds don't get marked down on the balance sheet as rates rise, so the bank's reported equity stays smooth even when the market value of its assets is falling. The cost is that classifying a bond as HTM is a near-irreversible commitment — sell more than a token amount of your HTM book and accounting rules can force you to mark the *entire* book to market, an event banks treat as catastrophic. So by parking \$91 billion in HTM, SVB didn't just hide the loss; it locked itself out of selling those bonds without detonating the whole portfolio. It traded reported smoothness for real, dangerous rigidity. When the run came and SVB needed liquidity, the largest, most liquid pile of assets it owned was the one it had effectively frozen.

The second choice was about hedging — or the lack of it. A bank that holds long fixed-rate bonds can neutralize most of the rate risk with an *interest-rate swap*: a contract that pays the bank more as rates rise, offsetting the fall in the bonds' value. Plenty of banks do exactly this; it's the standard tool for keeping the duration gap closed. SVB had been winding *down* its hedges through 2021 and into 2022, partly because hedging costs money and shaves the spread the bank was enjoying, and partly on a view that rates would not rise far or fast. Stripping the hedges made the reported earnings look better in the boom — and left the bank fully exposed to the one thing that would kill it. The combination is what makes SVB a teaching case rather than just a casualty: it chose accounting smoothness over flexibility *and* current income over protection, and both choices pointed the same way.

![The duration trap pipeline from cheap deposits to seizure](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-6.png)

#### Worked example: a bond's mark-to-market loss after the hikes

Let's make the bond math concrete, because it is the physical cause of everything that followed. Take one stylized 10-year bond with a 1.5% coupon and a face value of \$100 — roughly the kind of low-coupon, long-dated paper a bank bought in 2021.

When market yields are around 1.5%, the bond is worth about its face value: \$100. Now the Federal Reserve hikes, and the market yield for that maturity climbs to 5%. We re-price the bond by discounting its remaining cash flows — ten years of \$1.50 coupons plus the \$100 face at the end — at the new 5% rate:

$$
P = \sum_{t=1}^{10} \frac{1.5}{(1.05)^t} + \frac{100}{(1.05)^{10}} \approx 73
$$

The bond's market price falls from \$100 to about \$73 — a drop of roughly **27%**, more than a quarter of its value, even though not a single payment was missed and the U.S. Treasury is as creditworthy as ever. The chart below plots that decline across the whole range of yields.

![Ten-year bond price falling as yield rises from 0.5 to 5 percent](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-7.png)

The intuition: a long, fixed-rate bond is a promise to pay you yesterday's low rate for a decade, and when today's rates are far higher, that promise is worth dramatically less. Now multiply a 20%-plus haircut across roughly \$91 billion of HTM bonds, and you can see the hole forming.

## The rate shock: 2022, the fastest tightening in forty years

The bond bet was a bet that rates would stay low. In 2022, the bet lost spectacularly. To fight the worst inflation in four decades, the Federal Reserve raised its policy rate from essentially 0% at the start of the year to over 4.25% by December, and onward past 5% in early 2023 — the fastest, steepest tightening cycle since the early 1980s. Every one of those hikes pushed bond prices down, and SVB's long book was right in the path.

The damage was not on the part of the balance sheet everyone watches. SVB's loans were fine; its borrowers weren't defaulting. The damage was on the securities book, and most of it was sitting in the HTM bucket where accounting rules let it stay invisible on the headline balance sheet. By the end of 2022, the *unrealized* loss across SVB's securities — the gap between what the bonds were carried at and what they were actually worth in the market — had grown to roughly \$17 billion. That number deserves a moment of silence, because of what it sat next to.

#### Worked example: the HTM loss versus equity

Here is the calculation that should have set off every alarm. At the end of 2022, SVB reported total shareholder equity of about \$16 billion. That is the cushion — the buffer that absorbs losses before depositors are touched.

The unrealized loss on its securities was about **\$17 billion**.

$$
\text{Unrealized loss} \approx \$17\text{bn} \;>\; \text{Total equity} \approx \$16\text{bn}
$$

Set the two side by side and the conclusion is unavoidable: the hidden loss on the bond book had grown *larger than the entire equity of the bank*. On an amortized-cost basis, SVB looked well capitalized. On a mark-to-market basis, its equity had already been wiped out — the bank was, in economic terms, insolvent, and had been for some time. The only reason it didn't show was the HTM footnote.

![SVB unrealized bond loss compared to total equity](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-2.png)

The intuition: a 12.5×-levered balance sheet can only survive a loss of a few percent of assets before the equity is gone, and a multi-year duration book in the fastest hiking cycle in forty years lost far more than a few percent. SVB's leverage and its duration were a lit fuse.

This is the precise mechanism that interest-rate risk managers are supposed to watch — the sensitivity of a bank's economic equity to a change in rates, measured by frameworks with names like *EVE* (economic value of equity) and the *duration gap*. SVB's gap was enormous and largely unhedged. The dedicated treatment of how this risk is measured and managed lives in [Interest rate risk in the banking book (IRRBB) and the duration gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap); SVB is the textbook case of what happens when that gap is left wide open.

### Why the trade looked so attractive while it lasted

To understand why a bank's management would build such a trap, you have to see how good the trade looked before it blew up — because the profit was real right up to the moment it wasn't. The whole appeal of the duration trade is that a longer bond pays a higher yield than a short one in a normal, upward-sloping yield curve, so a bank that funds long bonds with near-free deposits books a fat spread today and worries about the rate risk later, if at all.

#### Worked example: the seductive spread that built the trap

Let's price the trade the way it looked to SVB in 2021. Take \$10 billion of deposits the bank paid essentially **0.1%** on — corporate operating cash earns almost nothing in a zero-rate world. Invest it in 10-year bonds yielding **1.6%**. The net interest margin on that slice is the difference:

$$
\text{Spread} = 1.6\% - 0.1\% = 1.5\% \implies 0.015 \times \$10\text{bn} = \$150\text{m per year}
$$

That \$150 million of annual profit, on this one \$10bn slice, was almost pure margin — the deposits cost nearly nothing and the bonds carried no credit risk. Scale it across the book and you see why management felt it was being prudent: it was earning a healthy spread on the safest borrowers on earth, the U.S. government and its agencies. What that calculation leaves out is the *price risk*. The same \$10bn of bonds that earned \$150m a year would, when yields rose 3.5 points to 5%, lose roughly \$2.5–\$2.7 billion of market value — wiping out almost twenty years of that spread in a single year of rate moves. The intuition: a high spread on a long bond is not free money, it is rent paid for taking duration risk, and the rent is dwarfed by the loss when the risk shows up. SVB collected the rent and ignored the bill.

This points to the governance failure underneath the accounting one. A bank's treasury function and its asset-liability committee (ALCO) exist precisely to ask "what does this spread cost us in rate risk, and have we hedged it?" — and SVB's controls there were notably weak. The bank had gone for an extended period without a permanent chief risk officer in 2022, its own internal models reportedly flagged the rising rate exposure, and its supervisors at the Federal Reserve had issued multiple findings about the bank's risk management before the failure. The trap was not invisible to the people whose job was to see it; it was tolerated because the spread it generated was real and immediate, while the risk was contingent and deferred. That asymmetry — sure profit now, possible ruin later — is the oldest temptation in banking, and it is why risk governance has to be a hard constraint rather than a polite suggestion.

## The hidden loss made visible: book value versus market value

The deepest lesson of SVB is about the difference between two ways of valuing the same bank, and how the comforting one can mask a fatal one. Let's lay the two balance sheets side by side.

At **book value** — the way SVB reported itself — the picture looked solid. Assets of about \$209 billion (with the bonds carried at the cost SVB paid), deposits and other borrowings of roughly \$193 billion, and equity of about \$16 billion. A well-capitalized bank, by the headline numbers.

At **market value** — marking the bonds to what they were actually worth — the picture was a corpse. Knock the roughly \$17 billion unrealized loss off the asset side and the bank's assets fall to about \$192 billion, against the same \$193 billion of liabilities. The equity is gone; the bank owes more than it owns. The before-and-after below is the single most important picture in this story.

![SVB balance sheet at book value versus marked to market](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-4.png)

The accounting was not fraudulent. SVB followed the rules, and the loss was disclosed — in a footnote, where the market eventually found it. But the episode is a permanent argument against ever taking a bank's headline capital ratio at face value when rates have moved a long way. Two banks with identical reported equity can be worlds apart if one is sitting on a giant unmarked HTM loss and the other isn't. The truth was always there; it just required someone to do the subtraction. In early 2023, the market did the subtraction.

#### Worked example: how a small asset loss becomes total equity loss

It is worth seeing the leverage amplifier in its own right, because it explains why "only" an 8% loss on assets is catastrophic. Take a stylized bank funded the standard way: \$100 of assets, \$92 of deposits and debt, \$8 of equity (8% — about 12.5× leverage).

Now the asset side falls by 9% — say a duration-driven mark-down on a long bond book — so assets drop from \$100 to \$91.

$$
\text{New equity} = \$91 - \$92 = -\$1
$$

A 9% fall in *assets* produced a *negative* equity: the bank now owes \$1 more than it owns. The loss didn't need to be large in percentage terms; it only needed to exceed the thin 8% sliver of equity. SVB's real loss, as a share of its bond-heavy assets, did exactly that. The intuition: leverage means a modest asset loss lands on equity magnified by roughly 12×, which is why banks die from problems that would barely dent an unlevered investor.

## The concentrated, uninsured deposit base: the run waiting to happen

A balance-sheet hole is dangerous, but a hole alone does not kill a bank overnight. Many banks were sitting on large unrealized bond losses in 2022 — the entire U.S. banking system had several hundred billion dollars of them. What turned SVB's hole into a thirty-six-hour death was the *liability* side: who its depositors were and how they behaved.

Recall that deposit insurance in the United States covers up to \$250,000 per depositor, per bank. Below that line, your money is guaranteed by the FDIC even if the bank fails; you have literally no reason to run, because you cannot lose. Above that line, you are an unsecured creditor of the bank — if it fails, you join the line of claimants and may not get everything back. Insured depositors are the calm, sticky base of a bank. Uninsured depositors are the flight risk.

SVB's deposit base was almost entirely flight risk. Because its customers were companies parking operating cash in large balances, an estimated **94% of its deposits were uninsured** — above the \$250,000 limit. Compare that to a typical U.S. bank, where roughly half of deposits are insured. The chart makes the contrast stark.

![Insured versus uninsured deposit share at SVB and a typical bank](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-5.png)

Now layer on a second feature: these uninsured depositors were not strangers to each other. They were a tightly connected community — startups backed by the same handful of venture funds, founders in the same accelerator cohorts, finance chiefs in the same Slack groups and WhatsApp threads. When a few prominent VCs began quietly telling their portfolio companies to pull money out, the advice did not trickle through the population over weeks the way a rumor reached depositors lining up outside a branch in 1933. It propagated through the entire networked base in *hours*. A concentrated, uninsured, networked deposit base is the most combustible funding structure a bank can have, and SVB had built the purest example of it in modern banking.

#### Worked example: deposit concentration and the uninsured run incentive

Put yourself in the seat of a startup CFO on Thursday morning with \$30 million of company cash at SVB. The insurance limit covers \$250,000 of it — less than 1% of your balance. If SVB fails, the other \$29.75 million is at risk; you might recover most of it eventually, but you have payroll to run *next week*, and "eventually" doesn't make payroll.

Your decision is brutally simple. If you move the money and SVB survives, you've lost nothing but a few hours of work. If you leave the money and SVB fails, you may not be able to pay your employees and you could sink your own company. The expected cost of running is tiny; the expected cost of staying is potentially fatal. Every rational uninsured depositor faces the same arithmetic, and crucially, knows that *every other depositor faces it too*. The dominant move is to run first. Multiply one CFO's \$30 million by a base that is 94% uninsured, and you get the result on the next page.

The intuition: deposit insurance exists precisely to remove this incentive for ordinary depositors, which is why it is the bedrock of bank stability. Where insurance doesn't reach — the uninsured 94% — the old, self-fulfilling logic of the bank run is alive and well. The mechanics of why a run is rational for the individual and ruinous for the group are laid out in [The anatomy of a bank run: from whisper to collapse](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse); SVB is that model running at internet speed.

## The trigger: the Wednesday announcement that lit the match

Through 2022, the loss sat in the footnote and the depositors sat still. Something had to make the hidden hole visible and frighten the depositors at the same time. That something was a press release SVB issued on Wednesday, March 8, 2023 — an attempt to fix the problem that instead detonated it.

By early 2023, SVB faced a slow squeeze. The venture funding boom had cooled, so deposits were drifting *out* as startups burned cash and weren't replenishing it with new rounds. To meet those outflows, SVB had been selling some of its more liquid bonds — and selling bonds in a higher-rate world meant *realizing* losses it had previously been able to ignore. Management decided to clean it up in one move: sell about \$21 billion of the available-for-sale securities book, take the loss, and simultaneously raise fresh capital to plug the resulting hole. On March 8, SVB announced it had sold those securities at a loss of about **\$1.8 billion** and would raise roughly **\$2.25 billion** in new equity to shore up the balance sheet.

The intent was reassurance. The effect was the opposite. To a sophisticated audience of VCs and startup treasurers, the announcement screamed three things at once: the losses everyone had been able to ignore in the footnote were now *real and being booked*; the bank was suddenly scrambling for capital, which it would only do under stress; and the capital raise might not even succeed. Within hours, the read was unanimous — and the figure of the loss next to the bank's equity was a calculation any analyst could do on the back of an envelope. The match was lit. What happened next happened faster than any large bank had ever failed.

## The 36-hour digital run: hour by hour

The run on SVB compressed into a window so short it broke the historical model of how banks die. The classic image of a bank run — physical queues snaking around the block, depositors waiting to reach a teller — assumes friction. People have to travel, line up, and wait, which gives the bank and regulators time to respond. SVB's run had no friction. Its depositors moved money with a few taps on a phone, and they coordinated through the same digital channels in real time. This was a run at the speed of a wire transfer and a group chat.

#### Worked example: the day-by-day drain

Let's account for the money, because the magnitude is the point. SVB's total deposit base going into the crisis was about **\$175 billion**.

- **Wednesday, March 8 (evening):** SVB announces the bond sale loss and the capital raise. Word spreads through the VC network overnight.
- **Thursday, March 9:** Depositors attempt to withdraw about **\$42 billion** in a single day — roughly 24% of all deposits, gone or requested in hours. SVB's stock falls about 60%. The planned \$2.25bn capital raise collapses as investors back away from a bank visibly bleeding deposits.
- **Friday, March 10:** Another **\$100 billion** of withdrawals is queued and expected the moment the bank opens. That would bring the two-day total to roughly \$142 billion — about **81% of the entire deposit base** — far more cash than any bank holds. Before that wave can clear, regulators step in: at around 8 a.m. Pacific, California's DFPI closes the bank and appoints the FDIC as receiver.

$$
\frac{\$42\text{bn} + \$100\text{bn}}{\$175\text{bn}} \approx 81\%
$$

No bank on earth can pay out 81% of its deposits in two days, because no bank holds anything close to that much in cash — the whole business is built on lending the money out into illiquid assets. The chart below shows the deposit base being shredded.

![SVB deposit drain on March 9 and queued for March 10 against the deposit base](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-3.png)

The intuition: a bank's promise to return everyone's money on demand is only ever a promise about *some* people asking at a time. When 81% of the money tries to leave in two days, the promise was never physically keepable; the only question was how fast the truth would arrive. In 2023, it arrived in thirty-six hours.

The deeper point is that this was a **liquidity** death triggered by a **solvency** worry, and the two are not the same. A bank is *insolvent* when its assets are worth less than it owes — which, marked to market, SVB arguably was. A bank is *illiquid* when it can't turn its assets into cash fast enough to meet withdrawals, regardless of whether those assets are ultimately worth enough. SVB had to sell long bonds into a falling market to meet the run, crystallizing the very losses that made everyone want to run, in a doom loop. Even a perfectly solvent bank can be killed by a fast enough run, because being forced to dump long assets at fire-sale prices manufactures the insolvency the run feared.

It's worth dwelling on just how much the digital nature of this run changed the physics. The runs that shaped banking regulation — the waves of failures in 1930s America, the queues outside Northern Rock in 2007 — all moved at the speed of physical bodies and paper. A depositor had to learn there was trouble (newspapers, word of mouth), travel to a branch, stand in a line, and wait to reach a teller who counted out cash by hand. That friction is not a bug; it is a *circuit breaker*. It gives the bank time to summon liquidity, gives regulators time to arrange support, and gives panic time to cool. Deposit insurance was invented in 1933 precisely to remove the *reason* to join those queues. SVB's run removed the queues themselves. There was no branch to visit and no line to stand in — a treasurer logged in, clicked, and moved tens of millions in the time it takes to read this sentence, while coordinating with a hundred peers in a shared chat. The circuit breaker that all the old rules quietly assumed had simply been deleted by technology. A run that would once have taken a week to drain a bank now drained SVB faster than any human institution could respond. This is the single most important new fact about bank fragility in the smartphone era, and SVB is its first large proof.

## The seizure and why this was a bank run, not a fraud

It is worth being precise about what kind of failure this was, because the public conversation muddled it. SVB was not Enron or Wirecard. There was no faked cash, no off-books fraud, no hidden bad loans. Every number was disclosed; the auditors had signed off. The failure was a failure of *risk management* — a decision to run an enormous, unhedged interest-rate mismatch funded by the flightiest deposits in the country — not a failure of honesty.

When the FDIC takes over a failed bank, its normal playbook is to protect insured depositors up to \$250,000 (paid out within days, that's the whole point of insurance), sell off the failed bank's assets, and pay uninsured depositors whatever the asset sales recover, over time, with no guarantee. Under that normal playbook, SVB's uninsured depositors — 94% of the base, including thousands of startups with payroll due — faced an uncertain, possibly months-long wait to recover an uncertain fraction of their money. That prospect is what made the weekend of March 11–12 so tense across the entire technology economy.

## The systemic-risk exception: the backstop that stopped the dominoes

Here the story turns from one bank to the whole system. Over the weekend, regulators faced a frightening question: if SVB's uninsured depositors took losses, what would the uninsured depositors at *every other* mid-sized bank do on Monday morning? The honest answer was: they would run too. The same arithmetic that drove the SVB run — *I am uninsured, I can lose, the cost of running is small, run first* — applied to uninsured depositors everywhere. The risk was a cascade of runs on regional banks that had nothing to do with SVB's specific mistakes but shared its funding vulnerability.

So on Sunday evening, March 12, the Treasury, the Federal Reserve, and the FDIC invoked a rarely used legal tool called the **systemic-risk exception**. They declared that *all* deposits at SVB — insured and uninsured alike — would be made whole, backed not by taxpayers but by the FDIC's Deposit Insurance Fund (which is funded by levies on banks). At the same time, the Fed launched the **Bank Term Funding Program** (BTFP), a facility that let any bank borrow against its bonds at the bonds' *face value* rather than their depressed market value — directly defusing the duration trap by giving banks a way to raise cash against underwater bonds without selling them and crystallizing the loss.

The BTFP is worth understanding, because it was engineered to attack the exact doom loop that killed SVB. In a normal central-bank loan, a bank pledges bonds as collateral and gets cash worth slightly *less* than the bonds' market value (the lender keeps a safety margin called a *haircut*). The cruelty of 2023 was that the bonds' market value had collapsed — so a bank with \$100 of face-value bonds worth only \$80 in the market could borrow against the \$80, not enough to stop a run. The BTFP threw that logic out: it valued the collateral at *par* — full face value — ignoring the rate-driven loss entirely.

#### Worked example: how the BTFP defuses the trap

Suppose a regional bank, frightened by SVB, faces \$80 of deposit withdrawals and owns \$100 of face-value Treasuries now worth only \$80 in the market. Its options:

$$
\text{Sell the bonds: raise } \$80, \text{ but realize a } \$20 \text{ loss} \rightarrow \text{equity falls } \$20
$$

$$
\text{BTFP loan: pledge the bonds, borrow } \$100 \text{ at par} \rightarrow \text{raise the } \$80, \text{ realize } \$0 \text{ loss}
$$

The first path forces the bank to crystallize a \$20 loss and shrink its equity at the worst possible moment — exactly the spiral SVB fell into. The second path lets it meet the run with cash and *no* realized loss, because the loan is sized to face value, not market value. The intuition: the BTFP broke the link between "I need cash" and "I must sell at a loss," which is the mechanical heart of the duration trap. By making underwater bonds a source of full-value cash, it removed the reason a solvent-but-illiquid bank had to die. That is why the facility, more than the deposit guarantee itself, is what actually calmed the regional-bank system.

The move worked, in the narrow sense: the regional-bank run did not become a full system-wide panic, though Signature Bank failed the same weekend and First Republic followed weeks later. But it also reopened the oldest argument in banking. By guaranteeing uninsured deposits after the fact, regulators arguably told every large depositor and every bank that the \$250,000 limit is negotiable in a crisis — which weakens the incentive for depositors to ever monitor their bank's risk, and for banks to ever restrain it. This is the **moral hazard** trade-off at the center of every bailout, and the full treatment of why deposit insurance and a lender of last resort are both essential and corrosive lives in [Deposit insurance, the lender of last resort, and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard). SVB forced regulators to choose between two bad options — let depositors take losses and risk a cascade, or backstop them and feed moral hazard — and they chose the cascade-prevention side.

## Common misconceptions

**"SVB failed because it made bad loans."** This is the most common and most wrong belief about SVB, because it pattern-matches to 2008. SVB's *loans* were fine — defaults were minimal. The losses were on its *securities*, specifically long-dated, default-free government bonds that lost market value purely because rates rose. SVB is the rare large failure with almost no credit losses; its killer was interest-rate risk, not credit risk. Mistaking the two means you'd look for the next SVB in loan books, when you should be looking in the securities portfolio and the duration gap.

**"The bonds were risky."** They were among the safest assets on earth by credit standards — U.S. Treasuries and government-agency mortgage securities, with essentially zero default risk. The risk was not that they'd go unpaid; it was their *duration*. A perfectly safe bond can still lose a quarter of its market value when rates jump, as the worked example showed. "Safe from default" and "safe from rate moves" are different properties, and conflating them is how a bank convinces itself a duration trap is a conservative strategy.

**"Held-to-maturity accounting was a trick to hide losses."** HTM accounting is legitimate and disclosed — the unrealized loss appeared in SVB's footnotes; analysts who read them knew. The flaw is conceptual, not dishonest: HTM lets a bank carry bonds at cost *on the condition* that it will hold them to maturity, but that condition silently assumes the bank is never forced to sell. A bank facing a run *is* forced to sell, at which point the footnote loss becomes a balance-sheet loss instantly. HTM doesn't hide losses from a careful reader; it hides them from the headline ratio, and it dissolves exactly when you most need it to hold.

**"A bigger equity cushion would have saved SVB."** More equity would have helped, but the run was fast enough that capital was not the binding constraint that week — *liquidity* was. SVB couldn't have sold \$142 billion of long bonds in two days at any price that would have kept it solvent. The deeper fix was on the asset side (don't build a giant unhedged duration gap) and the liability side (don't fund yourself almost entirely with concentrated, uninsured, networked deposits) — not simply holding one or two more points of capital against a run that drained 81% of deposits in 36 hours.

**"This couldn't happen at a big bank."** The interest-rate-risk exposure SVB had was extreme, but unrealized bond losses were system-wide in 2022 — the FDIC tallied hundreds of billions across U.S. banks. What protected the largest banks was not the absence of bond losses but a more diversified, more insured, less networked deposit base, plus far more active hedging of their rate risk. The *mechanism* that broke SVB exists at every bank; what varies is how much of it each bank carries and how flight-prone its funding is.

## How it shows up in real banks: the wider 2023 episode

SVB was not an isolated accident; it was the first domino in a regional-banking stress that ran for weeks, and the same two mechanisms — duration losses plus run-prone funding — show up across the failures and near-failures. The graph below contrasts why SVB's depositor base ran while a diversified bank's stayed put, which is the single most portable lesson from the whole episode.

![Why SVB ran while a diversified bank did not](/imgs/blogs/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run-8.png)

**Silicon Valley Bank, March 10, 2023.** The case we've walked through: \$209bn of assets, a \$17bn unrealized bond loss against \$16bn of equity, 94% uninsured deposits, a \$1.8bn realized loss and failed \$2.25bn raise on Wednesday, \$42bn out on Thursday, and seizure on Friday. The purest example of the duration trap meeting the most flight-prone funding base imaginable. Every later event that weekend was, in part, a reaction to this one.

**Signature Bank, March 12, 2023.** Closed by New York regulators two days after SVB. Signature had a different customer twist — a large share of deposits tied to the crypto industry — but the same core flaw: a concentrated, largely uninsured deposit base that ran hard once confidence cracked. It was the third-largest U.S. bank failure at the time, and it failed in the same window as SVB precisely because uninsured depositors everywhere were suddenly asking the same question.

**First Republic Bank, May 1, 2023.** The slow-motion sequel. First Republic served wealthy clients with large, mostly uninsured deposits and held a lot of long-duration, low-rate jumbo mortgages — its own version of the duration trap on the asset side. After SVB, its uninsured depositors began leaving; a \$30 billion deposit infusion from eleven large banks bought time but not survival. It bled deposits for weeks before regulators seized it and sold it to JPMorgan Chase. It remains the second-largest U.S. bank failure by assets, behind only Washington Mutual in 2008.

**Credit Suisse, March 19, 2023.** A week after SVB, an entirely separate and much larger crisis came to a head in Europe — though through a different door. Credit Suisse's problem was a decade of scandals and eroding trust, not a single duration trap, but the timing was no coincidence: a market already primed to fear bank fragility by SVB pulled its confidence from the weakest globally important bank. Swiss authorities forced a shotgun merger into UBS over a weekend, and controversially wiped out roughly CHF 16 billion of Credit Suisse's riskiest bonds. The shared lesson with SVB is that a bank's deepest asset is trust, and trust can evaporate faster than any capital ratio can be rebuilt.

**The banks that did *not* run.** The most instructive comparison is the dog that didn't bark. The largest U.S. banks held large unrealized bond losses too, yet saw deposits *flow in* during the crisis — a flight to perceived safety. Their funding was granular (millions of small accounts), more insured, less networked, and hedged. Same disease on the asset side; very different outcome, because the liability side was built to withstand fear rather than amplify it. That contrast is the whole practical takeaway: the asset-side trap is dangerous, but it is the *funding structure* that decides whether a balance-sheet hole becomes a thirty-six-hour death.

## The takeaway: how to read a bank after SVB

Strip SVB down to its bones and it is the series' spine in its most violent form. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and lends long, earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive. SVB violated both halves at once. It stretched the maturity transformation to an extreme — overnight money funding multi-year bonds — so a rate shock blew a hole in the cushion. And it funded that stretch with the least trustworthy, most coordinated deposit base in the country, so the moment the hole became visible, the trust vanished in hours. The duration trap built the hole; the funding concentration turned it into a stampede.

So when you next read a bank's results, you now know the three questions SVB teaches you to ask, in order. First, *what is hiding in the securities footnote?* Find the held-to-maturity book, find its unrealized loss, and subtract that loss from the reported equity — if the answer is near zero or negative, the bank's real capital is gone no matter what the headline ratio says. Second, *who funds this bank?* Find the share of deposits that are uninsured, and ask how concentrated and how networked those depositors are; a high uninsured share is dry tinder, and a base that all shares one group chat is dry tinder next to an open flame. Third, *is the rate risk hedged?* A duration gap is only a trap if it's left open; a bank that swaps away its rate exposure can hold the same bonds and survive the same hikes.

The uncomfortable, durable insight is that SVB was *solvent on paper and dead in practice*, and the gap between those two states was made of trust and a footnote. No single number on the front page of its filings was a lie. The failure was in the relationship between the numbers — a loss as big as the equity, sitting under a deposit base that could leave overnight — and in the speed at which a connected, uninsured crowd could act on the arithmetic once it became obvious. A bank does not die when it becomes insolvent; it dies when its depositors *decide* it has, and in the digital age that decision can be made and executed before the close of business. That is the real lesson of the thirty-six hours: capital is what a bank reports, but confidence is what funds it, and confidence has no settlement delay.

## Further reading & cross-links

- [SVB and Credit Suisse 2023: the bank runs that shook the system](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the compact, system-level overview of both 2023 failures; start here for the bird's-eye view, then return for the balance-sheet depth.
- [Interest rate risk in the banking book (IRRBB) and the duration gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap) — the risk-management deep dive on exactly the mechanism (EVE, the duration gap, repricing risk) that SVB left unhedged.
- [The anatomy of a bank run: from whisper to collapse](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse) — why a run is individually rational and collectively ruinous, and how the digital age compressed it from weeks to hours.
- [Deposit insurance, the lender of last resort, and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — the trade-off behind the systemic-risk exception, and why backstopping uninsured depositors both prevents cascades and feeds the next risk.
- [Duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) — the bond-market foundation: why a long, low-coupon bond loses a quarter of its value when yields jump, which is the physical cause of SVB's hole.

*This is educational, not investment advice. The figures describe a specific historical episode and are drawn from regulatory filings and public reporting as of 2023.*
