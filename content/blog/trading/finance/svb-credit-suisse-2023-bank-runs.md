---
title: "Silicon Valley Bank and Credit Suisse 2023: The Speed of a Modern Bank Run"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How rising interest rates turned a bank's safe bonds into a hidden loss, how its connected depositors fled at smartphone speed, and how the panic toppled Credit Suisse and rewrote the rules on AT1 bonds."
tags: ["banking", "bank-run", "interest-rates", "duration-risk", "svb", "credit-suisse", "at1-bonds", "financial-crisis", "fdic", "case-study"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Silicon Valley Bank held tens of billions in "safe" long-dated bonds whose value collapsed when interest rates rose, and when its concentrated, uninsured, hyper-connected depositors all ran at the same time, the bank failed in roughly two days.
>
> - On March 8, 2023 SVB disclosed a ~\$1.8 billion loss on bond sales and a capital raise; depositors requested ~\$42 billion of withdrawals in a single day - about a quarter of all deposits.
> - The core mechanism was duration risk hidden by accounting: SVB's bonds were marked at face value under "held-to-maturity" rules, so a ~\$15 billion unrealized loss stayed invisible until a deposit run forced the bank to sell and make the loss real.
> - SVB was seized on March 10 (the second-largest US bank failure at the time), Signature Bank failed two days later, and by March 19 the panic had reached Europe, where UBS was pushed to absorb Credit Suisse and ~\$17 billion of its AT1 bonds were written to zero.
> - Insured deposits never lost a cent; the Fed and FDIC backstopped all deposits and launched an emergency lending facility, but bondholders and shareholders were largely wiped out.
> - The durable lesson: in a world of instant transfers and group chats, a slow 1930s run that took weeks can now happen in hours - and "safe" assets can still sink a bank if their prices fall while depositors flee.

On the morning of Wednesday, March 8, 2023, Silicon Valley Bank was the 16th-largest bank in the United States, the financial home of roughly half of all American venture-backed startups, and by most conventional measures a profitable, well-run institution. It had made almost no risky loans. Its assets were stuffed with US Treasury bonds and government-backed mortgage securities - the safest paper on Earth. Forty-eight hours later it was gone: seized by regulators, its name a synonym for how fast a modern bank can die.

The diagram above is the mental model: a single disclosure about bond losses on a Wednesday became, within eleven days, a transatlantic crisis that ended a 167-year-old Swiss institution and rewrote a corner of the global bond market. The shocking part is not that a bank failed. Banks fail. The shocking part is the *speed*. The 1930s runs that scarred a generation took days and weeks - lines forming around the block, tellers stalling, gold dwindling. SVB lost about a quarter of its entire deposit base in a single business day, with no lines and no tellers, just frightened founders tapping "transfer" on their phones while warning each other in group chats.

![Timeline of the March 2023 banking crisis from SVB to Credit Suisse](/imgs/blogs/svb-credit-suisse-2023-bank-runs-1.png)

This post takes that story apart slowly. We will build, from zero, the handful of ideas you need: what a bank's balance sheet actually is, why bonds fall in value when interest rates rise, the accounting trick that let a giant loss hide in plain sight, what deposit insurance does and does not cover, and the strange instrument called an AT1 bond that detonated in the second act. Then we will walk the blow-up step by step, dissect exactly why it broke, follow the contagion to Credit Suisse, clear up the myths, and place the whole episode in the long lineage of bank runs from 1929 to Northern Rock to Lehman. By the end you should be able to explain to anyone - precisely, with numbers - how a bank full of the world's safest assets managed to vanish in two days.

## Foundations: the building blocks of a bank run

Before the story makes sense, six ideas have to be solid. None of them is hard, but the crisis lives in the seams between them. Take your time here; the rest of the post leans on this section.

### A bank balance sheet, explained simply

A bank is not a vault full of your cash. When you deposit \$10,000, the bank does not lock your specific banknotes in a drawer. It records that it *owes you* \$10,000 - that is a **liability**, something the bank must hand back on demand. Then it takes the money and does something with it: it lends it out, or it buys bonds, or it keeps some as cash reserves. Those loans, bonds, and reserves are the bank's **assets** - things the bank owns that earn it money.

A balance sheet is just a list with two sides that must add up:

```
Assets  =  Liabilities  +  Equity
(loans,    (deposits,       (the owners'
 bonds,     borrowings)      cushion)
 cash)
```

The third term, **equity** (also called capital), is the gap between what the bank owns and what it owes. If a bank has \$100 of assets and \$92 of deposits and other debt, its equity is \$8. Equity is the owners' stake and the shock absorber: if the assets lose value, equity falls first. When equity hits zero, the bank is **insolvent** - its assets are no longer enough to cover what it owes everyone else.

The crucial feature, and the source of all the danger, is the *mismatch* between the two sides. Deposits can be withdrawn instantly - they are short-term and liquid. But the bank's assets are often long-term and illiquid: a 30-year mortgage, a 10-year Treasury bond. A bank promises everyone their money back on demand while holding that money in things it cannot turn back into cash quickly without a loss. This is **maturity transformation**, and it is the entire business model of banking. It works beautifully right up until everyone wants their money at once.

### Why bond prices fall when interest rates rise

This is the single most important mechanical fact in the whole story, so let us build it carefully.

A bond is a loan you make to a borrower (here, the US government). The borrower promises to pay you a fixed amount of interest each year - the **coupon** - and to return your principal at the end - the **maturity date**. If you buy a 10-year Treasury bond for \$100 that pays a 1.5% coupon, you get \$1.50 a year for ten years and your \$100 back at the end. That \$1.50-a-year deal is locked in.

Now suppose that a year later, the government is issuing brand-new 10-year bonds paying 4% - because interest rates in the economy have risen. A new \$100 bond pays \$4 a year. Yours pays \$1.50. Nobody will buy your bond for \$100 when they can get \$4 a year elsewhere for the same \$100. So the *price* of your bond has to fall until its lower coupon, relative to its new lower price, gives a buyer the same 4% return the market now demands. Your bond is still perfectly "safe" - the government will absolutely pay you back - but its market price has dropped.

The longer the bond's remaining life, the bigger the price drop, because the below-market coupon hurts for more years. This sensitivity of a bond's price to interest-rate moves is called **duration**, and it is roughly measured in years. A bond with a duration of 6 means: for every 1 percentage point that rates rise, the bond loses about 6% of its value. Duration is the hidden risk that destroyed SVB - not credit risk (the chance the borrower defaults), but **interest-rate risk** (the chance rates move and your fixed coupon becomes unattractive). We will put exact numbers on this in the first worked example.

### Held-to-maturity vs available-for-sale: the accounting that hides the loss

Here is where it gets subtle, and where the trick lives. A bank holding bonds has to tell the world what they are worth, but the rules let it choose *how*, depending on its stated intentions:

- **Available-for-sale (AFS):** bonds the bank might sell before maturity. These are marked to market - their value on the balance sheet moves up and down with current prices. If they fall, the loss shows up (in a special equity line called "accumulated other comprehensive income"), dragging reported capital down.
- **Held-to-maturity (HTM):** bonds the bank intends and is able to hold all the way to the maturity date. These are carried at **amortized cost** - essentially the price paid - and are *not* marked to market. If their market value falls, the loss does **not** appear on the balance sheet at all. The logic: if you really will hold the bond to maturity, you will get your full principal back, so the temporary price dip in between supposedly does not matter.

That logic has a fatal hole. "If you hold to maturity" assumes you are never *forced* to sell early. A bank that must raise cash in a hurry - say, because depositors are fleeing - has to sell those HTM bonds *now*, at today's depressed prices, turning the hidden loss into a real, realized one. The accounting category does not change the economics; it only changes when the world gets to see them.

By the end of 2022, SVB's HTM bond portfolio carried roughly \$15 billion of **unrealized losses** - losses on paper, invisible on the headline balance sheet, but large enough to wipe out most of the bank's equity if ever made real.

### FDIC insurance and uninsured deposits

In the United States, the Federal Deposit Insurance Corporation (FDIC) guarantees deposits up to **\$250,000 per depositor, per bank**. If your bank fails and you had \$250,000 or less, you get every cent, fast, from the FDIC - no questions, no panic needed. This guarantee is why ordinary people do not run on their banks: there is nothing to run from.

But the limit is per depositor, and \$250,000 is tiny for a company. A startup that just raised a \$30 million funding round and parks it at one bank has \$29.75 million **uninsured** - money it would have to fight to recover, perhaps for cents on the dollar, if the bank failed. Uninsured depositors have every incentive to run at the first whiff of trouble, because they are the ones who eat the loss. SVB's deposit base was extraordinary on exactly this dimension: roughly **94% of its deposits were uninsured** - far above a normal bank's roughly half. That is a vat of dry tinder.

### A bank run, and why it is rational

A **bank run** happens when depositors, fearing the bank cannot pay everyone, rush to withdraw at once. The cruel logic is that running is individually rational even if collectively ruinous. The bank cannot instantly convert its long-term assets to cash, so it pays out on a first-come, first-served basis until it runs dry. If you think others will run, you should run *first*, to be ahead of them - which guarantees the run. Economists call this a self-fulfilling prophecy: the fear of insolvency *causes* the insolvency by forcing fire sales. A perfectly solvent bank can be killed by a run; SVB was arguably already near insolvent on a mark-to-market basis, which made the run not just self-fulfilling but inevitable.

### AT1 / CoCo bonds and the liquidity coverage ratio

Two more terms, needed for the Credit Suisse act.

**Additional Tier 1 (AT1) bonds**, also called **contingent convertibles** or **CoCos**, are a peculiar instrument invented after 2008. They are bonds - they pay interest - but they are designed to absorb losses while the bank is still alive, not just in bankruptcy. If the bank's capital falls below a trigger, or if a regulator declares the bank unviable, AT1 bonds can be converted to equity or **written down to zero** entirely. They pay a fat coupon precisely because they carry this risk. The normal mental model is that AT1 sits *above* common equity in the pecking order: equity, the riskiest layer, should be wiped out *before* AT1 takes a hit. Remember that ordering - it is the controversy.

The **liquidity coverage ratio (LCR)** is a post-2008 rule requiring banks to hold enough high-quality liquid assets (cash and easily sellable bonds) to survive 30 days of heavy outflows. It is meant to ensure a bank can meet a run for a month. SVB had liquid assets, but the *speed and size* of the 2023 run blew past anything the 30-day framework imagined - and SVB's largest pool of "liquid" assets were exactly the underwater HTM bonds it could only sell at a loss.

With those six ideas in hand, the story tells itself.

## The setup: how SVB built a beautiful trap

To understand the failure, you have to understand how reasonable it all looked in 2021.

### A flood of money looking for a home

In 2020 and 2021, two forces collided. The Federal Reserve had slashed interest rates to near zero and was buying bonds to support the economy through the pandemic, flooding the system with cash. At the same time, venture capital was in a historic boom: startups were raising enormous rounds at sky-high valuations. All that freshly raised cash had to sit somewhere, and for the technology and life-sciences world, the default home was Silicon Valley Bank. SVB was not a side player here; it was the ecosystem's central bank, the place where the startup, its founders, its VCs, and its employees all kept their accounts.

The result was a deposit explosion. SVB's deposits roughly tripled from about \$60 billion at the start of 2020 to nearly \$190 billion by early 2022. This sounds like a triumph. For a bank, it is a problem. Deposits are a liability - money you owe and must pay interest on. To make a profit, you have to put that money into higher-yielding assets faster than it pours in. SVB had a tidal wave of cash and nowhere obvious to lend it, because its startup clients were flush and not borrowing.

### Parking the flood in long-dated "safe" bonds

So SVB did what looked prudent: it bought bonds. Specifically, it loaded up on long-dated US Treasury bonds and government-backed mortgage-backed securities (MBS) - assets with essentially zero credit risk. With rates near zero, these long bonds yielded only about 1.5%. To squeeze out that yield, SVB bought *long* maturities, pushing the average duration of its securities book to roughly six years. And it classified the bulk of this portfolio - around \$91 billion of it - as **held-to-maturity**, locking in the accounting treatment that would carry them at cost and keep any price swings off the headline balance sheet.

On paper this was conservative. No subprime mortgages, no exotic derivatives, no Archegos-style leveraged bets. Just government paper. The risk SVB took was not *credit* risk; it was *duration* risk - a giant, undiversified bet that interest rates would stay low. The bank had taken short-term, instantly-withdrawable deposits and locked them into six-year fixed-rate bonds. That is maturity transformation cranked to an extreme.

And for a while, the bet paid handsomely, which is exactly what made it so dangerous. In 2020 and 2021, with deposits costing almost nothing and the long bonds yielding ~1.5%, SVB earned a tidy spread on a fast-growing balance sheet. Its profits rose, its stock climbed, and its strategy looked like disciplined treasury management rather than a leveraged rate bet. This is the recurring shape of financial blowups: the riskiest strategies often *look like* the safest ones right up until the regime changes, because the risk is in a variable (here, interest rates) that has been quiet for years. A bank loading up on long bonds in a zero-rate world is collecting small, steady rewards in exchange for a large, rare risk - the textbook profile of "picking up pennies in front of a steamroller." The pennies were the extra yield from going long; the steamroller was a rate-hiking cycle. For a decade the steamroller did not move. In 2022 it did.

### 2022: the rate hikes detonate the bet

Then inflation arrived, and the Fed reversed course harder and faster than almost anyone expected. Over 2022 it raised its policy rate from near zero to over 4%, and kept going into 2023. For an explanation of exactly how the Fed moves rates and why, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

Every percentage point of rate increase, as we built above, drives down the price of existing bonds - and the longer the duration, the harder the fall. SVB's roughly six-year-duration book got hammered. By the end of 2022, the bonds it carried at about \$91 billion of par value were worth roughly \$76 billion in the market: an **unrealized loss of around \$15 billion**. SVB's total equity at the time was about \$16 billion. In other words, the hidden loss in the bond book was nearly the size of the entire capital cushion. On a mark-to-market basis, the bank was a whisper from insolvency - but thanks to HTM accounting, almost none of that loss appeared on the balance sheet that investors and depositors saw.

![SVB bond book shown at par value versus its lower market value](/imgs/blogs/svb-credit-suisse-2023-bank-runs-2.png)

### The depositors were concentrated, uninsured, and connected

A \$15 billion hidden hole is survivable if the bank can hold the bonds to maturity and earn its way out. That requires one thing above all: the deposits must stay put. SVB's deposits were the opposite of sticky. They were concentrated in a single, tight-knit industry - tech and venture - where everyone knew everyone. They were overwhelmingly uninsured (about 94%), so depositors had every reason to bolt at the first sign of trouble. And they were connected in real time through Twitter, Slack, and the private group chats of venture firms. A handful of prominent VCs could, with a few messages, tell hundreds of portfolio companies to pull their cash. That is not a customer base; it is a single, herd-like depositor wearing thousands of name tags.

### Meanwhile, in Zurich

Across the Atlantic, a different kind of rot had been spreading for years at Credit Suisse, one of Switzerland's two flagship banks and a globally systemic institution. Credit Suisse had not made a single concentrated rate bet; its problem was a long, grinding loss of confidence. It had been battered by a string of scandals and losses: a ~\$5.5 billion hit from the 2021 collapse of the Archegos family office (a leveraged equity blowup; see [the Archegos 2021 total-return-swaps blowup](/blog/trading/finance/archegos-2021-total-return-swaps-blowup)), the implosion of the Greensill supply-chain finance funds it had marketed to clients, spying scandals, leadership churn, and persistent losses. Customers had already been quietly pulling money for months. Credit Suisse was a tinder-dry forest waiting for a spark. SVB's collapse was the spark.

## The blow-up, step by step

Here is the chronology, hour by hour where it matters.

### Wednesday, March 8: the disclosure

SVB had a cash problem. Deposits had begun shrinking through 2022 as the venture boom cooled and startups burned cash without raising fresh rounds. To raise liquidity, SVB sold about \$21 billion of its *available-for-sale* bond portfolio - and, because those bonds were underwater, the sale crystallized a loss of roughly \$1.8 billion. To plug the resulting hole in its capital, SVB announced, after market close on March 8, a plan to raise about \$2.25 billion in new equity.

To management this was a tidy bit of balance-sheet housekeeping: sell low-yielding bonds, take the loss, raise capital, move on. To the market it read as a flare in the night sky. A bank does not sell bonds at a \$1.8 billion loss and scramble for capital unless something is wrong. The disclosure, combined with a poorly-timed downgrade and the recent failure of crypto-focused Silvergate Bank, lit the fuse.

### Thursday, March 9: the fastest run in history

This is the day the modern era announced itself. As the news spread, prominent venture capitalists began telling their portfolio companies, privately and then publicly, to get their money out of SVB. The advice was rational from each founder's point of view - 94% of these deposits were uninsured, so why risk it? - and self-fulfilling in aggregate. Through SVB's online portal, depositors initiated withdrawal and transfer requests totaling roughly **\$42 billion in a single day**, about a quarter of the bank's entire deposit base. There were no physical lines. The run happened at the speed of a tap on a screen, amplified by the very network effects that had made SVB dominant: the same group chats that fed it deposits now drained it.

![Digital bank run feedback loop from loss news to insolvency](/imgs/blogs/svb-credit-suisse-2023-bank-runs-3.png)

By the close of business, SVB had a negative cash balance of nearly \$1 billion and could not meet the outflows. The diagram above shows why the run fed on itself: each step made the next step more rational, so instead of calming, the panic compounded.

It is worth dwelling on *why* this run was so much faster than any historical precedent, because the answer is not just "the internet." Three structural features of SVB's depositor network turned a normal scare into a stampede. First, **homogeneity**: the depositors were not a diverse cross-section of the economy but a single industry that read the same news, used the same Slack channels, and trusted the same handful of venture capitalists. When one influential VC said "move your money," it was not one voice among thousands - it was the voice the whole herd was listening to. Second, **delegated decision-making**: many startups did not decide individually; their boards and lead investors decided for dozens of companies at once, so a single board's instruction could trigger scores of simultaneous withdrawals. Third, **the absence of friction**: there was no branch to visit, no cashier's check to wait for, no daily withdrawal limit that mattered at this scale - just an online wire that cleared in minutes. Combine a herd that all listens to the same shepherd, shepherds who can move many flocks with one command, and rails with no speed limit, and you get \$42 billion gone in a day. The very network density that made SVB the indispensable bank of the startup world is what made its run uncatchable.

### Friday, March 10: seizure

There was no time for the planned capital raise; no investor would touch a bank visibly hemorrhaging deposits. On the morning of Friday, March 10, California regulators closed Silicon Valley Bank and handed it to the FDIC. It was the second-largest bank failure in US history at the time, with roughly \$209 billion in assets. The whole sequence - from the Wednesday-evening disclosure to Friday-morning seizure - had taken less than 48 hours.

### Sunday, March 12: Signature falls, the backstop arrives

The fear did not stop at SVB. Over the weekend, depositors and markets looked for the next bank that shared SVB's profile: a concentrated, heavily uninsured deposit base. Signature Bank, a New York lender with a large book of crypto-industry and uninsured commercial deposits, suffered its own run and was seized by regulators on Sunday, March 12 - the third-largest US bank failure at the time.

Facing a possible cascade across the entire regional banking system, US authorities intervened dramatically on the evening of March 12. Invoking a "systemic risk exception," the FDIC announced it would guarantee **all** deposits at SVB and Signature - including the uninsured billions above the \$250,000 limit. And the Federal Reserve launched the **Bank Term Funding Program (BTFP)**, an emergency facility that let banks borrow against their underwater bonds *at the bonds' par value rather than their depressed market value* - directly neutralizing the very HTM-loss problem that sank SVB. In effect, the authorities reached into the mechanism we dissected and disarmed it for everyone else.

### Sunday, March 19: Credit Suisse falls to UBS

The panic crossed the Atlantic. With confidence already paper-thin, Credit Suisse's clients accelerated their withdrawals during the same week - the bank reportedly lost tens of billions of francs in outflows. A clumsy statement from its largest shareholder (the Saudi National Bank, which said it would not invest more) sent the share price into freefall. Even an emergency liquidity line of roughly 50 billion francs from the Swiss National Bank failed to stop the bleeding. Over the weekend of March 18-19, Swiss authorities orchestrated a shotgun marriage: **UBS agreed to acquire Credit Suisse** for about 3 billion francs, a fraction of its book value, in a deal announced Sunday, March 19.

Buried in that deal was the detonation that shocked the bond market. As part of the rescue, the Swiss regulator FINMA wrote **roughly \$17 billion of Credit Suisse's AT1 bonds down to zero** - while Credit Suisse shareholders, who should rank *below* bondholders, were not wiped out but received UBS shares worth about 3 billion francs. Bondholders had been wiped before equity holders. We will dissect why this was legal and why it was so explosive in a moment.

## The mechanism dissected: why a safe bank died fast

Now the deep part. Four ingredients combined, and removing any one of them would have meant survival.

### Ingredient 1: duration risk, the bet nobody called risky

SVB's fatal exposure was not to borrowers defaulting; US Treasuries do not default. It was to the *direction of interest rates*. By buying long-dated bonds with deposits that could leave at any moment, SVB had made a one-way, undiversified bet that rates would stay low. When they did not, the loss was mechanical and unavoidable. The first worked example shows exactly how brutal duration can be.

#### Worked example: a \$100 bond when yields go from 1.5% to 4%

Suppose SVB buys a 10-year Treasury bond for \$100 paying a 1.5% coupon - \$1.50 a year - just like its actual portfolio. A year and a bit later, yields on comparable bonds have risen to 4%. What is the old bond now worth?

A buyer today demands a 4% yield. They can compare your bond's future cash flows - roughly \$1.50 a year for the remaining years plus \$100 at the end - against that 4% standard. Each future dollar is "discounted" more heavily because money now earns 4% elsewhere. Working through the present-value arithmetic for a bond with about nine years left:

```
Old bond cash flows:  $1.50/yr for ~9 years, then $100 back
Discounted at 4% (the new market rate):
   price falls to roughly $80
```

The bond that cost \$100 is now worth about \$80 - a **20% loss** - even though not one cent of default has occurred and the government will still pay in full at maturity. A quick sanity check using duration: a 10-year bond has a duration of roughly 8-9 years, so a 2.5-percentage-point rise in rates (1.5% to 4%) implies a price drop of about 8.5 x 2.5% ≈ 21%. The two methods agree.

The intuition: a "safe" government bond can still lose a fifth of its value overnight if rates jump - safety from default is not safety from price moves.

There is a second, slower knife in the same wound, and it is worth naming because it deepened the trap throughout 2022. As the Fed raised rates, SVB had to start paying more to *keep* deposits - depositors could otherwise move cash into money-market funds yielding 4% or more. But SVB's asset side was frozen at the old 1.5% yields, because the bonds were long-dated and fixed-rate. The bank was earning ~1.5% on its bonds while its cost of funding crept toward 4%. That is a **negative carry**: the spread between what the assets earn and what the funding costs went from comfortably positive to negative. A negative carry slowly bleeds capital every quarter even if the bank never sells a bond, because each dollar of assets earns less than the dollar of deposits funding it costs. SVB was therefore being squeezed from two directions at once: the *market value* of its bonds had collapsed (the duration loss), and the *income* from those bonds no longer covered its funding (the carry loss). The first hole was hidden by HTM accounting; the second showed up as shrinking profits and was one reason management felt compelled to act in March 2023.

### Ingredient 2: HTM accounting that hid the loss until it was fatal

Duration losses are survivable if you can sit on the bonds until they mature and repay at par. HTM accounting is built on exactly that premise - so it lets the bank carry the bonds at cost and report no loss. The problem is that the premise ("I will never be forced to sell early") is precisely what fails in a run. The accounting did not reduce the risk; it *concealed* it, so that depositors and even many investors did not see how thin SVB's true, mark-to-market capital had become. The next example makes the concealment concrete.

#### Worked example: the same \$1 billion loss, hidden or shown

Imagine two banks, each holding \$10 billion of bonds bought at par, each now worth \$9 billion in the market - a \$1 billion unrealized loss. Each bank has \$1.2 billion of reported equity.

- **Bank A classifies the bonds as available-for-sale (AFS).** It must mark them to market. The \$1 billion loss flows through to equity. Reported equity drops from \$1.2 billion to about \$0.2 billion. Anyone reading the balance sheet sees a bank teetering on the edge.
- **Bank B classifies the identical bonds as held-to-maturity (HTM).** It carries them at the \$10 billion cost. No loss appears. Reported equity stays at \$1.2 billion. The balance sheet looks healthy.

The two banks are *economically identical* - same bonds, same prices, same true net worth of about \$0.2 billion. But Bank B looks fine and Bank A looks frightening, purely because of an accounting election. SVB was Bank B, with a roughly \$15 billion hidden loss against about \$16 billion of equity.

The intuition: HTM does not make a loss disappear; it just delays the day everyone sees it - and that day arrives instantly the moment the bank is forced to sell.

There is one more cruel wrinkle in HTM accounting that sealed SVB's fate. The rules treat the HTM designation as a serious commitment: if a bank sells a meaningful chunk of its HTM bonds, regulators can force it to reclassify the *entire* HTM portfolio to mark-to-market, which would instantly drag the whole ~\$15 billion hidden loss onto the balance sheet and crater reported capital. This is sometimes called "tainting" the HTM book. So SVB faced a trap with no good exit: it could not quietly sell a few HTM bonds to raise cash without risking the reclassification of all of them. That is partly why, on March 8, it sold from its *available-for-sale* book instead and took the visible \$1.8 billion loss there - and why the bank had so little room to maneuver. The accounting category that had protected the optics in calm times became a cage the moment the bank needed liquidity.

### Ingredient 3: concentrated, uninsured deposits that fled together

A diversified bank with millions of small, insured retail depositors barely feels a scare: ordinary savers are protected up to \$250,000 and have no reason to run. SVB's depositors were the opposite on every axis, as the comparison below lays out.

![Comparison matrix of SVB versus a diversified bank on four risk axes](/imgs/blogs/svb-credit-suisse-2023-bank-runs-5.png)

#### Worked example: how much of SVB's deposits were above the \$250,000 limit

The FDIC covers \$250,000 per depositor. Consider three accounts at SVB:

- A startup with \$30,000,000 from a fresh funding round. Insured: \$250,000. Uninsured and at risk: **\$29,750,000**.
- A growth-stage company with \$5,000,000 of payroll cash. Insured: \$250,000. At risk: **\$4,750,000**.
- A founder's personal account with \$200,000. Fully insured; at risk: \$0.

Now scale up. SVB held roughly \$173 billion in deposits, of which about 94% - around \$163 billion - was uninsured. Compare a typical large US bank, where closer to half of deposits are insured. For SVB's depositors, running was not paranoia; it was arithmetic. Each uninsured dollar stood to recover only cents if the bank failed and there were no backstop, so the rational move was to be first out the door.

The intuition: deposit insurance is what stops runs, and SVB had almost none of its deposit base protected - so almost its entire funding could, and did, decide to flee at once.

### Ingredient 4: the speed of a smartphone run

The final accelerant was technology. In 1930, running on a bank meant physically traveling there, standing in line, and withdrawing cash the teller counted by hand - a process that throttled the run's speed and gave the bank time to find help. In 2023, a depositor could move \$10 million with a few taps, at 2 a.m., from a phone, while reading a VC's tweet urging them to do exactly that. Information and money both moved at internet speed, and they reinforced each other. The pipeline below shows how this turns a harmless paper loss into a fatal one.

![Pipeline showing how an unrealized loss becomes insolvency once deposits flee](/imgs/blogs/svb-credit-suisse-2023-bank-runs-6.png)

#### Worked example: \$42 billion in a day versus a slow 1930s run

Picture a 1930s bank with \$100 million in deposits facing a classic run. Depositors queue at one branch; a teller can process, say, a few hundred withdrawals a day. Even a furious run might drain a few million dollars a day - a percent or two of deposits - giving the bank days to arrange emergency cash or call in the central bank. The friction of physical lines was, in effect, a circuit breaker.

Now SVB. On March 9, depositors requested about **\$42 billion of withdrawals in one day** out of roughly \$173 billion of deposits - **about 24% of the entire bank** gone (or trying to go) in a single business day. Annualized to the pace of an old-fashioned run, that is not days or weeks of stress compressed into hours; it is a multiple of any run the regulatory framework was designed for. The 30-day liquidity coverage ratio assumed you had a month. SVB had an afternoon.

The intuition: the same digital rails that let SVB gather deposits frictionlessly let those deposits leave frictionlessly - and a run with no friction is a run with no brakes.

### The AT1 controversy: bonds wiped before equity

The second act, Credit Suisse, broke a different rule - not of accounting but of *seniority*. Recall the loss-absorption ladder: when a company fails, the most senior claimants are paid first and the most junior absorb losses first. Depositors and senior bondholders sit near the top; AT1 bonds sit just above common equity; equity is the very bottom, the first-loss layer.

![Loss absorption stack from senior deposits down to common equity](/imgs/blogs/svb-credit-suisse-2023-bank-runs-4.png)

The diagram shows the *normal* ordering, where equity is wiped out before any bond. In the Credit Suisse rescue, that order was inverted. FINMA, the Swiss regulator, wrote the roughly \$17 billion of AT1 bonds down to **zero**, while shareholders received UBS stock worth about 3 billion francs. AT1 holders got nothing; equity holders got something. How was this legal?

The answer is in the fine print AT1 investors had signed. Swiss AT1 prospectuses contained a clause allowing a full write-down at the regulator's discretion if the bank received extraordinary government support - a "viability event" - regardless of what happened to equity. The instruments were *contractually* designed to be wiped in exactly this scenario. Legally, FINMA was within its rights. But the market had broadly assumed - and most other jurisdictions' rules implied - that equity would always be hit first. The write-down detonated that assumption.

#### Worked example: a \$1,000 AT1 bond goes to zero while equity recovers a little

Suppose you hold a Credit Suisse AT1 bond with a face value of \$1,000, and a friend holds \$1,000 of Credit Suisse common stock. You both believe you are safer than your friend, because bonds rank above equity.

The rescue happens. Under FINMA's order:

- Your AT1 bond is written down 100%. Your recovery: **\$0**.
- Your friend's equity is not zeroed; shareholders collectively receive UBS shares worth about 3 billion francs. Scaled to a \$1,000 stake, your friend recovers perhaps a small fraction of their money - call it tens of dollars, not zero.

You, the supposedly more senior creditor, recovered **less** than the supposedly junior shareholder. Across the whole instrument, roughly \$17 billion of AT1 value vanished while equity retained a sliver. The day after, AT1 bonds issued by *other* European banks fell sharply as investors repriced the risk that "above equity" was not a promise.

The intuition: AT1 bonds pay a fat coupon precisely because, in a crisis, they can be wiped out - sometimes even before the shares they supposedly outrank.

The aftershock was instructive. Within a day, regulators in the eurozone and the UK rushed out statements clarifying that *in their* jurisdictions the standard hierarchy still held - equity absorbs losses before AT1 - explicitly to stop the Swiss precedent from contaminating their own banks' funding. They feared that if AT1 investors lost faith that the seniority order would be honored, banks across Europe would find it impossible or ruinously expensive to issue new AT1, choking off a key source of regulatory capital. The episode is a clean lesson in how a single national regulator's decision can reprice a half-trillion-dollar global market overnight, and how much of "seniority" in finance rests not on intuition but on the exact words of a contract. AT1 holders who had skimmed the prospectus and assumed they ranked above equity learned that the write-down clause meant exactly what it said.

## The aftermath: who paid, what changed

The immediate crisis was contained, but at a cost and with lasting consequences.

**Insured depositors lost nothing - and neither, in the end, did uninsured ones at SVB and Signature.** The FDIC's systemic-risk exception guaranteed all deposits at the two failed US banks, so startups recovered their payroll cash. This was a controversial choice: critics argued it rewarded the reckless concentration of uninsured money and created moral hazard, signaling that large depositors might always be bailed out. The cost of covering the uninsured deposits was borne by the FDIC's Deposit Insurance Fund, which is funded by levies on all banks - so the broader banking industry ultimately paid through special assessments.

**Shareholders and most bondholders were wiped.** SVB's and Signature's equity went to zero. Credit Suisse's AT1 holders lost roughly \$17 billion; its shareholders lost most of their value in the forced sale to UBS. The people who "paid" were the investors who had funded these banks, which is how it is supposed to work.

**First Republic became the next domino.** A San Francisco bank with a wealthy, heavily uninsured clientele and its own pile of low-rate mortgages and bonds, First Republic suffered slow-motion deposit flight through March and April despite a \$30 billion lifeline from a consortium of large banks. It failed on May 1, 2023, and was sold to JPMorgan Chase - the largest US bank failure since 2008 by assets, surpassing SVB.

![Contagion graph from SVB to Signature, First Republic, and Credit Suisse](/imgs/blogs/svb-credit-suisse-2023-bank-runs-7.png)

**The unrealized-loss overhang remained.** SVB was the most extreme case, but it was not unique. Across the US banking system, banks were sitting on hundreds of billions of dollars of combined unrealized losses on their bond portfolios - estimated at over \$600 billion at one point in 2022-2023. SVB was the bank where those losses met the worst possible deposit base. The episode forced supervisors and investors to look much harder at every bank's HTM book and at the *quality* of its deposits, not just the quantity.

**The rules came under review.** Regulators and politicians debated raising or removing the \$250,000 insurance cap (especially for business payroll accounts), tightening rules on interest-rate risk and HTM accounting for mid-sized banks, and accounting for the new reality that runs can be far faster than any 30-day framework assumes. The BTFP, the emergency lending facility, ran for about a year before closing in March 2024.

A specific regulatory thread is worth pulling, because it explains why SVB in particular slipped through. After the 2008 crisis, the Dodd-Frank Act subjected large US banks to strict supervision: stringent stress tests, tighter liquidity rules, and a requirement to account for unrealized bond losses in their regulatory capital. But in 2018, a bipartisan law raised the asset threshold for the strictest oversight from \$50 billion to \$250 billion. SVB, with around \$200 billion in assets, sat in the gap created by that change - large enough to be systemically important when it failed, but no longer subject to the toughest stress tests and liquidity requirements that might have flagged its interest-rate exposure and forced it to act sooner. Whether tighter supervision would have caught the duration mismatch in time is debated, but the episode reopened the question of where to draw the line between "too small to bother" and "too big to fail." The Fed's own post-mortem was unusually self-critical, faulting both the bank's management and its own supervisors for moving too slowly on risks that were, in hindsight, plainly visible in SVB's filings. For the deeper background on how deposit creation and central-bank backstops actually work, see [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) and the broader map in the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions).

## Common misconceptions

The SVB story is widely misremembered. Here are the errors that matter, each corrected with the mechanism.

### "SVB failed because it made risky loans"

This is the most common and most wrong belief, probably because it pattern-matches to 2008's subprime mortgages. SVB's assets were essentially the *safest* on Earth: US Treasury bonds and government-guaranteed mortgage securities, with near-zero chance of default. The risk was not that borrowers would not pay; it was **duration** - the certainty that long-dated bonds lose market value when rates rise. SVB died of interest-rate risk dressed as prudence, not credit risk. A bank can hold nothing but the safest paper and still fail if it holds too much of it for too long while rates climb.

### "Held-to-maturity means there is no loss"

The accounting category says no loss appears *on the headline balance sheet*, but the economic loss is real the moment market prices fall. HTM only works if the bank is never forced to sell before maturity. A deposit run is precisely the event that forces the sale, converting the hidden loss into a realized one and the accounting fiction into insolvency. Calling a loss "unrealized" describes its visibility, not its existence.

### "Insured depositors lost money"

No insured depositor at SVB lost a cent, and none ever does within the \$250,000 limit - that guarantee held perfectly. The fear and the run came entirely from the *uninsured* depositors, who held roughly 94% of SVB's deposits and faced genuine potential losses (later covered by the systemic-risk exception). The lesson is the opposite of "insurance failed": deposit insurance worked exactly as designed; SVB's problem was that almost none of its deposits were within it.

### "AT1 bonds are safer than equity"

AT1 / CoCo bonds rank above common equity in the normal pecking order, so investors assumed they would always be hit *after* shareholders. Credit Suisse proved that assumption false: the contracts permitted a full AT1 write-down at the regulator's discretion in a viability event, and that is what happened - AT1 holders were zeroed while shareholders kept a sliver. AT1 bonds carry a higher coupon precisely *because* they can be wiped out early; the yield is compensation for exactly the risk that materialized.

### "The Fed bailed out SVB"

SVB itself was not rescued - it was seized, wound down, its shareholders and bondholders wiped, its management removed. What the authorities backstopped were the *depositors* (including uninsured ones) and the *rest of the banking system*, via the deposit guarantee and the BTFP lending facility. The distinction matters: the institution was allowed to fail; its depositors and the system were protected. Conflating the two obscures who actually bore the loss.

### "This was a uniquely American or uniquely Swiss problem"

The duration trap was an industry-wide condition - hundreds of billions in unrealized bond losses sat across the system - and the AT1 question rattled bond markets in Europe and beyond. SVB and Credit Suisse were the weakest links, not isolated freaks. The same mechanism (rate-driven asset losses plus a fragile funding base) is latent in any bank that funds long, fixed-rate assets with flighty money.

## How it echoes in other markets

The SVB-Credit Suisse episode rhymes with a long history. Each of the following shares some part of the mechanism - a run, a funding mismatch, a hidden loss, or a confidence collapse - which is what makes the 2023 lesson general rather than a one-off.

### The 1929-1933 American bank runs

The archetype. In the early years of the Great Depression, thousands of US banks failed as depositors, with no insurance, rushed to withdraw cash at the first rumor of trouble. The runs were self-fulfilling: fear of failure caused the failure. The crucial difference from 2023 is speed and remedy. Those runs unfolded over days and weeks of physical lines, and there was no FDIC - it was *created in 1933, in direct response*, precisely to break the run dynamic by guaranteeing small deposits. SVB shows both the enduring logic (uninsured money runs) and the new twist (it now runs in hours, not weeks). The 1930s also gave us the policy reflex - guarantee deposits, lend freely against good collateral - that the Fed and FDIC reached for in March 2023.

### Northern Rock, 2007

The closest modern cousin, and the first British bank run in over a century. Northern Rock, a UK mortgage lender, funded itself heavily in short-term wholesale markets rather than sticky retail deposits. When those markets froze in the opening act of the global financial crisis, it could not roll over its funding - a funding mismatch identical in spirit to SVB's. The sight of queues outside Northern Rock branches in September 2007 (a *physical* run, the last of its kind) forced the UK government to guarantee its deposits and eventually nationalize it. The lesson SVB repeated: it is not the quality of your assets that kills you first, it is the fragility of your funding.

### Bear Stearns and Lehman Brothers, 2008

Bear Stearns (March 2008) and Lehman Brothers (September 2008) were investment banks, not deposit-takers, but they died of runs all the same - **wholesale** runs. They funded long-term, hard-to-value assets with overnight **repo** borrowing (short-term loans secured by securities). When counterparties lost confidence and refused to roll the repo, both firms ran out of cash within days despite being, on paper, solvent. Bear was absorbed by JPMorgan in a Fed-brokered fire sale; Lehman was allowed to fail, triggering the global crisis. The mechanism is SVB's exactly: a maturity mismatch plus a sudden refusal of short-term funding equals death by liquidity, regardless of stated asset value.

### Washington Mutual, 2008

Until SVB, "WaMu" was the largest US bank failure ever - a roughly \$300 billion thrift seized in September 2008. Unlike SVB, WaMu *did* have a credit problem (a portfolio of risky mortgages going bad), but the proximate cause of its end was a run: customers withdrew about \$16.7 billion over a few weeks as confidence eroded, and regulators seized it and sold the banking operations to JPMorgan Chase. WaMu shows the same end-state - a deposit run overwhelming a wounded bank - reached by a different (credit) road, and it set the template for the FDIC-brokered weekend sale later used for First Republic.

### Continental Illinois, 1984

The original "too big to fail." Continental Illinois, then one of America's largest banks, suffered a run in 1984 - but a *wholesale* one, by large institutional and foreign depositors pulling uninsured funds after the bank's bad energy loans came to light. There were no retail lines; the money left by wire, presaging SVB's electronic flight. The US government rescued it, explicitly guaranteeing all depositors and coining the "too big to fail" doctrine. It is the direct ancestor of the 2023 systemic-risk exception that covered SVB's uninsured depositors.

### The 2022 UK gilt / LDI crisis

A few months before SVB, in autumn 2022, British pension funds running "liability-driven investment" (LDI) strategies were caught by the same rising-rates mechanism. They held leveraged positions in UK government bonds (gilts); when a poorly-received budget sent gilt yields spiking, the bonds' prices collapsed, triggering margin calls that forced the funds to sell more gilts, pushing yields higher still - a fire-sale doom loop. The Bank of England had to step in with emergency bond-buying. It is the SVB mechanism in a different costume: long-duration "safe" government bonds, plus a forced-seller dynamic, equals a self-reinforcing collapse when rates jump.

### The savings and loan crisis, 1980s-1990s

The slow-motion version of SVB's exact trap. US savings and loan institutions ("thrifts") had funded long-term, low-rate fixed mortgages with short-term deposits. When interest rates spiked in the late 1970s and early 1980s (the Volcker era), the thrifts' funding costs soared above the income from their old low-rate mortgages, and the market value of those mortgages collapsed - a classic duration mismatch. Hundreds of institutions became insolvent over the decade, and the cleanup cost taxpayers well over \$100 billion. The S&L crisis is SVB stretched across years instead of days: same interest-rate-risk mechanism, slower fuse, bigger eventual bill.

### MF Global, 2011, and the speed of confidence

MF Global, a brokerage led by a former Goldman chief, collapsed in days in 2011 after disclosing large bets on European sovereign bonds. The disclosure triggered counterparty flight, margin calls, and a liquidity run that ended the firm within a week - and revealed a roughly \$1.6 billion shortfall in customer funds. Like SVB, MF Global shows how a single disclosure can convert latent risk into a fatal run almost overnight once confidence evaporates. The instrument differs; the confidence-collapse mechanism does not.

## When this matters to you, and further reading

You will probably never run a bank. But this episode pays off in concrete ways for anyone who keeps money somewhere or invests in financial firms.

**If you hold cash above \$250,000 anywhere, the insurance limit is not a footnote - it is the whole game.** SVB proved that uninsured deposits are genuinely at risk and that the rational response to a scare is to move first. Individuals and especially businesses learned to spread balances across multiple banks, use sweep programs that distribute cash across many insured institutions, or hold Treasuries directly. The cheapest insurance against a bank run is not having uninsured money sitting in one place.

**If you invest in banks or buy their bonds, read the duration and the deposit mix, not just the loan book.** SVB's annual report disclosed both its long-duration HTM portfolio and its overwhelmingly uninsured deposit base; the danger was visible to anyone who connected the two. And the AT1 write-down is a permanent reminder to read the prospectus: a bond's seniority is only as strong as its specific contract, and a high yield is always paying you for a specific risk.

**If you want to understand modern banking risk, internalize the two-sided fragility.** A bank can be killed by what it owns (assets falling in value) *and* by how it is funded (liabilities fleeing), and the deadly cases are when both happen at once. Watch interest-rate direction (it drives bond and asset values) and watch the stickiness of funding (insured, diversified, patient money is safe; uninsured, concentrated, connected money is a fuse).

To go deeper on the surrounding machinery, three companion pieces help. To see why rate moves are the engine behind the whole story, read [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates). To understand how deposits and bank money are actually created and backstopped, read [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier). For a map of the players - commercial banks, investment banks, the FDIC, the central bank - see the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions). And for a sibling case study in how a single concentrated, leveraged exposure can blow up fast, see [the Archegos 2021 total-return-swaps blowup](/blog/trading/finance/archegos-2021-total-return-swaps-blowup).

The enduring image is simple. SVB held the safest assets in the world and still failed in two days, because "safe from default" is not "safe from price moves," because an accounting label can hide a loss but not erase it, and because in 2023 a bank run no longer needs a line outside a branch - it only needs a group chat and a smartphone. The runs of the 1930s took weeks. The next one may take an afternoon.

*This is an educational case study, not investment advice. Figures are drawn from widely reported public records of the March 2023 events and are approximate where noted; some totals (deposit shares, unrealized losses, outflow figures) are reported with small variations across sources.*
