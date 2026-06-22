---
title: "The Credit Cycle and Why Bank Earnings Are Pro-Cyclical"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why bank profits boom late in an expansion and crater early in a downturn — how provisions, reserve builds and releases, and lagging charge-offs make bank earnings swing harder than the economy, and how that fools investors."
tags: ["banking", "credit-cycle", "provisions", "pro-cyclicality", "ifrs9", "cecl", "bank-earnings", "charge-offs", "reserve-release", "expected-credit-loss", "bank-analysis", "loan-losses"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank's reported profit swings much harder than the economy does, because the single most volatile line on its income statement — the loan-loss provision — falls almost to zero (and can even turn into a gain) when times are good, then spikes violently when the downturn arrives. That is *pro-cyclicality*, and it is the main reason bank earnings flatter management at the top and savage them at the bottom.
>
> - **Provisions are the swing factor.** A bank's core engine — net interest income plus fees minus costs — is fairly steady. The provision line is not: it can go from a small expense to a release in a boom and to many times the normal level in a bust. That one line does almost all the work on the bottom line.
> - **The worst loans are made in the best times.** Loan books grow fastest at the top of the cycle, exactly when underwriting is loosest and spreads are thinnest. Those loans default years later — so the cause (loose lending) and the effect (the loss) are separated by years, and the damage looks sudden when it is not.
> - **The forward-looking reserve (IFRS 9 / CECL) makes earnings lumpy.** Because banks must book expected future losses *today* when the outlook darkens, a single quarter's change in the economic forecast can move billions through the P&L before a single borrower has missed a payment.
> - **The one number to remember:** in 2009, US banks earned an industry return on assets of roughly **0.65%** — about half their normal **1%+** — almost entirely because the provision line exploded. By 2021 the same banks *released* reserves and reported a near-record **1.23%** ROA. Same banks, same business; the provision line told two opposite stories.

In the spring of 2021, the largest banks in America reported some of the best quarters in their history. JPMorgan's first-quarter net income roughly doubled year on year. The headlines said "blowout earnings." But if you read past the headline to the income statement, something strange was happening: the banks were not earning more from lending or from fees. They were earning more because they were *putting money back*. After hoarding tens of billions of dollars of loan-loss reserves in 2020 — bracing for a pandemic-driven wave of defaults that, thanks to enormous government support, never fully arrived — they were now *releasing* those reserves. A reserve release is not revenue. Nobody paid the bank anything. It is an accounting reversal of a prior expense. Yet it flowed straight to the bottom line and lifted reported profit to a record.

Rewind twelve years. In 2009, the same industry — many of the same banks — reported its worst year in a generation. Net income for US commercial banks collapsed, and dozens of banks went under. Again, the core engine had not changed that much: banks still took deposits, still made loans, still collected interest. What changed was the provision line. It exploded, as banks set aside enormous sums against loans they now expected to go bad. The expense of preparing for losses, booked all at once, swamped everything else.

That is the puzzle this post is about. Why do bank profits *boom late in an expansion and crater early in a downturn* — far more violently than the underlying economy moves? The diagram above is the mental model: walk the credit cycle through its four phases, watch what the provision line does in each, and you can see the whole story. The answer is that a bank's earnings are *pro-cyclical* — they amplify the cycle rather than smoothing it — and the amplifier is the loan-loss provision. Understanding this one mechanism is the difference between an investor who buys a bank at a record ROE near the top (the most dangerous moment) and one who recognizes that the record is built on a provision line that can only get worse from here.

![The credit cycle and what provisions do in each phase](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-1.png)

This connects directly to the spine of this whole series. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and lends long, earns the spread, and survives only as long as its thin equity cushion absorbs losses faster than they arrive. The credit cycle is the rhythm at which those losses arrive — slow and small for years, then fast and large all at once. Pro-cyclical earnings are simply what that rhythm looks like when it is run through an accounting system and reported to investors every quarter.

## Foundations: the credit cycle, provisions, and what "pro-cyclical" means

Before we can see why bank earnings amplify the cycle, we need a handful of terms — defined from zero, with a number attached to each.

### The credit cycle

The **credit cycle** is the recurring expansion and contraction of borrowing, lending, and loan losses in an economy. It is related to the business cycle (the rise and fall of output and employment) but it is *not the same thing*. The credit cycle has its own rhythm: credit tends to grow, get cheaper, and get easier to obtain for years — then, when something cracks, it contracts, gets dearer, and dries up, often abruptly.

Here is an everyday version. A long dry summer is followed by a sudden storm. For years the weather is benign; the reservoir fills; everyone builds closer to the river because the river has been calm for a decade. Then the storm comes, and the very people who built closest to the water are hit hardest. The calm did not prevent the flood — it *invited* the building that the flood would later punish. Credit works the same way: the long good times invite the very lending that the bad times will turn into losses.

The cycle has, roughly, four phases. We will go through them in detail later, but as a map:

1. **Recovery** — losses from the last bust are still working through; reserves are heavy; lending is cautious.
2. **Expansion** — defaults are low and falling; loan books grow; spreads are healthy; provisions shrink.
3. **Late boom** — the best borrowers are already lent to, so the *marginal* new loan goes to a weaker borrower on looser terms; provisions are tiny or even reversed; everything looks wonderful.
4. **Downturn** — the trigger arrives (a rate shock, a recession, a sector blow-up); defaults surge; reserves are rebuilt fast; provisions spike.

### A loan loss, a charge-off, and a recovery

When a borrower stops paying, a bank does not lose money instantly. The sequence is:

- The loan becomes **delinquent** (a payment is late), then **non-performing** (typically 90+ days past due — covered in depth in our piece on [non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process)).
- If the bank concludes the loan will not be repaid, it records a **charge-off** (also called a *write-off*): it removes the unrecoverable amount from its loan book. A **net charge-off (NCO)** is the gross charge-off minus any cash the bank later claws back.
- A **recovery** is that clawed-back cash — from selling collateral, from a workout settlement, or from the borrower's eventual partial repayment. Recoveries reduce the net loss.

The **net charge-off rate** is net charge-offs as a percent of the loan book. In a benign year it might be 0.3–0.5% of loans for a diversified bank. At the depth of the 2008–09 crisis, US banks' aggregate net charge-off rate climbed toward 3% of loans — six to ten times normal. That multiple, not the absolute number, is what wrecks earnings.

### The provision (PCL): the expense, not the loss

Here is the crucial distinction that trips up almost everyone new to bank accounting. The charge-off is the *realized* loss — the moment the money is gone. The **provision** is the *expense the bank books to prepare for losses*, and it does not have to happen at the same time as the charge-off.

Formally, the line on the income statement is the **provision for credit losses (PCL)** — sometimes called the loan-loss provision. It is the amount the bank charges against current-period profit to top up (or, if negative, draw down) its **allowance for credit losses (ACL)**, also called the *loan-loss reserve*. The allowance is a stockpile sitting on the balance sheet as a contra-asset (it reduces the reported value of loans); the provision is the *flow* into or out of that stockpile each period.

The mechanical link is simple and worth memorizing:

$$\text{Provision (PCL)} = \Delta\text{Allowance} + \text{Net charge-offs}$$

In words: the provision you book this period equals the *change* in your reserve stockpile plus the losses you actually wrote off. If you wrote off \$10 and you want your reserve to be \$5 higher than it was, you must provision \$15. If you wrote off \$10 but you decide your reserve can be \$8 *lower* than it was (because the outlook improved), you only provision \$2. And if you wrote off almost nothing and you cut your reserve by \$5, your provision is *negative \$5* — a **reserve release** that *adds* to profit.

This is the engine of pro-cyclicality, so sit with it. The provision is not the loss. It is the loss *plus or minus your changing opinion about future losses*. And opinions about the future are violently cyclical: optimistic at the top, terrified at the bottom.

#### Worked example: a provision spike turning profit to loss

Let us make a small bank concrete. Suppose Riverbank has a \$1,000 loan book. In a normal year it earns:

- Net interest income: \$80
- Fee income: \$20
- Operating costs: −\$55
- **Pre-provision operating profit: \$45**

This pre-provision number is the bank's *earnings engine* — what it makes before any loan losses. (We dissect it in detail in [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions).) In a normal year, Riverbank provisions \$10 against losses (1% of the book), so pre-tax profit is \$45 − \$10 = \$35.

Now a downturn hits. Riverbank's analysts now expect 6% of the book to default with a 45% loss-given-default, and the forward-looking accounting rules force the bank to recognize much of that expected loss now. The provision jumps to \$60. The engine has not changed — still \$45 of pre-provision profit. But:

$$\text{Pre-tax profit} = \$45 - \$60 = -\$15$$

The bank just swung from a \$35 profit to a \$15 loss, and *not one thing about its lending or deposit business changed*. The entire swing — \$50 — came from the provision line. That is pro-cyclicality in a single arithmetic step: a stable engine, a violently moving provision, a bottom line that lurches.

![Same bank two years how provisions swing the bottom line](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-4.png)

The figure puts the two cases side by side: the same Riverbank, the same engine, in a good year (when a reserve release nudges the provision below zero and pre-tax profit up to \$50) and in a bad year (when the provision spikes to \$60 and the same bank reports a \$15 loss). Every line above the provision is identical between the two columns. The provision line alone does the work — and it does it in opposite directions at opposite ends of the cycle. Hold that picture; everything else in this post is an elaboration of it.

### Reserve build vs reserve release

A **reserve build** is a period when the bank's provision *exceeds* its charge-offs, so the allowance grows. Builds happen when the bank gets more worried — at the start of a downturn, or pre-emptively when the outlook darkens. A **reserve release** is the opposite: provisions fall below charge-offs (or go negative), the allowance shrinks, and the difference flows to profit. Releases happen when the bank gets *less* worried — typically deep in a recovery or late in an expansion, when prior fears proved overdone.

The asymmetry of *when* these happen is the whole point. Banks build reserves into a downturn (hurting earnings exactly when the economy is weak) and release them in good times (flattering earnings exactly when the economy is strong). The accounting moves the same direction as the cycle. That is what "pro-cyclical" means.

### Pro-cyclical vs counter-cyclical

A variable is **pro-cyclical** if it moves *with* the cycle — up in booms, down in busts — and **counter-cyclical** if it moves *against* it. Most of a bank's business is mildly pro-cyclical (more lending and more fees in good times). But the provision line makes reported *earnings* strongly pro-cyclical: the line is small or negative in booms (boosting profit when things are already good) and large in busts (slashing profit when things are already bad). Reported bank earnings therefore *amplify* the cycle rather than smoothing it.

Regulators have long wished bank provisioning were *counter*-cyclical — that banks would salt away reserves in the good times to spend in the bad — and the post-2008 accounting reforms (IFRS 9 and CECL) were partly meant to push in that direction. We will see why they only partly succeeded, and in one respect made earnings *lumpier*. Some regulators went further and added an explicit *countercyclical capital buffer* — a capital surcharge that supervisors can raise in booms and release in busts, forcing banks to carry more cushion when credit is growing fastest. It is a deliberate attempt to lean against the pro-cyclicality this post describes, by making the regulatory capital requirement itself move counter to the cycle even when reported earnings do not.

### Through-the-cycle vs point-in-time

There are two ways to think about a borrower's riskiness. A **through-the-cycle (TTC)** view asks: across an entire credit cycle — good years and bad — what is this borrower's average probability of default? A **point-in-time (PIT)** view asks: given *today's* economic conditions, what is the probability right now? In a boom, the PIT default probability is below the TTC average; in a downturn, it is above. (We build out the full credit-risk engine — probability of default, loss-given-default, exposure at default — in [credit-risk management: PD, LGD, EAD and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss).)

This distinction matters enormously for provisioning. A reserve built on a *point-in-time* view will be low in booms and high in busts — pro-cyclical. A reserve built on a *through-the-cycle* view would be steadier — counter-cyclical, or at least neutral. Modern accounting (IFRS 9 / CECL) is largely point-in-time, which is precisely why it makes earnings lumpy. We will return to this.

### The "worst loans are made in the best times"

This is the deepest and least intuitive idea in the whole post, so we state it plainly here and unpack it later: **the loans a bank will most regret are the ones it makes near the top of the boom.** Not because bankers are stupid at the top, but because of selection. Early in an expansion, a bank lends to the strongest, most obviously creditworthy borrowers. As the expansion runs on, those borrowers are already fully banked, so to keep growing the loan book the bank must reach for *weaker* borrowers on *looser* terms at *thinner* spreads — and it does so just as competition is fiercest and credit standards are loosest. Those late-boom loans carry the highest embedded losses. But the losses do not show up for years. So the riskiest lending coincides with the calmest-looking results, and the bill arrives long after the party.

## The cycle's four phases, and what provisions do in each

Now we walk the cycle. Keep one question in mind at each phase: *what is the provision line doing, and why?*

![Provisions spike first then charge-offs realize later](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-2.png)

The chart shows the shape we are about to narrate, for a single representative bank, as a percent of its loan book. The amber bars are the provision (the expense); the red bars are net charge-offs (the realized loss). Notice three things. First, in the good years (the left of the chart) both are low. Second, when the downturn hits, the provision bar *jumps first and highest* — because the forward-looking reserve front-loads the expected loss. Third, the charge-off bar peaks *later* — because the actual write-offs take a year or two to work through. The provision leads; the charge-off lags. That gap is where pro-cyclicality lives.

### Phase 1 — Recovery: heavy reserves, cautious lending

Coming out of a bust, the economy is healing but scarred. Default rates are still elevated as the last of the bad loans work through the system. The bank is sitting on a *heavy* allowance built up during the downturn. Lending is cautious — credit committees remember the last cycle, underwriting is tight, and demand from good borrowers is weak anyway because the economy is fragile.

What does the provision do? It is still meaningful but falling. Charge-offs are coming down from their peak, and the bank may begin trimming its allowance as the worst-case scenarios it feared fail to materialize. Reported earnings are *recovering faster than the economy* precisely because the provision line is easing — the first taste of pro-cyclicality on the upswing.

### Phase 2 — Expansion: defaults low, books growing, provisions shrinking

This is the long, pleasant middle of the cycle. Unemployment falls, incomes rise, asset prices climb, and defaults drop to multi-year lows. Borrowers who looked shaky a few years ago are now comfortably servicing their debt. The bank's loan book grows steadily, and because new loans rarely default in their first year or two (defaults peak a couple of years into a loan's life — the *seasoning* effect), the freshly grown book looks pristine.

The provision shrinks toward its floor. With charge-offs low and the outlook improving, the bank needs only a small provision to maintain its allowance — and may even release a little. Pre-provision profit is healthy and the provision is barely a rounding error, so reported net income looks strong and stable. Return on equity climbs. Everyone — management, investors, analysts — extrapolates the calm.

### Phase 3 — Late boom: the worst loans, the smallest provisions

Now the trap closes, quietly. The expansion has run for years. The best borrowers are fully banked. To keep growing — and growth is what the market rewards, what bonuses are tied to, what justifies the share price — the bank reaches further down the credit spectrum. Loan-to-value ratios drift up. Covenants loosen. Spreads compress as banks compete for the same shrinking pool of good credits. Underwriting standards, which everyone swears they are holding, quietly slip, because the recent data say lending is safe (defaults have been low for years) and because the loan officer who refuses to match a competitor's terms simply loses the deal.

And the provision line? It is at its smallest — possibly negative. Point-in-time default probabilities are at cycle lows, so the forward-looking reserve the bank must hold is minimal. The bank may *release* reserves built earlier, adding to profit. Reported earnings hit cycle highs. ROE looks spectacular. The stock often trades at its richest multiple.

This is the single most dangerous moment in the cycle for an investor, and it *feels* like the safest. The numbers are at their best exactly when the embedded risk is at its worst.

### Phase 4 — Downturn: the spike

Then the trigger arrives — a recession, a rate shock, a sector collapse, a credit event. Borrowers who were stretched thin start to miss payments. Collateral values fall. The bank's analysts revise their economic forecasts downward, and under forward-looking accounting that revision *immediately* forces a larger expected-loss reserve — before most borrowers have actually defaulted. The provision line explodes upward.

Reported earnings collapse, often into a loss, even though the bank's deposit and lending franchise is still intact and still generating pre-provision profit. The market, which extrapolated the calm a year ago, now extrapolates the storm. The bank's stock falls hard. If the losses are large enough to eat through the thin equity cushion, the bank fails — which is why bank failures, as we will see, cluster *after* the boom, not during it.

## Provisions through the cycle: the engine of pro-cyclicality

We have seen the shape. Now let us be precise about *why* the provision line is so much more volatile than the underlying economy.

The provision is the product of two things, each of which is itself cyclical, and which move *together*:

1. **Expected loss per loan** — driven by the probability of default (PD) and the loss-given-default (LGD), both of which rise sharply in a downturn.
2. **The size and risk of the loan book** — which is largest, and contains the most marginal credits, exactly as the downturn begins (because the book grew fastest in the late boom).

Because both factors swing the same way, the provision is the *product* of two cyclical numbers, so its swing is multiplicative, not additive. A 3× rise in default probability on a book that is also 20% larger and skewed toward weaker credits can produce a 5× or larger rise in the provision. That is how a moderate recession produces an extreme earnings swing.

![When PD jumps expected loss the bank must reserve jumps too](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-5.png)

The chart makes the first factor concrete using the bank's own credit-risk parameters. Expected loss on a single exposure is the product

$$\text{Expected loss} = \text{PD} \times \text{LGD} \times \text{EAD}$$

where PD is the probability of default, LGD is the fraction the bank loses if default happens, and EAD is the exposure at default — the amount on the line. The blue bars use *through-the-cycle* probabilities of default by borrower rating; the red bars stress those probabilities for a downturn. Watch what happens as the rating worsens: for an investment-grade BBB borrower, the through-the-cycle one-year default probability is about 0.24%, but a downturn can triple it. Run that through the formula on a \$100 exposure at a 45% loss-given-default and the expected loss the bank must reserve roughly *triples* too. The riskier the borrower, the bigger the absolute jump.

#### Worked example: expected loss rising as PD jumps in a downturn

Take a single \$100 corporate loan to a BBB-rated borrower, with a 45% loss-given-default. Through the cycle, the one-year probability of default is about 0.24%. The expected loss the bank reserves is:

$$\text{EL} = 0.24\% \times 45\% \times \$100 = \$0.108$$

About eleven cents on a hundred-dollar loan. Trivial.

Now a recession hits, and the point-in-time default probability for that same borrower triples to 0.72%. Nothing about the loan changed — same borrower, same collateral, same \$100. But the expected loss the bank must now carry is:

$$\text{EL} = 0.72\% \times 45\% \times \$100 = \$0.324$$

About thirty-two cents — three times as much. Multiply that across a book of millions of loans, many of them on weaker BB and B credits where the multiple is even larger in absolute terms, and the bank's required reserve — and therefore its provision — balloons. The borrower has not yet missed a single payment. The *outlook* did all the work. This is why the provision can spike before the charge-offs arrive, and it is the heart of forward-looking provisioning, which we turn to next.

The second factor — the *size and composition* of the book — is subtler but just as powerful. Because the book grew fastest in the late boom and is now stuffed with the marginal credits made on loose terms, the same downturn bites a larger and lower-quality book. The bank is most exposed exactly when the environment is worst. The two factors compound, and the provision swings far more than the economy.

## IFRS 9 and CECL: why the forward-looking reserve makes earnings lumpy

To understand why modern bank earnings are even *lumpier* than they used to be, you need to know what changed in loan-loss accounting after 2008 — and why the cure for one problem created a side effect.

### The old world: incurred loss

Before the post-crisis reforms, banks used an **incurred-loss** model. Put simply, a bank could only reserve for a loss once there was *objective evidence* that the loss had probably already happened — a missed payment, a covenant breach, a downgrade. You could not reserve for losses you merely *expected* in the future. The system was widely blamed for **"too little, too late"** provisioning in 2008: banks entered the crisis with thin reserves because, right up until borrowers actually defaulted, there was no incurred event to reserve against. The reserves arrived only after the losses did, deepening the pro-cyclical lurch.

### The new world: expected credit loss

The reforms replaced incurred loss with an **expected-credit-loss (ECL)** model — *forward-looking* by design. The two regimes are:

- **IFRS 9** (international accounting, in force since 2018), which uses a *three-stage* model. (We cover the staging mechanics in full in [collateral, security, and loan-loss provisioning: IFRS 9 and CECL](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl).) Stage 1 loans (performing, no significant deterioration) carry a *12-month* expected loss. Stage 2 loans (significant increase in credit risk since origination, but not yet impaired) jump to a *lifetime* expected loss. Stage 3 loans (credit-impaired) carry lifetime loss too, with interest recognized on the net amount. The big lurch is the *Stage 1 → Stage 2 migration*: the moment a loan is judged to have significantly deteriorated, its reserve jumps from covering one year of expected loss to covering the *whole remaining life* of the loan — a cliff.
- **CECL** (Current Expected Credit Loss, the US GAAP standard, in force since 2020), which is even more front-loaded: from the day a loan is originated, the bank must reserve for *lifetime* expected losses on the entire book. There is no 12-month versus lifetime staging — it is lifetime from day one.

### Why forward-looking provisioning makes earnings lumpy

Here is the side effect. Both regimes require the bank to translate an *economic forecast* into a reserve. The bank runs scenarios — a base case, an upside, a downside — weights them, and books the probability-weighted expected loss. That means a change in the *forecast itself* moves the reserve, and therefore the provision, and therefore reported earnings — *before any borrower behavior changes at all*.

A single quarter in which the bank's economists mark down GDP growth, mark up the unemployment forecast, or assign more weight to the downside scenario can send billions through the provision line. Earnings become a partial function of a forecast — and forecasts are revised in jumps, not smoothly. When the outlook turns, the whole expected lifetime loss is recognized at once (a build); when it improves, it is reversed at once (a release). Instead of losses dripping in as they are incurred, they arrive in lumps as the *forecast* changes. The reserve is more adequate — which is good — but the *timing* is front-loaded and discontinuous, which makes the P&L jumpier.

CECL's "day-one lifetime loss" requirement adds a particular sting: rapid loan growth *itself* drives a provision, because every new loan must immediately carry a lifetime reserve. A bank that grows its book fast — which, recall, happens most in the late boom — books provisions on loans that have not lost a cent and may never. This dampens the late-boom flattering somewhat (growth now costs an upfront provision) but front-loads the expense onto the growth quarters, adding yet another source of lumpiness.

![Why provisions are pro-cyclical two forces push the same way](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-8.png)

The diagram ties the two forces together. When the economic outlook turns down, *two* mechanisms push the provision the same direction at once: the forward-looking reserve books the expected future loss *now* (the left path), and the lagging, clustered nature of actual charge-offs means the realized losses pile up *later* (the right path). Both feed the provision spike, and the spike makes reported earnings swing more than the economy that caused it. Neither force alone would be so violent; together they make bank earnings amplify the cycle.

#### Worked example: a reserve release flattering EPS

Now run the mechanism in reverse — the 2021 story in miniature. Suppose in 2020 our Riverbank, fearing a pandemic wave of defaults, built its allowance from \$10 to \$70 by booking a \$60 provision (charge-offs that year were near zero, so almost the entire \$60 was a build). That \$60 expense crushed 2020 earnings.

In 2021, the feared defaults never materialize — government support kept borrowers current. Charge-offs come in at just \$2. The bank's economists now judge \$70 of reserve to be far too much; \$30 is plenty. So the change in allowance is −\$40 (from \$70 down to \$30). Using our identity:

$$\text{Provision} = \Delta\text{Allowance} + \text{Net charge-offs} = (-\$40) + \$2 = -\$38$$

The provision is *negative \$38* — a release. Instead of subtracting from profit, it *adds* \$38. Riverbank's pre-provision profit of \$45 becomes a reported pre-tax profit of \$45 + \$38 = \$83 — nearly double the engine's output. Earnings per share looks like it nearly doubled. But the bank did not earn \$38 from anyone; it merely admitted its earlier fear was overdone and reversed the expense. The intuition: a reserve release is *yesterday's pessimism flowing back through today's P&L* — it flatters EPS without a dollar of new economic value created.

## Loan growth peaks at the top: the worst loans in the best times

We previewed this idea in Foundations; now we go deep, because it is the part most investors miss and the part that most reliably destroys bank capital.

### Why the book grows fastest at the worst moment

Loan growth is not constant through the cycle. It is itself pro-cyclical, and it peaks late. Several forces push the same way:

- **Demand.** Borrowers want to borrow most when the economy feels strongest — to expand, to acquire, to buy property at rising prices. Credit demand peaks late in the expansion.
- **Supply / competition.** Banks compete hardest for loans when capital is plentiful and recent loss experience is benign. With defaults low for years, models say lending is safe, capital is cheap, and every bank wants to grow. They bid against each other, and the way you win a loan is to offer a lower spread, a higher loan-to-value, or a looser covenant.
- **Incentives.** Loan officers, divisions, and CEOs are rewarded for growth and for this year's reported earnings. The cost of a loan that defaults in three years lands on someone else's watch. The asymmetry — bonus now, loss later — pushes the whole system toward more lending at the top.

The result is that the marginal loan made in the late boom is the worst loan in the book: weakest borrower, thinnest spread, loosest terms, highest loan-to-value, made just as the cycle turns. The strong loans were made early, when standards were tight and the economy was uncertain. The famous banker's saying captures it: **"The worst loans are made in the best of times."** It is not a paradox; it is selection plus incentives.

### Underwriting drift: how standards slip without anyone deciding to slip them

It is worth being precise about *how* underwriting loosens, because no credit committee ever votes to "make bad loans." The drift is subtle and almost always defensible in the moment. A few mechanisms do the work:

- **The data say lending is safe.** A bank's models are calibrated on recent experience. After five or six years of low defaults, the models report that a given borrower is safe — because the recent past, which is all the model has seen, *was* safe. The model is not wrong about the past; it is blind to the turn. So a borrower who would have been declined at the start of the cycle now clears the score, and the loan officer is simply following the model.
- **Competitive matching.** A relationship banker who insists on a 60% loan-to-value and a tight covenant loses the deal to a rival offering 75% and a covenant-lite structure. Lose enough deals and the banker's book shrinks, their bonus shrinks, and eventually their job is at risk. The rational individual response — match the market — aggregates into a system that loosens together.
- **Reaching for yield as spreads compress.** As competition narrows spreads on the best credits, the only way to hit a target return on a loan is to lend to a riskier borrower who will pay a higher rate. The bank tells itself it is being *compensated* for the extra risk by the wider spread — but spreads late in a cycle systematically *under*-price the risk, because the risk is the cycle turning, which the spread does not see.

The tragedy is that each step is locally sensible and the whole is collectively dangerous. By the time anyone can prove standards have slipped, the loans are already made and sitting in the book, waiting for the cycle to reveal them. This is precisely why a regulator or an analyst should distrust a bank that is growing its book fastest at the moment standards are loosest — the growth itself is the evidence.

### Net charge-offs lag — by a lot

Compounding the problem, the losses from those late-boom loans do not appear immediately. A freshly made loan rarely defaults in its first year; defaults peak two to four years into a loan's life as the borrower's circumstances change and the cycle turns. So the worst loans — made at the top — *charge off well into the downturn*, often years after origination.

This **lag** is why the cause and effect look disconnected. The loose lending happens in, say, year five of an expansion, when results look fantastic. The charge-offs from those loans peak in, say, year eight, deep in the recession. By the time the losses show up, the loans that caused them are years old and the bonuses they generated are long since paid. An investor watching only the current quarter sees a sudden disaster; in truth it was baked in years earlier.

![From boom lending to a profit crater the lag that fools investors](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-6.png)

The pipeline traces the chain end to end: late-boom lending on loose terms, borrowers who stretch themselves thin, the downturn that hits their income and collateral, the loans going bad, the provision spike, and finally the profit crater. Read left to right it is a story that unfolds over years — but the market experiences only the last box, the crater, as if it appeared from nowhere.

#### Worked example: normalized vs reported earnings

This is where a careful analyst earns their keep. Suppose Greenfield Bank reports the following provisions (as a percent of its \$1,000 loan book) over a cycle, against a steady pre-provision profit of \$45 every year:

| Year | Phase | Provision | Reported pre-tax profit |
|---|---|---|---|
| Year 1 | Expansion | \$8 | \$37 |
| Year 2 | Late boom | \$2 | \$43 |
| Year 3 | Late boom | −\$5 (release) | \$50 |
| Year 4 | Downturn | \$60 | −\$15 |
| Year 5 | Recovery | \$25 | \$20 |

Reported profit swings from a record \$50 to a \$15 loss to a recovering \$20. An investor extrapolating year 3's \$50 would pay a rich multiple at exactly the wrong moment.

A *normalized* (through-the-cycle) view smooths the provision to its cycle average. Sum the provisions: \$8 + \$2 − \$5 + \$60 + \$25 = \$90 over five years, or **\$18 per year** on average. Normalized earnings are therefore \$45 − \$18 = **\$27 every year**. That \$27 is the bank's *true earning power* once you average the credit cost across the cycle. It is far below year 3's \$50 and far above year 4's −\$15.

The intuition: reported earnings are the engine *minus a wildly cyclical provision*; normalized earnings are the engine *minus the average provision*. The single most useful thing a bank analyst does is mentally replace this quarter's provision with a through-the-cycle average — because the market keeps pricing the reported number as if it were permanent.

## Why pro-cyclical earnings fool investors

Put the pieces together and you can see exactly how bank earnings mislead — not through fraud, but through the structure of the numbers.

![Reported earnings swing normalized earnings reveal the engine](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-7.png)

The chart shows the trap in one picture. The red line is *reported* earnings per share through a cycle: it spikes in the good years (the reserve-release year is the peak) and collapses in the downturn. The blue dashed line is *normalized* earnings — the engine minus an average through-the-cycle provision. The blue line barely moves. The gap between them is the provision distortion. Investors who anchor on the red line buy at its peaks and sell at its troughs; investors who track the blue line see a far steadier — and far more valuable to know — picture of what the bank actually earns.

The mechanisms by which this fools people:

- **The peak looks like the safest moment.** At the top of the cycle, ROE is highest, the provision is lowest, and the chart of past earnings looks like a smooth ascent. Every signal says "high quality, low risk." But the low provision *is* the risk — it is masking the embedded losses in a freshly grown, loosely underwritten book.
- **Multiples expand at the wrong time.** Because investors pay a price-to-earnings or price-to-book multiple on *reported* earnings, and reported earnings peak late in the cycle, the *valuation* peaks late too — investors pay the highest price for the highest (and least sustainable) earnings. We dig into bank valuation, and how through-the-cycle ROE should drive the multiple, in [valuing a bank: price-to-book, ROE and the warranted multiple](/blog/trading/banking/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple).
- **Reserve releases get mistaken for operating strength.** When a bank reports a big earnings beat driven by a release, the headline says "profit jumps." But a release is non-recurring and, worse, *counter-signaling*: it usually means the bank has decided the environment is benign — often near the top, just before it isn't.
- **The lag hides the cause.** Because charge-offs lag origination by years, the quarter the losses hit looks like a bolt from the blue. The market punishes the bank for a "surprise" that was years in the making.

The disciplined response is to separate the *engine* (pre-provision profit, deposit franchise, fee streams) from the *provision distortion*, normalize the provision to a through-the-cycle average, and ask: at the price on offer, what through-the-cycle ROE am I paying for? That is how you avoid buying the peak.

### Why management often believes its own numbers

It would be comforting to think that bank executives *know* the peak earnings are illusory and simply hope investors won't notice. The truth is more unsettling: management is often as fooled as the market, because the same forces that loosen underwriting also distort judgment. The recent data say losses are low. The models — built on that data — say the book is sound. Compensation is tied to this year's reported earnings and this year's loan growth, so there is little incentive to look harder. And the human tendency to extrapolate the recent past is as strong in a CEO as in a retail investor. The result is that the late-cycle bank genuinely believes it is doing well, lends aggressively in good faith, and is as surprised as anyone when the provision spikes. This matters for how you read a bank: do not assume management's confidence at the top is informed. The confidence is a *symptom* of the cycle, not a signal that this bank is different. The banks that survive the turn well are usually the ones whose management was *uncomfortable* at the peak — that grew slowest, held the line on terms, and looked boring in the good years.

### The role of dividends and buybacks

Pro-cyclical earnings have a second-order danger: banks tend to return the most capital to shareholders exactly when earnings are highest — at the top of the cycle. Dividends are raised and buybacks accelerated when reported ROE is at its peak, which is precisely when the embedded losses are largest and the capital will soon be needed to absorb them. Then the downturn hits, the provision spikes, capital is suddenly scarce, and the bank either slashes the dividend (signalling distress and often triggering the very deposit flight it feared) or, worse, must raise equity at a depressed price, diluting the shareholders it was enriching a year earlier. The disciplined bank does the opposite — it builds capital in the good years against the losses it knows the cycle will eventually deliver — but the pressure of pro-cyclical reported earnings pushes management the wrong way. Watch what a bank does with its capital at the peak: returning all of it is a sign management is reading the reported number, not the cycle.

## Common misconceptions

**"A reserve release is real profit — the bank made more money."** No. A release is the reversal of a previously booked expense. Nobody paid the bank; it simply decided its earlier provision was too large and added the reversal back to profit. It flatters reported earnings and EPS but creates no new economic value. In our worked example, a \$38 release nearly doubled reported profit while the actual business earned the same \$45 of pre-provision profit it always did. Treat releases as non-recurring and strip them out to see the engine.

**"Provisions and charge-offs are the same thing — they move together."** They are different and they move with a lag. The charge-off is the realized loss (the money is gone); the provision is the expense the bank books to maintain its reserve, which depends on its *forecast*. Under forward-looking accounting the provision spikes *first* (when the outlook turns) and the charge-off peaks *later* (when borrowers actually default, one to three years on). Confusing the two makes the timing of the cycle impossible to read.

**"Record bank earnings mean a safe, high-quality bank."** Usually the opposite, late in a cycle. Record earnings near the top are typically built on a provision line scraping its floor — possibly negative from releases — on a loan book that grew fast and loose. The record is a sign the provision can only go one way from here: up. The 2021 records preceded the 2023 regional-bank stress; the mid-2000s records preceded 2008. High reported ROE at a cycle peak is a yellow flag, not a green one.

**"The new IFRS 9 / CECL rules fixed pro-cyclicality."** They improved reserve *adequacy* — banks now carry bigger, earlier reserves than the old "incurred loss" model allowed — but they made the *timing* lumpier, not smoother. Because provisions now move with the bank's economic *forecast*, a single quarter's forecast revision can swing billions before any borrower misses a payment. The reserve is more honest; the earnings are jumpier. That is why regulators had to relax CECL's day-one impact during COVID — the rules threatened to amplify the shock.

**"Loan growth is always a sign of a healthy, well-run bank."** Fast loan growth, especially late in an expansion and especially faster than peers, is one of the most reliable predictors of future losses. It usually means the bank is winning deals by underpricing risk or loosening terms — taking on the marginal credits its competitors passed on. The strongest predictor of a bank's losses in the next downturn is often how aggressively it grew in the last boom. (This is one of the red flags we catalogue in [spotting the next bank failure: the early warning signs](/blog/trading/banking/spotting-the-next-bank-failure-the-early-warning-signs).)

## How it shows up in real banks

Theory is one thing; here is the cycle running through real balance sheets and real dates.

![Bank failures cluster after the boom not during it](/imgs/blogs/the-credit-cycle-and-why-bank-earnings-are-pro-cyclical-3.png)

The chart of FDIC-insured bank failures per year is the systemic fingerprint of the credit cycle. Failures are essentially *zero* during the boom years — 0 in 2005 and 2006, just 3 in 2007 — and then explode *after* the turn: 25 in 2008, 140 in 2009, a peak of 157 in 2010, 92 in 2011. The losses from loans made in the 2004–2007 boom did not show up until 2009–2011. Then failures fall back to near zero through the next expansion (0 in 2021 and 2022) before a small cluster in 2023. The pattern is unmistakable: banks fail *after* the boom, when the lagging losses finally arrive and eat through capital. The calmest-looking years are the ones that plant the failures.

### 2009: the provision spike that halved industry profitability

In 2009, US commercial banks' aggregate return on assets fell to about **0.65%** — roughly half the ~1.3% they earn in a good year and far below the 1%+ rule of thumb for a healthy bank. The collapse was overwhelmingly a provision story. Pre-provision profitability held up reasonably well — banks were still earning net interest income and fees — but the provision line exploded as banks reserved against the mortgage, construction, and consumer losses from the 2004–2007 boom. Net charge-offs across the industry climbed toward 3% of loans, several times their normal level. The earnings collapse lagged the lending excess by years: the loans were made in the boom; the provisions and charge-offs landed in 2008–2010. Many banks reported outright losses for the year even though their core franchises were intact — the textbook pro-cyclical lurch, and for the weakest banks, the losses ate through the thin equity cushion and they joined the failure count above. (For the systemic mechanics of how those losses froze the whole system, see [the SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs), which trace the same fragility a cycle later.)

### The early 1990s: commercial real estate and the same lag

The 2008 episode was not new. In the late 1980s, US banks lent aggressively into a commercial-real-estate boom — office towers, shopping centres, condominiums — on the comfortable logic that property values only rose. Loan growth was strong, defaults were low, and reported earnings looked healthy right through 1988 and 1989. Then the cycle turned: overbuilding met a 1990–91 recession, property values fell, and the loans made in the boom began to default. Provisions spiked, charge-offs followed with the usual lag, and a wave of bank failures crested in the early 1990s — the same shape you can read in the failure chart above, one cycle earlier. The lesson repeats verbatim: the losses of 1991 were underwritten in 1988, the worst loans were made when the market looked best, and the reported earnings gave no warning because the provision line was still small while the lending was at its loosest. Every credit cycle since has rhymed.

### 2020: the COVID build that never had to be spent

In the first half of 2020, as the pandemic shut down the economy, banks did exactly what forward-looking accounting demands: they marked down their economic forecasts and built enormous reserves against an expected wave of defaults. JPMorgan, Bank of America, Citigroup, Wells Fargo and the rest collectively reserved tens of billions of dollars in a single quarter — JPMorgan alone took a credit-loss provision of over \$10 billion in the second quarter of 2020, a number driven almost entirely by the *forecast*, not by actual defaults. Industry ROA fell to about **0.72%** for the year. This was CECL's debut in a crisis: the day-one-lifetime-loss model front-loaded the expected pain into the build, crushing reported earnings in 2020 even though, at that point, very few borrowers had actually missed payments. Regulators were so worried this would amplify the shock that they phased in CECL's regulatory-capital impact to soften the blow.

### 2021: the release that flattered a record

Then the feared defaults never arrived. Unprecedented fiscal support — stimulus checks, expanded unemployment benefits, the Paycheck Protection Program, loan forbearance — kept households and businesses current. Charge-offs stayed remarkably low. With the 2020 reserves now looking far too large, banks *released* them through 2021, and the releases flowed straight to profit. Industry ROA jumped to about **1.23%**, near a record, and the big banks reported blowout quarters. But the quality of those earnings was poor: a large slice was reserve release, not operating income. An investor who saw "record profit" and extrapolated it badly misread the bank. The engine had not improved; the provision line had simply swung from a huge build to a release in eighteen months — pro-cyclicality in fast-forward, courtesy of the forward-looking reserve reacting to a forecast that proved wrong in both directions.

#### Worked example: reading the 2020-2021 swing as one analyst

Tie the COVID episode to our identity. Take a bank with steady pre-provision profit of \$45 (in billions, scaled). In 2020 it built reserves with a \$60 provision (charge-offs near zero), reporting pre-tax profit of \$45 − \$60 = −\$15 — a loss. In 2021 it released, with a provision of −\$30 (charge-offs of \$2, allowance cut by \$32), reporting \$45 + \$30 = \$75 — a record. Average the two years' provisions: (\$60 + −\$30) / 2 = \$15 per year. Normalized profit is \$45 − \$15 = **\$30** in *both* years. The intuition: the −\$15 loss and the \$75 record are *the same bank earning \$30*, with the provision line borrowing from one year to flatter the other. An analyst who normalized would have ignored both the 2020 panic and the 2021 euphoria — and been right.

### The next downturn: where the lag is hiding now

The mechanism does not retire. Wherever loan books have grown fastest and on the loosest terms in the recent expansion — commercial real estate (especially offices), leveraged corporate loans, certain consumer-credit pockets, private-credit-adjacent exposures — is where the lagging losses are quietly accumulating. The provisions against them will look small right up until the outlook turns, at which point the forward-looking reserve will force a build, the charge-offs will follow over the next year or two, and the banks most exposed will report the sharpest earnings declines. The cycle's signature — small provisions and record earnings now, a spike and a crater later — is being written in real time. The analyst's job is to look at *where the book grew* and *how loosely*, not at this quarter's reassuringly low provision.

## The takeaway: how to use this

The single most useful idea in this post is also the simplest: **a bank's reported earnings are its earning engine minus a provision line that lies to you about the cycle.** The engine — net interest income, fees, costs — is fairly steady and tells you what the bank can actually do. The provision is the part that swings, and it swings *with* the cycle: smallest (even negative) at the top, largest at the bottom. So reported earnings flatter the bank exactly when it is most dangerous to own and savage it exactly when the worst may be passing.

If you take three working rules from this:

1. **Normalize the provision.** Before you value a bank or judge management, mentally replace this quarter's provision with a through-the-cycle average loss rate for its book. Reported ROE at a cycle peak — built on a floored or negative provision — is not the ROE you will earn through the next downturn. The normalized number is what to capitalize.
2. **Treat record earnings late in a cycle as a warning, not a reassurance.** When the provision is at its smallest and ROE at its highest, the embedded risk is at its largest. The low provision *is* the risk. Reserve releases especially should be stripped out as non-recurring — they are yesterday's pessimism flowing back, and they tend to appear just before the turn.
3. **Watch where the book grew, and how loosely.** The losses of the next downturn are being underwritten right now, in the late-boom loans no one will charge off for years. Fast growth, thin spreads, and loosening terms relative to peers are the tells. The bill is deferred, not avoided.

Step back to the spine of this series. A bank is a leveraged, confidence-funded maturity-transformation machine whose thin equity cushion must absorb losses faster than they arrive. The credit cycle governs the *rhythm* at which those losses arrive — and pro-cyclical accounting governs how that rhythm is *reported*. The danger is not that losses arrive; it is that they arrive in a cluster, years after the lending that caused them, on a book grown fat in the calm — and that the reported numbers, right up until the moment of failure, say everything is fine. The investor and the regulator who internalize that the best-looking quarter is often the most fragile have learned the one thing the income statement will never tell them outright.

## Further reading & cross-links

- [Collateral, security, and loan-loss provisioning: IFRS 9 and CECL](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) — the staging mechanics behind the provision line dissected here.
- [Credit-risk management: PD, LGD, EAD and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss) — the engine that computes the expected loss that becomes the reserve.
- [Non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process) — what happens between delinquency and charge-off.
- [The income statement of a bank: net interest income, fees and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) — where the provision line sits and how pre-provision profit is built.
- [Valuing a bank: price-to-book, ROE and the warranted multiple](/blog/trading/banking/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple) — why you should capitalize through-the-cycle ROE, not the peak.
- [Spotting the next bank failure: the early warning signs](/blog/trading/banking/spotting-the-next-bank-failure-the-early-warning-signs) — the red-flag dashboard, including the rapid-loan-growth tell.
- [The SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the same fragility a cycle later, where lagging losses met a funding run.
