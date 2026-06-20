---
title: "Credit Analysis: The Five Cs and How a Loan Actually Gets Approved"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank decides whether to lend, from the Five Cs to credit scores, DSCR, and LTV, to the credit memo and the committee that votes yes or no."
tags: ["banking", "credit-analysis", "five-cs-of-credit", "credit-score", "dscr", "loan-to-value", "underwriting", "credit-committee", "probability-of-default", "lending"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A loan decision is one question asked five ways: *will* the borrower pay (character), *can* they pay from their cash flow (capacity), what do they bring to the table (capital), what backs the loan if it goes wrong (collateral), and is the world around them about to turn (conditions). Modern banks turn those questions into three numbers and a vote.
>
> - **Character and capacity decide whether a loan repays; capital and collateral decide what happens if it does not; conditions place both inside the cycle.** Capacity is the one most loans actually fail on.
> - The three numbers that do the heavy lifting: a **credit score** (a FICO of 740+ is prime, and roughly maps to a low probability of default), a **debt-service-coverage ratio** (lenders want **1.25×** — \$1.25 of cash flow for every \$1 of debt payment), and a **loan-to-value ratio** (a 20% down payment is an **80% LTV**).
> - Default odds climb roughly **5× per rating notch**: about 0.24% a year at BBB, 1.2% at BB, 5.5% at single-B. A lower credit band means a higher probability of default, which is *why* it means a higher rate.
> - Collateral does not stop a default; it cuts the *loss*. A secured loan loses about **25 cents on the dollar** when it defaults; an unsecured one loses about **55 cents**. That gap is the whole reason banks ask for security.
> - **The one number to remember: 1.25×.** A debt-service-coverage ratio of 1.25 is the cushion that lets a borrower keep paying when income drops 20% in a downturn. A 1.05× borrower does not survive that drop. The cushion is the point.

Picture this. A friend asks to borrow \$20 until payday. You say yes without a second thought. Now picture a stranger on the street asking for the same \$20. You hesitate. What changed? Not the amount, not the interest (there is none), not even the risk in dollars. What changed is everything you *know*: whether they will pay you back, whether they *can*, what they would lose if they stiffed you, and whether they look like someone whose payday is actually coming.

That hesitation is credit analysis. A bank does the exact same calculation you just did, except it does it on a \$400,000 mortgage instead of a \$20 loan, it does it ten thousand times a day, and it has turned your gut feeling into a structured process with names, numbers, and a committee that votes. The questions are the same. The friend who always pays you back is the borrower with good *character*. The friend who is broke this month is the one with no *capacity*. The stranger who hands you their watch as security has offered *collateral*. And the friend who just lost their job is a borrower whose *conditions* have changed.

In March 2023, a bank in California called Silicon Valley Bank failed in 36 hours, and the post-mortem was about interest-rate risk and a digital run — not credit. But here is the quieter truth the headlines skip: the overwhelming majority of banks that fail across history fail not because of a market panic but because they made bad *loans*. They lent to borrowers who could not pay, against collateral that was worth less than they thought, into a cycle that turned. Every one of those failures traces back to a credit decision that should have been a no and was a yes. This post is about how that decision is supposed to be made.

![Five Cs of credit matrix character capacity capital collateral conditions](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-1.png)

The matrix above is the mental model for the whole post: five questions, what each one measures, how a banker actually checks it, and the single number that summarizes it. Hold onto it. Everything below is just those five rows in slow motion.

This connects to the spine of how a bank lives or dies. A bank is a leveraged, confidence-funded machine: it takes in deposits (which it must repay on demand) and lends them out long (which it cannot get back on demand), and it survives only as long as its thin equity cushion absorbs loan losses faster than they arrive. Credit analysis is the bank's primary defense against the losses that eat that cushion. Underwrite well and the spread is yours to keep. Underwrite badly and the losses arrive faster than the equity can soak them up, and the machine breaks. The Five Cs are not a checklist a regulator imposed. They are the bank trying not to die.

## Foundations: the words you need before we go deep

Before we open the engine, let us define every term from zero. If you already know these, skim. If you do not, you cannot follow the rest, so read slowly.

**Credit.** When a bank gives you money now in exchange for your promise to give it back later — usually with interest — that is *credit*. A loan is one unit of credit. The word comes from the Latin *credere*, "to believe." A bank that extends credit is, literally, believing you. Credit analysis is the discipline of deciding how much belief is warranted.

**Default.** When a borrower stops paying — misses payments, breaks the terms, goes bust — the loan is in *default*. Default is the event credit analysis exists to predict and price. Note carefully: default is not the same as *loss*. A borrower can default and the bank can still recover most of its money by seizing collateral. The two are different numbers, and keeping them separate is the single most important habit in this whole subject.

**Probability of default (PD).** The odds, usually measured over one year, that a given borrower defaults. A *basis point* is one hundredth of a percent (0.01%), and PDs are often quoted in basis points: a top-quality corporate borrower might have a PD of 1 to 2 basis points (0.01% to 0.02%) a year, while a deeply troubled one might have a PD of 26% — a better-than-one-in-four chance of going bust within twelve months. PD is the number the credit score is ultimately trying to estimate.

**Loss given default (LGD).** If a borrower *does* default, what fraction of the loan does the bank ultimately fail to recover? That fraction is the *loss given default*. A well-secured loan might have an LGD of 25% — the bank recovers 75 cents on the dollar by selling the collateral. An unsecured one might have an LGD of 55%. LGD is where collateral does its work.

**Exposure at default (EAD).** How much the borrower actually owes at the moment they default. For a simple term loan it is roughly the outstanding balance. For a credit card, it is harder, because the borrower can draw down more right before they go bust.

**Expected loss.** Put the three together and you get the number a bank uses to price the loan and reserve against it: $\text{Expected loss} = \text{PD} \times \text{LGD} \times \text{EAD}$. This is the master equation of credit. PD says how *often* a loan defaults; LGD says how *badly* it hurts when it does; EAD says *how big* the wound is. We will use it relentlessly.

**Credit score.** A single number, computed by a model from your past behavior, that compresses your creditworthiness into one digit-string. In the United States the famous one is the *FICO score*, which runs from 300 to 850. Higher is safer. The score is a *proxy* for PD: it is the bank's fast, cheap, statistical estimate of how likely you are to default.

**Debt-service-coverage ratio (DSCR).** A measure of *capacity* — can the borrower's cash flow actually cover the loan payments? It is the ratio of available cash flow to the required debt payment. A DSCR of 1.25× means \$1.25 of cash flow for every \$1 of debt service: a 25% cushion. Below 1.0× means the cash flow does not even cover the payment, which is a flashing red light.

**Loan-to-value ratio (LTV).** A measure of *collateral cushion* — how much of the asset's value the loan covers. A \$80,000 loan against a \$100,000 house is an 80% LTV, which also means the borrower put 20% down. Lower LTV means more of the borrower's own money is at risk first, which protects the bank.

**The Five Cs.** The traditional framework that organizes all of this: **Character, Capacity, Capital, Collateral, Conditions.** Each is a different question; together they are the loan decision. They are old — bankers were teaching them a century ago — and they survive because they are *complete*: every modern metric slots into one of the five.

That is the vocabulary. Now the depth.

## The Five Cs, one at a time

The Five Cs are not five equal slices. They split cleanly into two jobs. **Character and capacity ask: will this loan get repaid in the normal course of things?** **Capital and collateral ask: if it does not, how much do we lose?** And **conditions wraps both inside the world** the loan lives in. A great underwriter reads them as a system, not a checklist, because they interact: a borrower with thin capacity (C2) can still be a good loan if their collateral (C4) is rock-solid, and a borrower with sterling character (C1) can still be a terrible loan if conditions (C5) are about to crush their industry.

### C1 — Character: will they pay?

Character is *willingness* to repay, as distinct from *ability*. Two borrowers can have identical incomes; one always pays on time and one treats every bill as optional. The first has good character; the second does not. In the \$20-to-a-friend example, character is the entire reason you say yes to one friend and no to another with the same wallet.

How does a bank measure something as soft as willingness? It does not interview your soul. It looks at your *track record*, because past payment behavior is the best available predictor of future payment behavior. The hard evidence of character is your **credit history**: have you paid your past debts on time, how long have you been borrowing, how many accounts have you opened recently, how much of your available credit are you using. All of that is summarized into the credit score, which is why the score sits in the character row of our cover matrix.

The deep point about character is that it is the cheapest C to verify and the hardest to fake over time. You can dress up a single year's income; you cannot retroactively invent a decade of on-time payments. That is why lenders lean so heavily on it for small, fast loans where running a full cash-flow analysis would cost more than the loan earns.

### C2 — Capacity: can they pay?

Capacity is *ability* to repay, measured from cash flow. This is the C that most loans actually fail on, and it is the one beginners under-weight. A borrower can have a flawless payment history (great character) and still be unable to service a new loan because their income simply cannot stretch to cover it. Willingness without ability is a default waiting for a bad month.

Capacity is measured with ratios that compare income to obligations. For a consumer, the key one is the **debt-to-income ratio (DTI)** — total monthly debt payments divided by gross monthly income; lenders typically want it under about 43% for a mortgage. For a business or a property, the key one is the **debt-service-coverage ratio (DSCR)** we defined above. Both are asking the same thing in different units: *is there enough money coming in to cover the money that must go out?*

The reason capacity is so central is that it is the only C that pays the loan back in the normal case. Character tells you they will try; capacity tells you whether trying is enough. Collateral and capital are backups. Capacity is the plan.

### C3 — Capital: what do they bring?

Capital is the borrower's own *skin in the game* — the money they put in alongside the bank's. On a mortgage it is the down payment. On a business loan it is the owner's equity. On a personal loan it might be net worth or savings. Capital matters for two reasons, and both are about *incentives* and *cushion*.

First, incentives: a borrower who has put 20% of their own money into a house behaves very differently from one who put nothing in. The first has \$80,000 to lose before the bank loses a cent; they will fight to keep the house. The second has nothing at stake and walks away the moment it stops being convenient. Capital aligns the borrower with the lender.

Second, cushion: the borrower's capital absorbs losses *first*. If a \$100,000 house falls to \$85,000 and the borrower put \$20,000 down, the bank's \$80,000 loan is still fully covered — the borrower's capital ate the entire \$15,000 fall. Capital is the borrower's own equity tranche, and just like a bank's equity cushions the bank, the borrower's capital cushions the loan.

### C4 — Collateral: what backs it?

Collateral is the specific asset pledged to the loan — the house behind a mortgage, the car behind an auto loan, the inventory or receivables behind a business line. If the borrower defaults, the bank can seize and sell the collateral to recover its money. This is the C that does *not* change the probability of default but dramatically changes the *loss* if default happens. Collateral lives in the LGD term of the master equation, not the PD term.

The distinction between *secured* and *unsecured* lending is the distinction between having and not having collateral. A mortgage is secured by the house; a credit card is unsecured (there is nothing to seize but the borrower's future income). That is why credit cards charge 20%-plus and mortgages charge single digits even to the same borrower: the card's loss given default is far higher because there is no asset to sell.

The subtle trap with collateral is that its value is not fixed. A house worth \$400,000 in a boom can be worth \$300,000 in a bust — and busts are precisely when borrowers default, so the collateral is worth least exactly when the bank needs it most. This correlation between default and collateral value is why prudent lenders apply a *haircut* (a discount) to collateral and demand a margin of safety in the LTV.

### C5 — Conditions: what world is this loan in?

Conditions are the external environment: the level of interest rates, the health of the borrower's industry, the stage of the economic cycle, the purpose of the loan itself. A restaurant loan looks different in a boom than in a recession. A floating-rate loan looks different when rates are 2% than when they are 7%. Conditions are the C that turns a portfolio of individually sound loans into a wave of correlated defaults, because a recession hits everyone at once.

Conditions are why a bank cannot just underwrite each loan in isolation. The same borrower, the same DSCR, the same collateral can be a yes in 2021 and a no in 2023 because the world around the loan has changed. Conditions are also where the credit *cycle* lives: PDs are not constant. In a downturn, the one-year probability of default for a typical book can rise several-fold, which is the single most important reason banks are forced to set aside reserves and why their earnings crater early in a recession.

The takeaway on the Five Cs as a system: a loan is a yes when character and capacity say it repays *and* capital and collateral say a loss would be survivable *and* conditions are not about to turn against it. A weak C can sometimes be carried by a strong one — that is the whole art — but two weak Cs at once is usually a decline.

Here is the whole framework on one page. The most useful column is the last one: which part of the expected-loss equation each C actually moves. This is the habit worth building — every time you look at a credit, ask whether the factor in front of you changes how *likely* the loan is to go bad (PD) or how *much it costs* if it does (LGD), because those are two different defenses and confusing them is the classic error.

| The C | The question | How a banker checks it | Which term it moves |
|---|---|---|---|
| Character | *Will* they repay? | Credit score, payment history, bureau file | PD (lowers default odds) |
| Capacity | *Can* they repay? | DSCR, debt-to-income, verified cash flow | PD (lowers default odds) |
| Capital | Skin in the game? | Down payment, net worth, owner equity | Both (aligns incentives and cushions loss) |
| Collateral | What backs it? | Pledged asset, appraised value, LTV | LGD (cuts the loss) |
| Conditions | What world? | Rates, industry health, the cycle | PD (the cycle raises default odds) |

Read the right-hand column and a deep truth jumps out: three of the five Cs work on the *probability* of default and only one works purely on the *loss*. That is not an accident. The cheapest way to avoid a credit loss is to not make a loan that defaults in the first place — so most of the framework is aimed at the PD term. Collateral is the backstop for when the first four Cs are wrong, which they sometimes are. A bank that leans too hard on collateral ("we're secured, so who cares about the cash flow?") is a bank that has quietly decided its own credit judgment is worthless and the asset will save it — and as 2008 showed, the asset does not save you when everyone is selling at once.

## How a credit score becomes a default probability becomes a rate

The Five Cs are the *frame*. The modern bank fills that frame with *models*, and the most visible model is the credit score. Let us see how a score becomes a number a bank can actually use to make money.

A **credit score** is a statistical model's verdict on your character and, partly, your capacity. The FICO score, the dominant one in the US, runs 300 to 850 and is built from five ingredients with roughly these weights: **payment history (~35%)** — have you paid on time; **amounts owed / utilization (~30%)** — how much of your available credit you are using; **length of credit history (~15%)**; **new credit / recent inquiries (~10%)**; and **credit mix (~10%)**. The model was trained on millions of borrowers, and it outputs a number whose only job is to *rank-order* default risk: higher scores default less.

There are really two families of these models in a bank. An **application scorecard** scores you at the moment you apply, using the data on your application plus a credit-bureau pull — it answers "should we lend to this stranger?" A **behavioral scorecard** scores you *after* you are already a customer, using your actual account behavior with the bank — it answers "should we raise this existing customer's limit, or worry about them?" FICO is essentially a generic, industry-wide application-style score; big banks build their own behavioral scorecards on top, because they can see things the bureau cannot, like the wobble in your checking-account balance.

The reason scores exist is *cost*. A full, hand-built credit analysis — pull the financials, verify the income, value the collateral, write a memo, convene a committee — might cost the bank several hundred dollars in staff time. That is fine for a \$2 million corporate loan. It is absurd for a \$3,000 credit card. So for high-volume, small-ticket consumer lending, the score *is* the underwriting: an algorithm reads the score, checks a few rules, and approves or declines in seconds. The Five Cs are still in there — the score encodes character and parts of capacity — but they have been compressed into one automated number.

![credit default odds rise about five times per rating notch bar chart](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-2.png)

The chart above is the bridge from a *rating* to a *probability*. Banks map borrowers onto an internal rating scale that mirrors the agency scale (AAA at the top, CCC at the bottom — the same scale the rating agencies use, which is worth understanding in its own right). What the bars show is the through-the-cycle one-year probability of default at each notch, and the pattern is the single most important fact in credit: **risk is not linear; it compounds.** Default odds climb roughly five-fold per notch as you descend. The numbers are roughly 0.01% at AAA, 0.02% at AA, 0.06% at A, 0.24% at BBB (the bottom of investment grade), then a jump to 1.2% at BB, 5.5% at single-B, and 26% at CCC. The chart uses a log scale precisely because a linear one would crush the top half of the alphabet into an invisible smear at the bottom.

Now watch a score band turn into a price.

![credit score band maps to default probability and priced loan rate](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-3.png)

The dual-axis chart above is the engine of risk-based pricing. The bars (left axis) are the one-year probability of default by FICO band — a stylized mapping, computed in the chart's own script, not a lender's real table. The line (right axis) is the rate the bank would charge that band. As the score falls and the PD rises, the priced rate rises with it. This is *why* a worse credit band means a worse rate: the rate is not a punishment, it is the bank pricing in a higher expected loss. Let us do the arithmetic that the chart hides.

#### Worked example: how a FICO band maps to a PD and a rate

You apply for an unsecured \$10,000 personal loan. Your FICO is 745, which lands in the 740-779 band. From the bank's stylized scorecard, that band carries a one-year probability of default of about **1.0%**. Because the loan is unsecured, its loss given default is high — say **55%**. And the exposure is the full \$10,000.

Start with the expected loss. Using the master equation:

$$\text{Expected loss} = \text{PD} \times \text{LGD} \times \text{EAD} = 0.010 \times 0.55 \times \$10{,}000 = \$55 \text{ per year.}$$

So on average, across many borrowers like you, the bank expects to lose \$55 a year on this \$10,000 loan to default. As a rate, that is \$55 / \$10,000 = **0.55%** — the loan must earn at least 0.55% a year just to break even on *credit losses* before it earns a cent of profit.

Now build the rate. The bank's rate is a stack: its **cost of funds** (what it pays depositors and the market for the money it lends — say 3.0%), its **operating cost** (servicing, collections — say 0.5%), the **expected loss** we just computed (0.55%), a **capital charge** (the cost of holding equity against the loan — say 0.5%), and a **profit margin** (say 0.5%). Add them: $3.0\% + 0.5\% + 0.55\% + 0.5\% + 0.5\% \approx 5.05\%$, round to about **5.1%**.

Now repeat for a borrower with a FICO of 600, in the 580-619 band, whose PD is about **12%**. Their expected loss is $0.12 \times 0.55 \times \$10{,}000 = \$660$ a year, or **6.6%** as a rate — twelve times yours. Stack that onto the same costs and the rate balloons past 13%. Same loan, same lender, same day — but the rate more than doubles, because the *expected loss* embedded in it is twelve times larger.

The one-sentence intuition: a credit score's whole purpose is to estimate your PD, and your PD, multiplied by the loss the bank would take, *is* the credit cost baked into your rate. A higher rate for a lower score is not a moral judgment; it is arithmetic.

## Capacity, measured: the debt-service-coverage ratio

If the score is how a bank reads character cheaply, the DSCR is how it reads *capacity* honestly. This is the most important ratio in commercial and property lending, and it deserves its own worked example because it is where loans most often live or die.

The debt-service-coverage ratio is the cash flow available to pay debt, divided by the debt payment required:

$$\text{DSCR} = \frac{\text{Net operating income (cash available for debt service)}}{\text{Total debt service (principal + interest due)}}.$$

A DSCR of exactly 1.0× means the borrower's cash flow precisely covers the loan payment with nothing to spare — every dollar in goes straight out to the loan, and one bad month means a missed payment. A DSCR of 1.25× means there is \$1.25 coming in for every \$1 going out: a 25% cushion. That 1.25× threshold is the number lenders quote most often as a minimum for income-producing loans, and the cover matrix flags it for a reason.

For consumers, the same idea wears a different name: the **debt-to-income ratio (DTI)**, total monthly debt payments divided by gross monthly income. DSCR and DTI are mirror images — DSCR asks "how many times over does income cover the debt?" (higher is safer), while DTI asks "what share of income is already promised to debt?" (lower is safer). A DSCR of 1.25× corresponds, loosely, to a borrower whose debt service eats 80% of the cash available for it (1 / 1.25 = 0.80); mortgage lenders typically cap total DTI around 43%, which is a far thicker cushion than 1.25× because a household has to eat and pay rent as well as service debt. The unifying point is that capacity is always a *ratio of obligations to income*, never a level of income alone — which is why a high earner drowning in existing debt can fail the test that a modest earner with a clean balance sheet passes easily.

#### Worked example: computing a DSCR of 1.25× and what it can survive

You want a loan against a small rental property. The property throws off **net operating income** (rent collected minus operating expenses, before the loan payment) of **\$125,000** a year. The loan you are asking for would require **\$100,000** a year in total debt service (principal plus interest). Compute the ratio:

$$\text{DSCR} = \frac{\$125{,}000}{\$100{,}000} = 1.25\times.$$

So you clear the typical 1.25× minimum exactly. Now ask the question the lender really cares about: *what can this cushion absorb?* Suppose a recession hits and your rental income falls 20% — vacancies rise, you cut rents to keep tenants. Your net operating income drops to \$125,000 × 0.80 = **\$100,000**. Your debt service is still \$100,000. Your new DSCR is:

$$\frac{\$100{,}000}{\$100{,}000} = 1.00\times.$$

You are now at exactly break-even: covering the payment, but with zero margin. One more bad quarter and you miss. The 25% cushion was *exactly enough* to absorb a 20% income shock and leave you standing — barely. That is not a coincidence. A 1.25× DSCR is, roughly, the cushion that lets a borrower survive a 20% income drop, which is about what a normal recession does to many cash flows. The lender chose 1.25× because it is calibrated to the cycle.

The one-sentence intuition: DSCR is not a vanity ratio — it is the answer to "how much can your income fall before you stop paying me?", and 1.25× is the lender's bet that you can take a recession-sized hit and still make the payment.

![default odds fall as the debt-service-coverage cushion grows](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-5.png)

The curve above makes the point visually. As DSCR climbs from 1.0× toward 2.0×, the borrower's default odds collapse, because each extra increment of coverage is another bad month they can absorb without missing a payment. Below 1.0× — in the shaded red zone — the cash flow does not even cover the payment, and default is not a risk but a near-certainty without outside help. The shape is steep on the left and flat on the right: the difference between 1.0× and 1.3× is enormous; the difference between 1.7× and 2.0× barely matters. This is exactly why lenders cluster their minimums around 1.20× to 1.30× — that is the elbow of the curve, the cheapest place to buy a lot of safety.

To see *why* the cushion is the whole game, compare two borrowers who look identical in good times.

![DSCR cushion comparison 1.5x borrower survives downturn 1.05x defaults](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-6.png)

The before-and-after above tells the story in two columns. On the left, in good times, both borrowers cover their \$100,000 debt service — Borrower A with \$150,000 of income (DSCR 1.50×) and Borrower B with \$105,000 (DSCR 1.05×). To a careless underwriter, both pass; both have a DSCR above 1.0×. On the right, the downturn arrives and both incomes fall 20%. Borrower A drops to \$120,000 of income, a DSCR of 1.20× — tighter, but still paying. Borrower B drops to \$84,000, a DSCR of 0.84× — *below 1.0×*, which means the cash flow no longer covers the payment, and Borrower B defaults. Same shock, opposite outcome. The only difference was the cushion they walked in with. This is the single most important reason a lender does not just ask "is the DSCR above 1.0× today?" but "how far above, and what would a recession do to it?"

## Collateral, measured: loan-to-value and loss given default

Capacity decides whether the loan repays. Collateral decides what happens if it does not. The metric for collateral cushion is the loan-to-value ratio, and the place collateral pays off is the loss-given-default term.

The **loan-to-value ratio** is the loan amount divided by the value of the asset securing it:

$$\text{LTV} = \frac{\text{Loan amount}}{\text{Appraised value of the collateral}}.$$

An 80% LTV means the loan is 80% of the asset's value, and — equivalently — the borrower put 20% of their own money in (their capital, C3, and their collateral cushion, C4, are two sides of the same coin here). Lower LTV is safer for the bank because there is more of the borrower's money standing in front of the bank's money, and more room for the asset's price to fall before the loan is "underwater" (worth more than the asset).

#### Worked example: an LTV of 80% and the cushion it buys

You buy a house for **\$400,000**. You put **\$80,000** down (your capital) and borrow **\$320,000** from the bank. Compute the LTV:

$$\text{LTV} = \frac{\$320{,}000}{\$400{,}000} = 0.80 = 80\%.$$

Your down payment of \$80,000 is 20% of the value — the *equity cushion* in front of the bank's loan. Now stress it. Suppose house prices fall 15% in a downturn. The house is now worth \$400,000 × 0.85 = **\$340,000**. Your \$320,000 loan is still *below* the \$340,000 value, so the bank is still fully covered — your 20% down payment absorbed the entire 15% fall and left \$20,000 of cushion. If you default now and the bank sells for \$340,000, it recovers its \$320,000 in full (before selling costs); its loss given default is essentially zero.

Now run it with no down payment — a 100% LTV loan, \$400,000 borrowed against a \$400,000 house. The same 15% fall leaves the house worth \$340,000 against a \$400,000 loan. The bank is **underwater by \$60,000**. If the borrower defaults, the bank seizes and sells for \$340,000 and eats a \$60,000 loss on a \$400,000 loan — a loss given default of 15% from the price fall alone, before legal and selling costs push it higher. The only difference between a zero-loss outcome and a \$60,000 loss was the 20% down payment. That is why mortgage lenders price 80% LTV loans far cheaper than 95% LTV loans, and why a borrower who puts less than 20% down is usually forced to buy mortgage insurance: their thin collateral cushion makes the bank's loss-given-default dangerously high.

The one-sentence intuition: LTV is the bank measuring how far the collateral can fall before *the bank's* money is at risk, and a 20% down payment buys the bank a 20% margin of safety against a falling market — exactly when defaults spike.

Now connect LTV to the loss directly.

![loss given default secured 25 percent versus unsecured 55 percent bar chart](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-8.png)

The chart above is why collateral exists. It shows the loss given default — the fraction of the loan the bank fails to recover *after* a default — by how the loan is secured. A **senior secured** loan, backed by good collateral with a low LTV, loses only about **25%** when it defaults: the bank gets 75 cents back on the dollar by selling the asset. A **senior unsecured** loan loses about **45%**. A **subordinated** loan (one that gets paid only after other lenders) loses about **65%**. And an **unsecured retail** loan — a credit card, a personal loan with nothing behind it — loses about **55%**. The whole point of asking for collateral is to slide a loan down this chart, from the 55% bar to the 25% bar. It does not make the borrower less likely to default; it makes the default cost the bank less than half as much.

#### Worked example: how collateral cuts the expected loss, not the default odds

Take the same borrower, the same \$10,000, the same 1.0% probability of default — but lend it two ways. **Unsecured**, with an LGD of 55%, the expected loss is $0.010 \times 0.55 \times \$10{,}000 = \$55$ a year. Now **secured** by good collateral at a low LTV, with an LGD of 25%, the expected loss is $0.010 \times 0.25 \times \$10{,}000 = \$25$ a year. The probability of default did not move — it is the same borrower with the same character and capacity. But the expected loss fell from \$55 to \$25, a 55% reduction, purely because the collateral changed what a default *costs*. Feed that back into the rate stack and the secured loan can be priced roughly 0.3% cheaper for the same profit. That gap — the cheaper rate a secured borrower gets — is collateral paying for itself.

The one-sentence intuition: collateral never touches PD; it lives entirely in LGD, and the reason a secured loan is cheaper is that the bank expects to recover most of its money even when the loan goes bad.

## The credit memo and the committee: how the yes actually happens

We now have the three numbers — a score (PD), a DSCR (capacity), an LTV (collateral cushion). For a small consumer loan, an algorithm reads those and decides in seconds. But for anything large — a business loan, a commercial mortgage, a syndicated facility — a human writes those numbers up into a document and a *committee* votes. This is where the abstract analysis becomes an actual decision, and it is the part outsiders never see.

The document is the **credit memo** (sometimes the credit application, credit paper, or deal memo). It is the underwriter's written case for or against the loan, and it is a discipline in itself. A good credit memo states the request (who, how much, what for, what terms), walks the Five Cs with the numbers (here is the score, here is the DSCR, here is the LTV, here is the collateral and its appraised value, here are the conditions in the borrower's industry), names the *risks* explicitly, names the *mitigants* (what protects the bank against each risk), and ends with a recommendation and the proposed terms. The memo exists so that the decision is made on *written analysis*, not a salesperson's enthusiasm — and so that, years later when the loan goes bad, someone can read exactly why the bank thought it was a good idea.

The committee is the group that votes. Crucially, *which* committee depends on the *size* of the loan, through a structure called **delegated lending authority** (or approval authority). A bank does not send every loan to the same room. A small loan might be approved automatically by the scorecard, or by a single loan officer up to their personal limit. A medium loan goes to a regional or divisional credit committee. A large loan goes to the senior credit committee. And a loan large enough to threaten the bank's own capital goes all the way to the board. The bigger the exposure, the more eyes and the higher the seniority required to approve it — because the bigger the loan, the more of the bank's thin equity cushion it puts at risk.

There is a deliberate separation of powers buried in this structure. The banker who *originates* the loan — who has the relationship with the borrower and is paid, partly, for booking business — is usually *not* the person who approves it. Approval authority sits with credit risk, an independent function whose job is to say no. This is the first of the famous *three lines of defense*: the business takes the risk, an independent risk function challenges it, and internal audit checks that both did their jobs. The reason is incentives. A banker rewarded for volume will always find a loan attractive; a credit officer rewarded for low losses will always find a reason to push back. The tension between them is not dysfunction — it is the control. When a bank lets origination and approval collapse into the same person (or lets the salespeople overrule credit), it has dismantled the very mechanism that keeps bad loans off the book, and the losses follow on a lag.

The decision does not end when the committee votes yes. A real approval comes wrapped in **covenants** — ongoing promises the borrower must keep, such as maintaining a DSCR above 1.20× or an LTV below 75%, reporting financials quarterly, or not taking on new senior debt. Covenants turn a one-time analysis into a living one: if the borrower's DSCR drifts toward 1.0×, the covenant *breaches* and the bank gets the right to act — renegotiate, demand more collateral, or call the loan — *before* an actual missed payment. This is the bank's early-warning system, and it is why credit analysis is not a single event at origination but a process that runs the whole life of the loan. The "monitoring" box at the end of the pipeline is where a good bank catches a deteriorating loan a year before it defaults.

![loan approval pipeline application score credit memo committee decision](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-7.png)

The pipeline above is the path a real loan walks. An application comes in (1). It is scored and the income and collateral are verified (2). An underwriter writes the credit memo — the Five Cs with the numbers, the risks, and the mitigants (3). The size of the loan routes it to the right approval authority (4): small loans get auto-approved or sign-off from an officer; large ones go to a credit committee; the biggest go to the board. The committee debates and votes — and committee votes are often *not* unanimous; a single strong objection can sink a deal or attach conditions (5). Finally the decision is booked with its terms, covenants, and conditions, signed and documented, and the loan moves into monitoring (6). Notice the loop is not just "yes/no" — the most common real outcome for a marginal deal is "yes, *but*."

That "yes, but" is the third box in the decision and the one beginners forget exists.

![loan decision tree approve decline approve with conditions](/imgs/blogs/credit-analysis-the-five-cs-and-how-a-loan-gets-approved-4.png)

The decision graph above shows the three real outcomes. An application hits the **score gate** first — a hard cutoff (here, a FICO of 660) that filters out the clear declines cheaply, before anyone wastes time writing a memo on a hopeless file. A score below the cutoff (say 640) usually fails the screen and the loan is declined or counter-offered on much tougher terms. The files that clear the gate go to the **capacity and collateral test** — the DSCR and LTV check. A *strong file* (DSCR 1.40×, LTV 70%) is a clean **approve** at the standard rate. A *thin file* (DSCR 1.10×, LTV 92%) is not a no — it is an **approve with conditions**: the bank will lend, but only if the borrower puts more money down, brings in a guarantor, or accepts a higher rate that pays the bank for the extra risk. The middle box, "approve with conditions," is where most of the negotiation in real lending actually happens.

#### Worked example: approve, decline, or condition at a cutoff

A bank sets its consumer-loan policy: auto-decline below a FICO of 660; auto-approve above 720 if DSCR ≥ 1.25× and LTV ≤ 80%; everything in between goes to manual review. Three applicants arrive for a \$300,000 mortgage on a \$375,000 house (an 80% LTV — \$300,000 / \$375,000 = 0.80).

**Applicant 1: FICO 640, DSCR 1.30×.** The score is below the 660 cutoff. The hard gate fires first — this is an automatic **decline**, regardless of the healthy DSCR, because the bank has decided that below 660 the default odds are too high to lend at any rate it is willing to offer. (The bank may counter-offer a smaller loan with a bigger down payment, but the headline answer is no.)

**Applicant 2: FICO 760, DSCR 1.45×, LTV 80%.** Score well above the auto-approve line, capacity comfortably above 1.25×, LTV at the limit. Every C is strong. This is an automatic **approve** at the standard rate. No committee, no negotiation — the policy already said yes.

**Applicant 3: FICO 700, DSCR 1.10×, LTV 80%.** The score clears the hard cutoff but sits below the auto-approve line, and the DSCR of 1.10× is *below* the 1.25× target — a thin capacity cushion. This goes to manual review, and the likely outcome is **approve with conditions**: lend, but require either a larger down payment (cutting the LTV to 70% to thicken the collateral cushion and lower the loss-given-default) or a co-signer (adding a second source of repayment to shore up capacity), or charge a higher rate that prices the thinner cushion. The bank does not say no; it reshapes the deal until the Five Cs balance.

The one-sentence intuition: a cutoff is a cheap first filter that kills the obvious noes, but the real craft is in the middle band, where a loan is turned from a marginal yes into a safe one by *changing its terms* — more capital, more collateral, more coverage, or more rate.

## Common misconceptions

**"A high income means you'll get the loan."** No. Income feeds *capacity*, but a high earner with high debts can have a worse DSCR or DTI than a modest earner with no debts. A doctor earning \$400,000 with \$380,000 of existing obligations has a DTI near 95% and will be declined, while a teacher earning \$60,000 with almost no debt sails through. Capacity is a *ratio*, not a level — it is income relative to obligations, and the obligations are half the math.

**"Collateral means the bank can't lose."** Wrong, on two counts. First, collateral changes the loss given default, not the probability of default — a fully secured loan can still default and still cost the bank money. Second, collateral is worth *least* exactly when you need it most: house prices and business-asset values fall in the same recessions that cause defaults, so the bank that thought it had 25% LGD can discover it has 40% when it actually has to sell. In the 2008 crisis, lenders who believed houses could only go up learned that a 95% LTV loan against a falling house recovers far less than the model assumed.

**"The credit score is the whole decision."** For a \$3,000 credit card, nearly. For anything substantial, no. The score captures character and part of capacity, but it knows nothing about your *specific* cash flow on *this* deal, the *specific* collateral, or the *conditions* in your industry. A 780 FICO does not approve a commercial loan whose DSCR is 0.9× — the cash flow does not cover the payment, and no score fixes that. The score is one C, dressed up as a number; the other Cs still vote.

**"A default means the bank loses the whole loan."** No — this is the PD-versus-LGD confusion again, and it is the most expensive mistake a beginner makes. A default on a well-secured loan might cost the bank 25% of the balance; a default on an unsecured loan might cost 55%. The expected loss is PD × LGD, and confusing default (the event) with loss (the consequence) leads people to wildly over- or under-estimate how much a bad loan actually costs.

**"Getting declined is the bank punishing you."** A decline is the bank concluding that, at any rate it is willing to charge, the expected loss is too high or too uncertain. It is not a moral verdict; it is arithmetic about the cycle. The same file declined in a recession would often be approved in a boom, because *conditions* — the fifth C — changed the bank's read on how likely the loan is to repay. The decline is about the world as much as about you.

## How it shows up in real banks

**Automated consumer underwriting (every day, everywhere).** When you apply for a credit card online and get a decision in under a minute, you have met an application scorecard doing the Five Cs at machine speed. The model pulls your bureau file, computes a score that encodes character and utilization-based capacity, checks a handful of policy rules (income above a floor, no recent bankruptcy), and approves, declines, or refers to a human. The credit memo and committee are collapsed into a few hundred milliseconds of code. This is the high-volume, low-ticket end of the same process a corporate banker runs by hand on a \$50 million deal.

**The 2008 mortgage crisis (capacity and conditions abandoned).** The defining credit failure of the modern era was, at its core, a collapse of the Five Cs. Lenders made "no-doc" and "stated-income" loans that skipped verifying *capacity* — borrowers wrote down whatever income they liked. They made high-LTV loans that left thin *collateral* cushions. And they did it on the assumption that house prices only rise, ignoring the *conditions* C. When prices fell, borrowers were underwater, their thin collateral could not cover the loans, and the unverified capacity turned out to be fiction. The expected-loss models had assumed an LGD of perhaps 20%; actual losses on subprime pools ran far higher because LTVs were high and prices fell together. Every C the lenders skipped came back as a loss.

**The DSCR-driven commercial-real-estate squeeze (2023-2024).** When the Fed raised rates from near zero to over 5% in 2022-2023, floating-rate commercial-property loans that comfortably cleared a 1.25× DSCR at 4% interest suddenly faced a doubled debt service. A property whose net operating income covered the old payment 1.25× might cover the new payment only 0.9× — below 1.0×, meaning the cash flow no longer covered the loan. This is the *conditions* C (rates) crushing the *capacity* C (DSCR) on loans that were perfectly sound when written. Banks with heavy commercial-real-estate exposure had to renegotiate, demand more equity, or watch DSCRs fall through 1.0×. It is a live demonstration that capacity is not a fixed property of a borrower; it is a function of the rate environment.

**Risk-based pricing on auto loans (visible to anyone who has bought a car).** Walk into a dealership with a 780 FICO and you might be offered 4% financing; walk in with a 620 and the same car costs you 13% or more. That spread is not negotiation — it is the lender's scorecard mapping your band to a PD, multiplying by the auto loan's LGD (autos depreciate fast, so LGD is meaningful), and pricing the expected loss into your rate. The car is identical; the credit cost is not. This is the FICO-to-PD-to-rate chain from earlier in this post, running in real life on a Saturday afternoon.

**The credit committee that says "yes, but" (every large bank, every week).** When a relationship banker brings a \$30 million loan to the senior credit committee, the most common outcome is not a clean yes or no — it is conditions. The committee might approve the loan but require a personal guarantee from the owner (shoring up character and adding a second source of repayment), a tighter covenant package (early-warning triggers that let the bank act before a default), a lower advance rate (cutting LTV to thicken the collateral cushion), or a higher margin (pricing the residual risk). The deal that walks out of the room is rarely the deal that walked in. That reshaping — turning a marginal file into a safe one by changing its terms — is what credit analysis *is* at the large-ticket end.

**Provisions through the cycle (why bank earnings are pro-cyclical).** Because conditions drive PD, and PD drives expected loss, a bank's credit costs are not steady — they balloon in recessions and shrink in booms. When the cycle turns down, the same loan book sees its modeled PDs rise several-fold, the bank must set aside larger reserves against expected losses, and those reserves hit earnings immediately even before a single loan actually defaults. This is why bank profits boom late in an expansion (when provisions are low) and crater early in a downturn (when they spike). The Five Cs are not static; the fifth one, conditions, makes the whole framework breathe with the economy.

## The takeaway: how to use this

Strip away the jargon and credit analysis is the discipline of answering one question — *will I get my money back, and what does it cost me if I don't?* — with enough structure that the answer survives a salesperson's optimism and a borrower's best self-presentation. The Five Cs are the structure. Character and capacity tell you whether the loan repays in the normal course; capital and collateral tell you what a default costs; conditions tell you whether the world is about to make a liar of all your numbers.

If you take three things from this post, take these. **First, separate PD from LGD and never confuse them again.** The probability of default is about the borrower; the loss given default is about the structure of the loan. A great borrower on a badly-structured loan and a shaky borrower on a well-secured one can have the same expected loss for completely different reasons, and you only see that if you keep the two numbers apart. Almost every credit mistake a beginner makes is some version of conflating "they might not pay" with "I'll lose everything."

**Second, respect the cushion.** A DSCR of 1.25×, an LTV of 80%, a 20% down payment — these are not arbitrary thresholds a regulator picked. They are calibrated to what a recession actually does: a 20% income drop, a 15% asset-price fall. The cushion is the entire point of the number, because the loan is not made for the good times — every loan looks fine in the good times. It is made for the bad ones, and the cushion is what survives them.

**Third, read the rate as information.** When a lender quotes you a rate, they are telling you what they think of your Five Cs. A high rate is the bank's estimate of your PD and LGD, priced. If you can move a C — put more money down (capital and collateral, cutting LTV), pay down other debts (capacity, improving DSCR), or simply let your payment history lengthen (character, raising your score) — the rate moves with it, because the rate *is* the Five Cs converted into a price.

And tie it back to the spine of how a bank lives or dies. A bank is a leveraged, confidence-funded machine running on a thin equity cushion. Loans are where it earns its spread, and loan losses are what eat that cushion. Credit analysis is the bank deciding, one loan at a time, how much risk to put in front of its own equity. Do it well — separate PD from LGD, demand the cushion, price the risk — and the spread accrues to the franchise. Do it badly — skip capacity, ignore conditions, believe collateral can't fall — and the losses arrive faster than the equity can absorb them, and the machine that looked so solid in the boom is suddenly the headline of a bust. The Five Cs are not paperwork. They are the difference between a bank that survives the cycle and one that becomes a case study.

## Further reading & cross-links

- [The lending business: how a bank underwrites a loan end to end](/blog/trading/banking/the-lending-business-how-a-bank-underwrites-a-loan-end-to-end) — the full origination-to-collection pipeline that credit analysis sits inside.
- [Loan pricing: cost of funds, risk premium, and the capital charge](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge) — the rate stack from this post built out in full, including RAROC.
- [Credit risk management: PD, LGD, EAD, and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss) — the master equation taken from a single loan to the whole portfolio and the credit cycle.
- [Credit rating agencies: Moody's, S&P, Fitch](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch) — where the AAA-to-CCC scale in the default-odds chart comes from, and how the agencies estimate the same PDs banks do.
