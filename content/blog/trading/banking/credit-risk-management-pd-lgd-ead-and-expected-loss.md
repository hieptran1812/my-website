---
title: "Credit Risk Management: PD, LGD, EAD and Expected Loss"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank turns the chance a loan goes bad into a number it can price, provision for, and hold capital against — the PD times LGD times EAD engine, and why the surprise above the average is what really sinks banks."
tags: ["banking", "credit-risk", "expected-loss", "probability-of-default", "loss-given-default", "economic-capital", "credit-cycle", "risk-management"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank turns the messy question "will this loan go bad?" into a single dollar figure with one equation: expected loss equals probability of default times loss given default times exposure at default, or **EL = PD × LGD × EAD**.
>
> - **Expected loss is a cost, not a surprise.** A bank knows roughly what fraction of its loans will sour each year, prices that into the spread it charges, and books it as provisions. The average loss is *budgeted*, like spoilage in a grocery store.
> - **The surprise is what kills you.** Losses do not arrive smoothly at the average — they cluster in bad years far above it. That swing above the average is *unexpected loss*, and a bank holds **economic capital** to absorb it. Capital is for the bad year, not the average year.
> - **PD is not one number.** A *through-the-cycle* PD is a steady long-run average; a *point-in-time* PD tracks today's conditions and lurches in a downturn. Confusing the two is how banks under-provision right before a recession.
> - **The one number to remember:** a \$1,000,000 loan to a BBB borrower (PD about 0.24%) with 45% loss given default has an expected loss of just **\$1,080 a year** — but the *worst* year on a whole portfolio can be four or five times the average, and that is exactly the gap capital must cover.

In the spring of 2007, almost every large bank in the world was reporting record-low loan losses. Provisions — the money set aside for loans expected to go bad — were at multi-decade lows. The models said the world was safe. Default rates on mortgages, on corporate loans, on credit cards were all near historic floors. Bank executives told shareholders that risk management had never been better.

Eighteen months later, the same banks were taking losses so large that several of them ceased to exist. The loans had not changed overnight. What changed was the *economy* the loans lived in — and with it, the probability that any given borrower would stop paying. The models that had looked so reassuring in 2007 had been measuring the wrong thing: they had been reading the calm of a boom and projecting it forward, instead of remembering that lending is a business whose losses cluster in the years you least expect them.

This post is about the engine underneath all of that — the machinery a bank uses to convert "will I get paid back?" into a number it can charge for, set aside for, and survive. It is one of the most important calculations in finance, and at its heart it is multiplication. The diagram above is the mental model: three numbers — how likely default is, how much you lose when it happens, and how big the loan is — multiply together into a single expected loss, which the bank then prices into the spread. Get those three numbers right and you have a viable lending business. Get them wrong — especially in the direction the whole industry got them wrong in 2007 — and you have the next banking crisis.

![Expected loss equals probability of default times loss given default times exposure at default then priced in](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-1.png)

This connects directly to the spine that runs through this whole series: **a bank is a leveraged, confidence-funded maturity-transformation machine — it borrows short and lends long, earns the spread, and survives only as long as its thin equity cushion absorbs losses faster than they arrive.** Credit risk is the single largest source of those losses for a traditional bank. Everything in this post is ultimately about one question: will the equity cushion be big enough when the losses come?

## Foundations: the four numbers that define a loan's risk

Before any math, let's build the intuition with something everyone understands: lending money to a friend.

Imagine you lend \$100 to ten different friends. You know, roughly, that not everyone pays you back. Maybe one of the ten will flake. When that one flakes, you might still get *something* — perhaps they had borrowed against a bike they let you keep, so you recover \$60 and lose \$40. And the size of each loan matters: if one friend borrowed \$1,000 instead of \$100, that single loan carries more risk than the rest combined.

That everyday picture contains the three ingredients of credit risk, and a bank just gives them precise names.

### Probability of default (PD)

**Probability of default, or PD, is the chance that a borrower stops paying within a set period — almost always one year.** It is a probability, so it lives between 0% and 100%. A borrower who has never missed a payment and has plenty of income might have a PD of 0.1% — one chance in a thousand of defaulting this year. A borrower already stretched thin might have a PD of 20% — one chance in five.

What counts as "default" is itself a defined event, not a vibe. Under the Basel rules that govern bank capital worldwide, a borrower is in default when either they are **90 days past due** on a material obligation, or the bank judges they are **unlikely to pay** in full without the bank taking some action like seizing collateral. The 90-days line is the workhorse definition: miss three months of payments and, for risk-measurement purposes, you have defaulted, even if you eventually catch up.

A *basis point* — a term you'll see constantly — is one hundredth of a percent, or 0.01%. So a PD of 0.24% can be written as 24 basis points. We'll use both.

### Loss given default (LGD)

Default is not the same as total loss. When a borrower defaults, the bank usually recovers *something* — by selling collateral, by negotiating a partial repayment, by suing. **Loss given default, or LGD, is the fraction of the exposure the bank ultimately fails to recover, expressed as a percentage.**

If a bank lends \$100, the borrower defaults, and the bank eventually claws back \$60 (after the costs of the workout), then it lost \$40 — an LGD of 40%. The complement, the 60% it got back, is the *recovery rate*. LGD plus recovery rate always equals 100%.

The single biggest driver of LGD is **collateral**. A mortgage is secured by a house the bank can foreclose on and sell, so mortgage LGDs are low — often 10% to 25%. An unsecured credit card has nothing behind it but a promise, so its LGD is high — 50% to 75%. We will see the full picture shortly.

### Exposure at default (EAD)

**Exposure at default, or EAD, is the dollar amount the bank is actually owed at the moment default happens.** For a simple term loan, that's roughly the outstanding balance. But for a credit line — a card, a revolving business facility — EAD is trickier, because a distressed borrower tends to *draw down* their available credit right before they fail. A company about to go bankrupt will max out every line it has. So EAD on a \$100,000 credit line is rarely the current balance; it's the current balance plus an estimate of how much more they'll draw before the end. Banks model this with a *credit conversion factor* — the fraction of the unused limit they expect to be drawn before default.

### Expected loss (EL)

Now we can assemble the headline number. **Expected loss is the average amount a bank expects to lose on a loan over a year, and it is simply the product of the other three:**

$$EL = PD \times LGD \times EAD$$

where $PD$ is the probability of default (a fraction), $LGD$ is the loss given default (a fraction), and $EAD$ is the exposure at default (in dollars). The result is in dollars. Read it aloud: *the chance it goes bad, times how much you lose when it does, times how much is on the line.* Each term answers a different question, and you need all three.

The word "expected" is doing heavy lifting, and it's worth pausing on. Expected loss is a *statistical average* across many loans and many years. On any single loan, you do not lose \$1,080 — you either lose nothing (the borrower pays) or you lose a large chunk (they default). The \$1,080 is what you lose *on average* if you make that same loan thousands of times. A bank, making millions of loans, is exactly the place where averages become real.

### Unexpected loss (UL)

Here is the concept that separates people who understand banking from people who don't. **Losses do not arrive at the expected rate every year.** Some years are quiet and losses come in below the average; some years are brutal and losses come in far above it. **Unexpected loss is the amount by which actual losses in a bad year exceed the expected average.** It is the *swing*, not the mean.

Expected loss is the cost you can plan for. Unexpected loss is the surprise you cannot. And because you cannot price a surprise into individual loans, you need a different defense against it: a cushion of equity sized to absorb the bad year. That cushion is **economic capital**.

### Economic capital

**Economic capital is the amount of equity a bank decides it needs to hold to survive a bad year of a chosen severity — typically a one-in-a-thousand year.** If a bank wants to be confident it can absorb any loss short of the worst year in a millennium, it holds enough capital to cover the gap between expected loss and that 99.9th-percentile loss. Economic capital is the bank's *internal* answer to "how much equity keeps us solvent?", distinct from the *regulatory* capital that Basel rules require (the two are related but not identical).

### Through-the-cycle vs point-in-time

Finally, two ways to *measure* PD, because this distinction is where banks get into the most trouble.

A **through-the-cycle (TTC)** PD is a long-run average default rate that deliberately *smooths over* the economic cycle. It asks: across booms and busts, what fraction of borrowers like this one default in an average year? A TTC PD barely moves from year to year.

A **point-in-time (PIT)** PD asks the opposite: given conditions *right now*, what is the chance this borrower defaults in the next year? A PIT PD falls in a boom (everyone looks safe) and spikes in a recession (everyone looks risky).

Neither is wrong; they answer different questions. The danger is using a PIT measure that has drifted down in a boom, calling it the "true" risk, and pricing and provisioning as if the good times will last. We'll return to this — it is, in one sentence, the story of 2007.

With the vocabulary in hand, let's go deep on each factor.

## Probability of default: the chance the borrower stops paying

PD is the term people argue about most, because it is the hardest to pin down and it swings the most. Loss given default is anchored to physical collateral; exposure is roughly the balance. But PD is a forecast of human and economic behavior, and forecasts are where models go to die.

Banks estimate PD in a few ways. The simplest is the **rating-to-PD mapping**: assign each borrower an internal grade (or use an external agency rating — AAA, AA, A, BBB, and so on), then look up the historical default rate for that grade. Rating agencies publish decades of data on what fraction of, say, BB-rated companies defaulted within a year, and banks calibrate to those tables. More sophisticated banks build **statistical models** — logistic regressions or machine-learning classifiers — that take a borrower's financials, payment history, industry, and macro conditions and output a PD directly.

The headline fact about PD is how *non-linear* it is across the credit spectrum. The jump from a safe borrower to a risky one is not gentle; it is exponential. Look at the numbers:

![One-year default probability by rating shown on a log scale](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-2.png)

This chart is on a *logarithmic* scale — each gridline is ten times the one below it — and even so the bars climb steeply. A AAA borrower has a one-year PD around 0.01% (one in ten thousand). A BBB borrower, still investment grade, sits near 0.24% (about one in 420). Cross into high yield at BB and you are at 1.2% (one in 83). By single-B it's 5.5%, and by CCC — borrowers already in distress — it's 26%, more than one in four defaulting *every single year*.

That investment-grade/high-yield divide, marked here by the color change from BBB to BB, is one of the most important lines in finance. It is the boundary the bond market obsesses over, and it sits between a PD of 0.24% and 1.2% — a fivefold jump for one notch. (For a deeper treatment of how those grades get assigned and what they mean, see the cross-links at the end.)

#### Worked example: expected loss on a single loan

Let's run the core equation end to end, the calculation a bank does on every loan it books.

You are a bank. A mid-sized company walks in wanting a **\$1,000,000** term loan. Your credit team rates it **BBB** — solid, investment grade, but not pristine. From your PD table, a BBB borrower has a one-year **PD of 0.24%**, or 0.0024 as a decimal. The loan is **senior unsecured** — it ranks ahead of other unsecured creditors but has no specific collateral — so your **LGD is 45%**, meaning if it defaults you expect to recover 55 cents on the dollar after the workout. And the **EAD** is the full \$1,000,000, since it's a fully-drawn term loan.

Now multiply:

$$EL = 0.0024 \times 0.45 \times \$1{,}000{,}000 = \$1{,}080$$

The expected loss is **\$1,080 per year** — about 11 basis points of the loan amount (\$1,080 / \$1,000,000 = 0.108%). That is what this loan "costs" you in average credit losses annually. To break even on credit risk alone, the spread you charge over your cost of funds must be at least 0.108%. (You'll charge more — to cover operating costs, capital, and profit — but \$1,080 is the credit-loss floor.)

The intuition: on this one loan in any given year you almost certainly lose *nothing*. But if you made ten thousand identical BBB loans, about 24 of them would default, you'd lose roughly 45% on each, and the total across the book would land near \$1,080 per loan. **Expected loss is what you charge everyone so that the few who default are paid for by the many who don't.**

### Why PD is the term that breaks

The reason PD causes crises is that it is *unstable in exactly the wrong way*. LGD and EAD are relatively sticky — collateral values move, but slowly. PD, by contrast, can multiply several times over in a single recession. A pool of mortgages with a 1% PD in good times can hit 5% or 8% in a deep downturn, because the same job losses that push one borrower to default push *all* of them toward the edge at once. We will see this clustering effect — the real reason credit losses are so dangerous — when we get to portfolio risk.

## Loss given default: how much you lose when it goes wrong

If PD is a forecast, LGD is closer to an accounting exercise — but a contested one, because the recovery process is slow, expensive, and uncertain. When a loan defaults, the bank doesn't instantly get its money back. It enters a *workout*: it tries to restructure, or it seizes and sells collateral, or it joins a line of creditors in a bankruptcy that can take years. LGD captures the net result of all that, including the legal fees, the time value of money lost while waiting, and the haircut on a fire-sale of collateral.

The dominant lever on LGD is what stands behind the loan. Here is the spread across typical structures:

![Loss given default by collateral and seniority shown as horizontal bars](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-3.png)

A **senior secured** loan — backed by specific collateral and first in line — has an LGD around 25%: the bank recovers three quarters of its money. **Senior unsecured** debt, with no specific collateral but a senior claim, sits near 45%. **Subordinated** debt, which ranks behind senior creditors and gets whatever's left, has an LGD around 65% — you lose two thirds. And **unsecured retail** lending (think credit cards and personal loans) lands near 55%: there's no collateral, but consumer recoveries through collections and garnishment claw back a bit more than the worst corporate cases.

The pattern is the whole point of secured lending: **collateral converts a high-loss exposure into a low-loss one without changing the probability of default at all.** A borrower is exactly as likely to default whether the loan is secured or not — but the bank's loss when they do can be cut in half by taking good collateral. This is why mortgages are among the safest things a bank does despite enormous size: the house secures the loan.

#### Worked example: how collateral changes LGD and the loss

Take the same borrower defaulting on a **\$500,000** loan, and see how much the *structure* of the loan changes what the bank actually loses. The PD is identical in every case — say this borrower defaults — so we're isolating the LGD effect. EAD is \$500,000.

- **Senior secured (LGD 25%):** loss = 0.25 × \$500,000 = **\$125,000**. The bank seizes and sells collateral worth most of the loan.
- **Senior unsecured (LGD 45%):** loss = 0.45 × \$500,000 = **\$225,000**. No specific collateral, but a senior claim in bankruptcy.
- **Unsecured retail (LGD 55%):** loss = 0.55 × \$500,000 = **\$275,000**. Collections recover something, but most is gone.
- **Subordinated (LGD 65%):** loss = 0.65 × \$500,000 = **\$325,000**. Last in line; gets the scraps.

The same default, the same \$500,000, produces losses ranging from **\$125,000 to \$325,000** — a 2.6× difference — driven entirely by what backs the loan and where it ranks. The intuition: **PD tells you how often the accident happens; LGD tells you how bad the crash is, and collateral is the seatbelt.** A lender who takes good security can survive a much higher default rate than one who doesn't.

### The catch: LGD rises in the same downturn that raises PD

There is a cruel correlation hiding here. LGD is not constant across the cycle — it *worsens* precisely when PD spikes. Why? Because the collateral behind loans is usually worth less in a recession. When the economy turns, more borrowers default (PD up) *and* the houses, equipment, and inventory the bank seizes sell for less (LGD up). The two factors move together in the wrong direction. A mortgage book modeled with a calm-times LGD of 15% can see realized LGDs of 35% or more when house prices fall 30% and the bank is dumping foreclosed homes into a buyer's market. Modeling PD and LGD as independent — as if a bad year hits one but not the other — systematically *understates* expected loss in exactly the scenario that matters. This is one of the deep flaws that the 2007–2009 models shared.

## Putting it together: expected loss across the credit spectrum

Now combine PD and LGD across the full range of borrowers, holding EAD fixed at \$1,000,000 and LGD at 45% (senior unsecured), to see what expected loss actually looks like as you move from the safest borrower to the riskiest:

![Expected loss in dollars on a one million dollar loan by borrower rating](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-4.png)

This is the same data as the PD chart, now translated into the dollars a bank would lose on average per year — and again it's on a log scale, because the range is enormous. A AAA loan has an expected loss of **\$45 a year**. The BBB loan from our worked example: **\$1,080**. A BB loan jumps to **\$5,400**, a single-B to **\$24,750**, and a CCC loan to a staggering **\$117,000** a year — more than 11% of the loan, every year, on average.

Read down the chart and you can see why banks price risky loans so much more expensively. To break even on credit losses alone, a CCC borrower must pay roughly 11.7% more in spread than a risk-free borrower — and that's *before* the bank charges for the capital it must hold and the operating cost of managing a troubled loan. The reason high-yield borrowers pay double-digit interest rates is not greed; it's arithmetic. The expected loss really is that large.

This chart also quietly explains a strategic truth about banking: **you cannot out-volume your way out of bad credit selection.** Making ten times as many CCC loans does not dilute the risk — it concentrates it. Every loan carries its own \$117,000 expected loss, and they all tend to default together in a downturn. Growth in lending without discipline in *who* you lend to is the most reliable way to destroy a bank.

## Exposure at default: the number that grows when you least want it to

EAD gets less attention than PD and LGD, which is a mistake, because it has a nasty habit: it tends to be *largest* in exactly the circumstances where default is most likely. For a simple, fully-drawn term loan, EAD is uncontroversial — it's the outstanding balance, declining predictably as the borrower amortizes. But most bank credit isn't a static term loan. It's *revolving*: credit cards, corporate working-capital lines, overdraft facilities. The borrower has a *limit* and draws against it as needed, and the balance moves around.

The problem is that a borrower heading toward default behaves in a predictable, unhelpful way: they draw down everything they can. A company that can see bankruptcy coming will max out every credit line it has while the lines are still open, because cash in hand is worth more than an unused facility once the lawyers arrive. So a \$1,000,000 credit line that's only \$200,000 drawn today is *not* a \$200,000 exposure in a default scenario — it's something much closer to the full \$1,000,000, because the borrower will pull the other \$800,000 on the way down.

Banks handle this with a **credit conversion factor (CCF)** — an estimate of what fraction of the *undrawn* limit will be drawn before default. If history says distressed borrowers on this type of facility draw 75% of their remaining headroom before failing, the CCF is 75%, and:

$$EAD = \text{drawn balance} + CCF \times (\text{limit} - \text{drawn balance})$$

#### Worked example: exposure at default on a credit line

A corporate client has a **\$1,000,000** revolving line, currently **\$300,000 drawn**, leaving \$700,000 of unused headroom. The bank's CCF for this facility type is **70%**. The exposure at default isn't the \$300,000 on the books today:

$$EAD = \$300{,}000 + 0.70 \times (\$1{,}000{,}000 - \$300{,}000) = \$300{,}000 + \$490{,}000 = \$790{,}000$$

If this client is BB-rated (PD 1.2%) and the loan is senior unsecured (LGD 45%), the expected loss is:

$$EL = 0.012 \times 0.45 \times \$790{,}000 = \$4{,}266$$

Compare that to the naive calculation using only the drawn balance — $0.012 \times 0.45 \times \$300{,}000 = \$1{,}620$ — and you can see the trap: ignoring the undrawn commitment would have *understated* the expected loss by more than 60%. The intuition: **a credit line is a loan you've half-made; the borrower controls when to finish making it, and they'll finish it at the worst possible moment for you.** A bank that prices and reserves only against drawn balances is systematically under-charging for the risk it has actually taken on through its commitments.

This is also why bank regulators treat undrawn commitments as real exposure for capital purposes, and why a sudden surge in line drawdowns across a banking system — as happened in March 2020, when companies drew billions on their revolvers to hoard cash — is a leading indicator of stress. The exposure was always there; the drawdown just made it visible.

## Expected vs unexpected loss: the average you price, the surprise you capitalize

Here is the conceptual heart of credit-risk management, and the place where intuition most often goes wrong. People assume that if a bank has correctly calculated expected loss and charged for it, it is safe. It is not. Expected loss is only the *average*. The thing that destroys banks is the *deviation* from the average — and you cannot price a deviation into a loan, because by definition you don't know which year it will come.

![Expected loss is priced in while unexpected loss is absorbed by capital](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-5.png)

The figure splits losses into two species with two completely different treatments. **Expected loss** — the long-run average — is *priced in*: it's charged in the loan spread, every borrower pays a slice of it, and it's booked as provisions, an ordinary operating cost. If a year's actual losses come in right at the expected level, the bank still profits, because the spread already covered them. **Unexpected loss** — the swing above the average in a bad year — *cannot* be priced away, because it's a surprise. It can only be absorbed by equity. That equity buffer is economic capital. And if the buffer is too thin, a single bad year doesn't dent the bank — it wipes it out.

This is the cleanest possible statement of why banks hold capital at all. **Provisions are for the average loss; capital is for the surprise.** A bank with perfect expected-loss pricing and zero capital would still fail the first time losses ran above average, which they inevitably do. The whole edifice of bank capital regulation — every ratio, every stress test in this series — exists to make sure the surprise cushion is big enough.

To see why the surprise is so much larger than people expect, we have to look at the *shape* of credit losses.

## The loss distribution: why the tail is so fat

If you plotted a bank's annual credit losses over many years, you would not get a nice symmetric bell curve centered on the expected loss. You would get something lopsided and ugly: a tall cluster of quiet years near the average, and a long, thin tail of catastrophic years stretching far to the right.

![Credit loss distribution showing the average and the long right tail](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-6.png)

This skewed shape is the signature of credit risk, and it has a structural cause. In a normal year, defaults are roughly independent — one borrower's bad luck doesn't affect another's — so losses cluster tightly near the expected level (the amber region, the average you priced in). But in a recession, defaults become *correlated*: the same shock — a recession, a housing crash, a spike in rates — hits everyone at once, and many borrowers default together. That correlation is what creates the long right tail. Most years are calm; rare years are devastating.

The figure marks three zones. The amber region up to the expected loss is the average — the cost the bank budgeted for. The blue region between the average and the 99.9th-percentile worst year is **unexpected loss**, and the width of that blue band *is* the economic capital the bank must hold: enough equity to absorb everything up to a one-in-a-thousand year. The red tail beyond that point is the region of insolvency — losses so extreme that even the capital buffer is exhausted and the bank fails.

#### Worked example: expected versus unexpected loss on a portfolio

Let's make the two species of loss concrete with numbers. Suppose a bank holds a portfolio of corporate loans with an **expected loss of \$10 million per year** — that's its long-run average, fully priced into spreads and booked as provisions.

In a *calm* year, actual losses might come in at **\$7 million** — below average. The bank had charged enough to cover \$10 million, so it pockets the \$3 million difference as extra profit. No drama.

In a *bad* year — a recession — defaults cluster and actual losses balloon to **\$45 million**. The expected-loss provisions covered the first \$10 million. The remaining **\$35 million is the unexpected loss** — the surprise. Where does it come from? Not from spreads (those only covered \$10 million); it comes straight out of the bank's equity. If the bank holds, say, \$50 million of economic capital against this portfolio, it survives — the \$35 million hit eats most of the cushion but leaves the bank standing. If it held only \$20 million of capital, the \$35 million loss wipes out the capital entirely and the bank is insolvent.

The intuition: **the expected loss tells you the price of doing business; the unexpected loss tells you how much capital you need to stay in business.** A bank that sizes its capital to the average loss instead of the bad-year loss is not conservatively capitalized — it is one recession from failure. The whole job of economic capital is to make the bank survive the \$45 million year, not the \$10 million one.

### How much capital? The economics of the buffer

How does a bank decide the width of that blue band? It picks a *confidence level* — a probability of survival it's targeting — and holds enough capital to cover losses up to that point in the distribution. A common choice is 99.9%, meaning the bank wants to survive all but the worst one-in-a-thousand years. The economic capital is then the distance from the expected loss out to that 99.9th-percentile loss.

The mathematics of this — the formula that turns PD, LGD, and a correlation assumption into a capital number — is the Basel "internal ratings-based" risk-weight formula, built on a model called the Asymptotic Single Risk Factor model. We won't grind through the calculus here (the [risk-management series](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies) covers the value-at-risk machinery this rests on), but the intuition is exactly the picture above: capital is sized to the *tail*, not the average, and the single most important input besides PD is the assumed *correlation* between borrowers — how much they tend to default together. Underestimate that correlation and you radically understate the capital you need. That, in one sentence, is the modeling error that sat underneath the 2008 crisis.

## Through-the-cycle vs point-in-time: the same risk, measured two ways

Return now to the distinction we flagged in the foundations, because it is where good banks and reckless ones diverge. The same borrower has a different PD depending on *which question you're asking*, and getting the two confused is how banks end up under-reserved at the worst possible moment.

![Through the cycle versus point in time probability of default comparison matrix](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-7.png)

A **through-the-cycle (TTC)** PD is a long-run average that deliberately ignores where you are in the cycle. It asks: across an entire boom-and-bust cycle, what fraction of borrowers like this one default in a typical year? Because it averages over good and bad times, a TTC PD is *stable* — it barely moves in a boom and barely moves in a recession. It's the natural input for things that should be stable: capital planning, agency ratings, the buffers that shouldn't whipsaw with the economy.

A **point-in-time (PIT)** PD asks the opposite: given conditions *right now*, what's the chance this borrower defaults in the next twelve months? In a boom, a PIT PD *falls* — incomes are rising, defaults are rare, everything looks safe. In a recession, it *spikes* — the same borrower now looks much riskier because the environment is hostile. PIT measures are the natural input for things that should react to current conditions: the loss provisions under IFRS 9 and CECL (which are explicitly forward-looking and point-in-time), and risk-based loan pricing today.

The two measures diverge most exactly when it matters most: in the turn of the cycle. In a boom, a PIT PD can drift far below the TTC average, making loans look cheap and capital look ample. A bank that mistakes the low PIT reading for the *true* long-run risk will under-price loans and under-provision — and then get blindsided when the cycle turns and PIT PDs snap back up toward (and past) the TTC level.

#### Worked example: the same loan in a boom and a bust

Make it concrete. Take a portfolio of \$100 million of loans with a senior-unsecured LGD of 45%. Suppose the *through-the-cycle* PD for these borrowers is **2%** — that's the honest long-run average.

In the middle of a **boom**, the *point-in-time* PD has drifted down to **0.8%**, because defaults are currently rare. A bank using the PIT number for pricing and provisioning would set expected loss at:

$$EL_{boom} = 0.008 \times 0.45 \times \$100{,}000{,}000 = \$360{,}000$$

In a **recession**, the PIT PD spikes to **6%** as the environment turns hostile. The same portfolio now implies:

$$EL_{bust} = 0.06 \times 0.45 \times \$100{,}000{,}000 = \$2{,}700{,}000$$

The expected loss has gone from \$360,000 to \$2,700,000 — a **7.5× swing** — without a single new loan being made. The TTC view, meanwhile, sat steady at $0.02 \times 0.45 \times \$100{,}000{,}000 = \$900{,}000$ the entire time. The bank that priced and reserved off the \$360,000 boom-time PIT number was \$540,000 a year *under* its own long-run average, and a full \$2.34 million short of what the bad year demanded.

The intuition: **a point-in-time measure tells you the truth about today and lies about the future; a through-the-cycle measure tells you the durable truth but hides today's stress.** A well-run bank uses both — PIT to provision for the loss that's actually coming, TTC to hold capital that doesn't evaporate the moment the cycle turns. The bank that uses only the cheerful boom-time PIT number is, without knowing it, running on far less margin than it thinks.

## The credit cycle: why losses arrive in waves

Everything above implies that credit losses are not smooth. They come in *waves* — long stretches of quiet punctuated by sudden, severe clusters. This is the credit cycle, and it is the master fact that makes credit risk so dangerous and so easy to mismanage.

![US bank failures per year showing the 2009 to 2010 crisis wave](/imgs/blogs/credit-risk-management-pd-lgd-ead-and-expected-loss-8.png)

This chart shows US bank failures per year — a downstream consequence of credit losses, since banks fail when loan losses overwhelm their capital. The pattern is the whole story of the credit cycle. From 2005 to 2007, *zero to three* banks failed a year — the calm before the storm, the years when PIT default rates were at historic lows and everyone felt safe. Then the wave: **25 failures in 2008, 140 in 2009, and 157 in 2010** as the financial crisis worked through the system. Then the slow normalization back to a handful a year. And in 2023, a small new spike of 5 failures — Silicon Valley Bank, Signature, First Republic — though notably that episode was driven more by interest-rate and liquidity risk than by classic credit losses (a story told in this series' other risk posts).

The lesson is brutal and counterintuitive: **the safest-looking years are when the most risk is being built.** In 2005–2007, lending standards were loosening, spreads were thin, and PIT default rates were near zero — which is *exactly* when banks were writing the loans that would default en masse two years later. Credit risk is a phenomenon that hides during the boom and reveals itself all at once in the bust. A risk manager who feels comfortable because losses are low is often the one who should be most worried.

This wave behavior is why expected loss alone is so misleading. If you averaged losses over 2005–2010, you'd get a moderate number. But no bank experiences "the average" — it experiences the quiet years, banks the profits, gets comfortable, grows the book, and then takes the entire decade's losses in eighteen months. The capital cushion has to be sized for the *peak* of the wave, not the average of the cycle.

There is a behavioral engine driving the wave, and it's worth naming because it recurs in every cycle. In the quiet years, a bank that has been earning more in spread than it loses in defaults looks brilliant — its return on equity is high, its stock rises, its executives are praised. The competitive pressure that follows is lethal: rivals see the profits and chase the same borrowers, spreads compress, and the only way to keep growing earnings is to lower standards — lend to weaker borrowers, accept thinner collateral, waive covenants. Each individual loosening looks reasonable against the backdrop of low realized defaults. By the late boom, the whole industry is making loans it would have rejected five years earlier, at spreads too thin to cover the *true* through-the-cycle expected loss. The economist Hyman Minsky described this as the migration from "hedge finance" to "speculative" to "Ponzi" finance, but you don't need the framework to see it in the failure chart: the years of fewest failures are the years the seeds of the next 157-failure wave are being planted. A risk function that can't push back against its own bank's growth during the boom is, in practice, no risk function at all.

## Portfolio credit risk: when correlation eats your diversification

So far we've treated loans one at a time. But a bank holds thousands or millions of them, and the risk of the *portfolio* is not the sum of the individual risks — it's governed by how much the loans default *together*. This is where the most dangerous misconceptions live.

The intuition starts well. If you make a thousand loans and each defaults *independently* — one borrower's failure tells you nothing about another's — then the law of large numbers is your friend. Some default, some don't, and the total lands reliably near the expected loss. Diversification works: spreading across many independent borrowers makes the portfolio loss *predictable*, which shrinks unexpected loss and the capital you need.

The catch is that loans are *not* independent. Borrowers share an economy. When a recession hits, it raises the default probability of *every* borrower at once — and worse, it raises them in a correlated way, so they tend to fail together. This correlation is the single most important and most underestimated quantity in portfolio credit risk. Even a *small* positive correlation between defaults dramatically fattens the tail of the loss distribution, because it means the bad years aren't just "more defaults" — they're "everything goes wrong simultaneously."

It's worth dwelling on just how powerful correlation is, because the result is genuinely surprising. Take two portfolios with *identical* expected loss. Portfolio A holds a thousand loans whose defaults are perfectly independent; Portfolio B holds a thousand loans whose defaults are highly correlated — when one goes, they tend to all go. In an average year, both lose the same amount. But the *distribution* of losses is wildly different. Portfolio A's losses pile up tightly around the average: the law of large numbers smooths out the individual defaults, so a year far above average is almost impossible. Portfolio B has no such smoothing — because the loans move together, the whole book is effectively one big bet, and a bad draw of the common factor sends losses to the moon. Portfolio A might need a thin sliver of capital to reach the 99.9th percentile; Portfolio B might need ten times as much for the *same expected loss*. The expected loss told you nothing about which portfolio was dangerous. Only the correlation did. This is the precise mathematical reason that a bank's capital requirement is dominated by an assumption — the default correlation — that doesn't even appear in the EL = PD × LGD × EAD equation. The headline number and the survival number are governed by different inputs.

#### Worked example: a portfolio's expected loss across grades

Let's build a small portfolio and compute its expected loss, then talk about what the number does and doesn't tell you. Suppose a bank holds **four loans of \$1,000,000 each** (\$4 million total), all senior unsecured (**LGD 45%**), but spread across credit grades:

| Loan | Rating | PD | EL = PD × LGD × EAD |
|---|---|---|---|
| 1 | A | 0.06% | 0.0006 × 0.45 × \$1,000,000 = **\$270** |
| 2 | BBB | 0.24% | 0.0024 × 0.45 × \$1,000,000 = **\$1,080** |
| 3 | BB | 1.20% | 0.0120 × 0.45 × \$1,000,000 = **\$5,400** |
| 4 | B | 5.50% | 0.0550 × 0.45 × \$1,000,000 = **\$24,750** |

Total expected loss = \$270 + \$1,080 + \$5,400 + \$24,750 = **\$31,500 per year**, or **0.79%** of the \$4 million book. Notice how lopsided the contribution is: the single B-rated loan accounts for \$24,750 — **79% of the portfolio's expected loss** — despite being only a quarter of the dollars. Risk in a credit portfolio is almost always concentrated in a handful of the weakest names.

But here's the crucial limitation: that \$31,500 is the *expected* loss, the average year. It tells you nothing about the *bad* year, and the bad year depends entirely on correlation. If these four borrowers are in four unrelated industries and regions, their defaults are nearly independent, the portfolio loss stays close to \$31,500, and the bank needs only a modest capital buffer. But if all four are, say, regional construction firms that all sink together when the property market turns, then in a bad year you could lose *all four at once* — \$1.8 million — sixty times the expected loss. Same expected loss; wildly different capital requirement. The intuition: **expected loss is set by the loans; unexpected loss is set by their correlation — and concentration is just correlation by another name.**

### Concentration: the silent killer

This is why **concentration risk** is the thing credit-risk managers lose sleep over. A portfolio can have a perfectly reasonable expected loss and still be one shock from death if it's concentrated — too much in one industry, one region, one large borrower, one collateral type. The classic failures in this series are concentration stories: a bank that lent heavily to commercial real estate, or to oil and gas, or to one giant client, looks fine in the average year and dies when *that one thing* turns. Diversification across genuinely *different* risk drivers — not just a larger number of similar loans — is the only real defense, and it directly shrinks the unexpected-loss tail that capital has to cover. (The [risk-management series](/blog/trading/risk-management/marginal-and-component-var-where-the-risk-actually-lives) digs into how to measure where the risk actually concentrates.)

## Common misconceptions

**"If a loan has a low default probability, it's a low-risk loan."** Not necessarily — you've only looked at one of three factors. A loan with a 0.5% PD but a 90% LGD (no collateral, deeply subordinated) and a large exposure can carry more expected loss than a 2% PD loan that's fully secured. PD is the term people fixate on because it's the most uncertain, but EL = PD × LGD × EAD, and a low PD multiplied by a high LGD and a big EAD is not a low number. Risk is the product, not any single factor.

**"Expected loss is the loss the bank should plan to survive."** No — expected loss is the loss the bank plans to *absorb in pricing*, as an ordinary cost. The loss it must *survive* is the unexpected loss, the bad-year deviation, which can be three to five times the expected loss for a typical portfolio. A bank that holds capital equal to its expected loss is catastrophically under-capitalized. Capital exists for the surprise, and the surprise dwarfs the average. Confusing these two is the single most common conceptual error in thinking about bank safety.

**"Low default rates mean the bank is well-run."** Often the opposite, if you're at the wrong point in the cycle. The years with the lowest realized defaults — 2005 to 2007 — were precisely when banks were writing the loans that would default catastrophically in 2009. A point-in-time default rate near zero in a boom is a sign that lending standards have loosened, not that risk has disappeared. The honest measure is the through-the-cycle rate, which doesn't flatter you in the good years.

**"Diversification eliminates credit risk."** It reduces *idiosyncratic* risk — one borrower's bad luck — but it does nothing against *systematic* risk, the recession that hits everyone at once. Spreading across a thousand mortgages protects you if one borrower loses their job; it does not protect you if a housing crash pushes all thousand toward default simultaneously. The portion of credit risk that diversification cannot remove — the correlated, all-at-once part — is exactly the part that requires capital. More loans of the same kind is *concentration*, not diversification.

**"Recovering collateral makes the bank whole."** Rarely. LGD figures already net out recoveries, and even a "low" LGD of 25% means the bank lost a quarter of the loan after all the work of seizing and selling. Worse, collateral is worth least in a downturn — the same moment the bank is forced to sell the most of it. The recovery process is slow, costs money in legal fees and lost time-value, and produces fire-sale prices. Collateral cushions the loss; it does not erase it.

## How it shows up in real banks

**The 2007 calm before the 2009 storm.** The cleanest illustration of point-in-time PD's danger is the run-up to the financial crisis. From 2005 to 2007, realized default rates on mortgages and corporate loans were near historic lows, so banks' point-in-time risk models showed low PDs, low expected losses, and ample capital. Loss provisions across the industry hit multi-decade lows. Then the cycle turned. As the FDIC failure chart shows, bank failures went from 3 in 2007 to 140 in 2009 to 157 in 2010 — a more than fiftyfold increase in two years. The loans hadn't changed; the *economy* they lived in had, and PIT PDs that had drifted to the floor in the boom snapped violently upward. Banks that had mistaken the boom-time PIT reading for the true risk had under-priced and under-reserved for years.

**Why high-yield bonds pay double-digit yields.** The expected-loss-by-grade chart isn't just theory — it's why a CCC-rated company has to offer investors 11% or 12% when a Treasury pays 4%. With a one-year PD around 26% and an LGD around 45%, the expected annual loss is roughly 11.7% of face value. A lender or bondholder who charged less than that spread would, on average, *lose money* across a portfolio of such credits. The eye-watering yields on distressed debt are not a free lunch; they are the market's honest price for an expected loss that really is that large. The [credit-spreads post in the fixed-income series](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) traces this from the bond-market side.

**IFRS 9 and CECL forced the PIT view into the accounts.** After the crisis, accounting regulators concluded that the old "incurred loss" model — only book a provision once a loss has actually happened — let banks under-reserve right up until disaster. The replacements, IFRS 9 (international) and CECL (US), require banks to book *expected* losses up front and to use *forward-looking, point-in-time* estimates that respond to the economic outlook. The effect is that provisions now jump the moment the outlook darkens, rather than waiting for defaults to materialize. The trade-off is volatility: provisions swing with the macro forecast, which is exactly the PIT behavior we described. This series' [collateral and provisioning post](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) covers the mechanics in depth.

**Concentration killed the savings-and-loan thrifts.** The S&L crisis of the late 1980s and early 1990s — over a thousand thrifts failed, at a taxpayer cost of roughly \$124 billion — was a concentration story as much as a rate story. Many thrifts were concentrated in commercial real estate and regional property markets; when those markets turned, the correlated defaults overwhelmed institutions whose loss models had assumed the loans were more independent than they were. Expected loss looked manageable; the correlated bad-year loss did not, and there wasn't enough capital to absorb it.

**The EL engine becomes the loan price.** The most concrete place the PD × LGD × EAD calculation touches a customer is the interest rate they're quoted. When a bank prices a loan, it stacks up several components: its own cost of funds (what it pays for the deposits and debt that fund the loan), the operating cost of servicing the loan, a charge for the capital it must hold against the loan's *unexpected* loss, a profit margin, and — sitting right in the middle — the expected loss. A borrower with a higher PD or a higher LGD literally pays a higher rate, because the EL term in the price is larger. This is why two companies borrowing the same amount on the same day can be quoted rates that differ by several percentage points: the bank has run the EL engine on each and is charging each one for its own expected loss. Banks formalize this with a metric called RAROC — risk-adjusted return on capital — which asks whether the spread, after subtracting expected loss and the cost of the capital tied up, still earns an acceptable return on that capital. A loan that doesn't clear the RAROC hurdle gets repriced or declined. The [loan-pricing post](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge) walks through the full stack.

**Wells Fargo and the limits of credit models.** Not every banking disaster is a credit-modeling failure — and that's a lesson too. The Wells Fargo fake-accounts scandal, which drew roughly \$5.5 billion in fines across 2016–2022, had nothing to do with PD, LGD, or EAD. It was an *operational* and *conduct* failure. The point: a bank with flawless credit-risk machinery can still be destroyed by the other risks in the taxonomy, which is why credit risk is only one of [the four risks every bank runs](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational). The EL engine is necessary, not sufficient.

## The takeaway: how to use this

Strip away the jargon and credit risk reduces to a single discipline: *turn the unknowable question "will I get paid back?" into three estimable numbers, multiply them, and then — crucially — remember that the product is only the average.*

Here is how to actually read a bank, or any lender, through this lens. First, **never judge a loan by its default probability alone.** PD is one of three factors, and a low PD attached to a high LGD and a large exposure is not a safe loan. The risk is the product. Second, **distinguish the price from the buffer.** Expected loss is what the bank charges and provisions for — an operating cost. The capital is for the *surprise*, the unexpected loss, and the surprise is several times the average. A bank that talks confidently about its low expected losses while holding thin capital has answered the easy question and ignored the hard one. Third, **be most suspicious when defaults are lowest.** A point-in-time default rate near zero is a feature of the late boom, not evidence of safety; it's the years of cheapest credit that breed the next wave of losses. The through-the-cycle number is the one that won't flatter you. And fourth, **worry about correlation more than count.** A thousand similar loans is concentration wearing a diversification costume. The risk that capital must absorb is the correlated, all-at-once risk — and that is governed by how *different* your borrowers truly are, not how many you have.

Bring it back to the spine of this series. A bank is a leveraged machine running on a thin equity cushion. Credit losses are the largest force pushing against that cushion, and they don't push steadily — they push in waves, far above the average in the bad years. The entire purpose of the PD × LGD × EAD engine, the expected-versus-unexpected split, and the through-the-cycle discipline is to make sure the cushion is sized for the *wave*, not the calm. The banks that fail are almost never the ones that mis-estimated the average. They're the ones that forgot the average was never the thing that was going to kill them.

## Further reading & cross-links

- [The four risks every bank runs: credit, market, liquidity, operational](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational) — where credit risk sits in the full taxonomy, and how the four risks compound.
- [Credit analysis: the five Cs and how a loan gets approved](/blog/trading/banking/credit-analysis-the-five-cs-and-how-a-loan-gets-approved) — the human-judgment side of estimating a borrower's PD before the model ever runs.
- [Collateral, security, and loan-loss provisioning: IFRS 9 and CECL](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) — how LGD and expected loss flow into the accounts and the reserves.
- [Loan pricing: cost of funds, risk premium, and the capital charge](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge) — how the expected-loss number becomes the spread a borrower actually pays.
- [Credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) — the same machinery from the bond market's perspective, where it sets credit spreads.

*This is an educational explanation of how banks measure credit risk, not financial or investment advice.*
