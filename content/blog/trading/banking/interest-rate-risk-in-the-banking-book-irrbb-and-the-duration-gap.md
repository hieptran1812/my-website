---
title: "Interest-Rate Risk in the Banking Book: IRRBB and the Duration Gap"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank's habit of lending long and borrowing short turns a move in interest rates into a profit, or a death sentence, and the gap measures that explain Silicon Valley Bank."
tags: ["banking", "interest-rate-risk", "irrbb", "duration", "duration-gap", "repricing-gap", "eve", "nii", "svb", "asset-liability-management"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank makes its money by lending long at a fixed rate and funding that with short, repriceable money; interest-rate risk in the banking book (IRRBB) is the danger that sits inside that one trade, and it is exactly what killed Silicon Valley Bank.
>
> - The **repricing gap** measures how much of the balance sheet reprices in each time bucket; a bank with more short funding than short assets is *liability-sensitive* and loses income when rates rise.
> - **Duration** measures how far a fixed cash flow's value falls when rates rise. A bank's **duration gap** — the duration of its assets minus the duration of its liabilities, scaled — tells you how much its *net worth* moves.
> - Banks watch two numbers: **NII sensitivity** (how much next year's profit moves) and **EVE sensitivity** (how much the whole balance sheet is worth today). SVB looked fine on the first and was already dead on the second.
> - The one number to remember: SVB carried about **\$17 billion** of unrealized bond losses against roughly **\$16 billion** of equity. The hole was bigger than the cushion before a single depositor moved.

On the afternoon of March 9, 2023, depositors tried to pull about **\$42 billion** out of Silicon Valley Bank in a single day. By the next morning another **\$100 billion** was queued to leave. The bank had **\$209 billion** in assets, **\$175 billion** in deposits, and — this is the part that matters — it had spent years pouring cheap deposit money into long-dated, fixed-rate bonds. When the Federal Reserve raised interest rates from near zero toward 5% over 2022 and early 2023, those bonds fell in value. By the end, the paper loss on SVB's securities was around **\$17 billion**, while the bank's entire equity cushion was only about **\$16 billion**.

Read that again. The loss was *larger than the equity* — before the run even started. The deposit flight didn't cause the insolvency; it merely forced the bank to stop pretending. SVB had taken a bet that almost every bank takes, just more aggressively and with less hedging: it locked its money into long fixed deals and funded them with money that could walk out the door at the first sign of trouble. That bet has a name. It is **interest-rate risk in the banking book**, and the figure below is the trap in one picture.

![Short repriceable funding against long fixed-rate assets, before and after a rate spike](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-1.png)

The diagram above is the mental model for the whole post: on the left, calm rates make the spread between long assets and cheap deposits look like free money; on the right, a rate spike crushes the value of the assets *and* forces the funding to reprice upward at the same time. Both sides of the balance sheet move against the bank at once. Everything that follows is just the machinery for measuring how badly, and how soon.

This is the third leg of a simple idea that runs through this whole series: **a bank is a leveraged, confidence-funded maturity-transformation machine.** It borrows short and lends long, earns the spread, and survives only as long as depositors trust it and its thin equity absorbs losses faster than they arrive. Interest-rate risk is the precise mechanism by which a rate move can blow a hole in that thin equity — quietly, on paper, for years before anyone is forced to admit it.

## Foundations: maturity transformation, rates, and the words you need

Let me build every term from zero, because the whole subject is a small number of ideas stacked on top of each other.

A **bank** takes in money it owes back to other people — mostly **deposits**, the balances in your checking and savings accounts — and uses that money to make **loans** and buy **securities** (tradeable IOUs like government and corporate bonds). The deposits and other borrowings are the bank's **liabilities** (what it owes); the loans and securities are its **assets** (what it owns). The sliver in between, the difference between what it owns and what it owes, is its **equity** or **capital** — the owners' stake and the cushion that absorbs losses.

The core trade is **maturity transformation**: the bank borrows short and lends long. Your deposit can be withdrawn today; the 30-year mortgage the bank funded with it will not be repaid for decades. The bank pockets the difference between the low rate it pays on short money and the higher rate it earns on long money. That difference, expressed as a percentage of the bank's interest-earning assets, is the **net interest margin** (NIM), and the dollars it produces are **net interest income** (NII) — the engine of bank profit. (We cover that engine in [net interest margin and the spread business](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained).)

An **interest rate** is the price of borrowing money for a period of time, quoted per year. When people say "rates went up 1%," they usually mean a **basis point** move — a *basis point* is one hundredth of a percentage point, so 0.01%, and 1% is 100 basis points (bps). Central banks like the Fed set a very short-term rate; the rates on everything longer are set by the market, and they move around a lot.

Here is the whole problem in an everyday example before any formula. Say you run a small pawnshop that does one clever thing: it borrows money from neighbors who can ask for it back any day, paying them a tiny 1% a year, and it lends that same money out as five-year loans at 4%. Every year you pocket the 3% difference, and life is good. Now interest rates in the town jump: the bank down the street starts offering your neighbors 5% on their savings. Two bad things happen at once. First, your neighbors want 5% from you too, or they take their money back — but you are still only earning 4% on the loans you already made, so your 3% profit just turned into a 1% *loss* on every dollar. Second, if you wanted to sell those old 4% loans to someone else to raise cash, no one will pay full price for a 4% loan when they can make a new 5% one — so the loans are worth less than you paid. That is interest-rate risk in miniature: the squeeze on your profit, and the fall in the value of what you own, both triggered by the same move in rates, both made unsurvivable by the fact that your funding can leave.

Now the heart of it. **Interest-rate risk** is the danger that a change in interest rates hurts the bank — either by squeezing the profit it earns, or by lowering the value of what it owns. **Interest-rate risk in the banking book** (IRRBB) is the regulator's name for this risk as it lives in the *banking book*: the loans, deposits, and bonds a bank holds to earn the spread and collect to maturity, as opposed to the *trading book* of positions it buys and sells for short-term gain. IRRBB is the risk that hides in the ordinary, boring business of being a bank.

There are two distinct ways a rate move bites, and confusing them is the single most expensive mistake in this whole field:

1. **The repricing channel.** When rates change, the rates on the bank's assets and liabilities reset (*reprice*) on different schedules. A floating-rate loan reprices next month; a fixed-rate bond doesn't reprice for years; a checking deposit can be repriced tomorrow. The mismatch in *when* things reprice changes the bank's **net interest income** going forward. The tool for measuring this is the **repricing gap**.

2. **The valuation channel.** A fixed stream of future cash — a bond's coupons, a fixed-rate loan's payments — is worth more when rates are low and less when rates are high, because you discount those future dollars at the prevailing rate. When rates rise, the *present value* of long fixed assets falls. The tool for measuring this is **duration**, and the bank-wide version is the **duration gap** and the **economic value of equity** (EVE).

The first channel is about *flow* — next year's earnings. The second is about *stock* — what the whole balance sheet is worth today. SVB's tragedy is that it was watching the flow while the stock was already gone. Let's take each channel apart.

## The repricing gap: when does each side reset?

Start with the simpler channel. Forget valuation for a moment and ask a flow question: if rates jump 1% tomorrow and stay there, how does the bank's net interest income change over the next year?

The answer depends on *which side reprices first*. Group every asset and every liability into time buckets by when its rate next resets — overnight to 3 months, 3 to 12 months, 1 to 3 years, and so on. In each bucket, subtract rate-sensitive liabilities from rate-sensitive assets. That difference is the **repricing gap** (also called the **maturity gap** or **funding gap**) for that bucket.

- A **positive gap** in a bucket means more assets than liabilities reprice in that window. If rates rise, more of your *income* resets upward than your *cost*, so your margin widens. You are **asset-sensitive**.
- A **negative gap** means more liabilities reprice than assets. If rates rise, your *cost* resets upward faster than your *income*, so your margin shrinks. You are **liability-sensitive**.

Most retail-funded banks are liability-sensitive in the very short buckets, because deposits are technically repriceable overnight while loans and bonds are locked in for years. The figure below shows the shape: assets pile up in the long buckets, liabilities pile up in the short buckets, and the early gap is negative.

![Repricing gap by time bucket showing assets and liabilities and the gap line](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-2.png)

Notice the red gap line dips below zero in the first two buckets and turns positive only in the long buckets. That negative early gap is the classic bank position: short, runnable funding against long, sticky assets. It is profitable in normal times and dangerous when rates spike.

#### Worked example: a repricing gap into an NII change

Suppose a bank has a **negative one-year repricing gap of \$10 billion** — that is, over the next twelve months, \$10 billion more of its liabilities reprice than its assets. Now rates rise by **200 basis points** (2%) across the board.

The first-order estimate of the change in net interest income is just the gap times the rate move times the fraction of the year:

$$\Delta \text{NII} \approx \text{Gap} \times \Delta r \times t$$

where $\Delta\text{NII}$ is the change in net interest income, $\text{Gap}$ is the repricing gap (signed), $\Delta r$ is the rate change, and $t$ is the fraction of a year the gap is in effect. Over a full year ($t = 1$):

$$\Delta \text{NII} \approx (-\$10\text{bn}) \times 2\% \times 1 = -\$200 \text{ million}$$

The bank loses **\$200 million** of net interest income over the year, because \$10 billion of its funding now costs 2% more while the matching assets are still earning the old, lower rate. If the same bank had a *positive* \$10 billion gap, the same rate rise would *add* \$200 million. The figure below shows the symmetric pain-and-gain across a range of shocks.

![One year net interest income change by rate shock for a liability-sensitive bank](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-7.png)

The intuition: the repricing gap tells you, in dollars, how much of next year's profit is exposed to a rate move — and which way you are pointed.

### The deposit-beta wrinkle: deposits don't reprice the way the contract says

There is a subtlety in the repricing gap that wrecks naive calculations, and it is the wrinkle that does most of the damage in practice. On paper, a checking or savings deposit can be repriced overnight — it belongs in the shortest bucket. But banks rarely pass the *full* rate move through to depositors immediately. The fraction of a rate move a bank passes on to its deposit rate is called the **deposit beta**. A deposit beta of 0.3 means that when the Fed raises rates 1%, the bank raises its deposit rate only 0.3%.

A low deposit beta is the bank's friend on the way up: it means funding costs lag, so margins widen for a while even for a liability-sensitive bank. This is precisely why net interest income can look healthy in the first year of a hiking cycle. But deposit beta is not fixed — it *climbs* as the cycle goes on and as depositors notice they can earn more elsewhere. Through the 2022-2023 hiking cycle, the industry's cumulative deposit beta marched from about 0.10 in mid-2022 to roughly 0.50 by late 2023 — meaning that by the end, banks were passing on half of every hike. The repricing gap you compute on day one understates your eventual pain, because it assumes a deposit beta that rises against you over time.

There is a darker version of this. When a bank assumes its deposits are "core" — sticky, low-beta, long-effective-life — it is making a *behavioral* bet that lets it carry a bigger duration gap. The bet is fine until depositors behave differently, at which point the deposit that you modeled as 5-year funding turns out to be overnight funding that demands a market rate or leaves. The repricing gap is only as honest as the deposit assumptions feeding it.

### Why the repricing gap is necessary but not sufficient

The repricing gap is the oldest tool in the asset-liability manager's kit, and it has a fatal limitation: it only looks at *when the coupon resets*, not at *how much the value of the principal changes in the meantime*. A 10-year fixed bond and a 10-year fixed loan both sit in the "5y+" bucket and contribute the same to the gap. But the bond can be sold tomorrow at a price that has already collapsed, while the loan sits on the books at face value. The gap measures earnings risk; it is blind to the valuation bomb.

The gap also flattens a curve into a single number. A real rate move is rarely a clean parallel shift where every maturity moves the same amount; the **yield curve** — the line of rates plotted against maturity — can steepen, flatten, or invert. A bank can be hedged against a parallel shift and still bleed from a twist in the curve, because its assets cluster at one maturity and its funding at another. Regulators address this by requiring banks to test several prescribed shocks — parallel up, parallel down, steepener, flattener, short-rate up, short-rate down — not just a single number. But the deeper limitation remains: the gap is an earnings tool, and earnings are the slow lens. For the fast, honest lens, we need duration.

## Duration: how far value falls when rates rise

Here is the most important idea in fixed income, built from scratch. A **bond** is a promise to pay you a fixed set of dollars on fixed future dates: a yearly **coupon** (the interest) plus the **face value** (the principal) at the end. Because those dollars are fixed and in the future, their value *today* depends on what rate you discount them at. Discounting is just the reverse of earning interest: a dollar a year from now is worth less than a dollar today, and worth *even less* when prevailing rates are high, because you could have earned that high rate instead.

So when market rates rise, the fixed future dollars of a bond get discounted harder, and the bond's price falls. When rates fall, the price rises. Price and yield move in opposite directions — the seesaw at the heart of bonds, which the fixed-income series covers in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much).

**Duration** answers the quantitative question: *by how much?* Duration is, roughly, the percentage fall in a bond's price for each 1% rise in its yield. A bond with a duration of 9 loses about 9% of its value when its yield rises 1%, and about 18% when its yield rises 2%. (More precisely, *modified duration* is the price sensitivity per unit yield; *Macaulay duration* is the weighted-average time you wait for your money, measured in years. They are nearly equal at low rates, and people use "duration" loosely for both.) The longer the bond and the lower its coupon, the higher its duration, because more of your money is locked up further out where discounting bites hardest.

![A 10-year bond price falling as its yield rises from low to high](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-3.png)

The curve above is computed, not stylized: it is the actual price of a 10-year, 1.6%-coupon bond at every yield from 0.5% to 7.5%. Two things to notice. First, it slopes down — higher yield, lower price. Second, it is *curved* (convex): the price falls fast at first and then more slowly, which is why duration is only a first approximation and why the fixed-income series has a whole post on [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story). For a bank, the curve is the valuation bomb in one line: a portfolio of these bonds loses real, mark-to-market value the moment rates climb.

#### Worked example: a bond's price fall when rates rise

You buy a **10-year bond** with a **1.6% annual coupon** at **par** — \$100 of price for \$100 of face value — when yields are 1.6%. (This is roughly the world SVB was buying into: long Treasuries and mortgage bonds at sub-2% yields in 2020 and 2021.) Each year it pays you \$1.60, and at the end it returns your \$100.

Now suppose the market yield for that bond rises to **5%**. To price the bond, you discount every future payment at 5%:

$$P = \sum_{t=1}^{10} \frac{\$1.60}{(1.05)^t} + \frac{\$100}{(1.05)^{10}}$$

Working that out, the ten coupons are worth about \$12.36 in today's dollars, and the \$100 principal arriving in ten years is worth about \$61.39. So:

$$P \approx \$12.36 + \$61.39 = \$73.75$$

Your \$100 bond is now worth **\$73.75** — a fall of about **26%** — even though it will still pay every promised dollar. Nothing defaulted. The bond is simply worth less today because the dollars it pays are no longer competitive with new bonds paying 5%. The bond's modified duration here is about **9.2**, so the quick estimate ($9.2 \times 3.4\% \approx 31\%$) overshoots the true 26% — that overshoot is convexity working in the holder's favor, but it is cold comfort when you are down a quarter of your money.

The intuition: a fixed-rate bond's *credit* can be perfect and its *value* can still crater, purely because rates moved. That is the loss that lives in the banking book.

### What makes a bond's duration long: coupon, maturity, and convexity

Two bonds of the same maturity can have very different durations, and the bank's choices about *which* bonds to buy directly set how exposed it is. Three forces drive duration:

- **Maturity.** Longer bonds have higher duration, because more of your money is tied up further out where the discounting of a rate change compounds. A 30-year bond is far more rate-sensitive than a 2-year bond.
- **Coupon.** Lower-coupon bonds have higher duration. A low coupon means you get less money back early and more of your value sits in the final principal payment, which is the most rate-sensitive part. The bonds banks bought in 2020-2021 were a worst case on both counts: long maturity *and* tiny coupons, because that is what the zero-rate world offered. Long and low-coupon is the highest-duration combination there is.
- **Convexity.** The price-yield relationship is curved, not straight, so duration itself changes as rates move. **Convexity** measures that curvature. For a plain bond, convexity is the holder's friend — it means losses are slightly smaller, and gains slightly larger, than duration alone predicts (which is why our worked example's bond fell 26% rather than the 31% the duration estimate suggested). But for **mortgage-backed securities** — bonds backed by home loans, a huge slice of many banks' books — convexity can turn *negative*: when rates rise, homeowners stop prepaying their mortgages, so the bonds' effective maturity *extends* right when you least want it, deepening the loss. SVB held a large book of exactly these. The fixed-income series digs into this in [mortgage-backed securities and negative convexity](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity).

The practical upshot: a treasurer who wants to dial down rate risk doesn't have to sell everything. Shifting toward shorter and higher-coupon assets pulls the average duration down. SVB did the opposite — it reached for yield by buying long and low, which is the same as reaching for duration.

### From one bond to the whole bank: the duration gap

A bank doesn't hold one bond; it holds a portfolio of assets with some average duration, funded by liabilities with some (usually much shorter) average duration. The **duration gap** combines them into a single number that tells you how the bank's *net worth* moves when rates change.

The formula looks intimidating but says something simple:

$$D_{\text{gap}} = D_A - \left(\frac{L}{A}\right) D_L$$

where $D_A$ is the duration of assets, $D_L$ is the duration of liabilities, $A$ is total assets, $L$ is total liabilities, and $L/A$ scales the liability duration down because a bank has fewer liabilities than assets (the difference is equity). A **positive duration gap** means assets are "longer" than liabilities, so a rate rise hurts the value of assets more than it helps by shrinking the value of liabilities — net worth falls. Almost every traditional bank has a positive duration gap, because its assets (long loans and bonds) are far longer than its liabilities (short deposits).

The change in the bank's economic value of equity is then:

$$\Delta \text{EVE} \approx -D_{\text{gap}} \times A \times \Delta r$$

This is the valuation channel made bank-wide. Hold this thought — it is the number SVB ignored.

#### Worked example: the duration of equity

Take a bank with **\$200 billion** of assets at an average duration of **5 years**, and **\$184 billion** of liabilities at an average duration of **1 year**. Its equity is \$200bn − \$184bn = **\$16 billion** (an 8% capital ratio, the realistic base case for a commercial bank, giving about 12.5× leverage). The duration gap:

$$D_{\text{gap}} = 5 - \left(\frac{\$184}{\$200}\right) \times 1 = 5 - 0.92 = 4.08 \text{ years}$$

Now rates rise **200 basis points**. The change in economic value of equity:

$$\Delta \text{EVE} \approx -4.08 \times \$200\text{bn} \times 2\% = -\$16.3 \text{ billion}$$

The bank just lost **\$16.3 billion** of economic value — slightly *more than its entire \$16 billion of equity*. On an economic basis, a 2% rate rise wiped this bank out, even though every loan is performing and no depositor has moved. This is the **duration of equity**: because equity is the thin difference between two large, rate-sensitive numbers, its own effective duration is enormous. Here the equity's duration is roughly $D_{\text{gap}} \times A / E = 4.08 \times 200 / 16 \approx 51$ years. A 51-year duration on the owners' stake is a terrifying number, and it is hiding inside an ordinary-looking 8%-capitalized bank.

The intuition: leverage doesn't just multiply credit losses, it multiplies *rate* losses. A modest duration gap on the assets becomes a catastrophic duration on the sliver of equity underneath them.

## EVE vs NII: the two lenses, and why you need both

We now have the two channels formalized. **NII sensitivity** measures how much the bank's net interest *income* changes over a short horizon (usually one to two years) for a given rate shock — the flow, driven by the repricing gap. **EVE sensitivity** (economic value of equity sensitivity) measures how much the present value of the bank's net worth changes *today* for a given rate shock — the stock, driven by the duration gap.

They can tell completely different stories about the same bank, and the matrix below lays out what each one actually measures.

![Comparison of EVE and NII sensitivity across what they measure horizon driver and the SVB blind spot](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-6.png)

The crucial row is the last one. A bank can look comfortable on NII — its near-term earnings hold up because much of its asset book hasn't repriced yet and its deposit costs haven't fully caught up — while its EVE has been annihilated, because the *value* of those long, low-yielding assets has collapsed. The income looks fine precisely *because* the assets are stuck at old, low rates; the same stuck-ness is what destroyed their value. NII can lull you while EVE screams.

#### Worked example: a bank that passes NII and fails EVE

Take a bank holding **\$90 billion** of 6-year-duration bonds yielding **1.6%**, funded by deposits costing **0.2%**. Rates rise 300 bps.

*The NII view, year one.* The bonds keep paying 1.6% — they don't reprice. Deposit costs creep up, but slowly: in the first year the bank passes through only part of the hike (a deposit beta of, say, 0.3, so deposit cost rises about 0.9% to ~1.1%). Net interest income is *squeezed* but still positive: the bank earns 1.6% and pays ~1.1%, a 0.5% margin. On the NII dashboard, the bank is bruised, not broken. Management reports "manageable" earnings sensitivity.

*The EVE view, today.* The same bonds, at 6-year duration, fall about $6 \times 3\% = 18\%$ in value — roughly **\$16 billion** of lost value on a \$90 billion book. If the bank's equity is \$12-16 billion, that single mark wipes out most or all of it. On the EVE dashboard, the bank is already insolvent.

Same bank, same day, two numbers, opposite verdicts. The intuition: NII asks "can I still earn a margin next year?" and EVE asks "what is this balance sheet actually worth if I had to value it honestly today?" A bank that watches only the first is flying blind into exactly the cliff SVB drove off.

### The accounting trick that hid the loss: AFS vs HTM

There is one more piece that explains *why* the EVE loss could stay invisible on the official books, and it is a quirk of accounting that turned a measurement problem into a deception. Banks classify their bonds into two buckets:

- **Available for sale** (AFS): bonds the bank might sell. These are marked to market — their losses flow through to a line in equity (other comprehensive income), so a falling bond price visibly dents the reported capital ratio.
- **Held to maturity** (HTM): bonds the bank declares it intends to hold to the end. These are carried at **amortized cost** — essentially the price paid — *regardless of their current market value*. An HTM bond that has lost 20% of its value still sits on the balance sheet at par. The loss exists, but it is disclosed only in a footnote, not in the headline capital.

You can see the temptation. A bank watching its reported capital ratio shrink as rates rise can stop the bleeding *on paper* by reclassifying bonds from AFS to HTM, which freezes them at cost and makes the unrealized loss disappear from the headline numbers. SVB carried roughly **\$91 billion** in its HTM book, where the loss didn't dent reported capital. The catch is that HTM accounting is a promise, not a shield: the day the bank is forced to sell *any* HTM security to raise cash, the accounting privilege can collapse and the losses come roaring back onto the books. The accounting let SVB *report* solvency long after EVE said it was gone. The footnote was telling the truth; the front page was not. (The accounting for how losses hit a bank's books is covered in [collateral, security, and loan-loss provisioning](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl).)

## The duration trap: how a paper loss becomes a real one

Here is the subtle part that confuses even smart observers. If SVB intended to *hold its bonds to maturity*, why did the \$17 billion paper loss matter? The bonds would eventually pay back par; the loss was unrealized, an accounting shadow.

The answer is the **duration trap**, and it is the whole reason interest-rate risk in the banking book is lethal rather than merely uncomfortable. The trap is the interaction between two things a bank does at once: it locks money into long assets, *and* it funds them with money that can leave. Each alone is survivable. Together they form a noose that a rate spike pulls tight.

![Pipeline of a rate rise turning a bond book into insolvency step by step](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-5.png)

Walk the chain in the figure. Rates rise. Bond prices fall, creating a large unrealized loss. So far, only paper. But the rate rise also makes the bank's cheap deposits look unattractive — depositors can now earn 5% in a money-market fund instead of 0.1% at the bank — so they start to leave. To pay the leaving depositors, the bank needs cash *now*. Its long bonds are the only liquid thing it has, so it is **forced to sell them** at the collapsed price. The moment it sells, the paper loss becomes a **realized loss** that hits capital. And because the realized loss is larger than the thin equity, the bank is **insolvent**.

The "hold to maturity" defense fails because *you only get to hold to maturity if your funding holds too.* Long assets are a promise you can keep only if your depositors keep theirs. When they don't, duration that you thought was a paper risk becomes a solvency risk overnight. This is why interest-rate risk and **liquidity risk** are not two separate problems for a maturity-transforming bank — they are the same problem viewed from two sides. The duration gap sets the size of the loss; the deposit run sets the timing of the reckoning.

### Why the run is rational, not panic

It is tempting to dismiss a bank run as irrational mass psychology, but the duration trap makes it coldly rational for an uninsured depositor. Consider the arithmetic from the depositor's seat. If you suspect the bank's bond losses exceed its equity, you know the bank is economically insolvent. The first depositors to withdraw get paid in full at par; the bank sells its most liquid assets to cover them. The depositors who wait get paid out of whatever is left after the fire-sale losses are realized — which may be less than 100 cents on the dollar, and may take months in a receivership. There is no reward for loyalty and a real penalty for slowness. So the dominant strategy for every uninsured depositor is to *run first*, and because everyone reasons identically, the run is instantaneous.

This is why the deposit-insurance limit matters so much. A depositor with a balance under the FDIC's **\$250,000** limit is made whole regardless of the bank's solvency, so they have no reason to run. A depositor with \$20 million has every reason. SVB's **94%-uninsured** deposit base was a tinderbox: almost every dollar belonged to someone with both the incentive and, in the smartphone era, the technical ability to move it in minutes. The duration gap loaded the gun; the uninsured, concentrated deposit base put a hair trigger on it. A bank can carry the same duration risk safely if its funding is granular and insured, because then no one has a reason to be the first one out the door.

## Asset-sensitive vs liability-sensitive: who wins when rates move

Not every bank is hurt by rising rates. Whether a rate move helps or hurts depends entirely on which side of the balance sheet reprices first — and that is a *choice* the bank makes, consciously or not, when it sets the duration of its assets against its funding.

![Graph of asset sensitive and liability sensitive bank outcomes when rates rise or fall](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-8.png)

Trace the four outcomes:

- An **asset-sensitive** bank — assets reprice faster than funding, e.g. lots of floating-rate loans funded by sticky, slow-to-reprice deposits — *gains* when rates rise (its income resets up faster than its cost) and *loses* when rates fall. Many large diversified banks engineered themselves to be modestly asset-sensitive going into 2022 and printed record net interest income as the Fed hiked.
- A **liability-sensitive** bank — funding reprices faster than assets, e.g. long fixed bonds funded by runnable deposits — *loses* when rates rise and *gains* when rates fall. **This was SVB.** It had loaded up on long, low-yielding fixed bonds during the zero-rate years, funded by deposits it assumed were sticky. When rates rose, both the repricing channel (squeezed NII) and the valuation channel (collapsed EVE) turned against it simultaneously.

The position is not destiny — it is a steerable variable. A bank can shorten its asset duration, buy floating-rate assets, issue longer-term debt, or use **interest-rate swaps** (contracts that exchange a fixed rate for a floating one, synthetically shortening the duration of the bond book) to flatten its duration gap. SVB had hedges in 2021 and then, fatefully, let many of them roll off in 2022 to make its reported earnings look better in the short run — trading a smaller NII drag today for an uncapped EVE risk tomorrow. The trade-off between the two lenses is not academic; it is a steering decision a treasury makes every quarter, and getting it wrong is fatal.

## How a treasury actually hedges the gap

The job of managing all of this falls to the bank's **treasury** and its **asset-liability management** (ALM) function — the desk that watches the duration gap, the repricing buckets, and the EVE and NII sensitivities, and steers them within board-set limits. They have a toolkit, and understanding it tells you what a *well-run* bank does that SVB didn't.

**Receive-fixed vs pay-fixed swaps.** The workhorse hedge is the interest-rate swap. A bank with too-long assets (a positive duration gap) wants to shorten them synthetically, so it enters a **pay-fixed swap**: it agrees to pay a fixed rate and receive a floating rate on some notional amount. If rates rise, the floating leg it receives goes up, generating a gain that offsets the fall in its bond values. The swap is, in effect, a short-duration overlay glued on top of the long bond book. SVB had exactly these in 2021, covering a chunk of its securities, and then unwound most of them — removing the airbag right before the crash.

**Shortening the asset book.** A treasury can simply let long bonds mature and reinvest in shorter ones, or buy floating-rate loans whose coupons reset with the market. This costs current yield — short and floating assets pay less than long fixed ones in a normal upward-sloping curve — which is precisely the tension. Reaching for yield means reaching for duration; staying safe means giving up income today.

**Lengthening the funding.** The other side of the gap is the funding. A bank can issue long-term debt or term deposits (locked in for years) instead of relying on overnight deposits, which lengthens liability duration and shrinks the gap from below. This is more expensive than free checking balances, again the same trade-off.

#### Worked example: a swap that saves the bank

Return to the bank from the duration-of-equity example: \$200 billion of assets at 5-year duration, \$184 billion of liabilities at 1-year duration, \$16 billion of equity, and a duration gap of 4.08 years. We saw a 200 bps rate rise cost it \$16.3 billion of economic value — its whole cushion.

Now suppose the treasury hedges with **\$80 billion notional** of pay-fixed swaps, each behaving like a short position in a 5-year asset (a duration of about 4.5). The swap book gains when rates rise. The effective asset duration drops: the bank now has \$200bn of bonds at duration 5 *minus* a \$80bn hedge at duration 4.5, for a net rate exposure of roughly $5 - (80/200)\times 4.5 = 5 - 1.8 = 3.2$ years on the asset side. The new duration gap:

$$D_{\text{gap}} = 3.2 - \left(\frac{\$184}{\$200}\right) \times 1 = 3.2 - 0.92 = 2.28 \text{ years}$$

The same 200 bps shock now costs:

$$\Delta \text{EVE} \approx -2.28 \times \$200\text{bn} \times 2\% = -\$9.1 \text{ billion}$$

The loss falls from \$16.3 billion to **\$9.1 billion** — painful, but survivable against \$16 billion of equity. The hedge cost the bank some net interest income each quarter (the fixed leg it paid usually exceeded the floating leg it received while rates were low), which is exactly why a management team chasing this quarter's earnings is tempted to drop it. The intuition: a swap doesn't make the rate risk disappear, it *transfers* it from the bank to the swap counterparty for a running fee — and that fee is the insurance premium SVB chose to stop paying.

## Common misconceptions

**"If I hold the bond to maturity, the rate loss isn't real."** This is the single most dangerous belief in banking, and it is the one SVB's management appears to have held. The loss is unrealized *only as long as you are never forced to sell*. Your ability to hold is hostage to your funding. The moment depositors leave faster than your cash on hand can cover, you must sell into the loss, and the paper number becomes a capital number. Holding to maturity is a privilege your liability side grants you, not a right you control. Accounting can defer recognizing the loss; it cannot make the loss go away.

**"Net interest income looks healthy, so our rate risk is fine."** NII and EVE are different lenses, and NII is the slower, more flattering one. A bank stuffed with long low-yield assets can show resilient near-term earnings *because* those assets haven't repriced — the very fact that destroyed their present value. In the worked example above, a bank earning a 0.5% margin on paper had already lost 18% of its asset value. Watching NII without EVE is like checking your speed without checking the cliff edge.

**"Higher rates are good for banks."** Sometimes, for a while, for *some* banks. Rising rates widen margins for asset-sensitive banks and lift the return on their cash. But the same rise hammers the value of every long fixed asset on the books and tempts depositors to chase yield elsewhere, raising funding costs. The 2023 failures happened during a period of *rising* rates and record industry NIM. "Higher rates help banks" is a half-truth that ignores the duration of the assets and the runnability of the funding.

**"Deposits are stable, long-term funding."** On paper a checking deposit can be repriced or withdrawn overnight; in practice they have historically been remarkably sticky, and banks model them as "core" funding with a long effective life. That stickiness is a *behavioral assumption*, not a contractual fact — and assumptions about deposit behavior break exactly when you need them most. SVB's deposit base was 94% uninsured and concentrated in a tech-startup clientele that talked to each other constantly; when fear hit, "sticky" deposits became a 36-hour digital stampede. The repricing schedule of a deposit is not the contract; it is a guess about human behavior under stress.

**"Interest-rate risk is the trading desk's problem."** No — the most dangerous rate risk in 2023 lived in the *banking* book, in plain-vanilla held-to-maturity bonds, not in any exotic trading position. IRRBB is the boring risk, which is exactly why it is overlooked. The trading book is marked daily and watched obsessively; the banking book can carry a building unrealized loss for years with a footnote and a "hold to maturity" label.

**"A bank that's hedged is safe."** Hedges reduce rate risk; they don't make it vanish, and they can themselves create new problems. A swap that pays off when rates rise is an asset that may require the bank to *post collateral* if it temporarily moves against you, consuming cash at the worst moment. A hedge sized for a parallel shift can leave you exposed to a curve twist. And hedges cost running income, so there is always a manager somewhere arguing to trim them to flatter this quarter — which is precisely the decision that left SVB naked. "Hedged" is a spectrum and a moving target, not a binary safe/unsafe switch; the right question is *how much* of the gap is hedged, against *which* scenarios, and whether the hedge will still be there when rates actually move.

## How it shows up in real banks

### Silicon Valley Bank, 2023: the duration trap in full

SVB is the textbook case because it ran every part of this post simultaneously and to an extreme. By the numbers in our dataset: **\$209 billion** in total assets, **\$175 billion** in deposits, a held-to-maturity bond book of about **\$91 billion**, and an unrealized loss across its available-for-sale and held-to-maturity securities of roughly **\$17 billion** — against equity of about **\$16 billion**.

![SVB unrealized bond loss compared with its reported equity](/imgs/blogs/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap-4.png)

The chart makes the central fact unmissable: the loss bar is taller than the equity bar. SVB was *economically* insolvent on a mark-to-market basis before any run, purely from the duration channel. Here is how each piece of this post shows up:

- **The repricing gap.** SVB was profoundly liability-sensitive. Deposits — overnight-repriceable — funded a book of multi-year fixed bonds. When the Fed hiked, the funding side strained to reprice while the asset yields stayed locked at sub-2%.
- **Duration and the duration gap.** Its securities had an average duration in the ballpark of six years. A roughly 3% rise in yields therefore implied an ~18% fall in value — consistent with the ~\$17 billion loss on a book of that size. The duration gap was large and unhedged after SVB let interest-rate swaps roll off in 2022.
- **EVE vs NII.** SVB's near-term net interest income looked acceptable into 2022; its economic value of equity was already gone. It was watching, and reporting, the flattering lens.
- **The duration trap springs.** The hidden loss became visible when SVB announced it had sold \$21 billion of available-for-sale securities at a loss and needed to raise capital. That announcement was the spark: it told depositors the bond losses were real and the bank was short of cash. The **\$42 billion** withdrawal attempt on March 9 and the **\$100 billion** queued for March 10 forced exactly the fire-sale the bank had hoped to avoid, converting the paper loss into a realized one and the bank into a failure. Ninety-four percent of its deposits were above the FDIC's **\$250,000** insurance limit, so depositors had every incentive to run first and ask questions never.

The deeper lesson is in the sequencing. Interest-rate risk built the hole over two years, silently, in the most boring corner of the balance sheet. Liquidity risk merely chose the hour the hole was revealed. The two risks were one.

It is worth being precise about the timeline, because it shows how fast the conversion from paper loss to failure happens once the trap springs. On Wednesday, March 8, 2023, SVB disclosed it had sold a large block of available-for-sale securities at a loss and was trying to raise about \$2 billion of fresh capital to plug the gap. That disclosure did the bank in: it confirmed to a tightly-networked, mostly-uninsured depositor base that the bond losses were real and the bank was scrambling for cash. By Thursday, March 9, depositors and their venture-capital backers were telling each other to pull funds, and the **\$42 billion** withdrawal wave hit — about a quarter of all deposits in a single day, executed by app and wire, not by people lining up at a branch. With another **\$100 billion** queued for Friday morning, the bank could not possibly meet the demand, because its money was locked in long bonds that could only be sold at a deep loss. Regulators closed SVB on Friday, March 10. Two days, start to finish, for a risk that had been building for two years. That compression — slow build, instant collapse — is the signature of the duration trap. (The companion case study sits at [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

### The 2023 unrealized-loss backdrop: SVB was not alone

What made 2023 frightening was that SVB's core problem — large unrealized losses on long bonds bought during the zero-rate years — was *industry-wide*. As the Fed lifted rates by more than 5 percentage points across 2022 and 2023, the entire US banking system sat on hundreds of billions of dollars of paper losses on its securities portfolios, much of it parked in "held to maturity" buckets where it didn't dent reported capital. Industry net interest margin, meanwhile, actually *rose* — from a ZIRP-era trough of about 2.56% in 2021 to 3.04% in 2022 and 3.30% in 2023, per the FDIC's aggregate figures. The income lens looked great precisely while the value lens deteriorated.

The difference between SVB and the banks that survived was not the existence of the unrealized loss; almost everyone had one. It was the *combination*: SVB paired an unusually large, unusually long, unusually unhedged bond book with an unusually concentrated, unusually uninsured, unusually flighty deposit base. The duration gap was big and the funding was runnable. Banks with similar bond losses but stickier, more granular, more-insured deposits could hold to maturity, because their funding actually held. The episode is the cleanest natural experiment imaginable for this post's thesis: the same rate shock, the same paper losses, and the banks that died were the ones whose liability side couldn't keep its promise to the asset side.

### The Savings and Loan crisis: the same trap, an earlier decade

This is not new. The American Savings and Loan crisis of the 1980s and early 1990s was, at its core, a giant interest-rate-risk failure. Thrifts had made long-term fixed-rate mortgages at single-digit rates, funded by short-term deposits. When inflation forced rates into the high teens around 1980-1981, the thrifts' funding costs shot above the rate they earned on their old mortgages — a textbook liability-sensitive squeeze that turned their net interest margin sharply negative. On a mark-to-market basis, the industry was deeply underwater. More than a thousand thrifts failed between 1986 and 1995, and the cleanup cost taxpayers on the order of \$124 billion through the Resolution Trust Corporation. The instruments were different from SVB's — 30-year mortgages instead of Treasuries and mortgage bonds — but the trap was identical: long fixed assets, short repriceable funding, and a rate move that turned the spread negative.

### The other side of 2022: the asset-sensitive winners

For balance, look at who *thrived* in the same rate environment that killed SVB. The largest US banks went into the 2022 hiking cycle deliberately positioned to be modestly asset-sensitive: heavy on floating-rate commercial loans, holding huge piles of cash and reserves that reprice instantly, and funded by enormous, granular, sticky retail deposit bases with low deposit betas. For them, every Fed hike was a tailwind: their loan and cash yields jumped while their deposit costs lagged, so net interest income surged. Industry-wide net interest margin rose from 2.56% in 2021 to 3.30% in 2023, and the giants posted record net interest income even as regional peers buckled.

The contrast is the entire lesson of this post in one comparison. Same Fed, same rate path, same calendar — and one set of banks printed money while another set died. The difference was not luck or macro forecasting; it was *positioning*: the duration of the assets, the runnability and granularity of the deposits, and whether the hedges were on or off. Interest-rate risk is not a force of nature that happens *to* a bank. It is a position a bank *holds*, and the bank chooses the size of it.

### The mirror image: when low rates trap a bank the other way

Interest-rate risk cuts both directions, and the *fall* in rates has its own victims. Banks and insurers that wrote long-dated *fixed-rate liabilities* — guaranteed annuities, savings products promising a fixed return — got squeezed when rates collapsed toward zero in the 2010s, because the assets they bought to back those promises no longer earned enough. Japanese and European life insurers spent two decades fighting this. It is the same duration-gap arithmetic with the sign flipped: when your liabilities are longer than your assets, *falling* rates blow the hole. The general lesson is that any mismatch in duration is a directional bet on rates, whether you meant to place one or not.

## The takeaway / How to use this

The single most useful habit this post can leave you with is to **never read a bank's earnings without asking what its balance sheet is worth.** Net interest income is the lens management loves, because it is slow, flattering, and easy to spin; economic value of equity is the lens that tells you whether the bank is solvent if you marked everything honestly today. When you see a bank reporting healthy margins while sitting on a pile of long, low-yield, "held to maturity" bonds bought in a low-rate era, you are looking at exactly the configuration that hides a duration bomb. The income statement is the last place the damage shows up.

Concretely, when you size up any deposit-taking institution, ask four questions in order:

1. **What is the duration gap?** How long are the assets versus the funding? A large positive gap is a bet that rates won't rise — held by a bank that may not even know it placed the bet.
2. **How runnable is the funding?** Sticky, granular, insured deposits buy you the right to hold to maturity. Concentrated, large, uninsured deposits can revoke that right in a day. The funding side decides whether the duration risk is academic or fatal.
3. **What does EVE say, not just NII?** If the two disagree, believe the worse one. SVB's NII said "bruised"; its EVE said "dead." EVE was right.
4. **Is it hedged?** Swaps and shorter assets can flatten the gap. A bank that lets its hedges roll off to flatter near-term earnings is mortgaging its solvency for a better quarter.

For a depositor, the practical version of all this is humbler but no less important: if your balance exceeds the insurance limit, the bank's duration gap and the runnability of its deposit base are *your* problem too, because you are an unsecured creditor of a leveraged institution that may be sitting on losses its headline numbers don't show. Keeping balances within the **\$250,000** FDIC limit, or spreading them across institutions, is the one lever an ordinary person has against a risk that even sophisticated venture funds failed to manage in 2023. This is description, not advice — but it is the mechanism, and the mechanism is what protects you.

The deepest point ties back to the spine of this series. A bank is a leveraged, confidence-funded maturity-transformation machine, and interest-rate risk is the price of admission for the "maturity-transformation" part. The bank earns the spread *because* it lends long and borrows short — which is to say, it earns the spread *because* it carries a duration gap. There is no version of ordinary banking without this risk; there is only managed and unmanaged, hedged and naked, watched and ignored. The institutions that died in 1989 and in 2023 did not invent a new way to fail. They took the oldest bet in banking, made it bigger than their equity could absorb, and funded it with money that left exactly when the bet went wrong. The duration gap is where the spread comes from, and it is where the bank goes to die.

## Further reading & cross-links

- [Net interest margin and the spread business explained](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the engine that interest-rate risk both powers and endangers.
- [Reading a bank balance sheet: assets, liabilities, and equity](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) — the two-sided structure whose duration mismatch this whole post is about.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — why a modest asset loss becomes a solvency event, and why equity has a 50-year duration.
- [Why bond prices move when rates move, and by how much](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much) — the price-yield seesaw and duration, derived from the bond up.
- [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the full run dynamic that turned SVB's duration trap into a 36-hour failure.

*This article is educational, not financial advice. The figures for bank balance sheets, durations, and rate shocks in the worked examples are illustrative round numbers chosen to teach the mechanics; the SVB and macro figures are as of the 2023 episode and the dates cited.*
