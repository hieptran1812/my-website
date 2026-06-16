---
title: "From the ten-year yield to your mortgage: the transmission of rates"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into how a number in the bond market — the 10-year Treasury yield — reaches your wallet through your mortgage, your car loan, your credit card, and the jobs a company decides to fund, and why the Fed sets short rates but you mostly feel long ones."
tags: ["fixed-income", "bonds", "mortgage-rates", "interest-rates", "monetary-policy-transmission", "ten-year-treasury", "housing", "credit", "us-treasuries", "the-economy"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — the bond market sets the price of money, and that price reaches your life mostly through one number: the 10-year Treasury yield, which sets your mortgage rate.
> - The 30-year mortgage rate is, to a first approximation, just the **10-year Treasury yield plus a spread** of roughly 1.5–2.0% — so when the 10-year moves 1%, your mortgage rate moves about 1% too.
> - On a \$400,000 loan, a 1% higher mortgage rate adds roughly **\$230–\$270 to the monthly payment** and raises the income you need to qualify by about **\$10,000 a year**. That is the whole bond market, landing in one household budget.
> - The Fed sets a **short** overnight rate, but households and businesses mostly feel **long** rates and credit availability — which is why the Fed's power over your mortgage is indirect and sometimes weak.
> - Not every loan tracks the long rate: **credit cards, home-equity lines, and many auto loans track the short rate** (the prime rate), so they reprice fast; a fixed mortgage barely moves once you lock it.
> - Transmission runs on **long and variable lags** — markets reprice in days, but housing, hiring, and inflation take quarters to respond, which is why the Fed is always steering a car whose steering wheel is connected to the wheels by a long, springy rod.

Here is a question almost everyone has asked without realizing it is a question about the bond market. You are shopping for a house. Six months ago, the monthly payment you were quoted felt doable. Today, on the *exact same house, the exact same loan amount,* the payment is hundreds of dollars higher, and suddenly you do not qualify. Nothing about you changed. Nothing about the house changed. What changed was a number you have probably never looked at — the yield on the 10-year US Treasury note — and it changed in an office in the bond market, thousands of miles from the house you wanted.

This post is about that invisible wire. The bond market is where the **price of money** gets set, and that price does not stay in the bond market. It travels. It runs out of the Treasury market and into your mortgage, your car loan, your credit card, and into the boardroom where a company decides whether to build a new factory and hire the people to staff it. Economists call this *monetary policy transmission*; the rest of us experience it as "rates went up and now everything is more expensive to borrow." Either way, it is the most personal thing the bond market does, and by the end of this post you will be able to trace the chain link by link — and put dollar figures on each one.

![A pipeline showing the Fed setting the overnight policy rate, the Treasury market pricing the ten-year yield off that path, the ten-year yield anchoring the mortgage and auto and credit-card and corporate borrowing costs, households and firms feeling the changed payments and hurdle rates, and the economy speeding up or slowing down over many months](/imgs/blogs/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates-1.png)

The diagram above is the mental model for the whole post — read it left to right, because that is the direction causality flows. At the far left, the Federal Reserve sets one number: a short, overnight interest rate. That single rate gets translated by the Treasury market into a whole *curve* of yields — the 2-year, the 5-year, the 10-year, the 30-year. The 10-year yield is the one that matters most for households, because it anchors long-term borrowing. From there the chain fans out: the mortgage rate, the auto loan, the credit card, the corporate bond. Those costs land on real people and real companies, who refinance or don't, buy or don't, build or don't — and the sum of all those decisions is the economy speeding up or slowing down, over a lag of many months. Hold that picture. Everything that follows is just one link of that chain, examined closely and priced out in dollars. (This is educational, not financial advice; the goal is to understand the machine, not to tell you what to do with your money.)

## Foundations: the building blocks you need first

Let's assemble the vocabulary from zero. Some of this overlaps with the rest of [the bond series](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction); if a term is already familiar, skim it, but do not skip, because the whole post lives in how these pieces connect.

**A bond is a loan you can trade.** When you buy a bond, you are lending money to an **issuer** in exchange for a schedule of future payments: periodic interest (the **coupon**) and the return of your principal (the **face value**, or **par**) on a known date (the **maturity**). A **US Treasury** is a bond issued by the US government — the safest borrower in the world, because it prints the currency it owes. A **Treasury note** is just a Treasury with a maturity between 2 and 10 years; the **10-year note** is the most-watched bond on Earth.

**Yield is the bond's return, and it moves opposite to price.** The coupon printed on a bond never changes. What changes every second is the bond's **price** and, mirror-image to it, its **yield** — the single annual return that makes the bond's future cash flows worth exactly its current price. When the price of a bond falls, its yield rises; when the price rises, its yield falls. That inverse link is [the price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), and when this post says "the 10-year yield rose," picture investors selling 10-year notes, pushing the price down and the yield up.

**A basis point** is one hundredth of a percent — 0.01%. Rates are quoted in basis points ("bps"): a 25 bps cut is a quarter of one percent; a "1% move" is 100 bps, a large move that mostly happens over a cycle or in a crisis. Throughout this post a "1% rise in rates" means 100 bps.

**The Fed funds rate is the short rate.** The Federal Reserve does not set mortgage rates. It sets one thing directly: the **federal funds rate**, the interest rate banks charge each other to borrow overnight. It is the shortest, safest rate in the whole system — money for one night, between banks. Everything else in the economy is priced *relative* to it, but the Fed only has a hand on this one lever. (For how the Fed actually pushes that lever, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).)

**The prime rate is the short rate, repackaged for consumers.** Banks take the Fed funds rate, add a roughly fixed 3%, and call the result the **prime rate** — the benchmark they lend to their best customers at. When the Fed funds rate is 4.5%, prime is about 7.5%. Credit cards, home-equity lines, and many other consumer loans are quoted as "prime plus something," so they move the instant the Fed moves.

**The yield curve is all the rates at once.** Plot the yield of Treasuries against their maturity — overnight, 2-year, 10-year, 30-year — and you get [the yield curve](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance). The Fed controls the very left end (the overnight rate). The market sets the rest, especially the long end, based on where it thinks short rates are *headed* over the years plus a cushion called the **term premium** (the extra yield investors demand for locking their money up for a long time). This split — Fed at the short end, market at the long end — is the single most important fact in this entire post.

**A mortgage is a 30-year amortizing loan.** When you buy a house, you borrow a large sum (the **principal**) and repay it over 30 years in equal monthly payments. Each payment covers the **interest** on the balance you still owe plus a slice of **principal** that chips the balance down. Early on, almost all of the payment is interest; near the end, almost all is principal. The fixed monthly amount is set by three numbers: the loan size, the term (30 years), and the **mortgage rate**. Crucially, in the US the rate on a standard fixed mortgage is *locked for the life of the loan* — once you sign, your payment never changes even if rates move.

**The spread is the extra yield over the benchmark.** No lender lends to a homeowner at the Treasury rate; a house can lose value, a borrower can default, and the loan can be paid off early. So lenders charge the Treasury yield *plus* an extra slice — a **spread** — to cover those risks and their costs. The mortgage spread over the 10-year Treasury is the hinge of this whole post.

With those in hand, here is the one sentence that motivates everything: **the bond market sets the 10-year yield, the mortgage rate is that yield plus a spread, and your monthly payment is just arithmetic on the mortgage rate — so a move in the bond market is a move in your housing budget.** Now let's prove it, link by link.

## The mortgage rate is the 10-year yield plus a spread

Start with the single most useful fact a homebuyer can know. The 30-year fixed mortgage rate is *not* set by the Fed, and it is *not* set by your bank pulling a number out of the air. It is set, day by day, by the bond market — specifically, it tracks the **10-year Treasury yield** plus a spread of roughly 1.5 to 2.0 percentage points. If you want to know where mortgage rates are heading, you watch the 10-year, not the Fed.

![A line chart over time showing the ten-year Treasury yield rising from about one and a half percent to about four and a half percent and the thirty-year mortgage rate running consistently above it, rising from about three point two percent to about six point seven percent, with the constant gap between the two lines labeled as the spread of roughly one point seven percent](/imgs/blogs/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates-2.png)

The chart above is the centerpiece of the post, and it is illustrative — the exact wiggles are stylized, but the relationship is real and you can verify it on any rate-history site. The lower line is the 10-year Treasury yield. The upper line is the 30-year mortgage rate. Notice two things. First, **they move together** — when the 10-year rises, the mortgage rate rises; when the 10-year falls, the mortgage rate falls. They are not two independent prices; they are one price with a markup. Second, **the gap between them stays roughly constant** at around 1.7%. That gap is the spread, and it is the lender's extra yield for the risks of lending against a house instead of to the government.

Why the 10-year, when a mortgage lasts 30 years? Because almost nobody keeps a 30-year mortgage for 30 years. People move, refinance, sell, pay extra. The *average life* of a typical mortgage — how long the money is actually out before it comes back — is closer to 7 to 10 years. So the bond that best matches a mortgage's true horizon is not the 30-year Treasury but something nearer the 10-year. The market settled on the 10-year as the benchmark, and that is the wire that runs into your loan.

#### Worked example: building a mortgage rate from the bond up

*Setup.* Suppose the 10-year Treasury yield is **4.3%** today. You want to know roughly what a new 30-year fixed mortgage will cost.

*Step 1 — start from the benchmark.* The 10-year yield is 4.3%. This is the risk-free anchor: what the safest borrower on Earth pays to borrow for about a decade.

*Step 2 — add the spread.* Lenders charge a spread over the 10-year to cover default risk, prepayment risk, servicing costs, and their profit. In normal times that spread is about 1.7%. So the mortgage rate is roughly 4.3% + 1.7% = **6.0%**.

*Step 3 — sanity-check against the world.* When the 10-year was around 4.3% in 2024, 30-year mortgage rates were indeed clustered around 6.5–7%, a bit above our 6% because the spread was unusually *wide* (more on why below). The arithmetic gets you to the right neighborhood from one bond-market number.

*Step 4 — the takeaway.* You did not need to know what the Fed would do, what your bank's mood was, or anything about the housing market. *A single number from the bond market, plus a spread, lands you within a few tenths of a percent of the mortgage rate — which is why the 10-year yield is the most important number in any homebuyer's life.*

### Why the spread is not constant

The spread is "roughly 1.7%," but "roughly" is doing real work. The spread *widens* when lenders feel more risk or uncertainty, and *narrows* when they feel safe and competitive. Three things move it:

- **Prepayment risk.** A homeowner can refinance whenever rates fall, handing the lender's money back at the worst moment. When rates are volatile, that option is worth more, so lenders charge a wider spread. (This is the same negative-convexity story told in detail in the post on [mortgage-backed securities](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity).)
- **Demand for mortgage bonds.** Most mortgages are bundled into mortgage-backed securities and sold to investors. When those investors are eager to buy (or when the Fed itself is buying them, as it did during quantitative easing), the spread narrows; when they pull back, it widens. In 2023, with the Fed shrinking its bond holdings, the mortgage spread blew out to nearly 3% — far above its ~1.7% norm — which is why mortgage rates felt punishingly high even relative to Treasuries.
- **Lender competition and capacity.** When lenders are slammed with applications, they widen spreads to ration their capacity; when business is slow, they compete spreads tighter.

So the clean rule "mortgage = 10-year + 1.7%" is the skeleton. The flesh is a spread that breathes with risk and demand. But the skeleton holds: *the 10-year is the engine, and the spread is the gearbox.*

## What a 1% move does to your monthly payment

Now we make it personal. The mortgage rate is set by the bond market; the monthly payment is set by the mortgage rate. Let's price out exactly how a move in the bond market lands in a household budget. We will use one loan throughout — a **\$400,000, 30-year fixed mortgage** — and watch what happens as the rate changes.

![A line chart with the thirty-year mortgage rate on the horizontal axis from three to eight percent and the monthly payment on the vertical axis, showing a rising curve from about one thousand seven hundred dollars a month at three percent to about two thousand nine hundred dollars a month at eight percent, with annotations that each one percent of rate adds roughly two hundred thirty to two hundred seventy dollars to the monthly payment](/imgs/blogs/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates-3.png)

The chart above shows the whole relationship for our \$400,000 loan. On the horizontal axis is the mortgage rate; on the vertical axis is the monthly principal-and-interest payment. The curve rises — and it rises a little faster as rates climb, because the math of compounding is not linear. At 3%, the payment is about \$1,686 a month. At 8%, it is about \$2,935. The same loan, the same house, costs \$1,249 more *every month* — about \$15,000 more a year — purely because of where rates sat when you locked.

#### Worked example: the 1% that priced you out

*Setup.* You are buying a house and taking a \$400,000, 30-year fixed mortgage. You got pre-approved when the mortgage rate was **6%**. By the time you found a house and made an offer, the 10-year Treasury yield had risen 1%, dragging the mortgage rate up to **7%**.

*Step 1 — the payment at 6%.* Plug \$400,000, 30 years, and 6% into the standard mortgage formula. The monthly principal-and-interest payment is **\$2,398**.

*Step 2 — the payment at 7%.* Same loan, now at 7%. The payment jumps to **\$2,661**.

*Step 3 — the damage.* That is \$263 more per month — about **\$3,156 more per year** — for the identical loan. Over the life of the loan, if you held it the full term, the extra interest would total roughly \$95,000.

*Step 4 — the qualifying wall.* Lenders usually want your housing payment to be no more than about 28% of your gross income (the "front-end ratio"). At \$2,398/month you needed about **\$103,000** of income to qualify. At \$2,661/month you need about **\$114,000**. If you earn \$108,000, a 1% move in the bond market just disqualified you from the house — without you spending a dollar.

*Step 5 — the takeaway.* *A 100 bps move in a number you never look at can swing your monthly payment by hundreds of dollars and move the income you need to qualify by roughly ten thousand dollars a year — which is why housing demand cools the moment the 10-year backs up.*

### Why the curve bends upward

Notice the payment curve in the figure is not a straight line — it gets steeper at higher rates. This matters, and it is worth understanding. At low rates, most of your payment goes toward principal, so the rate has less to bite on. At high rates, interest dominates, and each extra percent piles onto a bigger interest base. In plain terms: **the first jump from 3% to 4% hurts a little; the jump from 7% to 8% hurts more.** This convex shape is the same curvature that shows up everywhere in fixed income — it is the cousin of bond [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story), and it means rate hikes do progressively more damage to affordability the higher rates already are.

#### Worked example: the same dollar payment buys a smaller house

*Setup.* You have decided you can afford exactly **\$2,400 a month** for principal and interest. The question is: how big a 30-year loan does that buy you at different rates? This flips the previous example around — instead of fixing the loan and watching the payment, we fix the payment and watch the loan.

*Step 1 — at 3%.* A \$2,400 monthly payment supports a loan of about **\$569,000**.

*Step 2 — at 5%.* The same \$2,400 supports only about **\$447,000**.

*Step 3 — at 7%.* Now \$2,400 buys a loan of just about **\$361,000**.

*Step 4 — the shrinkage.* Your budget did not change by a dollar, but your purchasing power fell from \$569,000 to \$361,000 — a **37% cut** in how much house your fixed paycheck can buy — entirely because rates rose from 3% to 7%.

*Step 5 — the takeaway.* *When the 10-year yield rises, every buyer's budget silently shrinks; the same monthly check buys a much smaller loan, which is exactly the mechanism by which the bond market cools the housing market.*

## The Fed sets short rates, but you feel long rates

Here is the part that confuses almost everyone, including smart people, including some financial journalists. **The Fed sets a short rate. You mostly feel a long rate. The two are connected, but loosely — and sometimes they move in opposite directions.**

The Fed's only direct lever is the overnight federal funds rate. When the Fed "raises rates," it raises that overnight rate. But your 30-year mortgage is priced off the 10-year Treasury yield, which the Fed does *not* set directly. The 10-year reflects the market's expectation of where short rates will average over the *next ten years*, plus a term premium. So the Fed influences the 10-year only by changing what the market expects about the *future* — not by decree.

This is why you sometimes see headlines like "The Fed cut rates, so why did my mortgage rate go *up*?" It is not a paradox. The Fed can cut the overnight rate today, and if the market reads that cut as a sign the Fed is panicking about inflation getting out of control later, long-term yields can *rise* on the news — and mortgage rates with them. The short end and the long end are different animals controlled by different forces.

![A matrix listing five household loans down the side and three columns across the top showing what each loan is priced off, how fast it moves, and its recent level, with the thirty-year mortgage priced off the ten-year Treasury while credit cards and home-equity lines and adjustable products are priced off the prime rate which tracks the Fed's short rate](/imgs/blogs/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates-4.png)

The matrix above is the practical version of this idea. Not every loan you carry tracks the same thing. Look at the first column — "what it is priced off":

- **Your 30-year fixed mortgage** tracks the **10-year Treasury** (the long rate). Once you lock it, it does not move at all; you carry whatever rate was set the day you signed.
- **Your credit card** tracks the **prime rate** (the short rate). When the Fed hikes, your card's APR rises within a statement cycle or two. Credit cards are the most rate-sensitive debt most households carry.
- **A home-equity line of credit (HELOC)** also tracks **prime** directly. It floats. When the Fed raised rates aggressively in 2022–2023, HELOC borrowers felt it almost immediately.
- **Auto loans** are tied more to short rates plus a credit margin, so new auto loans reprice fairly fast — though, like a fixed mortgage, an existing fixed auto loan is locked.
- **An adjustable-rate mortgage (ARM)** starts off a short index, is fixed for an introductory period (often 5–7 years), and then resets — handing you the future rate risk in exchange for a lower starting rate.

This is the deep reason the Fed's power over your life is uneven. When the Fed hikes, your *revolving* debt — credit cards, HELOCs — gets more expensive almost instantly. But your *existing fixed* mortgage does not budge, and *new* mortgage rates only move to the extent the bond market reprices the 10-year. The Fed has a fast, hard grip on the short-rate stuff and a slow, soft grip on the long-rate stuff.

#### Worked example: a Fed hike hits two borrowers very differently

*Setup.* The Fed raises the federal funds rate by 1% (100 bps). Two people feel it. **Alice** has a \$400,000 fixed mortgage at 4%, locked three years ago, and carries a \$10,000 credit-card balance. **Ben** has no mortgage but carries a \$30,000 balance on a HELOC tied to prime.

*Step 1 — Alice's mortgage.* Nothing happens. Her mortgage rate is locked at 4% for the life of the loan. The Fed can hike all it wants; her \$1,910 monthly payment does not change. This is the gift of the US 30-year fixed.

*Step 2 — Alice's credit card.* Her card APR is prime-plus, so it rises about 1%, from say 21% to 22%. On a \$10,000 balance that is about \$100 a year of extra interest — real, but small.

*Step 3 — Ben's HELOC.* Ben's \$30,000 HELOC is prime-plus and floats. A 1% rate rise adds 1% to his rate, which is about \$300 a year of extra interest, and it shows up almost immediately on his next statement.

*Step 4 — the takeaway.* *The same 1% Fed hike barely touches Alice's biggest debt but lands fast on Ben's floating debt — because transmission depends entirely on whether your loan is pegged to the short rate or the long rate, and whether it is fixed or floating.*

### The locked-in effect: when transmission jams

The US 30-year fixed mortgage creates a peculiar and powerful brake on transmission, called the **lock-in effect** (or "golden handcuffs"). Suppose millions of homeowners refinanced into 3% mortgages when rates were low. Now rates rise to 7%. Those homeowners are sitting on a 3% loan that would cost 7% to replace. They are *frozen* — they will not sell, because moving means giving up a 3% mortgage and signing a 7% one. The result: housing inventory dries up, existing-home sales collapse, and the rate hike's intended cooling of the economy partly jams, because the people with the cheapest credit in history simply refuse to move. This is a transmission channel that *fails to transmit*, and it was a defining feature of the US economy in 2023–2024. The Fed pushed the lever; the gears slipped.

### The fast lane: credit cards and the prime rate

If the fixed mortgage is where transmission is *slowest*, the credit card is where it is *fastest*. Recall from the foundations that a credit-card rate is quoted as "prime plus a margin," and prime is just the Fed funds rate plus a roughly fixed 3%. So a credit-card APR is, mechanically, the Fed's overnight rate plus about 3% plus the card's own margin. When the Fed moves, the prime rate moves the same day, and your card's APR follows on your next statement cycle — usually within a month or two, with no negotiation, no application, and no choice on your part.

This is why credit-card debt is the most rate-sensitive money most households touch, and why it transmits Fed policy almost perfectly. There is no benchmark to misbehave, no spread that breathes with the bond market, no lock-in. The Fed says "up 1%," and within weeks roughly every revolving balance in the country is earning the lender an extra 1%. For a household carrying a balance, this is the channel where a Fed hike shows up first and most directly — long before it touches the mortgage market.

#### Worked example: the same hike on a card versus a mortgage

*Setup.* A household carries a \$12,000 credit-card balance at a 22% APR and also holds a \$300,000 fixed mortgage locked at 4%. The Fed raises rates 1%, and the bond market pushes the 10-year up 1% too, so new mortgage rates rise from 6% to 7%.

*Step 1 — the card.* The card is prime-plus, so its APR rises about 1%, from 22% to 23%. On the \$12,000 balance that is roughly \$120 a year of extra interest, hitting within a statement cycle.

*Step 2 — the existing mortgage.* Nothing happens. The 4% rate is locked for the life of the loan. The mortgage transmits the Fed move with a delay of *infinity* — it never resets.

*Step 3 — the next mortgage.* If this household tried to *buy a new house*, the new mortgage would cost 7% instead of 6%, adding about \$200 a month on a \$300,000 loan. But that channel only fires if they actually transact.

*Step 4 — the takeaway.* *The same Fed hike reaches this household instantly through the card, never through the existing mortgage, and only-if-they-move through the next mortgage — three completely different transmission speeds inside one household, all set by what each loan is pegged to.*

Auto loans sit in the middle of this spectrum. A new auto loan reprices fairly quickly because it is freshly originated off short-rate funding plus a credit margin, so a Fed hike makes the next car loan more expensive within weeks. But an *existing* fixed auto loan, like a fixed mortgage, is locked — you keep your old rate. So the auto channel transmits to *new* buyers fast and to *existing* borrowers not at all, which is why rate hikes show up as a drop in car *sales* and a rise in the average loan rate on *new* vehicles, while people who already financed their car feel nothing.

## The refinancing channel and the wealth effect

Rate moves do not only reach you when you take out a new loan. They reach you through two more subtle channels — **refinancing** and the **housing wealth effect** — and these are how falling rates *stimulate* the economy.

![A branching diagram showing the ten-year yield falling and the mortgage rate dropping, which splits into a refinancing channel that lowers monthly payments and a house-price channel that lifts home values and creates a wealth effect, with both channels feeding into more household spending and then a faster economy](/imgs/blogs/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates-5.png)

The diagram above traces the two channels. When the 10-year yield falls and mortgage rates drop, two things happen at once. On the left branch, existing homeowners can **refinance** — swap their old, expensive mortgage for a cheaper one — which cuts their monthly payment and puts cash in their pocket every month. On the right branch, lower rates raise what buyers can afford to pay, so **house prices rise**, homeowners feel richer, and they spend more (and can borrow against the higher equity). Both branches converge on the same place: **more household spending**, which speeds up the economy. This is the engine by which a central bank's rate cuts are supposed to filter into the real world.

#### Worked example: the refinancing windfall

*Setup.* You have a \$400,000 mortgage at 7%, with a \$2,661 monthly payment. The 10-year yield falls 2%, and the mortgage rate drops to 5%. You refinance the remaining balance (assume it is still close to \$400,000 early in the loan).

*Step 1 — the old payment.* At 7%, you were paying \$2,661 a month.

*Step 2 — the new payment.* Refinance to 5% on \$400,000 over 30 years: the payment falls to \$2,147 a month.

*Step 3 — the windfall.* That is **\$514 a month freed up** — about \$6,200 a year — that you can now spend, save, or use to pay down other debt. Across millions of households, a refinancing wave like this pumps tens of billions of dollars of spending power into the economy.

*Step 4 — the catch for the lender.* Your gain is the lender's loss: they were earning 7% on your money and now earn nothing on it, because you handed it back to refinance. This is the **prepayment** that makes mortgage bonds behave so strangely (see the [MBS post](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity)).

*Step 5 — the takeaway.* *Falling rates do not just help new buyers; they reach back and cut the payments of everyone who already owns a home and can refinance, which is one of the most powerful stimulus channels a central bank has — when it works.*

### Why the refinancing channel is asymmetric

There is a cruel asymmetry here, and it is worth understanding because it shapes how recessions and recoveries feel. Refinancing only helps you when rates *fall* below your current rate. If you locked in at 3% and rates rise to 7%, you have nothing to refinance — you just keep your cheap loan and feel lucky. But if you locked at 7% and rates fall to 3%, you refinance and capture the windfall. So the refinancing channel **amplifies the good news of falling rates** but does nothing on the way up. Combined with the lock-in effect, this means the US household sector tends to *win on both ends*: it grabs cheap rates when they fall and is shielded from expensive rates when they rise (as long as it does not need to move). That is a uniquely American feature — in most other countries, mortgages reset periodically or float, and households feel rate hikes far more directly.

#### Worked example: the housing wealth effect in dollars

*Setup.* You own a home worth \$500,000 with a \$300,000 mortgage, so you have \$200,000 of equity. The 10-year yield falls, mortgage rates drop, and buyers can now afford to pay more — so home prices in your area rise 10%.

*Step 1 — the new value.* Your home is now worth \$550,000.

*Step 2 — the new equity.* Your mortgage is still \$300,000, so your equity jumps from \$200,000 to **\$250,000** — a \$50,000 gain, on paper, from a rate move you had nothing to do with.

*Step 3 — the behavioral effect.* Research on the *wealth effect* suggests households spend a few cents of every extra dollar of housing wealth. If even 5 cents on the dollar gets spent, your \$50,000 paper gain might translate to \$2,500 of extra spending — and multiplied across millions of homeowners, that is a meaningful boost to consumption.

*Step 4 — the reverse.* This runs backward too. When rates rise and home prices fall, homeowners feel poorer and pull back — the *negative* wealth effect that makes rate hikes bite even people who never take out a new loan.

*Step 5 — the takeaway.* *Lower rates lift home values, and rising home values make owners feel richer and spend more — so the bond market reaches even homeowners who never refinance, through the value of the roof over their heads.*

## How rates reach businesses: the hurdle rate

So far we have followed rates into households. But the other half of transmission runs into *businesses* — and it works through a concept called the **hurdle rate**. This is where rate moves reach jobs.

Every company that thinks about a big investment — building a factory, buying equipment, launching a product — asks one question: *will this earn more than it costs to fund?* The cost of funding is the company's **cost of capital**, and the minimum return a project must clear to be worth doing is the **hurdle rate**. When borrowing is cheap, lots of projects clear the hurdle and get funded — which means construction, orders, and hiring. When borrowing is expensive, the hurdle rises, marginal projects get shelved, and hiring slows.

A company's cost of borrowing is built the same way a mortgage rate is: a benchmark Treasury yield plus a **credit spread** for the company's default risk. (That decomposition is the subject of the post on [corporate credit](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads).) So when the 10-year yield rises, every company's borrowing cost rises, every hurdle rate rises, and the pipeline of funded projects thins. The bond market does not just price houses; it prices factories and the jobs inside them.

#### Worked example: a factory that pencils, then doesn't

*Setup.* Northwind Corp — our recurring fictional issuer from across this series — is deciding whether to build a \$50 million factory it expects will earn a 9% return on the money invested. Northwind borrows at the 10-year Treasury yield plus a 2% credit spread.

*Step 1 — the low-rate world.* The 10-year is 3%, so Northwind borrows at 3% + 2% = **5%**. The factory earns 9% and costs 5% to fund. It clears the hurdle by a comfortable 4 points. *Build it* — and hire the 200 workers to run it.

*Step 2 — the high-rate world.* The 10-year rises to 5.5%, so Northwind now borrows at 5.5% + 2% = **7.5%**. The factory still earns 9%, but now the margin over funding cost is only 1.5 points — barely worth the risk.

*Step 3 — the cancellation.* If the 10-year rises a bit more, to 6%, Northwind's borrowing cost hits 8%, and a 9% project clearing an 8% hurdle is no longer worth the execution risk. The board shelves the factory. The 200 jobs never appear.

*Step 4 — the economy-wide version.* Northwind is one company. When the bond market repriced the 10-year up by 3 percentage points, *thousands* of marginal projects across the economy crossed from "build it" to "shelve it." That is how a bond-market move becomes a hiring slowdown.

*Step 5 — the takeaway.* *Higher long rates raise every firm's hurdle rate, quietly killing the marginal projects — and the marginal jobs — that only made sense when money was cheap.*

### The two business channels: cost and credit availability

There are actually two business channels, and the second is sneakier than the first. The first is the **cost** channel we just walked through: higher rates make borrowing more expensive, so fewer projects clear the hurdle. The second is the **availability** channel: when rates rise and the economy looks shakier, banks and bond investors get *choosier*. They do not just charge more — they lend *less*, to fewer borrowers, on stricter terms. A small business might face not a higher rate but a flat "no." This credit-availability channel can tighten faster and harder than the rate channel alone, and it is why credit conditions, not just the level of rates, matter for transmission. The deepest recessions happen when both channels slam shut at once — rates high *and* credit unavailable — which is exactly what happened in 2008.

There is also a third, quieter business channel that runs through the *stock market*. A company's stock price is, at its core, the present value of its future profits discounted back to today — and the rate you discount at is built on the same 10-year Treasury yield. When the 10-year rises, the discount rate rises, and the present value of those far-off profits falls; when it drops, the present value swells. This is why a back-up in long yields can knock the stock market down even before any company has reported worse earnings: the bond market re-priced the *denominator* under every stock. (The mechanics of that discount-rate link are the subject of [the stock–bond correlation and the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).) For a business, a lower stock price means it is more expensive to raise money by issuing new shares, and a falling stock dents the confidence of executives and the wealth of the company's owners. So the 10-year reaches business investment three ways at once — through the cost of debt, the availability of credit, and the price of equity — and all three tighten together when long rates rise. The same number that sets your mortgage payment sits, quietly, under the value of every company you might work for or invest in.

## Long and variable lags: why transmission takes time

Here is the final, crucial complication, and it is the one that makes a central banker's job so hard. **Transmission is not instant. It runs on long and variable lags** — a phrase coined by the economist Milton Friedman that has haunted the Fed ever since.

![A timeline showing a rate change at day zero repricing markets in minutes, mortgage and auto and card rates resetting over weeks, refinancing and home-buying decisions responding over one to two quarters, housing starts and construction jobs and big purchases shifting over two to four quarters, and the full effect on GDP and hiring and inflation arriving over one to one and a half years](/imgs/blogs/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates-6.png)

The timeline above lays out the sequence. The *financial* part of transmission is fast: when the 10-year reprices, mortgage and auto and credit-card rates reset within days to weeks. But the *economic* part is slow. A family deciding whether to buy a house takes months to act. A homebuilder deciding whether to break ground takes a quarter or two. A company deciding whether to fund a factory takes longer still. And the full effect on GDP, hiring, and inflation — the things the Fed actually targets — can take **12 to 18 months or more** to fully arrive.

This lag is why monetary policy is so treacherous. The Fed is steering a car whose steering wheel is connected to the wheels by a long, springy rod. It turns the wheel today, and the car does not respond for a year. If the Fed waits to see the economy slow before it stops hiking, it has already over-tightened — the hikes already "in the pipeline" will keep biting for a year after it stops. This is why the Fed talks endlessly about acting *preemptively* and watching *forecasts* rather than current data. By the time the damage shows up, it is far too late to undo.

#### Worked example: the hike that bites a year later

*Setup.* In March, the Fed raises rates and long yields rise with it, pushing the mortgage rate from 6% to 7%. The Fed wants to know: when will this cool the economy?

*Step 1 — month 1.* Mortgage rates reset to 7% almost immediately. But existing homeowners are locked in and unaffected. New buyers grumble but many are already mid-purchase. Almost nothing visible happens to the economy.

*Step 2 — months 3 to 6.* New mortgage applications drop as buyers do the math and step back. Refinancing dries up. Home-tour traffic falls. The slowdown is now visible in *leading* housing data, but not yet in GDP or jobs.

*Step 3 — months 6 to 12.* Home sales fall, homebuilders slow new projects, construction hiring softens, and the furniture-and-appliance spending that follows a home purchase fades. Now the real economy is cooling.

*Step 4 — months 12 to 18.* The full effect on GDP, employment, and finally inflation shows up. The hike from over a year ago is still working its way through.

*Step 5 — the takeaway.* *A rate move reprices markets in days but takes more than a year to fully reach jobs and inflation — so the Fed is always acting on where it thinks the economy will be, not where it is, and that lag is the deepest reason monetary policy so often overshoots.*

## Putting it all together: the full payment table

Let's tie the whole chain into one table — the bond market on the left, your household budget on the right.

![A matrix with five rows for ten-year Treasury yields from one and a half to five and a half percent and three columns showing the resulting mortgage rate at the ten-year plus one point seven percent spread, the monthly payment on a four hundred thousand dollar loan, and the income needed to qualify under the twenty-eight percent rule, all rising together as the yield rises](/imgs/blogs/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates-7.png)

The matrix above is the entire post in one picture, for our \$400,000 loan. Read across any row: a 10-year yield (left) sets a mortgage rate (10-year + 1.7%), which sets a monthly payment, which sets the income you need to qualify. Then read *down* the rows and watch the whole column climb. As the 10-year rises from 1.5% to 5.5% — a 4-point move, large but well within the range of the last few years — the qualifying income climbs from about \$74,000 to about \$116,000. The same loan, the same house, now requires \$42,000 more of annual income to buy. That \$42,000 is the bond market, expressed as a household.

#### Worked example: reading the table as a story

*Setup.* Imagine a single household with \$95,000 of income shopping for the same \$400,000 loan across the rate environments in the table.

*Step 1 — the 1.5% world (mortgage 3.2%).* The payment is \$1,730 and the qualifying income is about \$74,000. Our \$95,000 household qualifies easily, with room to spare.

*Step 2 — the 3.5% world (mortgage 5.2%).* The payment rises to \$2,196 and the qualifying income to about \$94,000. Our household *just barely* qualifies. The same house is now near the edge of their reach.

*Step 3 — the 5.5% world (mortgage 7.2%).* The payment is \$2,715 and the qualifying income about \$116,000. Our \$95,000 household no longer qualifies. They are priced out of the exact house they could comfortably afford two scenarios ago.

*Step 4 — the takeaway.* *Nothing about this household changed across the three scenarios — same income, same house, same loan — yet the bond market moved them from "easily affordable" to "priced out" through one variable: the 10-year yield.*

This is the whole point of the series spine — bonds are the price of money, and the price of money sets every other price. Here it sets the price of a roof over your head, and decides who can afford one.

## Common misconceptions

**"The Fed sets mortgage rates."** It does not. The Fed sets the overnight federal funds rate. Your 30-year mortgage tracks the 10-year Treasury yield plus a spread, and the 10-year is set by the bond market based on expectations of where short rates will *average* over a decade. The Fed influences mortgage rates only indirectly, by shaping those expectations. This is why mortgage rates can rise on a day the Fed cuts, or fall before the Fed has done anything — the bond market moves first, on what it expects the Fed to do next.

**"If the Fed cuts rates, my mortgage rate will drop the same amount."** Not necessarily, and sometimes not at all. A Fed cut at the short end can coexist with *rising* long yields if the market reads the cut as inflationary or as a sign the Fed is behind the curve. In late 2024, the Fed cut its policy rate while the 10-year yield — and mortgage rates — actually rose, frustrating would-be buyers who expected relief. The short end and the long end are different markets with different drivers.

**"Lower rates are always good for me."** Lower rates help borrowers and new buyers, but they hurt savers, retirees living on bond income, and anyone holding cash. And lower rates that come *because the economy is collapsing* (a recession-driven rate cut) arrive alongside job losses and falling asset prices — cold comfort if you have just been laid off. The phrase "rates fell" is not automatically good news; it depends entirely on *why*.

**"My fixed mortgage payment will rise if rates go up."** No. The defining feature of a US 30-year fixed mortgage is that the rate is locked for the life of the loan. Once you sign at 4%, you pay 4% for 30 years no matter what rates do. (This is exactly why the lock-in effect is so powerful — and why an adjustable-rate mortgage, which *does* reset, is a fundamentally different and riskier product.)

**"Mortgage rates track the 30-year Treasury, since the loan is 30 years."** They track the **10-year**, not the 30-year. Because homeowners move and refinance, the average life of a mortgage is closer to a decade than three, so the 10-year is the better-matched benchmark. The 10-year, not the 30-year, is the number to watch.

**"The credit-card rate and the mortgage rate move together."** They move off *different* anchors. Credit cards (and HELOCs) are priced off the prime rate, which tracks the Fed's short rate, so they reprice within a billing cycle of a Fed move. A fixed mortgage tracks the long rate and doesn't move once locked. In 2022–2023, credit-card APRs shot up almost lockstep with the Fed while existing mortgage payments sat frozen — the same household feeling the Fed on one debt and not the other.

## How it shows up in real markets

**The 2020–2021 refinancing boom.** When the pandemic hit, the Fed slashed rates to zero and bought trillions of dollars of Treasuries and mortgage bonds, dragging the 10-year yield to around 0.5% and mortgage rates to record lows near 2.7%. The refinancing channel fired on all cylinders: tens of millions of American households refinanced into sub-3% mortgages, cutting monthly payments by hundreds of dollars and freeing up an estimated tens of billions of dollars of annual spending power. It was monetary-policy transmission working exactly as designed — the bond market reached into millions of household budgets and loosened them, helping fuel the spending boom (and, later, the inflation) of 2021–2022.

**The 2022–2023 affordability shock.** Then it ran in reverse, hard. As the Fed hiked from zero to over 5% to fight inflation and as the term premium rose, the 10-year yield climbed from about 1.5% to nearly 5%, and the 30-year mortgage rate roughly tripled, from under 3% to over 7%. For the buyer in our worked examples, the qualifying income on a \$400,000 loan jumped by tens of thousands of dollars. Existing-home sales collapsed to multi-decade lows. This was transmission at full force — the bond market priced an entire generation of buyers out of the market in under two years.

**The lock-in effect of 2023–2024.** The same period showed transmission *failing* to transmit. With so many homeowners holding 3% mortgages and new rates at 7%, almost no one wanted to sell — selling meant trading a 3% loan for a 7% one. Housing inventory dried up, existing-home sales fell to the lowest levels since the mid-1990s, and the rate hikes that were supposed to cool the housing market instead *froze* it. The Fed pushed the lever; the gears slipped, because the US 30-year fixed mortgage had handed millions of households a brake against the very policy the Fed was running.

**The 2023 mortgage-spread blowout.** Even relative to Treasuries, mortgages felt unusually painful in 2023, and the reason was the spread, not just the 10-year. As the Fed ran quantitative tightening (shrinking its bond holdings) and the regional-banking stress of March 2023 spooked investors, the spread of mortgage rates over the 10-year widened from its ~1.7% norm to nearly 3%. So even when the 10-year stabilized, mortgage rates stayed stubbornly high. It was a vivid reminder that "mortgage = 10-year + spread" has two moving parts, and the spread can betray you. (For the banking stress behind it, see [SVB and Credit Suisse, 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**The 1981 affordability nadir.** Go back further for the extreme case. In late 1981, after the Volcker Fed had pushed short rates toward 20% to break inflation, the 30-year mortgage rate peaked above **18%**. On a modest loan, that meant a monthly payment several times what the same loan costs at today's rates. Home sales cratered and the homebuilding industry nearly shut down. It is the historical proof that mortgage transmission, when rates go to extremes, can simply halt the housing market — and the bond bull market that followed Volcker's victory was, viewed through this lens, a forty-year tailwind for every American homebuyer.

**The international contrast.** The US 30-year fixed is unusual. In the UK, Canada, and much of Europe, mortgages are fixed for only a few years and then reset, or float outright. So when central banks hiked in 2022–2023, households in those countries felt it *fast* — their payments jumped within months as their fixes expired, and the transmission to consumer spending was far quicker and harder than in the US. The same bond-market move that froze the US housing market through lock-in instead slammed directly into British and Canadian household budgets. The *structure* of the mortgage market, as much as the level of rates, decides how fast the bond market reaches the economy.

## When this matters to you

The next time you read a headline about the Fed, or hear that "rates are going up," you now have the chain to decode what it actually means for you. Ask three questions. *Short rate or long rate?* — a Fed move hits your credit card and HELOC fast, your fixed mortgage not at all, and your *next* mortgage only to the extent the 10-year reprices. *Fixed or floating?* — if your debt is locked, you are insulated; if it floats, you are exposed. *What's the lag?* — the financial effect is days, the economic effect is quarters, so the full consequences of today's move are a year or more away.

If you want to go deeper, the natural next steps are the macro view of this same machine — [monetary-policy transmission, how rate changes reach markets](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets) and [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — and the bond-market mechanics underneath it: [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) for where long rates come from, and [mortgage-backed securities](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity) for what happens to your loan after the bank sells it. The bond market can feel abstract — a world of yields and spreads and curves traded by people you will never meet. But it is not abstract. It is the rod that connects a decision in the Treasury market to the size of your monthly payment, the value of your home, and whether the company down the road builds the factory that would have hired you. The price of money is the most personal price there is.
