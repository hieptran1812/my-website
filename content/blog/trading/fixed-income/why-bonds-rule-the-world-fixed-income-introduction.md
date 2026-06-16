---
title: "Why bonds rule the world: an introduction to fixed income"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly tour of what a bond really is, why the bond market is the biggest and most powerful market on earth, and how the price of money sets every other price."
tags: ["fixed-income", "bonds", "interest-rates", "treasuries", "yield", "credit", "introduction", "the-price-of-money"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> - **A bond is a loan you can buy and sell.** You hand over cash today, and in return you get a stream of fixed payments and your money back on a set date — a contract you can resell to someone else at any time.
> - **The bond market is bigger than the stock market.** Globally, bonds add up to roughly \$130 trillion of debt outstanding versus around \$115 trillion of listed equity (both figures illustrative, late-2025 ballpark) — yet most people barely know it exists.
> - **Bonds set "the price of money."** The interest rate on a bond is what it costs to borrow; the most-watched bond on earth, the 10-year US Treasury, is the benchmark price of money for the whole world.
> - **That one price sets every other price.** The 10-year yield reaches into your mortgage, the rate companies pay to borrow, how much a stock is "worth," the value of the dollar, and how much your government owes in interest.
> - **This is the first post of a 42-part series** that builds fixed income from zero — what a bond is, how it is priced, how its risks work, and how it quietly steers the entire financial world.
> - **No finance background required.** Every term is defined on first use, every idea is grounded in a worked dollar example.

Here is a question almost nobody can answer, even people who follow markets closely: what is the single most important number in finance? It is not the level of the S&P 500. It is not the price of Bitcoin or gold or oil. It is a number most people have never looked up — the **yield on the 10-year US Treasury bond**. When that number moves, the rate on your mortgage moves, the value of every stock in the world is quietly re-rated, the dollar strengthens or weakens, and the interest bill on the US government's \$36 trillion of debt goes up or down by billions. It is, in the most literal sense, the price of money. And it is set in the bond market — the largest, deepest, and least understood market on the planet.

This series is about that market. We are going to build the whole thing from the ground up, starting from a question a curious eight-year-old might ask — *what is a bond?* — and ending, 42 posts later, with how the bond market disciplines governments, prices every stock, and occasionally breaks the global financial system. You do not need any finance background. We will define every term the first time it appears, and we will ground every abstract idea in a concrete dollar example you can check with a calculator.

Why is something so important so poorly understood? Partly because bonds *seem* boring. A stock can ten-bag or go to zero; a high-quality bond just... pays you back, on schedule, as promised. The drama is hidden. But the drama is real, and it is enormous — it just plays out in the language of yields and basis points rather than meme stocks and IPOs. Partly, too, because the bond market is mostly *wholesale*: it is dominated by giant institutions trading in the millions and billions, often over the phone and through dealers rather than on a public exchange, so ordinary investors rarely see it directly. And partly because the ideas have a reputation for being mathematical and forbidding — duration, convexity, the yield curve. They are not. Every one of them is a simple intuition dressed in intimidating clothes, and our job in this series is to undress them one at a time.

![A cash-flow timeline showing a bond as one outflow of one thousand dollars today followed by forty dollars of coupon each year for five years and the thousand dollars returned at maturity](/imgs/blogs/why-bonds-rule-the-world-fixed-income-introduction-1.png)

The timeline above is the mental model for everything that follows. A bond is just a loan with a schedule. You pay \$1,000 today (an outflow, in red), then you collect \$40 of interest every year for five years (inflows, in green), and at the end you get your \$1,000 back. That is the entire idea. Everything else in fixed income — duration, convexity, credit spreads, the yield curve, the whole \$130 trillion edifice — is a refinement of that one picture. Master the picture, and the rest is detail.

## Foundations: the building blocks of a bond

Before we can talk about why bonds rule the world, we need a shared vocabulary. This section defines, from scratch, every term we will lean on for the rest of the series. A practitioner can skim it; a beginner should read it slowly, because nothing later will make sense without it.

### What "fixed income" means

The phrase **fixed income** is the formal name for the world of bonds and bond-like instruments. It is called "fixed" income because the payments are usually fixed in advance: you know, on the day you buy, exactly how many dollars you will receive and exactly when. That is the defining contrast with stocks. When you buy a *share* of stock, you own a slice of a company and your return is whatever the company earns and chooses to pay you — it could be a fortune, it could be nothing. When you buy a *bond*, you are not an owner; you are a **lender**, and the borrower has promised you a specific schedule of payments. Your upside is capped at getting paid exactly what you were promised. Your risk is that the borrower fails to pay.

That asymmetry — capped upside, real downside — shapes everything about how bonds behave and who buys them.

### A bond is a tradable loan

The single most important sentence in this entire series is this: **a bond is a loan you can buy and sell.**

Think about an ordinary loan. Your friend borrows \$1,000 from you and promises to pay you back with interest. That is a private contract between the two of you. If you suddenly need your money, you are stuck — you cannot easily hand the loan to a stranger and get cash.

A bond removes that limitation. When a government or a large company needs to borrow, instead of one private loan it issues thousands of identical, standardized IOUs — bonds — that can be freely traded. The original lender does not have to wait until the loan is repaid; they can sell the bond to someone else, who collects the remaining payments. This *tradability* is the magic ingredient. It is what turns a sleepy IOU into a living, breathing market where prices move every second.

### The anatomy of a bond: par, coupon, maturity, issuer

Every bond is described by four numbers and a name. We will spend an entire later post on each, but here is the starter set:

- **Issuer** — *who is borrowing.* The US Treasury, the German government, Apple, your local city. The issuer's trustworthiness is everything: a promise from the US government is treated as rock-solid; a promise from a struggling company is not.
- **Par value (also called face value or principal)** — *how much you get back at the end.* The standard is \$1,000 per bond. This is the loan amount that is repaid at maturity.
- **Coupon** — *the interest the bond pays,* expressed as an annual percentage of par. A "4% coupon" on a \$1,000 bond means \$40 of interest per year. (The name comes from an era when bonds were paper certificates with detachable coupons you clipped and mailed in to collect interest.)
- **Maturity** — *the date the loan ends* and par is repaid. A "5-year bond" repays in five years; a "30-year bond" in thirty. Maturity is why a bond is "fixed income with an end date," unlike a stock you can hold forever.

We will use a recurring example bond throughout this series so the numbers compound: **a 5-year, \$1,000 par note with a 4% coupon.** Whenever you see that bond, picture the timeline from Figure 1.

### Yield: the bond's actual return

The **coupon** tells you the dollars a bond pays, but it does not tell you your *return*, because you might not pay \$1,000 for the bond. Prices move. If you buy our 4% bond for only \$900, your \$40 coupon is a bigger return on your smaller outlay. The number that captures your real return — accounting for what you actually paid — is the **yield**.

For now, hold one rough definition: **a bond's yield is the return you earn if you buy it at today's price and hold it to maturity, collecting every payment.** We will dissect the several precise kinds of yield in a dedicated post; the crucial fact for today is that *price and yield move in opposite directions.* When a bond's price goes up, its yield goes down, and vice versa. This seesaw is the heartbeat of the bond market, and we will return to it shortly.

### The time value of money: why a dollar later is worth less than a dollar now

There is one more foundation we need, and it is the deepest idea in all of finance: **a dollar today is worth more than a dollar in the future.** Not because of inflation (though that is part of it), but because a dollar you have now can be put to work earning interest. If you can earn 4% a year, then \$1 today grows to \$1.04 in a year. Run that backward: \$1.04 *next* year is worth only \$1 *today*, because \$1 today would grow into it. We say the *present value* of \$1.04 received in a year, at a 4% rate, is \$1.

This is the machinery of **discounting**, and it is how every bond — indeed every investment that pays in the future — is valued. To find what a future payment is worth today, you *discount* it by the interest rate: divide by (1 + rate) for each year you have to wait. A \$1,000 payment due in one year, at a 4% discount rate, is worth \$1,000 ÷ 1.04 ≈ \$962 today. Due in two years, it is worth \$1,000 ÷ 1.04² ≈ \$925. The farther away the money, the more it is shrunk, and the higher the rate, the harder it shrinks. A bond's price is nothing more than the present value of all its future coupons and its final principal, added up. We will build that sum brick by brick in Track B; for now, just hold the intuition that *future money is discounted to the present, and the discount rate is the price of money.*

#### Worked example: the present value of a single future dollar

Suppose someone offers you a contract that will pay you exactly \$1,000 in five years, with no risk at all. What is it worth today, if safe interest rates are 4%? You discount \$1,000 by 4% for five years: \$1,000 ÷ (1.04)^5 = \$1,000 ÷ 1.217 ≈ **\$822**. So you should be willing to pay about \$822 today for a guaranteed \$1,000 in five years.

Now watch what happens if rates rise to 6%. The same \$1,000 in five years is worth \$1,000 ÷ (1.06)^5 = \$1,000 ÷ 1.338 ≈ **\$747** — about \$75 less, a 9% drop in value, from nothing more than the discount rate rising. The future payment never changed; only the price of money did.

*Every bond is a bundle of future dollars, and a higher discount rate shrinks each of them, which is the deep reason rising rates push bond prices down.*

### A few units you will see constantly

- **Basis point (bp)** — *one hundredth of a percent,* i.e. 0.01%. Bond people talk in basis points because the moves that matter are small. "The 10-year rose 40 bps" means it went up 0.40 percentage points, say from 4.10% to 4.50%.
- **Notional** — *the face amount* a contract is based on. A "\$10 million notional" position is one whose payments are calculated on \$10 million of par.
- **Spread** — *the extra yield* one bond pays over a safer benchmark. If a company bond yields 6% and a same-maturity Treasury yields 4%, the **credit spread** is 2% (200 bps), the market's price for the company's extra default risk.
- **The risk-free rate** — *the yield on the safest possible bond,* in practice a US Treasury. It is the floor under every other interest rate, the number from which everything else is built up by adding spreads.

With that vocabulary in hand, we can ask the real question.

## Why the bond market is the most powerful market on earth

Most people, asked to picture "the financial market," picture a stock ticker — green and red numbers, the Dow, the Nasdaq, the trading floor of the New York Stock Exchange. Stocks get the headlines, the movies, the water-cooler talk. But the stock market is not the biggest market, and on most days it is not the most important one. The bond market is.

![A side-by-side comparison showing the global stock market at roughly one hundred fifteen trillion dollars next to the global bond market at roughly one hundred thirty trillion dollars of debt outstanding](/imgs/blogs/why-bonds-rule-the-world-fixed-income-introduction-3.png)

By size, the global bond market — all the government, corporate, mortgage, and municipal debt outstanding — runs to roughly **\$130 trillion** (an illustrative, late-2025 ballpark; market-size estimates vary by source and definition). The global stock market, all listed company shares added together, is around **\$115 trillion**. Bonds win. And size is only the start of the story.

The bond market is more powerful than the stock market for three reasons, and each one will get its own track later in this series.

**First, bonds are where the price of money is set.** Every loan in the economy — your mortgage, a company's expansion loan, a country's deficit financing — is priced off the interest rates established in the bond market. Stocks are *priced by* interest rates; bonds *are* the interest rates. That makes the bond market upstream of almost everything else.

**Second, the bond market is where governments fund themselves.** The US government spends far more than it collects in taxes, and it covers the gap by selling Treasury bonds. So does almost every other government. If the bond market loses confidence in a government and refuses to lend except at punishing rates, that government has a crisis on its hands — a dynamic so dramatic it earned a nickname, the **bond vigilantes**, which we will meet in Track G.

**Third, the bond market is the plumbing of the financial system.** Banks hold bonds as their safest assets. Central banks buy and sell bonds to run monetary policy. Trillions of dollars of short-term lending are secured against bonds (the "repo" market). When the bond market seizes up — as it briefly did in March 2020 — the whole system shakes, and central banks intervene with overwhelming force, because a broken bond market means a broken economy.

### The bond universe: a map of who borrows

The \$130 trillion is not one homogeneous blob. It is a set of distinct neighborhoods, each with its own borrowers, risks, and rules, and you should carry a rough map of them from the very start. We will spend all of Track F walking these neighborhoods one by one, but here is the aerial view.

![A grid of the bond market segments listing US Treasuries, mortgage bonds, US corporates, municipal bonds, agency debt, and emerging-market debt with illustrative sizes and who borrows in each](/imgs/blogs/why-bonds-rule-the-world-fixed-income-introduction-2.png)

Reading the map from biggest to smallest (all sizes illustrative, late-2025 ballpark):

- **US Treasuries (~\$28 trillion).** Debt of the US federal government. This is the largest, deepest, and safest segment, and it is the *benchmark* — the risk-free rate from which every other bond is priced. When you hear "the 10-year," this is the neighborhood.
- **Mortgage-backed securities, or MBS (~\$12 trillion).** Thousands of individual home loans pooled together and sold as bonds. When you pay your mortgage, your payment flows through to the holders of one of these bonds. MBS have a strange and important quirk — homeowners can refinance, which lets them "prepay" — that makes them behave unlike any other bond, a topic we will dig into in Track F.
- **US corporate bonds (~\$10 trillion).** Debt of companies, from rock-solid giants like Apple and Johnson & Johnson down to struggling firms whose bonds trade at distressed prices. The range of credit quality here is enormous, which is why credit risk (Track E) is its own discipline.
- **Municipal bonds (~\$4 trillion).** Debt of US states, cities, and public authorities, funding roads, schools, and water systems. Their interest is often exempt from federal (and sometimes state) tax, which makes their lower headline yields competitive for high-tax-bracket investors.
- **Agency debt (~\$2 trillion).** Debt of government-sponsored entities like Fannie Mae and Freddie Mac and various federal agencies — a notch riskier than Treasuries in theory, but widely treated as nearly as safe.
- **Emerging-market and other sovereign debt (trillions, varied).** Debt of governments outside the major developed economies. It pays higher yields to compensate for *country risk* — the chance of default, currency collapse, or political upheaval — and is split between "hard currency" (borrowing in dollars) and "local currency" (borrowing in the country's own money).

Each neighborhood answers the same two questions differently: *how likely am I to be paid back* (credit risk) and *how much will my price move if rates change* (interest-rate risk). Those two axes — credit and duration — organize the entire bond market, and most of this series is really about measuring them.

#### Worked example: feeling the size of \$130 trillion

Abstract trillions are hard to feel, so let's make them concrete. The US economy produces about \$28 trillion of goods and services a year (its GDP). The global bond market, at roughly \$130 trillion, is therefore around **4.6 times the size of the entire US economy** — every farm, factory, app, and haircut in America, multiplied by nearly five. Put differently: if you tried to count \$130 trillion at one dollar per second, it would take you over **four million years**. The US Treasury market alone, at roughly \$28 trillion, trades around \$900 billion of bonds *per day* — more dollars changing hands in a single day than many countries produce in a year.

*The bond market is not a sleepy corner of finance; it is the largest pool of capital humanity has ever assembled, and it reprices itself every single day.*

## The influence thesis: the price of money sets every other price

Here is the central idea of this whole series, the thread we will pull on for 42 posts. We call it the **influence thesis**, and it goes like this: *a bond's interest rate is the price of money, and the price of money sets the price of everything.*

To see why, you have to understand what an interest rate actually *is*. An interest rate is the rent on money — the price of having a dollar now instead of later. If that rent is high, money is "expensive": borrowing costs more, saving pays more, and a dollar in the future is worth much less than a dollar today. If the rent is low, money is "cheap": borrowing is easy and future dollars are nearly as good as present ones. The bond market is the giant auction where the rent on money is set, maturity by maturity, every day.

The single most important rent in the world is the yield on the **10-year US Treasury**, often just called "the 10-year." It is the benchmark borrowing cost for the world's safest borrower over a medium horizon, and it serves as the reference point from which almost every other rate is built. Watch what happens when it moves.

![An influence web showing the Fed setting the overnight rate, the bond market setting the ten-year Treasury yield, and that yield fanning out to mortgage rates, corporate borrowing, stock valuations, the dollar, and the government budget](/imgs/blogs/why-bonds-rule-the-world-fixed-income-introduction-4.png)

The diagram traces the channels. The Federal Reserve sets the very shortest rate — the overnight rate banks charge each other — but it does *not* directly set the 10-year. That is set by the bond market, by millions of buyers and sellers betting on growth, inflation, and Fed policy years into the future. And once the market sets the 10-year, it cascades outward:

- **Mortgages.** A 30-year US mortgage rate is, very roughly, the 10-year Treasury yield plus a spread of around 2 to 3 percentage points. When the 10-year rose from about 1.5% in 2021 to around 4.5% in 2023, 30-year mortgage rates roughly tripled, from under 3% to over 7%, and the US housing market froze. Nothing about houses changed; the price of money changed.
- **Corporate borrowing.** A company's bond yield is the Treasury yield plus a credit spread. When Treasuries rise, every company's cost of borrowing rises with them, which slows hiring, investment, and buybacks across the economy.
- **Stock valuations.** This one surprises people. A stock is worth the value *today* of all the cash a company will pay you *in the future*. To convert future dollars into today's dollars you "discount" them by an interest rate — and the higher that rate, the less those future dollars are worth now. When the 10-year rises, the discount rate rises, and stocks, especially high-growth stocks whose payoff is far in the future, fall. This is why a "bond move" can crater the stock market.
- **The dollar.** Higher US yields make US bonds more attractive to global savers, who must buy dollars to buy those bonds. So rising yields tend to pull money into the dollar and strengthen it, which in turn moves the price of everything traded in dollars — oil, gold, the cost of imports.
- **The government budget.** The US has roughly \$36 trillion of debt. When the rate it must pay to roll that debt over rises, the annual interest bill balloons — past \$1 trillion a year recently — crowding out other spending. The bond market literally sets how much the government owes.

One number. Six channels. That is the influence thesis in a single picture, and it is why a bond trader once joked that he wanted to be reincarnated as the bond market, because "you can intimidate everybody."

#### Worked example: how a 1% rate move resets a stock's value

Let's make the stock channel concrete, because it is the least intuitive. Imagine a simple company that you expect to pay you \$100 a year forever (a perpetual stream — economists call it a perpetuity). How much is that stream worth today? The math of a perpetuity is mercifully simple: its value is the annual payment divided by the discount rate.

At a discount rate of 4% (think: the 10-year at 4%), the company is worth \$100 ÷ 0.04 = **\$2,500**.

Now suppose the bond market pushes the 10-year, and so the discount rate, up by just one percentage point, to 5%. The company's cash flows have not changed at all — it still pays \$100 a year. But its value is now \$100 ÷ 0.05 = **\$2,000**.

A 1-percentage-point move in the price of money knocked **20%** off the value of a business whose actual prospects were unchanged. That is the bond market reaching into the stock market and repricing it.

*Stocks are not valued in a vacuum; they are valued against the yield on bonds, so when bonds move, stocks must move too.*

### Why bonds often move first

There is a reason bond traders carry a certain swagger: the bond market is frequently *ahead* of the others. Two structural facts make it a leading indicator.

The first is *who* trades it. The bond market is dominated not by retail investors chasing stories but by enormous, professional, often unemotional institutions — central banks, pension funds, insurers, sovereign-wealth funds, and the trading desks of the world's largest banks. These players are paid to forecast growth and inflation years out, because that is what determines whether a 10-year bond is a good deal. Their collective judgment about the economy is expressed, in real time, as the level and shape of the yield curve. When that judgment shifts, the bond market reprices before the slower-moving real economy shows it in the data.

The second is *what* it prices. Bonds are claims on the future — a 10-year bond is a bet about the next decade of inflation and rates. So the bond market is intrinsically forward-looking in a way that, say, this quarter's earnings are not. The most famous example is the **inverted yield curve**: when shorter-term bonds yield *more* than longer-term ones (an inversion), it has preceded almost every US recession of the past half-century, often by a year or more. The bond market, in effect, forecasts the downturn and is usually right. That is why economists and central bankers watch the curve obsessively, and why we devote an entire track (Track D) to reading it.

This does not mean bonds are always right or always first — markets are messy, and there are famous head-fakes. But as a rule of thumb, when stocks and bonds disagree about the economy, the smart money pays close attention to what the bond market is saying, because it is usually the more sober forecaster.

### The two great risks every bond carries

Strip away the jargon and a bond exposes you to exactly two big risks, and almost everything in this series is a tool for measuring or managing one of them.

**Interest-rate risk** is the risk that rates *change* while you hold the bond, moving its price via the seesaw. It applies even to a perfectly safe bond that will certainly pay you back — a US Treasury has zero default risk but plenty of interest-rate risk, as 2022 brutally proved. The measure of interest-rate risk is *duration*, the subject of Track C.

**Credit risk** is the risk that the borrower *fails to pay* — misses a coupon, or cannot return your principal. A Treasury has effectively none; a shaky company's bond has a lot. The market charges for credit risk through the *credit spread*, the extra yield a risky bond pays over a Treasury, the subject of Track E.

Hold these two axes in your head as you read everything that follows. A short, high-quality bond (a 2-year Treasury) has little of either risk and pays little. A long, low-quality bond (a 30-year junk bond) is loaded with both and pays a lot. The yield on any bond is, fundamentally, the market's price for the particular mix of duration and credit risk that bond carries — nothing more, nothing less.

## The price-yield seesaw: the one mechanic you must internalize

If the influence thesis is the *why* of this series, the price-yield relationship is the *how*. It is the single mechanic that confuses more beginners than anything else in fixed income, so we will take it slowly and return to it many times.

The puzzle is this: a bond's coupon is fixed. Our example bond pays \$40 a year, no matter what. So how can its price move? And why does it move *opposite* to interest rates?

![A convex curve charting bond price on the vertical axis against market yield on the horizontal axis, sloping down from a premium above one thousand dollars at low yields, crossing par at the four percent coupon, to a discount below one thousand dollars at high yields](/imgs/blogs/why-bonds-rule-the-world-fixed-income-introduction-5.png)

The chart shows the answer. Picture our bond: it pays \$40 a year on a \$1,000 face value, a 4% coupon. The day it is issued, market interest rates are also 4%, so the bond is worth exactly its \$1,000 par — there is no reason to pay more or less for a stream that yields exactly the going rate. That is the middle of the seesaw, marked "PAR."

Now suppose, the day after you buy it, the market interest rate jumps to 5%. Brand-new bonds are now being issued that pay \$50 a year. Your old bond, stuck paying \$40, is suddenly worse than the new ones. Nobody will buy it from you for \$1,000 — why pay full price for \$40 when \$50 is available at the same price? The only way to sell your bond is to drop its price until the \$40 it pays represents a competitive 5% return for the buyer. Its price falls — into the red "DISCOUNT" zone of the chart. **Rates up, price down.**

The reverse is just as true. If market rates fall to 3%, new bonds pay only \$30. Your old bond, paying \$40, is now a prize. Buyers will bid its price *up*, into the green "PREMIUM" zone, until its rich \$40 coupon represents only a 3% return on the higher price they paid. **Rates down, price up.**

That is the seesaw: the coupon is fixed, so when the going rate changes, the *price* must move in the opposite direction to keep the bond competitive. The curve is not a straight line — it bends, a property called **convexity** that we will devote a whole post to — but the direction never changes. Price and yield always move opposite each other.

#### Worked example: pricing our 5-year note when rates jump

Let's put real numbers on the seesaw. Take our recurring bond: 5-year, \$1,000 par, 4% coupon, so \$40 a year and \$1,000 back at year five. Price a bond by adding up the value *today* of each future payment, discounted at the market rate. (The full machinery of discounting gets its own post; here we just want the punchline.)

At a 4% market rate, the present value of \$40 a year for five years plus \$1,000 at year five works out to exactly **\$1,000** — it trades at par, as it should when the coupon equals the market rate.

Now let the market rate jump to 5%. We discount the same \$40-a-year-plus-\$1,000 stream at 5% instead of 4%, and the value drops to roughly **\$957**. The bond fell about \$43, or **4.3%**, even though a one-year bond would have fallen only about 1%.

Why did a 1% rate move cause a 4.3% price drop? Because the bond locks in its below-market coupon for *five years*, and the longer that lock, the more the price has to fall to compensate. This sensitivity has a name — **duration** — and it is the most important number in fixed income after the yield itself. (Our bond has a duration of roughly 4.5 years, which is why a 1% rate move produced a ~4.5% price move.)

*A 1% change in rates is never a 1% change in price; the longer the bond, the harder the seesaw tips.*

#### Worked example: why the same rate move hurts a 30-year bond far more

To feel duration in your bones, compare two bonds when rates rise 1 percentage point, from 4% to 5%:

- A **2-year** bond falls about **2%**.
- Our **5-year** bond falls about **4.3%**.
- A **30-year** bond falls about **17%**.

Same 1% move in the price of money; wildly different damage. The 30-year bond has locked in its below-market coupon for three decades, so its price must collapse far more to bring its effective yield up to the new 5%. This is why, in 2022, when the Fed pushed rates up at the fastest pace in 40 years, long-dated US Treasuries — supposedly the *safest* asset in the world — lost more than 30% of their value, a worse drawdown than many stock-market crashes. "Safe from default" is not the same as "safe from price moves," a distinction that destroyed a bank in 2023, as we will see.

*The "safest" bond can still be a dangerous trade, because a long maturity makes its price violently sensitive to the price of money.*

## Who issues bonds, and who buys them

A market needs two sides. To understand the bond market's power, you have to see who is on each side and why they are there.

![A flow diagram showing governments, companies, and cities issuing bonds into the bond market on one side, and banks, pensions and insurers, central banks, and households buying them on the other](/imgs/blogs/why-bonds-rule-the-world-fixed-income-introduction-6.png)

On the **borrowing** side stand the issuers, and their motive is always the same: they need cash now and are willing to pay for it over time.

- **Governments** issue the most by far. They run deficits — spending more than they tax — and bonds fund the gap. US Treasuries are the giant here; we will give them a whole post, because they are the benchmark the entire world prices against.
- **Companies** issue corporate bonds to fund factories, acquisitions, or simply because borrowing is cheaper than issuing stock. Apple, despite hundreds of billions in cash, has issued tens of billions in bonds, because the after-tax cost of bond interest was lower than the cost of other financing.
- **Cities and states** issue municipal bonds to build roads, schools, and water systems, often with a tax advantage that makes their lower yields competitive.

On the **lending** side stand the buyers, and their motive is the mirror image: they have cash now and want safe, predictable income later.

- **Banks** hold bonds as their safest, most liquid assets — a place to park deposits that earns interest and can be sold or borrowed against in a pinch.
- **Pension funds and insurers** are the natural home for bonds. They have known future obligations — a pension check in 2050, an insurance payout whenever a policyholder dies — and a bond's fixed, scheduled payments are tailor-made to match those liabilities. This matching is called **immunization**, and it is one of the deepest ideas in the field.
- **Central banks** hold enormous bond portfolios, both as foreign-exchange reserves and as the lever of monetary policy (buying bonds to push rates down, selling them to push rates up).
- **Households and funds** — through bond mutual funds and ETFs, or directly — buy bonds for income and as ballast against the wilder swings of stocks.

### Two markets in one: primary and secondary

There is a subtlety worth naming now, because it explains how a "fixed" loan can have a constantly moving price. The bond market is really two markets stacked together.

The **primary market** is where bonds are *born*. When the US Treasury needs to borrow, it holds an auction and sells brand-new bonds directly to large buyers. When a company wants to raise money, it works with banks to issue new bonds to investors. This is the moment the loan is actually made — cash flows from lender to borrower, and a new bond comes into existence. The borrower only ever raises money in the primary market.

The **secondary market** is where those same bonds are *traded* afterward, investor to investor, for the rest of their lives. The borrower is not involved and gets no new money; ownership simply changes hands. This is where prices move every second, where the seesaw operates, and where the \$900-billion-a-day Treasury turnover happens. When you read that "the 10-year yield rose today," that is the secondary market repricing existing bonds.

The two are linked: the secondary-market yield tells the borrower what rate it will have to offer on its *next* primary issue. If the secondary market is demanding 5% for a company's existing bonds, that company cannot issue new bonds at 4% — nobody would buy them. So the secondary market, by continuously repricing old debt, sets the cost of all new debt. That feedback loop is how a market full of tradable old loans ends up governing the price of money for everyone who wants to borrow tomorrow.

#### Worked example: why a pension fund buys a 30-year bond

Suppose a pension fund owes a retiree \$1,000 in exactly 30 years. How should it set aside money today to be *sure* it can pay? It could invest in stocks and hope, but "hope" is not a plan a pension can run on. Instead it buys a 30-year, \$1,000 zero-coupon Treasury — a bond that pays nothing along the way and returns \$1,000 at maturity. At a 4% yield, that bond costs about **\$308 today** (\$1,000 discounted at 4% for 30 years). The pension pays \$308 now, locks in the exact \$1,000 it will need, and is *immunized*: it no longer cares what rates, stocks, or the economy do over the next 30 years.

*Bonds let an institution convert a known future obligation into a known present cost, which is exactly why the entities that owe the future — pensions, insurers, governments — are the bond market's biggest players.*

## How to read this series

This is post #1 of 42. The series is organized into eight tracks that climb from the simplest idea to the most powerful one. You can read straight through, or jump to the track you care about — each post stands alone, but they compound if read in order.

![A roadmap grid listing the eight tracks of the series, from the bond itself and pricing, through rate risk and the yield curve, credit and the sectors, to the influence on the wider world and portfolio construction](/imgs/blogs/why-bonds-rule-the-world-fixed-income-introduction-7.png)

Here is the map:

- **Track A — Foundations.** What a bond is, line by line: par, coupon, maturity, issuer, who borrows, who lends, and the price-yield seesaw. (You have just done the express version of this track.)
- **Track B — Pricing and the time value of money.** Where a bond's price comes from: discounting future cash flows, why one rate is a simplification, and the spot and forward rates hidden inside the curve.
- **Track C — Interest-rate risk: duration and convexity.** The math of the seesaw. How to measure a bond's sensitivity to rates in years (duration) and in dollars (DV01), and the curvature correction (convexity) that duration misses.
- **Track D — The yield curve.** The most important chart in finance: how yields differ by maturity, why the curve usually slopes up, and why it inverting has predicted nearly every US recession.
- **Track E — Credit and default risk.** What changes when the borrower might *not* pay: credit spreads, ratings, the great divide between investment-grade and high-yield, and where bondholders stand in a bankruptcy.
- **Track F — The sectors of the bond market.** A tour of the neighborhoods: Treasuries, inflation-linked bonds, corporates, municipals, mortgage-backed securities, and emerging-market debt.
- **Track G — Bonds and the wider world.** The influence track. The thesis of this post, in full: how the risk-free rate anchors all valuation, disciplines governments, and transmits monetary policy to your mortgage.
- **Track H — Portfolio, trading, and crises.** How bonds are actually owned and traded — ladders, funds, the dealer market — and the great bond blowups of 1994, 2008, 2020, 2022, and 2023.

If you take one habit from this series, let it be this: **whenever you read a financial headline, ask what it implies for the price of money.** Stocks fell? Check whether yields rose. The dollar surged? Check whether US yields climbed relative to the rest of the world. A government is in trouble? Watch its bond yields, because that is where the market renders its verdict first. Once you start watching bonds, you start seeing the gears behind everything else.

For the policy and trading lens on these same forces, the sibling macro series covers [interest rates as the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession); for the allocation view, see [government bonds as the risk-free anchor](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).

## Common misconceptions

Fixed income is a field where intuition often points the wrong way. Here are the beliefs that trip up almost every beginner.

**"Bonds are safe; stocks are risky."** This is half-true and dangerously incomplete. Bonds are safer than stocks against *one* risk: a high-quality bond is very likely to pay you back, while a stock can go to zero. But bonds carry their own risks that stocks largely do not. If rates rise, a long bond's *price* can fall sharply even though it will eventually pay in full — long Treasuries lost over 30% in 2022. And a low-quality bond can default and pay you cents on the dollar. "Safe" depends entirely on which bond and which risk.

**"A bond's price can't change — it pays a fixed amount."** The *payments* are fixed; the *price* is not. Because you can buy and sell a bond before maturity, its market price floats up and down every day as interest rates move, exactly as the seesaw shows. Only if you hold to maturity and the issuer pays in full do you lock in the yield you bought at; sell early and you get the going market price, which may be far from what you paid.

**"Higher yield is always better."** A higher yield is the market paying you for taking more risk — longer maturity (more price sensitivity), or weaker credit (more chance of default), or worse liquidity. A 9% bond is not a better deal than a 4% bond; it is a *riskier* one, and the extra 5% is the price of that risk. The skill in fixed income is judging whether the extra yield compensates for the extra risk, not chasing the biggest number.

**"The Federal Reserve sets interest rates."** The Fed sets *one* rate — the overnight rate banks lend to each other — with great precision. It powerfully *influences* all the others, but it does not set them. The 10-year yield, the one that drives your mortgage, is set by the bond market's collective bet on growth, inflation, and future Fed policy. The Fed can cut its overnight rate and watch the 10-year *rise* if the market fears inflation — a humbling reminder of who is really in charge of the price of money.

**"Bonds and stocks always move in opposite directions."** They often do — when growth scares hit, money flees stocks for the safety of bonds, lifting bond prices. That negative correlation is the engine of the classic 60/40 portfolio. But it is not a law. In 2022, *both* fell hard at the same time, because the shock was rising rates, which hurts both. When the dominant risk is inflation rather than growth, stocks and bonds can move together — and that is precisely when "diversified" portfolios fail their owners.

**"If I hold to maturity, I can't lose money on a bond."** Hold a high-quality bond to maturity and you will indeed get back exactly the yield you bought at, regardless of how its price bounced around in between — the price risk washes out if you never sell. But two things can still cost you. First, *default*: if the issuer fails, "holding to maturity" gets you a fraction of your money, not all of it. Second, *opportunity cost and inflation*: lock in 2% for ten years and then watch rates and inflation climb to 6%, and you have technically "not lost money" while badly losing purchasing power and the chance to earn more elsewhere. "No loss if held to maturity" is true only in nominal terms and only if the borrower pays.

**"Bonds are an obscure thing only Wall Street cares about."** Almost every adult is, quietly, a bondholder. Your bank deposits sit largely in bonds. Your pension or 401(k) almost certainly holds them. Your insurance company backs its policies with them. And on the borrowing side, the rate on your mortgage, your car loan, and your government's deficit all flow from the bond market. You are connected to it from both ends whether or not you have ever bought a single bond yourself.

## How it shows up in real markets

The influence thesis is not theory. Here are episodes where the price of money, set in the bond market, visibly steered everything else.

**The 2022 rate shock — the worst bond year in modern history.** Coming out of the pandemic, the US 10-year yield sat near 1.5% and inflation was surging. The Fed responded with the fastest rate-hiking cycle in 40 years, and the bond market repriced violently: the 10-year climbed past 4%. Because of duration, long Treasuries — the planet's "safe" asset — lost more than 30%, and the classic 60/40 portfolio had one of its worst years ever, as *both* stocks and bonds fell together. The lesson of the series in one year: the price of money is the master variable, and when it moves fast, nothing is safe.

**Silicon Valley Bank, March 2023.** SVB took in a flood of deposits and parked them in long-dated US Treasuries and mortgage bonds — assets with no default risk at all. But when the Fed jacked up rates in 2022, the *price* of those long bonds collapsed (the seesaw, plus duration). The bonds were perfectly safe from default and catastrophically unsafe in price. When depositors got nervous and asked for their money, SVB had to sell those bonds at a deep loss, the loss became visible, the run accelerated, and the bank failed in days. A bank brought down not by bad loans but by interest-rate risk on the world's safest bonds. For the full anatomy, see [the SVB and Credit Suisse bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

**The UK gilt crisis, September 2022.** A new British government announced large unfunded tax cuts, and the bond market revolted: yields on UK government bonds (gilts) spiked in days. That spike detonated a hidden problem in UK pension funds, which had used leverage in their bond-matching ("LDI") strategies and now faced collapsing collateral. The Bank of England had to intervene with emergency bond-buying to stop a doom loop, and the policy was reversed within weeks. The bond market, in the role of vigilante, fired and removed a government's fiscal plan — and very nearly a prime minister.

**The March 2020 dash for cash.** When the pandemic hit, investors stampeded for cash and even sold US Treasuries — the supposedly most liquid market on earth briefly stopped functioning smoothly. Because Treasuries are the collateral underneath the entire financial system, the Fed responded with overwhelming force, buying trillions of dollars of bonds to restore order. It was a vivid demonstration that the bond market is not just *a* market; it is the plumbing, and when it clogs, central banks will do almost anything to clear it.

**The Volcker shock, 1979–1982.** To break the runaway inflation of the 1970s, Fed Chair Paul Volcker pushed short-term rates above 19%. The 10-year Treasury yield reached roughly 15%. The price of money went to extraordinary heights, mortgage rates topped 18%, the economy fell into a deep recession — and inflation finally broke. The episode is the clearest case study of the chain in this post: move the price of money far enough, and you can reshape the entire economy. It also launched a 40-year bull market in bonds as yields fell from those peaks all the way to near zero by 2020. See [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) for the mechanics.

**The era of zero rates, 2009–2021.** After the 2008 crisis, central banks held short-term rates near zero for years and bought bonds in bulk (a policy called quantitative easing, or QE) to push longer-term yields down too. With the price of money pinned near zero, the influence thesis ran in reverse: cheap money lifted the value of nearly every asset. Stocks, housing, and even speculative corners like crypto soared, because when the discount rate is near zero, future cash flows are barely shrunk and risky assets look irresistible. Some European and Japanese government bonds even traded at *negative* yields — investors paying for the privilege of lending — a once-unthinkable state that, at its peak, applied to well over \$15 trillion of bonds. The lesson was the mirror image of Volcker's: push the price of money low enough, for long enough, and you inflate the value of everything else. When that cheap-money era ended abruptly in 2022, the reversal is exactly what produced the rout described above.

**Greece and the European debt crisis, 2010–2012.** Greece had borrowed heavily, and when markets began to doubt it could repay, the yield on Greek 10-year bonds spiraled from single digits past 30%. At those rates, borrowing became impossible and the debt unpayable — a self-fulfilling doom loop in which fear of default *causes* default by making refinancing too expensive. The crisis spread to Italy and Spain and threatened the euro itself, until the European Central Bank's Mario Draghi promised in 2012 to do "whatever it takes," and bond yields collapsed back down on the strength of that single sentence. It is the purest demonstration that for a government, the bond market is not a sideshow — it is the line between solvency and crisis.

## When this matters to you and where to go next

You may never buy a single bond directly. But the bond market touches your life every day anyway — through the rate on your mortgage and car loan, through the value of the stocks in your retirement account, through the strength of your currency when you travel, and through the size of the deficit your government runs in your name. Learning to watch the bond market is learning to see the force behind all of those.

Concretely, here is what to do with this post. When you next consider buying a home, look up the 10-year Treasury yield first; it will tell you more about where mortgage rates are heading than any headline about house prices. When your stock portfolio swings sharply on a day with no company news, check whether yields moved — odds are the bond market repriced the discount rate and dragged equities along. When a country dominates the news with a budget or debt crisis, find the chart of its government bond yields, because that is the scoreboard the market keeps and it usually moves before the politics resolves. Each of these is the influence thesis in miniature, and once you have trained your eye to look at the price of money first, you will see the bond market quietly running underneath nearly every financial story you read.

The next post in this series, **Track A2**, zooms into the bond contract itself — par, coupon, maturity, and the fine print of the indenture — line by line. From there we build pricing (Track B), risk (Track C), the yield curve (Track D), and onward. If you want the heavier mathematics of how a bond is priced, the [bond pricing deep-dive](/blog/trading/quantitative-finance/bond-pricing) in the quantitative-finance series develops the discounting machinery in full.

For now, hold on to the one picture from Figure 1 and the one sentence that started this post: a bond is a loan you can buy and sell, its interest rate is the price of money, and that price sets every other price. Everything else in fixed income is a refinement of that idea — and over the next 41 posts, we are going to refine it all the way down to the gears.

*This series is educational, not financial advice. It explains how the bond market works and how its forces propagate; it does not recommend buying or selling any security.*
