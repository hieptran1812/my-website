---
title: "Archegos and Bill Hwang: Concentration, Ego, and Secrecy"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "How a family office built a hidden $160bn book of concentrated bets through total-return swaps, why overconfidence and secrecy removed every external brake, and the concentration-and-secrecy protocol that catches it before the margin call does."
tags:
  [
    "trading-psychology",
    "archegos",
    "bill-hwang",
    "concentration-risk",
    "leverage",
    "total-return-swaps",
    "margin-call",
    "overconfidence",
    "risk-management",
    "behavioral-finance",
  ]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Archegos was not a strange, unlucky accident. It was overconfidence, concentration, and secrecy compounding on borrowed money until one ordinary price drop turned into the fastest wipeout of a large fortune in modern market history.
>
> - **A family office turned roughly \$1.5bn into ~\$160bn of hidden market exposure** by renting stock through total-return swaps at several banks at once — so each lender saw only its own slice, and the public saw nothing.
> - **Extreme concentration felt like conviction.** A handful of names — ViacomCBS above all — made up the book. Concentration is what makes a levered position lethal: it removes the diversification that would otherwise cushion a shock.
> - **Secrecy removed the external checks.** A family office answers to no outside investors, files no 13F on swap positions, and — spread across brokers — nobody could aggregate the risk. Ego kept it that way.
> - **The math is brutal and simple.** At ~5x leverage, a 20\% drop erases 100\% of your equity. A concentrated single stock can fall that far in two days — and in late March 2021, one did.
> - **When it broke, it broke in days.** Around \$20bn of Bill Hwang's wealth evaporated in roughly two trading sessions; the banks that were slow to sell lost more than \$10bn combined, led by Credit Suisse (~\$5.5bn) and Nomura (~\$2.9bn). Hwang was convicted of fraud in 2024 and sentenced to 18 years.
> - **The lesson is a protocol, not a moral.** Aggregate your true exposure, cap concentration in advance, treat secrecy as a red flag rather than an edge, and count ego itself as a position-sizing risk.

In late March 2021, a man most of the investing public had never heard of lost around \$20 billion in about two days. Not a fund's assets under management — his own paper fortune. Bloomberg would later call it one of the fastest destructions of personal wealth in history. The banks that had lent to him lost more than \$10 billion between them. Two of the biggest, Credit Suisse and Nomura, would spend the next two years explaining the hole to their regulators, and one of them would not survive the decade intact.

The man was Bill Hwang, and his firm was Archegos Capital Management. What makes Archegos worth studying is not that it was exotic. It is that it was a textbook — a clean, almost laboratory-pure demonstration of how three ordinary human tendencies, given enough leverage and enough silence, turn a good investor into a systemic event. Those three tendencies are **concentration**, **ego**, and **secrecy**. This post is about how they fit together, why they feel so reasonable from the inside, and what a disciplined trader actually does to keep from becoming the next case study.

We will build the whole machine from parts. If you have never heard the terms *total-return swap*, *prime broker*, or *margin call*, you will by the end — and you will understand not just what they mean but why they were the exact tools this particular psychology needed.

![Diagram of Archegos spread across five prime brokers, each seeing only its own slice while the true concentrated book reaches $160bn](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-1.webp)

Picture this as the mental model for everything that follows. One entity, Archegos, sits at the center. It faces five or more banks at once. Each bank — Credit Suisse, Nomura, Morgan Stanley, Goldman Sachs, UBS — holds a chunk of the same trade and sees only its own chunk. On the far side, all those chunks point at the *same* small handful of stocks. No single lender ever saw the full ~\$160bn. The public never saw any of it. That structure is the whole story in one picture: enormous, concentrated, and invisible.

## Foundations: the building blocks

Before we can talk about what went wrong, we need the vocabulary. Every term here gets defined from zero. A professional can skim; a beginner should read straight through, because the later sections lean on all of it.

### What a family office is (and why it matters)

A **family office** is a private company that manages the wealth of a single family. That is the entire definition. It has no outside clients, no public shareholders, no depositors. Bill Hwang founded Archegos as a family office in 2013 to manage his own money — reportedly a few hundred million dollars to start.

Why does this matter? Because regulation scales with *whose* money is at risk. A hedge fund manages other people's money, so it must register, disclose, and answer to investors who can pull their capital and ask hard questions. A family office managing only its founder's money faces far lighter rules. It does not have to tell the public what it owns. It has no investors performing due diligence. It has, in short, almost no external eyes.

That is not automatically sinister — thousands of family offices operate quietly and prudently. But hold onto the point: a family office is a structure with the checks turned *off* by design. Whether that is safe depends entirely on the discipline of the person inside it.

There is one more piece of backstory that matters. Before Archegos, Hwang ran a hedge fund called Tiger Asia Management — he was a "Tiger cub," a protégé of the legendary investor Julian Robertson. In 2012, Tiger Asia and Hwang settled U.S. insider-trading charges tied to Chinese bank stocks, paying \$44 million (SEC, December 2012), and Hwang was effectively barred from managing outside money in some jurisdictions. The family office was, in part, the vehicle that let him keep trading his own capital with minimal oversight. The man who ended up with no external checks was a man who had already run into the ones that existed.

### What leverage is

**Leverage** means using borrowed money to control more of an asset than your own cash could buy. That is it. If you have \$100 and you borrow \$400, you can control \$500 of stock. Your leverage is 5-to-1, usually written 5x.

Leverage is a magnifier, and — this is the part people forget when it is working — it magnifies in *both* directions, symmetrically.

#### Worked example: what 5x leverage really does

Suppose you have \$100,000 of your own money. You borrow another \$400,000 and buy \$500,000 of a single stock. You are 5x levered.

- The stock rises 10\%. Your \$500,000 position is now worth \$550,000. You repay the \$400,000 loan and keep \$150,000. On your \$100,000 you made \$50,000 — a **50\%** gain. Leverage turned a 10\% move into a 50\% return.
- The stock falls 10\%. Your position is worth \$450,000. After repaying \$400,000, you have \$50,000 left. You lost \$50,000 — a **50\%** loss. The same 10\% move, in reverse, cut your equity in half.

Notice the multiplier is the same number, 5, in both directions. There is no version of leverage that amplifies your gains and gently forgives your losses.

> The intuition here: leverage does not make you a better investor. It multiplies whatever you already are — and it multiplies the losses at exactly the same rate as the wins.

### What a prime broker is

A **prime broker** is the bank that services a big trader. It lends the trader money and securities, holds the positions, clears the trades, and — crucially — demands collateral to protect itself against the trader's losses. Goldman Sachs, Morgan Stanley, Credit Suisse, Nomura, and UBS were all prime brokers to Archegos.

A prime broker is a lender, and like any lender it wants a cushion. The cushion is called **margin**.

### What margin and a margin call are

**Margin** is the collateral you post — your own money set aside — so the broker knows it can be made whole if your position moves against you. If you control \$500,000 of stock with \$100,000 of your own money posted as margin, your margin is 20\%.

A **margin call** is the moment the broker looks at your shrinking position, decides the cushion has gotten too thin, and demands more collateral *right now*. If you cannot post it, the broker sells your position to protect itself. You do not get a vote. This is the single most important mechanism in the whole story, and we will return to it in detail.

### What concentration is

**Concentration** is the opposite of diversification. A diversified book spreads money across many uncorrelated bets, so no single one can sink you. A concentrated book puts a large share of the money into a few names — sometimes one. Archegos was extraordinarily concentrated: a handful of media and technology stocks, with ViacomCBS (now Paramount) the largest single exposure, reportedly around \$20bn — roughly a third of that company's equity value (per Bloomberg and Reuters reporting, 2021), controlled by one family office.

Concentration is not inherently wrong; many great fortunes were built by betting big on a few things. But concentration and leverage together are a specific, dangerous compound, and understanding *why* is most of this post.

### What a 13F is

A **13F** is the quarterly filing in which large investment managers must disclose their U.S. stock holdings to the SEC. It is how the public learns what big funds own. Here is the loophole that mattered: a 13F covers *shares you own*. If you get your stock exposure through a swap — where the bank owns the actual shares and you own only the profit and loss — you may have no shares to report. The position becomes invisible to the 13F system. Keep that in mind; it is the hinge the secrecy turned on.

With that vocabulary in hand, we can now look at the instrument that made the whole thing possible.

## The instrument that hid everything: total-return swaps

Everything unusual about Archegos flows from one contract: the **total-return swap**, or TRS. If you understand this instrument, you understand how a few hundred million dollars became \$160 billion of exposure that nobody could see.

![Diagram of a total-return swap: Archegos posts margin and a financing fee, the broker buys and holds the shares and passes back the total return, so no 13F is filed](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-3.webp)

Here is how the contract actually works, step by step, following the figure above. A total-return swap is a contract between a trader (Archegos) and a bank (the prime broker) that lets the trader get all the economic experience of owning a stock *without ever owning the stock*.

1. Archegos posts a slice of collateral — margin — often around 15–20\% of the position's value.
2. The bank goes into the market and buys the actual shares with its own money. The bank is now the legal, registered owner of the stock.
3. The two sides agree to *swap* cash flows. Archegos pays the bank a financing fee (an interest rate plus a spread) for putting up the money. In return, the bank pays Archegos the stock's **total return** — every dollar of price gain and every dividend if it rises, and Archegos owes the bank every dollar if it falls.

The intuition here is worth stating plainly: the swap splits a stock into two halves and hands each party a different one. The bank keeps the shares, the voting rights, and the disclosure duty. Archegos keeps the entire profit and loss — the *economics* — plus the leverage. Ownership and exposure, normally fused in a single share of stock, are pulled apart.

Three consequences fall out of that split, and each one fed the disaster.

**First, it manufactures leverage.** Because Archegos only posted ~15–20\% margin, it controlled roughly five to seven times its cash in market exposure — the same 5x magnifier from the worked example above, but built into the contract itself.

**Second, it manufactures invisibility.** Since the *bank* owns the shares, Archegos owns no shares to disclose. No 13F. The public — and, more importantly, other banks — could not see the position by looking at filings. The stock never touched Archegos's books.

**Third, and most subtly, it fragments the risk across lenders.** Archegos ran the *same* swap strategy at Credit Suisse, at Nomura, at Morgan Stanley, at UBS, at Goldman. Each bank saw its own book — Credit Suisse's own exposure to Archegos swaps reportedly reached around \$9.5bn (Credit Suisse special-committee report, July 2021) — but no bank saw the sum. Each lender thought it was facing a large, aggressive, but manageable client. None knew it was one of six lenders to a ~\$160bn concentrated bet.

#### Worked example: how a swap turns \$150m into \$1bn of hidden exposure

Suppose Archegos wants \$1,000,000,000 of exposure to a single stock, and the bank requires 15\% margin.

- Archegos posts \$150,000,000 of collateral.
- The bank buys \$1,000,000,000 of the shares with its own cash and holds them in its own name.
- Archegos now receives the full return on \$1bn of stock while having put up \$150m. That is about **6.7x** leverage on that trade.
- Because the bank is the shareholder of record, Archegos reports *no* position. And if Archegos does the identical thing for \$1bn at a second bank, each bank sees a \$1bn client relationship; the true \$2bn — concentrated in the same stock — is visible to no one.

Now scale that from \$1bn to \$160bn across five or six banks, in fewer than ten names. That is Archegos.

> A total-return swap is leverage and invisibility sold together in one contract — you rent the exposure, and the disclosure stays with the landlord.

There is nothing illegal about a total-return swap itself; they are a standard, useful instrument used across finance every day. What turned this one toxic was the *purpose* it served here: to build a position so concentrated and so large that, had anyone been able to see it whole, they would have stopped it. The instrument did not create the psychology. It served it.

## Concentration as conviction: the leverage that grew in the dark

Great investors are often concentrated. They find a few things they understand deeply and bet on them. Conviction — the willingness to size up when you are right — is a genuine edge. So the dangerous idea does not arrive as an obvious error. It arrives dressed as a virtue.

The trap is that concentration *feels* identical to conviction from the inside. Adding to your biggest winner feels like discipline, like backing your best idea. What the feeling hides is that concentration is the specific ingredient that converts leverage from risky into lethal, because it removes the one thing that saves a levered book in a shock: diversification.

![Before-and-after diagram showing Archegos growing from $1.5bn equity and $10bn exposure in March 2020 to $36bn equity and $160bn exposure in March 2021](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-2.webp)

In essence, the figure shows the position eating itself larger over twelve months. According to the SEC's later account, Archegos went from a value of roughly \$1.5bn with about \$10bn of exposure in March 2020 to a value of more than \$36bn with about \$160bn of exposure at its peak in March 2021. The equity grew because the concentrated bets were winning; the exposure grew even faster because every gain was recycled into more leveraged swaps. A winning streak did not make the book safer. It made it bigger, more concentrated, and more fragile — all at once, and all while feeling like success.

Here is why diversification is the thing that matters. If you are 5x levered across 100 uncorrelated stocks, a bad day in any one of them barely moves your book; they rarely all fall together. But if you are 5x levered in essentially one theme — a few correlated media and tech names — then a single bad catalyst hits your *entire* book at once. Concentration means your positions all breathe together. When one lung fails, they all do.

Archegos also had a second-order effect on its own concentration. Because its swap buying was so large relative to the size of some of these stocks, Archegos's own purchases helped push those stocks up. The rising prices validated the thesis and enlarged the collateral, which supported more swaps, which bought more stock, which pushed prices higher still. That reflexive loop is intoxicating on the way up. It is also exactly the loop that runs in reverse on the way down, at the same speed.

### What this costs, and when it breaks

The cost of concentration is invisible until the moment it is catastrophic. For months or years the concentrated, levered book simply outperforms — the diversified investor next door looks timid by comparison. Then a single catalyst arrives, hits every position simultaneously, and the leverage that felt free presents its bill all at once. The break does not come gradually. It comes on one specific afternoon.

## Why nobody could add it up: the aggregation problem

The single most important question a risk manager can ask is: *what is our total exposure to this?* Archegos was arranged — deliberately or not — so that no one could answer it. Understanding why is worth one more mechanism, because the aggregation problem is what turns a large risk into an invisible one.

A bank's risk desk models each client on the exposure *it* can see. To Credit Suisse, Archegos looked like a big, aggressive client with a swap book on the order of \$9.5bn (per Credit Suisse's July 2021 special-committee report) — large, but a size a bank of its scale believed it could manage. To Nomura, Archegos was a different but comparably sized relationship. Each desk stress-tested its own slice and concluded, reasonably, that it could survive a bad scenario. None of them was wrong about its own book. They were all wrong about the same thing: the book was not one of six; it was one book, cut into six pieces.

#### Worked example: six "safe" slices, one lethal whole

Say six banks each extend Archegos \$2,000,000,000 of exposure to the same basket of stocks, each requiring 15\% margin, and each judging \$2bn a comfortable relationship.

- Each bank sees \$2bn and stress-tests a 20\% drop: a \$400,000,000 loss — painful, but survivable for a large bank.
- The true position is 6 × \$2bn = \$12,000,000,000 concentrated in the *same* names. Scale each slice up toward the roughly \$27bn-per-broker average behind Archegos's real ~\$160bn, and the identical arithmetic runs at the size that actually broke the banks.
- When the drop comes, all six banks call for margin *at the same moment*, on the *same* falling stocks, and all six try to sell the *same* names into the same shrinking pool of buyers. The slices that looked independent turn out to be perfectly correlated — because they were always one trade.

The lesson: risk that is individually acceptable to each lender can be collectively fatal, and the party best placed to see the whole — the borrower — is the one with the least incentive to reveal it.

Two things defeated aggregation here. First, the swaps meant no public filing tied the pieces together: no 13F, no paper trail linking one bank's book to another's. Second, competition — prime brokers do not share client exposures with each other, because those relationships are confidential and lucrative. Each bank's blindness was, from its own seat, perfectly rational. The blindness only became catastrophic when it was summed.

> A risk nobody can total is a risk nobody can manage. The number you refuse to compute is the number that ends you.

## Leverage decides the size of the drop that ends you

Let us make the fragility precise, because the arithmetic is the whole reason this ended the way it did. The question every levered trader should be able to answer instantly is: *how far can my positions fall before I am wiped out?* Leverage answers it with a single division.

![Chart plotting equity remaining against the adverse move in the stock, showing the unlevered line losing 25% on a 25% drop while the 5x line hits zero at a 20% drop](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-5.webp)

Picture the two lines. The gentle dashed line is an unlevered investor: if the stock falls 25\%, they lose 25\% of their equity — painful, survivable. The steep solid line is a 5x-levered book: it hits zero equity at a mere 20\% drop. The gap between the lines, widening as the move deepens, is the pure cost of leverage. And the point where the steep line touches the floor is the number that matters most: the move that ends you.

#### Worked example: the drop that erases a 5x book

The rule is a one-line division. At leverage *L*, the adverse move that wipes out your equity is about 100\% divided by *L*.

- At **2x** leverage, a 50\% drop erases you. Stocks rarely halve in a day, so you have room.
- At **3x**, a 33\% drop erases you.
- At **5x**, a **20\%** drop erases you.
- At **7x**, a **~14\%** drop erases you.

Now put Archegos's real numbers in. With roughly \$36bn of equity supporting about \$160bn of exposure — call it ~4.4x — the wipeout move was around 100\% ÷ 4.4 ≈ **23\%**. A 23\% fall across the book, and the equity is gone.

> The intuition: leverage does not just raise your risk in some vague way. It sets an exact number — the percentage drop that takes you to zero — and the more levered you are, the smaller and more ordinary that number becomes.

Here is where concentration and leverage finally shake hands, fatally. A diversified book almost never falls 23\% in a few days — its hundred names do not all crater together. But a *concentrated* book, packed into a few correlated stocks, absolutely can. A single high-flying stock can drop 20–30\% in two sessions on one piece of bad news. Concentration is what makes the wipeout move *reachable*. Leverage sets the trap; concentration walks the book into it.

That is not a hypothetical for Archegos. It is precisely what happened in the last week of March 2021.

## The margin-call death spiral

We now have every part needed to watch the collapse mechanically. When a concentrated, levered book starts to fall, it does not decline in a smooth, orderly way. It falls into a self-reinforcing spiral, and the spiral runs faster than any human can react.

![Three-step pipeline of the margin-call death spiral: concentrated stock falls and breaches margin, the call goes unmet and forces liquidation, block sales drive the price down and trigger a bigger call](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-4.webp)

The way it works is a loop with three stations, and each pass around the loop makes the next pass worse:

1. **The concentrated stock falls.** Because the book is levered, even a moderate drop — say 20–25\% — breaches the margin the broker requires. The cushion is gone.
2. **The call goes unmet, and the broker liquidates.** The broker demands more collateral. A book that is already fully deployed and concentrated cannot produce billions in fresh cash on a few hours' notice. So the broker exercises its right to sell the position to protect itself.
3. **The forced sale drives the price down further.** Dumping a huge block of a stock into the market pushes its price down — and that lower price breaches margin on the *rest* of the book and at the *other* brokers, triggering the next, larger call. Back to station one, bigger.

#### Worked example: the margin call that could not be met

Suppose the book is \$36,000,000,000 of equity supporting \$160,000,000,000 of concentrated exposure. The stocks in it fall 20\% over a couple of sessions.

- A 20\% fall on \$160bn of exposure is a loss of **\$32,000,000,000**.
- That loss comes straight out of equity: \$36bn − \$32bn = **\$4bn** of equity left, against a book still notionally around \$128bn. Leverage has just exploded from ~4.4x toward ~30x.
- The brokers, seeing the cushion nearly gone, call for many billions in fresh margin — simultaneously, because they are all watching the same falling stocks.
- There is no \$32bn of spare cash. There cannot be; it was all deployed into swaps. The calls go unmet. The brokers begin to sell.

The intuition to carry out of this: a concentrated, levered book has no gentle way down. The same drop that vaporizes the equity also removes any ability to meet the call, which forces the selling that deepens the drop. It is not a slope. It is a trapdoor.

#### Worked example: the fire-sale tax

Now add the liquidity cost, because it is what turned a large loss into a total one for the slow banks. Suppose the brokers must collectively sell around \$20,000,000,000 of stock concentrated in a few names, over a few days.

- A stock trades a certain amount per day — its liquidity. Trying to sell many times the normal daily volume forces you to accept lower and lower prices to find buyers.
- Say the forced selling pushes the average sale price 15\% below where the stock started the week. On \$20bn of stock, that discount alone is another **\$3,000,000,000** of value destroyed purely by the *act of selling in size*.
- And every block sold cheaper marks down the shares still unsold — and the shares still held at the slower brokers, who are now competing to sell the same names into the same shrinking pool of buyers.

The lesson in one sentence: forced selling pays a liquidity tax, and in a concentrated position that tax compounds — the faster you must sell, the worse the price, and the worse the price, the more you must sell.

This is why *speed* determined who survived. Goldman Sachs and Morgan Stanley moved first, dumping their blocks early while buyers still existed, and largely escaped. Credit Suisse and Nomura moved slower, and sold into a market the fast banks had already crushed. Same client, same trade, wildly different outcomes — decided by who hesitated.

## Ego and secrecy: how the brakes were removed

We have built the machine. Now the human question: why did nobody stop it? A position this large, this concentrated, this levered, is exactly the kind of thing that risk officers, investors, regulators, and lenders exist to catch. Every one of those brakes failed — and they failed because ego and secrecy had, one by one, disabled them.

![Diagram showing four psychological drivers each disabling a specific external check, letting the position grow to $160bn unchecked](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-8.webp)

Conceptually, the figure maps four ordinary psychological drivers onto the four external checks each one switched off. This is the core of the case, so take the mapping seriously — it is the transferable lesson.

**Overconfidence removed the internal trim.** When you are certain you are right — and Hwang had years of extraordinary returns telling him he was — you do not trim your winners. Conviction becomes size, and size becomes the whole book. The first brake on any position is the trader's own willingness to say "that is enough." Overconfidence disables it. (This is the mechanism explored in depth in [overconfidence and the illusion of control](/blog/trading/trading-psychology/overconfidence-and-the-illusion-of-control): the more control you *feel*, the less you hedge against the control you *lack*.)

**Concentration-as-conviction removed the diversification cushion.** Believing that his few names were special, Hwang held few names. So there was no diversification left to absorb a shock — the very cushion that lets a levered book survive a bad week was gone by design.

**Ego and secrecy removed the outside eyes.** A family office has no investors demanding transparency, no allocators running due diligence, no board asking why one position is a third of the book. Hwang had already had run-ins with regulators; the structure he built afterward answered to almost no one. Secrecy was not a side effect here. For a man who did not want to be second-guessed, it was the point.

**Hidden leverage removed the felt cost.** Because the leverage was embedded in swaps and spread across banks, it did not feel like debt. There was no monthly loan statement, no single lender totting up the total and getting nervous. Leverage that you cannot see does not frighten you — until the margin call, which is the first and last time you feel it.

Stack those four together and you get the figure's punchline: with every external check disabled, the position could grow to \$160bn with nothing and no one positioned to say stop. The brokers each saw a slice. The regulators saw no 13F. The investors did not exist. And the one person who *could* see the whole thing was the one person whose ego most needed it to keep growing.

> Secrecy feels like an edge — nobody can copy a trade they cannot see. But the people you are hiding from include the ones who would have caught your mistake. You cannot selectively hide only your genius.

This is the deep link between the parts. Concentration was the *shape* of the risk. Leverage was the *size* of it. Secrecy was what kept anyone from *seeing* it. And ego was the engine that chose concentration, chose leverage, and chose secrecy — not by accident, but because each one felt like sophistication rather than danger.

## What it looks like at the screen

Case studies can make a blowup feel like something that happens to other, obviously reckless people. It is more useful to describe how this actually feels from the inside, at the desk, day to day — because the tells are quiet, and they feel *good*. If you trade, some of these will be uncomfortably familiar. That discomfort is the point.

**The P&L only goes up, and you stop reading the risk report.** In a concentrated, winning, levered book, the account value climbs almost every day. The pleasure of watching it becomes the main feedback loop. The risk report — the boring page that says "you are 4x levered and 70\% in three names" — becomes something you glance at and dismiss, because it keeps warning about a disaster that keeps not happening. The absence of the disaster feels like proof you were right to ignore the warning.

**You have a favorite position, and you check it first.** There is one name you look at before anything else each morning — the big one, the one you *understand*. You know its story better than the analysts do. Adding to it on a dip feels like loyalty and insight at once. What you have stopped noticing is that this single name now determines whether you have a good year, and that your intimate knowledge of its story does nothing to protect you from a catalyst you did not predict.

**The broker's risk desk calls, and you find it slightly insulting.** Someone junior at a bank asks you to post more margin or trim an exposure. You are one of their most important clients; you have made them a fortune in fees. The request feels like small-minded box-ticking from people who do not see the big picture the way you do. So you push back, or you move some of the position to a broker who asks fewer questions. Every time you do this, you are personally disabling one of the external checks.

**Nobody knows how big you really are, and that feels like power.** You take a private satisfaction in the fact that the market cannot see your hand. Other traders would be stunned if they knew your true size. You read this secrecy as a competitive moat. You do not read it as the reason nobody is positioned to warn you before it is too late.

**Leverage feels like room, not like debt.** Because it is embedded in swaps and spread across counterparties, your borrowing never shows up as a scary number in one place. You experience it as *capacity* — the ability to put on the next great trade — rather than as an obligation that will be called at the worst possible moment. The first time it feels like debt is the morning the calls come in, all at once, and there is no cash to meet them.

**You argue with the risk report instead of obeying it.** When the weekly risk summary flashes a warning, your instinct is no longer to reduce the position but to explain the warning away — the model is too crude, the correlation assumption is stale, the scenario is implausible, the desk that built it does not understand your names the way you do. You have begun arguing with the smoke detector instead of checking for fire. The moment risk management becomes something you rationalize rather than something you obey, it has already stopped protecting you — and you will not notice, because being talked out of a warning feels exactly like being right.

None of these feelings is stupidity. Each one is a reasonable response to a genuine winning streak. That is exactly why they are dangerous: the psychology of a blowup does not feel like recklessness from the inside. It feels like being good at your job.

## Common misconceptions

**"Archegos blew up because Hwang was a bad investor."** He was, by most accounts, a genuinely talented stock picker with years of strong returns. That is the unsettling part. Skill at picking stocks and skill at *sizing and financing* those picks are different skills. Archegos is a monument to being excellent at the first and catastrophic at the second. Good analysis got magnified by bad risk structure until the risk structure was all that mattered.

**"The swaps were the villain."** Total-return swaps are ordinary, useful instruments used safely across finance every day. They did not cause the blowup; they were the tool a particular psychology reached for because it wanted leverage and invisibility. Blaming the instrument misses the lesson. The same result could have been built other ways. The driver was concentration, ego, and secrecy — the swap just made all three easy.

**"Diversification would only have lowered his returns."** Diversification does dampen returns in the good years — that is precisely what makes concentration tempting. But the correct comparison is not "concentrated returns vs diversified returns in a good year." It is "concentrated returns that occasionally go to zero vs diversified returns that survive." A strategy that produces spectacular numbers and then a total loss has a long-run return of negative 100\%. Surviving is not a drag on compounding; it is the precondition for it.

**"The banks were just victims of a rogue client."** The banks chose to extend enormous leverage to a single client on concentrated positions, competed with each other to do so for the fees, and — per Credit Suisse's own later report — failed to force Hwang to post adequate collateral or to grasp the concentration they were financing. The client hid the aggregate, yes. But each bank could see its own slice was dangerously large and lent anyway. Greed for prime-brokerage fees was its own failed brake.

**"The post-Archegos reforms fixed this."** Regulators did propose more disclosure for large swap positions and family offices after 2021, and some rules tightened. But a family office managing only its principal's money still faces lighter oversight than a fund with outside investors, and swap exposure is still harder for anyone to aggregate than plain share ownership. The structural blind spots that let Archegos hide are narrower than they were — not closed. Treat any confident claim that "the system won't allow it again" as the same overconfidence this whole post is about, moved up one level to the regulators.

**"This was a black swan — unforeseeable."** It was not. Everything about the mechanism was known. The margin-call death spiral is textbook. The wipeout arithmetic is a single division. The concentration was visible to each lender in its own book. What was hidden was only the *sum*, and the sum was hidden by choice, not by nature. The event was not unforeseeable; it was un-*aggregated*.

## How it shows up in real markets

The abstract mechanics only matter because they played out in real time, with real dates and real dollars. Here is the episode itself, and the pattern echoing across other blowups.

### 1. The week Archegos vanished (March 2021)

![Timeline of the week of March 22, 2021, from ViacomCBS near its peak through the failed stock sale, unmet margin calls, default, and the multi-day fire sale](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-6.webp)

The trigger was ViacomCBS. Through early 2021 the stock had roughly tripled — from around \$36 at the start of the year toward nearly \$100 by March 22 — helped along by Archegos's own swap buying. On March 22–24, ViacomCBS announced a roughly \$3bn stock sale (common shares priced at \$85, convertible preferred at \$100 on March 24). The market's response was tepid; a stock that had run up on scarcity suddenly faced fresh supply, and it fell. By the close on March 26, the newly issued shares were trading far below their offering prices — common around \$48, a loss of roughly half in days for those who bought the placement.

That fall breached Archegos's margin. On March 25 the calls came; Archegos could not meet them. On March 26 it defaulted, and the prime brokers began the fire sale — more than \$20bn of block trades in a handful of concentrated names, dumped into a market that quickly understood what was happening. Goldman and Morgan Stanley sold first and fast. Credit Suisse and Nomura were slower. Within days, a ~\$36bn book was gone, and with it around \$20bn of Bill Hwang's personal paper wealth — a destruction of fortune so fast it has few precedents. (These figures trace to the SEC's 2022 complaint and contemporaneous reporting from Bloomberg, Reuters, and the Financial Times; see Sources.)

### 2. Who lost what — speed decided everything

![Matrix of the prime brokers showing that banks which sold first lost near zero while the slow ones — Credit Suisse and Nomura — absorbed the bulk of the more-than-$10bn combined loss](/imgs/blogs/archegos-and-bill-hwang-concentration-ego-and-secrecy-7.webp)

The combined losses to the banks exceeded \$10bn (per Reuters and Financial Times tallies, 2021), but they were distributed almost entirely by reaction speed. Credit Suisse reported the largest hit, close to \$5.5bn, and its own special-committee investigation (July 2021) described a prime-services culture that had failed to rein in the client or demand adequate collateral. Nomura's loss came to roughly \$2.9bn. Morgan Stanley disclosed around \$0.9bn, UBS around \$0.8bn, with smaller losses at Mitsubishi UFJ and others (as reported in the banks' 2021 earnings disclosures). Goldman Sachs, which sold aggressively and early, reported an immaterial loss.

The distribution is the lesson. Every bank had lent to the same client on the same concentrated trade. What separated a rounding error from a multi-billion-dollar wound was who recognized the run first and sold without hesitation. For Credit Suisse the damage compounded existing troubles and fed the crisis of confidence that, two years later, ended in its emergency takeover by UBS. A single client relationship, mispriced for risk, helped topple a 167-year-old bank.

### 3. The legal aftermath (2022–2024)

In April 2022, the SEC charged Hwang and Archegos with a sweeping market-manipulation and fraud scheme, and the Department of Justice brought parallel criminal charges. Prosecutors argued that Hwang had not merely taken risk but lied to his banks about the size and composition of his positions so he could borrow more aggressively. After a trial lasting roughly two months, Hwang was convicted in July 2024 of racketeering conspiracy, securities fraud, and market manipulation. In November 2024, a federal judge sentenced him to **18 years** in prison — an unusually long white-collar sentence — and he was ordered to pay restitution exceeding \$9bn. His former chief financial officer, Patrick Halligan, was also convicted. (Sources: DOJ and SEC releases, 2022 and 2024.)

The criminal case reframes the psychology. This was not only overconfidence and concentration; the government proved deception — telling different banks different things so that none could assemble the true picture. Secrecy, in the end, was not just a personality trait. It was the crime.

### 4. The pattern echoes: LTCM and the family of blowups

Archegos rhymes with earlier disasters, which is why the pattern is worth memorizing rather than the single case. Long-Term Capital Management in 1998 was run by Nobel laureates with brilliant models, enormous hidden leverage across many counterparties, and concentrated correlated bets. When markets moved against it, the same aggregation problem appeared — no single counterparty saw the whole book — and the same spiral followed, requiring a Fed-organized rescue. Different instruments, different decade, identical skeleton: concentration, hidden leverage, and a book too large and too opaque for anyone to see whole until it broke. The names and tools change. The mechanism does not.

### 5. The family-office loophole, still open

Archegos prompted regulators to propose more disclosure for large swap positions and family offices, and some rules tightened. But the structural fact remains: a family office managing only its principal's wealth still operates with fewer external checks than a fund managing outsiders' money, and swap-based exposure is still harder to aggregate than share ownership. The conditions that let Archegos hide are institutional, not personal. Which means the defense has to be personal — a discipline you impose on yourself, because the system will not reliably impose it for you.

### 6. The bubble Archegos inflated in its own names

The reflexive loop deserves its own entry, because it shows how a concentrated buyer distorts the very prices it is betting on. Through late 2020 and early 2021, Archegos's relentless swap buying was a large part of the demand in names like ViacomCBS and Discovery. That buying pushed the prices up; the higher prices enlarged Archegos's collateral, which supported more swaps, which bought more stock, which pushed the prices higher still. ViacomCBS roughly tripled in the first months of 2021 — from around \$36 toward nearly \$100 by March 22 — a move far larger than any change in the company's fundamentals, inflated in part by one hidden, price-insensitive buyer.

The trap in being the marginal buyer of your own position is that you cannot tell how much of the price is *you*. Archegos's book looked more valuable, and its collateral looked healthier, precisely because Archegos kept buying — a mirror reflecting its own demand back as apparent conviction. When ViacomCBS issued new shares into that thin, propped-up demand and the price cracked, the mirror shattered and the loop ran backward at the same speed it had run forward. The lesson generalizes past this one fund: a concentrated buyer large enough to move a price is exposed not just to the stock, but to their own footprint in it — and that footprint disappears the instant they need to sell, which is exactly when they need it most.

## The drill: a concentration-and-secrecy protocol

Reading about a blowup is entertainment unless it changes what you do on Monday. Here is the protocol — four checks, drawn directly from the four failures above. It is written for a trader of any size, because the psychology is identical at \$10,000 and \$36bn; only the scale of the wreckage differs. This is educational, not individualized financial advice.

### Check 1 — Aggregate your true exposure

The first thing that killed Archegos was that nobody, including possibly Hwang, could see the whole position in one number. Do for yourself what no one did for him: build the aggregate.

- Once a week, write down your **total economic exposure** to each underlying — not per account, not per broker, not per instrument, but summed across everything: shares, options (by delta), swaps, correlated names. Archegos's mistake was letting the number live in six places so it never got added up.
- Then divide by your equity to get your **true leverage**. If that number surprises you, you have found the exact blind spot that the swap structure created for Hwang. The surprise *is* the risk.
- Treat any exposure you cannot state as a single number as if it were larger than you think. Hidden size is always underestimated by the person hiding it.

### Check 2 — Cap concentration in advance, in writing

Concentration is a decision you must make *before* the winning streak, because during the streak it will feel like cowardice.

- Set a hard ceiling — a maximum share of the book any single name or correlated theme may occupy — and write it down while you are calm.
- When a position grows past the cap *because it is winning*, trim it back to the cap. This will feel wrong every single time. Do it anyway. The feeling that trimming your best idea is foolish is the precise feeling that ran Archegos to \$160bn.
- Remember the wipeout division: your concentration and your leverage together set the exact drop that ends you. Choose that number deliberately instead of discovering it during a margin call.

### Check 3 — Treat secrecy as a red flag, not an edge

Hwang's secrecy felt like an advantage right up to the moment it meant no one could warn him.

- Ask of any position: *who could catch my mistake here, and have I accidentally hidden it from them?* If the honest answer is "no one can see this," that is not a moat. It is the removal of your last brake.
- Keep at least one informed, honest outside view on your book — a partner, a risk-minded friend, a written pre-commitment you cannot quietly edit. The value of an external check is precisely that it is external; you cannot supply it to yourself, because the part of you that needs checking is the part doing the checking.
- Be especially suspicious when you find yourself *routing around* oversight — moving a position to the broker who asks fewer questions, choosing the structure that discloses less. That impulse is the disaster protecting itself.

### Check 4 — Count ego as a position-sizing input

The deepest lesson is that a long winning streak is a *risk factor*, not just a reward.

- After a run of wins, deliberately assume your confidence is now miscalibrated *upward*, and size down rather than up. The winner's high is real and it makes you bet bigger exactly when the regime that rewarded you is most likely to turn. (This is the emotional-regulation core of [position sizing as emotional regulation](/blog/trading/trading-psychology/position-sizing-as-emotional-regulation): the size of your bet should be governed by a rule, not by how right you currently feel.)
- Watch for the tells from the screen section — ignoring the risk report, resenting the broker's call, savoring your invisibility. Each is a signal that ego is now driving the size. Treat the *feeling* of invincibility as a sell signal on your own leverage, the same way [greed and euphoria](/blog/trading/trading-psychology/greed-euphoria-and-the-anatomy-of-a-blowup) mark the top of a blowup sequence.
- Ask the humbling question on your best day, not your worst: *if my biggest position fell 25\% tomorrow, would I survive?* If the answer is no, you are not running a portfolio. You are running Archegos, and you simply have not met your ViacomCBS yet.

Run these four checks and you are doing, for yourself, the aggregating and trimming and second-guessing that Archegos's structure switched off. The protocol is not sophisticated. Its difficulty is entirely emotional: every check asks you to act against a feeling that, during a winning streak, feels like wisdom.

## When this matters to you

You are almost certainly not about to build \$160bn of hidden swap exposure. But the shape of this mistake scales all the way down, and it will find you at whatever size you trade. The retail investor who puts their whole account into one stock they are sure about, on margin, has built a tiny Archegos. The founder whose entire net worth is their company's stock has a concentration problem the moment they borrow against it. Any time conviction, leverage, and a lack of outside eyes line up in the same position, you are standing in the same machine — the scale is different, the mechanism is not.

The reason this case study endures is that it isolates the psychology so cleanly. Strip away the billions and the swaps and the prison sentence, and what remains is a person who was genuinely good, who won for a long time, who let winning talk him into concentration, who used leverage he could not fully feel, and who arranged things so no one could tell him to stop. Every one of those moves felt intelligent in the moment. That is the warning. The blowup does not announce itself as recklessness. It arrives as a series of reasonable decisions, each one slightly bolder than the last, in a book that only ever seems to go up — until the week it does not, and there is no cash for the call, and no one who saw it coming, because you made sure of that yourself.

Aggregate the exposure. Cap the concentration. Distrust the secrecy. Size against the ego. None of it is hard to understand. All of it is hard to do while you are winning — which is, of course, exactly when it is the only thing that matters.

## Sources & further reading

- [SEC charges Archegos and its founder with market manipulation](https://www.sec.gov/newsroom/press-releases/2022-70) — SEC press release and complaint (April 2022): the source for the growth from ~\$1.5bn/\$10bn (March 2020) to ~\$36bn value / ~\$160bn exposure (March 2021), and the manipulation charges.
- [Founder and head of Archegos Bill Hwang sentenced to 18 years](https://www.justice.gov/usao-sdny/pr/founder-and-head-archegos-capital-management-bill-hwang-sentenced-18-years-prison) — U.S. Department of Justice (November 2024): conviction details, the 18-year sentence, and the more-than-\$9bn restitution.
- [SEC: hedge-fund manager Bill Hwang / Tiger Asia to pay \$44 million](https://www.sec.gov/newsroom/press-releases/2012-2012-264htm) — SEC press release (December 2012): the earlier insider-trading settlement that preceded the family-office structure.
- [Credit Suisse special committee report on Archegos](https://finance.yahoo.com/news/credit-suisse-report-5-5-122558493.html) — coverage of the ~170-page Paul, Weiss report (July 2021) documenting the ~\$5.5bn loss and the risk-management failures.
- Bank loss figures (combined over \$10bn; Nomura ~\$2.9bn; Morgan Stanley ~\$0.9bn; UBS ~\$0.8bn) — [The Irish Times summary](https://www.irishtimes.com/business/financial-services/total-bank-losses-from-archegos-implosion-exceed-10bn-1.4549192) and [Archegos Capital Management (Wikipedia)](https://en.wikipedia.org/wiki/Archegos_Capital_Management), cross-referenced with Reuters and Bloomberg reporting.
- ViacomCBS offering and price collapse — [CNBC coverage of the stock sales](https://www.cnbc.com/2021/04/01/viacomcbs-stock-sales-amid-archegos-debacle-raise-questions-for-banks.html) (April 2021), for the \$85/\$100 offering prices and the late-March fall.
- On this blog: [Overconfidence and the illusion of control](/blog/trading/trading-psychology/overconfidence-and-the-illusion-of-control), [Greed, euphoria, and the anatomy of a blowup](/blog/trading/trading-psychology/greed-euphoria-and-the-anatomy-of-a-blowup), and [Position sizing as emotional regulation](/blog/trading/trading-psychology/position-sizing-as-emotional-regulation) — the psychological mechanisms this case study puts on display.

*This article is educational and historical, not individualized financial advice. Figures are drawn from the sources above; live-market values move, and dated magnitudes reflect the reporting as of the events described.*
