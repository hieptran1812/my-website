---
title: "The Wall Street Crash of 1929 and the Great Depression: Margin, Bank Runs, and Policy Failure"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How leverage from buying stocks on margin caused the 1929 crash, and why the far worse Great Depression came from bank runs, a collapsing money supply, and a Federal Reserve that failed to act."
tags: ["great-depression", "1929-crash", "margin", "bank-runs", "monetary-policy", "gold-standard", "deflation", "federal-reserve", "financial-history", "fdic"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The 1929 crash was a leverage accident built on stocks bought with borrowed money, but the crash alone did not cause the Great Depression; a three-year cascade of bank runs, a one-third collapse in the money supply, and a passive Federal Reserve did.
>
> - In October 1929 the stock market fell about 25% in a few days, and by 1932 the Dow was down roughly 89% from its 1929 peak.
> - The crash was amplified by buying on margin: investors put down as little as 10% and borrowed the rest, so a 10% price drop could wipe out 100% of their money and force selling.
> - The real catastrophe was monetary: from 1930 to 1933 roughly 9,000 banks failed, the money supply shrank by about a third, and prices fell about 25%, turning a recession into a depression.
> - The Federal Reserve stayed passive while banks collapsed, there was no deposit insurance to stop runs, and the gold standard exported the deflation around the world; unemployment reached about 25%.
> - The durable lesson, learned the hard way: a central bank must be the lender of last resort and protect the money supply in a panic, which is exactly what policymakers did in 2008 and 2020 to avoid repeating 1929.

In the summer of 1929, a shoeshine boy was reportedly giving stock tips to Joseph Kennedy, and Kennedy decided that when the shoeshine boy is in the market, it is time to get out. Whether or not the story is literally true, it captures something real: by 1929, owning stocks had stopped being something a few rich families did and had become a national pastime, fueled by an idea that felt like magic. You did not need to pay for the stocks you bought. You could put down a tenth of the price, borrow the rest from your broker, and ride the gains on money that was not yours. When prices only ever went up, this looked like free wealth. When prices fell, it turned out to be a machine for manufacturing ruin.

The diagram below is the mental model for the whole story. It is tempting to picture the Great Depression as one event: the market crashed, everyone got poor. The truth is two events stitched together. The crash was fast and dramatic and over in weeks. The depression was slow, grinding, and largely the product of a different mechanism entirely, one that played out over the following three years while policymakers watched and did almost nothing useful. Keep that split in your head, because nearly every popular misconception about 1929 comes from collapsing the two into one.

![Timeline from the 1929 crash through the 1933 banking collapse to the New Deal reforms](/imgs/blogs/wall-street-crash-1929-great-depression-1.png)

This post does three things. First, it builds every concept from zero, so that even if you have never bought a stock you can follow exactly what broke. Second, it dissects the mechanism in two layers: the leverage that turned a market correction into a crash, and the policy failure that turned a recession into a decade-long depression. Third, it shows how the same machinery has reappeared, and how the lessons of 1929 directly shaped the very different responses in 2008 and 2020. This is financial history, not investment advice; the point is to understand the gears, not to time a market.

## Foundations: the building blocks you need first

Before the story makes sense, you need a small vocabulary. Each term below is something the 1929 disaster turns on. None of it requires prior finance knowledge; read it once and the rest of the post will read easily.

**A stock.** A *stock* (also called a *share*) is a slice of ownership in a company. If a company is divided into a million shares and you own one, you own one-millionth of the company, including a claim on a millionth of its future profits. Stock prices move with what buyers and sellers think those future profits are worth. In a calm market, a stock's price is roughly tethered to the company's earnings. In a mania, the price can float far above anything earnings justify, because people are buying not for the profits but because they expect to sell to someone else at a higher price.

**Buying on margin.** Normally if you want \$10,000 of stock you pay \$10,000. *Buying on margin* means you pay only part of the price in cash and borrow the rest from your broker, using the stock itself as collateral. In the 1920s the cash portion (the *margin*) could be as low as 10%. So with \$1,000 of your own money you could control \$10,000 of stock, with the broker lending the other \$9,000. The borrowed money is a *broker's loan* (also called a *margin loan*), and you pay interest on it.

**Leverage.** *Leverage* is the general name for using borrowed money to control a position larger than your own cash. We measure it as a ratio: \$10,000 of stock on \$1,000 of your money is *10-to-1 leverage*, or "10x." Leverage is a multiplier in both directions. If the stock rises 10%, you make \$1,000 on your \$1,000 stake, a 100% gain. If it falls 10%, you lose \$1,000 on your \$1,000 stake, a 100% loss. The same percentage move that is a rounding error for an unleveraged owner is total ruin for the leveraged one. This asymmetry is the engine of the 1929 crash.

**A speculative bubble.** A *bubble* is a period when an asset's price rises far above its fundamental value because people buy it expecting to sell it higher, not for the income it produces. Bubbles feed on themselves: rising prices attract buyers, whose buying pushes prices higher, which attracts more buyers. They are self-sustaining right up until the inflow of new buyers stops, at which point they reverse just as violently.

**A margin call.** When you buy on margin, the broker requires that your own equity stay above a minimum, the *maintenance margin*. If the stock falls and your equity shrinks below that line, the broker issues a *margin call*: a demand that you immediately deposit more cash to restore the cushion. If you cannot pay, the broker sells your shares to recover its loan, whether you like it or not. A margin call is the moment a paper loss becomes a forced, real sale.

**A bank run.** A *bank run* happens when many depositors try to withdraw their money at the same time because they fear the bank will fail. The cruel logic is that the fear is self-fulfilling: a bank cannot give everyone their money at once (we will see why in a second), so a run can topple even a healthy bank. Once one bank is seen failing, depositors at other banks rush to withdraw too, and the panic spreads.

**Fractional-reserve banking.** This is why runs are possible. When you deposit \$1,000 at a bank, the bank does not lock your cash in a vault. It keeps a small fraction as *reserves* (say \$100) and lends out the rest (\$900) to borrowers, earning interest. This is *fractional-reserve banking*, and it is how essentially every bank on earth works. It is wonderful in normal times and catastrophic in a panic, because the bank simply does not have enough cash on hand to repay all depositors at once. If everyone demands their money the same day, the bank is insolvent on the spot, no matter how sound its loans were.

**The money supply.** The *money supply* is the total amount of money circulating in the economy: physical cash plus the balances in checking and savings accounts. Most money is not cash; it is bank deposits. Crucially, when banks lend, they create new deposit money (a topic explored in depth in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier)). The flip side, which 1930 to 1933 demonstrated with brutal clarity, is that when banks fail and lending collapses, deposit money is destroyed and the money supply *shrinks*.

**The gold standard.** Under the *gold standard*, a country's currency is legally convertible into a fixed amount of gold, and the amount of money a country can issue is tied to how much gold it holds. The dollar in 1929 was defined as 1/20.67 of an ounce of gold. The gold standard was prized as a discipline against governments printing money, but it had a fatal property in a depression: it prevented a central bank from creating money to fight a collapse, because more money required more gold, and it chained countries together so that deflation in one was transmitted to all.

**Deflation, and how it raises real debt.** *Inflation* is rising prices; *deflation* is the opposite, a general fall in prices. Deflation sounds pleasant (things get cheaper) but it is poison for anyone who owes money. Debts are fixed in dollar terms. If you owe \$1,000 and prices fall 25%, the goods, wages, and revenue you would use to repay that \$1,000 are now worth 25% less, so the *real* burden of the debt has risen by a third even though the number on the loan never changed. This is the *real debt* effect, and it is the link that turned falling prices into a wave of defaults.

**The lender of last resort.** Because a run can kill a solvent bank purely through panic, modern systems give the central bank a job: in a crisis it lends freely to solvent banks against good collateral, so that no bank fails merely because it cannot get cash fast enough. This is the *lender of last resort* function, articulated by Walter Bagehot in 1873. The Federal Reserve was created in 1913 precisely to do this. In 1930 to 1933 it largely declined to.

**Deposit insurance (FDIC).** *Deposit insurance* is a government guarantee that your bank deposit is safe up to a limit even if the bank fails. The *Federal Deposit Insurance Corporation (FDIC)* was created in 1933 to provide it. Its genius is that it removes the *reason* to run: if you know your money is safe whether or not you rush to the teller, you do not rush, and the run never starts. It is one of the single most effective pieces of financial plumbing ever invented.

**Glass-Steagall.** The *Glass-Steagall Act* of 1933 separated ordinary commercial banking (taking deposits, making loans) from investment banking (underwriting and trading securities), on the theory that banks gambling depositors' money in the stock market had made the system fragile. It also created the FDIC. Much of it was repealed in 1999, a fact that returns in the 2008 echo.

With that toolkit, the rest of the story is mechanical.

## The setup: the Roaring Twenties and the leverage machine

The 1920s in America were a genuine boom. Electrification, automobiles, radio, and mass production created real, rapid economic growth, and corporate profits rose. A bull market on those foundations was not crazy. What turned a justified bull market into a mania was the structure of how people bought in.

By the late 1920s, stock ownership had spread far beyond the wealthy, and the favored way to buy was on margin. The mechanics were intoxicating. You opened a *brokerage* account, put down a fraction of the price, and the broker financed the rest with a *broker's loan*, charging you interest and holding your shares as collateral. Margin requirements of 10% were common, meaning \$1 of your money controlled \$10 of stock.

The scale of this borrowing was enormous and growing. *Brokers' loans*, the total amount lent to investors to buy stock on margin, roughly doubled in the two years before the crash, reaching around \$8.5 billion by October 1929, a staggering sum against the economy of the time. Banks lent to brokers, brokers lent to speculators, and out-of-market corporations even parked spare cash in the brokers' loan market because the interest rates were so attractive. The entire edifice rested on stock prices continuing to rise.

![Stack showing a margin position resting on a broker loan above a thin cash slice](/imgs/blogs/wall-street-crash-1929-great-depression-4.png)

The picture above is the leverage stack, and it is worth staring at. Your \$1,000 is the thin slice at the bottom. Above it sits \$9,000 of borrowed money that has to be repaid first, in full, no matter what the stock does. As long as the total position rises, the borrowed layer is fixed and every extra dollar of gain flows to your thin slice, magnifying your return spectacularly. But the borrowed layer is also a hard floor under which your equity cannot go without triggering a forced sale. When the position falls, your slice absorbs the entire loss first, and it is small.

A second, more exotic amplifier sat on top of the margin frenzy: the *leveraged investment trust*. An investment trust was an early version of a mutual fund, a company that pooled money to buy a portfolio of stocks. The twist in 1929 was that these trusts were themselves leveraged, and they often owned shares in other trusts, which owned shares in still other trusts. A trust might raise money by issuing its own bonds and preferred shares (debt-like claims) and use the proceeds to buy common stocks. Because the trust's debts were fixed, any rise in its portfolio flowed disproportionately to its common shares, just like personal margin. Layer trusts owning trusts owning trusts, and you get leverage stacked on leverage stacked on leverage, a pyramid where a modest move in the underlying stocks became a violent move at the top. Goldman Sachs Trading Corporation, launched in 1928, became the most infamous example; the structure that multiplied its gains in 1928 and early 1929 annihilated it on the way down.

The mania, in short, was self-feeding in the classic bubble pattern, and it was wired with leverage at every level: individuals on margin, trusts on debt, trusts on trusts. The system was built to amplify a rise. It was therefore equally built to amplify a fall, and almost no one had run that second calculation.

It is worth pausing on why so many smart people missed the danger, because the same blind spot recurs in every bubble. In a long bull market, leverage stops looking like risk and starts looking like prudence. If a stock has risen 20% a year for several years, then borrowing at 8% interest to buy more of it is, in hindsight, a fantastic trade, and everyone who did it got richer than the people who did not. The cautious looked foolish and the leveraged looked brilliant, year after year, which is precisely the conditioning that maximizes the damage when the trend finally turns. The market had also created an aura of permanence; the Yale economist Irving Fisher, one of the most respected economists in the world, declared in October 1929 that stock prices had reached "what looks like a permanently high plateau," a remark that became the most infamous bad forecast in economic history. When the experts, the trend, and your own recent experience all say leverage is safe, the structural fragility hiding underneath is invisible until it is not.

## The blow-up, step by step: October 1929

The peak came in early September 1929, when the Dow Jones Industrial Average closed at 381.17 on September 3. From there it drifted lower through September and into October, with sharp down days unsettling the optimism. Then came the week that gave the crash its name.

**Black Thursday, October 24, 1929.** The market opened sharply lower and selling accelerated into something close to a panic, with prices dropping so fast that the ticker tape ran hours behind the actual trades, so investors could not even tell what their stocks were worth. Around midday a group of leading bankers, organized by J.P. Morgan partner Thomas Lamont, pooled funds and ostentatiously bought blocks of major stocks to restore confidence, a tactic that had worked in the *Panic of 1907*. It worked for a day. The market steadied and even recovered some ground by the close.

**Black Monday, October 28, 1929.** The reassurance did not hold. Monday brought no organized bank support and the market fell about 13% in a single day, one of the worst single-day percentage declines in its history. There was now no floor.

**Black Tuesday, October 29, 1929.** The worst day. Around 16 million shares traded, a volume record that would stand for decades, as everyone tried to sell at once and there were almost no buyers. The Dow fell roughly another 12%. Over the crash week the market lost on the order of 25% of its value, and tens of billions of dollars of paper wealth evaporated.

![Pipeline showing a price drop breaching the maintenance margin, triggering a margin call, and ending in forced selling](/imgs/blogs/wall-street-crash-1929-great-depression-5.png)

Here is where leverage turned an ugly correction into a self-reinforcing collapse, and the pipeline above is the loop. A falling price thins a margin investor's equity. Once that equity drops below the maintenance line, the broker issues a margin call. The investor, who put down only 10%, usually does not have the cash to meet it, so the broker sells the shares to protect its loan. That forced selling pushes the price down further, which breaches the maintenance line for the *next* set of margin investors, who get their own calls, who are also sold out, and so on. Selling begets selling. Unlike an unleveraged market, where a lower price simply tempts bargain hunters, a leveraged market manufactures forced sellers at every step down. This is the precise mechanism by which a 25% drop became unstoppable in the moment.

The crash did not stop in October. It was the beginning of a grinding decline. There were rallies, but the trend was relentlessly down for nearly three years. The Dow finally bottomed on July 8, 1932, at 41.22, a fall of about 89% from the September 1929 peak. An investor who had bought the index at the top and held would not see their money back, in nominal terms, until 1954.

But, and this is the central argument of the post, the 89% stock decline was not the Great Depression. It was a symptom and an accelerant. The Depression itself, the 25% unemployment, the bread lines, the decade of misery, was manufactured by a different machine that started running in 1930.

## The deeper catastrophe: the banking panics of 1930 to 1933

A 25% stock crash, even an 89% one, is survivable for an economy. Most households in 1929 did not own stock at all. The mechanism that turned a sharp recession into the worst depression in modern history ran through the banking system and the money supply, and it is here that the story stops being about Wall Street and becomes about the plumbing of money itself.

Recall fractional-reserve banking: banks hold only a fraction of deposits in cash and lend out the rest. This makes them inherently vulnerable to a loss of confidence. Starting in late 1930, a series of *banking panics* swept the country in waves.

The first wave hit in the autumn of 1930, triggered in part by the failure of the Bank of United States in New York in December 1930, a large bank whose name (despite being a private bank) made the public think a national institution had collapsed. Depositors, frightened, rushed to pull cash from banks everywhere. A second wave hit in 1931, and a third, the worst, in late 1932 and early 1933, culminating in the nationwide bank holiday of March 1933 when the newly inaugurated President Franklin Roosevelt closed every bank in the country to stop the bleeding.

The toll was staggering. Over roughly 1930 to 1933, on the order of 9,000 banks failed, somewhere around a third to 40% of all the banks in the United States. Each failure wiped out the deposits held there (there was no FDIC yet), destroying the savings of millions of ordinary families who had never touched the stock market.

Why was the American banking system so spectacularly fragile in the first place? A key structural reason was *unit banking*. Unlike Canada, which had a handful of large banks each with branches spread across the whole country, the United States in 1929 had tens of thousands of small, independent, single-location banks, a legacy of laws that restricted or banned branch banking. A small-town bank's loans were concentrated in one local economy, often heavily in agriculture, so when that local economy faltered, the bank had no diversification to fall back on and no large parent to lean on. This is part of why Canada, with the same Depression and the same gold standard, suffered *zero* bank failures while the United States lost thousands: the difference was the resilience of the banking structure, not the severity of the shock. The contrast is a clean demonstration that the banking collapse was not an unavoidable feature of the downturn but a product of a uniquely brittle system meeting a passive central bank.

![Graph of the deflationary spiral: bank runs to bank failures to a falling money supply to deflation, raising real debt and cutting spending into the next round of runs](/imgs/blogs/wall-street-crash-1929-great-depression-3.png)

Now follow the spiral in the diagram, because this is the actual engine of the Depression. When a bank fails, the deposits it held simply vanish from the money supply. When fear of failure spreads, surviving banks hoard cash and stop lending, and frightened citizens pull cash out and stuff it under mattresses, which also pulls deposits out of the banking system. Both effects do the same thing: they shrink the money supply. And shrink it did. Between 1929 and 1933 the U.S. money supply contracted by roughly a third. That is not a typo. One out of every three dollars of money, in the broad sense that includes bank deposits, ceased to exist.

A shrinking money supply forces prices down. With far less money chasing goods, sellers must cut prices to sell anything; this is *deflation*. Consumer prices fell roughly 25% from 1929 to 1933. And deflation, as we saw in the foundations, raises the real burden of every fixed debt: farmers, businesses, and households who owed money found that their revenues and wages were collapsing while their loan balances stood still, so they defaulted in droves. Those defaults bankrupted more banks, which destroyed more deposits, which shrank the money supply further, which deepened the deflation. The spiral fed itself. Industrial production roughly halved, and unemployment climbed to around 25%, one in four workers, by 1933.

### The Friedman-Schwartz critique: the Fed could have stopped it

The single most influential explanation of why the banking collapse was allowed to happen comes from Milton Friedman and Anna Schwartz's 1963 book *A Monetary History of the United States*. Their argument, now broadly accepted across the economics profession, is devastating in its simplicity: the Federal Reserve had both the tools and the mandate to stop the money supply from collapsing, and it failed to use them.

The Fed had been created in 1913 specifically to be the lender of last resort, to prevent exactly this kind of banking panic by supplying cash to solvent banks under stress. In 1930 to 1933 it largely did not. It allowed thousands of banks to fail without intervening. It did not flood the system with the reserves that would have let banks meet withdrawals and keep lending. At moments it actually *tightened*, most notoriously in October 1931, when it raised interest rates to defend the dollar's gold convertibility after Britain left the gold standard, precisely when the economy needed easier money. The Fed confused the symptoms of a panic (lots of bank borrowing would have looked like loose credit) with its causes, and a leadership vacuum after the death of the powerful New York Fed governor Benjamin Strong in 1928 left the system without a hand on the wheel.

Friedman and Schwartz's verdict: the Great Contraction of the money supply was not an inevitable consequence of the crash. It was a policy failure. Had the Fed acted as a lender of last resort and kept the money supply roughly stable, the banking panics could have been contained, the deflation arrested, and the recession kept to something far milder than the decade-long depression that resulted. This is not a fringe view; Ben Bernanke, who would chair the Fed in 2008, built his academic career on this literature and said so explicitly, a fact that matters enormously when we get to the echoes.

### The gold standard: a straitjacket that exported the disaster

There is one more actor that turned a national banking crisis into a global depression: the gold standard. Under it, a country's money supply was tied to its gold reserves, and exchange rates between gold-standard currencies were fixed. This created two destructive effects.

First, it tied the Fed's hands, or rather gave it an excuse. To create more money you needed more gold, or at least had to worry that creating money would prompt people to convert dollars to gold and ship it abroad, draining reserves. So the very tool needed to fight the depression, expanding the money supply, conflicted with the rule the Fed felt bound to defend.

![Graph showing the gold standard transmitting deflation: US deflation drains gold from others, who tighten to defend the peg and slump, while countries that leave gold early recover first](/imgs/blogs/wall-street-crash-1929-great-depression-7.png)

Second, and this is the contagion shown above, the gold standard transmitted deflation across borders. When the United States ran tight money and deflated, gold flowed toward it, draining other countries' reserves. To stop the drain and defend their own gold pegs, those countries had to raise interest rates and tighten money too, importing the deflation. Country after country was forced to choose between defending the peg (and slumping) or abandoning gold (and recovering). The historical evidence is stark and is one of the most robust findings in macroeconomic history: the countries that left the gold standard earliest, like Britain in 1931, recovered earliest, while those that clung to gold longest, like France and the United States until 1933 to 1936, suffered the deepest and longest depressions. The gold standard, far from being a stabilizer, was the transmission belt that turned an American mistake into a worldwide catastrophe.

## The mechanism dissected: two failures, not one

It is worth stating the full mechanism cleanly, because the entire argument of this post is that the Depression had two distinct stages with two distinct causes.

**Stage one: leverage turned a correction into a crash.** Stocks in 1929 were arguably overvalued, and some pullback was healthy and inevitable. What made the pullback catastrophic was that it was financed with borrowed money at every level: individuals on 10% margin, investment trusts on debt, trusts owning trusts. Margin transforms a price decline into a wave of forced selling through the margin-call loop, because leveraged holders cannot ride out a drop; they are sold out automatically. A market that should have corrected 20% to 30% instead went into a self-reinforcing downward spiral. Leverage is the answer to "why was the crash so violent?"

**Stage two: policy failure turned a recession into a depression.** The crash and the early downturn would likely have produced a sharp but ordinary recession, the kind the U.S. had weathered before (including the severe but short depression of 1920 to 1921). What turned it into the Great Depression was a triple policy failure that let a banking crisis become a monetary collapse:

1. *The Fed stayed passive* and did not act as lender of last resort, allowing the money supply to contract by a third (the Friedman-Schwartz critique).
2. *There was no deposit insurance,* so bank runs were rational and self-fulfilling; depositors who waited lost everything, so everyone rushed, so banks died.
3. *The gold standard was a straitjacket* that prevented monetary expansion and exported the deflation worldwide.

The result was the deflationary debt spiral: bank failures destroyed deposits, the money supply shrank, prices fell, the real value of every debt rose, debtors defaulted, more banks failed, and around again, each loop deeper than the last. Leverage is the answer to "why was the crash violent?" Policy failure is the answer to "why did the economy stay broken for a decade?" These are different questions with different answers, and conflating them is the original sin of how the period is usually remembered.

## Worked examples: the arithmetic of ruin

Numbers make the mechanism concrete. Each example below uses round figures so you can do the math in your head, and each ends with the single idea it teaches. These are illustrations of the mechanics, not investment recommendations.

#### Worked example: how 10% margin turns a 10% drop into a wipeout

You want exposure to a stock. You have \$1,000. In 1929 you could buy \$10,000 of stock by putting down your \$1,000 and borrowing \$9,000 from your broker.

- Your position: \$10,000 of stock.
- Your equity (your own money): \$1,000.
- Your debt (the broker's loan): \$9,000.

Now the stock falls 10%. The position is worth \$10,000 - \$1,000 = \$9,000. But the \$9,000 loan does not shrink; you still owe every dollar of it. So your equity is now:

- Position value: \$9,000.
- Minus the loan you still owe: \$9,000.
- Equity remaining: \$9,000 - \$9,000 = \$0.

A 10% fall in the stock produced a 100% loss of your money. The leverage ratio was 10-to-1, so the price move was multiplied tenfold against your equity. If the stock fell 15%, you would owe more than the position is worth: equity of \$8,500 - \$9,000 = -\$500, meaning you have lost not just your \$1,000 but owe an additional \$500.

![Before-and-after of a ten percent margin account where a ten percent drop wipes out the equity while the loan is unchanged](/imgs/blogs/wall-street-crash-1929-great-depression-2.png)

The before-and-after above shows the same arithmetic visually: the cash slice that was \$1,000 is simply gone, while the borrowed layer stands untouched. **Intuition: at 10-to-1 leverage, the same 10% wiggle that an outright owner barely notices is the difference between everything and nothing.**

#### Worked example: the margin-call cascade

Now multiply that one investor by a market full of them, and watch the prices feed the calls. Suppose a stock trades at \$100 and many investors hold it on margin with a maintenance requirement that their equity stay at 25% of the position.

- An investor holds 100 shares (\$10,000) financed with \$2,500 of equity and \$7,500 of debt. Equity is 25% of the position, exactly at the line.
- The stock falls to \$90. Position is \$9,000; debt is still \$7,500; equity is \$1,500, which is only 16.7% of \$9,000, below the 25% line.
- The broker issues a margin call. The investor cannot pay, so the broker sells the 100 shares into the market.
- That selling, repeated across thousands of accounts hitting their calls at the same price level, pushes the stock down further, say to \$80.
- At \$80, the next tier of margin investors, who were comfortable at \$90, now breach *their* maintenance line and get their own calls, and are sold out, pushing the price to \$70.

There is no natural stopping point on the way down, because each price decline manufactures a new batch of forced sellers rather than waiting for voluntary ones. In an unleveraged market, a lower price brings in bargain hunters; in a heavily margined one, a lower price brings in more forced sellers. **Intuition: leverage converts a price decline into a feedback loop, where selling causes lower prices which cause more selling.**

#### Worked example: a bank run under fractional reserves

This example is the heart of the banking collapse. A small bank takes in \$1,000 of deposits. Under fractional-reserve banking it keeps 10% as reserves and lends out the rest.

- Reserves (cash on hand): \$100.
- Loans made to borrowers: \$900.
- Deposits it owes back to customers: \$1,000.

In normal times this is fine; only a few depositors withdraw on any given day, and the \$100 of cash covers them easily while the \$900 of loans earns interest. Now a rumor spreads that the bank is in trouble.

- Depositors holding the first \$100 of withdrawals are paid in full from reserves. The cash is now gone.
- The 101st depositor arrives. The bank has \$900 in loans, but those are out with borrowers and cannot be called back instantly; you cannot phone a farmer and demand next year's mortgage today. The bank has no cash.
- The bank cannot pay, the run accelerates as everyone realizes the cash is gone, and the bank fails, even though its \$900 of loans may have been perfectly good and would have been repaid over time.

The bank was *solvent* (its assets of \$100 + \$900 = \$1,000 matched its deposits) but *illiquid* (it could not turn the loans into cash fast enough). A run kills a bank by attacking its liquidity, not its solvency. **Intuition: because a bank lends out most of what it takes in, no bank can survive everyone asking for their money at once; the only cure is something that removes the reason to ask, like deposit insurance or a lender of last resort.**

#### Worked example: the money supply contracting by a third

Bank failures do not just hurt the bank's own customers; they destroy money for everyone, because most money is bank deposits. Imagine a tiny economy with \$3,000 of total money supply, all of it held as deposits across three identical banks holding \$1,000 each.

- Total money supply: \$3,000.
- One bank fails in a run. Its \$1,000 of deposits is wiped out. Money supply falls to \$2,000.
- Fear spreads. Surviving banks, terrified of their own runs, stop making new loans and call in old ones to build up cash. Recall that lending is how banks create deposit money; when they stop, that money creation reverses. Suppose this destroys another \$667 of deposit money.
- Money supply is now about \$1,333, down from \$3,000. It has contracted by roughly a third.

This is exactly what happened to the United States between 1929 and 1933 at national scale: the broad money supply fell by about a third, partly from failed banks' destroyed deposits and partly from the surviving banks' refusal to lend. **Intuition: in a system where money is mostly bank deposits, a banking panic does not merely transfer wealth, it annihilates money, and an economy starved of a third of its money cannot help but collapse into deflation.**

#### Worked example: deflation raising the real burden of a fixed debt

Finally, the link that turned falling prices into mass defaults. A farmer borrows \$1,000 to plant a crop, expecting to repay it from selling the harvest.

- At the time of the loan, wheat sells for \$1 per bushel, so the \$1,000 debt equals 1,000 bushels of wheat.
- Deflation hits: prices fall 25%, so wheat now sells for \$0.75 per bushel.
- The debt is still \$1,000 in dollar terms; the bank does not forgive a cent. But to raise \$1,000 the farmer must now sell \$1,000 / \$0.75 = about 1,333 bushels of wheat.

The real burden of the debt rose from 1,000 bushels to 1,333 bushels, a one-third increase, without the loan ever changing. The farmer's income (the value of the harvest) fell while the debt stood still, so the farmer defaults, the bank takes a loss, and the bank inches closer to its own failure. Multiply across millions of farmers, homeowners, and businesses and you have the wave of defaults that fed the banking collapse. Economist Irving Fisher named this the *debt-deflation* theory in 1933. **Intuition: deflation is a silent transfer that makes every fixed debt heavier in real terms, so a falling price level can bankrupt borrowers who were perfectly solvent before prices dropped.**

## The aftermath: the New Deal rebuilt the plumbing

The policy response, mostly under Roosevelt after 1933, was less about ending the immediate crisis (the recovery was halting and incomplete until wartime spending in the 1940s) than about rebuilding the financial system so the specific failure machinery of 1929 to 1933 could not run again. Four changes did the heavy lifting, and all four remain central to how finance works today.

**Going off gold (1933).** One of Roosevelt's first acts was to break the gold straitjacket. In 1933 the U.S. suspended gold convertibility for citizens, and in 1934 it devalued the dollar against gold (from \$20.67 to \$35 per ounce). Freed from defending the peg, the Fed could finally let the money supply expand. The evidence that this mattered is direct: leaving gold was the single best predictor of when a country's recovery began, across the whole world.

**The FDIC (1933).** The *Federal Deposit Insurance Corporation*, created by the Banking Act of 1933, guaranteed deposits up to a limit (initially \$2,500, today \$250,000). This was arguably the most important reform of all, because it attacked the *reason* to run. If your deposit is safe whether or not you rush to the teller, you do not rush, and the run never forms. Bank runs by ordinary depositors, which had been a recurring feature of American finance for a century, essentially vanished from the United States after 1934. The deposit insurance fund is paid for by premiums on the banks themselves.

**Glass-Steagall (1933).** The *Glass-Steagall Act* separated commercial banking from investment banking, walling off ordinary depositors' money from securities speculation. The wall stood until much of it was repealed by the Gramm-Leach-Bliley Act in 1999, a repeal that critics blame for enabling the conglomerate banks at the center of 2008.

**The SEC (1934).** The *Securities and Exchange Commission*, created in 1934, brought the stock market under federal regulation: mandatory disclosure by public companies, rules against fraud and manipulation, and crucially the power to set *margin requirements*. The Fed was given authority over margin (Regulation T), and the days of 10% margin ended; the standard initial margin for stocks has been 50% since 1974, meaning you can borrow at most half the price, not 90% of it. The single biggest accelerant of the 1929 crash was regulated away.

Together these reforms did not prevent recessions, but they did dismantle the precise mechanism that turned the 1929 crash into the Great Depression. The lender-of-last-resort function was reaffirmed, deposit insurance killed the bank run, the gold standard's grip was broken, and margin was capped. When the next great financial crisis arrived in 2008, policymakers reached for exactly these levers, and a man who had spent his career studying 1929 was at the controls.

## 1929 versus 2008: the lesson applied

The clearest proof that the lessons of 1929 were learned is the 2008 financial crisis, where policymakers faced a structurally similar shock and did almost the exact opposite of what was done in 1929, with a dramatically milder outcome.

![Matrix comparing 1929 to 2008 on trigger, money supply, Fed action, deposit safety, and outcome](/imgs/blogs/wall-street-crash-1929-great-depression-6.png)

The matrix above lays the two side by side. Both began with a leverage-fueled asset bubble: stocks on margin in the 1920s, housing financed with mortgage debt and bank leverage in the 2000s (the full anatomy is in [the Lehman Brothers collapse](/blog/trading/finance/lehman-brothers-2008-financial-crisis)). Both saw a crash in the leveraged asset and the threat of a banking panic. There the paths diverged completely.

In 1929 to 1933 the money supply fell by a third; in 2008 to 2009 the Fed expanded the money supply aggressively through *quantitative easing* (creating money to buy bonds), and the broad money supply kept growing. In 1929 the Fed stayed passive and even tightened; in 2008 it slashed rates to near zero and acted as a lender of last resort on a colossal scale, lending against all manner of collateral. In 1929 there was no deposit insurance and runs spread; in 2008 the FDIC was in place and Congress temporarily raised the insured limit to \$250,000 to head off runs. The outcomes match the choices: unemployment peaked around 25% and stayed elevated for a decade in the 1930s, versus around 10% peak in 2009 with a recovery underway within a year or two.

This was not luck. Ben Bernanke, the Fed chair in 2008, was a scholar of the Great Depression whose academic work focused on exactly the Friedman-Schwartz critique and the role of bank failures. At Milton Friedman's 90th birthday in 2002, Bernanke addressed Friedman directly about the Fed's role in the Depression and said: "You're right, we did it. We're very sorry. But thanks to you, we won't do it again." In 2008 he kept that promise. Whatever one thinks of the bailouts, the counterfactual that haunts every central banker is 1929, and avoiding it shaped the entire response. The Fed's modern interest-rate and emergency-lending toolkit, described in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates), is in large part a set of answers to the questions 1929 posed.

## Common misconceptions

The 1929 to 1933 period is more misremembered than almost any other in financial history. Six beliefs are worth correcting, because each one hides the real mechanism.

**"The 1929 crash caused the Great Depression."** This is the big one. The crash was a leverage accident that destroyed paper wealth and signaled trouble, but a stock crash alone, even an 89% one, does not cause a decade of 25% unemployment. The Depression was caused by the banking collapse and the policy failure that followed: the money supply falling by a third, the deflation, the bank runs, the passive Fed, the gold standard. The crash and the depression were two events with two causes. Most Americans in 1929 owned no stock at all; what destroyed them was the loss of their bank deposits and their jobs, which came from the monetary collapse of 1930 to 1933, not from the ticker tape of October 1929.

**"The Fed did everything it could."** It did almost the opposite of what it should have. As lender of last resort it was supposed to lend freely to solvent banks in a panic; instead it let roughly 9,000 banks fail, allowed the money supply to contract by a third, and at the worst possible moment, October 1931, raised interest rates to defend the gold peg. The Friedman-Schwartz critique, now mainstream, is that the Fed had the tools and the mandate to prevent the monetary collapse and simply did not use them. This is precisely the mistake Bernanke promised, on the record, never to repeat.

**"The gold standard was a stabilizer that protected the economy."** The gold standard was the transmission belt of the disaster, not a brake on it. It tied the Fed's hands when expansion was desperately needed, and it forced every country on gold to import the deflation by defending their pegs. The single most reliable predictor of when a country's recovery began was when it *left* gold; the countries that clung to it longest suffered the most. Discipline against inflation in good times became a doomsday device in a deflation.

**"Buying stocks on margin is the same as investing."** Buying on margin is investing with borrowed money, which is a fundamentally different and far riskier activity than buying stock outright. An outright owner of a stock that falls 50% has lost half their money and can wait for a recovery. A margin buyer at 10% down is wiped out by a 10% drop and forced to sell at the bottom, locking in the loss and sometimes owing more than they invested. Leverage changes not just the size of the outcome but its character: it removes your ability to wait, because the margin call decides for you. Conflating the two is how people in 1929, and in every bubble since, underestimated their risk.

**"Everyone lost money in the crash, so it was an equal-opportunity disaster."** The pain was extremely uneven and ran mostly through channels other than stocks. Stock losses hit the relatively wealthy minority who owned shares. The far broader damage came through unemployment (one in four workers) and bank failures (the destroyed deposits of millions who never owned a single share). The Depression was a macroeconomic and banking catastrophe that reached almost everyone, but not primarily through the stock market.

**"It could never happen again because we are smarter now."** We are not smarter; we have better plumbing. The specific 1929 mechanism, runs destroying deposits and shrinking the money supply while the central bank watches, is largely disarmed by deposit insurance, a lender of last resort that now acts, and a money supply no longer chained to gold. But leverage-fueled bubbles still form, runs still happen in corners without deposit insurance (money-market funds in 2008, certain banks in 2023), and policy can still be too slow or too tight. The lesson is not that crises are abolished; it is that this *particular* failure mode is now defended against, as long as the defenses are maintained.

## How it echoes in other markets

The deep mechanism of 1929 to 1933, leverage amplifying a crash, then a run and a monetary contraction turning a recession into a depression, shows up again and again across financial history. Recognizing the pattern is the whole point of studying it.

**2008: the crisis where the Fed deliberately avoided 1930s mistakes.** The subprime mortgage bubble was a leverage story (banks at 30-to-1, households with little equity in their homes), and its collapse threatened exactly the monetary contraction of 1929: failing banks, a run on the shadow-banking system, and a credit freeze. The difference was the response. Bernanke's Fed, drawing directly on its Depression scholarship, slashed rates to zero, lent against everything as lender of last resort, and expanded the money supply through quantitative easing, while Congress backstopped the banks and the FDIC's existence prevented retail bank runs. The result was a severe recession, not a depression. It is the cleanest controlled experiment in macroeconomic history: same disease, opposite treatment, far better outcome. The mechanics of that crisis are dissected in [the Lehman Brothers collapse](/blog/trading/finance/lehman-brothers-2008-financial-crisis).

**Japan's lost decade and deflationary trap (1990s onward).** Japan's late-1980s bubble in stocks and real estate was extraordinary; at the peak, the land under the Imperial Palace in Tokyo was reputedly worth more than all the real estate in California. When it burst in 1990, Japan slid into a textbook debt-deflation trap: asset prices collapsed, banks were saddled with bad loans, and gentle but persistent deflation raised the real burden of debt for years. The Bank of Japan was criticized, much like the 1930s Fed, for acting too slowly and timidly to break the deflation. Japan never had a 1930s-style collapse, because its deposit system held, but it spent the better part of two decades fighting the same debt-deflation dynamic that crushed American debtors in the early 1930s. It is the modern proof that deflation plus debt is a slow poison even when the banking system survives.

**The Panic of 1907: the run that created the Fed.** Two decades before 1929, the United States suffered a classic banking panic with no central bank to stop it. A failed attempt to corner the stock of a copper company triggered runs on the trust companies that had financed it, and the runs spread through New York's banking system. With no lender of last resort, the panic was halted only when J.P. Morgan personally organized a consortium of bankers to inject liquidity, the same private-rescue tactic that was tried and failed on Black Thursday in 1929. The lesson of 1907, that a modern economy cannot rely on one banker's balance sheet to stop a panic, led directly to the creation of the Federal Reserve in 1913. The tragedy of 1929 to 1933 is that the institution built in response to 1907 then failed to do the one job it was created for.

**The savings-and-loan crisis (1980s).** When U.S. savings-and-loan institutions failed by the hundreds in the 1980s, deposit insurance worked exactly as designed: insured depositors were made whole and there were no mass retail runs, even as the institutions collapsed. The crisis was costly to taxpayers (the cleanup ran to roughly \$130 billion) and exposed real problems with how insurance can encourage reckless lending (*moral hazard*), but it demonstrated the central New Deal insight in action: with deposit insurance, banks can fail without the failure metastasizing into a system-wide run and monetary collapse. The contrast with 1930 to 1933, when there was no insurance and bank failures fed directly into a one-third money contraction, could not be sharper.

**The 2023 regional bank runs (Silicon Valley Bank, Credit Suisse).** In March 2023, Silicon Valley Bank suffered the fastest bank run in history, with tens of billions of dollars withdrawn in a single day, partly because its large concentration of uninsured deposits gave depositors every reason to flee. The episode was a vivid reminder that the run mechanism of 1929 is dormant, not dead: deposit insurance only removes the incentive to run for *insured* balances, so a bank funded by big uninsured depositors is still vulnerable, and digital banking lets a run happen in hours rather than days. Regulators responded in the 1933 spirit, guaranteeing the deposits and providing emergency liquidity to stop the contagion, precisely the lender-of-last-resort move the 1930s Fed refused to make.

**The 1920 to 1921 depression: the recession that did not become a depression.** Just before the Roaring Twenties, the U.S. suffered a sharp deflationary downturn: prices fell sharply and unemployment spiked. Yet it was over within about 18 months. Why did 1920 to 1921 stay a recession while 1929 to 1933 became a depression? The key difference is that 1920 to 1921 did not trigger a self-reinforcing banking panic and monetary collapse; the system absorbed the shock and recovered. The contrast is itself evidence for the central thesis: a sharp recession, even a deflationary one, is survivable, but a sharp recession that triggers an uncontained banking collapse becomes a depression.

**Margin-fueled crashes in modern markets.** The pure leverage-amplification mechanism of October 1929 recurs whenever borrowed money concentrates in a falling asset. The 1987 crash (the Dow fell about 22% in a single day) was amplified by mechanical, leverage-like selling from portfolio insurance strategies. The 2021 collapse of the family office Archegos showed the same shape on a smaller scale: hidden leverage through derivatives meant that when its concentrated stocks fell, forced selling by its lenders cascaded into a multi-billion-dollar loss in days. The instruments change, but the margin-call loop, where falling prices force sales that drive prices lower, is the same engine that ran on Black Tuesday.

## When this matters to you, and further reading

You do not need to trade stocks for 1929 to be relevant to your life. Its lessons are baked into the financial system you already rely on, and recognizing them changes how you read the news.

If you keep money in a bank, the reason you do not have to think about a run is the FDIC, a direct product of 1933. Knowing this is practical: deposit insurance has a limit (currently \$250,000 per depositor per insured bank in the U.S.), and the 2023 runs were largely about deposits above that limit. If you ever hold a large balance, the limit is the line between "the government guarantees this" and "you are an unsecured creditor of a bank that can run."

If you ever consider borrowing to invest, whether stock margin, leveraged ETFs, or crypto with leverage, the first worked example is the one to remember: leverage multiplies the percentage move against your own money, and it removes your ability to wait out a downturn because a margin call can force you to sell at the worst moment. The mechanism that ruined the speculators of 1929 has not changed; only the margin limits and the instruments have.

And if you want to understand the news, watch what central banks *do* in a crisis against the 1929 template. When the Fed cuts rates to zero, lends against unusual collateral, or expands its balance sheet in a panic, it is running the anti-1929 playbook: be the lender of last resort, protect the money supply, do not let a recession curdle into a debt-deflation spiral. When you hear a central banker invoke "we will not repeat the mistakes of the 1930s," now you know exactly which mistakes they mean.

For going deeper, the indispensable source is Milton Friedman and Anna Schwartz's *A Monetary History of the United States, 1867-1960* (1963), whose chapter on the Great Contraction is the foundation of the modern view. John Kenneth Galbraith's *The Great Crash, 1929* is the classic, highly readable narrative of the bubble and the crash itself. Ben Bernanke's *Essays on the Great Depression* (2000) collects the academic work that informed the 2008 response and extends the analysis to the gold standard's international transmission, building on Barry Eichengreen's *Golden Fetters*. On this site, the companion pieces that connect directly to this story are [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier), which explains the money-supply mechanics the Depression turned on; [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates), the modern toolkit built in answer to 1929; [the Lehman Brothers collapse](/blog/trading/finance/lehman-brothers-2008-financial-crisis), the crisis where the lessons were deliberately applied; and [Paul Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation), the mirror-image episode where a Fed chair used the same monetary power to crush inflation rather than fight deflation.

The Great Depression is, in the end, a story about plumbing. The crash was the leak that everyone saw; the depression was the slow flood that came because the people who could have shut off the water decided, for reasons that seemed principled at the time, to leave the valve closed. The reforms that followed were not glamorous, deposit insurance, a cap on margin, a central bank willing to act, but they are the reason a 2008 or a 2020 was a hard recession rather than a second Great Depression. That is the lesson worth keeping: in finance, the unglamorous plumbing is what stands between a bad year and a lost decade.
